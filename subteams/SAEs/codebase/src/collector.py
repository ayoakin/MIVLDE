import os
import pickle
import gzip
import pickletools
import threading
import contextlib

from queue import Queue
from tqdm.auto import tqdm
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import src
from src.Keys import Keys
from src.ModulePathMapper import ModulePathMapper
from src.activations import collect_activations

from odeformer.model import SymbolicTransformerRegressor

# Thread-local storage for model instances, MISHAX need it
thread_local = threading.local()
path_mapper_lock = threading.Lock()

def _get_model(model_args: dict):
    """
    Retrieve a model instance for the current thread, creating one if necessary.

    Ensures that each thread has its own instance of the SymbolicTransformerRegressor.
    Also maps the model instance to a unique thread-specific path in src.path_mapper.
    """
    if not hasattr(thread_local, "model"):
        thread_local.model = SymbolicTransformerRegressor(from_pretrained=True)
        # thread_local.model.model = thread_local.model.model.to('mps')  # Uncomment if using MPS
        thread_local.model.set_model_args(model_args)

        thread_id = threading.get_ident()
        with path_mapper_lock:
            if thread_id not in src.path_mapper:
                src.path_mapper[thread_id] = ModulePathMapper(thread_local.model)

    return thread_local.model

def producer_thread(
    queue: Queue, 
    batch: List[dict], 
    keys: Keys, 
    batch_index: int, 
    model_args: dict
) -> None:
    """
    Producer function that collects activations for a given batch of solutions
    and places the results into a queue for processing by consumers.

    Args:
        queue (Queue): The queue to store processed activations.
        batch (List[dict]): A list of solution dictionaries.
        keys (Keys): A filtering mechanism for activation keys.
        batch_index (int): The index of the batch.
        model_args (dict): Model configuration parameters.
    """
    model = _get_model(model_args)

    collected_activations = [
        keys.filter(
            collect_activations(model, solution['time_points'], solution['solution'])[0]
        )
        for solution in batch
    ]

    queue.put((batch_index, collected_activations))

def consumer_thread(queue: Queue, output_folder: str, total_batches: int) -> None:
    """
    Consumer function that retrieves processed activations from the queue
    and saves them as compressed pickle files.

    Args:
        queue (Queue): The queue containing processed activation data.
        output_folder (str): The folder where output files will be saved.
        total_batches (int): The number of batches this thread should process.
    """
    processed_count = 0
    while processed_count < total_batches:
        batch_index, collected_act = queue.get()

        output_path = os.path.join(output_folder, f'batch_{batch_index + 1}.pkl.gz')
        with gzip.open(output_path, 'wb') as file:
            file.write(
                pickletools.optimize(
                    pickle.dumps(collected_act, protocol=pickle.HIGHEST_PROTOCOL)
                )
            )
        processed_count += 1
        queue.task_done()

def _update_bar(bar: tqdm, status: Dict) -> None:
    """
    Updates the tqdm progress bar with a status message.

    Args:
        bar (tqdm): The tqdm progress bar instance.
        status (Dict): A dictionary containing the status message.
    """
    bar.set_postfix(status)
    bar.display()


def collect(
    input_path: str, 
    output_folder: str, 
    model_args: Optional[dict] = None,
    keys: Keys = Keys,
    num_threads: int = 4,
    batch_size: int = 10
) -> None:
    """
    Processes symbolic regression data in parallel, extracts activations, filters them using `Keys`,
    and saves the processed activations into multiple files.

    Args:
        input_path (str): Path to the input `.pkl` file containing solutions.
        output_folder (str): Folder where the processed activation files will be saved.
        model_args (dict, optional): Arguments for configuring the symbolic transformer model.
        keys (Keys): An instance of the `Keys` class to filter activations.
        num_threads (int): Number of threads to use for parallel processing.
        batch_size (int): Number of solutions per batch.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load input data
    with open(input_path, 'rb') as file:
        solutions = pickle.load(file)

    # Initialize model path mapping
    src.path_mapper = dict()

    # Set default model arguments if none are provided
    model_args = model_args or {
        'beam_size': 20, 
        'beam_temperature': 0.1
    }
    
    # Determine the total number of batches
    total_batches = (len(solutions) + batch_size - 1) // batch_size

    # Progress bar initialization
    bar = tqdm(
        total=len(solutions), 
        desc='Activation Collection', 
        postfix={'Stage': 'Initializing'}, 
        leave=True
    )

    # Create a queue for inter-thread communication
    queue = Queue(maxsize=20)

    # Determine the number of consumer threads (1/3rd of num_threads)
    consumer_threads = num_threads // 3
    batches_per_thread = total_batches // consumer_threads
    remainder = total_batches % consumer_threads

    # Start consumer threads
    for i in range(consumer_threads):
        threading.Thread(
            target=consumer_thread,
            args=(queue, output_folder, batches_per_thread + int(i < remainder)),
            daemon=True
        ).start()

    _update_bar(bar, {'Stage': f'Collection'})
    
    # Producer processing
    with (
        contextlib.redirect_stdout(None),
        ThreadPoolExecutor(max_workers=num_threads) as producers
    ):
        # Submit producer tasks
        futures = [
            producers.submit(
                producer_thread, 
                queue, solutions[i:i + batch_size], 
                keys, batch_index, model_args
            )
            for batch_index, i in enumerate(range(0, len(solutions), batch_size))
        ]
        
        # Update progress bar as producers finish
        for i, future in enumerate(as_completed(futures)):
            future.result()
            bar.update(batch_size)
            _update_bar(bar, {'Stage': f'[{i+1:04d}/{total_batches:04d}] batches collected'})
    
    # Finalize processing
    _update_bar(bar, {'Stage': 'Saving'})
    queue.join()
    _update_bar(bar, {'Stage': 'Completed'})
