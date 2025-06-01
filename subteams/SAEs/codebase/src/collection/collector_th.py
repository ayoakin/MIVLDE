import os
import pickle
import gzip
import pickletools
import threading
import contextlib

from tqdm.auto import tqdm
from typing import Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import src
from src.data import *
from src.instrumentation import *

from odeformer.model import SymbolicTransformerRegressor

# Thread-local storage for model instances
thread_local = threading.local()
path_mapper_lock = threading.Lock()

def get_model(model_args: dict):

    if not hasattr(thread_local, "model"):
        thread_local.model = SymbolicTransformerRegressor(from_pretrained=True)
        thread_local.model.set_model_args(model_args)

        thread_id = threading.get_ident()  # Get unique thread ID
        with path_mapper_lock:
            if thread_id not in src.path_mapper:
                src.path_mapper[thread_id] = ModulePathMapper(thread_local.model)

    return thread_local.model


def _update_bar(bar: tqdm, status: Dict):
    bar.set_postfix(status)
    bar.display()

def process_batch(batch: List[dict], keys: Keys, batch_index: int, output_folder: str, model_args: dict):
    model = get_model(model_args)
    collected_act = []
    
    for solution in batch:
        trajectory = solution['solution']
        times = solution['time_points']

        fit_activations, _ = collect_activations(model, times, trajectory)
        fit_activations = keys.filter(fit_activations)

        collected_act.append((solution, fit_activations))
    
    pickle_data = pickle.dumps(collected_act, protocol=pickle.HIGHEST_PROTOCOL)
    optimized_data = pickletools.optimize(pickle_data)
    
    output_path = os.path.join(output_folder, f'batch_{batch_index + 1}.pkl.gz')
    with gzip.open(output_path, 'wb') as file:
        file.write(optimized_data)
    
    return batch_index

def collect(
    input_path: str, 
    output_folder: str, 
    model_args: Optional[dict] = None,
    keys: Keys = Keys,
    num_threads: int = 4,
    batch_size: int = 10,
    **kwargs
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
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_path, 'rb') as file:
        solutions = pickle.load(file)

    src.path_mapper = dict()

    model_args = model_args or {
        'beam_size': 20, 
        'beam_temperature': 0.1
    }
    
    batch_size = batch_size or len(solutions)

    bar = tqdm(total=len(solutions), desc='Activation Collection', postfix={'Stage': 'Collecting'}, leave=True)
    
    futures = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor, contextlib.redirect_stdout(None):
        for batch_index, i in enumerate(range(0, len(solutions), batch_size)):
            batch = solutions[i:i + batch_size]
            future = executor.submit(process_batch, batch, keys, batch_index, output_folder, model_args)
            futures.append(future)
        
        for future in as_completed(futures):
            batch_index = future.result()
            bar.update(batch_size)
            _update_bar(bar, {'Stage': f'Batch [{batch_index+1:2d}/{(len(solutions) + batch_size - 1) // batch_size:2d}] completed'})
    
    _update_bar(bar, {'Stage': 'Completed'})
