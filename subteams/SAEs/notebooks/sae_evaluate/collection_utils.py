import sys
import time
from collections import defaultdict
import pickle
import gzip
import pickletools
import re
from dataclasses import dataclass, field  # Fixed this line
from typing import Dict, Set
import torch
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing import Process, Queue, set_start_method
import queue
from tqdm import tqdm
from tqdm.auto import tqdm
import numpy as np
np.infty = np.inf

import enum
import traceback

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from mishax import ast_patcher
from mishax import safe_greenlet

sys.path.append("../odeformer/")
sys.path.append("../odeformer/odeformer")
sys.path.append("../odeformer/odeformer/envs")
import odeformer
from odeformer.model import SymbolicTransformerRegressor


# Global variables for multiprocessing
models = None
model_args = None
keys = None


class Site(enum.StrEnum):
    """Instrumentation sites within an ODEFormer forward pass."""
    # Attention sites
    ATTN_SCORES, ATTN_PROBS, ATTN_OUTPUT, ATTN_MLP_OUTPUT, RESIDUAL_STREAM = (
        enum.auto(), enum.auto(), enum.auto(), enum.auto(), enum.auto()
    )

    # Layer norm sites
    PRE_ATTN_LAYERNORM, PRE_MLP_LAYERNORM = enum.auto(), enum.auto()

    # MLP sites
    MLP_INPUT, MLP_HIDDEN, MLP_OUTPUT, POST_MLP_RESIDUAL = enum.auto(), enum.auto(), enum.auto(), enum.auto()

    # Cross attention (decoder only)
    CROSS_ATTN_SCORES, CROSS_ATTN_PROBS, CROSS_ATTN_OUTPUT = enum.auto(), enum.auto(), enum.auto()


@dataclass
class ModulePathMapper:
    """
    Maps neural network modules to their hierarchical paths within the model.
    
    This class maintains a mapping of module instances to their corresponding
    names within the model architecture, allowing for structured access.
    """
    model: object  # The model whose module paths need to be mapped
    path_map: dict = field(default_factory=dict)  # Stores module paths by their ID

    def __post_init__(self):
        """
        Constructs the module-to-path mapping.
        Iterates through the model's encoder and decoder sections to
        build hierarchical paths for named modules.
        """
        model = getattr(self.model, 'model', self.model)

        def _name(section: str):
            """Registers modules under the given section (encoder/decoder)."""
            if not (module := getattr(model, section, None)):
                return

            for name, sub_module in module.named_modules():
                self.path_map[id(sub_module)] = f"{section}.{name if name else 'outer'}"
        
        _name('encoder')
        _name('decoder')

    def get_layer_path(self, module: nn.Module, accessing_component: str = None) -> str:
        """
        Retrieves the full hierarchical path of a given module.
        
        Args:
            module (nn.Module): The module whose path is being queried.
            accessing_component (str, optional): Additional component name
                (e.g., an attribute) to append to the path.
                Defaults to None.
        
        Returns:
            str: The full path of the module, including any accessed component.
        """
        base_path = self.path_map.get(id(module))
        return f"{base_path}.{accessing_component}" if base_path and accessing_component else base_path

_path_mapper = None

def _tag(module: nn.Module, site: Site, value: torch.Tensor, accessing: str = None) -> torch.Tensor:
    """Tags a value at a particular site for instrumentation."""
    try:
        parent = safe_greenlet.getparent()
        if parent is None:
            return value

        # Get full path including component
        path = None
        if _path_mapper is not None:
            path = _path_mapper.get_layer_path(module, accessing)

        ret = parent.switch((site, value, path))
        return ret if ret is not None else value
    except Exception as e:
        print(f"Error in tag at {site}: {e}")
        return value

def collect_activations_during_fit(model, times, trajectories):
    """Collects activations during model training."""
    global _path_mapper
    _path_mapper = ModulePathMapper(model)
    return collect_activations(lambda: model.fit(times, trajectories))

def install():
    """Installs patches for instrumentation."""
    print("Installing patches...", flush=True)
    
    PREFIX = f"from {__name__} import Site, _tag as tag"
    
    patcher = ast_patcher.ModuleASTPatcher(
        odeformer.model.transformer,
        ast_patcher.PatchSettings(prefix=PREFIX),
        MultiHeadAttention=[
            "scores = torch.matmul(q, k.transpose(2, 3))",
            "scores = tag(self, Site.ATTN_SCORES, torch.matmul(q, k.transpose(2, 3)), accessing='scores')",
            
            "weights = F.softmax(scores.float(), dim=-1).type_as(scores)",
            "weights = tag(self, Site.ATTN_PROBS, F.softmax(scores.float(), dim=-1).type_as(scores), accessing='weights')",

            "context = torch.matmul(weights, v)",
            "context = tag(self, Site.ATTN_OUTPUT, torch.matmul(weights, v), accessing='context')",
        ],
        TransformerModel=[        
            """
            attn = self.encoder_attn[i](
                    tensor, src_mask, kv=src_enc, use_cache=use_cache
                )
            """,
            """
            attn = tag(
                        self, Site.ATTN_MLP_OUTPUT, 
                        self.encoder_attn[i](
                            tensor, src_mask, kv=src_enc, use_cache=use_cache
                        ),
                        accessing=f'cross_attention{i}'
                    )
            """,
            
            "attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)",
            "attn = tag(self, Site.ATTN_MLP_OUTPUT, self.attentions[i](tag(self, Site.RESIDUAL_STREAM, tensor, accessing=f'residual{i}'), attn_mask, use_cache=use_cache), accessing='attention_layer{i}')",
        ],
        TransformerFFN=[
            "x = self.lin1(input)",
            "x = self.lin1(tag(self, Site.MLP_INPUT, input, accessing='input'))",
            
            "x = self.lin2(x)",
            "x = tag(self, Site.MLP_OUTPUT, self.lin2(x), accessing='output')",
        ]
    )

    try:
        patcher.install()
        print("Patches installed successfully")
    except Exception as e:
        print(f"Error installing patches: {e}")
        traceback.print_exc()
    
    return patcher

patcher = install()


def _process_activations(activations):
    """Processes collected activations into a structured format."""
    processed = {}
    for site, name_data in activations.items():
        processed[site] = {}
        for name, tensors in name_data.items():
            grouped = defaultdict(list)
            for tensor in tensors:
                grouped[tuple(tensor.shape)].append(tensor)
            processed[site][name] = {
                shape: torch.stack(tensors) 
                for shape, tensors in grouped.items()
            }
            # processed[site][name] = tensors
    return processed

def collect_activations(model_fn):
    """Collects activations during a model function execution."""
    # print("\nStarting activation collection")
    activations = defaultdict(lambda: defaultdict(list))
    
    with patcher():
        def run_in_greenlet():
            try:
                # print("Starting model execution in greenlet...")
                return model_fn()
            except Exception as e:
                print(f"Error in greenlet execution: {e}")
                traceback.print_exc()

        glet = safe_greenlet.SafeGreenlet(run_in_greenlet)
        with glet:
            result = glet.switch()
            while glet:
                try:
                    site, value, name = result
                    site = str(site)
                    if torch.is_tensor(value):
                        activations[site][name].append(value.detach().cpu())
                    result = glet.switch(value)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Error during activation collection: {e}")
                    traceback.print_exc()
    
    # print(f"Collection complete. Found sites: {list(activations.keys())}")
    return _process_activations(activations), result

@dataclass
class Keys:
    """
    Stores configuration settings for filtering activations.

    Attributes:
        encoders (Set[int]): Set of encoder layer indices to keep.
        decoders (Set[int]): Set of decoder layer indices to keep.
        encoder_attn (Set[str]): Set of cross-attention-related activations to collect. 
                                 Can contain any of: ['attn_scores', 'attn_probs', 'attn_output'].
        cross_attention (bool): Whether to keep cross-attention activations.

        to_collect (Dict[str, bool]): Dictionary mapping activation types to boolean values
                                      indicating whether to collect them.
    """
    encoders: Set[int] = field(default_factory=set)
    decoders: Set[int] = field(default_factory=set)
    encoder_attn: Set[str] = field(default_factory=set)
    cross_attention: bool = False

    to_collect: Dict[str, bool] = field(default_factory=lambda: {
        'residual_stream': False,
        'attn_scores': False,
        'attn_probs': False,
        'attn_output': False,
        'attn_mlp_output': False,
        'mlp_input': False,
        'mlp_output': False
    })

    def __post_init__(self):
        """
        Ensures that `to_collect` always contains all possible keys with default values.
        This prevents missing keys from causing errors when accessing `to_collect`.
        """
        default_flags = {
            'residual_stream': False,
            'attn_scores': False,
            'attn_probs': False,
            'attn_output': False,
            'attn_mlp_output': False,
            'mlp_input': False,
            'mlp_output': False
        }

        # Merge user-defined `to_collect` values with default values
        self.to_collect = {**default_flags, **self.to_collect}

    def _encoders_attn_to_remove(self) -> Set[str]:
        """
        Determines which attention types should be removed from activations.

        Returns:
            Set[str]: A set of attention keys to remove.
        """
        return {'attn_scores', 'attn_probs', 'attn_output'} - self.encoder_attn


    def filter(self, activations: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Filters activations based on the `to_collect` settings and encoder/decoder indices.

        Args:
            activations (Dict[str, Dict[str, torch.Tensor]]): 
                Dictionary containing activation tensors, structured as:
                {
                    "attn_scores": { "encoder.attentions.0.scores": tensor, ... },
                    "attn_probs": { "encoder.attentions.1.weights": tensor, ... },
                    ...
                }

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: The filtered activations dictionary.
        """

        # Keep only activation keys that are enabled in `to_collect`
        activations = {key: sub_dict for key, sub_dict in activations.items() if self.to_collect.get(key, False)}

        # Remove encoder attention entries that are NOT specified in `encoder_attn`
        if (to_remove := self._encoders_attn_to_remove()):
            for key in filter(lambda k: k in activations, to_remove):
                activations[key] = {
                    k: v for k, v in activations[key].items() 
                    if 'encoder_attn' not in k
                }

        # If `cross_attention` is disabled, remove all `cross_attention` activations from 'attn_mlp_output'
        if not self.cross_attention and (mlp_output := activations.get('attn_mlp_output', None)):
            activations['attn_mlp_output'] = {
                k: v for k, v in mlp_output.items() 
                if 'cross_attention' not in k
            }

        def check(x: str) -> bool:
            """
            Determines whether a given activation key corresponds to a valid encoder or decoder layer.

            Args:
                x (str): The activation key, e.g., 'encoder.attentions.3.scores'.

            Returns:
                bool: True if the activation should be kept, False otherwise.
            """
            if not (match := re.search(r"\d+", x)):  # Extracts the first number in the key
                return False
            
            layer = int(match.group())
            indices = self.encoders if x.startswith("encoder") else self.decoders

            return layer in indices  # Keep only layers that exist in `self.encoders` or `self.decoders`

        # Apply layer filtering to each activation type
        return {
            key: {k: v for k, v in values.items() if check(k)}
            for key, values in activations.items()
        }


def worker_process(process_id, task_queue, result_queue, keys):  
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    
    print(f"Worker {process_id} starting on device {device}")
    
    model = SymbolicTransformerRegressor(from_pretrained=True)
    

    model.model.to(device)
    if hasattr(model.model, 'embedder'):
        model.model.embedder.to(device)
    if hasattr(model.model, 'embeddings'):
        model.model.embeddings.to(device)
    
    while True:
        task = task_queue.get()
        if task is None:
            break
            
        index, solution = task
        try:
            trajectory = solution['solution']
            times = solution['time_points']
            
            if torch.is_tensor(trajectory):
                trajectory = trajectory.to(device)
            if torch.is_tensor(times):
                times = times.to(device)
            
            model.model.eval()
            
            with torch.cuda.device(device):
                fit_activations, outputs = collect_activations_during_fit(model, times, trajectory)
            
            for site in fit_activations:
                for name in fit_activations[site]:
                    if isinstance(fit_activations[site][name], dict):
                        for shape in fit_activations[site][name]:
                            fit_activations[site][name][shape] = fit_activations[site][name][shape].cpu()
            
            fit_activations = keys.filter(fit_activations)
            result = (solution, fit_activations)
            result_queue.put((index, result))
            
            del outputs, fit_activations
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error in worker {process_id} processing solution {index}: {str(e)}")
            result_queue.put((index, None))
            traceback.print_exc()
            torch.cuda.empty_cache()
    
    del model
    torch.cuda.empty_cache()

def collect(
    input_path: str, 
    output_path: str, 
    num_workers,
    keys_instance,
    model_args_dict: dict = None,
):
    import torch.multiprocessing as mp
    import sys
    from tqdm import tqdm

    start_time = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.init()
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp.set_start_method('spawn', force=True)

    print("Loading solutions...")
    with open(input_path, 'rb') as file:
        solutions = pickle.load(file)
    print(f"Loaded {len(solutions)} solutions")

    task_queue = mp.JoinableQueue()
    result_queue = mp.JoinableQueue()
    
    print(f"Starting {num_workers} workers...")
    
    processes = []
    for i in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(i, task_queue, result_queue, keys_instance)
        )
        p.start()
        processes.append(p)

    for i, solution in enumerate(solutions):
        task_queue.put((i, solution))
    
    for _ in range(num_workers):
        task_queue.put(None)

    collected_act = [None] * len(solutions)
    processed_count = 0

    #solutions=solutions[:10]
    
    pbar = tqdm(total=len(solutions), desc='Processing Solutions', 
                file=sys.stdout, dynamic_ncols=True, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    while processed_count < len(solutions):
        try:
            index, result = result_queue.get(timeout=120)  # 2 minutes 
            if result is not None:
                collected_act[index] = result
            processed_count += 1
            pbar.update(1)
            
        except  queue.Empty:
            print("\nQueue timeout - checking process status...")
            if not any(p.is_alive() for p in processes):
                print("All processes have died - terminating")
                break
    
    pbar.close()

    print(f"Execution time: {(time.time() - start_time)} seconds")
    
    print("Joining queue threads...")
    task_queue.close()
    result_queue.close()

    print("Cleaning up processes...")
    # Terminate and clean up processes
    for p in processes:
        if p.is_alive():
            print(f"Terminating process {p.pid}")
            p.terminate()
            p.join(timeout=5)

    task_queue.cancel_join_thread()
    result_queue.cancel_join_thread()
    
    collected_act = [r for r in collected_act if r is not None]
    print(f"Successfully processed {len(collected_act)}/{len(solutions)} solutions")

    print('Saving results...')
    pickle_data = pickle.dumps(collected_act, protocol=pickle.HIGHEST_PROTOCOL)
    optimized_data = pickletools.optimize(pickle_data)
    
    with gzip.open(output_path, 'wb') as file:
        file.write(optimized_data)

    print(f'Results saved to {output_path}')