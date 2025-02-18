import traceback
from collections import defaultdict
from typing import List


import torch

from mishax import safe_greenlet

from odeformer.model import SymbolicTransformerRegressor

import src
from src.module_mapper import ModulePathMapper

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


def collect_activations(model: SymbolicTransformerRegressor, times: List, trajectories: List):
    """Collects activations during a model function execution."""

    src.path_mapper = ModulePathMapper(model)
    activations = defaultdict(lambda: defaultdict(list))
    
    def _collect_activations(model_fn):
        with src.patcher():
            def run_in_greenlet():
                try:
                    return model_fn()
                except Exception as e:
                    print(f"Error in greenlet execution: {e}")
                    traceback.print_exc()
                    raise

            glet = safe_greenlet.SafeGreenlet(run_in_greenlet)
            with glet:
                result = glet.switch()
                while glet:
                    try:
                        site, value, name = result
                        if torch.is_tensor(value):
                            activations[site][name].append(value.detach().cpu())
                        result = glet.switch(value)
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"Error during activation collection: {e}")
                        traceback.print_exc()
                        break
        
        return _process_activations(activations), result
    
    return _collect_activations(lambda: model.fit(times, trajectories))