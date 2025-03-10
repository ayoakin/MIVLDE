import traceback
from collections import defaultdict
from typing import List, Dict, Tuple

import torch

from mishax import safe_greenlet
from odeformer.model import SymbolicTransformerRegressor
import src

def _process_activations(
    activations: Dict[str, Dict[str, List[torch.Tensor]]]
) -> Dict[str, Dict[str, Dict[torch.Size, torch.Tensor]]]:
    """
    Processes collected activations into a structured format.
    Groups activations by tensor shape and stacks tensors accordingly.

    Args:
        activations (dict): Dictionary containing activation data.

    Returns:
        dict: Processed activation dictionary with tensors grouped by shape.
    """
    processed = dict()
    for site, name_data in activations.items():
        processed[site] = dict()
        for name, tensors in name_data.items():
            grouped = defaultdict(list)
            
            # Group tensors by their shape
            for tensor in tensors:
                grouped[tuple(tensor.shape)].append(tensor)
            
            # Stack tensors for each shape
            processed[site][name] = {
                shape: torch.stack(tensors) 
                for shape, tensors in grouped.items()
            }
    
    return processed

def collect_activations(
    model: SymbolicTransformerRegressor, 
    times: List, 
    trajectories: List
) -> Tuple[Dict[str, Dict[str, Dict[torch.Size, torch.Tensor]]], torch.Tensor]:
    """
    Collects activations during the execution of a model function.

    Uses a safe greenlet to execute the model function and capture activations.
    Handles exceptions and ensures activations are detached from computation graph.

    Args:
        model (SymbolicTransformerRegressor): The model instance to collect activations from.
        times (List): Time data for the model fitting.
        trajectories (List): Trajectory data for the model fitting.

    Returns:
        tuple: Processed activations dictionary and the final model result.
    """
    activations = defaultdict(lambda: defaultdict(list))
    
    def _collect_activations(model_fn):
        with src.patcher():  # Apply any necessary patching for the model execution
            def run_in_greenlet():
                """Runs the model function in a greenlet and handles exceptions."""
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
                            # Store activation tensors in a detached form
                            activations[site][name].append(value.detach().cpu())
                        result = glet.switch(value)
                    except StopIteration:
                        break  # Exit loop when execution is complete
                    except Exception as e:
                        print(f"Error during activation collection: {e}")
                        traceback.print_exc()
                        break
        
        return _process_activations(activations), result
    
    return _collect_activations(lambda: model.fit(times, trajectories))
