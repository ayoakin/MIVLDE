import pickle
import gzip
import pickletools
import threading
from tqdm.auto import tqdm
from typing import Optional, Dict, Set

import src
from src.data import *
from src.instrumentation import *

from odeformer.model import SymbolicTransformerRegressor
from torch import nn

class TestMod(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        print('here!!')
        return x


def _update_bar(bar: tqdm, status: Dict):
    bar.set_postfix(status)
    bar.display()

def inject_and_pass(
    input_path: str, 
    output_path: str, 
    model_args: Optional[Dict] = None,
    to_inject: Optional[Set] = set(),
    **kwargs
) -> None:
    """
    Processes symbolic regression data, extracts activations, filters them using `Keys`,
    and saves the processed activations.

    Args:
        input_path (str): Path to the input `.pkl` file containing solutions.
        output_path (str): Path where the processed activations will be saved.
        model_args (dict, optional): Arguments for configuring the symbolic transformer model.
        keys (Keys): An instance of the `Keys` class to filter activations.
    """

    with open(input_path, 'rb') as file:
        solutions = pickle.load(file)

    model_args = model_args or {
        'beam_size': 20, 
        'beam_temperature': 0.1
    }

    model = SymbolicTransformerRegressor(from_pretrained=True)
    # model.model = model.model.to('mps')
    model.set_model_args(model_args)

    src.path_mapper = {threading.get_ident(): ModulePathMapper(model, to_inject=to_inject)}
    src.sae = {threading.get_ident(): TestMod()}
        
    bar = tqdm(solutions, desc='Activation Collection', postfix={'Stage': 'Collecting'}, leave=False)

    collected_out = [
        (solution, model.fit(solution['time_points'], solution['solution']))
        for solution in bar
    ]

    _update_bar(bar, {'Stage': 'Saving'})

    pickle_data = pickle.dumps(collected_out, protocol=pickle.HIGHEST_PROTOCOL)

    optimized_data = pickletools.optimize(pickle_data)

    with gzip.open(output_path, 'wb') as file:
        file.write(optimized_data)

    _update_bar(bar, {'Stage': 'Completed'})
    


