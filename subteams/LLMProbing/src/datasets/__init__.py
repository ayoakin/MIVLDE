"""
This module contains code which creates dataset classes derived from the PyTorch Dataset class
TODO: Possibly create a new SamplesDataset class?
"""

from .activations_dataset import ActivationsDataset

__all__ = ["ActivationsDataset"]