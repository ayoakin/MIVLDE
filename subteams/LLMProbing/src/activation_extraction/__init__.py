"""
This module contains code for extracting activations from given samples
TODO: Possibly integrate with a new SamplesDataset class?
"""

from .activations_extractor import ActivationsExtractor  # Import ActivationsExtractor from activations_extractor.py

__all__ = ["ActivationsExtractor"]  # Defines what gets imported with "from activation_extraction import *"