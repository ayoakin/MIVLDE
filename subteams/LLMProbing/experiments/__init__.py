"""
This module contains code for running experiments using the probing pipeline
"""

from .run_experiment import separability_testing  # Import separability_testing from run_experiment.py

__all__ = ["separability_testing"]  # Defines what gets imported with "from experiments import *"