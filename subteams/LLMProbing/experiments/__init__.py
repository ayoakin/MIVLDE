"""
This module contains code for running experiments using the probing pipeline
"""

from .run_experiment import separability_testing, r2_prediction_experiment, scalar_prediction_experiment  # Import from run_experiment.py

__all__ = ["separability_testing", "r2_prediction_experiment", "scalar_prediction_experiment"]  # Defines what gets imported with "from experiments import *"