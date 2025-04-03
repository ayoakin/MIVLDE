import os
import pandas as pd
from torch.utils.data import DataLoader
from src.datasets.activations_dataset import ActivationsDataset, R2ActivationsDataset
from src.datasets.utils import split_dataset, get_d_in
from src.probes.lr_probe import LRProbe
from src.probes.utils import train_classifier_probe, train_regression_probe, eval_classifier_probe, eval_regression_probe, save_probe_to_path

def train_and_save_probe_separation_expt(target_layer_idx, target_feature, activations_path, \
                                         probe_name, probes_path, \
                                         lr, num_epochs, \
                                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False):
  """
  Trains a logistic regression probe on activations from a specified transformer layer 
  to classify a target binary feature. The trained probe is then evaluated and saved.

  Args:
  - target_layer_idx (int): Index of the transformer layer to extract activations from.
  - target_feature (str): Name of the binary feature to classify.
  - activations_path (str): Path to the directory containing activation files.
  - probe_name (str): Name for saving the trained probe.
  - probes_path (str): Path where the trained probe should be saved.
  - lr (float): Learning rate for training the logistic regression probe.
  - num_epochs (int): Number of training epochs.
  - shuffle_datasets (bool, optional): Whether to shuffle the datasets when creating dataloaders. Default is True.
  - use_val (bool, optional): Whether to use a validation dataset during training. Default is True.
  - data_split (list of float, optional): Proportions for splitting dataset into train, val, and test. Default is [0.8, 0.1, 0.1].
  - write_log (bool, optional): Whether to log training details. Default is False.

  Returns:
  - test_loss (float): Loss on the test dataset.
  - test_acc (float): Accuracy on the test dataset.
  - test_fail_ids (list): List of sample IDs that were incorrectly classified in the test set.
  - final_train_loss (float): Final training loss after the last epoch.
  - final_train_acc (float): Final training accuracy after the last epoch.
  - final_val_loss (float or None): Final validation loss after the last epoch (None if `use_val=False`).
  - final_val_acc (float or None): Final validation accuracy after the last epoch (None if `use_val=False`).
  """

  # Test dataset, dataloaders, and splitting
  full_dataset = ActivationsDataset(activations_path=activations_path, feature_label=target_feature, layer_idx=target_layer_idx)
  train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
  train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=shuffle_datasets)
  if use_val:
    val_dataloader = DataLoader(val_dataset, shuffle=shuffle_datasets)
  test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)

  d_in = get_d_in(full_dataset)
  probe = LRProbe(d_in)

  # Training loop
  if use_val:
    probe, train_losses, train_accuracies, val_losses, val_accuracies = train_classifier_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs, val_dataloader=val_dataloader)
  else:
    probe, train_losses, train_accuracies, val_losses, val_accuracies = train_classifier_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs)

  # Evaluation on test set
  test_loss, test_acc, test_fail_ids = eval_classifier_probe(probe, test_dataloader)
  print(f'Probe trained on layer {target_layer_idx}: Test Set Loss {test_loss}, Test Set Accuracy {test_acc}')

  # Save probe
  os.makedirs(probes_path, exist_ok=True)
  probe_path = os.path.join(probes_path, probe_name)
  save_probe_to_path(probe, probe_path)

  final_train_loss = train_losses[-1]
  final_train_acc = train_accuracies[-1]
  final_val_loss = val_losses[-1]
  final_val_acc = val_accuracies[-1]
  return test_loss, test_acc, test_fail_ids, final_train_loss, final_train_acc, final_val_loss, final_val_acc

def separability_testing(target_feature, activations_path, \
                         probes_path, \
                         lr, num_epochs, \
                         layers, num_repeats=1, \
                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False):
  """
  Trains and evaluates logistic regression probes on activations from a specified list of transformer layers 
  to test feature separability. Each probe training is repeated multiple times to account for initialisation randomness.

  Args:
  - target_feature (str): The binary feature to be classified by the probes.
  - activations_path (str): Path to the directory containing activation files.
  - probes_path (str): Path where trained probes should be saved.
  - lr (float): Learning rate for training each logistic regression probe.
  - num_epochs (int): Number of training epochs.
  - layers (list of int): List of layer indices to train probes on.
  - num_repeats (int, optional): Number of times to repeat the experiment per layer for statistical reliability. Default is 1.
  - shuffle_datasets (bool, optional): Whether to shuffle the datasets when creating dataloaders. Default is True.
  - use_val (bool, optional): Whether to use a validation dataset during training. Default is True.
  - data_split (list of float, optional): Proportions for splitting dataset into train, val, and test. Default is [0.8, 0.1, 0.1].
  - write_log (bool, optional): Whether to log training details. Default is False.

  Returns:
  - experiment_data (pd.DataFrame): A DataFrame containing results for each layer and run, with the following columns:
    - "layer" (int): Transformer layer index.
    - "run" (int): Experiment repetition index.
    - "test_loss" (float): Loss on the test dataset.
    - "test_accuracy" (float): Accuracy on the test dataset.
    - "test_fail_ids" (list): List of sample IDs that were incorrectly classified in the test set.
    - "final_train_loss" (float): Final training loss after the last epoch.
    - "final_train_accuracy" (float): Final training accuracy after the last epoch.
    - "final_val_loss" (float or None): Final validation loss after the last epoch (None if `use_val=False`).
    - "final_val_accuracy" (float or None): Final validation accuracy after the last epoch (None if `use_val=False`).

  Notes:
  - This function iterates over the specified `layers` and trains a probe for each one.
  - Each probe is trained `num_repeats` times per layer to account for randomness in weight initialization.
  - Results are stored in a DataFrame for further analysis.
  """

  experiment_data = []

  # Iterate over the specified layers
  for layer_idx in layers:
    # Repeat a specified number of times
    for run in range(num_repeats):
      print(f'Repeat {run} of layer {layer_idx}')
      
      # Set probe name for saving
      probe_name = f'probe_{target_feature}_{layer_idx}_{run}.pt'
      
      # Train and save the probe for the correct layer
      test_loss, test_acc, test_fail_ids, final_train_loss, final_train_acc, \
        final_val_loss, final_val_acc = train_and_save_probe_separation_expt(target_layer_idx=layer_idx, target_feature=target_feature, activations_path=activations_path, \
          probe_name=probe_name, probes_path=probes_path, \
          lr=lr, num_epochs=num_epochs, \
          shuffle_datasets=shuffle_datasets, use_val=use_val, data_split=data_split, write_log=write_log)

      # Add relevant data to the experiment results
      experiment_data.append({
            "layer": layer_idx,
            "run": run,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_fail_ids": test_fail_ids,
            "final_train_loss": final_train_loss,
            "final_train_accuracy": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_accuracy": final_val_acc
        })

  experiment_data = pd.DataFrame(data=experiment_data)

  return experiment_data

def train_and_save_r2_probe(target_layer_idx, activations_path, \
                                    probe_name, probes_path, \
                                    lr, num_epochs, \
                                    shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False):
  """
  Trains a regression probe on activations from a specified transformer layer 
  to predict the R^2 score of the odeformer's candidate equation. The trained probe is then evaluated and saved.

  Args:
  - target_layer_idx (int): Index of the transformer layer to extract activations from.
  - target_feature (str): Name of the binary feature to classify.
  - activations_path (str): Path to the directory containing activation files.
  - probe_name (str): Name for saving the trained probe.
  - probes_path (str): Path where the trained probe should be saved.
  - lr (float): Learning rate for training the logistic regression probe.
  - num_epochs (int): Number of training epochs.
  - shuffle_datasets (bool, optional): Whether to shuffle the datasets when creating dataloaders. Default is True.
  - use_val (bool, optional): Whether to use a validation dataset during training. Default is True.
  - data_split (list of float, optional): Proportions for splitting dataset into train, val, and test. Default is [0.8, 0.1, 0.1].
  - write_log (bool, optional): Whether to log training details. Default is False.

  Returns:
  - test_loss (float): Loss on the test dataset.
  - test_acc (float): Accuracy on the test dataset.
  - test_fail_ids (list): List of sample IDs that were incorrectly classified in the test set.
  - final_train_loss (float): Final training loss after the last epoch.
  - final_train_acc (float): Final training accuracy after the last epoch.
  - final_val_loss (float or None): Final validation loss after the last epoch (None if `use_val=False`).
  - final_val_acc (float or None): Final validation accuracy after the last epoch (None if `use_val=False`).
  """

  # Test dataset, dataloaders, and splitting
  full_dataset = R2ActivationsDataset(activations_path=activations_path)
  train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
  train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=shuffle_datasets)
  if use_val:
    val_dataloader = DataLoader(val_dataset, shuffle=shuffle_datasets)
  test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)

  d_in = get_d_in(full_dataset)
  probe = LRProbe(d_in)

  # Training loop
  if use_val:
    probe, train_losses, val_losses = train_regression_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs, val_dataloader=val_dataloader)
  else:
    probe, train_losses, val_losses = train_regression_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs)

  # Evaluation on test set
  test_loss = eval_regression_probe(probe, test_dataloader)
  print(f'Regression probe trained on layer {target_layer_idx}: Test Set Loss {test_loss}')

  # Save probe
  os.makedirs(probes_path, exist_ok=True)
  probe_path = os.path.join(probes_path, probe_name)
  save_probe_to_path(probe, probe_path)

  final_train_loss = train_losses[-1]
  final_val_loss = val_losses[-1]
  return test_loss, final_train_loss, final_val_loss

def r2_prediction_experiment(activations_path, probes_path, \
                            lr, num_epochs, \
                            num_repeats=1, \
                            shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False):
  """
  Trains and evaluates a regression probe to predict R^2 scores from activations. 
  The experiment is repeated multiple times to account for randomness.

  Args:
  - activations_path (str): Path to the directory containing activation files.
  - probes_path (str): Path where trained probes should be saved.
  - lr (float): Learning rate for training the regression probe.
  - num_epochs (int): Number of training epochs.
  - num_repeats (int, optional): Number of times to repeat the experiment. Default is 1.
  - shuffle_datasets (bool, optional): Whether to shuffle the datasets when creating dataloaders. Default is True.
  - use_val (bool, optional): Whether to use a validation dataset during training. Default is True.
  - data_split (list of float, optional): Proportions for splitting dataset into train, val, and test. Default is [0.8, 0.1, 0.1].
  - write_log (bool, optional): Whether to log training details. Default is False.

  Returns:
  - experiment_data (pd.DataFrame): A DataFrame containing results for each experiment run, with the following columns:
    - "run" (int): Experiment repetition index.
    - "test_loss" (float): Loss on the test dataset.
    - "final_train_loss" (float): Final training loss after the last epoch.
    - "final_val_loss" (float or None): Final validation loss after the last epoch (None if `use_val=False`).

  Notes:
  - This function trains a regression probe to predict R^2 scores.
  - The experiment is repeated `num_repeats` times to account for randomness in weight initialization and dataset shuffling.
  - Results are stored in a DataFrame for further analysis.
  - The `target_layer_idx` is set to `-1` because we only know the odeformer prediction corresponding to the final decoder layer.
  """

  experiment_data = []

  # Repeat a specified number of times
  for run in range(num_repeats):
    print(f'Repeat {run}')

    # Set probe name for saving
    probe_name = f'probe_r2_{run}.pt'

    # Train and save the probe (note we only consider the final decoder layer, and that we are implicitly considering r2_score)
    test_loss, final_train_loss, final_val_loss = train_and_save_r2_probe(target_layer_idx=-1, activations_path=activations_path, \
                                                                             probe_name=probe_name, probes_path=probes_path, \
                                                                             lr=lr, num_epochs=num_epochs, \
                                                                             shuffle_datasets=shuffle_datasets, use_val=use_val, data_split=data_split, write_log=write_log)
    
    # Add relevant data to the experiment results
    experiment_data.append({
          "layer": 15,
          "run": run,
          "test_loss": test_loss,
          "final_train_loss": final_train_loss,
          "final_val_loss": final_val_loss,
      })

  experiment_data = pd.DataFrame(data=experiment_data)

  return experiment_data

def train_and_save_scalar_prediction_probe(target_layer_idx, target_feature, activations_path, \
                                         probe_name, probes_path, \
                                         lr, num_epochs, \
                                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False):
  """
  Trains a linear regression probe on activations from a specified transformer layer to predict a scalar feature of odeformer's candidate equation. The trained probe is then evaluated and saved.

  Args:
  - target_layer_idx (int): Index of the transformer layer to extract activations from.
  - target_feature (str): Name of the scalar feature to classify.
  - activations_path (str): Path to the directory containing activation files.
  - probe_name (str): Name for saving the trained probe.
  - probes_path (str): Path where the trained probe should be saved.
  - lr (float): Learning rate for training the logistic regression probe.
  - num_epochs (int): Number of training epochs.
  - shuffle_datasets (bool, optional): Whether to shuffle the datasets when creating dataloaders. Default is True.
  - use_val (bool, optional): Whether to use a validation dataset during training. Default is True.
  - data_split (list of float, optional): Proportions for splitting dataset into train, val, and test. Default is [0.8, 0.1, 0.1].
  - write_log (bool, optional): Whether to log training details. Default is False.

  Returns:
  - test_loss (float): Loss on the test dataset.
  - final_train_loss (float): Final training loss after the last epoch.
  - final_val_loss (float or None): Final validation loss after the last epoch (None if `use_val=False`).
  """

  # Test dataset, dataloaders, and splitting
  full_dataset = ActivationsDataset(activations_path=activations_path, feature_label=target_feature, layer_idx=target_layer_idx)
  train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
  train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=shuffle_datasets)
  if use_val:
    val_dataloader = DataLoader(val_dataset, shuffle=shuffle_datasets)
  test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)

  d_in = get_d_in(full_dataset)
  probe = LRProbe(d_in)

  # Training loop
  if use_val:
    probe, train_losses, val_losses = train_regression_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs, val_dataloader=val_dataloader)
  else:
    probe, train_losses, val_losses = train_regression_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs)

  # Evaluation on test set
  test_loss = eval_regression_probe(probe, test_dataloader)
  print(f'Regression probe trained on layer {target_layer_idx}: Test Set Loss {test_loss}')

  # Save probe
  os.makedirs(probes_path, exist_ok=True)
  probe_path = os.path.join(probes_path, probe_name)
  save_probe_to_path(probe, probe_path)

  final_train_loss = train_losses[-1]
  final_val_loss = val_losses[-1]
  return test_loss, final_train_loss, final_val_loss

def scalar_prediction_experiment(target_feature, activations_path, \
                         probes_path, \
                         lr, num_epochs, \
                         layers, num_repeats=1, \
                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False):
  """
  Trains and evaluates regression probes on activations from a specified list of transformer layers to predict a scalar feature. Each probe training is repeated multiple times to account for initialisation randomness.

  Returns:
  - experiment_data (pd.DataFrame): A DataFrame containing results for each experiment run, with the following columns:
    - "run" (int): Experiment repetition index.
    - "test_loss" (float): Loss on the test dataset.
    - "final_train_loss" (float): Final training loss after the last epoch.
    - "final_val_loss" (float or None): Final validation loss after the last epoch (None if `use_val=False`).

  Notes:
  - This function iterates over the specified `layers` and trains a probe for each one.
  - Each probe is trained `num_repeats` times per layer to account for randomness in weight initialization.
  - Results are stored in a DataFrame for further analysis.
  """

  
  experiment_data = []

  # Iterate over the specified layers
  for layer_idx in layers:
    # Repeat a specified number of times
    for run in range(num_repeats):
      print(f'Repeat {run} of layer {layer_idx}')
      
      # Set probe name for saving
      probe_name = f'probe_{target_feature}_{layer_idx}_{run}.pt'
      
      # Train and save the probe for the correct layer
      test_loss, final_train_loss, final_val_loss = train_and_save_scalar_prediction_probe(target_layer_idx=layer_idx, target_feature=target_feature, activations_path=activations_path, \
          probe_name=probe_name, probes_path=probes_path, \
          lr=lr, num_epochs=num_epochs, \
          shuffle_datasets=shuffle_datasets, use_val=use_val, data_split=data_split, write_log=write_log)

      # Add relevant data to the experiment results
      experiment_data.append({
          "layer": layer_idx,
          "run": run,
          "test_loss": test_loss,
          "final_train_loss": final_train_loss,
          "final_val_loss": final_val_loss,
        })

  experiment_data = pd.DataFrame(data=experiment_data)

  return experiment_data