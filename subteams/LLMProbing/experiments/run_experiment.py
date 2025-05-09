import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.datasets.activations_dataset import ActivationsDataset, R2ActivationsDataset
from src.datasets.utils import split_dataset, get_d_in
from src.probes.lr_probe import LRProbe
from src.probes.utils import train_classifier_probe, train_regression_probe, eval_classifier_probe, \
  eval_regression_probe, save_probe_to_path, load_probe_from_path, train_regression_probe_w_solver, \
  train_classifier_probe_w_solver, eval_solver_classifier_probe

# TODO: fix val_loss etc. when use_val = False

def train_and_save_probe_separation_expt(target_layer_idx, target_feature, activations_path, \
                                         probe_name, probes_path, \
                                         lr, num_epochs, \
                                         r2_threshold=None, \
                                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False, use_solver=False):
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
  full_dataset = ActivationsDataset(activations_path=activations_path, feature_label=target_feature, layer_idx=target_layer_idx, r2_threshold=r2_threshold)
  train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
  train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=shuffle_datasets)
  if use_val:
    val_dataloader = DataLoader(val_dataset, shuffle=shuffle_datasets)
  test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)

  d_in = get_d_in(full_dataset)
  probe = LRProbe(d_in)

  # Training loop
  if use_solver:
    if use_val:
      probe, train_losses, train_accuracies, val_losses, val_accuracies = train_classifier_probe_w_solver(probe, train_dataset, val_dataset=val_dataset)
    else:
      probe, train_losses, train_accuracies, val_losses, val_accuracies = train_classifier_probe_w_solver(probe, train_dataset, val_dataset=val_dataset)
  else:
    if use_val:
      probe, train_losses, train_accuracies, val_losses, val_accuracies = train_classifier_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs, val_dataloader=val_dataloader)
    else:
      probe, train_losses, train_accuracies, val_losses, val_accuracies = train_classifier_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs)

  # Evaluation on test set
  if use_solver:
    test_loss, test_acc, test_fail_ids = eval_solver_classifier_probe(probe, test_dataset)
  else:
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
                         r2_threshold=None, \
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
          r2_threshold=r2_threshold, \
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

def separability_testing_w_solver(target_feature, activations_path, \
                         probes_path, \
                         layers, \
                         r2_threshold=None, \
                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1]):
  """
  Trains and evaluates logistic regression probes on activations from a specified list of transformer layers 
  to test feature separability. Probe training is done using a direct solver from scikit-learn.

  Args:
  - target_feature (str): The binary feature to be classified by the probes.
  - activations_path (str): Path to the directory containing activation files.
  - probes_path (str): Path where trained probes should be saved.
  - layers (list of int): List of layer indices to train probes on.
  - shuffle_datasets (bool, optional): Whether to shuffle the datasets when creating dataloaders. Default is True.
  - use_val (bool, optional): Whether to use a validation dataset during training. Default is True.
  - data_split (list of float, optional): Proportions for splitting dataset into train, val, and test. Default is [0.8, 0.1, 0.1].

  Returns:
  - experiment_data (pd.DataFrame): A DataFrame containing results for each layer and run, with the following columns:
    - "layer" (int): Transformer layer index.
    - "test_loss" (float): Loss on the test dataset.
    - "test_accuracy" (float): Accuracy on the test dataset.
    - "test_fail_ids" (list): List of sample IDs that were incorrectly classified in the test set.
    - "final_train_loss" (float): Final training loss after the last epoch.
    - "final_train_accuracy" (float): Final training accuracy after the last epoch.
    - "final_val_loss" (float or None): Final validation loss after the last epoch (None if `use_val=False`).
    - "final_val_accuracy" (float or None): Final validation accuracy after the last epoch (None if `use_val=False`).

  Notes:
  - This function iterates over the specified `layers` and trains a probe for each one.
  - Results are stored in a DataFrame for further analysis.
  """

  experiment_data = []

  # Iterate over the specified layers
  for layer_idx in layers:
    # Set probe name for saving
    probe_name = f'probe_{target_feature}_{layer_idx}_{0}.pt'
    
    # Train and save the probe for the correct layer
    test_loss, test_acc, test_fail_ids, final_train_loss, final_train_acc, \
      final_val_loss, final_val_acc = train_and_save_probe_separation_expt(target_layer_idx=layer_idx, target_feature=target_feature, activations_path=activations_path, \
        probe_name=probe_name, probes_path=probes_path, \
        lr=1, num_epochs=1, \
        r2_threshold=r2_threshold, \
        shuffle_datasets=shuffle_datasets, use_val=use_val, data_split=data_split, use_solver=True)

    # Add relevant data to the experiment results
    experiment_data.append({
          "layer": layer_idx,
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
                                    r2_threshold=None, \
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
  full_dataset = R2ActivationsDataset(activations_path=activations_path, r2_threshold=r2_threshold)
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
  test_loss, test_r2, test_spearman, test_pearson = eval_regression_probe(probe, test_dataloader)
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
                            r2_threshold=None, \
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
                                                                             r2_threshold=r2_threshold, \
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


def load_and_run_r2_prediction_experiment(activations_path, \
                         probes_path, \
                         r2_threshold=None, \
                         num_repeats=1, \
                         shuffle_datasets=True, use_val=True, data_split=[0.8, 0.1, 0.1]):
  """
  Loads and evaluates regression probes on activations from a specified list of transformer layers to predict R^2 score.
  Loads a number of probes up to num_repeats based on how many were repeats trained.

  Returns:
  - experiment_data (pd.DataFrame): A DataFrame containing results for each experiment run, with the following columns:
    - "run" (int): Experiment repetition index.
    - "test_loss" (float): Loss on the test dataset.

  Notes:
  - This function iterates over the specified `layers` and loads and evaluates a probe trained for each one.
  - Results are stored in a DataFrame for further analysis.
  """

  experiment_data = []

  full_dataset = R2ActivationsDataset(activations_path=activations_path, r2_threshold=r2_threshold)
  train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
  if use_val:
    val_dataloader = DataLoader(val_dataset, shuffle=shuffle_datasets)
  test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)
  # Load the specified number of repeats
  for run in range(num_repeats):
    print(f'Repeat {run} of layer {15}')

    d_in = get_d_in(full_dataset)

    # Load probe based on expected naming format
    probe_name = f'probe_r2_{run}.pt'
    probe = load_probe_from_path(f'{probes_path}/{probe_name}', d_in=d_in)

    # Evaluate the loaded probe on test set
    if use_val:
      val_loss, val_r2, val_spearman, val_pearson = eval_regression_probe(probe, val_dataloader)
    else:
      val_loss = -1
    test_loss, test_r2, test_spearman, test_pearson = eval_regression_probe(probe, test_dataloader)

    # Add relevant data to the experiment results
    experiment_data.append({
        "run": run,
        "test_loss": test_loss,
        "final_val_loss": val_loss
      })

  experiment_data = pd.DataFrame(data=experiment_data)

  return experiment_data

def train_and_save_scalar_prediction_probe(target_layer_idx, target_feature, activations_path, \
                                         probe_name, probes_path, \
                                         lr, num_epochs, \
                                         r2_threshold=None, \
                                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False, use_solver=False):
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
  - test_r2 (float) : test R² score
  - test_spearman (float) : test Spearman rank correlation coefficient
  - test_pearson (float) : test Pearson correlation coefficient
  """

  # Test dataset, dataloaders, and splitting
  full_dataset = ActivationsDataset(activations_path=activations_path, feature_label=target_feature, layer_idx=target_layer_idx, r2_threshold=r2_threshold)
  train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
  train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=shuffle_datasets)
  if use_val:
    val_dataloader = DataLoader(val_dataset, shuffle=shuffle_datasets)
  test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)

  d_in = get_d_in(full_dataset)
  probe = LRProbe(d_in)

  # Training
  if use_solver: # Use direct solver for linear regression
    if use_val:
      probe, train_losses, val_losses = train_regression_probe_w_solver(probe, train_dataset, val_dataset=val_dataset)
    else:
      probe, train_losses, val_losses = train_regression_probe_w_solver(probe, train_dataloader, val_dataset=val_dataset)
  else: # Gradient-based training
    if use_val:
      probe, train_losses, val_losses = train_regression_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs, val_dataloader=val_dataloader)
    else:
      probe, train_losses, val_losses = train_regression_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs)

  # Evaluation on test set
  test_loss, test_r2, test_spearman, test_pearson = eval_regression_probe(probe, test_dataloader)
  print(f'Regression probe trained on layer {target_layer_idx}: Test Set Loss {test_loss}')

  # Save probe
  os.makedirs(probes_path, exist_ok=True)
  probe_path = os.path.join(probes_path, probe_name)
  save_probe_to_path(probe, probe_path)

  final_train_loss = train_losses[-1]
  final_val_loss = val_losses[-1]
  return test_loss, final_train_loss, final_val_loss, test_r2, test_spearman, test_pearson

def scalar_prediction_experiment(target_feature, activations_path, \
                         probes_path, \
                         lr, num_epochs, \
                         layers, num_repeats=1, \
                         r2_threshold=None, \
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
      test_loss, final_train_loss, final_val_loss, test_r2, test_spearman, test_pearson = train_and_save_scalar_prediction_probe(target_layer_idx=layer_idx, target_feature=target_feature, activations_path=activations_path, \
          probe_name=probe_name, probes_path=probes_path, \
          lr=lr, num_epochs=num_epochs, \
          r2_threshold=r2_threshold, \
          shuffle_datasets=shuffle_datasets, use_val=use_val, data_split=data_split, write_log=write_log)

      # Add relevant data to the experiment results
      experiment_data.append({
          "layer": layer_idx,
          "run": run,
          "test_loss": test_loss,
          "final_train_loss": final_train_loss,
          "final_val_loss": final_val_loss,
          "test_r2" : test_r2,
          "test_spearman" : test_spearman,
          "test_pearson" : test_pearson
        })

  experiment_data = pd.DataFrame(data=experiment_data)

  return experiment_data

def scalar_prediction_experiment_w_solver(target_feature, activations_path, \
                         probes_path, \
                         layers, \
                         r2_threshold=None, \
                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1]):
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
  for layer_idx in tqdm(layers, desc='\nTraining on each layer'):
    # Set probe name for saving
    probe_name = f'probe_{target_feature}_{layer_idx}_{0}.pt'
    
    # Train and save the probe for the correct layer
    test_loss, final_train_loss, final_val_loss, test_r2, test_spearman, test_pearson = train_and_save_scalar_prediction_probe(target_layer_idx=layer_idx, target_feature=target_feature, activations_path=activations_path, \
        probe_name=probe_name, probes_path=probes_path, \
        lr=1, num_epochs=1, \
        r2_threshold=r2_threshold, \
        shuffle_datasets=shuffle_datasets, use_val=use_val, data_split=data_split, use_solver=True)

    # Add relevant data to the experiment results
    experiment_data.append({
        "layer": layer_idx,
        "test_loss": test_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "test_r2" : test_r2,
        "test_spearman" : test_spearman,
        "test_pearson" : test_pearson,
      })

  experiment_data = pd.DataFrame(data=experiment_data)

  return experiment_data

def load_and_run_scalar_prediction_experiment(target_feature, activations_path, \
                         probes_path, \
                         layers, num_repeats=1, \
                         r2_threshold=None, \
                         shuffle_datasets=True, use_val=True, data_split=[0.8, 0.1, 0.1]):
  """
  Loads and evaluates regression probes on activations from a specified list of transformer layers to predict a scalar feature. Loads a number of probes up to num_repeats based on how many were repeats trained.

  Returns:
  - experiment_data (pd.DataFrame): A DataFrame containing results for each experiment run, with the following columns:
    - "run" (int): Experiment repetition index.
    - "test_loss" (float): Loss on the test dataset.

  Notes:
  - This function iterates over the specified `layers` and loads and evaluates a probe trained for each one.
  - Results are stored in a DataFrame for further analysis.
  """

  experiment_data = []

  # Iterate over the specified layers
  for layer_idx in layers:
    full_dataset = ActivationsDataset(activations_path=activations_path, feature_label=target_feature, layer_idx=layer_idx, r2_threshold=r2_threshold)
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, lengths=data_split)
    if use_val:
      val_dataloader = DataLoader(val_dataset, shuffle=shuffle_datasets)
    test_dataloader = DataLoader(test_dataset, shuffle=shuffle_datasets)
    # Load the specified number of repeats
    for run in range(num_repeats):
      print(f'Repeat {run} of layer {layer_idx}')

      d_in = get_d_in(full_dataset)

      # Load probe based on expected naming format
      probe_name = f'probe_{target_feature}_{layer_idx}_{run}.pt'
      probe = load_probe_from_path(f'{probes_path}/{probe_name}', d_in=d_in)

      # Evaluate the loaded probe on test set
      if use_val:
        val_loss, val_r2, val_spearman, val_pearson = eval_regression_probe(probe, val_dataloader)
      else:
        val_loss = -1
        val_r2 = -1
        val_spearman = -1
        val_pearson = -1
      test_loss, test_r2, test_spearman, test_pearson = eval_regression_probe(probe, test_dataloader)

      # Add relevant data to the experiment results
      experiment_data.append({
          "layer": layer_idx,
          "run": run,
          "val_loss": val_loss,
          "val_r2" : val_r2,
          "val_spearman" : val_spearman,
          "val_pearson" : val_pearson,
          "test_loss": test_loss,
          "test_r2" : test_r2,
          "test_spearman" : test_spearman,
          "test_pearson" : test_pearson
        })

  experiment_data = pd.DataFrame(data=experiment_data)

  return experiment_data