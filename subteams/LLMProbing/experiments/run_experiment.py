import os
import pandas as pd
from torch.utils.data import DataLoader
from src.datasets.activations_dataset import ActivationsDataset
from src.datasets.utils import split_dataset, get_d_in
from src.probes.lr_probe import LRProbe
from src.probes.utils import train_probe, eval_probe, save_probe_to_path

def train_and_save_probe_separation_expt(target_layer_idx, target_feature, activations_path, \
                                         probe_name, probes_path, \
                                         lr, num_epochs, \
                                         shuffle_datasets = True, use_val = True, data_split=[0.8, 0.1, 0.1], write_log=False):
  """
  TODO: add description
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
    probe, train_losses, train_accuracies, val_losses, val_accuracies = train_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs, val_dataloader=val_dataloader)
  else:
    probe, train_losses, train_accuracies, val_losses, val_accuracies = train_probe(probe, train_dataloader, lr=lr, write_log=write_log, num_epochs=num_epochs)

  # Evaluation on test set
  test_loss, test_acc, test_fail_ids = eval_probe(probe, test_dataloader)
  print(f'Probe trained on layer {target_layer_idx}:')
  print(f'Test Set: Loss {test_loss}, Accuracy {test_acc}')

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
  TODO: add description
  """
  experiment_data = []

  for layer_idx in layers:
    for run in range(num_repeats):
      probe_name = f'probe_{target_feature}_{layer_idx}_{run}.pt'
      test_loss, test_acc, test_fail_ids, final_train_loss, final_train_acc, \
        final_val_loss, final_val_acc = train_and_save_probe_separation_expt(target_layer_idx=layer_idx, target_feature=target_feature, activations_path=activations_path, \
                                                                             probe_name=probe_name, probes_path=probes_path, \
                                                                             lr=lr, num_epochs=num_epochs, \
                                                                             shuffle_datasets=shuffle_datasets, use_val=use_val, data_split=data_split, write_log=write_log)
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