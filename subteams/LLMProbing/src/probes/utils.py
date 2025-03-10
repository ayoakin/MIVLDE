import torch
import datetime
import os
from tqdm import tqdm
from src.probes.lr_probe import LRProbe

def eval_probe(probe, dataloader):
  '''
  Evaluate a given probe on a specified dataset (via its corresponding dataloader)
  '''
  with torch.no_grad():
    total_loss = 0
    correct = 0
    total_preds = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    probe.eval()

    fail_ids = []

    for acts, labels, ids in dataloader:
      outputs = probe(acts)
      loss = criterion(outputs, labels)
      total_loss += loss.item()

      pred_labels = torch.nn.functional.sigmoid(outputs).round()
      correct += (pred_labels == labels).float().sum()
      total_preds += len(labels)

      fail_mask = (pred_labels != labels)
      fail_indices = fail_mask.nonzero()

      batch_fail_ids = [ids[fail_idx] for fail_idx in fail_indices]
      fail_ids += batch_fail_ids

    accuracy = (correct / total_preds).item()
    avg_loss = total_loss / total_preds
    return avg_loss, accuracy, fail_ids

def train_probe(probe, train_dataloader, val_dataloader=None, \
                lr=0.01, num_epochs=20, device='cpu', \
                logs_path='/content/drive/MyDrive/aisc/logs', write_log=False): # TODO: determine if default hyperparameters are good
  '''
  Train an instantiated probe using specified training and validation data
  '''
  # Use Adam optimizer for now
  # TODO: investigate adding learning rate scheduler (e.g. cosine annealing?)
  opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-3)
  # TODO: investigate if weight decay is actually necessary
  criterion = torch.nn.BCEWithLogitsLoss()

  # Open log files to write to if desired
  # Include the current time of the experiment in filename to avoid collisions
  today_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
  if write_log:
    train_f = open(os.path.join(logs_path, f'{today_str}_train_acc_per_epoch.txt'), 'w')
    train_f.write(f'Learning rate: {lr}, Num. epochs: {num_epochs}\n')
    if val_dataloader is not None:
      val_f = open(os.path.join(logs_path, f'{today_str}_val_acc_per_epoch.txt'), 'w')
      val_f.write(f'Learning rate: {lr}, Num. epochs: {num_epochs}\n')

  losses = []
  accuracies = []
  val_losses = []
  val_accuracies = []

  # Epoch 0 (for comparison against epoch 1)
  e0_train_loss, e0_train_acc, e0_train_fail_ids = eval_probe(probe, train_dataloader)
  losses.append(e0_train_loss)
  accuracies.append(e0_train_acc)
  if val_dataloader is not None:
    e0_val_loss, e0_val_acc, e0_val_fail_ids = eval_probe(probe, val_dataloader)
    val_losses.append(e0_val_loss)
    val_accuracies.append(e0_val_acc)

  # Main training loop
  for epoch in tqdm(range(num_epochs), desc='Training LR Probe'):
    probe.train()
    total_loss = 0
    correct_preds = 0
    total_preds = 0

    for i, (train_acts, train_labels, train_fail_ids) in enumerate(train_dataloader):
      # Zero the gradients
      opt.zero_grad()

      # Calculate loss
      outputs = probe(train_acts)
      loss = criterion(outputs, train_labels)

      loss.backward()

      # Update model weights
      opt.step()

      total_loss += loss.item()
      total_preds += len(train_labels)

      # Calculate correct batch predictions
      pred_labels = torch.nn.functional.sigmoid(outputs).round()
      correct_preds += (pred_labels == train_labels).float().sum()

    # Calculate epoch stats
    accuracy = (correct_preds / total_preds).item()
    avg_loss = total_loss / total_preds

    losses.append(avg_loss)
    accuracies.append(accuracy)

    # Write to specified log file
    if write_log:
      train_f.write(f'Epoch {epoch+1}: Loss {avg_loss}, Accuracy {accuracy}\n')
    # print(f' Epoch {epoch+1}: Loss {avg_loss}, Accuracy {accuracy.item()}\n')

    # Run evaluation on validation set
    # TODO: maybe implement early stopping? Need to test on larger dataset
    if val_dataloader is not None:
        avg_val_loss, val_accuracy, val_fail_ids = eval_probe(probe, val_dataloader)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        if write_log:
          val_f.write(f'Epoch {epoch+1} (Validation): Loss {avg_val_loss}, Accuracy {val_accuracy}\n')
        # print(f' Epoch {epoch+1} (Validation): Loss {avg_val_loss}, Accuracy {val_accuracy.item()}\n')
  print(f'\nEpoch {epoch+1} (Final): Loss {avg_loss}, Accuracy {accuracy}')

  return probe, losses, accuracies, val_losses, val_accuracies

def save_probe_to_path(probe, probe_path):
  '''
  Save a probe's state dictionary to a specified path
  (saving only the state dictionary is suggested by PyTorch)
  '''
  torch.save(probe.state_dict(), probe_path)
  print(f'Saved state dictionary to {probe_path}')

# TODO: make the following function general enough to work with other probes (if needed)
def load_probe_from_path(probe_path, d_in=512):
  '''
  Returns a probe ready for evaluation loaded from the given path, with specified input dimension
  '''
  probe = LRProbe(d_in=d_in)
  probe.load_state_dict(torch.load(probe_path, weights_only=True))
  probe.eval()
  return probe