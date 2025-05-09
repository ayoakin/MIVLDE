import torch
import datetime
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge, RidgeClassifier
from src.probes.lr_probe import LRProbe



def eval_classifier_probe(probe, dataloader):
  '''
  Evaluate a given probe on a specified dataset (via its corresponding dataloader).

  Args:
  - probe (LRProbe): (classifier) probe to be evaluated
  - dataloader (torch.utils.data.DataLoader): dataloader for dataset on which the probe is to be evaluated

  Returns:
  - avg_loss (float): average loss (BCEWithLogitsLoss) on the given dataset
  - accuracy (float): accuracy of the probe on the given dataset
  - fail_ids (list(str)): IDs of datapoints which the probe fails to classify correctly
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
  
def eval_solver_classifier_probe(probe, dataset):
  '''
  Evaluate a given probe on a specified dataset (via its corresponding dataloader)
  which was trained using the direct solver.

  Args:
  - probe (LRProbe): (classifier) probe to be evaluated, trained using the direct solver
  - dataset (src.datasets.ActivationsDataset): dataset on which the probe is to be evaluated

  Returns:
  - avg_loss (float): average loss (BCEWithLogitsLoss) on the given dataset
  - accuracy (float): accuracy of the probe on the given dataset
  - fail_ids (list(str)): IDs of datapoints which the probe fails to classify correctly
  '''
  with torch.no_grad():
    total_loss = 0
    correct = 0
    total_preds = 0
    criterion = torch.nn.MSELoss(reduction='sum')
    # Corresponds to training objective in scikit-learn for RidgeClassifier (linear least squares)

    probe.eval()

    fail_ids = []

    # Iterate through the dataset using indices:
    # acts, labels, ids = [], [], []
    # for i in range(len(dataset)):
    #     act, label, id = dataset[i]
    #     acts.append(torch.tensor(act))
    #     labels.append(label.item())
    #     ids.append(id)

    dataloader = DataLoader(dataset, batch_size=len(dataset))

    for acts, labels, ids in dataloader:
      # Transform labels from {0,1} to {-1,1} to correspond with scikit-learn probe
      sk_labels = 2.0 * labels - 1.0
      
      outputs = probe(acts)
      loss = criterion(outputs, sk_labels)
      total_loss += loss.item()

      pred_labels = ((outputs + 1.0) / 2.0).round()
      correct += (pred_labels == labels).float().sum()
      total_preds += len(labels)

      fail_mask = (pred_labels != labels)
      fail_indices = fail_mask.nonzero()

      batch_fail_ids = [ids[fail_idx] for fail_idx in fail_indices]
      fail_ids += batch_fail_ids

    accuracy = (correct / total_preds).item()
    avg_loss = total_loss / total_preds
  
  return avg_loss, accuracy, fail_ids
  
def eval_regression_probe(probe, dataloader):
  '''
  Evaluate a given regression probe (e.g. for predicting R^2 score) on a specified dataset (via its corresponding dataloader).

  Args:
  - probe (LRProbe): (regression) probe to be evaluated
  - dataloader (torch.utils.data.DataLoader): dataloader for dataset on which the probe is to be evaluated

  Returns:
  - avg_loss (float): average loss (MSE loss) on the given dataset
  - r2 (float): R² score
  - spearman (float): Spearman rank correlation coefficient
  - pearson (float): Pearson correlation coefficient

  '''
  with torch.no_grad():
    total_loss = 0
    total_preds = 0
    all_preds = []
    all_labels = []
    # criterion = torch.nn.MSELoss()

    probe.eval()

    for acts, labels, ids in dataloader:
      outputs = probe(acts)
      # loss = criterion(outputs, labels)
      diff = (labels - outputs)
      total_loss += torch.square(diff).sum().item()

      total_preds += len(labels)

      if outputs.ndim == 0:  
          all_preds.append(outputs.item())
          all_labels.append(labels.item())
      else:  
          all_preds.extend(outputs.cpu().numpy())  
          all_labels.extend(labels.cpu().numpy())  


    avg_loss = total_loss / total_preds
    r2 = r2_score(all_labels, all_preds)
    spearman, _ = spearmanr(all_labels, all_preds)
    pearson, _ = pearsonr(all_labels, all_preds)

    return avg_loss, r2, spearman, pearson

def verbose_eval_regression_probe(probe, dataloader):
  '''
  Evaluate a given regression probe (e.g. for predicting R^2 score) on a specified dataset (via its corresponding dataloader).

  Args:
  - probe (LRProbe): (regression) probe to be evaluated
  - dataloader (torch.utils.data.DataLoader): dataloader for dataset on which the probe is to be evaluated

  Returns:
  - eval_results (list[list]): 
  - avg_loss (float): average loss (MSE loss) on the given dataset
  '''
  with torch.no_grad():
    total_loss = 0
    total_preds = 0

    eval_results = []

    probe.eval()

    for acts, labels, ids in dataloader:
      outputs = probe(acts)
      diff = (labels - outputs)
      square_errors = torch.square(diff).item()
      if len(acts) == 1:
        datapoint = [outputs, labels, ids, square_errors]
        eval_results.append(datapoint)
      else:
        for batch_idx in range(len(acts)):
          datapoint = [outputs[batch_idx], labels[batch_idx], ids[batch_idx], square_errors[batch_idx]]
          eval_results.append(datapoint)
      total_loss += torch.square(diff).sum().item()

      total_preds += len(labels)

    avg_loss = total_loss / total_preds
    return eval_results, avg_loss

def train_classifier_probe(probe, train_dataloader, val_dataloader=None, \
                lr=0.01, num_epochs=20, device='cpu', \
                logs_path='/content/drive/MyDrive/aisc/logs', write_log=False): # TODO: determine if default hyperparameters are good
  '''
  Train an instantiated classifier probe using specified training and validation data.

  Args:
  - probe (LRProbe): (classifier) probe to be trained
  - train_dataloader (torch.utils.data.DataLoader): dataloader for the training set
  - val_dataloader (torch.utils.data.DataLoader | bool, optional): dataloader for the validation set. Default is None
  - lr (float, optional): learning rate for training. Default is 0.01
  - num_epochs (int, optional): number of epochs for training. Default is 20
  - device (str, optional): device on which to run training. Default is 'cpu'
  - logs_path (str, optional): path to which log files should be saved, if desired. Default is '/content/drive/MyDrive/aisc/logs'
  - write_log (bool, optional): whether to write training logs to a file in logs_path. Default is False

  Returns:
  - probe (LRProbe): a trained (classifier) probe
  - losses (list(float)): list of losses at each epoch on the training set
  - accuracies (list(float)): list of accuracies at each epoch on the training set
  - val_losses (list(float)): list of losses at each epoch on the validation set
  - val_accuracies (list(float)): list of accuracies at each epoch on the validation set
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
  e0_train_loss, e0_train_acc, e0_train_fail_ids = eval_classifier_probe(probe, train_dataloader)
  losses.append(e0_train_loss)
  accuracies.append(e0_train_acc)
  if val_dataloader is not None:
    e0_val_loss, e0_val_acc, e0_val_fail_ids = eval_classifier_probe(probe, val_dataloader)
    val_losses.append(e0_val_loss)
    val_accuracies.append(e0_val_acc)

  # Main training loop
  for epoch in tqdm(range(num_epochs), desc='\nTraining LR Probe'):
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
        avg_val_loss, val_accuracy, val_fail_ids = eval_classifier_probe(probe, val_dataloader)

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        if write_log:
          val_f.write(f'Epoch {epoch+1} (Validation): Loss {avg_val_loss}, Accuracy {val_accuracy}\n')
        # print(f' Epoch {epoch+1} (Validation): Loss {avg_val_loss}, Accuracy {val_accuracy.item()}\n')
  print(f'\nTraining Set (Epoch {epoch+1} - Final): Loss {avg_loss}, Accuracy {accuracy}')

  return probe, losses, accuracies, val_losses, val_accuracies

def train_classifier_probe_w_solver(probe, train_dataset, val_dataset=None):
  '''
  Train an instantiated classifier probe using specified training and validation data.

  Args:
  - probe (LRProbe): (classifier) probe to be trained
  - train_dataset (src.datasets.ActivationsDataset): the full training dataset
  - val_dataset (src.datasets.ActivationsDataset | bool, optional): the full validation dataset. Default is None

  Returns:
  - probe (LRProbe): a trained (classifier) probe
  - losses (list(float)): list of losses at each epoch on the training set
  - accuracies (list(float)): list of accuracies at each epoch on the training set
  - val_losses (list(float)): list of losses at each epoch on the validation set
  - val_accuracies (list(float)): list of accuracies at each epoch on the validation set
  '''

  losses = []
  accuracies = []
  val_losses = []
  val_accuracies = []

  # Initial performance
  e0_train_loss, e0_train_acc, e0_train_fail_ids = eval_solver_classifier_probe(probe, train_dataset)
  losses.append(e0_train_loss)
  accuracies.append(e0_train_acc)
  if val_dataset is not None:
    e0_val_loss, e0_val_acc, e0_val_fail_ids = eval_solver_classifier_probe(probe, val_dataset)
    val_losses.append(e0_val_loss)
    val_accuracies.append(e0_val_acc)

  train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))

  # There should only be one batch since the batch size is set to the length
  for acts, labels, ids in train_dataloader:
    acts = acts.numpy()
    labels = labels.numpy()

    sklearn_classifier_probe = RidgeClassifier(alpha=0.01, fit_intercept=False)
    sklearn_classifier_probe.fit(acts, labels)

    coeffs = sklearn_classifier_probe.coef_

    with torch.no_grad():
      probe.hidden.weight.copy_(torch.tensor(coeffs).unsqueeze(0))

    # Trained performance
    e1_train_loss, e1_train_acc, e1_train_fail_ids = eval_solver_classifier_probe(probe, train_dataset)
    losses.append(e1_train_loss)
    accuracies.append(e1_train_acc)
    if val_dataset is not None:
      e1_val_loss, e1_val_acc, e1_val_fail_ids = eval_solver_classifier_probe(probe, val_dataset)
      val_losses.append(e1_val_loss)
      val_accuracies.append(e1_val_acc)

    # Break in case there are more batches...
    break

  return probe, losses, accuracies, val_losses, val_accuracies

def train_regression_probe(probe, train_dataloader, val_dataloader=None, \
                           lr=0.01, num_epochs=20, device='cpu', \
                           logs_path='/content/drive/MyDrive/aisc/logs', write_log=False):
  '''
  Train an instantiated regression probe using specified training and validation data.

  Args:
  - probe (LRProbe): (regression) probe to be trained
  - train_dataloader (torch.utils.data.DataLoader): dataloader for the training set
  - val_dataloader (torch.utils.data.DataLoader | bool, optional): dataloader for the validation set. Default is None
  - lr (float, optional): learning rate for training. Default is 0.01
  - num_epochs (int, optional): number of epochs for training. Default is 20
  - device (str, optional): device on which to run training. Default is 'cpu'
  - logs_path (str, optional): path to which log files should be saved, if desired. Default is '/content/drive/MyDrive/aisc/logs'
  - write_log (bool, optional): whether to write training logs to a file in logs_path. Default is False

  Returns:
  - probe (LRProbe): a trained (regression) probe
  - losses (list(float)): list of losses at each epoch on the training set
  - val_losses (list(float)): list of losses at each epoch on the validation set
  '''
  # Use Adam optimizer for now
  # TODO: investigate adding learning rate scheduler (e.g. cosine annealing?)
  opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-3)
  # TODO: investigate if weight decay is actually necessary
  criterion = torch.nn.MSELoss()

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
  val_losses = []

  # Epoch 0 (for comparison against epoch 1)
  e0_train_loss = eval_regression_probe(probe, train_dataloader)
  losses.append(e0_train_loss)
  if val_dataloader is not None:
    e0_val_loss = eval_regression_probe(probe, val_dataloader)
    val_losses.append(e0_val_loss)

  # Main training loop
  for epoch in tqdm(range(num_epochs), desc='\nTraining LR Probe'):
    probe.train()
    total_loss = 0
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

    # Calculate epoch stats
    avg_loss = total_loss / total_preds

    losses.append(avg_loss)

    # Write to specified log file
    if write_log:
      train_f.write(f'Epoch {epoch+1}: Loss {avg_loss}')

    # Run evaluation on validation set
    # TODO: maybe implement early stopping? Need to test on larger dataset
    if val_dataloader is not None:
        avg_val_loss = eval_regression_probe(probe, val_dataloader)
        val_losses.append(avg_val_loss)

        if write_log:
          val_f.write(f'Epoch {epoch+1} (Validation): Loss {avg_val_loss}\n')
  print(f'\nTraining Set (Epoch {epoch+1} - Final): Loss {avg_loss}')

  return probe, losses, val_losses

def train_regression_probe_w_solver(probe, train_dataset, val_dataset=None):
  '''
  Train an instantiated regression probe using specified training and validation data.

  Args:
  - probe (LRProbe): (regression) probe to be trained
  - train_dataset (src.datasets.ActivationsDataset): the full training dataset
  - val_dataset (src.datasets.ActivationsDataset | bool, optional): the full validation dataset. Default is None

  Returns:
  - probe (LRProbe): a trained (regression) probe
  - loss (float): list of losses at each epoch on the training set
  - val_loss (float): list of losses at each epoch on the validation set
  '''

  losses = []
  val_losses = []

  train_dataloader = DataLoader(train_dataset)
  if val_dataset is not None:
    val_dataloader = DataLoader(val_dataset)

  # Initial performance
  e0_train_loss, e0_train_r2, e0_train_spearman, e0_train_pearson = eval_regression_probe(probe, train_dataloader)
  losses.append(e0_train_loss)
  if val_dataset is not None:
    e0_val_loss, e0_val_r2, e0_val_spearman, e0_val_pearson = eval_regression_probe(probe, val_dataloader)
    val_losses.append(e0_val_loss)

  train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset))

  # There should only be one batch since the batch size is set to the length
  for acts, labels, ids in train_dataloader:
    acts = acts.numpy()
    labels = labels.numpy()

    sklearn_probe = Ridge(alpha=0.01, fit_intercept=False)
    sklearn_probe.fit(acts, labels)

    coeffs = sklearn_probe.coef_

    with torch.no_grad():
      probe.hidden.weight.copy_(torch.tensor(coeffs).unsqueeze(0))

    # Trained performance
    e1_train_loss, e1_train_r2, e1_train_spearman, e1_train_pearson = eval_regression_probe(probe, train_dataloader)
    losses.append(e1_train_loss)
    if val_dataset is not None:
      e1_val_loss, e1_val_r2, e1_val_spearman, e1_val_pearson = eval_regression_probe(probe, val_dataloader)
      val_losses.append(e1_val_loss)

    # Break in case there are more batches...
    break

  return probe, losses, val_losses

def save_probe_to_path(probe, probe_path):
  '''
  Save a probe's state dictionary to a specified path
  (saving only the state dictionary is suggested by PyTorch)

  Args:
  - probe (LRProbe): probe to be saved
  - probe_path (str): path to which the probe should be saved (incl. filename)
  '''
  torch.save(probe.state_dict(), probe_path)
  print(f'Saved state dictionary to {probe_path}')

# TODO: make the following function general enough to work with other probes (if needed)
def load_probe_from_path(probe_path, d_in=512):
  '''
  Returns a probe ready for evaluation loaded from the given path, with specified input dimension

  Args:
  - probe_path (str): path from which the probe should be loaded
  - d_in (int): dimension of data which is input to the probe
  '''
  probe = LRProbe(d_in=d_in)
  probe.load_state_dict(torch.load(probe_path, weights_only=True))
  probe.eval()
  return probe