import os
import pickle
import re
import torch
from torch.utils.data import Dataset

class ActivationsDataset(Dataset):
  """
  A PyTorch dataset class for loading and processing activations extracted from the odeformer model.

  This dataset loads activation tensors from saved `.pt` files and retrieves corresponding labels
  for supervised learning tasks.

  Args:
  - activations_path (str): Path to the directory containing activation files.
  - feature_label (str): The key used to extract the target feature from the activation metadata.
  - layer_idx (int): The index of the layer from which activations should be extracted.
  - module (str, optional): The type of module to extract activations from (default: 'ffn').

  Methods:
  - __len__(): Returns the number of activation samples.
  - __getitem__(idx): Loads and returns an activation sample, its corresponding label, and its unique ID.
  - get_layer_name(idx): Returns the formatted layer name corresponding to a given index.
  - get_id_from_path(act_path): Extracts and returns the sample ID from the activation file path.
  """

  def __init__(self, activations_path, feature_label, layer_idx, module='ffn'):
    """
    Initialises the dataset by collecting activation file paths and setting parameters.

    Args:
    - activations_path (str): Path to the directory containing activation files.
    - feature_label (str): The key used to extract the target feature from the activation metadata.
    - layer_idx (int): The index of the layer from which activations should be extracted.
    - module (str, optional): The type of module to extract activations from (default: 'ffn').
    """
    self.act_paths = [os.path.join(activations_path, f) for f in os.listdir(activations_path)]
    self.feature_label = feature_label
    self.layer_idx = layer_idx
    self.module = module

  def __len__(self):
    """
    Returns the total number of activation files in the dataset.

    Returns:
    - int: Number of activation samples.
    """
    return len(self.act_paths)

  def __getitem__(self, idx):
    """
    Loads an activation sample, its corresponding label, and its ID.

    Args:
    - idx (int): Index of the activation file to retrieve.

    Returns:
    - act_data (torch.Tensor): The extracted activation tensor.
    - act_label (torch.Tensor): The corresponding label tensor for supervised learning.
    - act_id (str): The unique identifier of the activation sample.

    Raises:
    - ValueError: If the activation shape does not match the expected dimensions.
    """
    act_path = self.act_paths[idx]
    with open(act_path, 'rb') as f:
      activation = pickle.load(f)
    layer_name = self.get_layer_name(self.layer_idx)
    if 'encoder' in layer_name:
      act_data = activation['encoder'][layer_name]
    else:
      act_data = activation['decoder'][layer_name]
    # # TODO: need to update the below functionality to be neater (needs to work with activation generation script)
    # if act_data.shape[0] != 512: # TODO: need to update this for when we consider multiple beams
    #   act_data = act_data[-1, :, :].flatten()
    act_data = act_data[:, :, :].flatten()  
    act_label = torch.tensor(activation['feature_dict'][self.feature_label], dtype=torch.float)
    act_id = self.get_id_from_path(act_path)
    return act_data, act_label, act_id

  def get_layer_name(self, idx):
    """
    Returns the formatted layer name for the given index.

    Args:
    - idx (int): The index of the layer.

    Returns:
    - str: The corresponding layer name.

    Raises:
    - ValueError: If the provided index is out of the expected range (-16 to 15).
    """
    layers = [f'encoder_{self.module}_{num}' for num in range(4)] + [f'decoder_{self.module}_{num}' for num in range(12)]
    layer_name = layers[idx]
    if -16 <= idx < 16:
      return layer_name
    else:
      raise ValueError("Layer index should be in -16 to 15")
  
  def get_id_from_path(self, act_path):
    """
    Extracts and returns the sample ID from the activation file path.

    Args:
    - act_path (str): The file path of the activation.

    Returns:
    - str: The extracted sample ID.

    Raises:
    - ValueError: If the activation filename does not match the expected format.
    """
    act_filename = act_path.split('/')[-1]
    match = re.match(r'activation_([a-zA-Z0-9]+_\d+)\.pt', act_filename)
    if match:
      return match.group(1)
    else:
      raise ValueError(f"Activation filename not formatted as expected. Expected format: activation_[descriptor]_[index].pt, Actual format: {act_filename}")
    

class R2ActivationsDataset(ActivationsDataset):
  """
  A specialized dataset class that extends `ActivationsDataset`, filtering activations based on R^2 score.

  This dataset is used for loading activations while ensuring that samples with an infinite R^2 score are excluded.
  Also allows for activations with r2_score below a specified threshold to be discarded.

  Args:
  - activations_path (str): Path to the directory containing activation files.
  - r2_threshold (float, optional): Threshold value for r2_score, below which scores should be discarded (default: None).
  - module (str, optional): The type of module to extract activations from (default: 'ffn').

  Methods:
  - __len__(): Returns the number of valid activation samples.
  - __getitem__(idx): Retrieves an activation sample, its label, and its ID from the filtered dataset.

  Notes:
  - Recall that r2_score should vary from (-inf, 1] when setting a value for the threshold
  """

  def __init__(self, activations_path, r2_threshold=None, module='ffn'):
    """
    Initializes the dataset by filtering out samples with infinite R^2 scores.

    Args:
    - activations_path (str): Path to the directory containing activation files.
    - r2_threshold (float, optional): Threshold value for r2_score, below which scores should be discarded (default: None).
    - module (str, optional): The type of module to extract activations from (default: 'ffn').

    Notes:
    - Filters out activation files where the `r2_score` is infinite.
    - Inherits from `ActivationsDataset`, setting `feature_label='r2_score'` and `layer_idx=-1`.
    """
    all_paths = [os.path.join(activations_path, f) for f in os.listdir(activations_path)]
    
    # Filter paths based on r2_score
    filtered_paths = []
    for path in all_paths:
        try:
            with open(path, 'rb') as f:
                activation = pickle.load(f)
            feature_value = activation['feature_dict']['r2_score']

            allow_path = True            
            
            # Exclude paths where r2_score is inf
            if torch.isinf(torch.tensor(feature_value, dtype=torch.float)).item():
              allow_path = False
            
            # Exclude paths where r2_score falls below a specified threshold (optional)
            if r2_threshold is not None and torch.tensor(feature_value, dtype=torch.float) < r2_threshold:
              allow_path = False

            # Append path if valid
            if allow_path:
              filtered_paths.append(path)
        except Exception as e:
            print(f"Warning: Failed to process {path} due to {e}")

    # Initialise parent class with filtered paths
    super().__init__(activations_path, feature_label='r2_score', layer_idx=-1, module=module)
    self.act_paths = filtered_paths

  def __len__(self):
    """
    Returns the number of valid activation samples (excluding those with infinite R^2 scores).

    Returns:
    - int: Number of filtered activation samples.
    """
    return super().__len__()

  def __getitem__(self, idx):
    """
    Retrieves an activation sample, its label, and its ID from the filtered dataset.

    Args:
    - idx (int): Index of the activation file to retrieve.

    Returns:
    - Tuple: (torch.Tensor, torch.Tensor, str) -> (activation tensor, label tensor, sample ID).
    """
    return super().__getitem__(idx)