import os
import pickle
import re
import torch
from torch.utils.data import Dataset

class ActivationsDataset(Dataset):
  def __init__(self, activations_path, feature_label, layer_idx, module='ffn'):
    self.act_paths = [os.path.join(activations_path, f) for f in os.listdir(activations_path)]
    self.feature_label = feature_label
    self.layer_idx = layer_idx
    self.module = module

  def __len__(self):
    return len(self.act_paths)

  def __getitem__(self, idx):
    act_path = self.act_paths[idx]
    with open(act_path, 'rb') as f:
      activation = pickle.load(f)
    layer_name = self.get_layer_name(self.layer_idx)
    if 'encoder' in layer_name:
      act_data = activation['encoder'][layer_name]
    else:
      act_data = activation['decoder'][layer_name]
    # TODO: need to update the below functionality to be neater (needs to work with activation generation script)
    if act_data.shape[0] != 512: # TODO: need to update this for when we consider multiple beams
      act_data = act_data[-1, :, :].flatten()
    act_label = torch.tensor(activation['feature_dict'][self.feature_label], dtype=torch.float)
    act_id = self.get_id_from_path(act_path)
    return act_data, act_label, act_id

  def get_layer_name(self, idx):
    '''
    Helper function to return the correct name of a layer in the ODEFormer given
    its index
    '''
    layers = [f'encoder_{self.module}_{num}' for num in range(4)] + [f'decoder_{self.module}_{num}' for num in range(12)]
    layer_name = layers[idx]
    if -16 <= idx < 16:
      return layer_name
    else:
      raise ValueError("Layer index should be in -16 to 15")
  
  def get_id_from_path(self, act_path):
    '''
    Helper function to return the id of the current sample from its path
    '''
    act_filename = act_path.split('/')[-1]
    act_id = re.findall(r'\d+', act_filename)[0]
    return act_id