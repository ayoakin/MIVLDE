import torch
from torch.utils.data import random_split

def split_dataset(dataset, lengths=[0.8, 0.0, 0.2], seed=42):
  '''
  Split into training, validation, and testing datasets
  Default is to have no validation dataset (i.e. empty) and randomized splitting
  Seed can be set for deterministic splitting
  '''
  if seed is not None:
    generator = torch.Generator().manual_seed(seed)
  else:
    generator = torch.Generator()
  return random_split(dataset, lengths, generator)

def get_d_in(dataset):
  '''
  Return the input dimension a probe requires for a given dataset of activations
  '''
  d_in = dataset[0][0].shape[0]
  return d_in