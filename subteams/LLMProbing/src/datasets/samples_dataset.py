import os
import pickle
import re
from torch.utils.data import Dataset

class SamplesDataset(Dataset):
  def __init__(self, samples_path):
    self.sample_paths = [os.path.join(samples_path, f) for f in os.listdir(samples_path)]

  def __len__(self):
    return len(self.sample_paths)

  def __getitem__(self, idx):
    sample_path = self.sample_paths[idx]
    with open(sample_path, 'rb') as f:
      sample = pickle.load(f)
    sample_id = self.get_id_from_path(sample_path)
    return sample, sample_id
  
  def get_id_from_path(self, sample_path):
    '''
    Helper function to return the id of the current sample from its path
    '''
    sample_filename = sample_path.split('/')[-1]
    match = re.match(r'sample_([a-zA-Z0-9]+_\d+)\.pt', sample_filename)
    if match:
      return match.group(1)
    else:
      raise ValueError(f"Sample filename not formatted as expected. Expected format: sample_[descriptor]_[index].pt, Actual format: {sample_filename}")