import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import gzip
import pickle
from tqdm import tqdm
import os
from datetime import datetime


def load_activations(filepath):
    print(f"Loading activations from {filepath}...")
    with gzip.open(filepath, 'rb') as file:
        collected_act = pickle.loads(file.read())
    print(f"Loaded {len(collected_act)} samples")
    return collected_act


def load_data_and_model(data_path, model_path, model_class):
    print(f"Loading data from {data_path}...")
    with gzip.open(data_path, 'rb') as file:
        collected_act = pickle.loads(file.read())
    print(f"Loaded {len(collected_act)} samples")
    
    print(f"Loading model from {model_path}...")
    model = model_class(input_dim=256, latent_dim=1024)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return collected_act, model


def prepare_residual_stream_data(collected_act, site_name='residual_stream',layer_name='encoder.outer.residual1'):
    training_data = []
    total_vectors = 0
    
    residual_shapes = {}
    
    for i, sample in enumerate(collected_act):
        if not (site_name in sample[1] and layer_name in sample[1][site_name]):
            continue
            
        residual_data = sample[1][site_name][layer_name]
        
        tensor_found = False
        for key in residual_data.keys():
            if key in residual_shapes:
                residual_shapes[key] += 1
            else:
                residual_shapes[key] = 1

            if isinstance(key, tuple) and len(key) == 3 and key[0] == 1 and key[2] == 256:
                tensor_found = True
                
                tensor = residual_data[key]
                if isinstance(tensor, torch.Tensor):
                    tensor = tensor.numpy()
                
                try:
                    vectors = tensor.reshape(-1, 256)
                    training_data.append(vectors)
                    total_vectors += vectors.shape[0]
                except Exception as e:
                    print(f"Error reshaping tensor with shape {tensor.shape}: {e}")

    if training_data:
        print(f"Final dataset: {len(training_data)} samples, {total_vectors} total vectors")
        return np.concatenate(training_data)
    else:
        raise ValueError("Could not find activations with the required structure in the data")

def extract_activations_and_trajectories(model, collected_data, layer_name="encoder.outer.residual3"):
    device = next(model.parameters()).device
    model.eval()

    samples = []
    
    for i, sample in tqdm(enumerate(collected_data), desc="Extracting activations"):
        try:
            solution = sample[0]['solution']
            time_points = sample[0]['time_points']
            equation = sample[0].get('equations', None)
            
            residual_data = sample[1]['residual_stream'][layer_name]
            
            for key in residual_data.keys():
                traj_residuals = residual_data[key]
                if isinstance(traj_residuals, torch.Tensor):
                    traj_residuals = traj_residuals.numpy()
                
                traj_residuals = traj_residuals.reshape(-1, 256)
                
                with torch.no_grad():
                    inputs = torch.tensor(traj_residuals, dtype=torch.float32).to(device)
                    model_output = model(inputs)
                    activations_np = model_output[1].squeeze(0).cpu().numpy()
                    mean_activation = activations_np.mean(axis=0)
                    max_activation = activations_np.max(axis=0)
                
                sample_data = {
                    'trajectory': solution,
                    'time_points': time_points,
                    'equation': equation,
                    'activations': activations_np,
                    'mean_activation': mean_activation,
                    'max_activation': max_activation
                }

                samples.append(sample_data)
                break
                
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"Extracted data from {len(samples)} trajectories")
    
    return samples
