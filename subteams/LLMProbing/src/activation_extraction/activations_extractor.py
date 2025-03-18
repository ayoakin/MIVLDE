import os
import io
import pickle
import numpy as np
import torch
from odeformer.metrics import r2_score
from contextlib import redirect_stdout
from tqdm import tqdm
from src.datasets import SamplesDataset

class ActivationsExtractor():
    def __init__(self):
        pass

    # Function to store the output of each layer
    def hook_fn(self, module, input, output, layer_name, layer_outputs):
        layer_outputs[layer_name] = output.detach().cpu() #  detach to avoid unnecessary gradient tracking, and move to store in cpu

    # Registering hooks for layers in the encoder and decoder
    def register_hooks(self, model_part, part_name, layer_outputs, layers_to_extract):
        if 'mha' in layers_to_extract:
            for idx, module in enumerate(model_part.attentions):  # MultiHeadAttention layers
                layer_name = f"{part_name}_attention_{idx}"
                module.register_forward_hook(lambda module, input, output, name=layer_name: self.hook_fn(module, input, output, name, layer_outputs))

        if 'ffn' in layers_to_extract:
            for idx, module in enumerate(model_part.ffns):  # FeedForward layers
                layer_name = f"{part_name}_ffn_{idx}"
                module.register_forward_hook(lambda module, input, output, name=layer_name: self.hook_fn(module, input, output, name, layer_outputs))

        if 'ln1' in layers_to_extract:
            for idx, module in enumerate(model_part.layer_norm1):  # LayerNorm 1 layers
                layer_name = f"{part_name}_layer_norm1_{idx}"
                module.register_forward_hook(lambda module, input, output, name=layer_name: self.hook_fn(module, input, output, name, layer_outputs))

        if 'ln2' in layers_to_extract:
            for idx, module in enumerate(model_part.layer_norm2):  # LayerNorm 2 layers
                layer_name = f"{part_name}_layer_norm2_{idx}"
                module.register_forward_hook(lambda module, input, output, name=layer_name: self.hook_fn(module, input, output, name, layer_outputs))

    def extract_activations(self, dstr, samples_path, activations_path, layers_to_extract=['ffn']): # We currently look at ouputs of ffn layers since they come before layer norm
        # Setup
        layer_outputs = {}
        os.makedirs(activations_path, exist_ok=True)

        # Construct SamplesDataset from path
        samples = SamplesDataset(samples_path)

        # Register hooks in odeformer
        self.register_hooks(dstr.model.encoder, 'encoder', layer_outputs, layers_to_extract)
        self.register_hooks(dstr.model.decoder, 'decoder', layer_outputs, layers_to_extract)
        
        # Loop over all samples in the dataset
        for test_sample, test_sample_id in tqdm(samples, desc='Extracting Activations'):
            # Use same ID as sample for filename
            activation_filename = f'activation_{test_sample_id}.pt'
            activation_filepath = os.path.join(activations_path, activation_filename)

            # Check if activation file already exists
            if not os.path.exists(activation_filepath):
              # print(f"\nProcessing sample: {test_sample_id}")
              
              # Fit odeformer
              with torch.no_grad():
                  dstr.fit(test_sample['times'], test_sample['trajectory'])
              
              # Get outputs of specified layer parts
              encoder_layer_outputs = {}
              decoder_layer_outputs = {}
              for layer_name, output in layer_outputs.items():
                  if 'ffn' in layer_name: # TODO: may need to change this in future to work with layers_to_extract (if we want it to be more general)
                      if 'encoder' in layer_name:
                          encoder_layer_outputs[layer_name] = output
                      if 'decoder' in layer_name:
                          decoder_layer_outputs[layer_name] = output

              # Add relevant info to activations dict
              activations = {}
              activations['encoder'] = encoder_layer_outputs
              activations['decoder'] = decoder_layer_outputs
              if 'operator_dict' in test_sample:
                  activations['operator_dict'] = test_sample['operator_dict']
              activations['feature_dict'] = test_sample['feature_dict']
              
              # Compute and add R^2 score (this adds a little extra overhead each iteration)
              pred_trajectory = dstr.predict(test_sample['times'], test_sample['trajectory'][0])
              if pred_trajectory is None or np.isnan(pred_trajectory).any():
                print(f'\nnan in trajectory of sample {test_sample_id}')
                test_r2 = float('-inf')
              else:
                test_r2 = r2_score(test_sample['trajectory'], pred_trajectory)
              activations['feature_dict']['r2_score'] = test_r2
              
              # Add the odeformer predicted expression from sample
              f = io.StringIO()
              with redirect_stdout(f):
                  dstr.print(n_predictions=1)
              pred_expression = f.getvalue()
              activations['pred_expression'] = pred_expression
              
              # Add ground truth expression from sample (depending on whether manual or random)
              if 'tree' in test_sample:
                activations['expression'] = test_sample['tree']
              elif 'expression' in test_sample:
                activations['expression'] = test_sample['expression']

              # Save activations file to specified path
              with open(activation_filepath, 'wb') as f:
                  pickle.dump(activations, f)
            else:
              # Activation file already exists, skip processing
              print(f"\nSkipping sample: {test_sample_id} (activation file already exists)")

        print(f'\n[INFO] Activation extraction complete. Activations saved to {activations_path}')