import os
import io
import re
import pickle
import torch
from odeformer.metrics import r2_score
from contextlib import redirect_stdout
from tqdm import tqdm

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
        layer_outputs = {}
        os.makedirs(activations_path, exist_ok=True)
        samples_dir = os.fsencode(samples_path)

        self.register_hooks(dstr.model.encoder, 'encoder', layer_outputs, layers_to_extract)
        self.register_hooks(dstr.model.decoder, 'decoder', layer_outputs, layers_to_extract)
        
        for sample in tqdm(os.listdir(samples_dir), desc='Extracting Activations'):
            # Load sample
            sample_name = os.fsdecode(sample)
            sample_path = os.path.join(samples_path, sample_name)
            with open(sample_path, 'rb') as f:
                test_sample = pickle.load(f)
            # test_id = re.findall(r'\d+', sample_name)[0]
            # print(f"[INFO] Loaded sample with id {test_id} from {sample_path}")

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
            test_r2 = r2_score(test_sample['trajectory'], pred_trajectory)
            activations['r2_score'] = test_r2
            
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

            # Save activations dict using same id as sample - TODO: determine if there is a smarter way of assigning ids to samples
            # Probably it makes sense to just replace 'sample' with 'activation', e.g. 'sample_exp_0' --> 'activation_exp_0'
            # Currently it will overwrite files...
            # activation_filename = f"activation_{test_id}.pt"
            activation_filename = sample_name.split('/')[-1].replace('sample', 'activation')
            activation_filepath = os.path.join(activations_path, activation_filename)
            with open(activation_filepath, 'wb') as f:
                pickle.dump(activations, f)
            # print(f"[INFO] Saved activations with id {test_id} to {activation_filepath}")

        print(f'[INFO] Activation extraction complete. Activations saved to {activations_path}')