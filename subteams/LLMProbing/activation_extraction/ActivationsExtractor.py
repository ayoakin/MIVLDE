import os
import re
import pickle
import torch

class ActivationsExtractor():
    def __init__(self):
        pass
        # TODO: add self.extract_activations with the correct parameters
        # This will make ActivationsExtractor behaviour consistent with ActivationsDataset and SamplesGenerators
        # i.e. won't have to call an extra class method

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

    def extract_activations(self, dstr, samples_path, activations_path, layers_to_extract=['ffn']):
        layer_outputs = {}
        os.makedirs(activations_path, exist_ok=True)
        samples_dir = os.fsencode(samples_path)

        self.register_hooks(dstr.model.encoder, 'encoder', layer_outputs, layers_to_extract)
        self.register_hooks(dstr.model.decoder, 'decoder', layer_outputs, layers_to_extract)
        
        for sample in os.listdir(samples_dir):
            sample_name = os.fsdecode(sample)
            sample_path = os.path.join(samples_path, sample_name)
            with open(sample_path, 'rb') as f:
                test_sample = pickle.load(f)
            test_id = re.findall(r'\d+', sample_name)[0]
            print(f"[INFO] Loaded sample with id {test_id} from {sample_path}")

            with torch.no_grad():
                dstr.fit(test_sample['times'], test_sample['trajectory'])
            
            # Get outputs of specified layer parts
            encoder_layer_outputs = {}
            decoder_layer_outputs = {}
            for layer_name, output in layer_outputs.items():
                if 'ffn' in layer_name: # TODO: may need to change this in future to work with layers_to_extract (if we want it to be more general)
                # We currently look at ouputs of ffn layers since they come before layer norm
                    if 'encoder' in layer_name:
                        encoder_layer_outputs[layer_name] = output
                    if 'decoder' in layer_name:
                        decoder_layer_outputs[layer_name] = output

            # Add relevant info to activations dict
            activations = {}
            activations['encoder'] = encoder_layer_outputs
            activations['decoder'] = decoder_layer_outputs
            activations['operator_dict'] = test_sample['operator_dict']
            activations['feature_dict'] = test_sample['feature_dict']
            # TODO: copute and add R^2 score (this will add extra computational overhead which we will need to test)
            # TODO: copy the odeformer predicted expression from sample
            # TODO: copy ground truth expression from sample
            # if 'tree' in test_sample:
            #   activations['expression'] = test_sample['tree']
            # if 'expression' in test_sample:
            #   activations['expression'] = test_sample['expression']

            # Save activations dict using same id as sample
            activation_filename = f"activation_{test_id}.pt"
            activation_filepath = os.path.join(activations_path, activation_filename)
            with open(activation_filepath, 'wb') as f:
                pickle.dump(activations, f)
            print(f"[INFO] Saved activations with id {test_id} to {activation_filepath}")

        print(f'[INFO] Activation extraction complete. Activations saved to {activations_path}')