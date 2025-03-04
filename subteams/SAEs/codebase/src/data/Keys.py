from dataclasses import dataclass, field
from typing import Dict, Set
import re
import torch

@dataclass
class Keys:
    """
    Stores configuration settings for filtering activations.

    Attributes:
        encoders (Set[int]): Set of encoder layer indices to keep.
        decoders (Set[int]): Set of decoder layer indices to keep.
        encoder_attn (Set[str]): Set of cross-attention-related activations to collect. 
                                 Can contain any of: ['attn_scores', 'attn_probs', 'attn_output'].
        cross_attention (bool): Whether to keep cross-attention activations.

        to_collect (Dict[str, bool]): Dictionary mapping activation types to boolean values
                                      indicating whether to collect them.
    """
    encoders: Set[int] = field(default_factory=set)
    decoders: Set[int] = field(default_factory=set)
    encoder_attn: Set[str] = field(default_factory=set)
    cross_attention: bool = False

    to_collect: Dict[str, bool] = field(default_factory=lambda: {
        'residual_stream': False,
        'attn_scores': False,
        'attn_probs': False,
        'attn_output': False,
        'attn_mlp_output': False,
        'mlp_input': False,
        'mlp_output': False
    })

    def __post_init__(self):
        """
        Ensures that `to_collect` always contains all possible keys with default values.
        This prevents missing keys from causing errors when accessing `to_collect`.
        """
        default_flags = {
            'residual_stream': False,
            'attn_scores': False,
            'attn_probs': False,
            'attn_output': False,
            'attn_mlp_output': False,
            'mlp_input': False,
            'mlp_output': False
        }

        # Merge user-defined `to_collect` values with default values
        self.to_collect = {**default_flags, **self.to_collect}

    def _encoders_attn_to_remove(self) -> Set[str]:
        """
        Determines which attention types should be removed from activations.

        Returns:
            Set[str]: A set of attention keys to remove.
        """
        return {'attn_scores', 'attn_probs', 'attn_output'} - self.encoder_attn


    def filter(self, activations: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Filters activations based on the `to_collect` settings and encoder/decoder indices.

        Args:
            activations (Dict[str, Dict[str, torch.Tensor]]): 
                Dictionary containing activation tensors, structured as:
                {
                    "attn_scores": { "encoder.attentions.0.scores": tensor, ... },
                    "attn_probs": { "encoder.attentions.1.weights": tensor, ... },
                    ...
                }

        Returns:
            Dict[str, Dict[str, torch.Tensor]]: The filtered activations dictionary.
        """

        # Keep only activation keys that are enabled in `to_collect`
        activations = {key: sub_dict for key, sub_dict in activations.items() if self.to_collect.get(key, False)}

        # Remove encoder attention entries that are NOT specified in `encoder_attn`
        if (to_remove := self._encoders_attn_to_remove()):
            for key in filter(lambda k: k in activations, to_remove):
                activations[key] = {
                    k: v for k, v in activations[key].items() 
                    if 'encoder_attn' not in k
                }

        # If `cross_attention` is disabled, remove all `cross_attention` activations from 'attn_mlp_output'
        if not self.cross_attention and (mlp_output := activations.get('attn_mlp_output', None)):
            activations['attn_mlp_output'] = {
                k: v for k, v in mlp_output.items() 
                if 'cross_attention' not in k
            }

        def check(x: str) -> bool:
            """
            Determines whether a given activation key corresponds to a valid encoder or decoder layer.

            Args:
                x (str): The activation key, e.g., 'encoder.attentions.3.scores'.

            Returns:
                bool: True if the activation should be kept, False otherwise.
            """
            if not (match := re.search(r"\d+", x)):  # Extracts the first number in the key
                return False
            
            layer = int(match.group())
            indices = self.encoders if x.startswith("encoder") else self.decoders

            return layer in indices  # Keep only layers that exist in `self.encoders` or `self.decoders`

        # Apply layer filtering to each activation type
        return {
            key: {k: v for k, v in values.items() if check(k)}
            for key, values in activations.items()
        }
