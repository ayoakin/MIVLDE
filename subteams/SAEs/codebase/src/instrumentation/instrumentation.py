import enum
import threading
from typing import Optional

import torch
from torch import nn

from mishax import safe_greenlet
from mishax import ast_patcher

import odeformer

import src

class Site(enum.StrEnum):
    """Instrumentation sites within an ODEFormer forward pass."""
    ATTN_SCORES, ATTN_PROBS, ATTN_OUTPUT, ATTN_MLP_OUTPUT, RESIDUAL_STREAM = (
        enum.auto(), enum.auto(), enum.auto(), enum.auto(), enum.auto()
    )
    MLP_INPUT, MLP_OUTPUT = enum.auto(), enum.auto()

def _get_module_path(
    module: nn.Module, 
    accessing: Optional[str] = None
) -> str:
    assert src.path_mapper, 'Path Mapper not set!'
    _path_mapper = src.path_mapper.get(threading.get_ident(), None)
    return _path_mapper.get_layer_path(module, accessing) if _path_mapper else None

def _tag(
    module: nn.Module, 
    site: Site, 
    value: torch.Tensor, 
    accessing: Optional[str] = None
) -> torch.Tensor:
    """
    Tags a value at a particular site for instrumentation.
    
    This function is used for tracking values at specific locations in a model.
    It attempts to switch to a parent greenlet and pass activation data.

    Args:
        module (nn.Module): The neural network module where the tagging occurs.
        site (Site): The site in the computation graph being tagged.
        value (torch.Tensor): The tensor value to be tagged.
        accessing (str, optional): The specific attribute being accessed. Defaults to None.

    Returns:
        torch.Tensor: The original or modified tensor value.
    """
    try:
        parent = safe_greenlet.getparent()
        if parent is None:
            return value

        path = _get_module_path(module, accessing)
        ret = parent.switch((site, value, path))

        return ret if ret is not None else value
    except Exception as e:
        print(f"Error in tag at {site}: {e}")
        return value
    
def install_collection():
    """Installs patches for instrumentation."""
    print("Installing patches...", end=' ', flush=True)
    print(__name__)
    exit(0)
    PREFIX = f"from {__name__} import Site, _tag as tag"
    
    src.patcher = ast_patcher.ModuleASTPatcher(
        odeformer.model.transformer,
        ast_patcher.PatchSettings(prefix=PREFIX),
        MultiHeadAttention=[
            "scores = torch.matmul(q, k.transpose(2, 3))",
            "scores = tag(self, Site.ATTN_SCORES, torch.matmul(q, k.transpose(2, 3)), accessing='scores')",
            
            "weights = F.softmax(scores.float(), dim=-1).type_as(scores)",
            "weights = tag(self, Site.ATTN_PROBS, F.softmax(scores.float(), dim=-1).type_as(scores), accessing='weights')",

            "context = torch.matmul(weights, v)",
            "context = tag(self, Site.ATTN_OUTPUT, torch.matmul(weights, v), accessing='context')",
        ],
        TransformerModel=[        
            """
            attn = self.encoder_attn[i](
                    tensor, src_mask, kv=src_enc, use_cache=use_cache
                )
            """,
            """
            attn = tag(
                        self, Site.ATTN_MLP_OUTPUT, 
                        self.encoder_attn[i](
                            tensor, src_mask, kv=src_enc, use_cache=use_cache
                        ),
                        accessing=f'cross_attention{i}'
                    )
            """,
            
            "attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)",
            "attn = tag(self, Site.ATTN_MLP_OUTPUT, self.attentions[i](tag(self, Site.RESIDUAL_STREAM, tensor, accessing=f'residual{i}'), attn_mask, use_cache=use_cache), accessing=f'attention_layer{i}')",
        ],
        TransformerFFN=[
            "x = self.lin1(input)",
            "x = self.lin1(tag(self, Site.MLP_INPUT, input, accessing='input'))",
            
            "x = self.lin2(x)",
            "x = tag(self, Site.MLP_OUTPUT, self.lin2(x), accessing='output')",
        ]
    )

    try:
        src.patcher.install()
        print("Patches installed successfully")
    except Exception as e:
        print(f"Error installing patches: {e}")
        import traceback
        traceback.print_exc()


def _should_inject(
    module: nn.Module, 
    accessing: Optional[str] = None
) -> str:
    assert src.path_mapper, 'Path Mapper not set!'
    _path_mapper = src.path_mapper.get(threading.get_ident(), None)
    return _path_mapper.should_inject(module, accessing) if _path_mapper else False

def _conditional_sae(
    input: torch.Tensor,
    module: nn.Module,
    accessing: Optional[str] = None
):
    if not src.sae:
        raise ValueError("SAE not set!")    
    
    sae_instance = src.sae.get(threading.get_ident())
    if not sae_instance:
        raise ValueError("SAE not set for the current thread!")

    if not _should_inject(module, accessing):
        return input
    
    return sae_instance(input)

def install_sae():
    """Installs patches for instrumentation."""
    print("Installing patches...", end=' ', flush=True)
    
    PREFIX = f"from {__name__} import Site, _tag as tag, _conditional_sae as conditional_sae"
    
    src.patcher = ast_patcher.ModuleASTPatcher(
        odeformer.model.transformer,
        ast_patcher.PatchSettings(prefix=PREFIX),
        TransformerModel=[
            "attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)",
            "attn = self.attentions[i](conditional_sae(tensor, self, accessing=f'residual{i}'), attn_mask, use_cache=use_cache)",
        ],
        TransformerFFN=[
            "x = self.lin2(x)",
            "x = self.lin2(conditional_sae(x, self))"
        ]
    )

    try:
        src.patcher.install()
        print("Patches installed successfully")
    except Exception as e:
        print(f"Error installing patches: {e}")
        import traceback
        traceback.print_exc()



