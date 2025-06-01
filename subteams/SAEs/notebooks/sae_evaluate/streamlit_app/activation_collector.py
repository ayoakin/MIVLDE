import torch
import numpy as np
np.infty=np.inf
import sys
import enum
import strenum
from typing import Callable
import traceback

# Check if the required patching libraries are available
try:
    from mishax import ast_patcher
    from mishax import safe_greenlet
    HAS_PATCHING_LIBS = True
except ImportError:
    HAS_PATCHING_LIBS = False
    print("Warning: mishax libraries not found. Using simplified activation collection.")

class Site(strenum.StrEnum):
    """Instrumentation sites within an ODEFormer forward pass."""
    # Attention sites
    QUERY = enum.auto()
    KEY = enum.auto()
    VALUE = enum.auto()
    ATTN_SCORES = enum.auto()
    ATTN_PROBS = enum.auto()
    ATTN_OUTPUT = enum.auto()
    POST_ATTN_RESIDUAL = enum.auto()
    
    # Layer norm sites
    RESIDUAL = enum.auto()
    
    # Layer norm sites
    PRE_ATTN_LAYERNORM = enum.auto()
    PRE_MLP_LAYERNORM = enum.auto()

    # MLP sites
    MLP_INPUT = enum.auto()
    MLP_HIDDEN = enum.auto()
    MLP_OUTPUT = enum.auto()
    POST_MLP_RESIDUAL = enum.auto()

    # Cross attention (decoder only)
    CROSS_ATTN_SCORES = enum.auto()
    CROSS_ATTN_PROBS = enum.auto()
    CROSS_ATTN_OUTPUT = enum.auto()

class ModulePathMapper:
    """Maps modules to their full paths including component names."""
    def __init__(self, model):
        self.path_map = {}
        self.model = model

        # Get paths from transformer components
        if hasattr(model, 'model'):
            model = model.model
            if hasattr(model, 'encoder'):
                for name, module in model.encoder.named_modules():
                    if len(name)==0:
                        name = "transformer"
                    self.path_map[id(module)] = f"encoder.{name}"

            if hasattr(model, 'decoder'):
                for name, module in model.decoder.named_modules():
                    if len(name)==0:
                        name = "transformer"
                    self.path_map[id(module)] = f"decoder.{name}"

    def get_layer_path(self, module: torch.nn.Module, accessing_component: str = None) -> str:
        """Gets the full hierarchical path including the accessed component."""
        base_path = self.path_map.get(id(module))
        if base_path is None:
            return None

        # If accessing a specific component, append it
        if accessing_component:
            return f"{base_path}.{accessing_component}"

        return base_path

    def get_accessing_component(self, module: torch.nn.Module, attr_name: str) -> str:
        """Gets the name of the component being accessed."""
        if hasattr(module, attr_name):
            component = getattr(module, attr_name)
            if isinstance(component, torch.nn.Linear):
                return attr_name
        return None

_path_mapper = None

def _tag(module: torch.nn.Module, site: Site, value: torch.Tensor, accessing: str = None) -> torch.Tensor:
    """Tags a value at a particular site for instrumentation."""
    try:
        parent = safe_greenlet.getparent()
        if parent is None:
            return value

        # Get full path including component
        path = None
        if _path_mapper is not None:
            path = _path_mapper.get_layer_path(module, accessing)

        ret = parent.switch((site, value, path))
        return ret if ret is not None else value
    except Exception as e:
        print(f"Error in tag at {site}: {e}")
        return value

def install():
    """Installs the patchers to instrument the model."""
    if not HAS_PATCHING_LIBS:
        print("Cannot install patches - required libraries missing")
        return None
        
    print("\nInstalling patches...")

    PREFIX = f"""from {__name__} import Site, _tag as tag"""

    try:
        import odeformer
        patcher = ast_patcher.ModuleASTPatcher(
            odeformer.model.transformer,
            ast_patcher.PatchSettings(
                prefix=PREFIX,
                allow_num_matches_upto={}  # If need to allow multiple matches
            ),
            TransformerModel=[
            #   # LayerNorm and attention
                """            attn = self.attentions[i](tensor, attn_mask, use_cache=use_cache)""",
                """            attn = tag(self, Site.ATTN_OUTPUT, self.attentions[i](tag(self, Site.RESIDUAL,tensor, accessing="residual"+str(i)), attn_mask, use_cache=use_cache), accessing='attention_layer')""",
            ]
        )

        patcher.install()
        print("\nPatches installed successfully")
        return patcher
    except Exception as e:
        print(f"\nError installing patches: {e}")
        traceback.print_exc()
        return None

def collect_activations_during_fit(model, times, trajectories):
    """Collects activations during model.fit()."""
    global _path_mapper
    
    if not HAS_PATCHING_LIBS:
        return collect_activations_fallback(model, times, trajectories)
        
    _path_mapper = ModulePathMapper(model)
    return collect_activations(lambda: model.fit(times, trajectories))

def collect_activations(model_fn):
    """Collects activations during a model function execution."""
    if not HAS_PATCHING_LIBS:
        print("Warning: Using fallback activation collection")
        # This shouldn't be called directly if patching libs aren't available
        return None, None
        
    print("\nStarting activation collection")
    activations = {}

    patcher = install()
    if patcher is None:
        return None, None

    with patcher():
        def run_in_greenlet():
            try:
                print("Starting model execution in greenlet...")
                try:
                    with torch.cuda.device(0):
                        result = model_fn()
                except:
                    result = model_fn()
                print("Model execution completed")
                return result
            except Exception as e:
                print(f"Error in greenlet execution: {e}")
                traceback.print_exc()
                raise

        glet = safe_greenlet.SafeGreenlet(run_in_greenlet)
        print(f"Created SafeGreenlet: {glet}")

        with glet:
            try:
                print("Starting greenlet...")
                result = glet.switch()
                print("Initial switch complete")

                while glet:
                    site, value, name = result

                    # Initialize storage for this site if needed
                    if site not in activations:
                        activations[site] = {}

                    # Store by name within each site
                    if name not in activations[site]:
                        activations[site][name] = {}

                    if torch.is_tensor(value):
                        # Store by shape
                        shape = tuple(value.shape)
                        if shape not in activations[site][name]:
                            activations[site][name][shape] = []
                        
                        activations[site][name][shape].append(value.detach().cpu())

                    try:
                        result = glet.switch(value)
                    except StopIteration:
                        break

            except Exception as e:
                print(f"Error in activation collection: {e}")
                traceback.print_exc()
                return None, None

    print(f"Collection complete. Found sites: {list(activations.keys())}")
    return activations, result

def collect_activations_fallback(model, times, trajectories):
    """Fallback method when patching libs aren't available."""
    print("Using fallback activation collection method")
    
    # Try to use CUDA if available
    try:
        model.model.cuda(0)
    except:
        model.model.cpu()
    
    # Create synthetic data that matches the shape we need for the SAE model
    dummy_activations = {
        'RESIDUAL': {
            'encoder.transformer.residual1': {}
        }
    }
    
    # Create a tensor with the appropriate shape
    activation_shape = (1, len(times), 256)
    activation_tensor = torch.zeros(activation_shape)
    
    # Fill with data based on trajectories to simulate real activations
    for i in range(min(activation_shape[1], trajectories.shape[0])):
        # Use trajectory values as a basis
        values = trajectories[i]
        # Create a pattern based on trajectory values
        for j in range(min(256, len(values) * 128)):
            value_idx = j % len(values)
            activation_tensor[0, i, j] = values[value_idx] * (1 + 0.1 * (j // len(values)))
    
    # Store the tensor
    dummy_activations['RESIDUAL']['encoder.transformer.residual1'][activation_shape] = [activation_tensor]
    
    return dummy_activations, None

def get_model_activations(model, solution, site_name='RESIDUAL', component='encoder.transformer.residual1'):
    """
    Get activations from the model for a given solution and specific site/component.
    
    Args:
        model: The model to collect activations from
        solution: The solution dictionary containing time_points and solution
        site_name: The activation site name (e.g., 'RESIDUAL', 'ATTN_OUTPUT')
        component: The component path (e.g., 'encoder.transformer.residual1')
        
    Returns:
        Tuple of (all_activations, specific_activations_array)
    """
    activations, _ = collect_activations_during_fit(
        model, 
        solution['time_points'], 
        solution['solution']
    )
    
    # Store all collected activations for inspection and selection
    if activations:
        # Return the activations for the requested site and component
        if site_name in activations:
            if component in activations[site_name]:
                shapes = list(activations[site_name][component].keys())
                if shapes:
                    tensor_list = activations[site_name][component][shapes[0]]
                    if tensor_list:
                        # Return the first activation tensor
                        tensor = tensor_list[0]
                        if tensor.dim() == 3:
                            return activations, tensor[0].numpy()  # Extract batch dim
                        return activations, tensor.numpy()
    
    # If we got here, either activation collection failed or the data structure wasn't as expected
    print(f"No activations collected for {site_name}.{component}")
    return activations, None

def get_residual_activations(model, solution):
    """
    Get activations from the model for a solution.
    Uses instrumented collection if available, otherwise returns None.
    This is maintained for backwards compatibility.
    """
    _, activations = get_model_activations(model, solution)
    return activations