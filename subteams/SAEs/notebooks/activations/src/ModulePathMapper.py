from dataclasses import dataclass, field
import torch.nn as nn

@dataclass
class ModulePathMapper:
    """
    Maps neural network modules to their hierarchical paths within the model.
    
    This class maintains a mapping of module instances to their corresponding
    names within the model architecture, allowing for structured access.
    """
    model: object  # The model whose module paths need to be mapped
    path_map: dict = field(default_factory=dict)  # Stores module paths by their ID

    def __post_init__(self):
        """
        Constructs the module-to-path mapping.
        Iterates through the model's encoder and decoder sections to
        build hierarchical paths for named modules.
        """
        model = getattr(self.model, 'model', self.model)

        def _name(section: str):
            """Registers modules under the given section (encoder/decoder)."""
            if not (module := getattr(model, section, None)):
                return

            for name, sub_module in module.named_modules():
                self.path_map[id(sub_module)] = f"{section}.{name if name else 'outer'}"
        
        _name('encoder')
        _name('decoder')

    def get_layer_path(self, module: nn.Module, accessing_component: str = None) -> str:
        """
        Retrieves the full hierarchical path of a given module.
        
        Args:
            module (nn.Module): The module whose path is being queried.
            accessing_component (str, optional): Additional component name
                (e.g., an attribute) to append to the path.
                Defaults to None.
        
        Returns:
            str: The full path of the module, including any accessed component.
        """
        base_path = self.path_map.get(id(module))
        return f"{base_path}.{accessing_component}" if base_path and accessing_component else base_path
