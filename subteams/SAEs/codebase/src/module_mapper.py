import torch.nn as nn

class ModulePathMapper:
    """Maps modules to their hierarchical paths within the model."""
    def __init__(self, model):
        self.path_map = dict()
        self.model = model
        self._build_path_map()


    def _build_path_map(self):
        """Constructs the module-to-path mapping."""
        model = getattr(self.model, 'model', self.model)

        def _name(section):
            if not (module := getattr(model, section, None)):
                return

            for name, sub_module in module.named_modules():
                self.path_map[id(sub_module)] = f"{section}.{name if name else 'outer'}"
        
        _name('encoder')
        _name('decoder')

    def get_layer_path(self, module: nn.Module, accessing_component: str = None) -> str:
        """Returns the full hierarchical path including accessed component if provided."""
        base_path = self.path_map.get(id(module))
        return f"{base_path}.{accessing_component}" if base_path and accessing_component else base_path
