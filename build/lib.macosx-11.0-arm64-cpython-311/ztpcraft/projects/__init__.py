"""ztpcraft.projects package

Automatically imports immediate subpackages so they are accessible as
`ztpcraft.projects.<subpkg>` and re-exports them in `__all__`."""

from importlib import import_module
from pkgutil import iter_modules
from types import ModuleType
from typing import List

__all__: List[str] = []

for module_info in iter_modules(__path__):
    if module_info.ispkg:
        mod: ModuleType = import_module(f"{__name__}.{module_info.name}")
        globals()[module_info.name] = mod
        __all__.append(module_info.name)

# Cleanup internal names
del import_module, iter_modules, ModuleType, List
