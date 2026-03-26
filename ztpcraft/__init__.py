from ztpcraft.version import version as __version__

# import submodules
# when changed, remember to change setup.py
import ztpcraft.projects as prj
import ztpcraft.toolbox as tb
import ztpcraft.bosonic as bosonic
import ztpcraft.decoherence as decoherence
import ztpcraft.utils as utils
import ztpcraft.misc as misc

# public modules
__all__ = [
    "prj",
    "tb",
    "bosonic",
    "decoherence",
    "utils",
    "misc",
    "__version__",
]


def reload_all():
    """Dynamically reload all ztpcraft modules."""
    import sys
    import importlib

    # Get all loaded ztpcraft modules
    modules = [m for m in sys.modules if m.startswith("ztpcraft.")]
    # Sort by dependency (deeper modules first)
    modules.sort(key=lambda m: m.count("."), reverse=True)
    # Reload each module
    for module_name in modules:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    print(f"Reloaded {len(modules)} modules!")
