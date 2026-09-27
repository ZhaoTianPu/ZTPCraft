"""Fluxonium models; the specialized array_mode module requires ninatool."""

from importlib import import_module
from . import multiloop_fluxonium

__all__ = ["array_mode", "multiloop_fluxonium"]


def __getattr__(name):
    if name == "array_mode":
        module = import_module(f"{__name__}.array_mode")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
