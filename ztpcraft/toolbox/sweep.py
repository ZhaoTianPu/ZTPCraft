from __future__ import annotations

"""Generic parameter-sweep engine with optional NamedSlotsNdarray outputs."""

from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import importlib
import inspect
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray

_IndexTuple = tuple[int, ...]
_ParamsDict = dict[str, Any]


@dataclass(frozen=True)
class SweepSpec:
    """Specification for one derived quantity in a parameter sweep."""

    fn: Callable[..., Any]
    kwargs: dict[str, Any]


class GenericSweep:
    """Domain-agnostic sweep engine inspired by scqubits ParameterSweep.

    Notes
    -----
    - Leading sweep axes are defined by ``paramvals_by_name`` in insertion order.
    - Each ``add_sweep`` function is evaluated pointwise with signature
      ``fn(sweep, index, params, **kwargs)``.
    - Results are stored as regular ndarrays and can be wrapped into
      ``NamedSlotsNdarray`` via :meth:`as_namedslots`.
    """

    def __init__(
        self,
        paramvals_by_name: Mapping[str, Iterable[Any] | NDArray[Any]],
        *,
        autorun: bool = False,
    ):
        ordered = OrderedDict(
            (name, np.asarray(list(values))) for name, values in paramvals_by_name.items()
        )
        if len(ordered) == 0:
            raise ValueError("paramvals_by_name must contain at least one axis.")

        self.paramvals_by_name: OrderedDict[str, NDArray[Any]] = ordered
        self.param_names: list[str] = list(self.paramvals_by_name.keys())
        self.shape: tuple[int, ...] = tuple(
            len(v) for v in self.paramvals_by_name.values()
        )

        self._specs: OrderedDict[str, SweepSpec] = OrderedDict()
        self._data: dict[str, NDArray[Any]] = {}
        self._point_cache: NDArray[Any] = np.empty(self.shape, dtype=object)
        self._point_cache[:] = None
        self._computed_sweeps: set[str] = set()
        self._has_run = False
        self._active_index: _IndexTuple | None = None
        self._active_params: _ParamsDict | None = None

        if autorun:
            self.run()

    def add_sweep(
        self,
        sweep_function: Callable[..., Any],
        sweep_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Add one sweep quantity and compute it immediately.

        Signature intentionally mirrors scqubits ParameterSweep.add_sweep:
        ``add_sweep(sweep_function, sweep_name=None, **kwargs)``.
        """
        if sweep_name is None:
            if not hasattr(sweep_function, "__name__"):
                raise ValueError(
                    "Sweep function name cannot be inferred; provide sweep_name."
                )
            sweep_name = sweep_function.__name__
        self._specs[sweep_name] = SweepSpec(
            fn=sweep_function,
            kwargs=kwargs,
        )
        self._computed_sweeps.discard(sweep_name)
        self.run(sweep_names=[sweep_name], recompute=True)

    def _call_sweep_function(
        self,
        sweep_function: Callable[..., Any],
        index: _IndexTuple,
        params: _ParamsDict,
        sweep_kwargs: dict[str, Any],
    ) -> Any:
        """Call sweep function with flexible signatures.

        Supported styles:
        - ``fn(sweep)``
        - ``fn(sweep, index, params)``
        - keyword style: ``fn(sweep=..., index=..., params=...)``
        - any of the above with ``**kwargs`` passed from ``add_sweep``.
        """
        try:
            signature = inspect.signature(sweep_function)
        except (TypeError, ValueError):
            return sweep_function(self, index, params, **sweep_kwargs)

        params_info = signature.parameters
        supports_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params_info.values()
        )

        call_kwargs: dict[str, Any] = {}
        if ("sweep" in params_info) or supports_var_kw:
            call_kwargs["sweep"] = self
        if ("index" in params_info) or supports_var_kw:
            call_kwargs["index"] = index
        if ("params" in params_info) or supports_var_kw:
            call_kwargs["params"] = params
        for key, value in sweep_kwargs.items():
            if (key in params_info) or supports_var_kw:
                call_kwargs[key] = value

        positional_only_count = sum(
            p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in params_info.values()
        )
        if positional_only_count >= 3:
            return sweep_function(self, index, params, **sweep_kwargs)
        if positional_only_count == 2:
            return sweep_function(self, index, **sweep_kwargs)
        if positional_only_count == 1:
            return sweep_function(self, **sweep_kwargs)
        return sweep_function(**call_kwargs)

    def run(
        self,
        sweep_names: Iterable[str] | None = None,
        *,
        recompute: bool = False,
    ) -> None:
        """Evaluate selected sweeps on the full parameter grid.

        Parameters
        ----------
        sweep_names:
            Names of sweeps to run. If omitted, runs all pending sweeps; with
            ``recompute=True``, reruns all registered sweeps.
        recompute:
            If True, rerun selected sweeps even if they were computed before.
        """
        if sweep_names is None:
            if recompute:
                target_names = list(self._specs.keys())
            else:
                target_names = [
                    name for name in self._specs.keys() if name not in self._computed_sweeps
                ]
        else:
            target_names = list(sweep_names)

        for name in target_names:
            if name not in self._specs:
                raise KeyError(f"Unknown sweep name: {name}")
            if (name not in self._data) or recompute:
                self._data[name] = np.empty(self.shape, dtype=object)

        if len(target_names) == 0:
            self._has_run = len(self._computed_sweeps) > 0
            return

        ranges = [range(n) for n in self.shape]
        for index in product(*ranges):
            params = {
                param_name: self.paramvals_by_name[param_name][axis_index]
                for axis_index, param_name in enumerate(self.param_names)
            }
            self._active_index = index
            self._active_params = params
            for name in target_names:
                spec = self._specs[name]
                value = self._call_sweep_function(spec.fn, index, params, spec.kwargs)
                self._data[name][index] = value
        self._active_index = None
        self._active_params = None
        self._computed_sweeps.update(target_names)
        self._has_run = True

    def keys(self) -> list[str]:
        """Return names of registered sweep quantities."""
        return list(self._specs.keys())

    def __getitem__(self, key: str) -> NDArray[Any]:
        if key not in self._computed_sweeps:
            raise RuntimeError(
                f"Sweep result '{key}' is not available; run this sweep first."
            )
        return self._data[key]

    def as_namedslots(self, key: str):
        """Return a NamedSlotsNdarray view of one computed sweep quantity."""
        try:
            module = importlib.import_module("scqubits.core.namedslots_array")
            namedslots_cls = getattr(module, "NamedSlotsNdarray")
        except Exception as exc:
            raise ImportError(
                "NamedSlotsNdarray is unavailable. Install/import scqubits first."
            ) from exc
        return namedslots_cls(self[key], self.paramvals_by_name)

    @property
    def current_index(self) -> _IndexTuple:
        """Current point index while inside a sweep callback."""
        if self._active_index is None:
            raise RuntimeError("No active sweep point. Access only inside sweep function.")
        return self._active_index

    @property
    def current_params(self) -> _ParamsDict:
        """Current point parameters while inside a sweep callback."""
        if self._active_params is None:
            raise RuntimeError("No active sweep point. Access only inside sweep function.")
        return self._active_params

    def get_value(self, sweep_name: str, index: _IndexTuple | None = None) -> Any:
        """Get value from another computed sweep at a given point.

        If ``index`` is omitted, uses the current point during callback execution.
        """
        if sweep_name not in self._computed_sweeps:
            raise KeyError(f"Sweep '{sweep_name}' is not computed yet.")
        if index is None:
            index = self.current_index
        return self._data[sweep_name][index]

    def set_point_cache(self, index: _IndexTuple, value: Any) -> None:
        """Set cached object associated with one sweep point."""
        self._point_cache[index] = value

    def get_point_cache(self, index: _IndexTuple, default: Any = None) -> Any:
        """Get cached object associated with one sweep point."""
        value = self._point_cache[index]
        if value is None:
            return default
        return value

    def get_or_build_point_cache(
        self,
        index: _IndexTuple,
        builder: Callable[[], Any],
    ) -> Any:
        """Get point cache or build/store it lazily."""
        value = self.get_point_cache(index, default=None)
        if value is None:
            value = builder()
            self.set_point_cache(index, value)
        return value

