from __future__ import annotations

import numpy as np
import pytest

from ztpcraft.toolbox.sweep import GenericSweep


def test_generic_sweep_scalar_and_matrix_outputs() -> None:
    sweep = GenericSweep(
        {
            "phi_a": np.array([0.0, 1.0]),
            "phi_b": np.array([10.0, 20.0, 30.0]),
        }
    )

    def scalar_fn(_sweep: GenericSweep, index, params) -> float:
        assert index == (index[0], index[1])
        return float(params["phi_a"] + params["phi_b"])

    def matrix_fn(_sweep: GenericSweep, _index, params) -> np.ndarray:
        x = float(params["phi_a"])
        y = float(params["phi_b"])
        return np.array([[x, y], [x + y, x - y]], dtype=np.float64)

    sweep.add_sweep(scalar_fn, "sum")
    sweep.add_sweep(matrix_fn, "mat")

    out = sweep["sum"]
    assert out.shape == (2, 3)
    assert np.isclose(out[1, 2], 31.0)

    mat = sweep["mat"][1, 2]
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (2, 2)
    assert np.isclose(mat[1, 0], 31.0)


def test_generic_sweep_point_cache_and_namedslots_bridge() -> None:
    sweep = GenericSweep({"x": np.array([0, 1, 2])})

    def cache_fn(sw: GenericSweep, index, params) -> int:
        val = int(params["x"] ** 2)
        sw.set_point_cache(index, {"x2": val})
        return val

    sweep.add_sweep(cache_fn, "x2")

    assert sweep.get_point_cache((2,)) == {"x2": 4}

    try:
        arr = sweep.as_namedslots("x2")
        assert np.asarray(arr).shape == (3,)
        assert int(arr[2]) == 4
    except ImportError:
        pytest.skip("scqubits not available for NamedSlotsNdarray bridge test")


def test_add_sweep_run_immediately_is_incremental() -> None:
    sweep = GenericSweep({"x": np.array([0, 1, 2, 3])})
    calls = {"a": 0, "b": 0}

    def sweep_a(_sw: GenericSweep, _index, params) -> int:
        calls["a"] += 1
        return int(params["x"] + 1)

    def sweep_b(_sw: GenericSweep, _index, params) -> int:
        calls["b"] += 1
        return int(params["x"] * 2)

    sweep.add_sweep(sweep_a, "a")
    assert calls["a"] == 4
    assert calls["b"] == 0

    sweep.add_sweep(sweep_b, "b")
    # adding b should not rerun a
    assert calls["a"] == 4
    assert calls["b"] == 4
    assert np.array_equal(sweep["a"], np.array([1, 2, 3, 4], dtype=np.int64))
    assert np.array_equal(sweep["b"], np.array([0, 2, 4, 6], dtype=np.int64))


def test_sweep_object_style_access_and_dependency() -> None:
    sweep = GenericSweep({"x": np.array([1, 2, 3])})

    def base_quantity(sw: GenericSweep) -> int:
        p = sw.current_params
        idx = sw.current_index
        return int(p["x"] + idx[0])

    def dependent_quantity(sw: GenericSweep) -> int:
        return int(sw.get_value("base") * 10)

    sweep.add_sweep(base_quantity, "base")
    sweep.add_sweep(dependent_quantity, "dep")

    assert np.array_equal(sweep["base"], np.array([1, 3, 5], dtype=object))
    assert np.array_equal(sweep["dep"], np.array([10, 30, 50], dtype=object))

