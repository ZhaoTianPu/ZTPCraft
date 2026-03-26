import numpy as np
from numpy.typing import NDArray
import pytest
import ztpcraft as ztpc
from typing import Any

from ztpcraft.projects.fluxonium.multiloop_fluxonium import (
    TwoLoopArrayFluxonium,
)

PhiArray = NDArray[np.float64]


@pytest.fixture
def model() -> TwoLoopArrayFluxonium:
    param_set: dict[str, object] = {}
    param_set["qC_standalone_symmetric"] = {
        "EJa": 58.5,
        "CJa": ztpc.tb.units.EC_2_capacitance(0.685),
        "Cga": 0.25,  # fF
        "EJb": 6.16,
        "CJb": ztpc.tb.units.EC_2_capacitance(0.825),
        "N": 117,
    }

    return TwoLoopArrayFluxonium(param_set["qC_standalone_symmetric"])


@pytest.fixture
def built_system(
    model: TwoLoopArrayFluxonium,
) -> tuple[TwoLoopArrayFluxonium, Any]:
    system = model.build_system(
        phi_ext_A=0.0,
        phi_ext_B=1.0,
        sectors=[(0.0, 0.0), (0.0, 1.0), (0.0, -1.0)],
    )
    return model, system


def _minima_points(system: Any) -> dict[str, PhiArray]:
    """
    Map each stored minimum name -> phi_eq (oscillator center).
    """
    # OscillatorRegistry stores LocalHarmonicOscillator objects in `oscillators`.
    return {
        name: np.asarray(osc.center, dtype=np.float64)  # type: ignore[attr-defined]
        for name, osc in system.registry.oscillators.items()
    }


def test_minima_are_stationary_points(
    built_system: tuple[TwoLoopArrayFluxonium, Any],
) -> None:
    model, system = built_system
    minima = _minima_points(system)

    for name, phi_eq in minima.items():
        _, grad, _ = model.value_grad_hessian(
            phi_eq,
            phi_ext_A=0.0,
            phi_ext_B=1.0,
        )

        grad_norm = float(np.linalg.norm(np.asarray(grad, dtype=np.float64)))
        assert grad_norm < 1e-4, f"{name}: gradient not zero, got {grad_norm}"


def test_minima_have_positive_hessian(
    built_system: tuple[TwoLoopArrayFluxonium, Any],
) -> None:
    model, system = built_system
    minima = _minima_points(system)

    for name, phi_eq in minima.items():
        _, _, hess = model.value_grad_hessian(
            phi_eq,
            phi_ext_A=0.0,
            phi_ext_B=1.0,
        )

        hess = np.asarray(hess, dtype=np.float64)
        eigvals = np.linalg.eigvalsh(hess)
        min_eig = float(np.min(eigvals))

        assert min_eig > 0, f"{name}: not a minimum, min eigenvalue = {min_eig}"


def test_minima_are_local_energy_minima(
    built_system: tuple[TwoLoopArrayFluxonium, Any],
) -> None:
    model, system = built_system
    minima = _minima_points(system)

    rng = np.random.default_rng(seed=0)

    for name, phi_eq in minima.items():
        E0 = float(model.potential(phi_eq, phi_ext_A=0.0, phi_ext_B=1.0))

        # Small random perturbations should increase energy locally.
        for _ in range(5):
            delta = 1e-3 * rng.normal(size=phi_eq.shape)
            E1 = float(model.potential(phi_eq + delta, phi_ext_A=0.0, phi_ext_B=1.0))

            assert E1 > E0, f"{name}: perturbation lowered energy"


def test_minima_are_consistent_with_direct_optimization(
    built_system: tuple[TwoLoopArrayFluxonium, Any],
) -> None:
    """
    Compare with a direct local minimization starting from phi_eq.
    """
    import scipy.optimize as opt

    model, system = built_system
    minima = _minima_points(system)

    for name, phi_eq in minima.items():

        def energy_fn(x: PhiArray) -> float:
            return float(model.potential(x, phi_ext_A=0.0, phi_ext_B=1.0))

        result = opt.minimize(
            energy_fn,
            phi_eq,
            method="BFGS",
            options={"gtol": 1e-6, "maxiter": 200},
        )

        phi_opt = result.x
        dist = float(np.linalg.norm(phi_opt - phi_eq))

        assert dist < 1e-5, f"{name}: direct minimization moved point by {dist}"
