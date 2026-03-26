import numpy as np
import pytest

from ztpcraft.bosonic.oscillator_integrals.oscillators import LocalHarmonicOscillator
from ztpcraft.bosonic.oscillator_integrals.normal_modes import (
    diagonalize_quadratic_hamiltonian,
)
from ztpcraft.bosonic.oscillator_integrals.oscillator_overlap_cache import (
    OverlapManager,
)
from ztpcraft.bosonic.oscillator_integrals.oscillator_operators import (
    n_operator_element,
)


def make_engine_oscillator(EC: float, EL: float) -> LocalHarmonicOscillator:
    """
    Build your engine oscillator matching scqubits results.
    """
    nm = diagonalize_quadratic_hamiltonian(np.array([[EC]]), np.array([[EL]]))
    return LocalHarmonicOscillator(
        transform_xi_theta=nm.transform_xi_theta,
        transform_qn=nm.transform_qn,
        frequencies=nm.frequencies,
        center=np.array([0.0]),
    )


def test_charge_operator_against_scqubits() -> None:
    scq = pytest.importorskip("scqubits")

    EC = 2.0
    EL = 1.3

    cutoff = 20
    truncated_dim = 10

    # --- scqubits system ---
    fluxonium = scq.Fluxonium(
        EJ=0.0,
        EC=EC,
        EL=EL,
        flux=0.0,
        cutoff=cutoff,
        truncated_dim=truncated_dim,
    )

    n_op = np.asarray(fluxonium.n_operator(), dtype=np.complex128)

    # --- your engine ---
    osc = make_engine_oscillator(EC, EL)
    manager = OverlapManager()

    # --- compare matrix elements ---
    for n in range(6):
        for m in range(6):

            val_engine = n_operator_element(
                osc,
                osc,
                (n,),
                (m,),
                charge_index=0,
                overlap_manager=manager,
            )

            val_scq: complex = complex(n_op[n, m])

            assert np.allclose(
                val_engine, val_scq, atol=1e-10
            ), f"Mismatch at ({n},{m}): {val_engine} vs {val_scq}"
