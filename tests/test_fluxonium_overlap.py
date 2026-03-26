import numpy as np
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidSector,
)


def test_overlap_identity(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s = sectors[0]

    O = system.overlap_matrix(s, s)

    # should be identity
    assert np.allclose(O, np.eye(O.shape[0]), atol=1e-6)


def test_overlap_hermitian(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s1, s2, _ = sectors

    O12 = system.overlap_matrix(s1, s2)
    O21 = system.overlap_matrix(s2, s1)

    assert np.allclose(O12, O21.conj().T, atol=1e-6)


def test_overlap_norm(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s1, s2, _ = sectors

    O = system.overlap_matrix(s1, s2)

    # columns should have norm ≤ 1
    norms = np.linalg.norm(O, axis=0)
    assert np.all(norms <= 1.0 + 1e-6)
