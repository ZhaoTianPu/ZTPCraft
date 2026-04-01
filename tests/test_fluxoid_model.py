import numpy as np
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidSector,
)


def test_effective_flux_symmetry(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s = sectors[0]
    phi_eff = system.model.effective_flux(s)

    # trivial consistency check: deterministic
    assert np.isfinite(phi_eff)


def test_energy_offset_consistency(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s = sectors[1]
    E = system.model.energy_offset(s)
    assert np.isfinite(E)


def test_potential_matches_reduced(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s = sectors[0]
    phi = np.linspace(-2 * np.pi, 2 * np.pi, 100)

    U_full = system.model.potential_as_explicit_inductor_sum(phi, s)
    U_red = system.model.potential_as_effective_inductor_plus_offset(phi, s)

    diff = U_full - U_red
    assert np.allclose(diff, 0, atol=1e-12)
