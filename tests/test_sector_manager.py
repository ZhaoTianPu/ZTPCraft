from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidSector,
)


def test_sector_state_shapes(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s: FluxoidSector = sectors[0]
    state = system._manager.get_sector_state(s) # type: ignore

    assert state.evecs.shape[0] == state.cutoff
    assert state.evals.ndim == 1


def test_sector_caching(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s: FluxoidSector = sectors[0]

    state1 = system._manager.get_sector_state(s) # type: ignore
    state2 = system._manager.get_sector_state(s) # type: ignore

    assert state1 is state2
