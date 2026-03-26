import numpy as np
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidSector,
)


def test_S_cache_reuse(
    system: TwoLoopFluxoidSystem,
    sectors: tuple[FluxoidSector, FluxoidSector, FluxoidSector],
):
    s1: FluxoidSector = sectors[0]
    s2: FluxoidSector = sectors[1]

    # trigger cache
    O1 = system.overlap_matrix(s1, s2)

    cache_size_before = len(system._S_cache)  # type: ignore

    # second call
    O2 = system.overlap_matrix(s1, s2)

    cache_size_after = len(system._S_cache)  # type: ignore

    assert cache_size_before == cache_size_after
    assert np.allclose(O1, O2)
