import pytest

from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidModelParams,
    FluxoidSector,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)


@pytest.fixture
def params():
    return FluxoidModelParams(
        EL_a=1.0,
        EL_b=1.2,
        EJ=5.0,
        EC=0.5,
        phi_ext_a=0.2,
        phi_ext_b=-0.1,
        flux_allocation_alpha=0.3,
    )


@pytest.fixture
def system(params: FluxoidModelParams) -> TwoLoopFluxoidSystem:
    return TwoLoopFluxoidSystem(params, cutoff=60, evals_count=8)


@pytest.fixture
def sectors() -> tuple[FluxoidSector, FluxoidSector, FluxoidSector]:
    return (
        FluxoidSector(0, 0),
        FluxoidSector(1, 0),
        FluxoidSector(0, 1),
    )
