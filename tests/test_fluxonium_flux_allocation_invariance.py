import numpy as np
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_system import (
    TwoLoopFluxoidSystem,
)
from ztpcraft.projects.fluxonium.multiloop_fluxonium.two_loop_fluxoid_model import (
    FluxoidSector,
    FluxoidModelParams,
)
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def test_flux_allocation_invariance(params: FluxoidModelParams):
    s = FluxoidSector(0, 0)

    params1: FluxoidModelParams = params
    params2: FluxoidModelParams = params.__class__(
        **{**params.__dict__, "flux_allocation_alpha": 0.0}
    )
    params3: FluxoidModelParams = params.__class__(
        **{**params.__dict__, "flux_allocation_alpha": 1.0}
    )

    sys1: TwoLoopFluxoidSystem = TwoLoopFluxoidSystem(params1)
    sys2: TwoLoopFluxoidSystem = TwoLoopFluxoidSystem(params2)
    sys3: TwoLoopFluxoidSystem = TwoLoopFluxoidSystem(params3)

    E1: FloatArray = sys1.eigenvalues_with_offset(s)
    E2: FloatArray = sys2.eigenvalues_with_offset(s)
    E3: FloatArray = sys3.eigenvalues_with_offset(s)

    assert np.allclose(E1, E2, atol=1e-8)
    assert np.allclose(E1, E3, atol=1e-8)
