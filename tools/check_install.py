"""Run with python -I against an installed wheel, outside the source tree."""
from pathlib import Path
import sys

import ztpcraft
from ztpcraft.projects.fluxonium.multiloop_fluxonium import SectorBasis, FluxoidSector
from ztpcraft.bosonic.oscillator_integrals import _oscillator_integrals_1d_quadrature as integrals

source_root = Path(__file__).resolve().parents[1]
assert not Path(ztpcraft.__file__).resolve().is_relative_to(source_root)
assert "jax" not in sys.modules
assert "ninatool" not in sys.modules
assert integrals.hermite_complex(0, 1j) == 1
assert abs(integrals.cSij(0, 0, 0.0, 0.0, 1.0, 1.0) - 1.0) < 1e-12
basis = SectorBasis((FluxoidSector(0, 0), FluxoidSector(1, 0)), 2)
assert basis.size == 4
print(f"Installed package verified: {ztpcraft.__file__}")
