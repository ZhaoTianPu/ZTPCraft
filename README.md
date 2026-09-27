# ZTPCraft

Tianpu Zhao's physics and quantum research toolbox, including fluxonium models,
fluxoid jump rates, oscillator integrals, and numerical utilities.

## Installation

After the first PyPI release, collaborators can install with:

```sh
python -m pip install ztpcraft
# To run the accompanying Jupyter notebooks:
python -m pip install 'ztpcraft[notebook]'
```

For development, clone this repository and run from its root:

```sh
python -m pip install -e '.[dev]'
```

Python 3.10–3.12 is required. The release workflow builds and tests wheels for
CPython 3.10–3.12 on Linux x86_64 and macOS Apple Silicon/Intel. SciPy is constrained to
1.13.x at build time and runtime because its Cython API changed in later versions.
Support for newer Python/SciPy versions requires updating and testing the extension.
Source builds require a C compiler; pip installs the declared Cython, NumPy, and
SciPy build dependencies automatically. Windows wheels are not currently built;
the C99 complex-number extension needs porting before Windows support is promised.

## Fluxoid jump rates

```python
from ztpcraft.projects.fluxonium.multiloop_fluxonium import (
    FluxoidSector, SectorBasis, TwoLoopFluxoidSystem,
    jump_matrix, prepare_rates, calculate_sector_rates,
)

basis = SectorBasis((FluxoidSector(0, 0), FluxoidSector(1, 0)), nlevels=3)
print(basis.size)  # six states, grouped by sector
```

See [the matrix-rate guide](ztpcraft/projects/fluxonium/multiloop_fluxonium/MATRIX_RATES.md)
for the model, operator conventions, and a complete calculation.

## Optional features

- `ztpcraft[arrays]`: JAX-based `TwoLoopArrayFluxonium` normal modes.
- `ztpcraft[simulation]`: dynamiqs/JAX simulations in the `f1f2` project.
- `ztpcraft[gpu]`: NVIDIA GPU monitoring.
- `ztpcraft[notebook]`: JupyterLab and a Python notebook kernel.

The separate `fluxonium.array_mode` module requires `ninatool`, installed separately
from its upstream source. It is loaded only when requested. Neither `ninatool` nor
JAX is needed for the fluxoid jump-rate API above. Importing an optional feature
without its dependencies raises the normal Python import error.

## Releases and license

See [RELEASING.md](RELEASING.md) for build checks and the one-time PyPI setup.
Licensed under BSD-3-Clause; see [license.md](license.md) and the retained
[CHENcrafts attribution](license_chencrafts.md).
