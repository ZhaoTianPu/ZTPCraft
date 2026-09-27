from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extension = Extension(
    "ztpcraft.bosonic.oscillator_integrals._oscillator_integrals_1d_quadrature",
    ["ztpcraft/bosonic/oscillator_integrals/_oscillator_integrals_1d_quadrature.pyx"],
    include_dirs=[np.get_include()],
    language="c",
)

setup(ext_modules=cythonize([extension], compiler_directives={"language_level": 3}))
