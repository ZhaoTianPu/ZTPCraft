from setuptools import setup, Extension

try:
    import numpy as _np
except Exception as exc:
    raise RuntimeError(
        "NumPy is required at build time to compile Cython extensions."
    ) from exc

# Configure the extension; fall back to a pre-generated C file if Cython is not available
ext = Extension(
    name="ztpcraft.bosonic._oscillator_integrals_1d",
    sources=["ztpcraft/bosonic/_oscillator_integrals_1d.pyx"],
    include_dirs=[_np.get_include()],
    language="c",
)

try:
    from Cython.Build import cythonize

    ext_modules = cythonize([ext])  # , language_level="3")
except Exception:
    # If Cython is not available, attempt to build from a generated C source
    ext.sources = ["ztpcraft/bosonic/_oscillator_integrals_1d.c"]
    ext_modules = [ext]

setup(ext_modules=ext_modules)
