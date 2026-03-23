# Auto-discover the installed package version.
# Falls back to '0.0.0+dirty' when the package metadata is unavailable
# (e.g. when running from a source checkout without installation).

from importlib.metadata import version as _pkg_version, PackageNotFoundError


try:
    version: str = _pkg_version("ztpcraft")
except PackageNotFoundError:  # pragma: no cover – running from source tree
    version = "0.0.0+dirty"
