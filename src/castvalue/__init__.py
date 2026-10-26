"""`__version__` is read from installed package metadata, not hardcoded --
a hardcoded string here already drifted out of sync with pyproject.toml
across the 1.3.1/1.3.2 patch releases (every result's `provenance.
castvalue_version` was silently wrong). Reading it dynamically means it
can never drift again, by construction.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("castvalue_bench")
except PackageNotFoundError:
    # Not installed via pip (e.g. a raw source checkout with no install
    # step at all) -- fall back rather than crash on import.
    __version__ = "0.0.0+unknown"
