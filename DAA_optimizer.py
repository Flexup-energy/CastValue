"""Backward-compatibility shim.

The real implementation moved to `castvalue.bess.daa.optimizer` as part of the
1.3.0 package restructure (CastValue is now a multi-benchmark package, not a
single flat module). `import DAA_optimizer` still works so existing scripts
and notebooks that predate the restructure don't break outright, but it's
deprecated — switch to `from castvalue.bess.daa import Optimizer` (or
`from castvalue.bess.daa.optimizer import Optimizer`).

Plan to remove this shim in a later major version.
"""

import warnings

from castvalue.bess.daa.optimizer import Optimizer

warnings.warn(
    "DAA_optimizer is deprecated and will be removed in a future release; "
    "use `from castvalue.bess.daa import Optimizer` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Optimizer"]
