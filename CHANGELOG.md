# Changelog

## 1.3.3

Bug fix: `castvalue.__version__` was a hardcoded string that drifted out
of sync with `pyproject.toml`'s version across the 1.3.1 and 1.3.2 patch
releases (stuck at `"1.3.0"`) -- caught via `castvalue-compute` integration
testing, where every result's `provenance.castvalue_version` was silently
wrong. Now read dynamically from installed package metadata
(`importlib.metadata.version("castvalue_bench")`), so it can't drift again.
No methodology/numerics changes versus 1.3.2.

## 1.3.2

PyPI distribution renamed to `castvalue_bench` -- `castvalue` collided
with PyPI's typosquat-prevention check (too similar to the existing,
unrelated `cast-value` package). Python import path is unchanged: still
`import castvalue`. No runtime/methodology changes versus 1.3.1.

## 1.3.1

Packaging fix: `pandas` is needed by the BESS DAA golden regression test
(reads `data/master_energy_data_v2.csv`) but was never declared as a
dependency anywhere -- it only worked locally because the dev environment
already had it installed globally. Caught by CI on the real v1.3.0 tag
(the `test-bess` job failed with `ModuleNotFoundError`). Added to the
`test` extra. No runtime/methodology changes versus 1.3.0.

## 1.3.0

Multi-benchmark package restructure. `castvalue` is no longer a single flat
module -- it's now a `src/` package with shared machinery in
`castvalue.core` (coalitions, Shapley/Harsanyi attribution, imbalance
pricing, sensitivity curves, ceiling checks) and one subpackage per
benchmark.

- **`castvalue.bess.daa`** -- the existing day-ahead-only BESS LP
  optimizer, moved from the repo root unchanged (numerics verified
  byte-for-byte identical via a golden regression test). Root-level
  `DAA_optimizer.py` is now a backward-compat shim (`DeprecationWarning`).
- **`castvalue.core`** -- new. Generic `Benchmark` protocol, `DayResult`,
  provenance, frozen/hashable `PolicyConfig`, the 2^n coalition runner,
  Shapley values (two independent formulas, cross-checked), Harsanyi
  dividends/interactions, grouped-game remapping, sensitivity curves, and
  the Perfect Foresight ceiling check (`"hard"`/`"soft"`).
- **`castvalue.pv.imbalance`** -- new. PV imbalance-cost benchmark v2: a
  two-stage (D-1 + intraday) decision policy over 7 forecast inputs
  (production, DA price, ID1 price, and imbalance price -- each a
  shortage/surplus pair -- at two forecast vintages), TenneT NL dual
  imbalance pricing, and an exact closed-form vertex-enumeration solver
  (no LP solver in the runtime path; validated in tests against both a
  real LP oracle and a brute-force grid search).
- **`castvalue.bess.multimarket`** -- placeholder only, raises
  `NotImplementedError`. Scope documented, not implemented.
- `pyproject.toml` switched to a `src/` layout; base dependency is now
  `numpy` only. New extras: `bess` (`pyomo`, needs a GLPK binary
  installed separately), `test` (`pytest`, `scipy` -- scipy is a PV
  test-only oracle dependency, never imported at runtime), `all`.
- CI: tests run with and without the `bess` extra (core + PV must pass
  with zero `pyomo` installed); a tag-triggered PyPI publish workflow is
  wired up via trusted publishing, pending the PyPI-side project
  configuration.

## 1.2.0 and earlier

See git tags `v1.0`, `v1.1`, `v1.2` -- predates this changelog.
