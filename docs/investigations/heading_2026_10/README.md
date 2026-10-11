# Heading investigation: executable evidence

Read the [complete design](../../HEADING_INVARIANCE_DESIGN_2026_10.md) for the
recommendation, production file map, migration, certificate and limitations.
This directory contains a research wrapper, not an alternate production
environment or a certified training entry point.

## Contents

- `prototype.py`: projected-gravity and yaw-removed-quaternion candidates; the
  latest version includes a covariant finite fallback for vertical pelvis X.
- `test_prototype.py`: frame algebra, target-bearing preservation, quaternion
  sign, local sensor/root agreement, terminal frames, live command rotation and
  the correlated-panel counterexample.
- `run_study.py`: pointwise comparisons, full-floor-truth rollouts and short PPO
  integration experiments. It imports the current production stance settings.
- `pointwise_*.json`: 100 tilted/moving states × seven rotations per species;
  `_terminal_fallback` results recheck the later fallback implementation.
- `rollouts_*.json`: eight fresh seeds × six headings; T. rex includes the
  seed-50 checkpoint and its paired normalizer, both species include zero action.
  T. rex's statue rows agree to about 10⁻¹¹ at multiples of 90° but differ
  slightly at ±45°, where its pyramidal friction cone is anisotropic (the
  design's [Scope of the symmetry](../../HEADING_INVARIANCE_DESIGN_2026_10.md#scope-of-the-symmetry)).
- `pilot_*_{11,12}.json`: four 8,192-step PPO smoke experiments, with 16 paired
  evaluation episodes each at −90°, 0°, +90° and 180° only. The policies
  effectively stayed at the zero-action statue, so these validate plumbing
  (parameters updated, rotated episodes completed), not learned heading
  robustness. These are not production-budget or gate results.

All initial results and their exact source files are preserved in `221d44e`.
JSON `base_commit` is HEAD when the run started; `scripts_sha256` identifies the
actual research source, including then-uncommitted files. Those source versions
are retained in the branch history. Do not infer a different code version from
the name of a JSON file. The two downloaded checkpoint files are not committed.
The local capture commit `7b334bc` and published commit `221d44e` have identical
tree `1b860f0690061ef5134a330b5b3331a12f0a8372`; their commit metadata differs.

## Commands

From the repository root, in an environment with the repository, its test and
SB3 dependencies installed (the recorded runtime used an editable repository
install, so direct script execution can import `environments`):

```bash
python -m pytest -o addopts='' -q docs/investigations/heading_2026_10/test_prototype.py

python docs/investigations/heading_2026_10/run_study.py pointwise \
  --species trex --output /tmp/pointwise_trex.json
python docs/investigations/heading_2026_10/run_study.py pointwise \
  --species compsognathus --output /tmp/pointwise_compsognathus.json

python docs/investigations/heading_2026_10/run_study.py rollouts \
  --species trex --seed 18042 --episodes 8 \
  --model /path/to/seed50_model.zip --vecnorm /path/to/seed50_vecnorm.pkl \
  --output /tmp/rollouts_trex.json
python docs/investigations/heading_2026_10/run_study.py rollouts \
  --species compsognathus --seed 18042 --episodes 8 \
  --output /tmp/rollouts_compsognathus.json

python docs/investigations/heading_2026_10/run_study.py pilot \
  --species trex --training-seed 11 --timesteps 8192 \
  --output /tmp/pilot_trex_11.json
python docs/investigations/heading_2026_10/run_study.py pilot \
  --species compsognathus --training-seed 11 --timesteps 8192 \
  --output /tmp/pilot_compsognathus_11.json
# Repeat the two pilot commands with --training-seed 12 for the second seed.
```

The exact initial runtime is recorded in every JSON: Python 3.12.14, MuJoCo
3.10.0, NumPy 2.5.3, Gymnasium 1.4.0, SB3 2.9.0, torch 2.14.1+cpu. Torch differs
from the repo's current 2.13.0 CI pin. Use the source files at `221d44e` to
reproduce the initial runs exactly; the final source changes are documented in
the design. Numerical differences from contact solves need not be bit-identical
across machines.

## Verification

| Command/check | Result |
|---|---|
| `pytest -o addopts='' -q` | 5,417 passed, 1 skipped, 7 warnings; 852.34 seconds |
| Research prototype plus `test_stance_heading_probe.py`, `test_direction_commands.py`, `test_command_frame.py` | 113 passed in 6.93 seconds; 46 research tests and 67 existing tests |
| `pytest -o addopts='' -q environments/shared/tests/test_landing_records.py` after the doc link | 3 passed |
| `python -m environments.shared.plant_contract --check` | Current |
| `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check` | Current; 649 lines, zero errors |
| `python -m environments.shared.species_catalog --check` | Current |
| `ruff check docs/investigations/heading_2026_10` and `ruff format --check docs/investigations/heading_2026_10` | Passed with ruff 0.16.9 |
| `git diff --check` | Passed |
| SHA-256 validation of source files recorded by all ten JSON results | All matched their retained source versions |
| 2026-10-10 documentation revision (Python 3.13): this directory's `test_prototype.py`, `test_landing_records.py`, `git diff --check` | 46 passed; 3 passed; passed. Research code and JSON results unchanged |

The full suite includes the 67 existing focused tests, so these counts should
not be added as independent coverage. The 46 research tests live outside the
configured production test directories and were run explicitly. Production
manifests, configurations and generated files are unchanged.
