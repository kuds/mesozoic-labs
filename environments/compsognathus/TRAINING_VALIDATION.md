# Compsognathus SB3 integration validation

Measured September 8, 2026. Both the anatomical and Rev B robot models now
support the repository's SB3 training workflow. This validates the pipeline;
it does not establish a converged walking controller.

These are the original integration results. The subsequent
[September 8 recipe review](TRAINING_RECIPE_REVIEW.md) records the revised
PPO settings, their separate validation, and the full-run qualification that
is still required. Passing the integration suite does not validate a recipe's
convergence or the older species' compatibility with current training code.

Machine-readable evidence: [training validation](data/training_validation_v1.json)
and [current model preflight](data/preflight_training_v1.json).

## Executed checks

| Check | Result |
|---|---|
| Shared infrastructure and Compsognathus regression suite | 2,507 passed; 113 optional-backend tests skipped |
| Affected modules after the final loader/interface fixes | 249 passed |
| Gymnasium, physical/task regressions and sweep-config resolution | 20 passed |
| Real CPU training integration | 8 passed: four shared-trainer cases and four notebook cases |
| Existing model-validation protocol | 26/26 trials passed, including 15% mass growth and motors-off controls |
| Notebook code | All 80 code cells parsed |
| Ruff lint/format and CI-style mypy | Passed; mypy checked 303 source files |
| Website production build | Passed |
| External and head cameras | Both variants rendered 640 × 480 RGB frames through OSMesa |

The runtime used MuJoCo 3.10.0, Gymnasium 1.3.0, SB3 2.9.0 and PyTorch
2.9.0+cpu. The CI typing check used the workflow's minimal dependency set.
A one-line conversion to a Python list fixes an existing recovery-harness
type error exposed by that check, without changing the held action values.

Each shared-trainer case ran PPO or SAC through balance, locomotion and
target reaching, saving and loading matched policy/VecNormalize pairs between
stages. Checks require actual optimizer updates, finite parameters, matching
deterministic predictions after reload, finite evaluation episodes, embedded
plant/task identities, and rejection of the other variant's checkpoint.

Each notebook case executed its real setup, selection, environment and
`train_stage` cells, then generated the shared stance report. This exposed
and fixed the report loader's unconditional use of `PPO.load` for SAC files.
The loader now identifies PPO/SAC from the archive's JSON optimizer metadata
and refuses ambiguous or unsupported metadata before loading.

Smoke runs use 64 training steps per stage, 32-step horizons and smaller
networks. Production advancement thresholds are retained. All four notebook
stance gates correctly failed; their measured reports were still generated.
Those failures are expected and are not evidence against the full recipes.

The new environment tests also cover full 20-second neutral-control standing,
seeded resets, the pelvis quaternion's body axes, action bounds, invalid
actions, horizon truncation, healthy/slow target arrival, and rejection of a
fall at the goal. The robot retains exactly 12 leg actuators and a fixed head
and tail. The original four species' complete plant-manifest entries are
unchanged.

## Reproduce

```bash
python -m pip install -e ".[train,test,viz]"
pytest environments/compsognathus/tests environments/shared/tests/test_compsognathus_training.py
python -m environments.shared.plant_contract --check
python -m environments.shared.species_catalog --check
python -m environments.compsognathus.scripts.validate_models --output /tmp/compsognathus-preflight.json
```

Run full training next, comparing learned behavior with the zero-action
baseline and enforcing the recorded gates. JAX/MJX, distributed Ray execution,
onboard state estimation, hardware transfer and learned walking performance
are outside this validation. The initial robot policy uses privileged
simulator state; the camera is available separately, not as an MLP input.
