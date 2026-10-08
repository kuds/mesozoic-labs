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
and [current model preflight](data/preflight_training_v1.json). The checks
below ran on the anatomical physics r1 plant; the preflight file has since
been regenerated on physics r2 (the appended section at the end).

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
baseline and enforcing the recorded gates. Onboard state estimation, hardware transfer and learned walking performance
are outside this validation. The initial robot policy uses privileged
simulator state; the camera is available separately, not as an MLP input.

## Anatomical physics r2 (appended 2026-10-07)

Decision D-D26 revised the anatomical model (physics r1 → r2, policy
interface r2 → r3, visual r1 → r2; `configs/plant_versions.toml` note 15) and
its stance task and gate; the robot is unchanged. Measured on the stance task
(`frame_skip` 10) with the zero-action statue, r1 → r2:

| Check | r1 | r2 |
|---|---|---|
| Settled metatarsus clearance | 1.07 mm | 4.81 mm |
| Centre of mass ahead of the pad's rear edge (of the 78 mm foot) | 19.6 mm (0.25) | 29.3 mm (0.38) |
| Largest 0.1 s pelvis push held, 5/5 seeds: forward / backward / lateral | 0.225 / 0.125 / 0.45 BW | 0.275 / 0.20 / 0.45 BW |
| Largest initial tilt held, 3/3 seeds: nose-down / nose-up / roll | 2° / 4° / 15° | 5° / 7° / 15° |
| Full episodes under zero-mean action noise, σ 0.02 | 5/20 | 20/20 |
| ... σ 0.135, the stance recipe's initial σ | 0/20 (median 12 steps) | 39/40 (151/160 over four seed blocks) |
| ... σ 0.10 / σ 0.20 | — | 40/40 / 0/40 (median 180 steps; 129 with an independent noise stream) |
| Stance reward of the statue, 40 episodes from seed 3042 | 2998.74 ± 1.22 | 2999.1 under the r1 terms; 4570.4 ± 7.2 under the stance-quality terms |
| `stance_quality/v2` statue panel (seeds 3042–3081; 7042–7081; 9042–9081) | — | clean on 40/40 each |

Each revision runs through its own action interface (the r1 linear residual,
the r2 soft-cubic leg residual), with paired noise streams. The regenerated
`data/preflight_training_v1.json` records `passed: true`: with the toe
armature of `configs/plant_versions.toml` note 15 (e) every one of the
anatomical model's ten seeded holds passes `supports_weight` (worst settled
ground-force error 0.31% against the 3% bar), where the default armature's
numerical 19–21 Hz heel chatter of the MTP servo failed seeds 49 and 51
(5.4%). The new tests
(`tests/test_biological_rewards.py`, `tests/test_noise_tolerance.py`, the
additions to `tests/test_models.py` and `tests/test_training_env.py`, and
`environments/shared/tests/test_compsognathus_stance_gate_config.py`) pin the
plant, the interface, the reward terms, the noise cliff and the gate
declaration. No r2 policy has been trained to convergence; the training
recipe's appended amendment says what to run next.
