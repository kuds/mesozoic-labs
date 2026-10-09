# Stance reward experiments, October 2026

The finalized October 8 Tyrannosaurus Rex runs survive 40/40 episodes but
fail the current stance gate: seed 42 stands near a sole edge in every
episode; seed 48 has three pressure-margin failures and one tilt failure.
The October 7 Velociraptor policy also survives 40/40, but typically hops
and unloads intended support geoms. This study prices those failures and
then compares fresh PPO training under paired seeds.

These are **research pilots**, not curriculum runs or certificates. All
production environments, observations, physical plants, configs, Recovery
inheritance, and gate thresholds remain unchanged. The experimental rewards
live in wrappers in this directory. A winning variant still needs a full
training replication, task fingerprint, statue calibration, and review
before adoption.

## Fixed experiment matrix

| Species | Arm | Reward delta |
| --- | --- | --- |
| Tyrannosaurus Rex | `control` | Current reward |
| Tyrannosaurus Rex | `cop_margin` | Worst foot's squared fore-aft pressure deficit; safe fraction 0.65, weight 0.10 |
| Velociraptor | `control` | Current reward |
| Velociraptor | `support_fraction` | Increase the support-conditioned fraction of the alive bonus from 0.20 to 0.50 |
| Velociraptor | `contact_quality` | Airborne fraction weight 0.50, impact excess over 2 body weights weight 0.25, support-coverage deficit weight 0.15 |

Every arm uses training seeds **42, 48, 50**, four environments, the existing
`[512, 256]` network and each species' PPO recipe, for **1,048,576 steps**.
That is 15 pilots / 15,728,640 training steps. The impact/coverage/flight
arm is deliberately separate from the alive-fraction arm. It tests a bundle;
it cannot tell which of its three terms caused an outcome without another
ablation. Stronger flatness penalties and heading changes are deferred.

The learning-rate schedule follows elapsed steps against the original
11-million / 6-million-step budget; it is not compressed into this pilot.
Entropy decay keeps its original elapsed-step schedule. VecNormalize uses
the production gamma and clipping limits. CPU execution and the installed
Python/Torch versions are recorded; this is not a bitwise reproduction of
the Colab training jobs.

## Measurement and pricing

`StepFloorProbe` chains the env's existing physics-substep observer, reads
normal forces and positions, and reduces one control step in constant
memory. It reuses the floor-truth recorder's normal-force decoder. CoP is
force-weighted over the entire step in each sole's local frame, normalized
by its own half-length. Missing or unloaded sole support counts as an edge.
The pressure cost takes the worst foot, with no left/right averaging. It
starts after the existing settle window (200 steps for Tyrannosaurus Rex).
This per-step surrogate is intentionally distinct from the gate's
worst-foot mean over the episode window.

Velociraptor has no box sole. Its contact arm uses whole-limb floor forces
and the existing morphology registry's digit III, digit IV, and metatarsal
support geoms. Forces are normalized by body weight. Coverage is the weaker
foot's mean registered-geom loaded fraction across substeps. Flight, impact,
and coverage costs respect the recorder's 0.10-second spawn grace.

Before training, calibration uses 40 reset seeds **3042–3081** for the
passive statue, both selected Tyrannosaurus Rex checkpoints, the selected
Velociraptor checkpoint, and an initial-exploration proxy. All candidate
rewards rescore the *same* trajectories. The proxy samples clipped raw
normal actions at the configured initial standard deviation; it is a
pricing stress test, not a PPO policy or training result. Compressed per-step
traces retain CoP and contact evidence that the finalized summary CSVs lack.

The preflight checks compare streaming measurements with the independently
reduced episode recorder, compare normal forces with `mj_contactForce`,
prove control trajectories are bit-identical, and compare the support
delta with the actual env's `support_conditioned_alive_fraction=0.5`
configuration. They also test missing-support behavior and schedule anchoring.

## Evaluation and interpretation

Every completed pilot gets a deterministic, 40-episode panel on **23042–23081**,
with a passive baseline on those same seeds. These seeds are disjoint from
calibration. Final checkpoint evaluation avoids selecting the best result
on this panel. Compare pressure position, sole tilt, intended support
coverage, flight/landings, saturation, falls, and clean-episode confidence.
An improvement must reproduce across training seeds and avoid trading one
physical failure for another. A later full-budget replication must also
pass the publication panel and the existing consecutive-panel protocol.

`original_task_gate_diagnostic` explicitly uses original-task rewards and
unchanged original thresholds alongside the physical measurements. Candidate
returns are logged separately. A favorable diagnostic never certifies an
altered-reward task. No gate threshold is relaxed, no robust handoff is
generated, and no Recovery stage starts from these runs.

## Reproduction

Install the project and SB3 training/test dependencies, with MuJoCo 3.10.0,
Gymnasium 1.3.0, SB3 2.9.0, and NumPy 2.1.3. The local CPU pilots use Torch
2.8.0+cpu and Python 3.12; all exact versions are saved with each result.

Place the selected checkpoints and their paired statistics in `$STANCE_INPUTS`:
`trex_seed42_model.zip`, `trex_seed42_vecnorm.pkl`,
`trex_seed48_model.zip`, `trex_seed48_vecnorm.pkl`,
`velociraptor_seed42_model.zip`, `velociraptor_seed42_vecnorm.pkl`.
These are the October 8 Tyrannosaurus Rex and October 7 Velociraptor
`robust_best_model` pairs. The loader checks embedded plant identities and
handles inference schedules across Python minor versions.

From the repo root, with task-specific directories set:

```bash
python -m pytest docs/investigations/stance_rewards_2026_10/test_prototype.py -q
python docs/investigations/stance_rewards_2026_10/run_experiments.py calibrate \
  --species trex --inputs "$STANCE_INPUTS" --output "$STANCE_CALIBRATION"
python docs/investigations/stance_rewards_2026_10/run_experiments.py calibrate \
  --species velociraptor --inputs "$STANCE_INPUTS" --output "$STANCE_CALIBRATION"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python docs/investigations/stance_rewards_2026_10/run_experiments.py batch \
  --calibration "$STANCE_CALIBRATION" --output "$STANCE_RUNS" --workers 4
```

Inspect calibration before starting the batch: preserve the passive stance's
reward, remove the edge-loaded policy's reward advantage, and check that
candidate costs do not eliminate positive nonterminal reward under initial
exploration. The batch refuses incomplete or source-mismatched calibration.
Use a fresh output directory; there is no silent overwrite or resume.

Each run saves its exact configs, initial policy hash, progress every 32,768
steps, a paired model/normalizer, a training log, and final episode evidence.
The batch manifest records queued/running/completed/failed states and PIDs.
Pilot processes run locally; their continued execution depends on the host
remaining available. Check both the manifest and actual processes before
reporting a job as running.
