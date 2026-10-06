# Physical gait reports and certification

The shared gait observer measures contact, stepping, slip and progress for the exact saved policy and normalization pair. Existing training configs and historical verdicts remain unchanged. They produce **report-only** development evidence; enforcement requires an explicit `locomotion_gait/v1` curriculum declaration.

A certificate qualifies a declared engineering event on a particular simulated plant. It does not establish biological plausibility, identify walking versus grounded running from energetics, or establish hardware performance. See the broader [gait quality plan](GAIT_QUALITY_PLAN_2026_09.md) and [historical audit](investigations/GAIT_AUDIT_2026_09.md).

## Measurement

- Record the initial reset state and every physics substep through the existing probe hook. The observer does not advance dynamics, alter observations/rewards, or consume reset RNG. A previous hook runs first and is restored on exit.
- Sum active solved normal forces between explicitly registered distal-foot geometries and declared terrain. Whole legs are not feet. Foot-on-foot force and nonfoot ground force are separate diagnostics; inactive proximity contacts carry no support. This follows [MuJoCo's contact representation](https://mujoco.readthedocs.io/en/latest/computation/index.html#contact).
- Normalize load by the animal subtree's weight, excluding prey/food. `L` is the declared home-keyframe mean hind-hip anchor height above the authored reference plane; it is a scale reference, not a reconstructed anatomical measurement.
- Detect load with on/off hysteresis and dwell times in seconds. Count only complete observed touchdown–liftoff–touchdown cycles inside the analysis window. Initial contact and unconfirmed terminal transitions create no steps. Valid cycles also require swing clearance and horizontal foot repositioning.
- Report duty factor as stance duration/stride duration. Interlimb phase is touchdown offset/stride period modulo one; matching uses circular distance and includes missing matches in its denominator. Cadence, support counts and per-foot diagnostics remain available for interpretation.
- Measure tangential velocity at each loaded contact point relative to terrain, including rigid-foot rotation. A rolling foot can move at its site while its contact point does not slide. `max_slip_distance_over_leg` is the worst loaded-stance integral of slip speed divided by `L`, including partial boundary stances and positively loaded contacts below detector thresholds.
- `body_support_fraction` is nonfoot ground-force impulse/total animal ground-force impulse. It is distinct from time in body contact. Flight fraction and longest flight are time-based. Missing, malformed or nonfinite required telemetry fails qualification; unavailable JSON metrics are `null`.

The `GaitProtocol` defaults are **provisional detector settings**: load on/off at 0.02/0.01 body weights, stance/swing/simultaneous windows of 0.02 s, phase tolerance 0.15 cycles, clearance 0.01 `L`, and repositioning 0.02 `L`. Calibrate detector settings and engineering acceptance criteria on separate development data before locking them. These numbers are not biological standards.

## Profiles and criteria

| Profile | Required timing, in addition to real cycles, progress and support limits |
| --- | --- |
| `biped_alternating` | Left/right alternation; reject synchronous or badly staggered hops |
| `quadruped_walk` | Fore and hind left/right alternation plus separated four-beat touchdown sequences |
| `quadruped_trot` | Fore and hind alternation plus synchronous diagonal pairs |
| `quadruped_pace` | Fore and hind alternation plus synchronous ipsilateral pairs |

There is no automatic gait-profile selection. Flight limits are explicitly declared: an alternating aerial run must not be rejected merely for flying, and an absence of flight does not distinguish a walk from grounded running. [Rubenson et al. (2004)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1691699/) measured running mechanics without an aerial phase in ostriches.

Each qualifying episode must jointly meet its completion, analysis duration, forward displacement speed, per-foot complete/valid cycle counts, phase, simultaneous-touchdown, flight, slip and body-support criteria. Optional foot-on-foot and panel reward rails are explicit. All mandatory fields are listed in [`GaitGateThresholds`](../environments/shared/curriculum/gait_gate.py); enforcement supplies no hidden acceptance defaults. `required_consecutive`, when present, must be 1 because the gate judges one fixed panel.

## Freeze the protocol without viewing the panel

Reserve the registered publication block **3042–3081** for the 40-episode certification panel. Do not use it for reward design, detector/threshold calibration, checkpoint selection or retries after inspecting outcomes. A code check can exclude recorded seed roles; it cannot prove that a human never tuned against held-out results.

Plan the hash before collecting held-out telemetry:

```bash
python -m environments.shared.scripts.gait_report trex --stage locomotion \
  --protocol-only --env-json frozen_env.json \
  --episodes 40 --seed 3042 --settle-s 1.0 > planned_protocol.json
```

`--protocol-only` constructs and validates the plant, prints the measurement payload/hash, and rolls no episodes. It needs no output directory and writes no report, even if `--out-dir` is supplied. `--protocol-json` can provide a plain object of detector options. `--direction X Y` is normalized before hashing.

Freeze the resulting `measurement_protocol_sha256`, the chosen profile, all gate criteria, exact `min_eval_episodes`, and `gait_panel_seed_start` in the proposed stage TOML. The hash pins detector options, implementation bytes, geometry registry, MuJoCo version, sampling rates, direction, settling exclusion, horizon and panel. Source changes require a new planned hash before evaluation.

## Development reports and saved-policy replay

Extract the exact recorded environment constructor kwargs into a plain JSON object. `--env-json` replaces the current config's environment kwargs; it does not merge them or silently recreate an old task from current defaults.

```bash
python -m environments.shared.scripts.gait_report trex --stage locomotion \
  --model replay/models/policy.zip --vecnorm replay/models/policy_vecnorm.pkl \
  --env-json frozen_env.json --episodes 3 --seed 9000 --settle-s 1.0 \
  --out-dir fresh_development_report
```

Three pilot episodes are development evidence, never a certification result. `--zero-action` replaces `--model` for a negative control and cannot certify. With existing reward gates, even a saved-policy report that clears provisional diagnostic criteria remains report-only.

Copy the checkpoint's recorded `stage_config.json` beside the checkpoint or one directory above it, and copy nearby run `provenance.json` when available. Keep the explicit matched VecNormalize file with the pair. The saved config must record `run.seed` and `run.n_envs`, corroborated by the checkpoint's JSON metadata. Known training seeds `seed+rank`, selection seed `seed+1000`, and recorded development/calibration/selection roles must not overlap certification seeds. Missing or conflicting seed provenance makes diagnostics ineligible and strict certification refuse.

Saved policies require compatible plant metadata and the exact task fingerprint. Normalization is frozen and reward normalization is disabled during inference; there is no sidecar guessing or unnormalized fallback. `--allow-legacy-plant` is diagnostic only and cannot mint a certificate.

After locking a new gait gate, use its explicit `--config gait_gate.toml`, the same frozen environment/protocol options, the selected pair, and the declared 40 episodes starting at 3042. Do not extend a failing panel or choose a favorable subset.

## Evidence and interpretation

- `gait_traces/episode_0000.npz` etc. retain numeric physics-substep telemetry for every episode.
- `gait_panel.csv` retains episode metrics and checkpoint, normalization, task and protocol bindings; `metrics_json` includes complete per-foot and timing diagnostics.
- `gait_report.json` records plant/task identities, geometry/scale references, detector/panel settings, seed provenance, file hashes, qualification failures and confidence statistics. Readers verify bindings and replay metrics from the raw traces rather than trusting a stored pass flag.
- Changed checkpoint/normalization bytes during measurement refuse the result. Validation/load failures invalidate an earlier report with an incomplete, noncertifying record. Completed result bundles, including nested or symlinked output paths, are immutable; use a fresh directory outside them.

For `k` joint successes out of fixed `N`, the one-sided exact 95% Clopper–Pearson lower bound is `BetaQuantile(0.05; k, N-k+1)` (`0` when `k=0`). At `N=40` and a declared lower-bound floor of `0.80`, **37/40 passes and 36/40 fails**. The certificate concerns the specified reset/scenario distribution and episode event; it does not cover arbitrary terrain, perturbations or species.

CLI exit 0 means a report or preflight completed, **not that a certificate passed**. Read `report_only`, `certification_eligible`, `certified`, failures and statistics. Exit 2 indicates command usage errors; exit 3 indicates an evaluation refusal.

Mac CPU replay is fresh evidence on that runtime, not confirmation of an original GPU trajectory. Contact-sensitive runs can diverge across hardware/backends; preserve runtime and plant bindings, and do not certify unverified model/physics revisions. The observer does not modify frozen MJX code or species physics. The [historical audit's replay limitations](investigations/GAIT_AUDIT_2026_09.md) explain this distinction.
