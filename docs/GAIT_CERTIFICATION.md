# Physical gait reports and certification

The shared gait observer measures contact, stepping, slip and progress for the exact saved policy and normalization pair. Existing training configs and historical verdicts remain unchanged. They produce **report-only** development evidence; enforcement requires an explicit `locomotion_gait/v2` curriculum declaration.

A certificate qualifies a declared engineering event on a particular simulated plant. It does not establish biological plausibility, identify walking versus grounded running from energetics, or establish hardware performance. See the broader [gait quality plan](GAIT_QUALITY_PLAN_2026_09.md) and [historical audit](investigations/GAIT_AUDIT_2026_09.md).

## Measurement

Recording is unchanged from v1:

- Record the initial reset state and every physics substep through the existing probe hook. The observer does not advance dynamics, alter observations/rewards, or consume reset RNG. A previous hook runs first and is restored on exit.
- Sum active solved normal forces between explicitly registered distal-foot geometries and declared terrain. Whole legs are not feet. Foot-on-foot force and nonfoot ground force are separate diagnostics; inactive proximity contacts carry no support. This follows [MuJoCo's contact representation](https://mujoco.readthedocs.io/en/latest/computation/index.html#contact).
- Normalize load by the animal subtree's weight, excluding prey/food. `L` is the declared home-keyframe mean hind-hip anchor height above the authored reference plane; it is a scale reference, not a reconstructed anatomical measurement.
- Measure tangential velocity at each loaded contact point relative to terrain, including rigid-foot rotation.

The v2 measurement core (`gait/events.py`, `gait/metrics.py`, `gait/labels.py`) treats a gait as a phase-locked pattern of limb oscillators (Collins and Stewart 1993; Golubitsky et al. 1999) and confirms it with Hildebrand's footfall estimator:

1. **Contact segmentation per limb.** A foot is loaded above 0.01 BW / n limbs. Unloads are merged back into stance when they are sub-dwell chatter (at most 10 ms), scuffs (never clear 0.01 `L` and move the foot less than 0.03 `L`, or are short) or impact bounces (shorter than 0.25 and lower than 0.3 of the limb's reference swing, the upper quartile of its clear swings). Loads shorter than 6 ms are dropped only after merging, so a touchdown is the first contact of a stance and an impact bounce never delays it. A swing is valid when it clears 0.01 `L` and repositions the foot 0.03 `L`.
2. **Continuous phase.** Each limb's phase rises linearly between touchdowns. A stride longer than twice the limb's median stride is a pause and leaves the phase undefined; before the first and after the last touchdown the phase is extrapolated for one median stride. Undefined phase is coverage loss, so a limb that stops cycling, a stand or a one-leg bout costs coverage in proportion to its duration.
3. **Pair statistics.** For every limb pair, the relative phase (lag of the second limb behind the first, in cycles) gives a time-weighted circular mean and mean resultant length (the pair's phase-locking index), a centred one-stride sliding estimate, and Hildebrand's per-stride footfall phases pooled in both directions. A duty-normalised stance-overlap index, which does not depend on touchdown timing, is 0 for perfect anti-phase occupancy and 1 for in-phase occupancy.
4. **Templates.** A profile is a list of limb pairs with target phase sets. Template coverage is the time during which every templated pair is locally locked within its target set widened by 0.05 cycles. A sample is grossly off-template when a phase is undefined, the pair is locally unlocked, or its local mean is 0.10 cycles outside the target. `wrong_locked_fraction` is time locked at a wrong relative phase (another gait).
5. **Physical quantities.** Duty factor and load share per limb; contralateral load and duty ratios; lead-limb exchange (fraction of strides in which the fore-aft order of a contralateral pair flips past ±0.01 `L` in both directions; step-through gaits flip twice per stride, step-to gaits never); the per-limb **skid fraction** (stance slip distance over trunk travel during the same stances, a ratio of sums so one bad stance cannot fail an episode); `body_support_fraction` (nonfoot ground impulse over total animal ground impulse); time with foot-on-foot force above 0.01 BW; flight fraction.
6. **Diagnostics (never verdict inputs).** Per-stride Hildebrand labels and their time-weighted distribution (walk, grounded or aerial run, hop, staggered hop, skip; lateral- or diagonal-sequence walk or amble, walking/running/flying trot and pace, pronk, bound, half-bound, canter, transverse or rotary gallop), Froude number, stride length over `L` and over Alexander's (1976) prediction, limb-phase mean and concentration, hind duty, phase coordination index (Plotnik et al. 2007), alternation index, biped hop-flight fraction, per-foot median step length, and the contralateral phase offset anchored at mid-stance as well as at touchdown.

Every reduction that reaches a stored metric is order-deterministic (`math.fsum`, sequential `cumsum`, scalar `math` trigonometry and exact elementwise arithmetic; no BLAS reductions and no SIMD transcendental ufuncs), and stored values are rounded to six decimals. The gate judges only stored values, so the reader's re-judge from `metrics_json` and its hash-checked replay from the raw traces reproduce the in-process verdict across machines. Missing, malformed, transposed, negative or nonfinite required telemetry, a non-positive body weight or leg length, or an unknown foot registry fails closed; unavailable JSON metrics are `null`.

`GaitProtocol` holds every metric-shaping setting above, including the template tolerances that define coverage (0.09 cycles around anti-phase, 0.125 around synchrony, limb-phase walk band [1/8, 3/8] ∪ [5/8, 7/8], local widening 0.05, gross widening 0.05). They are hashed into `measurement_protocol_sha256`; changing one requires a new planned hash and a fresh panel. The gate bars below are separate, so retuning a bar is a re-judge of stored evidence.

## Profiles and criteria

| Profile | Templated pairs (lag of the second limb behind the first) | Flight cap (provisional) |
| --- | --- | --- |
| `biped_alternating` | r→l at 0.5 | 0.65: aerial running is a run |
| `quadruped_walk` | fr→fl and rr→rl at 0.5; limb phase rr→fr and rl→fl in [1/8, 3/8] ∪ [5/8, 7/8] | 0.05: walks and ambles have no suspension |
| `quadruped_trot` | fr→fl and rr→rl at 0.5; diagonals fr→rl and fl→rr at 0 | 0.50: flying trots are trots |
| `quadruped_pace` | fr→fl and rr→rl at 0.5; ipsilaterals fr→rr and fl→rl at 0 | 0.50: flying paces are paces |

**Quadruped boundary conventions (need sign-off by the gait owners).** The limb-phase circle is split at the 1/8 midpoints between pace (0), single-foot lateral-sequence walk (1/4), trot (1/2) and single-foot diagonal-sequence walk (3/4), so the three quadruped profiles are mutually exclusive and leave no gap. This is coarser than Hildebrand's gait graph, which places lateral- and diagonal-couplet categories between the single-foot walks and the pace or trot: a diagonal-couplets walk with limb phase 0.38–0.44 (the crocodylian high walk, Reilly and Elias 1998, relevant to *Dibothrosuchus*) is certified as `quadruped_trot`, not `quadruped_walk`, and a trot or pace with 21–50 ms of dissociation is never a walk. An amble or running walk (duty below 0.5 without suspension) is a walk; LS and DS walks are both accepted, which keeps the verdict invariant under fore/hind relabelling; flying trots and paces are trots and paces. The episode label shows the finer Hildebrand class.

Each qualifying episode must jointly meet every declared criterion. All criteria are explicit curriculum keys with no hidden defaults; a key that another profile consumes is refused rather than ignored. Failure reasons read `group/rail: value (worst limb or pair) op bar`; the `group/rail` prefix is stable, so reports aggregate failures by rail.

| Group | Criterion (gate key) | Provisional value | Worst genuine development value |
| --- | --- | --- | --- |
| episode | `completed_horizon` must be true | — | — |
| episode | `min_episode_duration_s` (analysis window; tolerance 1.5 × the largest sample interval absorbs float clock accumulation) | horizon − settle | 18.999999999999794 s ≥ 19 s |
| episode | `min_episode_forward_vel` | task speed bar | — |
| participation | `min_limb_phase_coverage` | 0.80 | 0.977 (velociraptor) |
| participation | `min_limb_duty` | 0.10 | 0.29 (velociraptor at 10 ms) |
| participation | `min_relative_limb_load_share` (share × n limbs) | 0.30 | 0.45 (puppet quadrupeds carry 75/25 fore/hind) |
| participation | `min_pair_load_ratio`, `min_pair_duty_ratio` | 0.60, 0.60 | 0.80, 0.77 |
| participation | `min_complete_cycles_per_foot` | 3 | 10 |
| stepping | `min_valid_swing_fraction` | 0.75 | 0.81 (compsognathus at 10 ms) |
| stepping | `min_median_swing_clearance_over_leg` | 0.02 | 0.035 (compsognathus at 10 ms) |
| stepping | `min_lead_exchange_fraction` | 0.15 | 0.26 (compsognathus 1001 at 10 ms) |
| stepping | `min_step_length_over_leg` (optional; see open decisions) | not declared | −0.10 (compsognathus 1001) |
| support | `max_flight_fraction` | per profile, above | 0.33 (velociraptor) |
| support | `max_body_support_fraction` | 0.01 | 0 |
| support | `max_foot_foot_contact_fraction` | 0.02 | 0 |
| support | `max_skid_fraction` | 0.50 | 0.29 (velociraptor touchdown skid) |
| coupling | `min_phase_locking` (continuous and footfall estimators) | 0.60 | 0.70 (velociraptor at 10 ms) |
| coupling | `max_alternation_phase_offset` (circular, both estimators) | 0.09 | 0.076 (compsognathus 0921) |
| coupling | `max_alternating_overlap_index` | 0.50 | 0.043 |
| coupling | `max_synchrony_phase_offset` (trot, pace) | 0.125 | 0.083 (puppet trot) |
| coupling | `min_synchronous_overlap_index` (trot, pace) | 0.50 | 0.96 |
| coupling | `min_walk_limb_phase`, `max_walk_limb_phase` (folded limb phase, walk) | 0.125, 0.375 | 0.215 (puppet walk) |
| persistence | `min_template_coverage` | 0.70 | 0.79 (velociraptor at 10 ms) |
| persistence | `min_segment_coverage` (each third of the window not grossly off-template) | 0.50 | 0.71 |
| persistence | `max_off_template_run_fraction`, or at most `max_off_template_run_strides` strides when that is longer, never above `max_off_template_run_fraction_ceiling` | 0.10, 2 strides, 0.15 | 0.055 |
| persistence | `max_wrong_locked_fraction` | 0.20 | 0.17 (velociraptor at 10 ms) |

The provisional values come from `provisional_gait_criteria(profile)` and were calibrated on a development split only (synthetic footfall patterns at low jitter, one seed of physics puppets with designed footfalls, and replays of the trained compsognathus, velociraptor, T. rex, robot and Dibothrosuchus policies). Held-out synthetic, puppet, replay and adversarial corpora scored them; see the bake-off record. They are development criteria for report-only panels; a certification config must restate each one. No genuine trained quadruped policy exists yet, so quadruped bars rest on synthetic and puppet gaits.

There is no automatic gait-profile selection. An absence of flight does not distinguish a walk from grounded running. [Rubenson et al. (2004)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1691699/) measured running mechanics without an aerial phase in ostriches. `required_consecutive`, when present, must be 1 because the gate judges one fixed panel. The optional panel reward rail is explicit.

### Open decisions and known limitations

- **Step-to versus step-through.** Lead exchange at 0.15 rejects an exact step-to gait but a policy with foot-placement noise of about 0.01 `L` can satisfy it. The robust rule is the optional `min_step_length_over_leg` (median touchdown step length of each foot ahead of the contralateral foot, about 0.05 `L`): every genuine development gait is at +0.09 `L` or more except the genuine-labelled compsognathus 1001 policy, whose left foot lands a median 0.06–0.10 `L` behind the right. Declaring the rail therefore requires relabelling that policy, or accepting that the profile certifies timing alternation only.
- **Thin genuine margins.** Compsognathus 0921's touchdown-anchored phase offset is 0.076 against 0.09, mostly the effect of its left-heavy duty (0.49/0.60); the mid-stance anchored offset reported beside it is about 0.02. Velociraptor's longest off-template interval reaches 0.092 of the window in one episode (the next is 0.054). Measured headroom (bar 0.10 offset, 0.12 run fraction) produced no new errors; confirm on fresh panels before freezing.
- **Intermittent wrong-phase bouts** shorter than the persistence bars and totalling under the wrong-locked cap (about 20 % of the window) can still pass; a 40/60 timing stagger lies at the symmetry boundary.
- **Long strides.** With strides of 1.5 s a 9 s window holds six strides; the stride-aware persistence rule allows up to two strides of gross excursion but never more than 15 % of the window.
- **Labels are descriptive.** Grounded running is labelled from duty and trunk bouncing (Cavagna-style), not energetics.
- **Other body plans.** The code paths are generic over limb pairs, but the foot registry, templates and contralateral pairs are defined for two and four legs only. A new body plan adds its templates (e.g. a hexapod tripod) and contralateral pairs.

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

After locking a new `locomotion_gait/v2` gate, use its explicit `--config gait_gate.toml`, the same frozen environment/protocol options, the selected pair, and the declared 40 episodes starting at 3042. Do not extend a failing panel or choose a favorable subset.

## Evidence and interpretation

- `gait_traces/episode_0000.npz` etc. retain numeric physics-substep telemetry for every episode.
- `gait_panel.csv` retains episode metrics and checkpoint, normalization, task and protocol bindings; `metrics_json` includes complete per-foot and timing diagnostics.
- `gait_report.json` records plant/task identities, geometry/scale references, detector/panel settings, seed provenance, file hashes, qualification failures and confidence statistics. Readers verify bindings and replay metrics from the raw traces rather than trusting a stored pass flag.
- Changed checkpoint/normalization bytes during measurement refuse the result. Validation/load failures invalidate an earlier report with an incomplete, noncertifying record. Completed result bundles, including nested or symlinked output paths, are immutable; use a fresh directory outside them.

For `k` joint successes out of fixed `N`, the one-sided exact 95% Clopper–Pearson lower bound is `BetaQuantile(0.05; k, N-k+1)` (`0` when `k=0`). At `N=40` and a declared lower-bound floor of `0.80`, **37/40 passes and 36/40 fails**. The certificate concerns the specified reset/scenario distribution and episode event; it does not cover arbitrary terrain, perturbations or species.

CLI exit 0 means a report or preflight completed, **not that a certificate passed**. Read `report_only`, `certification_eligible`, `certified`, failures and statistics. Exit 2 indicates command usage errors; exit 3 indicates an evaluation refusal.

Mac CPU replay is fresh evidence on that runtime, not confirmation of an original GPU trajectory. Contact-sensitive runs can diverge across hardware/backends; preserve runtime and plant bindings, and do not certify unverified model/physics revisions. The observer does not modify frozen MJX code or species physics. The [historical audit's replay limitations](investigations/GAIT_AUDIT_2026_09.md) explain this distinction.
