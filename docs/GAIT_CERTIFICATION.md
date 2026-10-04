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
2. **Continuous phase.** Each limb's phase rises linearly between touchdowns. A stride longer than twice its *local cadence* is a pause and leaves the phase undefined. The local cadence is the larger of the median strides of up to three neighbouring strides of the same limb on either side (the stride itself excluded; at the first and last stride, where only one side exists, the stride joins that side's median). A stop between normal strides is therefore a pause, but a genuine change of cadence is not: a walk-to-run transition from a 0.95 s to a 0.42 s stride matches the cadence of its own side, whereas the whole-episode median used before called every walking stride a pause. Before the first and after the last touchdown the phase is extrapolated for one stride. Undefined phase is coverage loss, so a limb that stops cycling, a stand or a one-leg bout costs coverage in proportion to its duration.
3. **Pair statistics.** For every limb pair, the relative phase (lag of the second limb behind the first, in cycles) gives a time-weighted circular mean and mean resultant length (the pair's phase-locking index), a centred one-stride sliding estimate, and Hildebrand's per-stride footfall phases pooled in both directions. A duty-normalised stance-overlap index, which does not depend on touchdown timing, is 0 for perfect anti-phase occupancy and 1 for in-phase occupancy.
4. **Templates and off-gait time.** A profile is a list of limb pairs with target phase sets. Template coverage is the time during which every templated pair is locally locked within its target set widened by 0.05 cycles. The **off-gait fraction** is the time demonstrably spent in something other than the declared gait: a templated limb's phase is undefined (not cycling), or a pair is locally locked *inside a competing gait's template* (an alternating pair within the synchrony tolerance of in-phase: a hop, bound or pronk; a pair the profile wants in phase within the alternation tolerance of anti-phase; a walk's limb phase within the synchrony tolerance of pace or trot timing), or the pairs are in an uncoordinated bout (undefined, competing-locked or unlocked without a break) lasting at least two pooled strides. Shorter unlocked moments (a stumble, a double step) and locking at an intermediate phase (a skip-like drift) are not off-gait time; they cost template coverage only. The components (`undefined_phase_fraction`, `other_gait_locked_fraction`, `uncoordinated_bout_fraction`) are stored beside it, with the earlier diagnostics (`gross_off_template_fraction`, `wrong_locked_fraction`, the longest grossly off-template interval and per-third segment coverage).
5. **Local travel frame.** Fore-aft quantities are measured along the trunk's own travel heading, the root displacement over a centred one-stride window (one stride cancels the stride-periodic lateral sway), never along the declared task direction. Where that displacement is shorter than 0.05 `L` the heading of the whole window is used, and the declared direction only when the trunk did not travel at all. A heading drift or a rigid rotation of an episode therefore cannot change lead exchange, step or stride length or skid; before, an exact step-to gait walking 1.5–3° off the task axis gained lead exchanges it never made, and a genuine walk rotated 10–30° lost them. Only the progress rail (`mean_speed_mps`) is projected on the declared direction.
6. **Physical quantities.** Duty factor and load share per limb; contralateral load and duty ratios; on four legs the lighter girdle's (fore pair or hind pair) share of the foot impulse and the ratio of the girdles' mean duty factors; lead-limb exchange (fraction of strides in which the fore-aft order of a contralateral pair flips past ±0.01 `L` in both directions; step-through gaits flip twice per stride, step-to gaits never); per-foot **stride length** (footprint-to-footprint advance of the same foot along the local heading, median over complete strides); **swing-phase floor contact** per foot (`swing_ground_fraction`: the share of swing time, between two stances, with any floor force or the foot lower than 0.005 `L`, so a drag under the contact threshold or a skim counts; `swing_slip_fraction`: the distance slid on the floor during swing over the swing travel); the per-limb **skid fraction** (stance slip distance over trunk travel during the same stances, a ratio of sums so one bad stance cannot fail an episode); `body_support_fraction` (nonfoot ground impulse over total animal ground impulse); time with foot-on-foot force above 0.01 BW; flight fraction.
7. **Diagnostics (never verdict inputs).** Per-stride Hildebrand labels and their time-weighted distribution (walk, grounded or aerial run, hop, staggered hop, skip; lateral- or diagonal-sequence walk or amble, walking/running/flying trot and pace, pronk, bound, half-bound, canter, transverse or rotary gallop), Froude number, median footprint stride length over `L` and over Alexander's (1976) prediction, limb-phase mean and concentration, hind duty, phase coordination index (Plotnik et al. 2007), alternation index, biped hop-flight fraction, per-foot median step length, and the contralateral phase offset anchored at mid-stance as well as at touchdown.

Every reduction that reaches a stored metric is order-deterministic (`math.fsum`, sequential `cumsum`, scalar `math` trigonometry and exact elementwise arithmetic; no BLAS reductions and no SIMD transcendental ufuncs), and stored values are rounded to six decimals. The gate judges only stored values, so the reader's re-judge from `metrics_json` and its hash-checked replay from the raw traces reproduce the in-process verdict across machines. Missing, malformed, transposed, negative or nonfinite required telemetry, a non-positive body weight or leg length, or an unknown foot registry fails closed; unavailable JSON metrics are `null`.

`GaitProtocol` holds every metric-shaping setting above, including the template tolerances that define coverage and the competing templates (0.09 cycles around anti-phase, 0.125 around synchrony, limb-phase walk band [1/8, 3/8] ∪ [5/8, 7/8], local widening 0.05, gross widening 0.05), the pause rule (factor 2, three neighbouring strides a side), the travel-heading window (one stride, 0.05 `L` minimum travel), the swing floor band (0.005 `L`) and the uncoordinated-bout length (two strides). They are hashed into `measurement_protocol_sha256`; changing one requires a new planned hash and a fresh panel. The gate bars below are separate, so retuning a bar is a re-judge of stored evidence.

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
| participation | `min_relative_limb_load_share` (share × n limbs) | 0.36 | 0.439 (puppet *Dibothrosuchus* pace at 10 ms; the token-limb, rearing and wheelbarrow probes are at 0.31–0.32) |
| participation | `min_pair_load_ratio`, `min_pair_duty_ratio` | 0.70, 0.70 | 0.753, 0.756 (velociraptor at 20 ms; 0.801, 0.774 at 2 ms); the antalgic-limp probe is at 0.62 / 0.63 |
| participation | `min_girdle_load_share` (quadrupeds: lighter girdle's share of the foot impulse) | 0.18 | 0.223 (fore-heavy puppets carry 72–78 % on the forelimbs); rearing and wheelbarrow probes 0.16 |
| participation | `min_girdle_duty_ratio` (quadrupeds: lighter-duty girdle's mean duty over the other's) | 0.60 | 0.847 (puppet trot); rearing, wheelbarrow and token-limb probes 0.20–0.36 |
| participation | `min_complete_cycles_per_foot` | 3 | 10 |
| stepping | `min_valid_swing_fraction` | 0.75 | 0.79 (compsognathus 1001 at 20 ms) |
| stepping | `min_median_swing_clearance_over_leg` | 0.02 | 0.031 (compsognathus 1001 at 20 ms) |
| stepping | `min_stride_length_over_leg` (every foot's median footprint stride) | 0.20 | 0.266 (puppet *Dibothrosuchus* walk), 0.271 (compsognathus 1001 at 20 ms); micro-step shuffles 0.09–0.15, real hops 0.10–0.21 |
| stepping | `max_swing_ground_fraction` | 0.40 | 0.176 (real policies, compsognathus 1001 at 20 ms); 0.267 (synthetic sin² swings at 10 ms); toe-drag probes ≥ 0.74 |
| stepping | `max_swing_slip_fraction` | 0.10 | 0.004; toe-drag probes ≥ 0.43 |
| stepping | `min_lead_exchange_fraction` (local travel frame) | 0.15 | 0.318 (compsognathus 1001 at 20 ms) |
| stepping | `min_step_length_over_leg` (optional; see open decisions) | not declared | −0.126 (compsognathus 1001 at 20 ms) |
| support | `max_flight_fraction` | per profile, above | 0.33 (velociraptor) |
| support | `max_body_support_fraction` | 0.01 | 0 |
| support | `max_foot_foot_contact_fraction` | 0.02 | 0 |
| support | `max_skid_fraction` (local travel frame) | 0.35 | 0.293 (velociraptor touchdown skid at 20 ms); skating probes 0.40–0.49, the *Dibothrosuchus* scramble ≥ 0.79 |
| coupling | `min_phase_locking` (continuous and footfall estimators) | 0.60 | 0.70 (velociraptor at 10 ms) |
| coupling | `max_alternation_phase_offset` (circular, both estimators) | 0.09 | 0.076 (compsognathus 0921) |
| coupling | `max_alternating_overlap_index` | 0.50 | 0.043 |
| coupling | `max_synchrony_phase_offset` (trot, pace) | 0.125 | 0.083 (puppet trot) |
| coupling | `min_synchronous_overlap_index` (trot, pace) | 0.50 | 0.96 |
| coupling | `min_walk_limb_phase`, `max_walk_limb_phase` (folded limb phase, walk) | 0.125, 0.375 | 0.215 (puppet walk) |
| persistence | `min_template_coverage` | 0.70 | 0.722 (velociraptor at 20 ms; 0.802 at 2 ms) |
| persistence | `max_off_gait_fraction` (one budget: undefined phase, competing-gait locking, uncoordinated bouts) | 0.05 | 0.034 (velociraptor at 2 ms), 0.042 (at 10–20 ms); see below |
| persistence | `max_off_gait_strides`, `max_off_gait_fraction_ceiling` (quadruped walk only: one mistimed footfall) | 0.9 strides, 0.15 | 0.82 strides (j5-jittered 1.5 s-stride walk at 20 ms; 0.54 at 2 ms) |

The provisional values come from `provisional_gait_criteria(profile)` and were calibrated on a development split only (synthetic footfall patterns at low jitter, the same patterns at j3/j5 jitter with development seeds, one seed of physics puppets with designed footfalls, and replays of the trained compsognathus, velociraptor, T. rex, robot and Dibothrosuchus policies), with the verification attack traces of the first proposal as negatives. Held-out synthetic, puppet, replay and adversarial corpora scored them once; see the bake-off record.

**How the hardened bars were set (hardening round 1).** Each bar sits between the worst genuine development value (2, 10 and 20 ms sampling) and the nearest pathological probe, and each has a physical reading:

- *Stride length ≥ 0.20 `L`*: each step moves a foot at least a tenth of the leg. The 2026-09 plan's GQ-8(c) proposed 0.3 `L`, but genuine development gaits reach 0.266 `L` (the *Dibothrosuchus* walk puppet) and 0.271 `L` (compsognathus 1001 at 20 ms), so 0.3 would reject genuine walkers; micro-step shuffles sit at 0.09–0.15 `L`.
- *Swing floor contact ≤ 0.40 of swing time, swing slip ≤ 0.10 of swing travel*: a step is a swing that is clear of the floor for most of its duration. Real policies touch or skim the floor for at most 18 % of swing (a sin² clearance bell, which lingers near the ground at both ends, for 27 %); the toe drag under the contact threshold does so for 74–92 % and slides for 43–80 % of its travel.
- *Skid ≤ 0.35*: at most about a third of the trunk's stance travel may come from the feet sliding. Genuine runs reach 0.29 (velociraptor touchdown skid); skating gaits that glide at 40–49 % of trunk speed fail.
- *Pair load and duty ratios ≥ 0.70* (a Robinson symmetry index of about 35 %): the simulated genuine walkers are more asymmetric than animals (compsognathus 0921 is left-heavy, 0.80), and the velociraptor reaches 0.753 at 20 ms; an antalgic limp at 0.62–0.66 fails.
- *Girdle participation*: mammals carry 55–65 % of their weight on the forelimbs, while sauropods and crocodylians are often hind-heavy; the simulated plants are more extreme (the *Dibothrosuchus* puppets carry 72–78 % forward, the *Dibothrosuchus* statue 79 % on its hind feet). The lighter girdle must still carry 18 % of the foot impulse (each quadruped limb at least 9 % through the per-limb share floor of 0.36), and its limbs must be in stance at least 60 % as long as the other girdle's: fore and hind duty factors of a symmetrical gait are nearly equal (Hildebrand's gait formula uses one), genuine puppets differ by at most 15 %, while rearing, wheelbarrow and token-limb gaits sit at 0.20–0.36.
- *Off-gait budget 5 % of the window*: the genuine velociraptor spends up to 0.034 (0.042 at 10–20 ms) of its window off-gait. Two-foot hop bouts filling 14–20 % of a walk measure 0.060–0.16 and fail every episode; 10 % bouts measure 0.045–0.065 and fail four of six episodes (no panel can reach 37/40); the scramble probe fails five of six. The quadruped walk alone has a stride allowance (0.9 stride, never above 15 % of the window), because its competing templates sit only 0.05 cycles beyond its on-template band: one mistimed footfall of a long-stride walk at 5 % touchdown jitter locks there for up to 0.82 stride, while two missed steps measure about one stride. The alternating, trot and pace templates are 0.235 cycles from their competitors and need no allowance. They are development criteria for report-only panels; a certification config must restate each one. No genuine trained quadruped policy exists yet, so quadruped bars rest on synthetic and puppet gaits.

There is no automatic gait-profile selection. An absence of flight does not distinguish a walk from grounded running. [Rubenson et al. (2004)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1691699/) measured running mechanics without an aerial phase in ostriches. `required_consecutive`, when present, must be 1 because the gate judges one fixed panel. The optional panel reward rail is explicit.

### Open decisions and known limitations

- **Step-to versus step-through.** Lead exchange at 0.15 rejects an exact step-to gait, now at any heading, but a policy with foot-placement noise of about 0.01 `L` can satisfy it. The robust rule is the optional `min_step_length_over_leg` (median touchdown step length of each foot ahead of the contralateral foot along the local heading, about 0.05 `L`), off by default: every genuine development gait is at +0.054 `L` or more (velociraptor at 20 ms; compsognathus 0921 +0.15) except the genuine-labelled compsognathus 1001 policy, whose left foot lands a median 0.05–0.13 `L` behind the right. With the rail declared at 0.05 `L`, compsognathus 1001 qualifies 0 of 8 development episodes (and none at 10 or 20 ms or rotated). Declaring the rail therefore requires relabelling that policy, or accepting that the profile certifies timing alternation only.
- **Thin genuine margins.** Compsognathus 0921's touchdown-anchored phase offset is 0.076 against 0.09, mostly the effect of its left-heavy duty (0.49/0.60); the mid-stance anchored offset reported beside it is about 0.02. The puppet girdle share (0.223 against 0.18) and per-limb share (0.439 against 0.36) rest on one development seed of one quadruped plant. The velociraptor's 20 ms phase locking (0.599 against 0.60) fails one development episode, as before this round.
- **Residual off-gait bouts.** Per episode, 10 % two-foot hop bouts pass 2 of 6 and an uncoordinated scramble that never locks near a competing gait passes about half the time at 28 % of the window (always fails at 40 %); neither can reach a 37/40 panel. The genuine velociraptor spends 10–15 % of some episodes in a sustained skip-like drift with double steps, so a budget that also counted intermediate-phase locking would reject it. A single two-foot hop stride in a 0.75 s-stride walk stays inside the 5 % budget.
- **Long strides.** With strides of 1.5 s a 9 s window holds six strides; one mistimed footfall of a jittered walk costs most of a stride of off-gait time, which only the walk profile forgives (0.9 stride, never above 15 % of the window).
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
