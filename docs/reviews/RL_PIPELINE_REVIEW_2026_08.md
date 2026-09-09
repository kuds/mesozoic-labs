# RL Pipeline Review — August 2026

Scope: the repository at `main` (30b186a), open PR #493 (head 259a147) and open
draft PR #494 (head c6d85cf), and the Drive artifacts of the most recent
training run — seed-43 trex PPO stage 1, run `20260806_133233` (gate FAIL,
`mean_unsupported_duty` 0.1958 vs ceiling 0.02). Focus: RL-correctness bugs,
measurement validity, PR feedback, and cleanup.

Method note: every finding below was independently re-derived from primary
sources (code at the cited commit, the raw Drive artifacts, the plant XML) by a
second adversarial pass before inclusion; arithmetic claims were recomputed.
One candidate finding was refuted in verification and is recorded as such
(§7.5). Line numbers refer to the tree named in each finding (`main`, `#493`,
or `#494`).

---

## 1. Headline: the stance gate measures a stroboscope, not airborne time

**Severity: major. Files: `environments/shared/base_env.py:826`,
`environments/trex/envs/trex_env.py:366`, `environments/shared/stance_diagnostics.py:74`,
`environments/shared/scripts/stance_gate_report.py:663`,
`environments/shared/scripts/stance_duty_validation.py:126`. Pre-existing on
`main` — not introduced by either PR — but it is the ground the whole bounce
investigation stands on.**

Physics runs at dt = 0.002 with `frame_skip = 5`, so control is 100 Hz while
contacts evolve at 500 Hz. `BaseDinoEnv.step` runs the 5-substep loop and only
then reads observations, reward info, and termination (`base_env.py:826-843`),
so every contact-derived quantity — the bilateral-support reward, the
support-conditioned alive gate, foot-load balance, the 2 foot-contact obs dims,
floor-contact termination, and (via `info["r/l_foot_contact"]` →
`derive_stance_info`) the gate's per-step duty flags — sees only the **last of
5 physics substeps**. The other 8 ms of every 10 ms control period is invisible
to the metric, the reward, and the observation alike.

The seed-43 logs are the proof this matters, not just a theoretical concern:

- In **all 40** gate-panel episodes, `bilateral_support_duty` is **exactly**
  0.8000 and `unsupported + single` is **exactly** 0.2000. Every duty value is
  an integer multiple of 1/800 (settle 200 of 1000 steps; e.g. 0.19875 =
  159/800, 0.04125 = 33/800).
- A seed-invariant non-bilateral total of exactly 160/800 with a seed-*varying*
  split between unsupported and single support is the fingerprint of a
  **period-5-control-step limit cycle** — a 20 Hz hop phase-locked to the
  control clock — with the 0.1 N classification threshold flickering the
  grazing sample between "single" and "unsupported". (800 mod 5 = 0, so the
  count is exactly 160 regardless of phase.) Commit 8bffb98 already documents
  the same subharmonics for seed 42 (duty 1/6 at 16.7 Hz, 1/5 at 20.0 Hz).
- Internal cross-check closing to ~1e-11: `reward_alive` 897.27 and
  `reward_bilateral_support` 476.73 jointly imply sampled support quality
  0.79454 — quality ≈ 0.99 on 4 of 5 samples, ≈ 0 on the 5th.
- The statue measured through the same pipeline reads bilateral 0.998, which
  rules out a settle-window accounting bug.
- Physical scale: height quality 0.99536 puts RMS pelvis-height error at
  ~4 mm. This is a **millimetric micro-bounce**, not hopping.

Consequences:

1. **The gated 0.1958 is a phase-locked sample statistic, not physical airborne
   time.** Any unloaded window between 0 and 20 ms per 50 ms cycle that covers
   exactly one sampling instant yields exactly 0.2. "The policy spends 19.6% of
   its time unsupported" is not established; "one unloaded sampling instant per
   50 ms" is.
2. **The false-PASS direction is open.** Reward, observation, and gate all
   share the same stroboscope, and this run proves policies phase-lock to the
   sampling clock. A policy that sharpens its unloading window to fall
   *between* the 10 ms samples reads duty ~0.000 and passes the 0.02 gate while
   unloading every cycle. This run's FAIL is the conservative direction; the
   next run's PASS may not be.
3. The gate's notion of "bearing load" is a 0.1 N threshold
   (`stance_diagnostics.py:74`) on a multi-tonne animal — 420x looser than the
   reward's 42 N (`foot_load_balance_min_support_force`) and 350 N
   (saturation) constants, which shape only the reward and never the gate.
4. Tail/head/torso floor strikes resolved within a control step cannot
   terminate the episode (`_check_floor_contact` sees only last-substep
   `ncon`).
5. `stance_duty_validation.py` cannot catch any of this: it reads its
   kinematic ground truth after `env.step` at the same aliased instants
   (`:126-127`), and its forced-hop driver runs at period 40 steps = 2.5 Hz
   (`:181-182`) — the frame-locked ~20 Hz regime the failing runs actually
   occupy is exactly the regime the validation never sweeps.
6. The UCB is *correct* but adds nothing here: the t-bound arithmetic
   reproduces the logged 0.1981 exactly (§6.1), but a Student-t bound on
   i.i.d. episode duties is a category mismatch for a deterministic limit
   cycle plus threshold flicker — 19 of 40 episodes sit at exactly 0.2000, and
   as s → 0 the bound collapses onto the mean. Worth a sentence in
   `stance_gate.py`'s docstring.
7. The reported "tremor at ~24.0 Hz" is best read as **20 Hz + broadband**:
   the `effective_freq_hz` estimator saturates at 33.33 Hz for white noise
   (recomputed from the formula at `stance_gate_report.py:522-524`), so any
   broadband component drags a 20 Hz tone upward, and the contact evidence
   pins the mechanical cycle at exactly 100/5 = 20 Hz. The saturation caveat
   lives only in a code comment; the rendered text prints the number
   unqualified. Suggested check: FFT one stored post-settle action trace — if
   the spectrum peaks at 20.0 Hz, the investigation's "22.6 Hz tremor"
   phrasing should become "20 Hz control-clock-locked cycle".

**Fix direction:** aggregate contact across substeps inside the `frame_skip`
loop (min-force-per-foot, or supported-substep fraction) for the info keys
feeding `derive_stance_info`, the support rewards, and floor-contact
termination — and add a 5-step-period driver to `stance_duty_validation.py` to
bound the error. This belongs **before or alongside** PR #493's plant revision:
re-running multi-seed experiments on a gate metric with unquantified bias
wastes exactly the runs the plan says now matter, and the action-filter and
impulse experiments will otherwise be read through the same aliased lens.

---

## 2. Reward-design findings (trex stage 1)

### 2.1 The reward is aligned with the gate but ~10x too shallow to enforce it (major)

`environments/trex/envs/trex_env.py:522-548`, `environments/shared/reward_functions.py:399-409`.

At the run's weights, the per-sample differential between a bilateral and an
unsupported sample is ≈ 1.65–1.7 (bilateral_support 0.6 + conditioned half of
alive 0.5 + foot-load-balance swing ~0.55). Moving the trained policy from its
0.196 duty to the 0.02 ceiling is worth ~300 of a 2470 return (~12%), while
**hop-invariant** terms — height 597.22 (quality 0.995: the 20 Hz cycle needs
<1 mm of vertical excursion, and the sampled pelvis height stays inside the
0.06 tolerance), head_clearance 350.00 (exactly saturated), leg_home_pose
371.33, neck 141.33, heading 50.69, and the unconditional half of alive 500 —
sum to 2010.6/ep, **~81% of return**. Nothing pays the hop more than standing
(the statue at 3271.8 is the documented optimum), so this is not a reward that
encourages hopping; it is a reward whose gradient toward gate compliance is
too shallow for PPO to follow, and three consecutive runs have now plateaued
in the gate-FAIL band (duty 0.319 → 0.1668 → 0.1958) while clearing the 1950
rail each time. The vertical hop is additionally invisible to every
velocity-shaped term (speed/drift/lateral/idle all read `qvel[0:2]` only) and
to the height term through the stroboscope.

If stage 1 is to converge into the gate's feasible region: steepen the support
terms (e.g. an episode-level duty penalty), gate more than half the alive
bonus (`support_conditioned_alive_fraction`), and/or fix the measurement
(§1) first so reward and gate price the same physical quantity.

### 2.2 Backward drift is near-free, and the pitch shaping points backward (minor)

`configs/trex/stage1_balance.toml:6-7`, `environments/shared/reward_functions.py:66-98, 269-283, 519-535`.

All 70 eval episodes drift backward (fwd vel −0.02..−0.119 m/s). Nothing
prices it: `forward_vel_weight = 0`, `backward_vel_penalty_weight = 0`
("redundant with drift_penalty"), the speed penalty has a 0.1 m/s dead band
(the logged `reward_speed` −25.00 comes from within-hop speed spikes, not the
drift), and the quadratic drift term charged −15.89/ep — 0.64% of return —
for ~0.65 m of displacement. Even if enabled, the backward term normalizes by
`forward_vel_max = 8.0`, so it is useless at balance speeds. Meanwhile the
nosedive penalty and terminator are one-sided (forward pitch only), making
"hedge backward" systematically cheaper than "hedge forward" — a plausible
driver of the uniformly negative drift. If stage 1 certifies standing
*still*, one of these needs to bind at ~0.5 m displacement.

### 2.3 The toe actuators that generate the gated signal are anchored by nothing (minor, known issue #491 with a gate-relevant twist)

`environments/trex/envs/trex_env.py:339-348`.

`leg_home_joint_names` covers hip/knee/ankle only. The six toe joints — which
carry their own touch sensors and directly shape the foot-floor contact the
gate measures — are anchored only by the energy term at ~0.0036/step per
saturated toe. The run confirms the free-rider: all six toe DCs at exactly
±1.000 with AC exactly 0 (also `neck_yaw`, `tail_1_yaw`, `tail_3_pitch`). The
TOML already documents the shaping gap ("the list, not the weight, is the
problem"); the twist worth recording is that the un-anchored joints are
precisely the contact-generating ones — a policy free to command digit
extremes can tune when each foot's summed touch reading crosses the 0.1 N gate
threshold. This is one more reason the passive-toes plant revision (#493's
recommendation 1) is sound.

---

## 3. Training-pipeline bugs (shared code, all species)

### 3.1 `warmup_ent_coef` is silently dead for PPO (major)

`environments/shared/curriculum/schedules.py:67-72`,
`environments/shared/curriculum/advancement.py:518-529`,
`environments/shared/train_base.py:758-780`.

`EntCoefDecayCallback._on_step` unconditionally reassigns `model.ent_coef`
every step from a value captured before warm-up starts.
`StageWarmupCallback._on_training_start` sets the configured boost (0.02) once
— and is clobbered on the next step, long before PPO's `train()` ever reads
it; the restore at warm-up end is clobbered identically. Callback order does
not matter. **Every** species' stage-2/3 PPO TOML sets both keys, so every PPO
curriculum advancement runs its transition with entropy decaying from the base
0.005 while the log claims "warm-up active … ent_coef=0.020". Warmup
`clip_range` is honored; SAC is unaffected. Fix: have the decay callback skip
while warm-up is active, or restart the decay from the warm-up value.

### 3.2 `obs_rms_decay_on_resume` and `ramp_attr` are dropped by both JAX library entry points (major)

`environments/shared/jax_curriculum.py:247-268`, `environments/shared/jax_training.py:289-318`.

All eight stage-2/3 TOMLs set `obs_rms_decay_on_resume = 0.01`; the helper
exists (`jax_normalization.py:75-105`) and the notebook honors it, but
`_JAX_KEY_MAP` and the CLI both drop it — `decay_running_stats` has zero
production callers. A JAX stage-2/3 resume keeps the prior stage's
multi-million sample count exactly when the obs distribution shifts — the
pathology the key exists to prevent — with no warning, while 8 TOML comments
assert the decay happens. `ramp_attr` is inert on the same two paths (harmless
today only because every TOML sets the default value). A prior review flagged
this as QoL; it is a real behavioral divergence between the notebook and the
library. Fix both, and give `[jax]` the same fail-closed unknown-key
validation `[curriculum]` already has (§8.4).

### 3.3 `collected_results.csv` `model_hash` is the stage-3 hash on every row (minor)

`environments/shared/reporting/csv_output.py:224-226`, `environments/shared/reporting/bundles.py:242-244`.

The per-stage rows are stamped with top-level provenance `model_hash`, which is
defined as `selected_checkpoints["3"]`'s hash. For the seed-43 stage-1 run the
column is empty even though the stage-1 hash sits in
`selected_checkpoints["1"]`; for a complete 3-stage bundle, stages 1–2 get
stamped with the stage-3 hash — a false mismatch for anyone verifying rows
against zips. The per-row fix already exists in the same dict:
`selected_checkpoints[str(stage)]["model_hash"]`. (The other "empty" columns in
the digest are intended sparsity, verified: quality_score/rank are sweep-only;
the `*_threshold` columns are structurally empty for stance-gated stages
because the gate schema forbids those keys; top-level
`selected_model_path`/`model_hash` null is by design for partial bundles.)

### 3.4 Stage-1 artifacts print a structurally impossible "Success rate 0%" (minor)

`environments/shared/train_base.py:1046-1051`, `environments/shared/eval_diagnostics.py:35-44, 971`.

Trex stage-1 success keys are prey-strike events a balance stage can never
emit, so the metric is a constant 0. The `success_metric_applicable()` guard
exists and prints "N/A (not an active Stage N gate)" — but only the
in-training callback uses it. `stage_summary.txt`, `metrics.json`,
`collected_results.csv`, and both evaluation CSVs all resurrect the misleading
0%. Wire the same guard into the post-training paths.

### 3.5 Hardcoded success keys omit `snap_success` (minor)

`environments/shared/curriculum/advancement.py:248, 388`.

Both the supplementary eval and the standalone fallback detect success with a
literal `("bite_success", "strike_success", "food_reached")` tuple, ignoring
`SpeciesConfig.success_keys`. Dibothrosuchus uses `snap_success` with a real
stage-3 `min_success_rate = 0.5` gate: on the fallback paths its stage can
never advance and the failure reads as a training problem. (The primary npz
path is species-agnostic via `info["is_success"]` and is unaffected.)

### 3.6 GCS upload omits `robust_best_model.zip` (minor)

`environments/shared/config.py:507-515`.

The upload list is hardcoded to `best_model`/`stage{N}_final` (+ vecnorms).
But `robust_best_model` is what the handoff prefers (`train_base.py:548`),
what the stance report evaluates, and what `training_summary.txt` names as
"Best model". A run restored from GCS alone silently substitutes the
risk-unadjusted `best_model` — exactly in the case where the two differ. (On
the seed-43 run they were byte-identical, which masks the gap; see §3.8.)

### 3.7 Same-stage resume re-applies stage-entry shaping (minor)

`environments/shared/train_base.py:762-780`, `environments/shared/curriculum/advancement.py:599-607`.

`if stage > 1 and load_path:` cannot distinguish "entering stage 2 from stage
1" from "continuing stage 2 from its own checkpoint" (which `--load`
explicitly supports). A stage-2 resume resets `forward_vel_weight` to 0.1 on a
converged policy, pins warm-up `clip_range` for 100k steps, and (because
`learn()` never passes `reset_num_timesteps=False`) restarts every schedule
from zero. The first ~500k resumed steps optimize a different reward than the
checkpoint was selected on.

### 3.8 `robust_best_model` degeneracy — decide and label it (question)

`environments/shared/curriculum/checkpoints.py:72-82`.

Verified: it is a real online argmax of (mean − std), saving the exact weights
just evaluated — not a copy. On a converged low-variance stage (std ~0.3–1% of
mean) it *predictably* degenerates to byte-identical `best_model`, which is
what this run produced (same sha256 for both zips and both vecnorm pkls). Not
a bug — but consider logging/manifesting "robust_best == best_model (same
argmax)" when hashes match, so the degeneracy is visible rather than inferred,
and the duplicate ~4 MB per stage is a conscious cost.

### 3.9 Evaluation cadence was never re-derived for 40-episode panels on a 10M budget (cleanup)

`environments/shared/train_base.py:152-166, 443, 694-704`.

Panels were raised to the gate's 40 episodes, but `eval_freq` stayed at 50k on
a 1-env DummyVecEnv: ~200 evals x 40 episodes x 1000 steps ≈ **8M serial eval
env-steps** per 10M-step stage (plus 10 supplementary episodes per eval and
the post-training 50+30 episode evals). For a full-horizon policy, evaluation
plausibly dominates the 8h wall clock. Halving cadence or vectorizing the eval
env returns hours per run without touching the gate's power, which is
specified per-panel.

### 3.10 The "best evaluation" reduction is triplicated; the CLI curriculum path has no bundle (cleanup)

`environments/shared/train_base.py:927-950, 1412-1423`, `environments/shared/reporting/stage_artifacts.py:69-88`.

Three copies parse `evaluations.npz` with drifting field sets, so
`metrics.json`, `curriculum_results.csv`, and `collected_results.csv` describe
"the best evaluation" differently, and a future change to selection must land
in three places. Separately, `generate_stage_artifacts`/`save_result_bundle`
are invoked only from the notebook and sweep scripts — a CLI
`train_curriculum` run produces no provenance, no stance report, no evidence
CSVs, just its own non-canonical `curriculum_results.csv`. Collapse onto
`build_stage_results_from_eval_data` + the bundle writer.

---

## 4. PR #493 (stance probes + seed-43 record) — mergeable; fix the ordering nit first

No critical or major defect found in the main path. The two handoff bugs the
PR confesses to are genuinely fixed and pinned (ablation variants built from
reset with the complement-and-sum identity asserted; impulse read-out keyed
off per-direction envelopes with the pooled margin demoted). The
probe/verdict separation (`_PROBE_MARKERS`, distinct stems, banner before the
verdict line, `stance_panel_selected.csv` never written by probes) is
disciplined and test-backed. Spot-checked doc claims recompute: the statue
controls reproduce 3271.8 (3271.0 hold / 3274.4 ablation), the 0.50/0.00
recovery envelope follows from the printed margins, and the toe/tail/hip-roll
degree claims match `trex.xml`.

**The toe-splay evidence and the 21→15 recommendation survive PR #494's
mapping correction** — verified from the XML: home ctrl equals the ctrlrange
midpoint for the toes (and 19 of 21 actuators), both mappings share the ±1
endpoints, and the ablation/hold/impulse step counts are physical rollouts
independent of the reporting mapping. Its weakest input is the unverifiable
stage-2 toe-AC 0.129 number, not the mapping.

Findings, all minor/cleanup:

1. **Probe orchestration sits inside the gate report's `try`, before the
   verdict is recorded** (`stage_artifacts.py:265-305`). ~296 probe episodes
   now run between "gate report written" and "verdict recorded / summary /
   replays written" (the window existed before at ~40 episodes; the PR widens
   it ~7x). A runtime death there (Colab timeout, OOM; `KeyboardInterrupt`
   sails through `except Exception`) loses a finished 8-hour run its summary
   and recorded verdict. Worse, an exception at the *call sites* (e.g.
   `settle_steps=int(report["settle_steps"])` at `:300`, or any future
   signature drift — the helpers' internal catch-alls can't catch caller-side
   TypeError) is swallowed by the outer handler, which returns None — and
   `stance_report=None` records **gate FAIL for a stage whose on-disk report
   says PASS** (`gates.py:128-134`): the exact report-vs-recorded disagreement
   this module exists to prevent, inverted. Move the four probe calls after
   `_apply_stage_gate`/`write_stage_summary` (or at least out of the outer
   `try`).
2. **`--hold-from-report` does not skip the panel its help text promises to
   skip** (`stance_gate_report.py:2122-2260`): `main()` unconditionally rolls
   the full 40-episode base panel first, then discards its per-actuator DC
   when the flag is set (settle/horizon are config-derivable). Also parses the
   JSON twice.
3. **The impulse read-out's final branch prints "Neither side recovers from
   anything" when both sides recover everything** (`:1562-1569`) — reachable
   for a more passively stable species or small `stance_probe_impulse_speeds`
   — and `test_a_statue_that_recovers_too_does_not_count_as_active_control`
   cements the wrong prose (it only asserts the ACTIVE-CORRECTION string is
   absent). Add a branch keyed on `statue_best >= max(speeds)`: "the sweep
   never exceeded the passive envelope; raise the impulse speeds."
4. **The AST wiring test would stay green through the exact bug class this PR
   hit twice** (`test_stance_gate_report.py:2250-2267`): `ast.walk` accepts
   unreachable calls and checks names only (no args), the four-name tuple
   means a fifth probe goes unwired invisibly (contra the CHANGELOG claim),
   and nothing tests that `_roll_episodes` actually applies the impulse at
   `steps == impulse.step` — a regression there silently scores the
   undisturbed policy and feeds finding 3's branch. The existing
   FakeEnv+recorder harness makes a step-timing test cheap. Also: nothing
   asserts the *outer* edge (`generate_stage_artifacts` →
   `_write_stance_gate_report`) at all.
5. **The four `stance_probe_*` keys validate on any stage/species, including
   ones where the probes can never run — silently** (`gate_schema.py:157-169,
   250`). Setting `stance_probe_impulse_speeds` on trex stage 2 (the obvious
   next use) validates cleanly, produces nothing, and warns nowhere — the
   exact trap the same function rejects for threshold keys. Warn or fail on
   probe keys under a non-stance gate kind.
6. **One leftover wrong-mapping sentence** survives both PRs:
   `docs/KNOWN_ISSUES.md:72` "Actions map linearly onto each actuator's own
   ctrlrange" — false for trex; #494 fixes the identical phrasing in the
   investigation doc but not here. One-phrase fix, best folded into #494.
7. **Probe cost has no off-switch** (question): ~296 extra episodes and ~28
   checkpoint reloads per run *and per sweep trial*; the only existing switch
   (`stance_report_episodes = 0`) kills the certification report too. If
   sweeps should not pay the battery fifty times, add a probe-only suppress
   flag.

---

## 5. PR #494 (action-mapping fix) — correct, numerically verified; finish four items

The central claim was verified by execution, not just reading: with the pinned
mujoco 3.10.0, the v2 report's probed mapping reproduces each species'
`_scale_action` **exactly** (max |env−report| ctrl error 0.0 over dense
grids), action 0 maps to the named home control for every actuator of every
species, ±1 hit the ctrlrange edges, the trex ankles carry exactly the +5.5°
key_ctrl-vs-key_qpos preload — and #493's retracted "4 of 21 off-home" numbers
reproduce exactly as midpoint-minus-key_qpos artifacts (+5.0/+5.0/+5.5/+5.5).
The per-sample applied-ctrl accumulation correctly implements E[f(a)]; the
velociraptor gear-50 claw exclusion is conservative and passes the other
species; no code outside the report module reads the schema string or degree
fields; the full test suite passes on the PR head. Note the mapping dispute
could only ever have affected `neck_pitch`/`head_pitch` — home ctrl equals the
ctrlrange midpoint for 19 of 21 trex actuators.

To finish:

1. **The PR body's validation section is wrong and should be corrected**: the
   "two pre-existing stale plant-manifest failures already present on the base
   branch" are a **MuJoCo-version environment artifact**, not a base-branch
   defect. Under the pinned mujoco 3.10.0 the manifest test subset passes
   23/23 and `build_plant_manifest()` matches the committed manifest
   entry-for-entry for all four species; the failures reproduce only under
   3.11.x, where the physics fingerprints legitimately change. Re-validate
   under 3.10.0 and reword. Related hardening: `current_plant_identity`
   (`plant_contract/manifest.py:263-276`) recomputes fingerprints under
   whatever MuJoCo is installed and reports version skew as "manifest is
   stale" — compare `mujoco.__version__` against the manifest's
   `generated_with.mujoco` first, as the CLI check already does. It has now
   misled one PR's validation narrative.
2. **The constant-hold probe still holds f(E[a])** — the PR's own
   E[f(a)] ≠ f(E[a]) thesis, unapplied one function below the fix
   (`stance_gate_report.py:928`): the held constant is the normalized mean
   `dc`, so for asymmetric-span actuators the held target is not the mean
   commanded target (trex: only neck/head pitch, bias ~0.01° for seed 43 —
   latent, but the docstring's "asked to stand where it was already standing"
   is false by the PR's own reasoning). Derive the hold from the recorded
   `ctrl_mean` inverted through the probed mapping, or say so in the
   docstring.
3. **Saturated geared motors vanish from the degree-ordered table**
   (`:1227-1228`): motors correctly carry no `dc_deg` in v2, so they sort as
   0.0° and fall out of the 12-row table — a velociraptor claw pinned at DC
   +1.0 disappears while the header promises the ordering no longer hides
   moved targets. Also the `zero_offset_deg` contract check can never fire for
   motors. Secondary: `_actuator_pose_mapping` indexes `jnt_qposadr` with
   `actuator_trnid` before any transmission-type check (`:452-453`); a future
   tendon/site actuator turns the whole mapping into `[]` via the
   function-wide except (all-or-nothing, latent today — all four models are
   joint-transmission-only).
4. **The "action = 0 is the home keyframe" conflation survives** in
   `eval_diagnostics.py:108-109, 282-284, 725` and — in the same file the PR
   rewrites — `stance_gate_report.py:914-916, 897-898`, including one
   **rendered artifact line** (`:1918`, the ablation header "commanded 0,
   which IS the home keyframe"). Sweep them in this PR while the distinction
   is fresh.

Smaller: ~6–9% of the diff is pure formatter line-rejoining on lines #493
introduced (four `stage_artifacts.py` hunks and several in the report/tests) —
needless conflict surface for a stacked PR; the fixture-only fallbacks at
`:557` (missing `angular_position_ctrl` defaults to *True*) and `:484-493`
(legacy `home` fallback resolves action-zero to key_qpos, mirroring neither
mapping) contradict the fail-safe design and should be failed closed or
deleted; and the v1→v2 `dc_deg` reference change (home qpos → action-zero
control; ankles move 5.5°) is properly schema-bumped but worth a
machine-readable definition line in the JSON for cross-version tooling.

---

## 6. Drive-log forensics (run 20260806_133233)

### 6.1 Positive result: the run is exactly reproducible from its own evidence

Recomputed from the raw Drive artifacts against the code at the recorded
commit: mean unsupported duty 0.19578125 and UCB 0.19811740557503926 reproduce
**to the last digit** from the shipped 40-episode evidence with the committed
t(39) = 1.6849 / ddof = 1 bound (implied multiplier backs out to
1.6849000000000096); every reward-decomposition term recomputes from the
config weights (the "suspicious" −25.00 is −25.0039 — a rounding coincidence
driven by hop-speed spikes; 350.00 is true saturation; component sum equals
the stored `reward_mean` exactly). Provenance, hashing, and the gate
arithmetic can be trusted for this run.

### 6.2 Items surfaced by the logs

1. **Shipped probe cutoffs truncate the sweep exactly where it becomes
   informative** (`configs/trex/stage1_balance.toml:397`): [5, 10, 20] Hz,
   and this run's curve jumps 128.6 → 912.8 steps between the last two
   points, with the saturation point above 20 Hz unmeasured — while the code's
   own docstrings cite 35 Hz measurements the default can no longer make. Two
   more cutoffs (30, 35) cost 20 episodes.
2. **Probe reports embed gate-verdict boilerplate that is false for probes**
   (`stance_gate_report.py:601`, `stance_gate.py:378-395`): every filtered
   report in the Drive JSON leads with "n_episodes 10 < 40 …" and "the bound
   would certify a smaller sample than the panel reports", and scores a
   deliberately handicapped policy against the unfiltered 1950 rail — noise
   ahead of the informative discriminators, in the machine-readable surface,
   for a measurement three docstrings say certifies nothing. Suppress gate
   evaluation for probe reports (passed/failures = null).
3. **`evaluation_final.csv`'s `evaluation_seed` column is a protocol-stream
   label, not a per-episode seed** (`csv_output.py:404-415`): episodes vary
   via successive draws from one RNG stream seeded once at 3043; the
   identically named `panel_seed` in `stance_panel_selected.csv` *is* a real
   per-episode seed (3042+i). Also "publication seed" means 3042 in the CLI
   help and 3043 in provenance's `seed_roles`. Rename or document.
4. **`verification_status` can never leave "unverified"** (question,
   `result_bundle/provenance.py:255-256`): no code path in the repository sets
   `verified` or `model_revision_status = "current"`, yet
   `claims_certified` and the website's stage-video gating key off exactly
   those values. Either the promotion step is a manual edit outside any
   committed tooling (worth documenting), or every bundle is permanently
   uncertifiable by the pipeline that creates it.
5. **The investigation doc's seed-43 toe table doesn't match the run's shipped
   gate report** (question, for the #493 doc): the doc's −0.998/+0.999/…
   column cannot come from the selected-checkpoint report, which pins all six
   toes (plus neck_yaw, tail_1_yaw, tail_3_pitch — 9 of 21 actuators) at
   exactly ±1.000 with AC exactly 0.000 across all 32,000 post-settle samples.
   The table is titled "final per-actuator pose", so the likely source is the
   *final* checkpoint — a different policy from the selected one the panel and
   gate measured. Both datasets support the splay conclusion, but the doc
   mixes selected-checkpoint action stats and final-checkpoint values without
   labeling which is which; state the source.
6. Small reading hazard: in the per-step stance CSVs
   (`trex_ppo_stage1_*_stance.csv`), the `*_duty` columns are **instantaneous
   binary flags** per control step (`stance_diagnostics.py:116`), not
   accumulated duties — the name invites the wrong reading next to the panel
   CSV where the same names are per-episode averages.

---

## 7. Environment / contract findings

1. **Energy/smoothness/jerk are computed on the raw pre-clip action while ctrl
   uses the clipped action** (`base_env.py:821-840`). Safe for SB3 and the
   probe tooling today (both clip), but any direct caller passing |a| > 1 gets
   physics from the clipped action and reward charged for the excess. Clip
   once at the top of `step()` and use it for both.
2. **The plant contract's `observation_segments` describes the trex
   foot-contact segment as 8 components; the observation carries 2**
   (`plant_contract/policy_layer.py:299-324`). Summed widths give 67 vs the
   real obs_dim 61. No programmatic consumer today, but it is the document
   downstream tooling is invited to reconstruct the layout from; anyone
   locating dims from it mis-indexes by 6. Fix on an intentional contract
   revision (it is fingerprinted).
3. Stale module docstring: the 2 foot-contact obs dims are documented as "2
   plantar-pad touch sensors" but are pad+digit sums — a ~22% scale error for
   anyone calibrating from the doc (`trex_env.py:15` vs `:366-371`).
4. `render_stance_gate_report` hardcodes trex-stage-1 statue calibration
   ("statue 0.998 / 0.002, not gated") into every species' report
   (`stance_gate_report.py:1192-1193`).
5. **Refuted during verification** (recorded so it isn't re-filed): a claimed
   SB3-vs-MJX termination divergence via the thigh height thresholds is
   geometrically impossible — the thigh body origin is a rigid ≤0.157 m offset
   from the pelvis, so `mjx_config.py`'s thigh entries (0.20 m) are dead
   configuration shadowed by the shared 0.70 healthy-z floor. Residue worth
   keeping: those two entries could be deleted, and no test pins
   termination-envelope parity for the torso/tail contact-vs-height
   approximation in strongly pitched poses.

---

## 8. Configs, CI, and repo hygiene

1. **Statue-derived gate constants have no freshness check**
   (`stage1_balance.toml:302-314, 391-396`, `early_stopping.py:313-320`):
   `min_avg_reward` 1950 and `collapse_peak_floor_reference` 3271.8 are
   documented as derived, the TOML warns forgetting the re-measurement "is
   silent", the failure has already bitten twice (floors that never armed; the
   4250.4 episode) — and the notebook already writes a freshly measured
   `zero_action_baseline.json` into every run dir that nothing cross-checks.
   One runtime comparison (measured statue vs configured reference, warn on
   drift) closes it with data the run already produces. The same applies to
   `early_stopping.py`'s docstring claim that the reference "re-anchors
   automatically" — it does not.
2. **The notebook's zero-action-baseline verdict is gate-kind-blind**
   (`sb3_training.ipynb`, baseline cell): for stance-gated trex stage 1 it
   compares the statue against `min_avg_reward` — which the statue clears *by
   design* (it is a collapse rail, not the gate) — so every trex run opens
   with a false "FAILS — a statue clears this gate" that is persisted into the
   saved baseline JSON. Dispatch on `gate_kind`.
3. **Stale trex configs**: the `[sac]` block's `buffer_size = 300_000 #
   Sufficient for 6M-step Stage 1` predates the 10M budget (and its gamma
   comment reasons against a PPO value of 0.998 this config abandoned for
   0.98); `sweep_ppo.json` stage 1 still runs 6M — under which the 7M
   `ent_coef` decay never completes, ending at ~0.0007 ≈ the abandoned 0.001
   floor the config blames for a documented failure run — and its
   `env_nosedive_weight` range [2.0, 4.0] excludes the production 1.5, so a
   sweep cannot even sample today's operating point; `sweep_sac.json` likewise
   6M.
4. **`[jax]` keys have no fail-closed validation** (contrast
   `gate_schema.py:247-255`): misspelled or unsupported keys are silently
   ignored — the mechanism behind §3.2, and a trap class the configs' own
   history documents ("species default was leaking into stage 1 via key
   mismatch bug").
5. **CI**: lint tools are unpinned while pre-commit pins ruff 0.4.4/mypy
   1.15.0 — CI can turn red on untouched code, and nothing in CI runs
   pre-commit at all (so `check-added-large-files --maxkb=500` is local-only;
   a 21 MB GIF is already in-tree). Path filters omit `conftest.py` (which
   bootstraps `sys.path` for the whole suite), `Dockerfile`, `scripts/`,
   `.pre-commit-config.yaml`, and `docs/` subdirectories — a PR touching only
   `conftest.py` gets no Python CI.
6. **~25 MB of GIFs committed twice byte-for-byte** (`results/velociraptor/*`
   == `website/static/img/*`, three md5-verified pairs): every clone carries
   the duplication forever and the copies can silently diverge. Make
   `results/` canonical and copy at site build.
7. **CHANGELOG.md has three concurrent "[Unreleased]" sections** (v0.3.6,
   v0.3.2, v0.3.0) above a dated 0.2.0, at 103 KB with no cutting policy —
   "Unreleased" no longer discriminates. Date and close the shipped sections;
   add one line to CONTRIBUTING.md on when sections close.
8. **Self-contradictory provenance comments in the files PRs mine for
   constants**: `posture_weight = 1.5 # Increased from 1.5`;
   `nosedive_weight = 1.5 # Increased from 3.0` (a decrease);
   stage-2's decay anchor justified by "matching stage 1's 3M-of-6M" (now
   7M-of-10M) with a dead `train_base` line pointer; stage-3's
   `fall_penalty` "proportional to reduced alive_bonus (0.05)" directly under
   `alive_bonus = 0.0`. In this repo the comments *are* the experiment record
   — #493 and #494 both quote them — so these are worth fixing like code.

---

## 9. Suggested priority order

1. **Substep contact aggregation** (§1) — it changes what the gate, the
   reward, and every probe measure, and it should land before the plant
   revision consumes multi-seed runs.
2. **`warmup_ent_coef` fix** (§3.1) and **JAX resume-decay fix** (§3.2) —
   silent, affect every species' multi-stage runs.
3. **PR #494**: correct the validation narrative (§5.1), sweep the remaining
   conflation phrasing (§5.4), then merge; **PR #493**: move the probes out of
   the certification `try` (§4.1), then merge. Both PRs are otherwise sound.
4. The reward-shallowness question (§2.1) — decide whether to steepen support
   terms before the next 8-hour run, since three runs have now plateaued in
   the same band; pair it with the measurement fix so the reward prices real
   airborne time.
5. Everything in §§3.3–3.10, 6.2, 7, 8 as batched cleanup.
