# Stage 1b Implementation Plan — T-Rex Recovery Stage

**Date**: 2026-08-15. **Baselined on**: `main` @ e02f5dc (the tree containing the
first certified stance-gate pass). **Companion to**:
`docs/STAGE1_SPLIT_PLAN.md` (revision 5) — that document argues *why* stage 1
splits into stance (1a) and recovery (1b) and specifies the mechanism; this one
maps the unbuilt half onto the tree as it stands after the gate-pass run.
Where the two disagree, the split plan's adopted design decisions stand; this
document only re-sequences the remaining work and records what the three
August runs changed. Statements about current code cite `file:line` in this
tree; they will drift and should be read as anchors, not contracts.

---

## 1. Where stage 1a stands

Run `20260810_145546` (seed 42, 10M steps, SB3 PPO, physics r7 / policy
interface r11) is the first certified `stance_quality/v1` PASS in the
repository's history
(`docs/investigations/TREX_STAGE1_GATE_PASS_RUN_2026_08.md`): full-horizon
40/40 vs ≥ 0.95 required, mean unsupported duty 0.0048 with one-sided 95% UCB
0.0080 vs ≤ 0.02 required, reward 3368.7 ± 46.7 against the 2100 rail — 96.4%
of the zero-action statue's 3495.2, with no actuator saturated and balance
carried by ±4–5° hip-roll sways around the home pose. The probes describe the
first healthy animal of the campaign: survival at every filter cutoff
including 5 Hz, a near-statically-stable constant-hold (3/10 full horizon,
mean 611 steps, gentle nosedives — vs 0/10 and backward tail-slams one run
earlier), and an over-determined release ablation.

Everything 1a needs from the split plan is shipped and verified in code: the
fail-closed gate schema (`environments/shared/curriculum/gate_schema.py`), the
`stance_quality/v1` statistic (`environments/shared/curriculum/stance_gate.py`),
the T-Rex stage-1 TOML converted to it at reset noise 0.05 with the reward
threshold demoted to a 0.60×-statue rail, JAX-path gate parity, notebook
enforcement that raises on a failing gate, the reset/plant repairs, and the
statue-constants freshness guard (`test_statue_constant_freshness.py`).

Two caveats the postmortem itself flags, which shape the run plan in §5:
one seed per condition across all three August runs, and the 10 Hz filter
(#503) and the shaping pack (#504) changed together, so their individual
contributions are not separable. The split plan's §12.7 open question — does a
policy satisfying both gate criteria exist at all — is now answered yes.

## 2. Stage-1 closeout: cleanup found by this review

Verdict: **nothing below blocks starting the 1b build.** The items in §2.1
should land before the next *training run* because they would pollute or
mislead its evidence; §2.2 is documentation the repo's own policy says to
prune; §2.3 is cosmetic. Each item was verified against the current tree.

### 2.1 Before the next training run — **landed 2026-08-15 on this branch**

1. **`diagnostics/action_saturation` logger-key collision.** The #504 pack
   added `action_saturation` to `INFO_KEYS`
   (`environments/shared/diagnostics.py:170`), recorded per rollout at `:251`
   — and later in the *same method* (`:315-316`) the callback records its
   pre-existing raw pre-clip |a| ≥ 0.99 fraction under the identical key,
   overwriting the env's filtered-command ramp fraction every rollout. Two
   different definitions share one TensorBoard key; the one the new reward
   term is built on is the one that never survives. Rename the callback's
   metric (e.g. `diagnostics/raw_action_saturation`), update
   `test_diagnostics` and the KNOWN_ISSUES metric-hazard paragraph.
2. **The filter-probe sweep predates the plant's own 10 Hz filter.**
   `stance_probe_filter_hz = [5, 10, 20, 30, 35]`
   (`configs/trex/stage1_balance.toml:473`) and its "close the curve at
   Nyquist" rationale were written for the unfiltered plant; r11 (#503) now
   low-passes every command at 10 Hz inside the plant, so the 30/35 Hz points
   stack a wider filter onto a narrower one and measure approximately the
   unprobed checkpoint, each costing 10 episodes. (A first-order pole is not a
   wall — 22.7 Hz content still passes at ~0.40 gain — but "passes nearly
   clean" is structurally impossible now.) Re-derive the sweep below the plant
   cutoff (e.g. `[1.0, 2.5, 5.0, 8.0]`, keeping one ≥ 10 Hz point for
   cross-run continuity) and rewrite the derivation comment.
3. **The three new release-ablation groups have no test coverage.**
   `test_every_group_gets_both_directions`
   (`environments/shared/tests/test_stance_gate_report.py:1748`) still
   iterates only the five r6-era groups; #501's `knees_ankles`, `left_leg`,
   `right_leg` (`stance_gate_report.py:2052-2054`) — the groups the
   knee-localization conclusion rests on — could be deleted or broken without
   a test noticing. Extend the tuple (and `_TREX_ACTUATORS` if needed).
4. **The factory parity test pins the pack weights with `> 0.0`.**
   `test_trex_mjx_reward_parity.py:205-207` asserts only positivity for
   `tail_home_pose_weight`, `action_saturation_weight`,
   `leg_home_pose_broad_fraction`, while every neighboring assertion pins the
   exact TOML value. The statue rails (2100 / 3495.2) were derived at exactly
   0.25 / 0.5 / 0.25 — a silent TOML drift invalidates the rails without
   failing the test. Pin exact values (and `tail_home_pose_tolerance = 0.05`).

### 2.2 KNOWN_ISSUES pruning — **landed 2026-08-15 on this branch** (the file's own policy: "when an item here gets fixed, delete it")

5. **"An action filter cannot be retrofitted" (`docs/KNOWN_ISSUES.md:82`)** —
   the entry prescribed training with the filter present; r11 did exactly
   that, invalidated the unfiltered checkpoints, and the gate-pass policy
   survives every cutoff. Reduce to a one-line historical pointer at the
   gate-pass postmortem §4 and the r11 CHANGELOG entry.
6. **HIGH hold-constant item (`docs/KNOWN_ISSUES.md:90`)** — its stated
   closure condition ("re-run on a policy trained on the 15-actuator plant")
   was met by the gate-pass run. Rewrite around the r7/r11 measurement (3/10
   full horizon, mean 611 steps, near-static stability) and drop the r6-era
   contraindication against raising smoothness/jerk, which now actively
   misdirects stage-2 tuning.
7. **Saturation item (`docs/KNOWN_ISSUES.md:68`)** — "nothing opposes it" is
   false since `action_saturation_weight = 0.5` landed, and "counts have not
   been re-measured on the 15-actuator plant" is false since the
   narrow-tolerance postmortem measured 5 of 15 parked. Update with both
   citations (and note the gate-pass run's 0 of 15).
8. **Stale line-number anchors in the latent-trap and metric-hazard items**
   (`docs/KNOWN_ISSUES.md:303, :320`) — the cited `base_env.py:765/783` moved
   ~230–280 lines in the August rewrites, and the raw-vs-clipped reward trap
   is now *closed for the T-Rex* (r11's `_filter_action` clips before the
   reward terms read the action, `base_env.py:963-989`) while remaining real
   for filter-free species. Re-anchor at function level and scope the claim.

### 2.3 Cosmetic / opportunistic — **landed 2026-08-15 on this branch** (item 12 as a rename with a legacy-reader fallback; item 13 as a real fix — the panel bug was NaN-key detection, not a missing skip)

9. `configs/trex/stage1_balance.toml:549`: the release-ablation cost comment
   says "13 panels"; the current plant's group set gives 17.
10. `stance_gate_report.py:413-422`: `_actuator_pose_mapping`'s docstring
    describes the deleted toe actuators in the present tense; add the same
    past-tense r6 framing #500 used elsewhere in the file.
11. Stale TOML provenance comments: `posture_weight` "Increased from 1.5" (a
    no-op), `nosedive_weight` "Increased from 3.0" (a decrease), the `[sac]`
    block's 6M-era sizing, `ent_coef_end`'s 21-action-dim reasoning.
12. The evaluation CSVs' `success` column is the stage-3 task event and reads
    False in every stage-1 row by construction (gate-pass postmortem item 1);
    rename to `task_success` or omit in stage-1 artifacts.
13. The biped-empty "diagonal pair" contact panel (narrow-tolerance
    postmortem item 6) — still cosmetic, still unfixed.

### 2.4 Flags — not cleanup, but check before 1b leans on them

- **`[jax] fall_penalty = -10` vs `[env] -100`**
  (`configs/trex/stage1_balance.toml:278` vs `:171`): a documented deliberate
  divergence, but it means every statue rail and postmortem baseline is
  SB3-referenced. Any 1b null baseline must be measured on the backend that
  will train (see §6, decision 6).
- **`[jax] num_updates = 500`** (`:265`) implies ~65.5M env steps against the
  10M curriculum budget — verify which one the JAX stage-1 path honors before
  any JAX 1b run.
- **`action_jerk = 3.0` / `smoothness = 2.0` derivation comments describe the
  pre-filter signal** — since r11 the terms read the filtered command, so the
  calibration numbers in the comments (init jerk 65.4/step etc.) describe a
  signal that no longer reaches them. The weights are empirically validated by
  the gate-pass run; the comments are not. Re-derive or annotate.
- **`settle_steps = 200` is still `[inferred]`**
  (`configs/trex/stage1_balance.toml:337`) — and 1b's push schedule interacts
  with it directly (§6, decision 4). Calibrate against stored rollouts before
  the 1b gate depends on it.

## 3. What 1b is (adopted design, in brief)

Same plant, same reward, same observations, same terminations as 1a — plus
scheduled external shoves, delivered as horizontal forces written to
`data.xfrc_applied[root, 0:3]`, and a gate that certifies **per-shove
recovery** rather than mere survival. Adopted parameters
(`STAGE1_SPLIT_PLAN.md` §3.3): `perturbation_capture_velocity_multiple = 1.5`
(dimensionless; ~150 N for 0.20 s on this plant), `perturbation_interval =
2.0` s with `± 0.5` s jitter (defeats a clock-timed brace),
`perturbation_duration = 0.20` s, `perturbation_direction =
"uniform_horizontal"`; derived force/impulse persisted per species. The
statue — global optimum of the undisturbed reward — survives 0/40 under the
push (exact 95% upper bound 7.2%), which is the whole point: only a
disturbance can measure feedback control. Explicitly provisional, to be set by
measurement, not adopted: `p_recovery`, `Δ_success`, `T_recover`, ramp-vs-fixed
onset, and the 3M warm-started budget.

## 4. Build map

Ordering follows the split plan's steps 8–14 with two updates: step 9's core
(the stance-gate statistic) already shipped, and everything below is rebased
onto the r7/r11 tree. Each workstream lands independently and default-off.

### W1 — Perturbation engine (split plan step 11). Build first.

The only workstream that touches physics. Default off — byte-identical
trajectories when disabled is the acceptance test.

- **Pure kernel**: `external_push_force(...)` in
  `environments/shared/reward_functions.py` — schedule in, per-control-step
  force vector out; shared verbatim by both backends (the file already hosts
  the backend-shared pure functions).
- **SB3/mujoco path**: `_apply_perturbation()` called at the top of
  `BaseDinoEnv.step()` (`environments/shared/base_env.py:982`) — write
  `data.xfrc_applied[root, 0:3]` while a pulse is active, zero it when not,
  zero it unconditionally in `reset()` (`:1299`).
  `scripts/hw_chassis_study.py:359-367` already demonstrates the exact
  write-hold-clear pattern against this plant.
- **MJX/JAX path**: same schedule unit via `data.replace(xfrc_applied=...)` in
  the `environments/shared/mjx_env.py` / `jax_*` step functions; auto-reset
  must clear it; one schedule definition must produce identical pushes for one
  SB3 env and thousands of vectorized MJX envs.
- **Schedule determinism**: push times, directions, and magnitudes
  pre-generated per episode from the episode seed; identical schedules for the
  policy and every null controller on the same seed (pairing is part of the
  1b gate's estimand); realized schedule persisted with the run. Magnitude
  derives from capture-velocity multiple × **kinematic-subtree mass** (not
  `mj_getTotalmass`, which includes the 65.45 kg prey body — split plan §6).
- **Config**: `perturbation_*` keys in `[env]`, defaulting to 0.0/off for
  every species and stage; register them in the fail-closed key validation the
  same way the `stance_probe_*` keys are registered, so a typo is fatal, and
  they should be rejected (or warned) under gate kinds that can never use
  them.
- **Early smoke test**: confirm `xfrc_applied` is honored by the MJX pipeline
  under the pinned mujoco 3.10.0 before building on it.
- **Tests**: force-off regression (disabled ⇒ byte-identical trajectories on
  both backends), clear-on-reset, schedule determinism and SB3↔MJX parity,
  pulse timing (a recorder-env test that the force is applied at exactly the
  scheduled steps — the impulse-probe test gap found by the August review is
  the cautionary tale), resume behavior.

### W2 — Task fingerprint and load modes (step 8a)

A `step()`-hook push moves no `policy_interface_revision` — old checkpoints
stay mechanically loadable while being unvalidated for the pushed task. Close
the provenance gap with a **task/evaluation fingerprint** (perturbation
parameters + schedule identity + gate kind) recorded in run provenance, and
two explicit load modes in the SB3 and JAX entry points:
`resume_same_stage` (fingerprint must match exactly) and
`initialize_next_stage` (mismatch expected once, recorded as lineage — the 1a
→ 1b warm-start uses this).

### W3 — Stage identity (step 8b). **Decision required — §6.1.**

The integer-stage assumption is pervasive: `_STAGE_FILE_PREFIX`
(`environments/shared/config.py:154`, plus `:457`), the bundle subset check
(`environments/shared/reporting/bundles.py:90-96`),
`species_catalog.py:300`, `train_base.py:1144`, the JAX CLI
(`jax_training.py:239`), the sweep CLI and orchestration
(`scripts/sweep/__main__.py:38,61`, `orchestration.py:823,949`), and the
website's generated species data. The split plan's answer is the semantic
stage manifest (`stance` / `recovery` / `locomotion` / `behavior` with
`legacy_alias` and backward readers) precisely because renumbering silently
changes what "stage 2" means in every historical artifact. The manifest is the
right end state; it is also the largest non-physics chunk of 1b. The minimal
alternative — a T-Rex-only `1b` entry threaded through the integer plumbing —
is smaller but leaks the string/int split into every one of the sites above.
Either way, the new stage's config lands as
`configs/trex/stage1b_recovery.toml` (or the manifest's naming), a copy of
`stage1_balance.toml` plus the `perturbation_*` block and the W4 gate keys.

### W4 — Recovery gate: `recovery_quality/v1` (steps 9-remainder, 12, 13)

- **New gate kind** in `GATE_KINDS` (`gate_schema.py:51`; the module
  documents the extension recipe at `:30`) with its own required threshold
  keys, fail-closed like the others.
- **Per-shove recovery events**: recovery = re-entering the safe
  pose/velocity/contact set within `T_recover` of pulse end and dwelling
  there; computed from per-step state on both backends (the substep-honest
  contact machinery from #499 is the foundation). Evidence rows per episode
  **and per shove**: controller ID, pair ID, seed, push vector and timing,
  recovery-entry step, dwell — the split plan makes pair identity part of the
  estimand.
- **Gate statistic** (provisional until calibrated):
  `LCB95(P(full horizon AND every shove recovered)) ≥ p_recovery`, plus the
  paired same-seed same-schedule superiority test against the null suite —
  zero action, constant/brace controllers, and the incoming 1a checkpoint —
  with the multiplicity rule declared in advance. If any reward threshold is
  retained, the paired test is authoritative.
- **Reporting**: extend `stance_gate_report.py` (or a sibling
  `recovery_gate_report.py`) so the certification artifact carries the
  per-shove table and the null margins; every metric reported as margin over
  the measured null, per split plan §6.

### W5 — Gate resolver (step 10). **Scope decision — §6.2.**

The split plan wants the full resolver — capability spec, null manifest, and
decision procedure frozen into an atomic `gate_resolution.json`, with
missing/stale baseline data *blocking* advancement — landed before any stage
that depends on it. The non-negotiable core for 1b's first run is smaller: the
null baselines must be measured on registered seeds at the frozen task
fingerprint, persisted, and their absence must fail closed. Whether the full
atomic-resolution machinery lands now or after the first calibration run is a
scope call.

### W6 — Diagnostic tool modes (split plan §7.3; blocked on W1)

`zero_action_baseline`, `joint_excursion_report`, `action_bound_report`, and
`observation_ablation_report` gain explicit `plant_sanity` (push forced off)
vs `task_gate` (push exactly matching the advancement evaluation) modes with
persisted fingerprints, so a statue baseline measured without the push is
never silently compared against a pushed run.

## 5. Runs and pilots

- **P1 — seed replicate of the gate-pass configuration** (postmortem item 4).
  Compute-only, no code conflict: can start now, in parallel with W1. This is
  "the only thing that distinguishes solved from lucky," and it also
  re-answers the hold-constant and saturation items in §2.2 on a second seed.
- **P2 — filter-only ablation** (10 Hz filter without the #504 pack) to
  separate the two interventions' contributions. Lower priority than P1;
  same parallelism.
- **P0 — §8.1 transfer pilot**, the moment W1 lands: enable the adopted push
  against the certified 1a checkpoint, ~10-minute eval. A coarse answer
  already exists — the gate-pass impulse probe's recovery envelope (0.50 m/s
  one direction, 0.00 the other) shows the stance is actively controlled but
  asymmetric — but the pilot answers it under the real force schedule. If the
  first shove destroys the stance, the ramp option (§3.3) gets its pilot
  before any 10M-step commitment.
- **P3 — re-measure every push-calibration figure** at current main on
  registered seeds (split plan §8.2: all §3.4 numbers predate the current
  plant and are not reproducible from the repository). Produces the null-suite
  baselines W5 freezes.
- **P4 — first 1b training run**: warm-start from the gate-pass
  `robust_best_model.zip` (9.55M, r11-native) via `initialize_next_stage`;
  provisional 3M budget; ramp-vs-fixed per P0.
- **P5 — calibrate and declare** `p_recovery`, `Δ_success`, `T_recover` from
  P3/P4 evidence; enable the recovery stage for T-Rex only (step 14); other
  species wait on their own plant preflights (velociraptor's foot-sensor
  under-read is still open — `raptor.xml`'s single `r_toe_d3` site reads
  ~55% of true load).

Dependency sketch: W1 → {P0, P3, W6}; W2, W3 → P4; W4 → P5; W5 → P5.
Cleanup §2.1 lands before P1/P4 so their evidence is clean. P1/P2 are
compute-parallel with everything.

## 6. Open decisions (for review before the corresponding workstream starts)

**Decided 2026-08-15** (review with the project owner): (1) **semantic stage
manifest** — recovery is a real fourth stage, curriculum order
stance → recovery → locomotion → behavior, with legacy aliases preserving
historical artifact meaning; (2) **full gate resolver before the first 1b
training run** (not the minimal core); (3) the seed replicate ran — seed 43,
gate FAIL at duty 0.0597, so the "passes or bounces" question is answered
seed-sensitive at n = 2 (see
`investigations/TREX_STAGE1_SEED43_REPLICATE_2026_08.md`); a third seed is
queued; (6) **SB3 first** for all 1b evidence. Standing constraint from the
same review: the perturbation engine must be **species-generic from the
first commit** — shared code only, per-species magnitudes derived from each
plant, config keys available to every species, off by default. Decisions 4,
5, and 7 below remain open and are settled inside W1/W4 design or by P3
measurement.

1. **Stage identity scope (W3)**: full semantic manifest now, or minimal
   T-Rex-only `1b` insertion with the manifest deferred? The manifest is the
   designed end state and avoids paying the migration twice; the insertion is
   faster to a first 1b run but touches the same ~10 sites with a hack.
2. **Gate resolver scope (W5)**: full atomic `gate_resolution.json` before
   P4, or the minimal frozen-null-baseline core first? (Fail-closed on
   missing baselines is non-negotiable either way.)
3. **Compute plan**: does P1 (seed replicate) start now in parallel with the
   W1 build? It occupies a training slot for ~11 h but upgrades the
   gate-pass from mechanistic to statistical evidence.
4. **Push clock vs settle window**: the first push draw at `2.0 ± 0.5` s
   lands at 1.5–2.5 s; `settle_steps = 200` ends at 2.0 s — so the first
   shove can land inside the still-uncalibrated settle window. Decide whether
   the push clock starts at episode start or settle end, and calibrate
   `settle_steps` (§2.4) before the 1b gate depends on either.
5. **Duty semantics under push**: the 1b gate is recovery-based and does not
   reuse `max_unsupported_duty` — but the eval harness reports duty
   everywhere, and a legitimate recovery step may lift a foot. Decide
   explicitly whether any stance screen applies between shoves in 1b, and
   label the duty columns in pushed panels as diagnostic, so a number that
   looks like the 1a gate is not read as one.
6. **Backend for the first 1b runs**: SB3 (where every rail, baseline, and
   postmortem lives) vs JAX (which trains against `fall_penalty = -10` and
   has the unverified `num_updates` budget, §2.4). Recommendation: SB3 for
   P0/P3/P4 evidence continuity; bring JAX to parity via the W1 tests before
   using it for 1b training.
7. **Reset noise for 1b evaluation**: the design's principle is robustness
   from declared pushes at the 1a operating point (0.05), but the split
   plan's step-12 evaluation text says noise 0.10, and the historical 34/40
   push-recovery figure was measured at 0.05. P3 should measure both once;
   the gate should then commit to one, explicitly.

---

*This plan was produced by the 2026-08-15 stage-1 review; the cleanup findings
in §2 were each verified against the tree at e02f5dc before inclusion.*
