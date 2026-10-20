# Per-foot CoP band: a code-only prototype of the stance4 fallback term

Written 2026-10-20, 00:35-01:30 UTC. Proposal only. Nothing trained. Nothing was pushed. Drive and GitHub were not
touched, and nothing was written in checkout. The maintainer approves any reward change.

"Measured" means I ran it or read it from stored files. "Inferred" is marked. Paths: SP =
scratchpad, C = SP/foot/stance4/copband.

Nothing here changes the stance4 criteria (SP/foot/stance4/criteria.txt, sha256 978a93da..., re-checked) or any
stance4 verdict. Fails stay fails.

## 0. In one paragraph

stance4 decision line 5 says: if the stagger penalty does not hold fore-aft from scratch, the next round tests "the
per-foot CoP band term (new contact-reward code)". This note builds that term as an opt-in [env] option of the T. rex
env, off by default, on a scratch branch. Each step, it reads each foot's whole-foot centre of pressure (CoP) exactly
as the judge's K2b statistic reads it, using the recorder's own code. It charges a foot whose CoP lies outside
[-0.6, +0.4] span half-lengths. Two guards close the side doors that spec 3.3 named. A foot that is unloaded, carries
under 35% of the load, or stands off its pad pays the full per-foot price. The pad's own CoP must also lie in
[-0.65, +0.65], so load moved onto the digits cannot pay the band. Off, it changes nothing (measured, byte for byte).
The proposed dose is w 0.30, the stagger term's maximum, with tolerance 0.357 (25 mm). On stored stance3 rollouts it
separates every failing panel from its clean episodes (AUC 1.000). It charges the statue 1.9 per episode.

## 1. Where it lives (scratch only)

- Clone: SP/fix/scratch_copband, from SP/fix/screen_s4_LTS at c1e146a8. Branch scratch/cop-band. Push URL disabled
  (`DISABLED-no-push`). One commit, f94452ad, on top of c1e146a8.
- The base already carries the stagger term (ff7114c0) and the stance4 config commit (stagger on). The band is a
  separate switch. Nothing turns it on.

Files:

| file | change |
|---|---|
| environments/shared/gait/recorder.py | The per-step reductions move into module functions: `decode_settings`, `decode_normal_forces`, `classify_floor_contacts`, `per_substep_force`, `sole_cop_steps`, `whole_foot_cop_steps`. Same operations, same order. `_reduce` calls them. |
| environments/shared/gait/stance_metrics.py | `whole_foot_cop(trace)` calls a new `normalized_foot_cop(foot_cop, leg_floor_max, span)`, which holds its normalization and loaded rule. |
| environments/shared/gait/step_foot_cop.py | New. `StepFootCop` copies each substep's contacts as the recorder's hook does, and reduces one step with the functions above. |
| environments/shared/reward_functions.py | New pure functions `band_excess` and `reward_stance_cop_band`. |
| environments/tyrannosaurus_rex/base_env.py | Seven opt-in kwargs, validation, the `OPT_IN_TASK_KWARGS` carve-out, `_cop_band()` (lazy reader), `_accumulate_substep`, `_invalidate_substep_aggregates`, reward section 8i, and the total. |
| environments/shared/behavior_checkpoint.py | The six non-weight band settings join `_TRANSITION_REWARD_SETTINGS`, as the stagger tolerance did. |
| tests | New environments/tyrannosaurus_rex/tests/test_anatomical_stance_cop_band.py (31 tests). One new mirrored-rollout case in test_symmetry.py. Two new transition cases in test_behavior_checkpoint.py. The layering test names one exception and checks it (section 6.1). |

## 2. The definition

Per step t and foot k (r, l):

```
c_k   = whole-foot CoP: every floor contact of the limb over the step's 5 substeps, force-weighted, in the sole
        frame of its own substep, in span half-lengths (-1 heel line, +1 toe line; 1 unit = 70 mm on r9)
        = the recorder's EpisodeTrace.foot_cop for step t, normalized by stance_metrics.whole_foot_cop
p_k   = pad CoP along the pad's forward axis, in pad half-lengths (EpisodeTrace.sole_cop[t, k, 0]); the span is the
        pad's own edges, so c and p share one scale
u_k   = max( excess(c_k, [-0.6, +0.4]),  excess(p_k, [-0.65, +0.65]) )        excess = distance outside, 0 inside
q_k   = exp(-(u_k / 0.357)^2)              if foot k is counted
      = 0                                   otherwise (guards, section 4)
reward_stance_cop_band = -w * mean_k (1 - q_k)     if step_count > 200 and command == 0
                       = 0                          otherwise
w = stance_cop_band_weight (default 0.0 = off; proposed 0.30)
```

The kwargs and defaults:

| kwarg | default | meaning |
|---|---|---|
| stance_cop_band_weight | 0.0 | the switch; proposed 0.30 |
| stance_cop_band_lower / _upper | -0.6 / +0.4 | the whole-foot band (spec 3.3's tight band) |
| stance_cop_band_pad_lower / _upper | -0.65 / +0.65 | the pad band (guard 2) |
| stance_cop_band_tolerance | 0.357 | 25 mm, in span half-lengths |
| stance_cop_band_min_load_share | 0.35 | guard 1 |

Info keys, only when on: per foot `<s>_cop_band_foot_cop`, `_pad_cop`, `_load_share`, `_digit_share`, `_counted`,
`_quality` (s = r, l; a reading on an unloaded foot is written 0.0, so every value is finite), plus
`stance_cop_band_quality` and `reward_stance_cop_band`. The term enters reward_total only when on.

- **Gate.** The stagger term's gate: after stance_width_settle_steps (200) and while the command is zero. The K2b
  window also starts at step 200. The spawn and settle are not charged.
- **Mirror invariance.** c and p are fore-aft readings, so the mirror only swaps the feet. The mean over two feet is
  exactly symmetric (a + b = b + a in floating point). Measured: the per-foot info keys swap and the reward is
  unchanged to 1e-9 under `mirror_env_state`, and over 260-step mirrored rollouts (test_symmetry, 2 seeds x 2 action
  sets, reward and info to 1e-9).
- **The same aggregation as the recorder.** `StepFootCop` copies, per substep, what the recorder's hook copies: the
  contact pairs, the normal forces (the same decode), the contact points and the sole poses. It reduces one step with
  the recorder's functions and the gate's normalization. `np.bincount` sums each step's contacts in the same order
  either way. Measured: on a 300-step noisy episode, every per-foot whole-foot CoP and pad CoP equals the recorder's
  trace byte for byte (600 of 600 foot-steps; pinned by a test).
- **States scored outside step()** (after reset, hand-posed tests) read the current contacts as one substep. A
  training step always reads its own 5 substeps.

## 3. The dose, calibrated as spec 3.1 did

All numbers here are measured on stored rollouts, with arithmetic only (C/scripts/size_copband.py ->
C/tables/size_copband.txt). The inputs are SP/foot/stance3/night/diag_split/out (Run 3 at 8.39M and 6.29M, Run 2 at
4.19M, the L+T statue, four counterfactual variants; per-step whole-foot CoP, pad CoP, forces and rewards) and
diag_drift/out (nine checkpoints, whole-foot CoP only). Window: steps 200-999, full-horizon episodes.

Checks first:
- K2b re-derived from the per-step CoP agrees with the judge's flags on 320 of 320 episodes.
- With the old split-diagnosis settings (w 0.25, tolerance 0.357, no guards) the script reproduces the split notes'
  "CoP band tight" column exactly (0.0729 / 0.0028 / 0.0020 / 0.1138 / 0.0073).

### 3.1 Which band, and which feet fail where (measured)

- Run 3 at 8.39M fails K2a heel-side: the forward foot's window CoP is -0.86 to -0.97 (u = 0.26-0.37 past -0.6).
- Run 2 at 4.19M fails K2b toe-side: one foot at +0.66 to +0.99, the other at -0.59 to -0.90.
- So the band must price both edges. [-0.6, +0.4] does; its AUC is 1.000 on both panels.
- The statue's per-step whole-foot CoP: p1 -0.40, median +0.05, p99 +0.59. So the statue's toe transients brush the
  upper edge. That is its whole tax (below).

### 3.2 Slope against the existing price

What the reward already charges, per step per 0.1 span half-length (7 mm) of heel depth (Run 3 8.39M, 40
episodes): reward_total 0.0097 by regression (r +0.53), 0.0131 split against clean. leg_home_pose is most of it
(0.0071). Run 2 4.19M: 0.0053.

The new term's slope for one foot u past the edge (the term is a mean over two feet):

| u past the edge (span units) | 0.26 (CoP -0.86) | 0.30 | 0.33 (K2b edge) | 0.37 | 0.40 |
|---|---|---|---|---|---|
| w 0.30, tolerance 0.357 (proposed), per 0.1 | 0.0360 | 0.0349 | 0.0331 | 0.0297 | 0.0268 |
| times the regression price (0.0097) | 3.7 | 3.6 | 3.4 | 3.1 | 2.8 |
| times the split-vs-clean price (0.0131) | 2.7 | 2.7 | 2.5 | 2.3 | 2.0 |
| total restoring slope, times today's | 4.7 | 4.6 | 4.4 | 4.1 | 3.8 |
| w 0.30, tolerance 0.25, per 0.1 (not chosen) | 0.0423 | 0.0341 | 0.0277 | 0.0199 | 0.0148 |

- Tolerance 0.357 puts the slope peak at u = 0.25, at the near edge of the failing heel cluster (0.26-0.37), and keeps
  at least 2x the existing price out to CoP -1.0. Tolerance 0.25 peaks at u = 0.18 and drops to 1.5x the regression
  price at u = 0.4. Tolerance 0.15 is flat past u = 0.3.
- This mirrors spec 3.1's choice for the stagger term (peak on the first failing cluster, at least 1.8x the existing
  price out to the deepest one).

### 3.3 Window costs (cost per step; per window = x 800)

| group | w 0.30 (proposed) | w 0.40 | w 0.50 | stagger term w 0.30 (spec 3.1) |
|---|---|---|---|---|
| R3 8.39M K2a episodes (6) | 0.0883 (71) | 0.1177 (94) | 0.1472 (118) | 0.177 (142) |
| R3 8.39M clean (34) | 0.0034 (2.7) | 0.0046 (3.6) | 0.0057 (4.6) | 0.022 (17) |
| R2 4.19M K2b episodes (14) | 0.1368 (109) | 0.1824 (146) | 0.2279 (182) | 0.044 |
| R2 4.19M clean (26) | 0.0087 (7.0) | 0.0116 (9.3) | 0.0145 (11.6) | 0.008 |
| R3 6.29M (40) | 0.0024 (1.9) | 0.0032 | 0.0040 | 0.0082 |
| statue (39) | 0.0024 (1.9) | 0.0032 (2.5) | 0.0040 (3.2) | 0.0015 (1.2) |
| AUC split vs clean, R3 8.39M / R2 4.19M | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.967 |
| split minus clean, R3 8.39M, against the 48 per window reward_total already charges | 68 (1.4x) | 91 (1.9x) | 113 (2.4x) | 125 (2.6x) |

- Live statue check (measured, zero action, seeds 3042-3081, L+T [env], band on at w 0.30): the band costs 1.91 per
  full-horizon episode (worst 27.0). The stored-rollout estimate was 1.9. So the term-on L+T statue is about 4457.5
  on full-horizon episodes (4459.40 - 1.91; inferred by subtraction).
- Maximum 0.30 per step = 240 per window = 5.4% of the L+T statue return. The stance pays about 4.4-4.5 per step and
  a fall costs 100, so the term never makes ending an episode pay.
- The drift sets (whole-foot band only; no pad or forces stored) agree: AUC 0.982-1.000 where a panel has K2b
  episodes; clean checkpoints cost 0.002-0.007 per step. Arm L s60 at 5.24M costs 0.072 per step on its full-horizon
  episodes, none of them outside K2b (its 2 K2b episodes are falls). That checkpoint's ankles sit +4.7 deg open
  (spec 1), so its CoP swings toward the edges without crossing K2b's.

### 3.4 The proposed dose and the alternative

- **Proposed: w 0.30, tolerance 0.357.** Its slope is 2.0-3.7x the existing price across the failing cluster. Its
  maximum equals the stagger term's, so a later screen can compare the two terms at one maximum price. It charges
  clean episodes far less than the stagger term (2.7 against 17 per window on Run 3 8.39M).
- **Alternative: w 0.50.** Use it if the maintainer wants the window price of a split, not only the slope, to be at
  least 2x what the reward charges today (2.4x). The statue then pays 3.2 per episode.
- Inferred: the stagger term's own from-scratch result (stance4) is the best guide to which matters more, the slope or
  the window price. Decide the dose after the stance4 verdict.

## 4. The guards (the two side doors of spec 3.3)

### 4.1 Guard 1: unloading or lightening a foot

**The door.** K2b and the band read only loaded steps. A penalty that skipped unloaded feet (as the split
diagnosis' band did) would pay the policy to lift the foot whose CoP is out of band.

**The guard.** A foot is counted only if all three hold on the step:
- the limb is loaded by the gate's own rule (a finite whole-foot CoP and more than 0.1 N on some substep);
- its share of the two limbs' mean floor force is at least 0.35 (the v2 gate's windowed bar,
  min_foot_load_share_windowed);
- its pad is loaded (guard 2b).

An uncounted foot has q = 0. It pays the full per-foot price, w/2, whatever its readings.

**Why it works.**
- Lifting a foot is never cheaper than the worst CoP.
- The band prices where the load sits, not how much. So lightening a foot between 35% and 50% changes nothing, and
  below 35% it costs the maximum. The term never pays the policy to shed load.
- A foot that only grazes the floor (0.1 N on one substep) is loaded by the gate's rule but fails the share bar.

**Measured.** On every stored panel (the statue, three policies and four counterfactual variants) guard 1 fires on
0.000% of window steps. The smallest per-step share seen is 0.435. Hand-posed tests: a lifted right foot pays exactly
0.15 (w/2); a foot at 43% passes the default bar and pays the maximum at a 0.45 bar.

### 4.2 Guard 2: moving load onto the digits

**The door (measured geometry).** On r9 the three digits hinge at the MTP axis, at -0.86 span units. Their capsules
run forward to +1.00, +1.86 and +1.29, past the pad's toe line (C/tables/digit_geometry.txt). The whole-foot CoP
counts digit load, but the pad CoP does not. So the digits are a lever on the whole-foot CoP alone.
- Moving the statue's digit share (0.17) from the MTP axis to the d3 tip moves the whole-foot CoP by +0.46 span units
  (32 mm). That is more than a K2a heel excursion (u 0.26-0.37). A heel-loaded foot could press its toe tips and look
  centred while its pad stays on its heel. K2a, which reads the pad, would still fail.
- The reverse works on the toe side: loading the digit bases pulls the whole-foot CoP back while the pad stays
  toe-loaded. Measured in a hand-posed test: toes extended 0.6 rad put 40% of the right foot's load on its digits.
  The whole-foot CoP reads +0.03 (inside) while the pad reads +0.63.

**The guard.**
- (2a) Pad band. The pad's own CoP must lie in [-0.65, +0.65]. The foot pays the larger of its two excesses. Digit
  load cannot move the pad's CoP.
  - The edges come from the data, not by hand. The digits shift the whole-foot CoP from the pad CoP by -0.25 to
    +0.04 (p1-p99, every stored panel). The whole-foot band widened by that shift is [-0.64, +0.65], rounded to
    [-0.65, +0.65].
  - So a normal stance whose whole-foot CoP is in band has its pad in band too. A digit-made whole-foot reading does
    not.
- (2b) Off the pad. A foot standing on its digits alone (pad unloaded: no sole contact above 0.1 N on any substep)
  is not counted and pays the maximum.

**Measured.**
- The pad band binds (its excess is the larger) on 0.005% of statue foot-steps, 0.04% on Run 3 6.29M, and 1.8% / 1.0%
  on the failing Run 3 8.39M and Run 2 4.19M panels. It is close to dormant on the stances seen so far.
- Hand-posed tests: toes flexed 0.4 rad lift the right pad (digit share 1.0), and the foot pays the maximum. The
  left foot, now under 35% of the load, pays it too (-0.30 in all). Toes extended 0.6 rad (the toe-side door) pay nothing at the default edge 0.65 (pad +0.63). With the edge at 0.5 they pay
  exactly the pad-excess formula.

**What guard 2 does not close (inferred).** A pad CoP between the band and the pad band (for example whole-foot +0.4
with pad +0.63) is free. Closing it would mean a tighter pad band, which starts to charge normal stances. Run 3
8.39M's toe-side feet already sit at pad +0.55 to +0.66.

## 5. Gaming analysis

| door | what the policy would do | what the term does | status |
|---|---|---|---|
| Lift a foot | unload the out-of-band foot | full per-foot price | closed (guard 1) |
| Lighten a foot | carry it lightly so its CoP is easy to place | under 35%: full price; above: no change | closed (guard 1) |
| Digit lever | press toe tips (heel side) or digit bases (toe side) | the pad band prices the pad | mostly closed (guard 2a); a small free zone, section 4.2 |
| Stand on the digits | lift the pad, carry the foot on its digits | not counted: full price | closed (guard 2b) |
| Rock the foot | keep the window mean in band while each step swings | every step is priced, so it is stricter than K2b's mean | no door |
| Settle window | build the posture during steps 0-200 | unpriced, by design (as the stagger and lean terms) | K2b's window also starts at 200 |
| Fall to escape | end the episode | at most 0.30/step against ~4.4/step and a -100 fall | no incentive |
| NaN or degenerate contact | readings undefined | uncounted: finite full price | tested |
| **Pay the judged statistic by another route** | put each foot's CoP in band with asymmetric ankle torques and keep the stagger (twist) | the band is satisfied; nothing prices the twist | **open: the main risk** (section 7) |

The last row is spec 3.3's "it pays the judged statistic itself". The split diagnosis found each foot's CoP is set by
its ankle torque over its load (corr +0.99). So the cheapest way to satisfy the band may be an ankle-torque asymmetry
that cancels the stagger's effect on the CoP, not the hip restoring torque the stagger term targets (inferred). K2
would then pass with a twisted stance. The stance4 stagger watches (foot_stagger at step 200 and in the window, yaw
change) would show it. Any band screen should keep them as report-only readouts.

## 6. Tests and checks (all measured on the scratch clone)

### 6.1 Tests

- test_anatomical_stance_cop_band.py, 31 tests, all pass:
  - defaults and the opt-in declaration;
  - off: byte-identical reward, reward_total, info and qpos over 240 steps; no band info key; the reader is never
    built and gait/step_foot_cop.py is never imported;
  - off: no task fingerprint or recorded stage config moves, on every T. rex stage; on: all seven kwargs are hashed,
    defaults included;
  - invalid values fail fast (10 cases);
  - the reading equals the recorder's trace byte for byte (300 steps);
  - on hand-posed states (as stood, ankles opened 0.06 rad, toes flexed 0.2 rad), the readings match an independent
    `mj_contactForce` loop to 1e-9, and the reward matches the formula to 1e-12;
  - the gates (steps 0, 1 and 200; a live command) pay 0 and still report the quality;
  - on: no dynamics change, and reward_total = off + the term, byte for byte, over 320 steps;
  - the guards (a lifted foot, a light foot, a foot off its pad, the pad band on the toe-side digit door);
  - the mirror: the per-foot readings swap and the term is unchanged;
  - the pure function: the formula, NaN-safety, exact invariance under swapping the feet, band_excess, and the slope
    table of section 3.2.
- test_symmetry.py: a new case, "screened options and CoP band on" (2 seeds x 2 action sets, 260 steps): mirrored
  rollouts agree in reward and info to 1e-9.
- test_behavior_checkpoint.py: a behavior transition may turn the band on and off (2 cases).
- test_gait_stance_metrics.py: the layering rule "no species env module imports the gait package" now names one
  exception, `TyrannosaurusRexEnv._cop_band`. A new test checks that the excepted import sits in that method's body
  alone: the env module imports nothing from gait at import time.
- The related files together: 319 passed, 0 failed (test_anatomical_stance_cop_band, test_anatomical_stance_stagger,
  test_gait_stance_metrics, test_gait_recorder, test_gait_pad_telemetry, test_symmetry, test_behavior_checkpoint).
- Full suite (environments/shared/tests and environments/tyrannosaurus_rex/tests, run from C, not the clone
  root): 6,324 tests, 6,276 passed, 32 failed, 15 errors, 1 skipped. Re-run from the same folder, the same 47 fail
  the same way on the base c1e146a8 (screen_s4_LTS): 0 differ. They are the base's known failures (the missing
  tensorboard package, the screen-config pins) plus tests that read repository-relative paths (recovery.toml, the
  MJCF, the notebook checkout) and so need the clone root as the working folder.
- The digest snapshot check (`digest_snapshot --block-optional-backends --check`) on the clone: 649 lines, 0
  errors, "Digest snapshot is current".

### 6.2 Identity checks against c1e146a8 (screen_s4_LTS), measured

- Off, byte for byte: four 1000-step noisy stance episodes (seeds 3042, 3050, 3052, 3081) under the stance4 [env]
  (stagger on) and under the L+T [env] (stagger removed). Every step's reward, reward_total, info keys and values, and
  qpos hash identically in both clones (C/logs/off_identity_*.txt).
- The stance task_sha256 is unchanged: sha256:3fa46708... with the stance4 config, and sha256:d4a3b8f8... with the
  stagger keys removed (both as the stance4 criteria record).
- The recorder refactor moves no recorded number: every EpisodeTrace field (balance and per-geom capture on) and the
  pad telemetry of three noisy episodes hash identically in both clones (C/logs/recorder_identity_*.txt).

### 6.3 Cost, measured

- Env step time, one thread, best of 3: 1.40 ms off, 2.02 ms on (+0.62 ms, +45%; C/tables/timing.txt).
- Under cProfile, about a third of it is the per-substep copy and two thirds the one-step reduction. Both are many
  small numpy calls.

## 7. Risks

1. **It pays the judged statistic** (section 5, last row). The policy may meet the band by the cheapest route, which
   may hide the split rather than remove it (inferred). The readouts that the band cannot pay must stay in any screen.
2. **The dose may be too weak in window terms.** The split's window price at w 0.30 is 1.4x what the reward already
   charges on Run 3 8.39M. The stagger term's was 2.6x. The slope is 2-3.7x. w 0.50 is the named alternative.
3. **Calibrated on few panels.** Four stored per-step panels with pad readings, plus nine whole-foot-only checkpoints.
   A from-scratch run will visit states none of them show.
4. **A reward cliff at the share bar.** Guard 1 is a step: a foot crossing 35% jumps to the full price. It never
   fired on stored data, but early training may sit near it (inferred). A smooth ramp (full price at 25%, none at
   35%) is a simple variant if this worries the maintainer.
5. **Layering and fingerprint coupling.** The reward now runs measurement code from gait/. The task fingerprint hashes
   only kwargs. A later change to the recorder's reduction would change the reward with no fingerprint move. The
   bit-identity test pins the two together, but only in this branch.
6. **Throughput.** +45% env step time single-threaded (measured). Inferred: up to that much more Colab wall time for
   12M steps, because the env dominates at n_envs 4.
7. **The single-substep fallback.** A state scored outside step() reads one substep. Only tests and the digest
   snapshot's state probes do that. A stage that turned the band on would see those probe values change.
8. **The pad band leaves a small free zone** (section 4.2).

## 8. What training it would need

1. The maintainer's approval of a reward change. This note is a proposal.
2. A layering decision. Either amend the D-D23 rule ("no species env module imports the gait package") to allow this
   one lazy import, or move the contact reduction out of gait/ into a module both the recorder and the env import. The
   bit-identity test should move with it.
3. A throughput pass. A single fused per-step path would cut most of the +0.62 ms (inferred). It would be new code, so
   the recorder bit-identity test must pin it.
4. Reward plumbing, as for the lean and stagger terms: add reward_stance_cop_band to DiagnosticsCallback.REWARD_KEYS
   and to the reward-decomposition weight map in the same change.
5. A config commit on a screen branch. Recommended base: L+T at 1af2d4d7 plus this term alone, so the stance3 L+T
   runs remain a valid control. A band-plus-stagger arm is the alternative if stance4 lands on line 4 rather than 5.
   Then, measured on that tree:
   - the term-on task_sha256 (and recovery.toml's, which inherits the stance [env]);
   - collapse_peak_floor_reference: the term-on statue (about 4457.5 L+T full-horizon, inferred above);
   - a fresh off-identity check.
6. A fixed criteria file before the first training step, as stance4's: the same K1-K4 and judge settings, a stance5
   judge with the band arm, a control, a credit rule and a decision table. Report-only readouts the band cannot pay:
   - the stagger (foot_stagger_m and its window sd);
   - the ankle servo asymmetry;
   - the pad CoP (K2a);
   - the digit share and digit load position;
   - the per-foot `_counted` fraction (how often a guard fires);
   - the yaw change.
7. Compute: three seeds from scratch, 12M steps, Colab L4 as in stance4, plus the throughput cost above.
8. Optional before any screen: a short local continuation from a failing checkpoint (Run 3 @ 8.39M), as the night
   continuation did for the stagger term. It would show whether the band moves the CoP split and which route it takes
   (ankle or hip). Report only.

## 9. Files

- Design: C/design.md (this file).
- Calibration: C/scripts/size_copband.py -> C/tables/size_copband.txt; C/scripts/digit_geometry.py ->
  C/tables/digit_geometry.txt; C/scripts/statue_band.py -> C/tables/statue_band.txt; C/scripts/timing.py ->
  C/tables/timing.txt.
- Checks: C/scripts/off_identity.py, C/scripts/recorder_identity.py (logs in C/logs), C/scripts/smoke_equality.py,
  C/scripts/probe_states.py, C/scripts/full_suite.sh (C/logs/full.log, full.xml).
- One-off edit script: C/scripts/refactor_recorder.py (the recorder extraction).
- Code: SP/fix/scratch_copband, branch scratch/cop-band, commit f94452ad on c1e146a8. Never pushed.
