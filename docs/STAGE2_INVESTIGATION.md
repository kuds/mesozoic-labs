# Stage-2 Locomotion Collapse Investigation (run 20260709_185946)

Root-cause analysis of the velociraptor PPO stage-2 failure on 2026-07-09,
with the evidence, the code/plant changes implicated, and a ranked list of
experiments to run next.

## Summary

Run `20260709_185946` used a training config **identical** to four runs that
passed stage 2 in mid-March (same reward weights, warmup/ramp curriculum,
PPO hyperparameters, seed, and net arch — verified against both runs' saved
`stage_config.json`). What changed is the **physics plant**: commit
`156a933` ("bound actuator forces; implicitfast") landed at 18:41 on
July 9 — **18 minutes before the run started** (the Colab notebook clones
`main` at launch). It added `forcerange ≈ 0.8×kp` to every raptor position
actuator and switched the integrator to implicitfast, verifying only
*static settling* ("settle behavior verified unchanged").

Static behavior is stage 1: it passed with the best scores on record
(1747 final eval, all 30 episodes full-length). Dynamic behavior is
stage 2: it collapsed and was early-stopped at 1.1M/8M steps.

## Evidence

### 1. The config is not the regression

| | March runs (4× passed) | July run (failed) |
|---|---|---|
| forward_vel weight / max | 2.0 / 3.0 | 2.0 / 3.0 |
| alive / fall | 0.5 / −50 | 0.5 / −50 |
| warmup / ramp | 300k @ clip 0.02 / 0.2→2.0 over 2M | identical |
| PPO lr / n_steps / batch / γ | 5e-5→1e-5 / 4096 / 256 / 0.995 | identical |
| net_arch / seed / n_envs | [512,256] / 42 / 4 | identical |
| Best stage-2 eval | 2581–2679, 3.1–3.9 m/s | 1068, 0.76 m/s |

(Sources: `stage_config.json` of `ppo_20260315_041632/stage2` vs
`20260709_185946/stage2` on Drive.)

### 2. The plant changed 18 minutes before launch

`156a933` (2026-07-09 18:41) → run start 18:59. Every passing run trained
on the pre-forcerange plant.

### 3. The bounds bite exactly in the dynamic regime

Measured with `environments/velociraptor/scripts/actuator_saturation_report.py`
(2.5 Hz alternating-leg sinusoids at 80 % ctrl amplitude, 2500 steps):

| actuator | kp | forcerange | settle clip | gait clip |
|---|---|---|---|---|
| l/r hip pitch | 150 | ±120 | 0 % | **34–40 %** |
| l/r ankle | 100 | ±80 | 0 % | **22–25 %** |
| everything else | — | — | 0 % | 0 % |

Static: zero clipping (the commit's verification holds). Dynamic: the two
joint groups that drive and catch a running gait spend a third of the
cycle at their force cap. Under identical commands, the bounded vs
unbounded plants diverge by ~4.5 cm RMS pelvis height. Pinned in CI by
`environments/velociraptor/tests/test_actuator_bounds.py`.

### 4. Collapse anatomy matches force-starved gait recovery

From `evaluations.npz`/`diagnostics.npz` of the failed run:

- Clean climb to mean 1040 ± 50 through 450k steps, zero failures.
- From 500k the eval distribution goes **bimodal**: good episodes
  (1000–1500) vs catastrophic early `body_contact` falls (15–150). The
  catastrophic fraction climbs monotonically 1/30 → 25/30; no recovery.
- Training-rollout metrics improve to the very end (reward-hacking
  signature: speed reward on surviving episodes outweighs cheap falls).
- PPO internals healthy throughout (approx_kl ~0.015, explained_var ~0.9);
  policy action std grew 1.18 → 1.49 under constant ent_coef 0.005.
- Peak speed 0.76 m/s vs 3.5+ m/s on the old plant — with hips clipped
  34–40 % of a moderate gait cycle, the **2.0 m/s curriculum gate may not
  be reachable on the current plant at all**.

## Secondary factor: fall penalty is cheap at γ=0.995

Independent of the plant, the reward math tolerates fragile-fast gaits:
pushing +0.5 m/s earns ≈ `2.0 × 0.5/3.0 = 0.33`/step, ~60 discounted over
300 steps, while a fall 300 steps ahead costs only `50 × 0.995³⁰⁰ ≈ 11`
discounted. Break-even needs fall_penalty ≈ −250 or a higher alive bonus.
The old plant masked this because robust fast gaits existed; the new plant
exposes it.

## Hardening already merged (this branch)

- **Risk-adjusted checkpointing** — `RobustBestModelCallback` saves
  `robust_best_model.zip` by `mean − std` eval score; next-stage loading
  and notebook gating prefer it. On the failed run this would have
  selected the ~450k zero-failure checkpoint instead of the already
  bimodal 800k "peak".
- **Entropy decay** — set `ent_coef_end` (and optionally
  `ent_coef_decay_timesteps`) under `[ppo]` in a stage TOML to linearly
  decay the entropy bonus (`EntCoefDecayCallback`).
- Actual-timesteps reporting, atomic npz artifacts, and the curriculum
  gate's `evaluations.npz` path fix (PR #432).

## Experiments to run next (ranked)

**A. Plant A/B — highest information, run first.**
Rerun stage 2 from the existing stage-1 best checkpoint on:
  1. current plant (control for the retry), and
  2. the pre-`156a933` raptor (`git revert 156a933 --
     environments/velociraptor/assets/raptor.xml`, or raise leg
     `forcerange` to ≥1.5×kp which measures ~0 % gait clipping).
If (2) passes like March and (1) collapses again, the plant is confirmed
as the regression and stage-2 tuning moves to experiment B; if both fail,
suspect the June/July training-code changes instead.

**B. Retune for the bounded plant (if keeping it — it is more realistic).**
- Raise leg actuator headroom: hips/ankles `forcerange` 0.8×kp → 1.2–1.5×kp
  (the 0.8 sizing was calibrated to clip "impact/reset spikes", but it
  clips gait torques), **or** raise kp with forcerange scaled to match.
- Re-measure with `actuator_saturation_report.py` targeting <10 % gait
  clipping on hips/ankles before spending GPU hours.
- Consider whether the `min_avg_forward_vel = 2.0` gate is physically
  reachable; recalibrate against a scripted-gait top-speed measurement.

**C. Reward/entropy tuning (cheap add-ons to any rerun).**
- `forward_vel_max = 2.5` (from 3.0): saturates the speed incentive just
  past the 2.0 m/s gate instead of paying for robustness-destroying speed.
- `fall_penalty = -150` to `-250` (from −50): break-even math above.
- `[ppo] ent_coef_end = 0.001`: stop paying for exploration noise late in
  the stage (action std grew 26 % during the collapse).
- Keep `RobustBestModelCallback` defaults (risk_coef 1.0).

**D. If A/B/C all fail:** bisect the June/July training-code changes
(`4a01162`, `c97ace8`) against the March behavior — VecNormalize, warmup,
and curriculum internals changed in that window.
