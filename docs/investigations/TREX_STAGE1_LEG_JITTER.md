# T-Rex Stage 1 — Jittery Leg Movement

**Status:** root cause identified; one-line config fix applied (pending a validation run).
**Run under review:** `logs/trex/ppo/20260723_204941/stage1` (SB3 PPO, 6.0M steps, 2026-07-24).
**Symptom:** in `trex_ppo_stage1_best.mp4` / `_final.mp4` the legs shake continuously. The
eval videos are rendered with `deterministic=True` (`evaluation.py`), so the jitter is baked
into the learned **mean** policy — it is not exploration noise that only shows up in training.

---

## TL;DR

The jitter is driven by the **exploration/entropy schedule, not the reward shaping.** The PPO
policy std (`algo_std`) is stuck at ~1.0 and never anneals — so every actuator is driven with
~1.0-std corrections and the mean policy is never pushed to be precise or smooth.

Root cause is a one-line config gap: **`configs/trex/stage1_balance.toml` never set
`ent_coef_decay_timesteps`.** When unset, the entropy-coefficient decay defaults to the *full*
6M-step budget (`train_base._maybe_ent_coef_decay_callback`), so `ent_coef` coasts near its
start value for essentially the whole run and only reaches the 0.001 floor at the very end —
too late to let std anneal. Stage 1 is the **only** balance stage missing this anchor:
velociraptor stage 1 and T-Rex stage 3 both set `ent_coef_decay_timesteps = 3000000`.

**Fix:** add `ent_coef_decay_timesteps = 3000000` to T-Rex stage 1. (Applied.)

---

## Quantitative diagnosis (T-Rex stage 1)

Parsed from `diagnostics.npz` (733 diagnostic samples, 612 algo samples over 6.0M steps).

| Signal | First 20% | Last 20% | Reading |
|---|---|---|---|
| `algo_std` (policy std) | 1.012 | **1.038** | rises — never anneals |
| `algo_entropy_loss` | −30.0 | −30.5 | entropy pinned at σ≈1.0 |
| `action_delta` (Σ Δa²/step, 21 act.) | 21.4 | 21.4 | flat for the whole run |
| **RMS action change / joint / step** | 1.010 | **1.010** | "severe" is 0.3–0.5; "smooth" is <0.1 |
| `reward_smoothness` | −0.0255 | −0.0255 | ~1.5% of the alive bonus — negligible |
| `reward_alive` | 1.750 | 1.750 | dominant, action-insensitive |
| eval reward | 479 | 2240 (best 2420 @5.75M) | task **is** learned; then a slight late decline |

The task is learned (survival time ~4× longer, value function explained-variance ~0.90); the
policy just balances *and shakes*. Two reinforcing causes:

1. **Std never anneals.** With `ent_coef` held high for the full budget, the entropy bonus
   (`ent_coef · entropy` ≈ 0.005 · 30 = 0.15 early) dominates the tiny policy-gradient signal
   (~0.03) in a balance task where reward is nearly action-insensitive (alive + height ≈ 2.74
   per step no matter what the legs do). Nothing pushes the policy to become confident, so
   std stays at ~1.0 and injects ~1.0-std noise per actuator per step.
2. **Action-rate penalty is a rounding error.** `reward_action_smoothness` =
   `−w · Σ(Δa)² / (n·4)` = `−0.1 · Σ(Δa)² / 84`. Even a full-scale reversal on all 21 joints
   every step caps at −0.1/step against a +1.75 alive bonus. It cannot shape behavior at any
   realistic magnitude.

---

## The deciding evidence: a 3-stage natural experiment

`ent_coef_decay_timesteps = 3000000` is set in exactly one of the three most recent balance/
locomotion stages. That stage is the only one whose std annealed and whose jitter dropped.

| Metric | T-Rex S1 (no anchor) | Raptor S1 (**anchor=3M**) | Raptor S2 (no anchor) |
|---|---|---|---|
| `smoothness_weight` | 0.1 | 0.05 | 0.05 |
| `energy_penalty_weight` | 0.075 | 0.075 | 0.003 |
| `algo_std` first → last 20% | 1.012 → **1.038** | 1.009 → **0.929** | 1.008 → **1.222** |
| std annealed? | no (Δ +0.026) | **yes (Δ −0.079)** | no (Δ +0.214) |
| RMS action change / joint (last 20%) | 1.010 | **0.925 (lowest)** | 0.979 |
| smoothness penalty / alive | −1.5% | −0.6% | −2.4% |

Read across the row: the T-Rex penalizes action-rate **twice as hard** as the raptor
(`smoothness_weight` 0.1 vs 0.05) yet jitters **more**. Smoothness weight is not the
differentiator. The anchor is: the only stage that annealed std (Raptor S1) is the only one
that got smoother, and it did so at the *lowest* smoothness weight. Both un-anchored stages
kept std high and legs at ~1.0 RMS/joint.

This matches the team's own prior finding for the raptor
(`STAGE2_RECOMMENDATIONS.md`): "action std grew 1.179 → 1.489 under constant `ent_coef`
(entropy still rising at stop)," fixed by decaying entropy and anchoring the decay before the
late drift.

---

## Fix (applied)

`configs/trex/stage1_balance.toml`, `[ppo]`:

```toml
ent_coef_decay_timesteps = 3000000
```

Rationale: entropy reaches its 0.001 floor by ~3M (half the budget), giving the policy ~3M
steps to consolidate at low std — the same schedule that let raptor S1 anneal 1.01 → 0.93.
No other value is changed (single-variable change for clean attribution). This is expected to
both cut the exploration-noise jitter and, by training the policy in a low-noise regime closer
to the deployed deterministic policy, push the mean policy toward smoother control.

## Validation plan (requires a training run — not run here)

1. Re-run T-Rex stage 1 (6M steps, ~3.75h on the L4). **Pass criteria:**
   - `algo_std` anneals to <~0.7 by end of training (vs 1.04 now);
   - `action_delta` last-20% RMS/joint drops materially below 1.0;
   - eval reward/ep-length hold at or above the current run (≥~2200 / ≥~850), no late decline;
   - the eval video shows a visibly steadier stance.
2. If std anneals but the deterministic gait is still not smooth enough, escalate the
   **secondary** lever — the action-rate penalty is currently ~1% of reward. Either raise
   `smoothness_weight` substantially (≥~0.5) or tighten the `n·4` normalization in
   `reward_functions.reward_action_smoothness`. Do this only after the anchor run, so the
   effect is attributable.

## Follow-up: stage 2

T-Rex **stage 2** also omits `ent_coef_decay_timesteps`, and raptor S2 (also un-anchored) shows
the same std blow-up (1.01 → 1.22). When the curriculum advances, stage 2 will likely need the
same anchor. Left out of this change to keep it scoped to the reported stage-1 jitter.
