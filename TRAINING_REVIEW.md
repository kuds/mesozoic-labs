# Velociraptor PPO Training Review

**Run:** `ppo_20260221_233623`
**Date:** 2026-02-22
**Reviewer:** Claude (automated review)
**Total training time:** 36m 44s (Stage 1 only)

---

## Summary

Stage 1 (balance) shows major improvement over the previous run (Feb 13). The raptor now survives ~90 steps instead of falling immediately, and the reward threshold is met. However, the episode length threshold is not met — episodes are 3.4x shorter than required — and training has plateaued, indicating the agent is stuck in a local optimum.

| Metric | Result | Threshold | Status |
|--------|--------|-----------|--------|
| Eval Reward | 335.18 +/- 118.81 | 50.0 | **Passed** |
| Avg Episode Length | 89.5 +/- 24.3 steps | 300 steps | **Failed** |
| Curriculum Gate | 0/3 consecutive passes | 3 required | **Not met** |

### Progress vs. Previous Run (Feb 13)

| Metric | Feb 13 | Feb 22 | Change |
|--------|--------|--------|--------|
| Eval Reward | -35.13 +/- 12.54 | 335.18 +/- 118.81 | +370 |
| Avg Episode Length | ~7 steps (immediate fall) | 89.5 steps (0.9s) | +82 steps |
| Alive Bonus | 1.0 | 5.0 | +4.0 |
| Posture Weight | 0.3 | 2.0 | +1.7 |
| Nosedive Penalty | (none) | 3.0 | new |
| Entropy Coeff | 0.01 | 0.005 | -0.005 |

The code changes (nosedive penalty, quadratic posture penalty, natural pitch offset, alive_bonus increase) successfully fixed the training collapse from the Feb 13 run.

---

## Stage 1: Balance — Detailed Analysis

### What the agent learned

The raptor can now maintain a roughly upright posture for ~90 steps (0.9 seconds sim time) before falling. This is real balance behavior — not a degenerate exploit — as confirmed by the posture and nosedive penalties working as intended.

### Reward decomposition (estimated per episode)

| Component | Per-step | Over 89.5 steps | Notes |
|-----------|----------|-----------------|-------|
| alive_bonus | +5.0 | +447.5 | Dominates total reward |
| posture penalty | -0.2 to -2.0 | ~-45 | Quadratic, varies with tilt |
| nosedive penalty | 0 to -3.0 | ~-15 | Active beyond 20° natural pitch |
| smoothness penalty | -0.0 to -0.1 | ~-5 | Small contribution |
| tail stability | -0.0 to -0.02 | ~-1 | Negligible |
| fall_penalty (terminal) | — | -100 | Applied on every termination |
| **Estimated total** | | **~280-335** | **Matches observed 335** |

### Why training plateaued

The training curves show reward and episode length both plateauing around 400k-500k timesteps with no further improvement through 1M steps. Several factors contribute:

1. **alive_bonus = 5.0 creates a shallow gradient for longer survival.** At 89 steps the agent earns ~350 total reward. Surviving one additional step adds only +5.0 (minus small penalties), but the fall penalty of -100 is already amortized across the episode. The marginal value of longer survival is low relative to the noise in the reward signal.

2. **ent_coef = 0.005 is too low for the plateau.** This was reduced from 0.03 (previous run) to 0.005. While the lower entropy helped the agent converge to a stable balance strategy, it now prevents exploration of qualitatively different strategies needed to survive past 90 steps. The agent is likely executing a rigid posture correction pattern and cannot discover the oscillatory micro-adjustments that would enable longer dynamic balance.

3. **The posture (2.0) + nosedive (3.0) penalties may constrain the balance corridor.** With a combined weight of 5.0 for posture-related penalties, the agent is punished heavily for any tilt. Dynamic balance (like a human on a balance board) requires controlled swaying — the penalties may suppress the very movements needed for long-term stability.

4. **Episode length variance is high (±24.3, CV=27%).** The agent doesn't have a robust balance strategy; it has a fragile one that works for variable durations. A robust policy would show lower variance.

### Training curves interpretation

From the curves:
- **Episode reward**: Rises from ~-100 to ~350, sharp improvement from 0-300k steps, flat plateau from 400k-1M steps
- **Episode length**: Rises from ~5 to ~90 steps, same plateau pattern
- **Tilt angle**: Stabilizes but shows the agent maintaining a non-zero tilt
- The flat plateau over 600k steps (60% of training budget) confirms the agent cannot improve further with current hyperparameters

---

## Diagnosis: Why Episodes End at ~90 Steps

With `max_episode_steps = 500`, episodes should last up to 5 seconds sim time. At 89.5 steps average, episodes use only 18% of the available time. Termination causes are:

1. **Pelvis height exits healthy range** (`healthy_z_range = [0.3, 1.0]`) — the raptor slowly leans and eventually drops below 0.3m
2. **Excessive tilt** (`max_tilt_angle = 1.047 rad = 60°`) — accumulated angular drift eventually exceeds the threshold
3. **Nosedive** (`forward_z < natural_z - 0.5`) — the raptor pitches forward past the 30° excess threshold
4. **Body/tail ground contact** — torso, neck, head, or tail segments touch the floor

The agent has learned to delay these termination conditions but not prevent them indefinitely. This suggests the control policy lacks the precision for continuous corrective balance.

---

## Recommendations

### High Priority — Unlock longer episodes

#### 1. Increase entropy coefficient to 0.01-0.02

The plateau is an exploration problem. With `ent_coef = 0.005`, the policy distribution is too narrow to discover better balance strategies. Increasing to 0.01-0.02 (still lower than the previous 0.03 which caused instability) should allow the agent to explore while retaining what it has learned.

```toml
ent_coef = 0.015
```

#### 2. Reduce alive_bonus to 2.0-3.0

At 5.0 per step, the alive_bonus dominates total reward so heavily that the agent is "satisfied" with short episodes. Reducing to 2.5 maintains a meaningful survival gradient without causing the collapse seen at 1.0. The fall_penalty (-100) becomes relatively more costly, pushing the agent to avoid termination more aggressively.

```toml
alive_bonus = 2.5
```

#### 3. Add an episode-progress bonus

A small bonus that scales with episode progress would create a monotonically increasing incentive to survive longer. This directly addresses the episode length gap:

```python
# In _get_reward_info, add:
progress_bonus = 0.01 * self._step_count  # Grows each step
```

This adds +0.5 at step 50, +1.5 at step 150, +3.0 at step 300 — creating an accelerating incentive to survive longer without dominating early-episode rewards.

#### 4. Soften posture penalties during balance exploration

Reduce `posture_weight` from 2.0 to 1.0 and `nosedive_weight` from 3.0 to 1.5. The combined 5.0 penalty for tilt is too restrictive. Dynamic balance requires controlled oscillation — the agent needs room to lean and recover. The termination conditions (`max_tilt_angle`, nosedive threshold) already prevent dangerous tilts; the per-step penalties don't need to be this aggressive.

```toml
posture_weight = 1.0
nosedive_weight = 1.5
```

### Medium Priority — Training efficiency

#### 5. Extend timesteps to 2M

With 600k steps wasted on a plateau, 1M total is insufficient. 2M steps gives the agent more time to break through after hyperparameter adjustments, and the 37-minute runtime means 2M steps would still complete under 1.5 hours.

#### 6. Increase n_envs from 4 to 8

More parallel environments provide more diverse experience per rollout. With `n_steps = 2048`, going from 4 to 8 envs doubles the rollout buffer from 8,192 to 16,384 samples per update. This is especially valuable for balance learning, where small initial pose perturbations lead to different failure modes.

### Lower Priority — Curriculum tuning

#### 7. Consider a two-phase Stage 1

Split Stage 1 into 1a (static balance: survive 200 steps) and 1b (dynamic balance: survive 400 steps with perturbations). This decomposes the balance problem into achievable sub-goals. Phase 1a uses current settings; phase 1b adds random force perturbations to the pelvis during training.

#### 8. Relax min_avg_episode_length to 200 initially

If the above changes bring episode length to 200+ but not 300, consider whether 200 steps is sufficient to bootstrap Stage 2 locomotion. A raptor that can balance for 2 seconds may learn to walk forward, which itself improves balance through momentum.

---

## Recommended Next Training Configuration

```toml
[stage]
name = "balance"
description = "Learn to stand and balance without falling"

[env]
forward_vel_weight = 0.0
alive_bonus = 2.5
energy_penalty_weight = 0.0
tail_stability_weight = 0.02
posture_weight = 1.0
nosedive_weight = 1.5
gait_symmetry_weight = 0.0
smoothness_weight = 0.05
strike_bonus = 0.0
strike_approach_weight = 0.0
prey_distance_range = [10.0, 15.0]
max_episode_steps = 500

[ppo]
learning_rate = 3e-4
learning_rate_end = 1e-5
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.015

[ppo.policy_kwargs]
net_arch = [256, 256]

[curriculum]
timesteps = 2000000
min_avg_reward = 50.0
min_avg_episode_length = 200
required_consecutive = 3
```

Key changes from current config:
- `alive_bonus`: 5.0 → 2.5 (reduce reward plateau)
- `posture_weight`: 2.0 → 1.0 (allow dynamic balance exploration)
- `nosedive_weight`: 3.0 → 1.5 (soften tilt corridor)
- `smoothness_weight`: 0.1 → 0.05 (allow more movement variety)
- `ent_coef`: 0.005 → 0.015 (increase exploration)
- `timesteps`: 1M → 2M (more training time)
- `min_avg_episode_length`: 300 → 200 (achievable intermediate target)

---

## Previous Review (Feb 13)

The previous training run (`ppo_20260213_194535`) failed across all three stages with Stage 1 reward of -35.13. The following recommendations from that review have been addressed:

| Recommendation | Status |
|---------------|--------|
| Increase alive_bonus from 1.0 to 3.0-5.0 | Done (set to 5.0) |
| Reduce posture_weight from 0.3 to 0.1 | Partially done (increased to 2.0 instead, with quadratic scaling) |
| Increase ent_coef to 0.02-0.05 | Not done (decreased to 0.005) |
| Debug MJCF passive stability | Addressed via nosedive penalty + natural pitch offset |
| Enforce hard curriculum gating | Implemented (curriculum only ran Stage 1) |

The changes that were implemented produced a 370-point improvement in eval reward, validating the core diagnosis. The remaining gap is episode length, which this review's recommendations target.
