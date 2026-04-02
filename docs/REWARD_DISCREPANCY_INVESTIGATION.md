# T-Rex Stage 1 Reward Discrepancy Investigation

**Date:** 2026-04-02
**Issue:** Best eval reward (2410.69) appears inconsistent with short episode lengths (196 steps / 1.96s)

## Root Cause

The high reward and the short episodes are **from different checkpoints**. The summary
conflates two evaluation points:

| Metric | Value | Source |
|--------|-------|--------|
| Best eval reward | 2410.69 | Checkpoint at 23M steps (~update 175) |
| Final eval reward | 17.26 +/- 7.76 | End of training (32.6M steps) |
| Avg ep length | 196.4 steps (1.96s) | End of training (final eval) |

The eval rollout (195 frames, 3.9s showing T-Rex collapsing) is from the **final model**,
not the best checkpoint. The best model at 23M steps likely had episodes of ~500+ steps
at ~4-5 reward/step, consistent with 2410.69 total return.

## Policy Collapse Evidence

The reward diagnostic charts show clear policy degradation after update ~175:

1. **Height reward** peaked ~1.7 at update 100-150, then dropped to ~0.5 by update 240
2. **pelvis_z** declined from ~0.85m to ~0.65m (well below target 0.90m)
3. **forward_z** went negative after update 200 — T-Rex nosediving forward
4. **Alive reward** dropped from ~1.0 to near 0 — early termination from falls
5. **Posture penalty** became increasingly negative after update 150

## Per-Step Reward Budget (Stage 1 Balance)

With ideal standing posture, the maximum per-step reward is:

| Component | Weight | Max per step | Notes |
|-----------|--------|-------------|-------|
| height | 2.0 | +2.0 | `clip((pelvis_z - 0.75) / 0.15, 0, 1) * 2.0` |
| alive_bonus | 1.2 | +1.2 | Gated on height fraction * foot contact |
| foot_contact | 0.8 | +0.8 | Binary: any foot on ground |
| energy | -0.075 | ~-0.05 | `-0.075 * mean(action^2)` |
| posture | -2.5 | 0 to -2.5 | Quadratic tilt penalty |
| nosedive | -4.0 | 0 to -4.0 | Forward pitch penalty |
| drift | -0.3 | 0 to -0.3 | Displacement penalty |
| speed | -0.3 | 0 to -0.3 | Speed above 0.1 m/s |
| spin | -0.1 | 0 to -0.1 | Angular velocity penalty |
| smoothness | -0.1 | 0 to -0.1 | Action jerk penalty |
| **Ideal total** | | **~3.8-4.0** | With minimal penalties |

Best eval: 2410.69 / ~4.0 per step = ~600 steps, consistent with mid-training performance.
Final eval: 17.26 / 196 steps = ~0.09 per step (mostly penalties + fall penalty of -50).

## Contributing Factors

1. **No early stopping on performance degradation**: Training continued for all 500 updates
   even after performance peaked at ~update 175 and began collapsing.

2. **Value function divergence**: The height reward (weight=2.0) dominates and creates
   high variance in value targets. With gamma=0.98, value overestimation may drive
   increasingly aggressive updates that destabilize the policy.

3. **Entropy collapse**: ent_coef=0.005 is quite low. Once the policy starts leaning
   forward, it may lack the exploration capacity to recover an upright stance.

4. **Nosedive cascade**: As forward_z decreases (forward lean), the nosedive penalty
   (-4.0 weight) becomes a strong gradient toward "don't lean forward", but the policy
   has already committed to a forward-leaning stance. This creates conflicting gradients
   that destabilize learning.

## Curriculum Gate Status

The stage 1 curriculum requirements were **NOT met**:

- `min_avg_reward = 100.0` — final: 17.26 (FAIL)
- `min_avg_episode_length = 750` — final: 196.4 (FAIL)
- `required_consecutive = 3` — never achieved

The best checkpoint likely met `min_avg_reward` but may not have sustained
`min_avg_episode_length >= 750` for 3 consecutive evaluations.

## Recommendations

1. **Use the best checkpoint for downstream stages**: The best model at 23M steps
   (saved as `best_model.zip`) should be used for stage 2, not the final model.

2. **Add early stopping on degradation**: Stop training if mean reward drops below
   e.g. 50% of the best observed reward for N consecutive evaluations.

3. **Reduce num_updates or add KL-based stopping**: 500 updates may be too many.
   The target_kl=0.05 may not be aggressive enough to prevent late-stage collapse.

4. **Consider learning rate warmup + cosine decay**: Linear decay to 1e-5 may not
   drop fast enough in the late phase where the policy is fragile.

5. **Increase entropy coefficient slightly**: Try ent_coef=0.01 to maintain
   exploration capacity throughout training.
