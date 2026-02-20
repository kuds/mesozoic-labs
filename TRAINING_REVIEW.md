# Velociraptor PPO Training Review

**Run:** `ppo_20260213_194535`
**Date:** 2026-02-13
**Reviewer:** Claude (automated review)
**Total training time:** 3h 38m 47s

---

## Summary

All three curriculum stages underperformed their target thresholds. The root cause is a Stage 1 (balance) failure that cascaded through the curriculum — the raptor never learned to stand, so it could not walk or strike.

| Stage | Eval Reward | Threshold | Status |
|-------|------------|-----------|--------|
| 1 — Balance | -35.13 +/- 12.54 | 50.0 | **Failed** |
| 2 — Locomotion | 2.94 +/- 46.06 | 100.0 | **Failed** |
| 3 — Strike | 118.37 +/- 67.37 | ~500 (1 strike) | **Failed** |

---

## Stage 1: Balance

**Result:** -35.13 +/- 12.54
**Curriculum threshold:** min_avg_reward = 50.0, min_avg_episode_length = 400

With `alive_bonus = 1.0` per step and `max_episode_steps = 500`, a raptor that stands still for the full episode would earn ~500. A mean reward of -35 indicates the raptor is falling almost immediately, with the `fall_penalty = -100` dominating every episode. The low standard deviation (12.54) confirms consistent failure rather than intermittent success.

### Diagnosis

- The initial pose or passive dynamics may be unstable — the raptor may fall even with zero action input
- Joint damping may be too low for the policy to learn corrective actions before toppling
- `posture_weight = 0.3` may penalize exploratory movements needed to discover balance
- `ent_coef = 0.01` may be too low for sufficient exploration given the difficulty of the initial balance problem

### Recommendations

1. Verify passive stability: run `view_model.py` with zero action and confirm the raptor doesn't fall under gravity alone
2. If passively unstable, increase leg joint `damping` and `stiffness` in `raptor.xml`, or adjust `qpos0` to a more stable initial pose
3. Increase `alive_bonus` from 1.0 to 3.0-5.0 to strengthen the survival gradient
4. Reduce `posture_weight` from 0.3 to 0.1 to avoid penalizing useful exploratory tilts
5. Increase `ent_coef` to 0.02-0.05 for broader initial exploration
6. Consider extending the timestep budget to 2M if 1M is insufficient after tuning

---

## Stage 2: Locomotion

**Result:** 2.94 +/- 46.06
**Curriculum threshold:** min_avg_reward = 100.0, min_avg_episode_length = 800

A reward near zero with extremely high variance (coefficient of variation ~15x) indicates no consistent locomotion. The loaded Stage 1 checkpoint had no functional balance policy, so Stage 2 had no foundation to build on.

The variance pattern suggests bimodal behavior: some episodes the raptor stumbles forward before falling (earning small approach/velocity rewards), while most episodes end in immediate falls.

### Recommendations

- Do not retrain until Stage 1 reliably meets its advancement threshold
- Once Stage 1 is fixed, consider setting `smoothness_weight = 0.0` initially so the policy can explore movement patterns freely
- Verify VecNormalize stats from Stage 1 are loaded and compatible with Stage 2's reward scale

---

## Stage 3: Strike

**Result:** 118.37 +/- 67.37
**Expected:** 500+ per episode (one strike = 500 bonus)

The reward of ~118 almost certainly comes from approach shaping alone, not actual strikes. With `strike_approach_weight = 20.0` and prey spawned at 3-8m, closing ~5m before falling yields `20 * 5 = 100` in approach reward — accounting for most of the observed mean. The raptor is stumbling toward prey but not reaching it.

### Recommendations

- This stage requires a working locomotion policy; fix Stages 1-2 first
- When retraining, consider starting prey closer (`prey_distance_range = [1.0, 3.0]`) initially to make first contact achievable, then widen the range
- Address the reward scale mismatch: `strike_bonus = 500` vs `alive_bonus = 0.1` (5000:1 ratio) may cause gradient instability; per-component reward normalization would help

---

## Systemic Issues

### 1. Curriculum did not gate on reward threshold

Training advanced through all 3 stages despite Stage 1 never meeting `min_avg_reward = 50.0`. The curriculum manager should enforce this as a hard gate — advancing on `target_timesteps` alone wastes compute when the policy hasn't converged.

### 2. Reward scale mismatch across stages

Stage 1 rewards are O(1) per step; Stage 3 strike bonus is O(500). `VecNormalize` normalizes total reward only, not per-component, so running statistics shift dramatically between stages and loaded normalization stats become stale.

### 3. Insufficient parallelism

Only 4 parallel environments were used. With `n_steps = 2048`, this yields 8,192 samples per rollout. Increasing to 8-16 parallel envs would provide more diverse experience per update at low cost.

---

## Recommended Action Plan

| Priority | Action | Rationale |
|----------|--------|-----------|
| 1 | Debug MJCF passive stability | Determine if the physics model can balance at all |
| 2 | Tune Stage 1 (damping, alive_bonus, ent_coef) | Root cause of all failures |
| 3 | Enforce hard curriculum gating | Prevent wasted compute on doomed later stages |
| 4 | Increase parallel envs to 8-16 | More diverse rollouts per update |
| 5 | Retrain full curriculum | Only after Stage 1 reliably converges |
| 6 | Add per-component reward normalization | Address scale mismatch for Stage 3 |
