# Reward Scale Redesign — Stage 3 Terminal Bonus

> **Date:** 2026-04-18
> **Status:** Analysis — implementation deferred
> **Related:** PR on `claude/review-sac-normalization-FK3SY` (SAC VecNormalize fix),
> [REWARD_DISCREPANCY_INVESTIGATION.md](REWARD_DISCREPANCY_INVESTIGATION.md)
> (precedent: sparse `-100` fall penalty caused bimodal return distribution
> and policy collapse)

## Summary

The Stage 3 terminal bonus (`strike_bonus_weight` / `bite_bonus_weight` /
`food_reach_bonus_weight`, all set to `1000`) is roughly two orders of magnitude
larger than accumulated per-step shaping over a typical episode. This makes the
reward distribution bimodal and non-stationary, which:

1. Forces VecNormalize's reward normalization to chase a moving std — fine for
   PPO (on-policy), harmful for SAC (replay buffer stores rewards at write
   time, so stale samples end up with an inconsistent scale relative to new
   ones).
2. Masks the underlying reward-design smell: the terminal bonus is doing all
   the work and the per-step shaping is almost irrelevant near the moment of
   strike.
3. May be a contributing factor to Brachiosaurus Stage 3 being stuck at 16.7%
   success (see `docs/RL_TRAINING_PLAN.md`). Food-reach is sparser than
   strike/bite and the large terminal gradient is easier to chase on the
   predator species.

The SAC VecNormalize fix (separate PR) handles the immediate replay-buffer
issue. This doc captures the deeper reward-design fix that should follow.

## Current State

### Per-step reward magnitudes

All three species share the same reward architecture (components clipped to
`[-1, 1]` then weighted). Typical per-step range:

- **Stage 1 (Balance):** `~0.1` to `~1.75` (alive bonus dominates); episode
  return `~100–1750`.
- **Stage 2 (Locomotion):** `~0.5` to `~2.0` (forward velocity ≤ 2.0, alive
  0.5, small penalties); episode return `~500–1500`.
- **Stage 3 (Task):** `~0` to `~3.0` (approach shaping, no alive bonus); plus
  a **single +1000 terminal bonus** on success. Episode return `~0–200`
  without strike, `~1000–2000` with strike.

Stage 3 is the only stage with a bimodal return distribution.

### Terminal bonus weights

Files and weights (verified 2026-04-18):

| File | Weight key | Value |
|---|---|---|
| `configs/velociraptor/stage3_strike.toml` | `strike_bonus_weight` | 1000 |
| `configs/trex/stage3_bite.toml` | `bite_bonus_weight` | 1000 |
| `configs/brachiosaurus/stage3_food_reach.toml` | `food_reach_bonus_weight` | 1000 |

### Why the `1000` is load-bearing today

With `norm_reward=True` (PPO default, SAC default before the recent fix),
VecNormalize divides rewards by the running std of discounted returns. Once
the policy starts striking, the running std grows to ~`O(100)`, scaling the
+1000 bonus to an effective `~+10`. This happens to land in a sane range
relative to the per-step shaping — but it's a side effect of normalization,
not an intentional design.

Once reward normalization is off (as it now is for SAC), the +1000 goes to
the critic unmodified. SAC handles this in principle, but:

- Q-target updates spike by ~1000 in a single step, slowing critic
  stabilization.
- SAC's `auto` entropy coefficient targets an entropy level relative to the
  reward scale; a sudden ~1000 makes α chase.
- `clip_reward=50.0` is a no-op when `norm_reward=False` (SB3 skips the clip
  path entirely — see `stable_baselines3/common/vec_env/vec_normalize.py`).

## Proposed Fix

### Principle

Terminal bonus should be comparable in magnitude to accumulated shaping, not
dwarf it. With per-step shaping maxing out at `~3.0` and typical successful
approach runs accumulating `~100–300` over 200–500 steps, a terminal bonus
in the range **`50–150`** keeps the success signal clear (5–50× a single
per-step reward) without producing a Q-target spike.

### Concrete changes

1. **Rescale terminal bonuses** in the three Stage 3 configs:
   - `configs/velociraptor/stage3_strike.toml`: `strike_bonus_weight: 1000 → 75`
   - `configs/trex/stage3_bite.toml`: `bite_bonus_weight: 1000 → 75`
   - `configs/brachiosaurus/stage3_food_reach.toml`:
     `food_reach_bonus_weight: 1000 → 100` (sparser task, slightly larger
     bonus)
   - Specific values are starting points — validate empirically.

2. **Re-derive curriculum gating thresholds** in each file's
   `[curriculum_kwargs]` section. `min_avg_reward` thresholds will need
   proportional adjustment. Quickest path: scale by the same factor as the
   bonus reduction, then validate with a calibration run.

3. **Add a reward-design contract** as a header comment in each species'
   Stage 3 TOML:

   ```toml
   # Reward scale contract:
   # - Per-step components clipped to [-1, 1] then weighted; total per-step
   #   magnitude should stay within ~[-3, +3].
   # - Terminal bonus should be roughly 10-30% of a typical successful
   #   episode return (comparable to accumulated shaping, not dwarfing it).
   # - Exceeding these bounds destabilises SAC's critic and makes the reward
   #   distribution non-stationary for both algorithms. See
   #   docs/REWARD_SCALE_REDESIGN.md.
   ```

4. **Optional cleanup once validated:**
   - Flip `DEFAULT_NORM_REWARD = False` in `environments/shared/constants.py`
     so PPO also runs without reward normalization (no longer needed as a
     safety net). Drop the algorithm-aware branching in `create_vec_env`.
   - Drop the `clip_reward=50.0` knob (no-op when `norm_reward=False`).
   - Simplify `load_vecnorm_stats` docstring (ret_rms discussion becomes
     moot).

### Validation plan

Before merging: one training run per species at Stage 3 with the new bonus
values, SAC only (most sensitive to reward scale):

- Confirm Stage 3 curriculum gate passes within the existing timestep budget
  (may need a `~20%` timestep bump since the gradient pull toward the bonus
  is weaker).
- Confirm SAC entropy coefficient `α` settles to reasonable values (not
  drifting upward indefinitely).
- Confirm success rate is at or above the current baseline (for
  Velociraptor/T-Rex; for Brachio, any improvement over 16.7% is
  informative).

If success rates regress, the bonus was cut too aggressively. Walk it up
toward `150–200` rather than back to `1000`.

## Open Questions for Implementer

1. **Should the bonus be species-specific?** Brachio's food-reach is sparser
   and slower; it may warrant a slightly larger bonus than strike/bite. The
   proposed `75 / 75 / 100` reflects this but is a guess.
2. **Should PPO reward normalization be turned off at the same time?** Or
   leave it on as defense-in-depth until the rescale is validated across
   all species?
3. **How to handle existing trained checkpoints?** The rescale is a breaking
   change to the reward landscape — no cross-boundary resume. New runs
   only. A `CHANGELOG.md` entry is warranted.
4. **W&B / TensorBoard historical comparisons** get a discontinuity at this
   change. Tag the commit clearly so retrospective reward-curve comparisons
   can filter accordingly.

## Risks

- **Stage 3 success rate could regress** if the bonus is too weak relative
  to shaping noise. The empirical validation run is the gate.
- **Brachio Stage 3 might still fail** even after the rescale — the reward
  design isn't the only suspect (see `RL_TRAINING_PLAN.md` for other
  factors). This fix removes one confounding variable.
- **Previous eval rewards become incomparable.** The leaderboards in
  `docs/RL_TRAINING_PLAN.md` will need footnotes or re-runs.

## Why Not Normalize Instead?

VecNormalize's reward normalization masks the magnitude problem without
solving it. Equivalent analysis:

- For PPO, normalization works but is a layer of indirection between the
  reward you design and the reward the policy sees. Debugging reward curves
  in TensorBoard requires mental de-scaling.
- For SAC, normalization is actively harmful (replay buffer staleness,
  entropy-coef chase) — confirmed by SB3's own guidance that off-policy
  algorithms should generally not use VecNormalize reward normalization.
- Observation normalization is a separate and genuinely useful tool (sensor
  channels live on wildly different scales). Keep it on for both algorithms.

The right fix is at the reward source, not the wrapper. This doc describes
that fix.
