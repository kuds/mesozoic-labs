# Composite ASHA Metric for Balance Reward Discrimination

> **Date:** 2026-03-16
> **Status:** Proposed
> **Scope:** Stage 1 (Balance) hyperparameter sweeps via Ray Tune

---

## Problem

ASHA (Async Successive Halving Algorithm) currently uses `best_mean_reward` as
its sole pruning metric. This works well in the early phase of a sweep when most
trials fail to balance — reward is spread out and ASHA can aggressively prune
non-learners.

Once a majority of trials solve balance (reward saturates at ~300+), the metric
loses discriminative power. ASHA can no longer distinguish a sloppy balancer
(reward 310, wobbly, inconsistent episode lengths) from a clean one (reward 350,
stable, full-length episodes). Pruning decisions among survivors become
effectively random.

This is the core inefficiency in our current Stage 1 sweeps: we spend compute
budget on mediocre survivors that crowd out exploration of better hyperparameter
regions.

---

## Current Architecture

```
┌──────────────────┐     ┌──────────────────┐
│  Search Algorithm │     │   ASHA Scheduler │
│  (Random Search)  │     │                  │
│                   │     │  metric:         │
│  Draws 50 configs │────▶│  best_mean_reward│
│  from search space│     │  mode: max       │
│                   │     │  grace: 20       │
└──────────────────┘     │  reduction: 2    │
                          └──────────────────┘
```

**Metrics currently reported per evaluation checkpoint:**

| Metric | Source | Description |
|--------|--------|-------------|
| `best_mean_reward` | `EvalCallback.best_mean_reward` | Best mean reward seen so far (30 eval episodes) |
| `last_mean_reward` | `EvalCallback.last_mean_reward` | Most recent eval mean reward |
| `timesteps` | `model.num_timesteps` | Training steps completed |

**Data available but not reported:**

| Data | Source | Description |
|------|--------|-------------|
| Per-episode rewards | `EvalCallback.evaluations_results[-1]` | Array of 30 episode rewards |
| Per-episode lengths | `EvalCallback.evaluations_length[-1]` | Array of 30 episode lengths |
| Max episode steps | Environment spec | Upper bound on episode length |

---

## Proposed Solution

### Composite Metric

Replace the scalar `best_mean_reward` with a composite score that blends reward
and a **utilization ratio** — the fraction of each episode the agent spends
alive and upright:

```
utilization_ratio = mean_episode_length / max_episode_steps
composite_score   = mean_reward + α × utilization_ratio
```

Where:
- `mean_episode_length` is averaged over 30 eval episodes
- `max_episode_steps` is the environment's hard time limit (e.g. 1000)
- `α` (alpha) is a scaling constant that controls blending

### Why Utilization Ratio

Episode length is a direct proxy for balance quality that **remains
discriminative after reward saturates**:

| Training Phase | `best_mean_reward` | `utilization_ratio` | ASHA Signal |
|---|---|---|---|
| Early (nothing balances) | Spread (0–100) | All low (~0.1) | Reward dominates — prunes non-learners |
| Mid (some balance) | Bimodal (50 vs 300) | Bimodal (0.3 vs 0.8) | Both contribute — strong signal |
| Late (most balance) | **Saturated (~300+)** | **Still spread (0.7–0.99)** | Utilization discriminates — prunes sloppy balancers |

Unlike reward, utilization is bounded [0, 1], interpretable, and doesn't depend
on reward shaping details.

### Alpha Selection

`best_mean_reward` is O(300) for Stage 1 balance. `utilization_ratio` is [0, 1].
To make utilization contribute meaningfully without dominating in the early phase:

- **α = 50**: Utilization contributes up to ~50 points. A trial with reward 280
  and utilization 0.95 scores 327.5, beating a trial with reward 310 and
  utilization 0.70 (score 345). Reward still dominates early.
- **α = 100**: Stronger discrimination. A perfect utilization (1.0) is worth 100
  points — roughly a third of the reward signal. More aggressive at promoting
  clean balancers.

**Recommendation:** Start with `α = 50` and review the composite score
distribution after one sweep.

An alternative is to normalize reward to [0, 1] and weight both equally, but
this requires knowing the reward range a priori, which varies across species and
reward configurations.

---

## Implementation

### Changes Required

**1. `RayTuneReportCallback._on_step()` in `environments/shared/scripts/sweep/ray_tune.py`**

Add utilization computation and composite score to the reported metrics:

```python
import numpy as np

# After confirming a new eval occurred...
last_lengths = self.eval_callback.evaluations_length[-1]  # (30,)
mean_length = float(np.mean(last_lengths))

# Get max episode steps from the eval environment
max_steps = self.eval_callback.env.envs[0].spec.max_episode_steps
utilization = mean_length / max_steps if max_steps else 0.0

mean_reward = float(self.eval_callback.last_mean_reward)
composite = mean_reward + COMPOSITE_ALPHA * utilization

tune.report(
    {
        "best_mean_reward": float(self.eval_callback.best_mean_reward),
        "last_mean_reward": mean_reward,
        "utilization_ratio": utilization,
        "composite_score": composite,
        "timesteps": self.num_timesteps,
    },
    checkpoint=checkpoint,
)
```

**2. ASHA scheduler configuration in `notebooks/ray_tune_sweep.ipynb`**

```python
COMPOSITE_ALPHA = 50  # Configurable in the notebook header

scheduler = ASHAScheduler(
    metric="composite_score",   # was: "best_mean_reward"
    mode="max",
    max_t=max_reports,
    grace_period=GRACE_PERIOD,
    reduction_factor=REDUCTION_FACTOR,
)
```

**3. `DriveProgressLogCallback` in `notebooks/ray_tune_sweep.ipynb`**

Add `utilization_ratio` and `composite_score` to the progress CSV so we can
analyze the metric distribution post-sweep.

### Backward Compatibility

- `best_mean_reward` and `last_mean_reward` continue to be reported, so existing
  analysis notebooks and result collection (`collect_ray_results()`) work
  unchanged.
- The composite metric is additive — reverting to the old behavior is a one-line
  change to the scheduler config.
- `COMPOSITE_ALPHA` is a notebook-level constant, trivially adjustable between
  sweeps.

---

## Relationship to Search Algorithm Improvements

This proposal addresses **ASHA's pruning quality** — given a set of trials, how
well can ASHA decide which to keep. It does not address **sampling quality** —
how well the search algorithm proposes new configurations.

These are complementary:

| Component | Current | Proposed Improvement |
|-----------|---------|---------------------|
| **Scheduler (pruning)** | ASHA on `best_mean_reward` | ASHA on `composite_score` (this doc) |
| **Search algorithm (sampling)** | Random search | Optuna/TPE (future work) |

The composite metric is simpler to implement (callback + scheduler arg change)
and provides immediate benefit. Adding a Bayesian search algorithm like Optuna
is a logical next step but introduces interaction effects with ASHA's early
stopping that need careful handling (pruned trials give noisy signal to the
surrogate model).

**Recommended order:**
1. Implement composite metric (this proposal)
2. Run one sweep and validate the metric distribution
3. Evaluate whether Optuna is still needed or if better pruning alone suffices

---

## Validation Plan

After one sweep with the composite metric:

1. **Plot composite score distribution** at each ASHA rung to confirm it remains
   spread out (not saturated like raw reward)
2. **Compare pruning decisions** to the old metric — did ASHA prune trials that
   the old metric would have kept (and vice versa)?
3. **Check top-5 trials by composite** vs top-5 by raw reward — are the
   composite winners qualitatively better balancers?
4. **Review α sensitivity** — plot composite rank vs raw reward rank to see if α
   is too aggressive or too weak

---

## Open Questions

- **Should the composite track `best` or `last`?** Currently using `last_mean_reward`
  (most recent eval) rather than `best_mean_reward` (historical max). Using `last`
  makes the composite responsive to regression (agent quality degrading), which
  is arguably what we want ASHA to penalize. Using `best` would be more forgiving
  of transient dips.
- **Per-stage alpha values?** Stage 2/3 rewards are on different scales. If we
  reuse this composite for later stages, α will need stage-specific tuning.
- **Reward variance as an additional signal?** `np.std(last_rewards)` over the 30
  eval episodes captures consistency. A trial with mean reward 300 ± 150 is worse
  than 300 ± 10. This could be a third component but adds complexity.
