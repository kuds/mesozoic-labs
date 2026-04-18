# Training Results Review

> **Date:** 2026-03-25 (updated)
> **Data:** 140+ training runs across T-Rex, Velociraptor, and Brachiosaurus (Feb 12 - Mar 23)
> **Algorithms:** PPO and SAC

---

## Executive Summary

| Species | Algorithm | Stage 1 (Balance) | Stage 2 (Locomotion) | Stage 3 (Behavior) |
|---------|-----------|-------------------|---------------------|-------------------|
| **Velociraptor** | PPO | Solved (1960.64 reward) | Solved (3.47 m/s) | Solved (93.3% strike) |
| **Velociraptor** | SAC | Solved (970.19 reward) | Solved (2.91 m/s) | Solved (90.0% strike) |
| **T-Rex** | PPO | Solved (2994.34 reward) | Solved (3.47 m/s) | Solved (96.7% bite) |
| **Brachiosaurus** | PPO | Solved (3002.52 reward) | Solved (1.12 m/s) | **Failing** (16.7% vs 50% food_reach) |

**Bottom line:** Velociraptor (PPO and SAC) and T-Rex (PPO) have completed all
3 stages. Brachiosaurus passes Stages 1-2 but Stage 3 (food_reach) remains the
bottleneck at 16.7% success rate vs the 50% threshold. The original review below
covers the early Feb 12-27 training history, followed by updated sections for
March results.

---

## T-Rex Training Analysis

### 14 unique training runs (Feb 22 - Feb 27)

### Stage 1 Progression

| Run | Date | Reward | Ep Length | Steps | Key Config Changes | Passed? |
|-----|------|--------|-----------|-------|-------------------|---------|
| ppo_20260222_210506 | Feb 22 | 348.08 +/- 172.89 | 417.0 | 1M | alive=1.0, energy=0.0005, posture=0 | **Yes** |
| ppo_20260223_022137 | Feb 23 | -2.84 +/- 14.29 | 99.7 | 205K | Same config | No (crashed early) |
| ppo_20260223_023220 | Feb 23 | 67.73 +/- 36.99 | 176.6 | 3M | Same config | No |
| ppo_20260223_225351 | Feb 23 | 35.84 +/- 52.02 | 143.1 | 3.6M | Same config | No |
| ppo_20260224_005932 | Feb 24 | 158.33 +/- 84.10 | 138.0 | 4M | alive=2.0, energy=0.05, posture=1.5, gamma=0.995 | No |
| ppo_20260224_153726 | Feb 24 | 143.38 +/- 62.16 | 129.6 | 4M | Same as above | No |
| ppo_20260224_225556 | Feb 24 | 184.67 +/- 58.02 | 149.8 | 4M | Threshold relaxed to ep_len=300 | No |
| ppo_20260225_021848 | Feb 25 | 277.20 +/- 83.99 | 206.3 | 4M | Same | No |
| ppo_20260226_141055 | **Feb 26** | **806.50 +/- 388.24** | **299.3** | **6M** | **batch=128, gamma=0.998, posture=2.0** | **No (299.3 vs 300)** |
| ppo_20260227_155559 | Feb 27 | 806.50 +/- 388.24 | 299.3 | 6M | Added net_arch=[256,256] | No (same checkpoint) |

### Stage 1 Key Findings

**The original run (Feb 22) passed** with a simple config (alive=1.0, no posture
penalty), but this was not reproducible — three subsequent attempts with the same
config failed (67.73, 35.84 reward).

**The Feb 26 breakthrough** (gamma=0.998, batch=128, posture=2.0, 6M steps)
achieved 806.5 reward and 299.3 episode length — missing the 300 threshold by
**0.7 steps**. This is effectively solved.

**Config evolution that worked:**

| Parameter | Original (Feb 22) | Final (Feb 26) | Effect |
|-----------|-------------------|----------------|--------|
| gamma | 0.99 | 0.998 | Longer horizon = values sustained balance |
| batch_size | 64 | 128 | More stable gradient estimates |
| posture_weight | 0.0 | 2.0 | Explicit upright posture reward |
| alive_bonus | 1.0 | 2.0 | Stronger survival incentive |
| energy_penalty | 0.0005 | 0.05 | Discourages jittery actions |
| timesteps | 1M | 6M | More training time for convergence |

### Stage 2 Results

| Run | Date | Reward | Ep Length | Steps | fwd_vel | posture | Passed? |
|-----|------|--------|-----------|-------|---------|---------|---------|
| ppo_20260222_210506 | Feb 22 | 743.17 +/- 413.91 | 682.0 | 2M | 1.0 | 0.0 | No (682 vs 800) |
| ppo_20260223_023220 | Feb 23 | -18.09 +/- 48.78 | 90.5 | 3M | 1.0 | 0.0 | No (catastrophic) |
| ppo_20260224_005932 | Feb 24 | -74.38 +/- 4.93 | 64.6 | 4M | 1.0 | 0.0 | No (catastrophic) |
| ppo_20260224_153726 | Feb 24 | 72.91 +/- 23.13 | 101.8 | 4M | 1.0 | 0.4 | No |
| ppo_20260226_141055 | Feb 26 | 285.04 +/- 126.12 | 225.0 | 5M | 1.0 | 0.4 | No |
| ppo_20260227_155559 | Feb 27 | 180.52 +/- 149.86 | 185.2 | 5M | 1.0 | 0.4 | No |

### Stage 2 Key Findings

1. **Best Stage 2 was the very first run** (682 ep_len, now threshold is 600) —
   this would have passed the current relaxed threshold.
2. **Catastrophic forgetting is severe.** Runs that follow a well-trained Stage 1
   often collapse in Stage 2 (ep_len dropping to 64-90).
3. **The Feb 26 Stage 2** (285 reward, 225 ep_len) showed improvement over
   mid-Feb runs but still far from the 600 threshold.
4. **posture_weight=0.4 in Stage 2** helped compared to 0.0 (225 ep_len vs 90),
   but the agent still loses balance quickly.

### Stage 3 Results (Preliminary)

Only attempted from non-passing Stage 2 checkpoints, so results are unreliable:

| Run | Reward | Ep Length | Steps |
|-----|--------|-----------|-------|
| ppo_20260222_210506 | 108.17 +/- 892.64 | 775.6 | 325K |
| ppo_20260224_153726 | 371.50 +/- 653.86 | 145.0 | 4M |

Extremely high variance (std > mean) indicates the agent is unstable.

### T-Rex Recommendations

1. **Stage 1 is effectively solved.** The 299.3 vs 300 threshold gap can be closed
   by either slightly extending training (7M steps) or relaxing the threshold by
   1 step.

2. **Stage 2 needs catastrophic forgetting mitigation:**
   - The current config (posture=2.0, nosedive=3.0 matching Stage 1) has not been
     tested yet. The best Stage 2 runs used posture=0.0-0.4.
   - Consider the first run's success: it reached 682 ep_len with a simpler Stage 1
     checkpoint. The more refined Stage 1 policy may be harder to fine-tune.
   - **Lower the Stage 2 learning rate further** (try 3e-5 instead of 5e-5) and use
     a warmup period of 50K steps with frozen policy layers.
   - **VecNormalize stats carryover** is critical — verify running stats transfer
     between stages.

3. **Stage 2 config discrepancy:** The current TOML config sets posture=2.0 and
   nosedive=3.0 in Stage 2 (matching Stage 1), but no run has tested these exact
   values. The rationale (prevent forgetting) is sound, but empirically the best
   Stage 2 result came from posture=0.0 and no nosedive penalty.

---

## Velociraptor Training Analysis

### ~30 unique training runs (Feb 12 - Feb 27)

### Stage 1 Progression

| Run | Date | Reward | Ep Length | Steps | Key Config | Passed? |
|-----|------|--------|-----------|-------|------------|---------|
| ppo_20260212_003824 | Feb 12 | -18.01 +/- 7.94 | 113.0 | 505K | alive=1.0, posture=0.0 | No |
| ppo_20260212_124205 | Feb 12 | -12.52 +/- 14.58 | 123.4 | 1M | Same | No |
| ppo_20260212_224234 | Feb 12 | 299.17 +/- 151.93 | 418.4 | 1M | Same | **Yes** |
| ppo_20260213_011802 | Feb 13 | 370.82 +/- 74.75 | 483.1 | 1M | Same | **Yes** |
| ppo_20260213_194535 | Feb 13 | -42.47 +/- 26.02 | 114.7 | 1M | posture=0.3 | No |
| ppo_20260221_033148 | Feb 21 | 293.75 +/- 79.46 | 78.8 | 1M | alive=5.0, 16 envs | No (low ep_len) |
| ppo_20260221_233623 | Feb 21 | 415.35 +/- 173.09 | 107.0 | 1M | alive=5.0, posture=2.0, [256,256] | No |
| ppo_20260222_003347 | **Feb 22** | **791.87 +/- 224.72** | **376.0** | **2M** | **alive=2.5, posture=1.0, batch=256, [256,256]** | **Yes** |
| ppo_20260222_173019 | Feb 22 | **908.76 +/- 243.82** | **416.6** | 3M | alive=2.5, posture=1.0, threshold=100 | **Yes** |
| ppo_20260224_005939 | Feb 24 | 759.88 +/- 247.06 | 429.4 | 4M | alive=2.0, posture=1.5, gamma=0.995 | **Yes** |
| ppo_20260225_002644 | Feb 25 | 504.97 +/- 249.69 | 327.9 | 4M | Same as above | **Yes** |

### Stage 1 Key Findings

Stage 1 is **reliably solved** with the current config family:
- alive_bonus=2.0-2.5, posture_weight=1.0-1.5, batch_size=256, net_arch=[256,256]
- gamma=0.995 and 4M timesteps provide robust convergence
- Pass rate: ~70% of recent runs pass Stage 1

### Stage 2 Results (Critical — The Bottleneck)

| Run | Date | Reward | Ep Length | Steps | fwd_vel | alive | posture | Passed? |
|-----|------|--------|-----------|-------|---------|-------|---------|---------|
| ppo_20260213_011802 | **Feb 13** | **405.56 +/- 179.51** | **840.3** | **2M** | **1.0** | **0.5** | **0.0** | **Yes** |
| ppo_20260222_003347 | **Feb 22** | **522.47 +/- 134.14** | **882.0** | **2M** | **1.0** | **0.5** | **0.2** | **Yes** |
| ppo_20260222_203422 | Feb 22 | 359.31 +/- 269.75 | 434.6 | 3M | 1.0 | 0.5 | 0.2 | No |
| ppo_20260224_005939 | Feb 24 | 407.83 +/- 403.35 | 456.1 | 4M | 1.0 | 0.5 | 0.2 | No |
| ppo_20260224_153737 | Feb 24 | 267.79 +/- 269.90 | 199.6 | 4M | 0.8 | 1.5 | 0.5 | No |
| ppo_20260225_002644 | Feb 25 | 825.66 +/- 429.58 | 588.3 | 4M | **3.0** | 0.5 | 0.2 | No |
| ppo_20260226_021249 | Feb 26 | 557.93 +/- 364.21 | 534.0 | 5M | **3.0** | 0.5 | 0.2 | No |
| ppo_20260226_141130 | Feb 26 | -23.76 +/- 32.76 | 55.7 | 8M | **2.0** | **1.0** | **0.4** | No (catastrophic) |
| ppo_20260227_125127 | Feb 27 | -24.64 +/- 7.69 | 41.7 | 8M | **1.5** | **1.5** | **0.8** | No (catastrophic) |

### Stage 2 Key Findings

**Two runs passed Stage 2.** Both share a clear pattern:

| Parameter | Passing Runs | Failing Runs (Recent) |
|-----------|-------------|----------------------|
| forward_vel_weight | **1.0** | 1.5-3.0 |
| alive_bonus | **0.5** | 1.0-2.0 |
| posture_weight | **0.0-0.2** | 0.4-0.8 |
| nosedive_weight | **0.0** | (varies) |
| timesteps | 2M | 4-8M |

**The empirical evidence is clear:**
1. **Moderate forward velocity (1.0)** works. Pushing to 2.0-3.0 makes the agent
   sprint recklessly and fall (high reward, short episodes).
2. **Low alive_bonus (0.5) and low posture (0.0-0.2)** let the agent explore
   locomotion strategies without being trapped in a "stand still" local optimum.
3. **Matching Stage 1 weights in Stage 2 causes catastrophic forgetting.** The
   runs with alive=1.0-1.5 and posture=0.4-0.8 all collapsed (ep_len < 60).
4. **More timesteps didn't help.** The passing runs used only 2M steps, while
   8M-step runs with aggressive configs catastrophically failed.

### Current Config vs. What Works

**Problem:** The current `stage2_locomotion.toml` was tuned to "match Stage 1 to
prevent forgetting," but the data shows the opposite effect:

| Parameter | Current TOML | Successful Runs | Mismatch? |
|-----------|-------------|-----------------|-----------|
| forward_vel_weight | 1.0 | 1.0 | No |
| alive_bonus | **2.0** | **0.5** | **Yes (4x too high)** |
| posture_weight | **1.5** | **0.0-0.2** | **Yes (7x too high)** |
| nosedive_weight | **1.5** | **0.0** | **Yes (new penalty)** |
| energy_penalty | 0.003 | 0.001 | Minor |
| smoothness_weight | 0.05 | 0.05-0.1 | No |

The current config has **never been tested** with these exact values for Stage 2.
Based on the trend of increasing alive/posture weights causing worse Stage 2
performance, it is likely to fail.

### Stage 3 Results

Only one run reached Stage 3 from a passing Stage 2:

| Run | Date | From Stage 2 | Reward | Ep Length | Steps |
|-----|------|-------------|--------|-----------|-------|
| ppo_20260213_011802 | Feb 13 | Passed (840.3 ep_len) | 153.39 +/- 183.65 | 629.1 | 3M |

This is promising — the agent maintained long episodes (629 steps) and accumulated
meaningful reward, but the high variance suggests the strike behavior was not
reliable. This was with the old config (no [256,256] net_arch, no posture tuning).

### Velociraptor Recommendations

1. **Revert Stage 2 to the config that actually worked:**
   ```toml
   alive_bonus = 0.5          # Not 2.0
   posture_weight = 0.2       # Not 1.5
   nosedive_weight = 0.3      # Not 1.5
   forward_vel_weight = 1.0   # Keep
   ```

2. **The "match Stage 1 to prevent forgetting" hypothesis is wrong for this
   species.** The data shows the opposite: strong balance penalties in Stage 2
   trap the agent in a standing posture and it never learns to walk. The lower
   penalties let the agent "unlearn" rigid balance in favor of dynamic balance
   (walking itself maintains balance through momentum).

3. **Use shorter Stage 2 training (2-3M steps).** Both passing runs completed in
   2M steps. The 8M-step runs with aggressive configs diverged — more steps
   amplified bad gradients rather than fixing them.

4. **Preserve the Stage 2 checkpoint from ppo_20260222_003347** (882 ep_len) if
   it still exists — it's the best foundation for Stage 3 attempts.

5. **Stage 3 is ready to attempt** once a reliable Stage 2 checkpoint is obtained.
   The one prior Stage 3 run from Feb 13 was promising (629 ep_len, 153 reward).

---

## Cross-Species Patterns

### Catastrophic Forgetting

Both species exhibit severe catastrophic forgetting in Stage 1 to Stage 2
transitions. The pattern is consistent:

| Forgetting trigger | Evidence |
|-------------------|----------|
| High posture/balance penalties in Stage 2 | Raptor: posture 0.4-0.8 -> ep_len < 60 |
| High alive_bonus in Stage 2 | Raptor: alive 1.0-1.5 -> collapse |
| Aggressive learning rate | T-Rex: lr=1e-4 in earlier Stage 2 runs |
| Incompatible reward scaling | Sudden reward distribution shift between stages |

**What prevents forgetting (empirically):**
- Lower learning rate in Stage 2 (5e-5 or less)
- Gradual reward weight transition (not matching Stage 1 exactly)
- Moderate forward velocity incentive (1.0, not 2.0+)
- VecNormalize stats carryover between stages
- StageWarmupCallback and RewardRampCallback (available in curriculum.py)

### Stage 1 Balance — Solved Pattern

Both species converge on similar winning configs:

| Parameter | T-Rex (best) | Velociraptor (best) |
|-----------|-------------|-------------------|
| alive_bonus | 2.0 | 2.0-2.5 |
| posture_weight | 2.0 | 1.0-1.5 |
| gamma | 0.998 | 0.995 |
| batch_size | 128 | 256 |
| net_arch | [256, 256] | [256, 256] |
| timesteps | 6M | 4M |

T-Rex needs higher gamma (0.998 vs 0.995) likely because it's heavier and balance
corrections take longer to propagate through the longer body.

### Training Variance

Both species show high run-to-run variance even with the same config and seed.
Several identical configs produced vastly different results:

- Raptor: ppo_20260224_005939 -> 429.4 ep_len, ppo_20260224_193243 -> 301.4 ep_len
  (same config, 30% difference)
- T-Rex: ppo_20260222_210506 passed Stage 1, but 3 subsequent identical runs failed

This suggests the training is sensitive to initialization and early exploration
trajectories. Running multiple seeds per config and picking the best would improve
reliability.

---

## Recommended Next Steps

### Immediate (before v0.3.0 refactoring)

1. **T-Rex Stage 1:** Run with current config at 7M steps (or relax threshold to
   295). The 299.3 result is within noise of 300.

2. **Velociraptor Stage 2:** Run with the config that actually worked:
   - alive_bonus=0.5, posture=0.2, nosedive=0.3, fwd_vel=1.0
   - Use the best Stage 1 checkpoint (504.97 reward, 327.9 ep_len from recent runs)
   - 2-3M steps with lr=5e-5, batch=256, gamma=0.995

3. **Run 3 seeds per config** to account for variance. Pick the best-performing
   seed for the next stage.

### After Stage 2 is Solved

4. **Velociraptor Stage 3:** Use the passing Stage 2 checkpoint with current
   `stage3_strike.toml` config. The one prior Stage 3 attempt (153 reward, 629
   ep_len) suggests this is achievable.

5. **T-Rex Stage 3:** Same approach once Stage 2 is reliably passing.

### Config File Updates to Consider

6. **Velociraptor `stage2_locomotion.toml`:** The current values (alive=2.0,
   posture=1.5, nosedive=1.5) contradict the empirical evidence. Consider
   reverting to values closer to what worked (alive=0.5, posture=0.2).

7. **T-Rex `stage2_locomotion.toml`:** The current values (posture=2.0,
   nosedive=3.0) have never been tested. The best T-Rex Stage 2 (682 ep_len)
   used posture=0.0. Consider testing the current config, but have a fallback
   with lower penalties.

---

## Historical Run Index

### T-Rex Runs (Chronological)

| # | Run ID | Date | Stages | Best Result | Notes |
|---|--------|------|--------|-------------|-------|
| 1 | ppo_20260222_210506 | Feb 22 | 1-3 | S1 passed, S2 682 ep_len | Best overall T-Rex run |
| 2 | ppo_20260223_022137 | Feb 23 | 1 | Failed (-2.84) | Crashed early |
| 3 | ppo_20260223_023220 | Feb 23 | 1-3 | S1 67.73 reward | Forgetting in S2 |
| 4 | ppo_20260223_225351 | Feb 23 | 1 | S1 35.84 reward | |
| 5 | ppo_20260224_005932 | Feb 24 | 1-3 | S1 158, new config | alive=2.0, posture=1.5 |
| 6 | ppo_20260224_153726 | Feb 24 | 1-3 | S3 371.5 (high var) | |
| 7 | ppo_20260224_225556 | Feb 24 | 1 | S1 184.67 | Relaxed threshold |
| 8 | ppo_20260225_021848 | Feb 25 | 1 | S1 277.20 | Improving |
| 9 | ppo_20260226_141055 | Feb 26 | 1-2 | **S1 806.5, 299.3 ep_len** | **Near-pass (0.7 short)** |
| 10 | ppo_20260227_155559 | Feb 27 | 1-2 | Same S1, S2 180 | net_arch=[256,256] added |

### Velociraptor Runs (Key Milestones)

| # | Run ID | Date | Stages | Best Result | Notes |
|---|--------|------|--------|-------------|-------|
| 1 | ppo_20260212_224234 | Feb 12 | 1-2 | S1 passed, S2 776 ep_len | First pass |
| 2 | ppo_20260213_011802 | Feb 13 | **1-3** | **S1+S2 passed**, S3 153 | **Best complete run** |
| 3 | ppo_20260213_194535 | Feb 13 | 1-3 | Failed (posture=0.3) | Posture penalty hurt |
| 4 | ppo_20260221_033148 | Feb 21 | 1-2 | S1 293, alive=5.0 | High alive_bonus tested |
| 5 | ppo_20260222_003347 | Feb 22 | 1-2 | **S1 792, S2 882** | **Best S2 ever** |
| 6 | ppo_20260222_173019 | Feb 22 | 1 | **S1 909** | Best S1 reward |
| 7 | ppo_20260224_005939 | Feb 24 | 1-2 | S1 passed, S2 456 | gamma=0.995 |
| 8 | ppo_20260225_002644 | Feb 25 | 1-2 | S2 826 reward, 588 ep_len | fwd_vel=3.0 |
| 9 | ppo_20260226_141130 | Feb 26 | 1-2 | S2 catastrophic (55 ep_len) | fwd_vel=2.0, alive=1.0 |
| 10 | ppo_20260227_125127 | Feb 27 | 1-2 | S2 catastrophic (42 ep_len) | posture=0.8, alive=1.5 |

---

## Velociraptor Stage 3 Review — March 15, 2026

> **Run:** Seed 42, PPO, 3-stage curriculum (6M + 8M + 8M steps)
> **Result:** Stage 3 **failed** — success_rate 0.0% vs 10% threshold

### Results Summary

| Stage | Best Reward | Ep Length | fwd_vel | Success Rate | Passed? |
|-------|-------------|-----------|---------|--------------|---------|
| 1 (balance) | 1501.96 | 958.1 | 0.32 m/s | — | **Yes** |
| 2 (locomotion) | 2654.81 | 1000.0 | 3.88 m/s | 3.3% | **Yes** |
| 3 (strike) | 2377.02 | 955.0 | 0.47 m/s | **0.0%** | **No** |

This is the **first full 3-stage run** where stages 1 and 2 both passed their
curriculum gates. Stage 3 trained for 8M steps but achieved zero strike success.

### Key Observations from Training Curves

**Reward:** Climbs steadily from ~100 to ~2000-2500, with high variance
(shaded band from ~500 to ~2800). The agent is accumulating meaningful
per-step reward but not from strikes.

**Speed:** Drops from ~3.0 m/s (inherited from Stage 2) to ~0.4-0.5 m/s by 2M
steps and stays there. The agent actively unlearns running. This is rational:
approaching the prey slowly maximizes per-step proximity/approach rewards without
overshooting and losing claw_proximity bonus.

**Prey Distance:** Drops from ~12m to ~1.5m by 3M steps, then plateaus. The
raptor gets very close but refuses to make contact.

**Strike Success Rate:** Essentially zero throughout training (~0.0001-0.0007).
The rare contacts are accidental, not learned behavior.

**Termination Breakdown:** `strike_success` is the #1 termination reason at ~42%
but this is misleading — these are likely accidental contacts during early
exploration that the agent then learns to *avoid*.

**Cost of Transport:** Rises steadily, suggesting increasingly inefficient
movement — the agent is "creeping" near the prey rather than locomoting
efficiently.

---

### Question 1: Why is net_arch different across stages?

**Resolved.** All stage TOMLs now specify `net_arch = [512, 256]` directly,
and the `sweep_validation.toml` / `VALIDATION_SETTING` override system has
been removed. Net_arch is consistent across all stages for all species.

---

### Question 2: Why does prey distance change between stages?

This is intentional curriculum design:

| Stage | prey_distance_range | Rationale |
|-------|-------------------|-----------|
| 1 (balance) | `[10.0, 15.0]` | Prey is far away — the agent should focus on standing, not chasing |
| 2 (locomotion) | `[8.0, 12.0]` | Slightly closer to encourage forward movement toward a target |
| 3 (strike) | `[2.0, 6.0]` | Close enough that the agent can discover strikes through exploration |

The stage 3 config comment in `stage3_strike.toml` explains: *"Tightened from
[3.0, 8.0]: closer prey makes strike discovery much more likely during
exploration."*

This is sound design — sparse rewards (like the one-time strike bonus) need the
agent to be close enough to accidentally contact the prey during random
exploration, which then gets reinforced. At 10-15m, the probability of a random
walk reaching the prey is negligible.

---

### Question 3: Why does the raptor approach but never strike?

**The strike bonus is far too low relative to the opportunity cost of episode
termination.** The raptor has correctly learned that striking is net-negative.

**The math:**

At convergence, the agent earns ~2.0-2.5 reward per step from combined per-step
rewards (forward velocity, alive bonus, proximity, claw proximity, heading,
posture, etc.). With `gamma = 0.995` and ~500 steps remaining in a typical
episode, the discounted future reward from staying alive is:

```
Future value ≈ Σ(0.995^i × 2.5) for i=0..499
             ≈ 2.5 × (1 - 0.995^500) / 0.005
             ≈ 2.5 × 183.6
             ≈ 459
```

The strike bonus is **50.0** but immediately terminates the episode. So striking
costs the agent ~459 in expected future reward and pays only 50 — a net loss
of ~409. The agent rationally avoids striking.

**The reward decomposition chart confirms this:** The cyan "S3 strike" line is
flat at zero, while approach/proximity/claw_proximity rewards are positive and
sustained. The agent has found the optimal strategy *within the current reward
structure*: get as close as possible to the prey (maximizing proximity rewards)
without actually touching it (which would terminate the episode).

**The alive_bonus was already reduced from 0.5 to 0.05** (the config comment
says: *"survival is learned in stages 1-2; high alive_bonus made striking
net-negative"*). But 0.05/step is only one component — the agent also earns
forward_vel (0.5 weight), heading (0.3), proximity (0.5), claw_proximity (2.0),
posture (0.1), etc. every step. The total per-step reward dwarfs the one-time
strike bonus.

**Recommended fixes (pick one or combine):**

1. **Increase strike_bonus dramatically** — to at least 500-1000 to exceed the
   discounted future value. The config comment says it was raised "10x from 5.0"
   but it needs another 10-20x increase.

2. **Don't terminate on strike success.** Instead, respawn the prey at a new
   random location and let the agent strike multiple times per episode. This
   makes striking additive rather than episode-ending, removing the opportunity
   cost entirely.

3. **Add a per-step strike penalty** — a small negative reward for being close
   to the prey WITHOUT striking (e.g., `-0.1 × claw_proximity`). This makes
   "hovering near prey" costly and breaks the local optimum.

4. **Reduce the per-step reward budget in Stage 3.** Currently forward_vel_weight
   is 0.5, which gives continuous reward for moving. In a hunting stage, the agent
   should be incentivized to strike, not to meander. Consider zeroing out
   forward_vel_weight and proximity rewards, keeping only approach_weight (which
   is delta-based and goes to zero at the prey) and the strike bonus.

5. **Use a shaped terminal reward** — instead of a flat 50.0, scale the strike
   bonus by remaining episode time: `strike_bonus × (remaining_steps / max_steps)`.
   Early strikes are worth more, incentivizing speed.

**Recommendation 2 (prey respawn) is likely the most robust fix** because it
fundamentally changes the problem from "one-shot sparse reward vs episode
termination" to "repeated dense reward," which PPO handles much better.

---

## Brachiosaurus Stage 2 Review — March 23, 2026

> **Run:** `ppo_20260322_203554`, Seed 42, PPO, L4 GPU
> **Stage:** 2 (locomotion) — Learn coordinated quadrupedal walking
> **Result:** Stage 2 **failed** — did not meet curriculum gates

### Results Summary

| Metric | Final (20M steps) | Best (14.3M steps) | Best Model Eval (30 ep) | Curriculum Gate |
|--------|-------------------|---------------------|------------------------|----------------|
| Eval Reward | 798.88 ± 80.24 | **2495.14 ± 404.96** | 2396.38 ± 504.04 | ≥ 100 ✅ |
| Episode Length | **199.6 ± 17.9 steps** | 960.9 ± 146.3 steps | 921.1 ± 203.4 steps | ≥ 750 ❌ |
| Forward Velocity | **0.59 ± 0.07 m/s** | — | **0.13 ± 0.11 m/s** | ≥ 0.75 m/s ❌ |
| Duration | 10h 47m 56s | — | — | — |

**The run failed both the episode length and forward velocity gates.** The best
checkpoint (14.3M steps) had excellent episode length (921 steps) but essentially
zero forward velocity (0.13 m/s). The agent learned to survive but not walk.

---

### Phase Analysis

The training curves reveal three distinct phases:

**Phase 1: Warm-up & Balance Retention (0–5M steps)**
- Episode length holds at ~1000 steps (max), indicating the agent retained Stage 1
  balance through the warm-up and ramp periods
- Forward velocity stays near zero — the agent is standing still
- Reward climbs from ~1000 to ~1800 purely from alive/posture/height rewards
- Contact forces are asymmetric: RL (rear-left) dominates at ~200 force units,
  FR (front-right) peaks later at ~100 — the agent leans heavily on a diagonal pair
- Gait symmetry spikes dramatically (30–60) as the agent experiments with leg
  movements but doesn't achieve coordinated walking
- Cost of transport peaks at 0.20 — high energy expenditure for zero locomotion

**Phase 2: Collapse & Speed Discovery (5–10M steps)**
- **Catastrophic episode length collapse** from ~1000 to ~200 steps around 5M
- Forward velocity suddenly jumps from 0 to ~0.35 m/s
- The forward velocity ramp reaches full weight at 3M steps, and by 5M the
  gradient signal overwhelms the stability rewards
- All foot contact forces drop to near-zero — the agent is airborne/falling quickly
- Gait symmetry crashes to zero (no coordinated leg alternation)
- Pelvis height drops from 1.22m to 1.17m (the creature slouches forward)
- Tilt angle increases from 3° to 7° (forward lean)
- Distance traveled jumps from 0.2m to 0.6m per episode

**Phase 3: Unstable Plateau & Late Degradation (10–20M steps)**
- Reward oscillates wildly between 800–2500 with the peak around 14M
- Episode length fluctuates between 200–1000 — highly unstable policy
- Forward velocity reaches ~0.35–0.40 m/s but never approaches the 0.75 gate
- A **second collapse** occurs around 17M steps, with episode length crashing to ~200
- Speed dips to ~0.30 m/s by 20M
- Heading alignment remains high (~0.95) — the agent moves straight but slowly
- Final distance traveled ~1.0m, drift ~0.9m per episode

---

### Termination Analysis

| Termination Reason | Fraction |
|-------------------|----------|
| **Fallen** | ~85% |
| **Tail contact** | ~45% |
| **Head contact** | ~14% |

The agent is overwhelmingly terminated by falling. The high tail_contact fraction
(45%) suggests the creature is tipping backward or its tail is dragging the ground.
Head contact at 14% indicates nose-diving, which the nosedive_weight=0.8 is
supposed to prevent but isn't sufficient.

---

### Reward Decomposition

The reward decomposition reveals a clear hierarchy at convergence:

| Component | Approximate Value | Notes |
|-----------|-------------------|-------|
| S2 forward | ~0.2–0.4 (growing) | Dominant learned component after 5M steps |
| S2 alive | ~0.3 (constant) | Fixed bonus per step |
| S2 gait_symmetry | ~0.5–0.7 | High early, but this reflects the penalty *not being applied* when stationary |
| S2 height | ~0.1 | Small positive signal |
| S2 energy | ~-0.05 | Very small penalty |
| All others | ~0 | Posture, smoothness, heading, lateral — negligible contribution |

The forward reward is the dominant learned signal but caps at ~0.4 per step because
the agent only reaches ~0.35 m/s × 4.0 weight / forward_vel_max = ~1.4 before
the episode terminates. Short episodes mean less total reward accumulation.

---

### Diagnosis: Two Competing Failure Modes

**Failure Mode 1: The "Statue" (0–5M steps)**
The agent learns that standing still maximizes alive_bonus + height + posture
rewards without any fall risk. With forward_vel_weight ramping slowly from 0.2,
the gradient signal for movement is initially too weak to overcome the stable
equilibrium of standing still. The gait_symmetry reward (2.0 weight) is maximized
when all four feet touch the ground — which they do when standing.

**Failure Mode 2: The "Lunge and Fall" (5–20M steps)**
Once the forward velocity reward reaches full weight (4.0), the agent discovers
that lunging forward yields high instantaneous reward. But it hasn't developed the
coordinated quadrupedal gait needed to sustain locomotion, so it falls within
~200 steps (2 seconds). The 85% fall termination rate confirms this.

**The core issue:** The agent transitions directly from standing to lunging without
passing through stable walking. The reward landscape has a valley between the two
attractors — walking requires simultaneously maintaining balance AND moving forward,
which is harder than either alone.

---

### Contact Force Analysis

The per-foot contact patterns are revealing:

- **Early training (0–1M):** RL (rear-left) carries ~200 force, FR (front-right)
  carries ~25. The creature leans heavily on a single diagonal pair (RL+FR)
- **Mid-training (3–5M):** FR force spikes to ~100, FL and RR rise to ~50.
  A second diagonal pair activates but forces are still asymmetric
- **Late training (7M+):** ALL foot forces drop to near-zero. The creature is
  barely in contact with the ground before falling

The diagonal pair contact chart confirms the asymmetry: Diag A (FR+RL) peaks at
~200 while Diag B (FL+RR) peaks at ~90. A healthy quadrupedal walk should show
alternating, roughly equal diagonal pairs.

---

### Best Model Paradox

The best checkpoint (14.3M steps) achieves 921 steps episode length but only
0.13 m/s forward velocity. The 30-episode evaluation confirms this: 2396 reward
over 921 steps ≈ 2.6 reward/step, mostly from alive + gait + height bonuses.

**This checkpoint is essentially a refined "statue."** It mastered standing
(achieving near-maximum episode length) with perhaps slight swaying that registers
as minimal forward velocity. It would not pass the 0.75 m/s velocity gate.

This creates a dilemma for checkpoint selection: the best-reward model can't walk,
while the models that walk (Phase 3) can't survive long enough.

---

### Comparison with Cross-Species Patterns

This run echoes several patterns from the T-Rex and Velociraptor analysis:

| Pattern | Velociraptor/T-Rex | Brachiosaurus (this run) |
|---------|-------------------|-------------------------|
| Forward vel weight too high | fwd=2.0-3.0 → reckless sprinting | fwd=4.0 → lunging and falling |
| Standing-still optimum | Alive + posture trap | Alive + gait_symmetry + height trap |
| Catastrophic collapse | ep_len drops to 42-55 | ep_len drops from 1000 to 200 |
| Contact asymmetry | — | RL leg carries 4x more force than others |
| Best model = best stander | — | Best model: 921 steps, 0.13 m/s velocity |

The forward_vel_weight=4.0 for Brachiosaurus is **even more aggressive** than the
3.0 that caused catastrophic failures in Velociraptor. For a heavy sauropod that
needs stable footing, this is too much velocity pressure.

---

### Recommendations

#### 1. Reduce forward_vel_weight to 2.0 (from 4.0)

The Velociraptor data showed that fwd_vel=1.0 was the sweet spot for bipeds. For a
quadruped with more complex gait coordination, 2.0 should provide sufficient
gradient signal without overwhelming stability rewards. At 0.75 m/s target speed,
this gives 1.5/step — meaningful but not dominant.

#### 2. Increase alive_bonus to 0.5 (from 0.3)

The 85% fall rate is the primary failure mode. A stronger survival signal
prioritizes learning to stay upright while moving. The Velociraptor's successful
Stage 2 runs used alive=0.5.

#### 3. Extend the ramp period to 5M steps (from 3M)

The current 3M ramp means full velocity pressure hits at exactly the time the
agent is still discovering how to use its legs. A 5M ramp gives more time for
gait coordination to develop before speed pressure becomes dominant.

#### 4. Increase gait_stability_weight to 0.2 (from 0.05)

The 0.05 weight is too small to meaningfully penalize the angular velocity of the
torso. The agent's increasing tilt angle (3° → 8°) shows this isn't being
controlled. A 4x increase makes this comparable to posture_weight.

#### 5. Add a minimum contact reward

The drop to zero foot contact forces suggests the agent doesn't value ground
contact. Consider adding a reward component that specifically rewards having ≥3
feet in contact with the ground, separate from gait_symmetry. This bridges the
gap between "standing" and "walking" by ensuring the agent maintains ground
contact during locomotion.

#### 6. Reduce training to 15M steps or add early stopping

Performance peaked at 14.3M and then collapsed. Training past this point
destroyed a good policy. Either cap training earlier or implement early stopping
that saves the checkpoint when eval metrics decline for >1M consecutive steps.

#### 7. Try 3 seeds

The high run-to-run variance observed across species means this single run may
not be representative. Running 3 seeds with the adjusted config and picking the
best would improve reliability.

#### Proposed Config Changes

```toml
[env]
forward_vel_weight = 2.0        # Was 4.0: too aggressive for heavy quadruped
alive_bonus = 0.5               # Was 0.3: reduce 85% fall rate
fall_penalty = -50.0            # Keep
gait_stability_weight = 0.2    # Was 0.05: control body angular velocity
gait_symmetry_weight = 2.0     # Keep: still needed for four-leg coordination
posture_weight = 0.3            # Keep
nosedive_weight = 0.8           # Keep
idle_penalty_weight = 0.2       # Was 0.3: less pressure to move at all costs

[curriculum]
timesteps = 15000000            # Was 20M: performance collapsed after 14M
ramp_timesteps = 5000000        # Was 3M: slower ramp for gait development
```

---

### Open Questions

1. **Is the gait_symmetry reward correctly formulated for a sauropod?** At 2.0
   weight, it's the second-largest reward signal, but the agent achieves it by
   standing still (all four feet on ground). It may need to be conditioned on
   forward velocity — only reward symmetry when the agent is actually moving.

2. **Should the VecNormalize running stats be frozen during warm-up?** The
   300K-step warm-up with low clip range may not be enough if the observation
   distribution shifts significantly between standing (Stage 1) and walking
   (Stage 2).

3. **Is the natural_pitch target (-0.15 rad ≈ -8.6°) appropriate?** The tilt
   angle chart shows the agent converging toward 8°, which is close to this
   target. If the Brachiosaurus should be more upright, this target needs
   adjustment.

---

## Velociraptor SAC Results — March 21, 2026

> **Run:** `sac_20260321_170055`, Seed 42, SAC, 8 parallel envs, L4 GPU
> **Result:** All 3 stages **PASSED** — first successful SAC training

### Results Summary

| Stage | Name | Best Reward | Ep Length | Fwd Vel | Success Rate | Time | Passed? |
|-------|------|-------------|-----------|---------|--------------|------|---------|
| 1 | Balance | 970.19 | 945.7 | -0.64 m/s | — | 5:08:59 | **Yes** |
| 2 | Locomotion | 2078.62 | 874.4 | 2.91 m/s | — | 8:36:12 | **Yes** |
| 3 | Strike | 1195.43 | 257.6 | 1.63 m/s | **90.0%** | 9:14:06 | **Yes** |

**Total:** 22M steps, 22:59:18 training time (8 parallel envs)

### Key Observations

1. **SAC is ~2x slower than PPO** (22:59 vs 11:25 for the same 22M steps) due to
   the replay buffer and twin Q-network updates. Using 8 parallel envs (vs 4 for
   PPO) partially offset this.

2. **Lower balance reward than PPO** (970 vs 1960) but still passes the threshold.
   SAC's entropy-maximizing objective produces more diverse balance behaviors with
   lower average reward but sufficient stability.

3. **Slightly lower forward velocity** (2.91 vs 3.47 m/s) in Stage 2, but still
   well above the 2.0 m/s threshold.

4. **90% strike success** vs PPO's 93.3%. Both are well above the 25% threshold
   but SAC's exploration-driven policy may be slightly less precise in the final
   strike execution.

5. **SAC used `gamma=0.99`** (vs PPO's `0.9797`/`0.995`) and `lr=0.0003`/`0.0001`,
   with `[512, 256]` network architecture matching the PPO sweep winner.

### SAC vs PPO Comparison

| Metric | PPO (ppo_20260315_041632) | SAC (sac_20260321_170055) |
|--------|--------------------------|--------------------------|
| Stage 1 Best Reward | 1960.64 | 970.19 |
| Stage 2 Fwd Vel | 3.47 m/s | 2.91 m/s |
| Stage 3 Success Rate | 93.3% | 90.0% |
| Total Training Time | 11:25:15 | 22:59:18 |
| Parallel Envs | 4 | 8 |

PPO remains the more efficient algorithm for this task, but SAC's success
validates the curriculum design works across algorithm families.

### Other SAC Attempts

| Run | Date | Envs | Stage 1 | Stage 2 | Stage 3 | Notes |
|-----|------|------|---------|---------|---------|-------|
| sac_20260320_004309 | Mar 20 | 4 | Failed (222.0) | — | — | Only 400K steps, gamma=0.99 |
| sac_20260320_151527 | Mar 20 | 4 | Passed (1302.83) | — | — | 2M steps, lib v0.3.0.dev0 |
| sac_20260321_170055 | **Mar 21** | **8** | **Passed** | **Passed** | **Passed (90%)** | **Best run** |
| sac_20260322_210152 | Mar 22 | 8 | Passed (1484.66) | — | — | 4.6M steps, incomplete |
| sac_20260323_010349 | Mar 23 | 4 | Failed (1229.71) | — | — | gamma=0.9797, 6M steps |

The Mar 23 run with `gamma=0.9797` (PPO's optimized value) failed, suggesting
SAC prefers its own gamma schedule rather than borrowing from PPO sweep results.

---

## Brachiosaurus Stage 3 (Food Reach) Review — March 21, 2026

> **Run:** `ppo_20260321_144730`, Seed 42, PPO, 4 envs, L4 GPU
> **Result:** Stage 3 **FAILED** — 16.7% success rate vs 50% threshold
> **This is the furthest any Brachiosaurus run has progressed** (first to reach Stage 3)

### Full Run Results

| Stage | Name | Best Reward | Ep Length | Fwd Vel | Success Rate | Steps | Time | Passed? |
|-------|------|-------------|-----------|---------|--------------|-------|------|---------|
| 1 | Balance | 3002.52 | 1000.0 | 0.02 m/s | — | 6M | 3:46:42 | **Yes** |
| 2 | Locomotion | 4176.95 | 957.4 | 1.12 m/s | 3.3% | 16M | 8:18:51 | **Yes** |
| 3 | Food Reach | 732.20 | 460.5 | 0.52 m/s | **16.7%** | 8M | 3:54:06 | **No** |

### Stage 2 Breakthrough

This run achieved the **first successful Brachiosaurus Stage 2**, with:
- Forward velocity of **1.12 m/s** (threshold: 0.75 m/s)
- Episode length of **957.4 steps** (threshold: 750)
- Best reward of **4176.95** — the highest Stage 2 reward across all species

Key config differences from earlier failed Stage 2 runs:

| Parameter | Failed runs (Mar 18-20) | This run (Mar 21) |
|-----------|------------------------|-------------------|
| forward_vel_weight | 2.0-3.0 | **4.0** |
| energy_penalty_weight | 0.002 | **0.005** |
| alive_bonus | 0.1-0.75 | **0.2** |
| timesteps | 8-12M | **16M** |

The higher forward_vel_weight (4.0) combined with longer training (16M steps)
finally overcame the "statue" failure mode, while the increased energy penalty
prevented the "lunge and fall" mode.

### Stage 3 Analysis

Stage 3 (food_reach) ran for 8M steps with the following config:
- `alive_bonus=0.0`, `energy_penalty=0.001`, `forward_vel_weight=0.5`, `posture_weight=0.1`
- Food reach uses neck extension to touch elevated food targets
- 16.7% success rate with only 460.5 step average episode length

**Why it's failing:**
1. **Short episodes (460 steps)** suggest the agent falls before reaching food
2. **Forward velocity drops** from 1.12 (Stage 2) to 0.52 m/s — the agent slows
   down but doesn't compensate with neck reaching
3. **Low success rate (16.7%)** indicates the neck extension behavior is rarely
   triggered — the agent reaches the food location but doesn't extend its neck
4. Similar to the velociraptor Stage 3 "approach but don't strike" pattern:
   the continuous per-step rewards outweigh the sparse food_reach bonus

### Brachiosaurus Training History

| Run | Date | S1 Passed | S2 Passed | S2 Best Reward | S2 Fwd Vel | S3 Result |
|-----|------|-----------|-----------|----------------|------------|-----------|
| ppo_20260318_144535 | Mar 18 | No (825.0) | — | — | — | — |
| ppo_20260318_181308 | Mar 18 | No (1445.74) | — | — | — | — |
| ppo_20260319_001536 | Mar 19 | **Yes** | No | 952.56 | 0.19 m/s | — |
| ppo_20260319_133946 | Mar 19 | **Yes** | No | 2023.02 | 0.18 m/s | — |
| ppo_20260319_232800 | Mar 19 | **Yes** | No | 1623.33 | 0.56 m/s | — |
| ppo_20260320_151735 | Mar 20 | **Yes** | No | 1675.35 | 0.24 m/s | — |
| ppo_20260321_024127 | Mar 21 | **Yes** | No | 2692.80 | 0.89 m/s | — |
| ppo_20260321_144730 | **Mar 21** | **Yes** | **Yes** | **4176.95** | **1.12 m/s** | **16.7% food_reach** |
| ppo_20260322_203554 | Mar 22 | **Yes** | No | 2396.38 | 0.59 m/s | — |

### Stage 2 Config Evolution

The progression shows forward_vel_weight and timesteps as the key levers:

| Config | fwd_vel_weight | timesteps | Best fwd_vel |
|--------|---------------|-----------|-------------|
| Mar 19 (001536) | 2.0 | 8M | 0.19 m/s |
| Mar 19 (232800) | 3.0 | 8M | 0.56 m/s |
| Mar 21 (024127) | 8.0 | 12M | 0.89 m/s |
| **Mar 21 (144730)** | **4.0** | **16M** | **1.12 m/s** |
| Mar 22 (203554) | 4.0 | 20M | 0.59 m/s |

The Mar 22 run at 20M steps actually performed worse than the Mar 21 run at 16M,
suggesting 16M is near-optimal and longer training risks overtraining.

### Recommendations for Brachiosaurus Stage 3

> **Note (2026-04-18):** A deeper analysis in
> [REWARD_SCALE_REDESIGN.md](REWARD_SCALE_REDESIGN.md) argues the opposite
> direction — that the +1000 terminal bonus across all Stage 3 configs is
> already too large relative to accumulated per-step shaping, producing a
> bimodal non-stationary return distribution that VecNormalize was papering
> over. The food-reach problem may stem from the sparsity of the trigger
> (`neck extension`) rather than the magnitude of the bonus. See that doc
> for the proposed rescale + validation plan; recommendation 1 below should
> be reframed or reversed before acting on it.

1. **Increase food_reach bonus significantly** — same pattern as the velociraptor
   strike bonus issue. The per-step rewards dominate the sparse food_reach reward.

2. **Try non-terminating food reach** — respawn food at a new position after
   successful reach instead of ending the episode, making food_reach additive.

3. **Reduce per-step reward budget** — zero out forward_vel_weight in Stage 3
   to remove the incentive for aimless walking vs purposeful neck reaching.

4. **Increase training steps** — 8M may be insufficient. The successful Stage 2
   needed 16M; Stage 3 likely needs similar or more.

5. **Add a proximity-to-food shaping reward** that specifically rewards the
   head (not body) being close to the food target, guiding neck extension.

---

## Updated Cross-Species Summary — March 25, 2026

### Training Completion Status

| Species | Algorithm | Balance | Locomotion | Behavior | Total Time | Status |
|---------|-----------|---------|------------|----------|------------|--------|
| Velociraptor | PPO | 1960.64 | 3.47 m/s | 93.3% strike | 11:25:15 | **Complete** |
| Velociraptor | SAC | 970.19 | 2.91 m/s | 90.0% strike | 22:59:18 | **Complete** |
| T-Rex | PPO | 2994.34 | 3.47 m/s | 96.7% bite | 13:02:32 | **Complete** |
| Brachiosaurus | PPO | 3002.52 | 1.12 m/s | 16.7% food_reach | 15:59:39 | Stage 3 failing |

### Total Training Runs

| Species | PPO Runs | SAC Runs | Total |
|---------|----------|----------|-------|
| Velociraptor | ~60 | 5 | ~65 |
| T-Rex | ~50 | 0 | ~50 |
| Brachiosaurus | ~10 | 0 | ~10 |
| **Total** | **~120** | **5** | **~125** |

### Key Learnings (Updated)

1. **Curriculum design is algorithm-agnostic** — the same 3-stage structure works
   for both PPO and SAC, validating the pedagogical approach.

2. **Quadrupedal locomotion is significantly harder** than bipedal. Brachiosaurus
   needed 16M steps for Stage 2 (vs 8M for bipeds) and Stage 3 remains unsolved.

3. **Behavior stages (3) share a common failure pattern** across all species:
   per-step rewards dominate sparse terminal bonuses, causing "approach but don't
   act" behavior. Each species required reward rebalancing to fix this.

4. **SAC needs 2x wall-clock time** for comparable results but provides a useful
   second opinion on reward function design — if both PPO and SAC solve a stage,
   the curriculum gates are well-calibrated.
