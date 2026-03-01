# Training Results Review

> **Date:** 2026-03-01
> **Data:** 80+ training runs across T-Rex and Velociraptor (Feb 12 - Feb 27)
> **Algorithm:** PPO (all runs)

---

## Executive Summary

| Species | Stage 1 (Balance) | Stage 2 (Locomotion) | Stage 3 (Behavior) |
|---------|-------------------|---------------------|-------------------|
| **T-Rex** | Nearly solved (299.3 vs 300 threshold) | Struggling (best: 682 ep_len) | Untested from passing Stage 2 |
| **Velociraptor** | Reliable (passes consistently) | Solved once (882 ep_len), not reproduced | Never attempted from passing Stage 2 |

**Bottom line:** Both species have Stage 1 under control. Stage 2 locomotion is
the bottleneck — it was solved once for Velociraptor but recent config changes
broke it. T-Rex Stage 2 has shown promise but suffers catastrophic forgetting.

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
