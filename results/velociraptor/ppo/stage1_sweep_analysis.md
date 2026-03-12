# Raptor Stage 1 (Balance) — Hyperparameter Sweep Analysis

> **Note**: This sweep was run *before* the spinning fix (`spin_penalty_weight`)
> and drift penalty (`drift_penalty_weight`) were added. Results should be
> interpreted with that caveat — the top configs here may behave differently
> with those corrections in place.

## Summary

40 trials were run across PPO hyperparameters and environment reward weights.
**36 of 40 trials passed** stage 1 (reward > 100, episode length > 300),
indicating the balance task is broadly solvable across a wide range of configs.
The 4 failures share clear patterns (see below).

**Best trial**: Trial 40 — `best_mean_reward = 4954`, `ep_len = 1000` (maxed out)

**Current config baseline reward**: ~2875 (Trial 1 is closest to the current
`stage1_balance.toml` settings).

## Top 10 Trials by Best Mean Reward

| Rank | Trial | Best Reward | Ep Len | alive_bonus | gamma | lr | batch | n_steps | n_epochs | arch | energy_pen |
|------|-------|-------------|--------|-------------|-------|----|-------|---------|----------|------|------------|
| 1 | 36 | 4968 | 1000 | 5.00 | 0.971 | 8.0e-5 | 256 | 2048 | 6 | [256,256,256] | 0.012 |
| 2 | 40 | 4954 | 1000 | 5.00 | 0.970 | 1.0e-5 | 64 | 4096 | 10 | [512,256] | 0.025 |
| 3 | 25 | 4730 | 1000 | 4.78 | 0.976 | 9.6e-5 | 256 | 2048 | 10 | [512,256] | 0.018 |
| 4 | 39 | 4723 | 965 | 5.00 | 0.970 | 1.0e-5 | 64 | 1024 | 10 | [512,256] | 0.017 |
| 5 | 33 | 4635 | 949 | 5.00 | 0.972 | 3.0e-4 | 256 | 2048 | 6 | [512,256] | 0.010 |
| 6 | 27 | 4576 | 1000 | 4.63 | 0.977 | 2.1e-5 | 64 | 2048 | 10 | [256,256,256] | 0.014 |
| 7 | 22 | 4517 | 951 | 4.83 | 0.975 | 1.0e-5 | 256 | 2048 | 10 | [512,256] | 0.020 |
| 8 | 20 | 4517 | 972 | 4.78 | 0.988 | 4.9e-5 | 128 | 1024 | 10 | [512,256] | 0.023 |
| 9 | 29 | 4428 | 988 | 4.60 | 0.980 | 1.2e-5 | 128 | 4096 | 6 | [512,256] | 0.014 |
| 10 | 24 | 4405 | 1000 | 4.46 | 0.970 | 2.7e-5 | 256 | 4096 | 10 | [512,256] | 0.024 |

All top-10 trials achieved episode lengths of 950-1000 (near or at the maximum),
meaning the raptor balances for the full episode.

## Failed Trials (4/40)

| Trial | Best Reward | Ep Len | Key Factors |
|-------|-------------|--------|-------------|
| 16 | 385 | 125 | gamma=0.992, lr=1e-5, batch=512, n_steps=4096 |
| 17 | 387 | 119 | **gamma=0.999**, **ent_coef=0.0485**, batch=512 |
| 34 | 824 | 196 | **lr=2.8e-4**, ent_coef=0.0045, gamma=0.987 |
| 37 | 407 | 106 | **lr=3e-4**, batch=64, n_steps=2048, arch=[256,256,256] |

## Key Lessons Learned

### 1. `alive_bonus` is the single most impactful parameter — raise it significantly

The current config uses `alive_bonus = 2.0`. Every single top-10 trial uses
**4.5 - 5.0** (the upper end of the sweep range). This makes intuitive sense:
for a balance task, the primary reward signal *is* staying alive. A higher bonus
creates a stronger, clearer gradient for the policy.

- **Top 10 average**: 4.83
- **Bottom 10 average**: 3.09
- **Current config**: 2.0

**Recommendation**: Increase `alive_bonus` to **4.5 - 5.0**.

### 2. `gamma` should be lower than 0.99 — sweet spot is 0.97 - 0.98

The current config uses `gamma = 0.99`. The sweep clearly shows that **lower
discount factors work better** for the balance task:

- **Top 10 average gamma**: 0.975
- **All 4 failures** had gamma >= 0.987
- Trial 17 (gamma=0.999) was the worst performer

High gamma makes the value function try to predict rewards far into the future,
which is unnecessary for balance — the agent just needs to stay upright *now*.
Lower gamma stabilizes value estimation and speeds convergence.

**Recommendation**: Reduce `gamma` to **0.975**.

### 3. Learning rate — current 3e-4 is risky, lower is safer

Two of the four failed trials used `lr >= 2.8e-4` (the current config's value).
Most top performers use learning rates an order of magnitude lower:

- **Top 10 median lr**: ~2.7e-5
- Trials 34 and 37 both failed with lr=~3e-4

However, trial 33 succeeded with lr=3e-4 and reached reward 4635, so high LR
isn't an automatic failure — it just increases variance. Lower LR gives more
consistent results.

**Recommendation**: Reduce `learning_rate` to **5e-5** (or use a cosine
schedule from 3e-4 down to 1e-5 to get the best of both).

### 4. `batch_size = 512` underperforms — avoid it

No trial with batch_size=512 cracked the top 10. Both trials 16 and 17
(failures) used batch_size=512. The best 512-batch trial (trial 7) peaked at
3433 — far below the top performers.

Smaller batches (64-256) provide noisier but more frequent gradient updates,
which appears to help for this task.

- **Current config**: 256 — this is fine, no change needed.

### 5. `n_epochs = 10` is slightly favored over 6 or 3

7 of the top 10 trials used `n_epochs = 10`. The current config uses 6, which
also appears in 3 of the top 10. Either works, but 10 may extract more value
from each rollout.

**Recommendation**: Consider increasing `n_epochs` to **10**.

### 6. Entropy coefficient — keep it low

- Trial 17 had the highest ent_coef (0.0485) and failed catastrophically
- Top performers cluster around **0.0001 - 0.003**
- Current config (0.005) is at the upper edge of the working range

**Recommendation**: Reduce `ent_coef` to **0.001**.

### 7. Network architecture — [512, 256] or [256, 256, 256] both work

The current [256, 256] is not represented in the sweep, but related archs
performed well:
- **[512, 256]** ("tapered"): Most common among top performers (7 of top 10)
- **[256, 256, 256]** ("deep"): Trials 27 and 36 both cracked top 10
- **[512, 512]**: Consistently weaker (best trial: 3433)
- **[512, 512, 256]**: Weakest (best trial: 2593)

Wider layers don't help — a tapered or deep-narrow architecture is better.

**Recommendation**: Switch to `net_arch = [512, 256]`.

### 8. `nosedive_weight` should be higher — sweet spot around 2.2-3.0

- **Top 10 average**: 2.58
- **Current config**: 1.5

Higher nosedive penalty keeps the raptor upright more consistently.

**Recommendation**: Increase `nosedive_weight` to **2.5**.

### 9. `energy_penalty_weight` should be lower

- **Top 10 average**: 0.018
- **Current config**: 0.05

Lower energy penalty lets the agent use whatever actuator effort is needed
to balance without being penalized for it.

**Recommendation**: Reduce `energy_penalty_weight` to **0.02**.

### 10. `posture_weight` — current value is reasonable

- **Top 10 range**: 0.5 - 2.1, average ~1.3
- **Current config**: 1.5

This is in the sweet spot. No change needed.

### 11. Policy stability — some configs collapse late in training

Several trials show significant gaps between `best_mean_reward` and
`last_mean_reward`, indicating the policy degraded after reaching its peak:

| Trial | Best | Last | Drop |
|-------|------|------|------|
| 25 | 4730 | 3590 | -24% |
| 28 | 3799 | 1240 | -67% |
| 5 | 2304 | 160 | -93% |
| 1 | 2875 | 1236 | -57% |

Trials with **lower learning rates** and **lower gamma** tend to be more
stable (less drop-off). Using a learning rate schedule (cosine decay) could
help maintain the best performance found during training.

## Recommended Updated Config for Stage 1

Based on the sweep results, here's what the updated `stage1_balance.toml`
could look like:

```toml
[env]
alive_bonus = 5.0            # was 2.0 — strongest signal for balance
nosedive_weight = 2.5        # was 1.5 — prevents forward pitch
posture_weight = 1.5         # unchanged — already in sweet spot
energy_penalty_weight = 0.02 # was 0.05 — let the agent use energy freely

[ppo]
learning_rate = 5e-5         # was 3e-4 — more stable convergence
gamma = 0.975                # was 0.99 — shorter horizon for balance
n_epochs = 10                # was 6 — extract more from each rollout
ent_coef = 0.001             # was 0.005 — less random exploration needed

[ppo.policy_kwargs]
net_arch = [512, 256]        # was [256, 256] — tapered arch works best
```

## Caveats

1. **No spin/drift penalties in this sweep**: The recent `spin_penalty_weight`
   and `drift_penalty_weight` additions were not included. Some top-performing
   trials may have achieved high rewards via spinning or drifting exploits that
   would now be penalized. A follow-up sweep with these corrections is
   recommended.

2. **Stage 1 only**: These results don't tell us how well these configs
   transfer to stages 2 and 3. A high alive_bonus in stage 1 may create
   overly conservative policies that resist transitioning to locomotion.

3. **4M timestep budget**: The sweep used 4M steps vs the full 6M in the
   curriculum config. Some trials might have improved with more training time.

4. **Single-stage evaluation**: The sweep doesn't measure downstream
   curriculum success. The "best" stage 1 config is the one that produces
   a policy that *also* succeeds in stages 2 and 3.
