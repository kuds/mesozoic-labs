# RL Training Plan

> **Date:** 2026-03-26 *(last reviewed 2026-04-18)*
> **Scope:** Remaining SAC & PPO training across all species
> **Notebooks:** `notebooks/sb3_training.ipynb`, `notebooks/ray_tune_sweep.ipynb`
>
> **Update (2026-04-18):** Active priority shifted to JAX training
> stabilization (Apr 2–3) and then to the SAC VecNormalize fix (Apr 18).
> The trial plan below is a dated planning snapshot. Before resuming SB3 trials,
> factor in (1) the SAC reward-normalization fix (see
> `environments/shared/train_base.py`, which now disables `norm_reward` for
> SAC), and (2) the pending Stage 3 terminal-bonus rescale documented in
> [investigations/REWARD_SCALE_REDESIGN.md](investigations/REWARD_SCALE_REDESIGN.md). Running SAC trials
> before the rescale will test the normalization fix in isolation; running
> them after will confound the two changes.

---

## Public Result Baseline

Do not copy run metrics or current stage budgets into this planning document.
The generated [public catalog](../README.md#training-results) is the authoritative
view of current configs and provenance-labelled result summaries. At this plan's
last review, T-Rex SAC and Brachiosaurus SAC had no published summary, and the
published Brachiosaurus PPO Stage 3 result had not met its configured gate.

---

## Remaining Trials (4 total)

### Trial 1: T-Rex SAC

**Notebook:** `notebooks/sb3_training.ipynb`

**Why the SB3 notebook (not Ray Tune):** The historical T-Rex PPO summary provides
a baseline for the configured fixed-head contact criterion. It does not validate
articulated biting, because the model has no jaw joint. The SAC hyperparameters in the TOML configs are
well-reasoned. The historical Velociraptor SAC summary records all three stages
as passed under that run's evaluation; it is a starting reference, not evidence
that T-Rex will behave similarly.

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `trex` |
| Algorithm | `SAC` |
| Stages | All 3 (curriculum auto-advances) |
| Stage budgets | Use the current TOML values shown in the generated catalog |
| Runtime sizing | Start conservatively and measure with a short pilot; no hardware minimum or runtime table has been validated |

Load SAC hyperparameters and environment settings directly from the current
`configs/trex/stage*.toml` files. Record any notebook overrides with the run;
this dated plan intentionally does not copy the values.

---

### Trial 2: Brachiosaurus PPO Stage 3 Sweep

**Notebook:** `notebooks/ray_tune_sweep.ipynb`

**Why Ray Tune (not the SB3 notebook):** The historical Stage 3 `food_reach`
summary did not meet its configured success gate. A systematic sweep is the
next experiment to consider. This criterion measures head-tip distance to the
food target; it does not require physical contact.

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `brachiosaurus` |
| Algorithm | `PPO` |
| Stage | `3` only (load a provenance-complete Stage 2 checkpoint) |
| Runtime/search settings | Resolve from `configs/brachiosaurus/sweep_ppo.json`; validate with one trial before increasing concurrency |

The species sweep JSON is the authoritative search space and job-settings
source. Save the resolved snapshot emitted by the notebook with every sweep.

---

### Trial 3: Brachiosaurus SAC (All Stages)

**Notebook:** `notebooks/sb3_training.ipynb`

**Why the SB3 notebook (not Ray Tune):** The SAC configs in the TOMLs are already written with
thoughtful rationale. SAC's replay buffer and auto-entropy tuning may naturally handle two
problems that plague Brachiosaurus PPO:

1. **Catastrophic forgetting between stages** -- buffer retains old transitions
2. **Discovering novel neck-extension behavior in Stage 3** -- auto-entropy maintains exploration

Try the defaults first.

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `brachiosaurus` |
| Algorithm | `SAC` |
| Stages | All 3 (curriculum auto-advances) |
| Stage budgets | Use the current TOML values shown in the generated catalog |
| Runtime sizing | Measure with a short pilot; no validated hardware/runtime table is published |

Load all effective values from `configs/brachiosaurus/stage*.toml` and record
overrides. Watch the generated Stage 2 velocity gate and Stage 3 success gate
rather than relying on the dated numeric notes that previously lived here.

---

### Trial 4 (Contingency): Brachiosaurus SAC Stage 3 Sweep

**Notebook:** `notebooks/ray_tune_sweep.ipynb`

**When to run:** Only if Trial 3 passes Stages 1-2 but fails Stage 3 food_reach (same
pattern as PPO).

#### Notebook Setup

| Setting | Value |
|---------|-------|
| Species | `brachiosaurus` |
| Algorithm | `SAC` |
| Stage | `3` only (load Stage 2 checkpoint from Trial 3) |
| Runtime/search settings | Resolve from `configs/brachiosaurus/sweep_sac.json` and validate with a one-trial pilot |

---

## Recommended Execution Order

```
Trial 1: T-Rex SAC (sb3_training.ipynb)      <-- high confidence, run first
Trial 2: Brachio PPO S3 sweep (ray_tune)     <-- can run in parallel with Trial 1
Trial 3: Brachio SAC full (sb3_training.ipynb) <-- run after Trial 2 completes
Trial 4: Brachio SAC S3 sweep (ray_tune)     <-- only if Trial 3 S3 fails
```

**Parallelism:** Trials 1 and 2 are independent and can run simultaneously on separate
Colab instances. Trial 3 benefits from any insights gained from Trial 2's sweep results
(reward weight findings apply to SAC too).

## Compute Planning

Resolve stage and sweep budgets from the current TOML/JSON configs, run a
small pilot on the intended hardware, and calculate total compute from the
measured throughput. Historical timings do not have enough provenance to serve
as a hardware sizing table.
