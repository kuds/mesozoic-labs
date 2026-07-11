# T-Rex JAX/MJX Stage 1 Review — 2026-04-01

> **Run:** `20260401_141554` | **Algorithm:** JAX PPO | **GPU:** A100-SXM4-80GB
> **Steps:** 65.5M (160/500 updates before KL halt) | **Duration:** 43m 28s
> **Best eval:** 992.7 @ update 132 | **Final eval:** 520.7 ± 63.8
>
> **Postscript (2026-04-18):** The bugs identified below are **fixed**.
> Termination body-height checks in `environments/trex/mjx_config.py` now
> include `torso` and other missing bodies. JAX reward-signal unification
> (commit `1957671`, 2026-04-03) brought the JAX path into parity with the
> SB3 `reward_functions.py`, and `a6d906f` restored the `fall_penalty` /
> `reset_noise_scale` overrides. Additional stabilization landed Apr 2–3
> (PPO ratio explosion, value-loss domination, dones broadcasting, LR
> schedule bug). This review is preserved for historical context; a re-run
> is needed to confirm convergence on current main.

---

## Verdict: FAIL — Agent learned to lie down and collect alive bonus

The T-Rex never learns to stand or balance. It falls to the ground within the
first few timesteps and remains prone for the entire episode (1000 steps),
exploiting a **broken termination condition** that fails to detect ground
contact. Training was halted at update 160 by KL divergence explosion (1.18e+06).

---

## Root Cause Analysis

### Bug 1: Missing `torso` in MJX termination body checks (PRIMARY)

**File:** `environments/trex/mjx_config.py:34-39`

The MJX path uses position-based body-height checks as a JAX-compatible
alternative to the CPU path's contact-pair iteration. The current config only
monitors:

```python
termination_body_heights={
    "skull": 0.12,
    "tail_3": 0.10,
    "tail_4": 0.08,
    "tail_5": 0.05,
}
```

**The `torso` body is completely missing.** When the T-Rex falls on its side or
belly, the torso geom (capsule radius 0.18) rests on the ground, but the
*pelvis body origin* stays at z ≈ 0.5m (propped up by the capsule radius).
Since `healthy_z_range[0] = 0.5`, the pelvis barely avoids the height
termination threshold, and no other check catches the fall.

The CPU Gymnasium path (`trex_env.py:479-504`) correctly terminates on torso
floor contact via contact-pair iteration — but that code path isn't used during
MJX GPU training.

**Fix:** Add torso (and ribcage/belly) to `termination_body_heights`:

```python
termination_body_heights={
    "skull": 0.15,       # capsule radius 0.10 + margin
    "torso": 0.25,       # capsule radius 0.18 + margin — MISSING
    "tail_3": 0.10,
    "tail_4": 0.08,
    "tail_5": 0.05,
},
```

### Bug 2: `healthy_z_range[0] = 0.5` is too low

**File:** `environments/trex/mjx_config.py:21`

The pelvis spawns at z = 0.90m. When the T-Rex is fully collapsed on the
ground, the pelvis body origin sits at z ≈ 0.50m due to the torso capsule
geometry acting as a prop. This means a clearly-fallen dinosaur *just barely*
avoids the height termination threshold.

**Fix:** Raise to 0.60–0.65m. A standing T-Rex has pelvis at 0.85–0.95m, so
0.60 gives ample margin for normal movement while catching falls.

### Issue 3: Alive bonus (1.2) dominates the reward landscape

**File:** `stage_config.json` → `reward_weights.alive_bonus = 1.2`

With no termination on falling, the per-step reward breakdown for a prone T-Rex
is approximately:

| Component | Value | Notes |
|-----------|-------|-------|
| alive_bonus | +1.20 | Constant, every step |
| height_maintenance | +0.00 | `2.0 * clip((0.5 - 0.5)/0.4, 0, 1) = 0` |
| posture | −0.28 | `−2.5 * (0.35/1.047)² ≈ −0.28` at ~20° tilt |
| nosedive | −0.05 | Small at moderate pitch |
| energy | −0.01 | Near zero (agent learns not to move) |
| drift/speed/spin | ~0.00 | Not drifting, not moving |
| **Total** | **≈ +0.86** | |

This matches the observed ~1.0 reward/step. The agent reaches a **degenerate
local optimum**: do nothing, collect 1.2 alive bonus per step × 1000 steps ≈
1000 episode return. This is confirmed by the episode return plateau at ~900-1000.

**Fix:** Reduce `alive_bonus` to 0.3–0.5 for stage 1, or make it conditional
on being "healthy" (standing). The alive bonus should incentivize *not dying*,
not reward existence unconditionally.

### Issue 4: Eval shows no movement — correct but misleading

The eval rollout correctly reflects the learned policy: the agent does nothing.
Forward velocity hovers at ~0 m/s, foot contacts are near-zero (sparse
collision spikes, not deliberate stepping), and pelvis height collapses to 0.5m
and stays there.

The eval setup itself is working correctly — it uses the same position-based
termination as MJX training (`jax_eval.py:224-237`), so the behavior is
consistent between training and evaluation.

---

## Training Dynamics

### Phase 1 (updates 0–40): Rapid reward climb
- Reward/step jumps from 0.4 → 0.7 as agent discovers "do nothing" strategy
- Entropy drops from 29.8 → 25.9 (policy narrowing quickly)
- Fall rate drops to <1% — not because balancing improved, but because
  termination never triggers

### Phase 2 (updates 40–130): Plateau at degenerate optimum
- Reward/step ≈ 1.0, episode return ≈ 900-1000
- Episode length maxes out at 1000 (truncation, never termination)
- Gradient norm stable at ~20, loss slowly decreasing
- Agent has fully converged on the "lie down" policy

### Phase 3 (updates 125–160): KL explosion and halt
- KL warnings start at update 125 (kl=161)
- Escalating: 1.35e4 → 2.24e3 → 2.83e3 → 2.27e5 → 1.18e6
- Gradient norm spikes to 175
- **Cause:** Policy is in a flat plateau with near-zero gradients for useful
  actions. Stochastic updates occasionally push the policy far from the
  current mode, causing massive ratio changes. The clip range (0.2) can't
  contain these jumps because the baseline policy is so narrow (entropy ≈ 23.7)
  in the degenerate region.

---

## Recommended Fixes (Priority Order)

### 1. Fix termination — add torso body-height check

```python
# environments/trex/mjx_config.py
termination_body_heights={
    "skull": 0.15,
    "torso": 0.25,       # ADD THIS
    "tail_3": 0.10,
    "tail_4": 0.08,
    "tail_5": 0.05,
},
```

### 2. Raise healthy_z_range lower bound

```python
healthy_z_range=(0.65, 1.6),  # was (0.5, 1.6)
```

### 3. Reduce alive_bonus for stage 1

```python
# stage_config.json or stage1_balance.toml
"alive_bonus": 0.4,  # was 1.2
```

### 4. Increase height_weight to make standing more valuable

```python
"height_weight": 4.0,  # was 2.0
```

This makes the standing vs lying reward gap:
- Standing (z=0.90): alive(0.4) + height(4.0) = +4.4/step
- Lying (z=0.50): alive(0.4) + height(0.0) = +0.4/step (if not terminated)
- Clear 10x incentive to stand, plus termination penalty on fall

### 5. Consider early stopping on KL divergence

Instead of halting at catastrophic KL (1e6), revert to the last checkpoint and
reduce LR when KL exceeds `target_kl` by 10x for 3 consecutive updates. The
current setup lets the policy degrade significantly before intervening.

---

## Key Metrics Summary

| Metric | Value | Expected (healthy) |
|--------|-------|--------------------|
| Mean reward/step | 1.0 | 0.5–2.0 (with height reward) |
| Episode return | 900–1000 | Variable (with falls) |
| Fall rate | 0% | 10–30% during learning |
| Mean episode length | ~1000 (max) | 300–800 during learning |
| Forward velocity | ~0 m/s | 0 m/s (stage 1 = balance only) |
| Pelvis height | ~0.5m (floor) | 0.85–0.95m (standing) |
| Entropy | 23.7 | 24–28 (exploring) |
| KL at halt | 1.18e6 | < 0.05 (target) |

---

## Comparison with Previous Stable-Baselines3 Runs

The earlier SB3 PPO runs (Feb 22–27) successfully solved Stage 1 with reward
2994.34 because they used the **CPU Gymnasium path** with contact-pair
termination. The MJX migration introduced the position-based termination
approximation, which works for the skull and tail extremities but fails to
catch torso ground contact — the most common fall mode for a top-heavy biped.
