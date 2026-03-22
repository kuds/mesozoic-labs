# GCP Notebook SAC Environment Recommendations

## Question

What is the recommended number of environments for an L4 Google Cloud notebook
using SAC (Soft Actor-Critic)?

## Answer

**8 parallel environments** is the recommended default for SAC training.

When the `--algorithm sac` flag is used, the training CLI automatically bumps
`n_envs` from the standard default of 4 to **8**
(`_SAC_DEFAULT_N_ENVS = 8` in `environments/shared/cli.py:159`).

### Why 8 environments for SAC?

SAC benefits from more parallel environments because:

1. **CPU-bound simulation** – MuJoCo physics steps are CPU-bound, so more
   environments keep the CPU busy while the GPU handles policy updates.
2. **Off-policy replay buffer** – Unlike PPO (on-policy), SAC stores
   transitions in a replay buffer and can reuse data from all environments
   efficiently, making additional environment throughput directly useful.

### Relevant configuration

| Parameter | Value | Source |
|---|---|---|
| `_SAC_DEFAULT_N_ENVS` | 8 | `environments/shared/cli.py:159` |
| CLI default (`--n-envs`) | 4 | `environments/shared/cli.py:93` |
| Sweep config (`n_envs`) | 2 | `configs/sweep_sac.json` (conservative for sweeps) |

### L4 GPU context

Training results in the README were obtained on a Google Colab L4 GPU. The L4
provides sufficient compute for SAC training across all three curriculum stages.
The automatic bump to 8 environments applies regardless of GPU type—it is
driven by SAC's algorithmic properties, not hardware constraints.

### Override

To use a different number of environments, pass `--n-envs <N>` explicitly:

```bash
python -m environments.trex.train --algorithm sac --n-envs 12 --stage 1
```
