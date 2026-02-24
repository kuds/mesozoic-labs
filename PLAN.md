# Plan: Bring Training Scripts to Notebook Parity + Persistent Storage

## Overview

Two deliverables:
1. **Script parity** — Port all notebook training features into shared modules and update all 3 species' `train_sb3.py` scripts
2. **Persistent storage** — Add `docker-compose.yml` and environment variable support for local Docker and Vertex AI/GCS

## Architecture

Rather than duplicating code 3x, extract shared training logic into two new modules under `environments/shared/`. The species scripts become thin wrappers that supply species-specific configuration.

---

## New Files

### 1. `environments/shared/diagnostics.py` — DiagnosticsCallback

Species-agnostic SB3 callback. Takes `reward_keys` and `info_keys` as constructor arguments so each species supplies its own list.

Features (from notebooks):
- Per-component reward breakdown → TensorBoard `diagnostics/reward_*`
- Environment state metrics → TensorBoard `diagnostics/<info_key>`
- Observation stats (mean, std, max abs) from rollout buffer
- Action stats (mean, std) from rollout buffer
- VecNormalize running variance tracking (`obs_rms.var`, `ret_rms.var`)
- Termination reason breakdown → TensorBoard `terminations/<reason>`
- Plateau detection with configurable window/threshold + console warning
- `diagnostics.npz` persistence of per-rollout info_key averages

### 2. `environments/shared/train_utils.py` — Shared training engine

Consolidates all duplicated training logic:

- `linear_schedule(initial_lr, final_lr)` — LR decay callable
- `create_vec_env(env_class, stage_configs, stage, n_envs, seed, use_subproc, vecnorm_path=None, clip_reward=10.0)` — with VecNormalize carry-over
- `prepare_algo_kwargs(config, algorithm, verbose, tensorboard_log)` — extracts `policy_kwargs`, handles `learning_rate_end`, pops non-algorithm keys
- `create_model(algorithm, config, train_env, load_path, algo_kwargs, policy_kwargs)` — PPO or SAC creation/loading
- `evaluate_with_forward_vel(model, env_class, stage_configs, stage, n_episodes, seed, vecnorm_path)` — per-episode forward velocity collection
- `build_callbacks(model_dir, stage_dir, stage, eval_env, eval_freq, save_freq, n_envs, diagnostics_reward_keys, diagnostics_info_keys, verbose)` — EvalCallback + CheckpointCallback + DiagnosticsCallback
- `write_training_summary(run_dir, stage_results_list, species, algorithm, seed, n_envs)` — text summary file
- `save_results_json(run_dir, stage_results_list, species, algorithm, seed)` — machine-readable JSON
- `record_stage_video(model, env_class, stage_configs, stage, stage_dir, vecnorm_path, seed)` — mp4 recording (optional, depends on mediapy)
- `plot_training_curves(stage_dirs, stage_configs, algo_name, species, save_path)` — 2x2 matplotlib grid (optional, depends on matplotlib)

### 3. `docker-compose.yml` — Local Docker training

Services:
- `train` — single-stage training with bind-mounted `./outputs`
- `curriculum` — full 3-stage curriculum
- Environment variables: `LOG_DIR`, `WANDB_API_KEY`, `WANDB_PROJECT`, `SPECIES`, `STAGE`

---

## Modified Files

### 4–6. `environments/{trex,velociraptor,brachiosaurus}/scripts/train_sb3.py`

Rewrite each to use shared modules. Each script becomes ~150 lines:

**Species-specific constants:**
- `SPECIES` name, `ENV_CLASS`, `STAGE_CONFIGS`
- `REWARD_KEYS` and `INFO_KEYS` for DiagnosticsCallback
- `CLIP_REWARD` (10.0 for trex/brachio, 50.0 for raptor)

**New CLI flags (all 3 scripts):**
- `--algo {PPO,SAC}` (default PPO) — algorithm selection
- `--wandb` — enable W&B logging
- `--no-video` — skip video recording in curriculum mode

**Shared behavior (via train_utils):**
- DiagnosticsCallback active in all training
- VecNormalize carry-over between curriculum stages
- Linear LR schedule from `learning_rate_end` in TOML
- `policy_kwargs` / `net_arch` extraction from TOML
- SAC support alongside PPO
- W&B integration via existing `wandb_integration.py`
- Post-curriculum: evaluate_with_forward_vel, training summary, results JSON, video recording, training curves PNG
- Environment variable fallbacks: `LOG_DIR` env var overrides `--log-dir` default

---

## Detailed Per-Species DiagnosticsCallback Keys

**T-Rex:**
- REWARD_KEYS: `reward_forward, reward_alive, reward_energy, reward_tail, reward_bite, reward_approach, reward_total`
- INFO_KEYS: `forward_vel, tilt_angle, pelvis_height, tail_instability, bite_success, prey_distance, approach_delta`

**Velociraptor:**
- REWARD_KEYS: `reward_forward, reward_alive, reward_energy, reward_tail, reward_posture, reward_nosedive, reward_smoothness, reward_strike, reward_approach, reward_gait, reward_heading, reward_lateral`
- INFO_KEYS: `forward_vel, prey_distance, pelvis_height, tilt_angle, tail_instability, contact_asymmetry, action_delta, heading_alignment, lateral_vel, forward_z, approach_delta`

**Brachiosaurus:**
- REWARD_KEYS: `reward_forward, reward_alive, reward_energy, reward_gait, reward_food, reward_approach, reward_total`
- INFO_KEYS: `forward_vel, tilt_angle, torso_height, gait_instability, head_food_distance, food_reached, approach_delta`

---

## Implementation Order

1. Create `environments/shared/diagnostics.py`
2. Create `environments/shared/train_utils.py`
3. Update `environments/trex/scripts/train_sb3.py`
4. Update `environments/velociraptor/scripts/train_sb3.py`
5. Update `environments/brachiosaurus/scripts/train_sb3.py`
6. Create `docker-compose.yml`
7. Verify imports work (run `python -c "from environments.shared.diagnostics import DiagnosticsCallback"`)
