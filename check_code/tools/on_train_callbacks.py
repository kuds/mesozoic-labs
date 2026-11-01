"""Switch on, the in-training callbacks with the phase model: StageAwareEvalCallback evaluations, checkpoints,
policy probe, diagnostics, rollout ledger and the K3 panel (each on a 128-step cadence), 2 PPO updates.
usage: on_train_callbacks.py OUTDIR"""
import json
import sys
from pathlib import Path

WT = "/home/user/work/pc_branch"
sys.path.insert(0, WT)
out = sys.argv[1]
import environments.shared.train_base as tb  # noqa: E402
from environments.shared.config import load_all_stages  # noqa: E402
from environments.shared.k3_panel import K3PanelConfig  # noqa: E402
from environments.shared.species_names import require_model_id  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest  # noqa: E402

mid = require_model_id("tyrannosaurus_rex", "anatomical")
cfg = get_species_config(mid)
st = load_all_stages(mid)
ref = load_stage_manifest(mid).resolve("stance").reference
st[ref]["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=2, critic_phase=True)
cur = st[ref]["curriculum_kwargs"]
print("curriculum cadence keys before:", {k: cur.get(k) for k in ("eval_freq", "eval_episodes", "checkpoint_freq", "min_eval_episodes")})
cur.update(eval_freq=128, eval_episodes=2, checkpoint_freq=128)
k3 = K3PanelConfig(sigma_file=f"{WT}/docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json",
                   sigma_key="source_vectors.42", every=128, seeds=(1, 2))
model = tb.train(cfg, st, ref, 256, n_envs=2, seed=42, output_dir=f"{out}/run", use_tensorboard=False, verbose=0,
                 report_metrics=False, save_on_interrupt=False, post_eval_episodes=0, k3_panel=k3)
run = Path(f"{out}/run")
print("classes:", type(model).__name__, type(model.policy).__name__, type(model.get_vec_normalize_env()).__name__)
print("files:", sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()))
import numpy as np  # noqa: E402

ev = np.load(run / "evaluations.npz")
print("evaluations.npz timesteps:", ev["timesteps"].tolist(), "results shape:", ev["results"].shape,
      "mean per eval:", ev["results"].mean(axis=1).round(2).tolist())
for name in ("k3_panel.csv", "train_episodes.csv"):
    p = run / name
    if p.exists():
        lines = p.read_text().splitlines()
        print(f"{name}: {len(lines)} lines; header {lines[0][:160]}")
        for line in lines[1:4]:
            print("   ", line[:200])
