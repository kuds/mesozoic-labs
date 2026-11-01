"""Off-path equivalence evidence: run the notebook's trainer entry (train_base.train, as notebooks/sb3_training.ipynb's
train_stage calls it: report_metrics=False, save_on_interrupt=False) on the real T. rex stance stage for 2 PPO
updates on a tiny config, from the code tree given, and dump what the run computed.

usage: python -I -B equiv_run.py TREE OUTDIR [--critic-phase]

Recorded (identically in every tree, by wrapping MirrorKLPPO.train, which the stance stage trains with): before each
update, the rollout buffer (observations, actions, rewards incl. the time-limit bootstrap, episode starts, values,
log-probs, returns, advantages); after training, every policy parameter and Adam tensor, the VecNormalize
statistics, num_timesteps and _n_updates.  The run directory (final pair, probe, diagnostics, ledger) stays in
OUTDIR/run for the byte comparison.  Nothing in the trainer is changed: the wrapper only copies arrays.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path


def main() -> int:
    tree, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    critic_phase = "--critic-phase" in sys.argv[3:]
    sys.path.insert(0, str(tree))
    import numpy as np

    from environments.shared import mirror_kl_ppo, train_base
    from environments.shared.config import load_all_stages
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import get_species_config

    assert Path(train_base.__file__).resolve().is_relative_to(tree), train_base.__file__
    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    configs = load_all_stages(model_id)
    config = copy.deepcopy(configs)
    config[1]["ppo_kwargs"].update(n_steps=256, batch_size=128)
    if critic_phase:
        config[1]["ppo_kwargs"]["critic_phase"] = True
    records: list[dict] = []
    original = mirror_kl_ppo.MirrorKLPPO.train

    def recording_train(self):
        buffer = self.rollout_buffer
        records.append(
            {
                name: np.array(getattr(buffer, name), copy=True)
                for name in (
                    "observations",
                    "actions",
                    "rewards",
                    "episode_starts",
                    "values",
                    "log_probs",
                    "returns",
                    "advantages",
                )
            }
        )
        return original(self)

    mirror_kl_ppo.MirrorKLPPO.train = recording_train
    out.mkdir(parents=True, exist_ok=True)
    model = train_base.train(
        get_species_config(model_id),
        config,
        1,
        1024,
        n_envs=2,
        seed=42,
        eval_freq=10**9,
        save_freq=10**9,
        verbose=0,
        algorithm="ppo",
        output_dir=str(out / "run"),
        use_tensorboard=False,
        task_load_mode="resume_same_stage",
        report_metrics=False,
        save_on_interrupt=False,
    )
    arrays: dict[str, np.ndarray] = {}
    for index, record in enumerate(records):
        for name, value in record.items():
            arrays[f"update{index}/{name}"] = value
    for name, value in model.policy.state_dict().items():
        arrays[f"param/{name}"] = value.detach().cpu().numpy()
    for key, state in model.policy.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            arrays[f"adam/{key}/{name}"] = np.asarray(value)
    venv = model.get_vec_normalize_env()
    for name in ("mean", "var", "count"):
        arrays[f"obs_rms/{name}"] = np.asarray(getattr(venv.obs_rms, name))
        arrays[f"ret_rms/{name}"] = np.asarray(getattr(venv.ret_rms, name))
    arrays["num_timesteps"] = np.asarray(model.num_timesteps)
    arrays["n_updates"] = np.asarray(model._n_updates)
    np.savez(out / "dump.npz", **arrays)
    print(
        f"tree={tree} critic_phase={critic_phase} updates={len(records)} num_timesteps={model.num_timesteps} "
        f"n_updates={model._n_updates} policy={type(model.policy).__name__} model={type(model).__name__} "
        f"buffer_obs={records[0]['observations'].shape} venv={type(venv).__name__}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
