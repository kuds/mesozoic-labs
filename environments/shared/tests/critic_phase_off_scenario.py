"""The critic phase's off path, run on a code tree: what main's trainer computes with no ``[ppo] critic_phase`` key.

usage: python -I -B critic_phase_off_scenario.py TREE OUT_JSON [--write GOLDEN --source-commit SHA]

Not a test module (pytest does not collect it): ``test_critic_phase_off_path.py`` runs it in a subprocess on the
repository root and compares its output with ``fixtures/critic_phase_off/golden_4c9d5e9b.json``, which this same
script wrote with ``--write`` on a ``git archive`` export of main at 4c9d5e9b, before the critic phase existed.  It
uses only APIs that commit has (``train_base.train``, ``config.load_all_stages``,
``species_registry.get_species_config``), so the golden is main's own behaviour, not this tree's.

Three tiny CPU runs of the notebook's trainer entry (``train_base.train`` with ``report_metrics=False``,
``save_on_interrupt=False``) on the committed T. rex stages, ``torch.set_num_threads(1)``, n_envs 2, seed 42,
n_steps 256, batch_size 128:

- S1: the stance from scratch, 1024 steps (2 updates), a periodic checkpoint at 512;
- S2: locomotion warm-started from S1's final checkpoint under ``initialize_next_stage`` (the node's edge), 1024
  steps;
- S3: a same-stage resume of S1's 512-step checkpoint in a copy of S1's stage directory, 512 more steps.

Recorded per run, as ``sha256(dtype | shape | bytes)`` digests ("digests", comparable only in the environment the
golden was made in): before each update the rollout buffer (observations, actions, rewards, episode starts,
values, log-probs, returns, advantages), by wrapping ``PPO.train``; after training every policy parameter, every
optimizer tensor, the VecNormalize statistics, ``num_timesteps`` and ``_n_updates``; and the final checkpoint's
VecNormalize sidecar bytes.  Recorded as plain values ("structure", comparable anywhere): the model, policy,
rollout-buffer and VecNormalize classes, the stage_config.json key sets and recipe digest, the final archive's
``data`` keys and ``policy_kwargs``, and whether ``environments.shared.critic_phase_ppo`` was ever imported.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

BUFFER_FIELDS = ("observations", "actions", "rewards", "episode_starts", "values", "log_probs", "returns", "advantages")
PHASE_MODULE = "environments.shared.critic_phase_ppo"


def digest(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}|{array.shape}|".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def environment() -> dict[str, Any]:
    """What the digests depend on beyond the code: compared field by field before any digest is."""
    import gymnasium
    import mujoco
    import numpy as np
    import stable_baselines3
    import torch

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__,
        "mujoco": mujoco.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }


def _record(model: Any, updates: list[dict[str, Any]], stage_dir: Path, final: str) -> dict[str, Any]:
    import numpy as np

    digests: dict[str, str] = {}
    for index, update in enumerate(updates):
        for name, value in update.items():
            digests[f"update{index}/{name}"] = digest(value)
    for name, value in model.policy.state_dict().items():
        digests[f"param/{name}"] = digest(value.detach().cpu().numpy())
    for key, state in model.policy.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            digests[f"optimizer/{key}/{name}"] = digest(np.asarray(value))
    venv = model.get_vec_normalize_env()
    for name in ("mean", "var", "count"):
        digests[f"obs_rms/{name}"] = digest(getattr(venv.obs_rms, name))
        digests[f"ret_rms/{name}"] = digest(getattr(venv.ret_rms, name))
    digests["num_timesteps"] = digest(np.int64(model.num_timesteps))
    digests["n_updates"] = digest(np.int64(model._n_updates))
    models = stage_dir / "models"
    digests["sidecar"] = hashlib.sha256((models / f"{final}_vecnorm.pkl").read_bytes()).hexdigest()

    recorded = json.loads((stage_dir / "stage_config.json").read_text())
    with zipfile.ZipFile(models / f"{final}.zip") as archive:
        data = json.loads(archive.read("data"))
    for name in ("policy_class", "rollout_buffer_class"):
        # The classes are pickled by reference: these bytes name the module and the class.
        digests[f"archive/{name}"] = hashlib.sha256(data[name][":serialized:"].encode()).hexdigest()
    structure = {
        "model": type(model).__name__,
        "policy": f"{type(model.policy).__module__}.{type(model.policy).__qualname__}",
        "rollout_buffer": f"{type(model.rollout_buffer).__module__}.{type(model.rollout_buffer).__qualname__}",
        "vec_normalize": f"{type(venv).__module__}.{type(venv).__qualname__}",
        "buffer_obs_shape": list(model.rollout_buffer.obs_shape),
        "updates": len(updates),
        "num_timesteps": int(model.num_timesteps),
        "stage_config_keys": sorted(recorded),
        "run_keys": sorted(recorded.get("run", {})),
        "hyperparameter_keys": sorted(recorded.get("hyperparameters", {})),
        "curriculum_keys": sorted(recorded.get("curriculum", {})),
        "hyperparameters_sha256": recorded.get("run", {}).get("hyperparameters_sha256"),
        "load_mode": recorded.get("run", {}).get("load_mode"),
        "archive_data_keys": sorted(data),
        "archive_policy_module": data["policy_class"].get("__module__"),
        "archive_buffer_module": data["rollout_buffer_class"].get("__module__"),
        "archive_buffer_kwargs": data["rollout_buffer_kwargs"],
        "archive_policy_kwargs": sorted(data["policy_kwargs"]) if isinstance(data.get("policy_kwargs"), dict) else None,
        "model_files": sorted(path.name for path in models.iterdir()),
    }
    return {"digests": digests, "structure": structure}


def run(tree: Path, work: Path) -> dict[str, Any]:
    sys.path.insert(0, str(tree))
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from environments.shared import train_base
    from environments.shared.config import load_all_stages
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import get_species_config

    if not Path(train_base.__file__).resolve().is_relative_to(tree):
        raise SystemExit(f"train_base was imported from {train_base.__file__}, not from {tree}")
    torch.set_num_threads(1)
    model_id = ModelId("tyrannosaurus_rex", "anatomical")
    species = get_species_config(model_id)
    committed = load_all_stages(model_id)
    configs = copy.deepcopy(committed)
    for stage in (1, 2):
        configs[stage]["ppo_kwargs"].update(n_steps=256, batch_size=128)

    updates: list[dict[str, Any]] = []
    original = PPO.train

    def recording_train(self: Any) -> None:
        buffer = self.rollout_buffer
        updates.append({name: np.array(getattr(buffer, name), copy=True) for name in BUFFER_FIELDS})
        original(self)

    PPO.train = recording_train  # type: ignore[method-assign]
    common: dict[str, Any] = {
        "n_envs": 2,
        "seed": 42,
        "eval_freq": 10**9,
        "verbose": 0,
        "algorithm": "ppo",
        "use_tensorboard": False,
        "report_metrics": False,
        "save_on_interrupt": False,
    }
    scenarios: dict[str, Any] = {}

    s1 = work / "s1_stance"
    model = train_base.train(species, configs, 1, 1024, save_freq=512, output_dir=str(s1), **common)
    scenarios["S1"] = _record(model, updates, s1, "stage1_final")

    updates.clear()
    s2 = work / "s2_locomotion"
    model = train_base.train(
        species,
        configs,
        2,
        1024,
        save_freq=10**9,
        output_dir=str(s2),
        load_path=str(s1 / "models" / "stage1_final.zip"),
        task_load_mode="initialize_next_stage",
        **common,
    )
    scenarios["S2"] = _record(model, updates, s2, "stage2_final")

    updates.clear()
    s3 = work / "s3_resume"
    shutil.copytree(s1, s3)
    for stale in (s3 / "models").glob("stage1_final*"):
        stale.unlink()
    model = train_base.train(
        species,
        configs,
        1,
        512,
        save_freq=10**9,
        output_dir=str(s3),
        load_path=str(s3 / "models" / "stage1_512_steps.zip"),
        task_load_mode="resume_same_stage",
        **common,
    )
    scenarios["S3"] = _record(model, updates, s3, "stage1_final")

    return {
        "environment": environment(),
        "phase_module_imported": PHASE_MODULE in sys.modules,
        "scenarios": scenarios,
    }


def main(argv: list[str]) -> int:
    tree, out = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    work = out.parent / f"{out.stem}_runs"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    result = run(tree, work)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    if "--write" in argv:
        golden = Path(argv[argv.index("--write") + 1])
        source = argv[argv.index("--source-commit") + 1]
        golden.write_text(json.dumps({"source_commit": source, **result}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {out}: phase module imported {result['phase_module_imported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
