"""Pilot 3's code (31d02e42, CHECK/tree_p3) side of the env identity check for ONE seed. CPU only, nothing trains.

Usage (cwd = CHECK/tree_p3):
    ISO_WT=CHECK/tree_p3 PYTHONPATH=CHECK/scripts python -I CHECK/tools/iso.py CHECK/scripts/p3_side.py SEED

1. Pilot 3's own locomotion [env] (31d02e42's load_stage_config) against the built tree's "p3x" kwargs
   (the built [env] minus the five pilot-4 keys).
2. Replays, in pilot 3's env code, the two action sequences the built tree recorded: the built rollout's actions in
   the twin [env] (pilot 3's [env] with prey_lateral_range [0, 0]) and pilot 3's walker's closed-loop actions in
   pilot 3's exact [env]; every obs, reward, info value, qpos, qvel and flag is compared bit for bit with the built
   tree's replay of the same actions.
3. Pilot 3's walker closed loop in pilot 3's exact env under pilot 3's code: its actions against the built tree's
   closed loop (bit for bit) and the heading metrics.
Writes CHECK/runs/s<SEED>/p3_side.json only.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from p4c_common import EPISODE_SEED, MAX_STEPS, C, heading_metrics, jsonable, my_pair, replay, same

SEED = int(sys.argv[1])
OUT = Path(C) / "runs" / f"s{SEED}"

import torch  # noqa: E402

torch.set_num_threads(1)

import environments  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.policy_loading import load_sb3_model  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402

MID = ModelId("tyrannosaurus_rex", "anatomical")
EnvClass = get_species_config(MID).env_class
res = {"seed": SEED, "environments_imported_from": environments.__file__}
kw = json.loads((OUT / "kwargs.json").read_text())
p3_env = load_stage_config(MID, 2)["env_kwargs"]
res["pilot3_env_kwargs_equal_built_p3x"] = same(jsonable(p3_env), kw["p3x"])
res["pilot3_env_kwargs_diff"] = {k: [jsonable(p3_env.get(k)), kw["p3x"].get(k)] for k in sorted(set(p3_env) | set(kw["p3x"]))
                                 if not same(jsonable(p3_env.get(k)), kw["p3x"].get(k))}


def compare(a: dict, b: dict) -> dict:
    out = {"steps": [len(a["reward"]), len(b["reward"])]}
    for field in ("obs", "reward", "terminated", "truncated", "qpos", "qvel", "heading", "xy"):
        out[f"{field}_mismatch_rows"] = sum(not same(x, y) for x, y in zip(a[field], b[field]))
    keys_mismatch, key_sets = {}, 0
    for ia, ib in zip(a["info"], b["info"]):
        key_sets += int(set(ia) != set(ib))
        for k in ia:
            if k in ib and not same(ia[k], ib[k]):
                keys_mismatch[k] = keys_mismatch.get(k, 0) + 1
    out["info_key_set_mismatch_rows"] = key_sets
    out["info_value_mismatch_counts"] = keys_mismatch
    out["n_info_keys"] = len(a["info"][0]) if a["info"] else 0
    out["bit_identical"] = (out["steps"][0] == out["steps"][1] and key_sets == 0 and not keys_mismatch
                            and all(v == 0 for k, v in out.items() if k.endswith("_mismatch_rows")))
    return out


rec_twin = replay(EnvClass(**kw["twin"]), np.load(OUT / "actions_built_rollout.npy"), EPISODE_SEED)
rec_p3x = replay(EnvClass(**p3_env), np.load(OUT / "actions_p3x_closed_loop.npy"), EPISODE_SEED)
res["twin_replay_vs_built_tree"] = compare(json.loads((OUT / "replay_twin_built_tree.json").read_text()), rec_twin)
res["p3x_replay_vs_built_tree"] = compare(json.loads((OUT / "replay_p3x_built_tree.json").read_text()), rec_p3x)
# postfix/code: the new walker's live-command action sequence, replayed in pilot 3's exact [env] in both trees.
rec_p3x_b = replay(EnvClass(**p3_env), np.load(OUT / "actions_built_rollout.npy"), EPISODE_SEED)
res["p3x_builtactions_replay_vs_built_tree"] = compare(
    json.loads((OUT / "replay_p3x_builtactions_built_tree.json").read_text()), rec_p3x_b
)

zip_path, pkl_path = my_pair(SEED)
orig = load_sb3_model(zip_path, algorithm=PPO, device="cpu")
with open(pkl_path, "rb") as f:
    vn = pickle.load(f)
env = EnvClass(**p3_env)
obs, _ = env.reset(seed=EPISODE_SEED)
acts, hs, xy = [], [env._heading()], [env.data.qpos[0:2].copy()]
for _ in range(MAX_STEPS):
    a = orig.predict(vn.normalize_obs(np.asarray(obs, dtype=np.float32))[None], deterministic=True)[0][0]
    acts.append(a)
    obs, r, te, tr, info = env.step(a)
    hs.append(env._heading())
    xy.append(env.data.qpos[0:2].copy())
    if te or tr:
        break
built = np.load(OUT / "actions_p3x_closed_loop.npy")
acts = np.asarray(acts, dtype=np.float32)
res["pilot3_closed_loop"] = {
    "steps": len(acts), "terminated": bool(te),
    "actions_bit_equal_built_tree_closed_loop": bool(acts.shape == built.shape and np.array_equal(acts, built)),
    "metrics": heading_metrics(hs, xy),
}
with open(OUT / "p3_side.json", "w") as f:
    json.dump(res, f, indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
