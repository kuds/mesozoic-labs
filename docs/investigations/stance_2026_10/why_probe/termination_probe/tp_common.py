"""Termination probe (lens B): shared paths, the checkpoint table and the loaders.  No rollout statistics here.

Imported by probe_static.py, run_probe.py and the analysis scripts (all in this folder).  The simulation helpers import
the sealed code copy (S2/r10round/bias_screen/code, read only) first on sys.path; nothing is ever written under it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ME = Path(S2) / "com_revisit/notes/termination_probe"
CODE = S2 + "/r10round/bias_screen/code"                     # sealed copy, commit 8737cd3d, read only
BS2 = S2 + "/r10round/bias_screen2"
F2 = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance"
OBS_TAIL3 = 6          # observation index of the tail_3 pitch angle (rad); checked against the model in probe_static
ACT_TAIL3 = 14         # action index of tail_3_pitch_act; checked against the model
GAMMA = 0.98
HORIZON = 1000

# The checkpoint table (label -> zip, pkl, stage_config.json, critic form).  Chosen from the task text before any
# rollout: the parent s42 at 1,048,576 and 2,097,152 (F2 copies), BS2's control_2511 and control_2531 at 3,145,728 and
# at E = 3,538,944 (stage1_final), and one blinded run at E for contrast (critic_blind_tail3_2511, the seed paired
# with control_2511).
CKPT = {
    "p1048576": (f"{F2}/models/stage1_1048576_steps.zip.copy1", f"{F2}/models/stage1_vecnormalize_1048576_steps.pkl.copy1",
                 f"{F2}/stage_config.json.copy1", "plain"),
    "p2097152": (f"{F2}/models/stage1_2097152_steps.zip.copy1", f"{F2}/models/stage1_vecnormalize_2097152_steps.pkl.copy1",
                 f"{F2}/stage_config.json.copy1", "plain"),
}
for _seed in (2511, 2531):
    _m = f"{BS2}/runs/control_{_seed}/01_stance"
    CKPT[f"c{_seed}_3145728"] = (f"{_m}/models/stage1_3145728_steps.zip", f"{_m}/models/stage1_vecnormalize_3145728_steps.pkl",
                                 f"{_m}/stage_config.json", "plain")
    CKPT[f"c{_seed}_E"] = (f"{_m}/models/stage1_final.zip", f"{_m}/models/stage1_final_vecnorm.pkl",
                           f"{_m}/stage_config.json", "plain")
_b = f"{BS2}/runs/critic_blind_tail3_2511/01_stance"
CKPT["b2511_E"] = (f"{_b}/models/stage1_final.zip", f"{_b}/models/stage1_final_vecnorm.pkl", f"{_b}/stage_config.json",
                   "blind")
ORDER = ("p1048576", "p2097152", "c2511_3145728", "c2531_3145728", "c2511_E", "c2531_E", "b2511_E")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup_simulation(tag: str) -> str:
    """Threads 1, TMPDIR under ME/tmp/<tag>, the code copy first on sys.path; returns the environments origin."""
    tmp = ME / "tmp" / tag
    (tmp / "mpl").mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "TMPDIR": str(tmp), "MPLCONFIGDIR": str(tmp / "mpl"),
                       "GITHUB_SHA": "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"})
    import tempfile

    tempfile.tempdir = None
    code = os.path.realpath(CODE)
    if sys.path[:1] != [code]:
        sys.path.insert(0, code)
    import environments

    origin = os.path.realpath(environments.__file__)
    if not origin.startswith(code + os.sep):
        raise SystemExit(f"environments imported from {origin}, not the code copy {code}")
    import torch

    torch.set_num_threads(1)
    return origin


def stance_ctor() -> dict:
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId

    return dict(load_stage_config(ModelId("tyrannosaurus_rex", "anatomical"), "stance")["env_kwargs"])


def env_factory():
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    ctor = stance_ctor()
    mid = ModelId("tyrannosaurus_rex", "anatomical")

    def factory():
        return env_class(mid)(**ctor)
    return factory


def ctor_matches_run(stage_config_path: str) -> tuple[bool, list]:
    """Compare the code copy's stance env_kwargs with a run's recorded reward_weights (its full env kwargs)."""
    ctor = stance_ctor()
    rec = json.loads(Path(stage_config_path).read_text())["reward_weights"]
    diffs = []
    for k in sorted(set(ctor) | set(rec)):
        a, b = ctor.get(k, "<absent>"), rec.get(k, "<absent>")
        if isinstance(a, tuple):
            a = list(a)
        if a != b:
            diffs.append((k, a, b))
    return (not diffs), diffs


def load_pair(label: str, factory):
    """(model, normalizer) through the reader's own loader (policy_loading.load_sb3_checkpoint; frozen statistics)."""
    from environments.shared.policy_loading import load_sb3_checkpoint

    z, p, _cfg, _form = CKPT[label]
    model, norm, _ = load_sb3_checkpoint(str(z), str(p), factory)
    return model, norm


def tail_capsule_lowest(model, data, geom_ids) -> list:
    """Lowest point (world z) of each capsule geom: centre z - |half-length * axis_z| - radius."""
    out = []
    for g in geom_ids:
        zc = float(data.geom_xpos[g][2])
        axz = float(data.geom_xmat[g].reshape(3, 3)[2, 2])
        r, hl = float(model.geom_size[g][0]), float(model.geom_size[g][1])
        out.append(zc - abs(hl * axz) - r)
    return out
