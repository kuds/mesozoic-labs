"""e2e lane: the REAL set-up path to learn() for one seed, then the edge-prepared model saved as stand-in pairs.
NOTHING TRAINS.

    cd CLONE && ISO_WT=CLONE python -I TOOLS/iso.py edge_pairs.py SEED PARAMS.json OUTDIR CLONE

1. Runs the pilot-4 notebook's cells through the pinned smoke runner TOOLS/nb_pilot_runner4.py (imported from its file;
   its own --stop-at-learn: learn() raises LearnBlocked; PPO/SAC train(), collect_rollouts(), the optimizers' step()
   and backward() raise and are counted). The only addition: the runner's install_learn_stops is wrapped so that each
   stopped learn() first records the model object it was called on (CAPTURE), then the runner's own stop raises.
   The manual cell (MANUAL_LOAD_MODE initialize_next_stage, the seed's own 8M pair) thus builds the model through
   the notebook's train_stage -> train_base.train -> _train_stage_body, which prepares it at the edge (the command
   columns zeroed, the command slice of obs_rms reseeded, ret_rms fresh) and attaches the task lineage.
2. Copies every file the stage folder holds at learn() (stage_config.json and the other records) to OUTDIR/stage/.
3. Saves the captured model with SB3's own CheckpointCallback (save_freq 1, name_prefix stage2, save_vecnormalize)
   at num_timesteps 500,000 / 1,000,000 / 1,500,000 / 2,000,000 (num_timesteps set before each save, as a periodic
   save would carry it; the weights are the edge's, identical in all four): OUTDIR/models/stage2_<steps>_steps.zip
   and stage2_vecnormalize_<steps>_steps.pkl. model.save() and VecNormalize.save() only; no update.
4. Checks: the training guards were never hit; the saved zip's JSON data member carries the lineage; the reloaded
   2,000,000 pair, rolled deterministically for one 1,000-step episode (reset 3042) in the tree's pilot-4 env, acts
   bit-identically to the original pilot-3 8M pair fed the same observation with obs[60:63] zeroed (the setup
   check's dA), and to the in-memory captured model.
Writes OUTDIR/edge_pairs.json.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[5] / ("pilot4/tools")
RUNS = Path(__file__).resolve().parents[5] / ("pilot3/runs/final")
STEPS = (500_000, 1_000_000, 1_500_000, 2_000_000)

SEED = int(sys.argv[1])
PARAMS = Path(sys.argv[2])
OUT = Path(sys.argv[3])
CLONE = Path(sys.argv[4]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
RESULT: dict = {"seed": SEED, "params": str(PARAMS), "clone": str(CLONE)}


def sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save() -> None:
    (OUT / "edge_pairs.json").write_text(json.dumps(RESULT, indent=1, default=str) + "\n")


spec = importlib.util.spec_from_file_location("nb_pilot_runner4", TOOLS / "nb_pilot_runner4.py")
runner = importlib.util.module_from_spec(spec)
sys.modules["nb_pilot_runner4"] = runner
spec.loader.exec_module(runner)
RESULT["runner_sha256"] = sha(TOOLS / "nb_pilot_runner4.py")
CAPTURE: dict = {}
_install = runner.install_learn_stops


def install_with_capture() -> dict:
    rec = _install()  # the runner's own stops and guards
    import stable_baselines3 as sb3
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    classes = [BaseAlgorithm, OnPolicyAlgorithm, OffPolicyAlgorithm, sb3.PPO, sb3.SAC, sb3.A2C]
    from environments.shared.mirror_kl_ppo import MirrorKLPPO

    classes.append(MirrorKLPPO)
    wrapped = []
    for cls in classes:
        if "learn" in cls.__dict__:
            stop = cls.__dict__["learn"]

            def learn(self, *args, _stop=stop, **kwargs):
                CAPTURE.setdefault("model", self)
                CAPTURE.setdefault("kwargs", kwargs)
                return _stop(self, *args, **kwargs)  # the runner's stop: raises LearnBlocked

            setattr(cls, "learn", learn)
            wrapped.append(f"{cls.__module__}.{cls.__qualname__}.learn")
    rec["e2e_capture_wraps"] = wrapped
    return rec


runner.install_learn_stops = install_with_capture
rc = runner.main(["--params", str(PARAMS), "--out", str(OUT / "nb"), "--clone", str(CLONE), "--stop-at-learn"])
sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
RESULT["runner_exit"] = rc
RESULT["runner_record"] = {k: runner.RECORD.get(k) for k in (
    "clone_head", "roles", "run_all", "manual_cell_stopped_at_learn", "learn_called", "guard_hits", "learn_stops",
    "environments_resolves_to", "environments_imported_from", "run_dir", "entry_lines", "parent_pair",
    "locomotion_curriculum_at_end", "notebook_sha256")}
RESULT["captured"] = "model" in CAPTURE
save()
if rc != 0 or "model" not in CAPTURE or runner.GUARD_HITS:
    print(f"edge_pairs: the set-up path did not end as expected (runner exit {rc}, captured {'model' in CAPTURE}, "
          f"guard hits {runner.GUARD_HITS}); see {OUT / 'edge_pairs.json'}", flush=True)
    sys.exit(1)

import environments  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.logger import Logger  # noqa: E402

from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.policy_loading import load_sb3_model  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.task_fingerprint import MODEL_TASK_LINEAGE_ATTRIBUTE  # noqa: E402

RESULT["environments_file"] = environments.__file__
model = CAPTURE["model"]
RESULT["model_class"] = f"{type(model).__module__}.{type(model).__qualname__}"
RESULT["lineage_in_memory"] = getattr(model, MODEL_TASK_LINEAGE_ATTRIBUTE, None)
RESULT["num_timesteps_at_learn"] = int(model.num_timesteps)
RESULT["learn_kwargs"] = {k: (repr(v) if k == "callback" else v) for k, v in CAPTURE["kwargs"].items()}

# 2. the stage folder's records at learn()
stage_dir = Path(runner.RECORD["run_dir"]) / "03_locomotion"
dst = OUT / "stage"
dst.mkdir()
records = {}
for p in sorted(stage_dir.rglob("*")):
    if p.is_file():
        rel = p.relative_to(stage_dir)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)
        records[str(rel)] = sha(p)
RESULT["stage_dir"] = str(stage_dir)
RESULT["stage_records"] = records
save()

# 3. the stand-in pairs, saved by SB3's own periodic-checkpoint code
models = OUT / "models"
models.mkdir()
model._logger = Logger(folder=None, output_formats=[])
cb = CheckpointCallback(save_freq=1, save_path=str(models), name_prefix="stage2", save_vecnormalize=True)
cb.init_callback(model)
saved = {}
for steps in STEPS:
    model.num_timesteps = steps
    cb.on_step()
    z, v = models / f"stage2_{steps}_steps.zip", models / f"stage2_vecnormalize_{steps}_steps.pkl"
    with zipfile.ZipFile(z) as zf:
        data = json.loads(zf.read("data"))
    saved[steps] = {"zip": sha(z), "vecnorm": sha(v), "data_num_timesteps": data.get("num_timesteps"),
                    "data_lineage": data.get("mesozoic_task_lineage")}
RESULT["saved"] = saved
RESULT["guard_hits_after_save"] = list(runner.GUARD_HITS)
save()

# 4. the reloaded 2M pair against the original 8M pair on command-zeroed observations, one deterministic episode
MID = ModelId("tyrannosaurus_rex", "anatomical")
kw = dict(load_stage_config(MID, 2)["env_kwargs"])
RESULT["env_kwargs_check"] = {k: kw.get(k) for k in ("command_mode", "prey_lateral_range", "walk_heading_price_weight",
                                                     "walk_heading_price_scale_rad")}
EnvClass = get_species_config(MID).env_class
orig_zip = RUNS / f"pilot3_walk_s{SEED}" / "03_locomotion" / "models" / "stage2_8000000_steps.zip"
orig_pkl = RUNS / f"pilot3_walk_s{SEED}" / "03_locomotion" / "models" / "stage2_vecnormalize_8000000_steps.pkl"
orig = load_sb3_model(str(orig_zip), algorithm=PPO, device="cpu")
with open(orig_pkl, "rb") as f:
    orig_vn = pickle.load(f)
new = load_sb3_model(str(models / "stage2_2000000_steps.zip"), algorithm=PPO, device="cpu")
with open(models / "stage2_vecnormalize_2000000_steps.pkl", "rb") as f:
    new_vn = pickle.load(f)
mem_vn = model.get_vec_normalize_env()


def act(m, x):
    return m.predict(np.asarray(x, dtype=np.float32)[None], deterministic=True)[0][0]


env = EnvClass(**kw)
obs, _ = env.reset(seed=3042)
d_orig = d_mem = 0.0
n = 0
te = tr = False
for _ in range(1000):
    o = np.asarray(obs, dtype=np.float32)
    z = o.copy()
    z[60:] = 0.0
    a_new = act(new, new_vn.normalize_obs(o))
    a_orig = act(orig, orig_vn.normalize_obs(z))
    a_mem = act(model, mem_vn.normalize_obs(o))
    d_orig = max(d_orig, float(np.max(np.abs(a_new - a_orig))))
    d_mem = max(d_mem, float(np.max(np.abs(a_new - a_mem))))
    obs, r, te, tr, info = env.step(a_new)
    n += 1
    if te or tr:
        break
env.close()
RESULT["reloaded_2M_vs_original_8M_on_zeroed_obs"] = {"reset_seed": 3042, "steps": n, "terminated": bool(te),
                                                       "truncated": bool(tr), "max_abs_action_delta": d_orig,
                                                       "max_abs_action_delta_vs_in_memory": d_mem}
RESULT["guard_hits_final"] = list(runner.GUARD_HITS)
try:
    model.get_env().close()
except Exception as exc:  # noqa: BLE001 - recorded
    RESULT["close_train_env"] = f"{type(exc).__name__}: {exc}"
for c in getattr(CAPTURE["kwargs"].get("callback"), "callbacks", []):
    ev = getattr(c, "eval_env", None)
    if ev is not None:
        try:
            ev.close()
        except Exception as exc:  # noqa: BLE001 - recorded
            RESULT["close_eval_env"] = f"{type(exc).__name__}: {exc}"
save()
print(json.dumps({k: RESULT.get(k) for k in ("seed", "runner_exit", "captured", "guard_hits_final",
                                            "reloaded_2M_vs_original_8M_on_zeroed_obs")}, default=str), flush=True)
sys.exit(0 if not runner.GUARD_HITS and d_orig == 0.0 and d_mem == 0.0 else 1)
