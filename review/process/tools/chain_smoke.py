"""Process review: run the notebook's own cells on a git-archive copy of WT HEAD (da7e0b92) with CRITIC_PHASE True,
INCLUDING cell 20 (the behavior chain loop: train_stage -> generate_stage_artifacts -> write_training_summary ->
save_run_bundle) and cell 30 (verify the result bundle) -- the Colab "Run all" path neither the builder's nor the
checker's smoke reached.  Harness-only shrink (the point is crash detection with a phase checkpoint, not numbers):
n_steps 64 x 2 envs, budget 256 (2 PPO updates), eval every 128 steps of 1 episode, the stance report / certificate /
probes at 2 episodes, one filter and one impulse speed, no videos (no GL here; Colab renders with EGL).

usage: chain_smoke.py on|off
"""

import functools
import json
import os
import re
import sys
import time
import traceback

TREE = "/home/user/work/pc_study/review/process/tree"
mode = sys.argv[1]
os.chdir(f"{TREE}/notebooks")
nb = json.load(open(f"{TREE}/notebooks/sb3_training.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"]]


class _IP:
    def system(self, cmd):
        raise RuntimeError("shell escape in a local run: " + cmd)


g = {"__name__": "__main__", "get_ipython": lambda: _IP()}


def set_param(src, name, value):
    pat = re.compile(rf"^{name}\s*=.*$", re.M)
    assert len(pat.findall(src)) == 1, name
    return pat.sub(f"{name} = {value}", src)


cell6 = cells[6]
for name, value in [
    ("SPECIES", '"tyrannosaurus_rex"'),
    ("BEHAVIOR", '"stance"'),
    ("TRUNK_FROM", '""'),
    ("USE_GOOGLE_DRIVE", "False"),
    ("AUTO_DISCONNECT", "False"),
    ("N_ENVS", "2"),
    ("SEED", "42"),
    ("RUN_ID", f'"pcr_chain_{mode}"'),
    ("RUN_LABEL", f'"pcr chain {mode}"'),
    ("CRITIC_PHASE", "True" if mode == "on" else "False"),
]:
    cell6 = set_param(cell6, name, value)


def run(i, src=None):
    t0 = time.time()
    print(f"\n######## CELL {i} ########", flush=True)
    exec(compile(src if src is not None else cells[i], f"<cell {i}>", "exec"), g)
    print(f"######## CELL {i} done in {time.time() - t0:.1f} s", flush=True)


for i in [3, 4, 6, 8, 9, 11, 15, 16]:
    run(i, cell6 if i == 6 else None)

ref = g["CHAIN"][0].reference
cfg = g["STAGE_CONFIGS"][ref]
print("stage config ppo critic_phase:", cfg["ppo_kwargs"].get("critic_phase"), flush=True)
cfg["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=2)
cfg["curriculum_kwargs"].update(
    timesteps=256,
    checkpoint_freq=128,
    eval_freq=128,
    eval_episodes=1,
    stance_report_episodes=2,
    stance_probe_filter_hz=[1.0],
    stance_probe_impulse_speeds=[0.5],
)
_orig = g["generate_stage_artifacts"]
g["generate_stage_artifacts"] = functools.partial(_orig, record_videos=False)

status = {}
for i in (20, 24, 30):
    try:
        run(i)
        status[i] = "ok"
    except BaseException as exc:  # noqa: BLE001 - record and continue; a gate FAIL halts by raising
        status[i] = f"{type(exc).__name__}: {exc}"[:400]
        traceback.print_exc()
print("CELL STATUS", json.dumps(status), flush=True)

from pathlib import Path  # noqa: E402

run_dir = Path(g["RUN_DIR"])
print("RUN_DIR", run_dir)
for p in sorted(run_dir.rglob("*")):
    if p.is_file():
        print("FILE", p.relative_to(run_dir), p.stat().st_size)
man = run_dir / "artifact_manifest.json"
print("artifact_manifest.json present:", man.exists())
stage_dir = run_dir / "01_stance"
if (stage_dir / "stage_config.json").exists():
    sc = json.load(open(stage_dir / "stage_config.json"))
    print("stage_config critic_phase:", sc.get("hyperparameters", {}).get("critic_phase"))
for name in ("gate_verdict.json",):
    f = stage_dir / name
    if f.exists():
        v = json.load(open(f))
        print(name, {k: v.get(k) for k in list(v)[:8]})
