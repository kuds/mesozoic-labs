"""Run the notebook's own cells (head_tree copy of WT HEAD) locally, as a user would after editing cell 6.
usage: nb_run.py on|off [smoke]
  off: cells 3,4,6,8,9 with CRITIC_PHASE False; checks STAGE_CONFIGS == load_all_stages (the default path changes nothing)
  on : cells 3,4,6,8,9,11,13,15,16,18 with CRITIC_PHASE True (stops before cell 20, the chain loop's training);
       'smoke': then train_stage for 1 PPO update (harness shrinks n_steps; checkpoint cadence 128) and the RESUME cell
       (cell 18, RESUME_STAGE="stance") over the deleted final pair for 1 more update."""
import copy
import json
import os
import re
import sys
import traceback

TREE = "/home/user/work/pc_study/check_code/head_tree"
mode = sys.argv[1]
smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
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
for name, value in [("SPECIES", '"tyrannosaurus_rex"'), ("BEHAVIOR", '"stance"'), ("TRUNK_FROM", '""'),
                    ("USE_GOOGLE_DRIVE", "False"), ("AUTO_DISCONNECT", "False"), ("N_ENVS", "2"), ("SEED", "42"),
                    ("RUN_ID", f'"pccheck_{mode}"'), ("RUN_LABEL", f'"pc check {mode}"'),
                    ("CRITIC_PHASE", "True" if mode == "on" else "False")]:
    cell6 = set_param(cell6, name, value)


def run(i, src=None):
    print(f"\n######## CELL {i} ########", flush=True)
    exec(compile(src if src is not None else cells[i], f"<cell {i}>", "exec"), g)


order = [3, 4, 6, 8, 9] + ([11, 13, 15, 16, 18] if mode == "on" else [])
for i in order:
    run(i, cell6 if i == 6 else None)

from environments.shared.config import load_all_stages  # noqa: E402

fresh = load_all_stages(g["MODEL_ID"])
ref = g["CHAIN"][0].reference
if mode == "off":
    print("CHECK default path: STAGE_CONFIGS == load_all_stages(MODEL_ID):", g["STAGE_CONFIGS"] == fresh)
else:
    exp = copy.deepcopy(fresh)
    for node in g["CHAIN"]:
        exp[node.reference].setdefault("ppo_kwargs", {})["critic_phase"] = True
    print("CHECK switch path: STAGE_CONFIGS == TOMLs + ppo_kwargs.critic_phase=True on the chain only:", g["STAGE_CONFIGS"] == exp)
print("CHAIN", [n.id for n in g["CHAIN"]], "RUN_DIR", g.get("RUN_DIR"), "TRUNK_DIR", g.get("TRUNK_DIR"))

if mode == "on" and smoke:
    from pathlib import Path

    cfg = g["STAGE_CONFIGS"][ref]
    # harness only: tiny rollout and a checkpoint cadence the smoke reaches; budget 256 = 2 updates of 2 x 64
    cfg["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=2)
    cfg["curriculum_kwargs"].update(checkpoint_freq=128, timesteps=256)
    print("\n######## SESSION 1: train_stage(stance, 128) ########", flush=True)
    out = g["train_stage"](stage=ref, timesteps=128, run_dir=g["RUN_DIR"], task_load_mode="resume_same_stage",
                           label=g["RUN_LABEL"] or None, evaluate=False)
    model = out[0]
    print("session1 classes:", type(model).__name__, type(model.policy).__name__, type(model.rollout_buffer).__name__,
          type(model.get_vec_normalize_env()).__name__, "num_timesteps", model.num_timesteps)
    stage_dir = Path(out[3])
    models = sorted(p.name for p in (stage_dir / "models").iterdir())
    print("session1 models:", models)
    sc = json.load(open(stage_dir / "stage_config.json"))
    print("session1 stage_config critic_phase:", sc["hyperparameters"].get("critic_phase"),
          "hyperparameters_sha256:", sc["run"]["hyperparameters_sha256"])
    # simulate a runtime reclaim before the final save: remove the final pair (my folder only)
    for suffix in (".zip", "_vecnorm.pkl"):
        p = stage_dir / "models" / f"stage1_final{suffix}"
        if p.exists():
            p.unlink()
    print("\n######## SESSION 2: RESUME cell with RESUME_STAGE = 'stance' ########", flush=True)
    run(18, set_param(cells[18], "RESUME_STAGE", '"stance"'))
    m2 = g["model_res"]
    print("resume classes:", type(m2).__name__, type(m2.policy).__name__, type(m2.rollout_buffer).__name__,
          type(m2.get_vec_normalize_env()).__name__, "num_timesteps", m2.num_timesteps,
          "buffer width", m2.rollout_buffer.observations.shape)
    import zipfile
    data = json.loads(zipfile.ZipFile(stage_dir / "models" / "stage1_final.zip").read("data"))
    print("final zip mesozoic_critic_phase:", data.get("mesozoic_critic_phase"))
    print("final zip observation_space shape:", data["observation_space"].get("_shape"))
    import pickle
    vn = pickle.load(open(stage_dir / "models" / "stage1_final_vecnorm.pkl", "rb"))
    print("final sidecar:", type(vn).__module__ + "." + type(vn).__name__, vn.obs_rms.mean.shape)
    print("models after resume:", sorted(p.name for p in (stage_dir / "models").iterdir()))
