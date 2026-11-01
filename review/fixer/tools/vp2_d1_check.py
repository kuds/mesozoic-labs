"""P1/VP2 (widened-pair clause) and D1 (loader alone), on a git archive of HEAD, read only on STANCE6 s80's F2 final
pair (sha256 checked against the F2 fetch manifest).  No training; writes only into OUT.
 (a) K3 panel: environments.shared.k3_panel.run_k3_panel (own and joint tests, K3_PANEL_SEEDS, horizon = the stage's
     max_episode_steps, joint sigma control_sigma.80 from HEAD's sigma_s70_s80.json) on the stock policy and on its
     zero-phase-column widening (manip_check.widen_stock_policy): rows compared bitwise.
 (b) D1: the widened model saved as a zip by HEAD's own save path (BaseAlgorithm.save with the phase policy class and
     the mesozoic_critic_phase record, as train_base sets it); M3's loader alone (stance_report._load_policy) on that
     zip and on the original zip, one predict each on the same 1000-step deterministic bank of the stock actor: equal.
usage: PY -I -B vp2_d1_check.py COPY OUT"""
import csv, hashlib, importlib.util, json, os, sys
from pathlib import Path
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
sys.dont_write_bytecode = True
COPY, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
S1 = Path("/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad")
FETCH = S1 / "foot/stance6/fetch"; RUN = "stance6_r10LTSB_s80"
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows = [r for r in csv.DictReader(open(FETCH / "manifest.csv", newline="")) if r["fetch"] == "F2" and r["run_id"] == RUN and r["status"] == "COPY"]
paths = []
for rel in ("01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"):
    p = FETCH / "F2" / RUN / "copies" / f"{rel}.copy1"; h = sha(p)
    assert any(r["path"] == rel and r["sha256"] == h for r in rows), rel
    print(f"{rel}.copy1 {h} in the F2 manifest: True"); paths.append(p)
spec = importlib.util.spec_from_file_location("manip_check", "/home/user/work/pc_study/tools/pc/manip_check.py")
MC = importlib.util.module_from_spec(spec); sys.modules["manip_check"] = MC; spec.loader.exec_module(MC)
sys.path.insert(0, str(COPY))
import numpy as np
import environments
assert Path(environments.__file__).resolve().is_relative_to(COPY)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environments.shared import k3_panel as K
from environments.shared.config import load_all_stages
from environments.shared.model_id import ModelId
from environments.shared.plant_contract import current_plant_identity
from environments.shared.reporting.episode_trace import load_trace_policy
from environments.shared.reporting import stance_report as SR
from environments.shared.species_registry import env_class as model_env_class
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.critic_phase import CHECKPOINT_ATTRIBUTE
from environments.shared.critic_phase_ppo import CriticPhaseActorCriticPolicy
mid = ModelId("tyrannosaurus_rex", "anatomical")
stage = load_stage_manifest(mid).resolve_behavior("stance").reference
env_kwargs = dict(load_all_stages(mid)[stage]["env_kwargs"]); env_class = model_env_class(mid)
factory = lambda: env_class(**env_kwargs)
ident = current_plant_identity(mid)
tp = load_trace_policy(str(paths[0]), str(paths[1]), factory, plant_identity=ident)
stock = tp.model.policy
wide = MC.widen_stock_policy(stock, 200, 0.98)
vn = VecNormalize.load(str(paths[1]), DummyVecEnv([factory])); vn.training = False
norm = K.copy_normalization(vn)
sig = K.read_joint_sigma(COPY / "docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json", "control_sigma.80")
horizon = int(env_kwargs.get("max_episode_steps", 1000))
res = {"horizon": horizon, "seeds": list(K.K3_PANEL_SEEDS), "joint_sigma_sha256": sig.sha256, "sigma_problem": sig.problem}
r_stock = K.run_k3_panel(stock, norm, factory, joint_sigma=sig.vector, horizon=horizon)
r_wide = K.run_k3_panel(wide, norm, factory, joint_sigma=sig.vector, horizon=horizon)
res["k3_rows"] = len(r_stock); res["k3_rows_equal_bitwise"] = r_stock == r_wide
res["k3_falls"] = {t: sum(r["fall"] for r in r_stock if r["test"] == t) for t in ("own", "joint")}
(OUT / "k3_rows_stock.json").write_text(json.dumps(r_stock, default=str))
(OUT / "k3_rows_wide.json").write_text(json.dumps(r_wide, default=str))
print("(a) K3 panel:", json.dumps(res))
# (b) D1: save the widened model by HEAD's own save path, then M3's loader alone
model = tp.model
model.policy = wide; model.policy_class = CriticPhaseActorCriticPolicy
setattr(model, CHECKPOINT_ATTRIBUTE, wide.critic_phase.record())
zp = OUT / "widened_s80_final.zip"; model.save(str(zp))
pred_w, desc_w = SR._load_policy(str(zp), str(paths[1]), factory, plant_identity=ident)
pred_s, desc_s = SR._load_policy(str(paths[0]), str(paths[1]), factory, plant_identity=ident)
env = factory(); obs, _ = env.reset(seed=3042); n_eq = n = 0
for _ in range(1000):
    a_s, a_w = np.asarray(pred_s(obs)), np.asarray(pred_w(obs))
    n += 1; n_eq += bool(np.array_equal(a_s, a_w))
    obs, _r, term, trunc, _i = env.step(a_s)
    if term or trunc:
        break
env.close()
print(f"(b) D1: widened zip {zp.name} sha256 {sha(zp)}; loader describe {desc_w!r}; stock describe {desc_s!r}; "
      f"predict equal on {n_eq} of {n} steps of a deterministic episode (seed 3042)")
ok = res["k3_rows_equal_bitwise"] and n_eq == n and n > 0
print("RESULT", "VP2 widened-pair panel equal; D1 loader+predict equal" if ok else "NOT EQUAL")
