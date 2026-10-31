"""PS build checker: an independent check of the critic_phase arm (own comparison code; the hook under test is
PS/tools/bs_hooks.install_arm).  No file is written except the JSON on stdout (the caller captures it under
PS/check_build/logs/) and scratch under TMPDIR (PS/check_build/tmp/c_arm, deleted by the caller).  The ONE optimizer
pass is a throwaway: one full MirrorKLPPO.train() (the run's n_epochs and batch size) on an in-memory copy; nothing is
saved.

    PY -I -B c_arm.py SEED
Parts:
  A  load fidelity: the loaded policy / optimizer = the start zip's tensors (torch.load weights_only, own reading).
  B  after install_arm: every actor tensor bitwise the zip's; value_net.0 bitwise but cols 60-62 (now 0); optimizer
     state bitwise; statistics: 60 dims + count bitwise, slots (0, 1).
  C  fresh states (4,096 random raw observations, random k in 0..1000, and random garbage in the slots): the hooked
     actor (mode, log_std, log_prob, entropy, deterministic forward) and critic against an UNHOOKED deep copy of the
     loaded policy on the same states with zero command slots under the START statistics: bitwise.
  D  the mirror path: MirrorKLPPO's own tensors (built by _mirror_begin_update) and the actor on mirrored states.
  E  the evaluation path: an eval VecNormalize synced the way EvalCallback syncs (sync_envs_normalization), its
     statistics' class and slots, and model.predict on 60 eval steps against the unhooked copy.
  F  one rollout (2,048 x 4) with the hooked training env: every buffer row's slots against the phase of the env's own
     counter reconstructed from infos["step"] (not from episode_starts); the gated reward terms against the
     observation's settled flag (off-by-one); the time-limit terminal observation; VecNormalize's 60 other dims and
     count against a control twin fed the same raw batches with the slots zero; actor / critic on the recorded buffer.
  G  the probe: a PolicyProbeCallback (the code's class, swapped by H.phase_probe_class) row on the parent's bank.
  H  one throwaway train(): actor phase columns and their Adam moments, critic phase columns, invariance after.
"""
import copy
import importlib.util
import io
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path

PS = Path("/home/user/work/phase_screen")
TOOLS = PS / "tools"
CB = PS / "check_build"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("bs_common")
H = _load("bs_hooks")
SEED = int(sys.argv[1])
assert SEED in C.VALIDATION_SEEDS
tmp = Path(os.environ["TMPDIR"])
assert str(tmp).startswith(str(CB) + "/")
code = os.path.realpath(C.CODE)
sys.path.insert(0, code)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.logger import Logger  # noqa: E402
from stable_baselines3.common.running_mean_std import RunningMeanStd  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization  # noqa: E402

import environments  # noqa: E402
from environments.shared.config import load_stage_config  # noqa: E402
from environments.shared.mirror_kl_ppo import MirrorKLPPO, SymmetricRunningMeanStd, enable_mirror_kl  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.policy_loading import load_sb3_model  # noqa: E402
from environments.shared.policy_probe import PolicyProbeCallback, StateBank  # noqa: E402
from environments.shared.species_registry import env_class  # noqa: E402
from environments.shared.symmetry import MirrorMap, pop_symmetry_config  # noqa: E402
from environments.shared.train_base import _prepare_alg_kwargs, create_vec_env  # noqa: E402

assert os.path.realpath(environments.__file__).startswith(code + os.sep)
LO, HI = 60, 63
GAMMA, SETTLE = 0.98, 200
out = {"seed": SEED}
fails = []


def check(name, ok, detail=None):
    out.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)


def my_phase(k):
    """My own reading of the design: ramp, settled, discount of an observation made at counter k."""
    k = int(k)
    return (min(k, SETTLE) / SETTLE, 1.0 if k >= SETTLE else 0.0, GAMMA ** max(0, SETTLE - k))


def mx(a, b):
    a, b = torch.as_tensor(a).double().flatten(), torch.as_tensor(b).double().flatten()
    return float((a - b).abs().max())


def beq(a, b):
    return bool(torch.equal(torch.as_tensor(a), torch.as_tensor(b)))


# ------------------------------------------------------------------------------------------------ A: load
zip_path, pkl_path = C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"])
assert C.sha256_file(zip_path) == C.START["zip_sha256"] and C.sha256_file(pkl_path) == C.START["pkl_sha256"]
with zipfile.ZipFile(zip_path) as z:
    P0 = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    O0 = torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
with open(pkl_path, "rb") as fh:
    VN0 = pickle.load(fh)
mean0 = np.array(VN0.obs_rms.mean, dtype=np.float64)
var0 = np.array(VN0.obs_rms.var, dtype=np.float64)
eps0, clip0 = float(VN0.epsilon), float(VN0.clip_obs)
out["start_slots"] = {"mean": mean0[LO:HI].tolist(), "var": var0[LO:HI].tolist(), "eps": eps0, "clip": clip0}

mid = ModelId("tyrannosaurus_rex", "anatomical")
cfg = copy.deepcopy(load_stage_config(mid, "stance"))


class _Species:
    model_id = mid
    env_class = None


species = _Species()
species.env_class = env_class(mid)
ppo = copy.deepcopy(cfg["ppo_kwargs"])
sym = pop_symmetry_config(ppo)
venv = create_vec_env(species, {1: {"env_kwargs": dict(cfg["env_kwargs"])}}, 1, C.N_ENVS, seed=SEED,
                      use_subproc=False, algorithm="ppo", gamma=float(ppo["gamma"]))
vn = VecNormalize.load(str(pkl_path), venv.venv)
vn.training = True
vn.norm_reward = True
alg_kwargs, _tb, _g = _prepare_alg_kwargs({"ppo_kwargs": copy.deepcopy(cfg["ppo_kwargs"])}, "ppo", 0,
                                          tmp / "_tb_unused", use_tensorboard=False)
pop_symmetry_config(alg_kwargs)
alg_kwargs.pop("policy_kwargs", None)
model = load_sb3_model(str(zip_path), algorithm="ppo", env=vn, device="cpu", **alg_kwargs)
mirror_map = MirrorMap.load(mid, current_plant_identity(mid))
model = enable_mirror_kl(model, sym, mirror_map)
assert isinstance(model, MirrorKLPPO)
out["rms_class_before_hook"] = type(vn.obs_rms).__name__
sd = model.policy.state_dict()
check("A_loaded_policy_equals_zip", list(sd) == list(P0) and all(beq(sd[k], P0[k]) for k in P0))
osd = model.policy.optimizer.state_dict()


def opt_equal(a, b):
    sa, sb = a["state"], b["state"]
    if sorted(sa) != sorted(sb):
        return False
    for i in sa:
        if sorted(sa[i]) != sorted(sb[i]):
            return False
        for n in sa[i]:
            if not beq(sa[i][n], sb[i][n]):
                return False
    return a["param_groups"] == b["param_groups"]


check("A_loaded_optimizer_equals_zip", opt_equal(osd, O0))
out["optimizer"] = {"class": type(model.policy.optimizer).__name__,
                    "param_groups": [{k: v for k, v in g.items() if k != "params"}
                                     for g in model.policy.optimizer.param_groups]}
ref = copy.deepcopy(model.policy)            # the UNHOOKED loaded policy (reference)
ref.eval()
twin_rms = SymmetricRunningMeanStd.from_stock(copy.deepcopy(vn.obs_rms), mirror_map.obs)   # control's statistics
check("A_twin_is_symmetric", isinstance(twin_rms, SymmetricRunningMeanStd), type(twin_rms).__name__)

# ------------------------------------------------------------------------------------------------ B: hook
rec = H.install_arm("critic_phase", model, vn)
out["arm_applied"] = {k: rec.get(k) for k in ("transform_verified", "mlp_extractor_class", "obs_rms_class",
                                              "train_env_class", "value_phase_cols_max_abs_before",
                                              "adam_phase_moments_max_abs_before", "mirror_obs_sign_slots_before")}
policy = model.policy
sd1 = policy.state_dict()
VN0W = "mlp_extractor.value_net.0.weight"
actor_keys = [k for k in P0 if not k.startswith("mlp_extractor.value_net") and not k.startswith("value_net")]
check("B_actor_tensors_bitwise_zip", all(beq(sd1[k], P0[k]) for k in actor_keys), actor_keys)
other = [k for k in P0 if k not in actor_keys and k != VN0W]
check("B_critic_other_tensors_bitwise_zip", all(beq(sd1[k], P0[k]) for k in other), other)
keep = torch.ones(63, dtype=torch.bool)
keep[LO:HI] = False
check("B_value_net0_other_cols_bitwise", beq(sd1[VN0W][:, keep], P0[VN0W][:, keep]))
check("B_value_net0_phase_cols_zero", bool(torch.all(sd1[VN0W][:, LO:HI] == 0.0)),
      float(P0[VN0W][:, LO:HI].abs().max()))
check("B_optimizer_bitwise_zip", opt_equal(policy.optimizer.state_dict(), O0))
rms = vn.obs_rms
check("B_rms_other_dims_bitwise", np.array_equal(np.asarray(rms.mean)[:LO], mean0[:LO])
      and np.array_equal(np.asarray(rms.var)[:LO], var0[:LO]) and float(rms.count) == float(VN0.obs_rms.count))
check("B_rms_slots_0_1", np.array_equal(np.asarray(rms.mean)[LO:HI], [0.0] * 3)
      and np.array_equal(np.asarray(rms.var)[LO:HI], [1.0] * 3), type(rms).__name__)
check("B_rms_is_symmetric_subclass", isinstance(rms, SymmetricRunningMeanStd))
# the actor's Adam moments on its slot columns (a constant input) and the critic's: 0 at the load
st_pi = policy.optimizer.state[policy.mlp_extractor.policy_net[0].weight]
st_vf = policy.optimizer.state[policy.mlp_extractor.value_net[0].weight]
check("B_adam_slot_moments_zero", all(float(st[n][:, LO:HI].abs().max()) == 0.0 for st in (st_pi, st_vf)
                                      for n in ("exp_avg", "exp_avg_sq")))


def norm_np(raw, mean, var):
    return np.clip((raw - mean) / np.sqrt(var + eps0), -clip0, clip0)


# ------------------------------------------------------------------------------------------------ C: fresh states
rng = np.random.default_rng(SEED)
N = 4096
raw = mean0 + rng.standard_normal((N, 63)) * np.sqrt(var0 + eps0)
raw = raw.astype(np.float32).astype(np.float64)        # the env's observation is float32 (l1027)
raw[:, LO:HI] = 0.0
ks = rng.integers(0, 1001, size=N)
ks[:8] = [0, 1, 199, 200, 201, 999, 1000, 100]
raw_phase = raw.copy()
raw_phase[:, LO:HI] = np.array([my_phase(k) for k in ks])
raw_phase = raw_phase.astype(np.float32).astype(np.float64)    # the env's observation is float32 (l1027)
# the float64 feeding path (the probe's and phase_bank_check's: phase in a float64 raw array, normalised, then
# float32) against the env's path (float32 raw first), for every k 0..1000
kk = np.arange(0, 1001)
f64 = np.array([my_phase(k) for k in kk])
s_eps = np.sqrt(1.0 + eps0)
env_path = (f64.astype(np.float32).astype(np.float64) / s_eps).astype(np.float32)
f64_path = (f64 / s_eps).astype(np.float32)
check('C_env_path_slot_equals_float32_phase_all_k', np.array_equal(env_path, f64.astype(np.float32)))
out['C_float64_path_ulp_diffs'] = {'n_values_differing': int((f64_path != f64.astype(np.float32)).sum()),
                                   'ks': sorted(set(kk[np.any(f64_path != f64.astype(np.float32), axis=1)].tolist()))[:20],
                                   'settled_k_exact': bool(np.array_equal(f64_path[200:], f64.astype(np.float32)[200:])),
                                   'max_rel': float(np.max(np.abs(f64_path.astype(np.float64) - f64.astype(np.float32).astype(np.float64)) / np.maximum(f64, 1e-30)))}
z0 = torch.as_tensor(norm_np(raw, mean0, var0), dtype=torch.float32)              # start stats, zero command
z1 = torch.as_tensor(norm_np(raw_phase, np.asarray(rms.mean), np.asarray(rms.var)), dtype=torch.float32)
check("C_z1_slots_equal_phase_float32", beq(z1[:, LO:HI], torch.as_tensor(raw_phase[:, LO:HI], dtype=torch.float32)))
check("C_z0_slots_zero", bool(torch.all(z0[:, LO:HI] == 0.0)))
check("C_other_dims_identical", beq(z0[:, :LO], z1[:, :LO]))
acts = torch.as_tensor(rng.uniform(-1, 1, size=(N, policy.action_space.shape[0])), dtype=torch.float32)


def actor_outputs(pol, x):
    with torch.no_grad():
        d = pol.get_distribution(x)
        mode = d.mode().clone()
        a, v, lp = pol.forward(x, deterministic=True)
        ve, lpe, ent = pol.evaluate_actions(x, acts[: len(x)])
        return {"mode": mode, "fwd_actions": a, "fwd_logp": lp, "eval_logp": lpe, "entropy": ent,
                "log_std": pol.log_std.detach().clone()}, {"fwd_value": v, "eval_value": ve,
                                                           "predict_values": pol.predict_values(x)}


a_ref, v_ref = actor_outputs(ref, z0)
a_hk, v_hk = actor_outputs(policy, z1)
check("C_actor_bitwise_vs_unhooked_start", all(beq(a_ref[k], a_hk[k]) for k in a_ref),
      {k: mx(a_ref[k], a_hk[k]) for k in a_ref})
check("C_critic_bitwise_vs_unhooked_start", all(beq(v_ref[k], v_hk[k]) for k in v_ref),
      {k: mx(v_ref[k], v_hk[k]) for k in v_ref})
garbage = z1.clone()
garbage[:, LO:HI] = torch.as_tensor(rng.uniform(-10, 10, size=(N, 3)), dtype=torch.float32)
a_g, v_g = actor_outputs(policy, garbage)
check("C_actor_invariant_to_garbage_slots", all(beq(a_ref[k], a_g[k]) for k in a_ref))
check("C_critic_invariant_at_zero_init", all(beq(v_ref[k], v_g[k]) for k in v_ref))
check("C_input_tensor_not_mutated", beq(z1[:, LO:HI], torch.as_tensor(raw_phase[:, LO:HI], dtype=torch.float32)))
# the unhooked policy DOES respond to the slots (the mask is what makes the hooked one invariant)
a_ru, _ = actor_outputs(ref, garbage)
out["C_unhooked_actor_response_to_slots"] = mx(a_ref["mode"], a_ru["mode"])

# ------------------------------------------------------------------------------------------------ D: mirror path
model.set_logger(Logger(folder=None, output_formats=[]))
model._mirror_begin_update()
T = model._mirror.tensors
check("D_mirror_tensors_slots_self_plus1", beq(T["obs_perm"][LO:HI], torch.arange(LO, HI))
      and bool(torch.all(T["obs_sign"][LO:HI] == 1.0)))
old = mirror_map.obs
check("D_mirror_other_dims_unchanged", np.array_equal(np.asarray(T["obs_perm"]), np.asarray(old.perm))
      and np.array_equal(np.asarray(T["obs_sign"])[:LO], np.asarray(old.sign, dtype=np.float32)[:LO]),
      {"old_slot_sign": np.asarray(old.sign)[LO:HI].tolist()})
om1 = z1[:, T["obs_perm"]] * T["obs_sign"]
om0 = z0[:, torch.as_tensor(old.perm)] * torch.as_tensor(old.sign, dtype=torch.float32)
check("D_mirrored_slots_keep_phase", beq(om1[:, LO:HI], z1[:, LO:HI]))
with torch.no_grad():
    mu_m1 = policy.get_distribution(om1).mode()
    mu_m0 = ref.get_distribution(om0).mode()
    v_m1, v_m0 = policy.predict_values(om1), ref.predict_values(om0)
check("D_mirrored_actor_bitwise_vs_unhooked", beq(mu_m1, mu_m0), mx(mu_m1, mu_m0))
check("D_mirrored_critic_bitwise_vs_unhooked", beq(v_m1, v_m0), mx(v_m1, v_m0))
model._mirror.tensors = {}                   # as found (built again by train())

# ------------------------------------------------------------------------------------------------ E: eval path
eval_venv = create_vec_env(species, {1: {"env_kwargs": dict(cfg["env_kwargs"])}}, 1, 1, seed=SEED + 7,
                           use_subproc=False, algorithm="ppo", gamma=float(ppo["gamma"]))
eval_vn = VecNormalize(eval_venv.venv if isinstance(eval_venv, VecNormalize) else eval_venv, training=False,
                       norm_obs=True, norm_reward=False)
sync_envs_normalization(vn, eval_vn)
check("E_eval_rms_stock_class", type(eval_vn.obs_rms) is RunningMeanStd, type(eval_vn.obs_rms).__name__)
check("E_eval_rms_slots_0_1", np.array_equal(eval_vn.obs_rms.mean[LO:HI], [0.0] * 3)
      and np.array_equal(eval_vn.obs_rms.var[LO:HI], [1.0] * 3))
check("E_eval_env_not_swapped", not any(getattr(type(getattr(e, "unwrapped", e)), "ps_phase_env", False)
                                        for e in eval_vn.venv.envs))
eobs = eval_vn.reset()
worst = 0.0
slots_seen = set()
for _ in range(60):
    raw_e = eval_vn.get_original_obs()
    slots_seen.update(np.asarray(raw_e)[:, LO:HI].reshape(-1).tolist())
    a_h, _ = model.predict(eobs, deterministic=True)
    z_ref = torch.as_tensor(norm_np(np.asarray(raw_e, dtype=np.float64), mean0, var0), dtype=torch.float32)
    with torch.no_grad():
        a_r = ref._predict(z_ref, deterministic=True).numpy()
    a_r = np.clip(a_r, policy.action_space.low, policy.action_space.high)
    worst = max(worst, float(np.max(np.abs(a_h - a_r))))
    eobs, _r, _d, _i = eval_vn.step(a_h)
check("E_eval_actions_bitwise_vs_unhooked_start", worst == 0.0, {"max_abs": worst, "raw_slots_seen": sorted(slots_seen)})
eval_vn.close()

# ------------------------------------------------------------------------------------------------ G: the probe
with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as zz:
    bank = StateBank(labels=tuple(str(x) for x in zz["bank_labels"]), seeds=zz["bank_seeds"],
                     steps=np.asarray(zz["bank_steps"], dtype=np.int64), ankle_offset=zz["bank_ankle_offset"],
                     raw_obs=np.asarray(zz["bank_raw_obs"], dtype=np.float64),
                     com_fore_aft_m=zz["bank_com_fore_aft_m"], ankle_actuators=zz["ankle_actuators"],
                     ankle_signs=zz["ankle_signs"], horizon=int(zz["bank_horizon"]), build_seconds=0.0)
    row_s = int(np.flatnonzero(np.asarray(zz["update_timesteps"]) == C.S)[0])
    parent_v = np.asarray(zz["bank_value_now"][row_s], dtype=np.float64)
    parent_a = np.asarray(zz["bank_action_now"][row_s], dtype=np.float64)
out["G_bank_steps"] = bank.steps.tolist()


def probe_row():
    p = PolicyProbeCallback.__new__(PolicyProbeCallback)
    BaseCallback.__init__(p, 0)
    p.init_callback(model)
    p.bank, p._warned, p.num_timesteps = bank, set(), int(model.num_timesteps)
    stt = PolicyProbeCallback._obs_stats(vn)
    p._frozen = (np.array(stt[0]), np.array(stt[1]), int(model.num_timesteps))
    rec_p = H.probe_phase("critic_phase", [p])
    row = p._row()
    return p, rec_p, row


def bank_fed(pol):
    fed = bank.raw_obs.copy()
    fed[:, LO:HI] = np.array([my_phase(int(s) + 1) for s in bank.steps])
    x = torch.as_tensor(norm_np(fed, np.asarray(vn.obs_rms.mean), np.asarray(vn.obs_rms.var)), dtype=torch.float32)
    with torch.no_grad():
        return pol.predict_values(x).flatten().numpy(), pol.get_distribution(x).mode().numpy()


p, rec_p, row = probe_row()
check("G_probe_k_is_kept_plus_1", rec_p["k"] == (bank.steps + 1).tolist() and all(k >= 200 for k in rec_p["k"]),
      rec_p["k"])
check("G_probe_bank_restored", p.bank is bank)
fv, fa = bank_fed(policy)
check("G_probe_value_now_equals_phase_fed_critic", np.array_equal(np.asarray(row["bank_value_now"]),
                                                                    fv.astype(np.float32)),
      mx(row["bank_value_now"], fv))
check("G_probe_value_frozen_equals_phase_fed_critic", np.array_equal(np.asarray(row["bank_value_frozen"]),
                                                                       fv.astype(np.float32)))
z_bank0 = torch.as_tensor(norm_np(bank.raw_obs, mean0, var0), dtype=torch.float32)
with torch.no_grad():
    rv0, ra0 = ref.predict_values(z_bank0).flatten().numpy(), ref.get_distribution(z_bank0).mode().numpy()
check("G_zero_init_probe_bitwise_vs_unhooked_start_bank", np.array_equal(np.asarray(row["bank_value_now"]), rv0.astype(np.float32)) and np.array_equal(np.asarray(row["bank_action_now"]), ra0.astype(np.float32)))
out["G_zero_init_probe_vs_unhooked_start_bank"] = {"value_max_abs": mx(row["bank_value_now"], rv0),
                                                   "action_max_abs": mx(row["bank_action_now"], ra0),
                                                   "value_vs_parent_row_S": mx(rv0, parent_v),
                                                   "action_vs_parent_row_S": mx(ra0, parent_a)}
# ------------------------------------------------------------------------------------------------ F: rollout
vn.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
model._last_obs = None
GATED = ("reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band")


class Rec(BaseCallback):
    def __init__(self):
        super().__init__(0)
        self.k_rows, self.trans, self.terminal, self.batches = [], [], [], []

    def _on_rollout_start(self):
        self.k_now = [int(e.unwrapped._step_count) for e in vn.venv.envs]

    def _on_step(self):
        infos, dones = self.locals["infos"], self.locals["dones"]
        self.k_rows.append(list(self.k_now))
        nxt = []
        for i, info in enumerate(infos):
            step = int(info["step"])                  # the env's own counter after this step (base_env l1365)
            self.trans.append((self.k_now[i], step, *[float(info.get(g, 0.0)) for g in GATED],
                               *[g in info for g in GATED]))
            if dones[i]:
                term = info.get("terminal_observation")
                self.terminal.append((step, bool(info.get("TimeLimit.truncated", False)),
                                      None if term is None else np.asarray(term, dtype=np.float32)[LO:HI].tolist()))
                nxt.append(0)
            else:
                nxt.append(step)
        self.k_now = nxt
        self.batches.append(np.array(vn.get_original_obs(), copy=True))          # float32, as VecNormalize updates
        return True


cb_rec = Rec()
_t, cb = model._setup_learn(model.n_steps * model.n_envs, cb_rec, False, "ps_check", False)
reset_raw = np.array(vn.get_original_obs(), copy=True)
out["rms_count_after_reset"] = float(vn.obs_rms.count)
assert model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps)
buf = model.rollout_buffer
bobs = np.asarray(buf.observations)                       # (2048, 4, 63) float32
K = np.asarray(cb_rec.k_rows, dtype=np.int64)              # (2048, 4)
expect = np.array([[my_phase(k) for k in row] for row in K], dtype=np.float32)
err = float(np.max(np.abs(bobs[..., LO:HI].astype(np.float64) - expect.astype(np.float64))))
check("F_buffer_slots_equal_phase_of_info_step_counter", err == 0.0, {"max_err": err, "k_min": int(K.min()),
                                                                     "k_max": int(K.max()),
                                                                     "rows_presettle": int((K < 200).sum())})
starts = np.asarray(buf.episode_starts) > 0
check("F_k0_iff_episode_start", np.array_equal(starts, K == 0))
tr = np.array(cb_rec.trans, dtype=np.float64)
k_obs, step_after = tr[:, 0], tr[:, 1]
check("F_counter_after_step_is_k_plus_1", np.array_equal(step_after, k_obs + 1))
gated = tr[:, 2:6]
present = tr[:, 6:10].astype(bool)
settled_obs = np.array([my_phase(int(k))[1] for k in k_obs])
pre = settled_obs == 0.0
out["F_gated_terms_present"] = dict(zip(GATED, present.all(0).tolist()))
check("F_gated_terms_zero_before_settle", bool(np.all(gated[pre] == 0.0)), {"transitions_presettle": int(pre.sum())})
first = (k_obs == 200)
out["F_gated_terms_at_k200_nonzero_frac"] = dict(zip(GATED, (gated[first] != 0.0).mean(0).tolist())) if first.any() \
    else None
out["F_gated_terms_after_settle_nonzero_frac"] = dict(zip(GATED, (gated[~pre] != 0.0).mean(0).tolist()))
check("F_some_gated_term_pays_from_k200", bool(first.any()) and bool(np.all(np.any(gated[first] != 0.0, axis=1))),
      int(first.sum()))
check("F_k199_unsettled", bool(np.all(gated[k_obs == 199] == 0.0)) and int((k_obs == 199).sum()) > 0)
out["F_terminals"] = cb_rec.terminal[:12]
tl = [t for t in cb_rec.terminal if t[1]]
check("F_time_limit_terminal_obs_settled", all(t[0] == 1000 and t[2] == [1.0, 1.0, 1.0] for t in tl), len(tl))
# statistics: slots frozen; the other 60 dims and the count = a control twin fed the same raw batches, slots zero
check("F_rms_slots_still_0_1", np.array_equal(np.asarray(vn.obs_rms.mean)[LO:HI], [0.0] * 3)
      and np.array_equal(np.asarray(vn.obs_rms.var)[LO:HI], [1.0] * 3), type(vn.obs_rms).__name__)
reset_raw_c = reset_raw.copy()
reset_raw_c[:, LO:HI] = 0.0
twin_rms.update(reset_raw_c)
for b in cb_rec.batches:
    bc = b.copy()
    bc[:, LO:HI] = 0.0
    twin_rms.update(bc)
check("F_rms_other_dims_equal_control_twin", np.array_equal(np.asarray(vn.obs_rms.mean)[:LO], twin_rms.mean[:LO])
      and np.array_equal(np.asarray(vn.obs_rms.var)[:LO], twin_rms.var[:LO])
      and float(vn.obs_rms.count) == float(twin_rms.count),
      {"count": float(vn.obs_rms.count), "twin": float(twin_rms.count),
       "max_mean_diff": float(np.max(np.abs(np.asarray(vn.obs_rms.mean)[:LO] - twin_rms.mean[:LO])))})
check("F_rms_symmetry_residual_zero", vn.obs_rms.residual() == (0.0, 0.0), vn.obs_rms.residual())
blob = pickle.loads(pickle.dumps(vn.obs_rms))
check("F_rms_pickles_stock_with_slots", type(blob) is RunningMeanStd and np.array_equal(blob.var[LO:HI], [1.0] * 3))
# the recorded buffer: actor and critic against the unhooked copy on the same rows with the slots zero
xb = torch.as_tensor(bobs.reshape(-1, 63), dtype=torch.float32)
xb0 = xb.clone()
xb0[:, LO:HI] = 0.0
acts = torch.as_tensor(np.asarray(buf.actions).reshape(-1, buf.actions.shape[-1]), dtype=torch.float32)
a_hb, v_hb = actor_outputs(policy, xb)
a_rb, v_rb = actor_outputs(ref, xb0)
check("F_buffer_actor_bitwise_vs_unhooked", all(beq(a_hb[k], a_rb[k]) for k in a_hb))
check("F_buffer_critic_bitwise_vs_unhooked", all(beq(v_hb[k], v_rb[k]) for k in v_hb))
out["F_buffer_values_vs_unhooked_critic_max_abs"] = mx(np.asarray(buf.values).reshape(-1), v_rb["predict_values"])
for trial in range(3):
    xg = xb.clone()
    xg[:, LO:HI] = torch.as_tensor(rng.uniform(-10, 10, size=(len(xb), 3)), dtype=torch.float32)
    a_g, _ = actor_outputs(policy, xg)
    check(f"F_buffer_actor_invariant_garbage_{trial}", all(beq(a_hb[k], a_g[k]) for k in a_hb))

# ------------------------------------------------------------------------------------------------ H: one train()
w_pi = policy.mlp_extractor.policy_net[0].weight
w_vf = policy.mlp_extractor.value_net[0].weight
pi_slots0 = w_pi[:, LO:HI].detach().clone()
pi_all0 = {k: v.detach().clone() for k, v in policy.state_dict().items()}
steps_before = max(float(s["step"]) for s in policy.optimizer.state.values())
model.train()
steps_after = max(float(s["step"]) for s in policy.optimizer.state.values())
out["H_adam_steps"] = [steps_before, steps_after]
sd2 = policy.state_dict()
check("H_some_optimizer_steps", steps_after > steps_before)
check("H_actor_phase_cols_bitwise_unchanged", beq(w_pi[:, LO:HI].detach(), pi_slots0))
check("H_actor_other_params_moved", not beq(sd2["mlp_extractor.policy_net.0.weight"][:, :LO],
                                            pi_all0["mlp_extractor.policy_net.0.weight"][:, :LO]))
st_pi = policy.optimizer.state[w_pi]
st_vf = policy.optimizer.state[w_vf]
check("H_actor_phase_adam_moments_zero", float(st_pi["exp_avg"][:, LO:HI].abs().max()) == 0.0
      and float(st_pi["exp_avg_sq"][:, LO:HI].abs().max()) == 0.0)
out["H_critic_phase_cols_max_abs"] = float(w_vf[:, LO:HI].detach().abs().max())
out["H_critic_phase_cols_per_feature_max_abs"] = w_vf[:, LO:HI].detach().abs().max(0).values.tolist()
check("H_critic_phase_cols_moved", out["H_critic_phase_cols_max_abs"] > 0.0)
check("H_critic_phase_adam_moments_nonzero", float(st_vf["exp_avg_sq"][:, LO:HI].abs().max()) > 0.0)
check("H_mirror_tensors_rebuilt_self_plus1", bool(model._mirror.tensors)
      and bool(torch.all(model._mirror.tensors["obs_sign"][LO:HI] == 1.0)))
check("H_rms_slots_still_0_1", H.phase_slots_frozen(vn.obs_rms))
# after the update: the actor still ignores the slots, on the buffer and on fresh states; the critic responds
a1, v1 = actor_outputs(policy, xb)
xg = xb.clone()
xg[:, LO:HI] = torch.as_tensor(rng.uniform(-10, 10, size=(len(xb), 3)), dtype=torch.float32)
a2, v2 = actor_outputs(policy, xg)
check("H_actor_invariant_after_update_buffer", all(beq(a1[k], a2[k]) for k in a1))
a3, v3 = actor_outputs(policy, z1)
a4, v4 = actor_outputs(policy, garbage)
check("H_actor_invariant_after_update_fresh", all(beq(a3[k], a4[k]) for k in a3))
xs = xb.clone()
xs[:, LO:HI] = 1.0
_, v5 = actor_outputs(policy, xs)
out["H_critic_response"] = {"garbage_vs_phase_buffer": mx(v1["predict_values"], v2["predict_values"]),
                            "settled_vs_own_phase_buffer": mx(v1["predict_values"], v5["predict_values"]),
                            "presettle_rows": int((K < 200).sum())}
check("H_critic_responds_after_update", out["H_critic_response"]["garbage_vs_phase_buffer"] > 0.0)
# the probe after the update: the row's critic is the phase-fed critic (and differs from the slots-zero one)
p, rec_p, row = probe_row()
fv, fa = bank_fed(policy)
z_bank1 = torch.as_tensor(norm_np(bank.raw_obs, np.asarray(vn.obs_rms.mean), np.asarray(vn.obs_rms.var)),
                          dtype=torch.float32)
with torch.no_grad():
    v_zero_slots = policy.predict_values(z_bank1).flatten().numpy()
check("G2_probe_after_update_value_equals_phase_fed", np.array_equal(np.asarray(row["bank_value_now"]),
                                                                      fv.astype(np.float32)))
out["G2_probe_after_update_phase_vs_zero_slots_max_abs"] = mx(fv, v_zero_slots)
out["fails"] = fails
out["RESULT"] = "PASS" if not fails else "FAIL"
vn.close()
print(json.dumps(out, indent=1, default=str))
sys.exit(0 if not fails else 1)
