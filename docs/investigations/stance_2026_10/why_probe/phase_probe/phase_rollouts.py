"""PHASE PROBE: training-like stochastic rollouts of one checkpoint pair, recorded per decision state.  No training:
no optimizer, no update, no continuation; the checkpoint files are opened read only.

    PY -I -B phase_rollouts.py --label L --zip Z --pkl P --stage-config J [--blind] [--episodes-per-env 12]
                               [--seed 7700] [--joint tail_3_pitch] --out OUT.npz
    PY -I -B phase_rollouts.py ... --equivalence N   (compares this loop's first N vector steps with SB3's own
                               collect_rollouts on a fresh build with the same seeds; writes nothing but OUT.json)

The env, VecNormalize and model are built as BS2's reader builds its on-policy states (read_bs.onpolicy_states): the
code copy's stance config (checked equal to the run's stage_config.json reward_weights), create_vec_env with 4 envs,
the pair's VecNormalize with training False (frozen statistics) and norm_reward True, load_sb3_model with the stage's
[ppo] kwargs, enable_mirror_kl.  Actions: policy.forward (sampled), clipped to the action space, exactly as SB3's
collect_rollouts.  --blind installs BS2's bs_hooks.critic_blind_class on the mlp_extractor (the EFFECTIVE critic of
the critic_blind_tail3 arm); the raw value MLP is evaluated after the rollout as well.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
CODE = S2 + "/r10round/bias_screen/code"
BS_HOOKS = S2 + "/r10round/bias_screen2/tools/bs_hooks.py"
COMMIT = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"
FD_DEG = 0.25
TERM_CODES = {"": 0, "fallen": 1, "nosedive": 2, "tail_contact": 3, "head_contact": 4, "skull_low": 5,
              "torso_contact": 6, "other": 9}
EXTRA_INFO = ("stance_width", "stance_width_target", "foot_stagger_m", "com_lateral_offset_m", "pelvis_height",
              "tilt_angle")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def setup(label):
    tmp = HERE / "tmp" / label
    (tmp / "mpl").mkdir(parents=True, exist_ok=True)
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "TMPDIR": str(tmp), "MPLCONFIGDIR": str(tmp / "mpl"), "GITHUB_SHA": COMMIT})
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


def load_hooks():
    spec = importlib.util.spec_from_file_location("bs_hooks", BS_HOOKS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bs_hooks"] = mod
    spec.loader.exec_module(mod)
    return mod


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, float) and v.is_integer():
        return v
    return v


def check_env_kwargs(env_kwargs, stage_config_path):
    """The env the code copy builds must match the run's recorded reward_weights key by key: a key the stance config
    sets is compared directly; a key it leaves out is compared with the env constructor's default (the record lists
    the effective values).  A recorded key that is neither is reported as a difference."""
    import inspect

    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    cls = env_class(ModelId("tyrannosaurus_rex", "anatomical"))
    defaults = {}
    for klass in reversed(cls.__mro__):
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, prm in inspect.signature(init).parameters.items():
            if prm.default is not inspect.Parameter.empty:
                defaults[name] = prm.default
    effective = dict(defaults)
    effective.update(env_kwargs)
    rec = json.loads(Path(stage_config_path).read_text())["reward_weights"]
    diffs = []
    for k in sorted(set(env_kwargs) | set(rec)):
        a, b = effective.get(k, "<missing>"), rec.get(k, "<missing>")
        if isinstance(a, tuple):
            a = list(a)
        if isinstance(b, tuple):
            b = list(b)
        if a != b and not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and float(a) == float(b)):
            diffs.append((k, a, b, "set" if k in env_kwargs else "default"))
    return diffs


def build(zip_path, pkl_path, seed, blind, H):
    """read_bs.onpolicy_states' build, with the arm hook for --blind."""
    import copy

    import numpy as np
    import torch
    from stable_baselines3.common.vec_env import VecNormalize

    from environments.shared.config import load_stage_config
    from environments.shared.mirror_kl_ppo import MirrorKLPPO, enable_mirror_kl
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.policy_loading import load_sb3_model
    from environments.shared.species_registry import env_class
    from environments.shared.symmetry import MirrorMap, pop_symmetry_config
    from environments.shared.train_base import _prepare_alg_kwargs, create_vec_env

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    cfg = copy.deepcopy(load_stage_config(mid, "stance"))

    class _Species:
        model_id = mid
        env_class = None

    species = _Species()
    species.env_class = env_class(mid)
    ppo = copy.deepcopy(cfg["ppo_kwargs"])
    sym = pop_symmetry_config(ppo)
    venv = create_vec_env(species, {1: {"env_kwargs": dict(cfg["env_kwargs"])}}, 1, 4, seed=seed,
                          use_subproc=False, algorithm="ppo", gamma=float(ppo["gamma"]))
    vn = VecNormalize.load(str(pkl_path), venv.venv)
    vn.training = False
    vn.norm_reward = True
    alg_kwargs, _tb, _g = _prepare_alg_kwargs({"ppo_kwargs": copy.deepcopy(cfg["ppo_kwargs"])}, "ppo", 0,
                                              Path(os.environ["TMPDIR"]) / "_tb_unused", use_tensorboard=False)
    pop_symmetry_config(alg_kwargs)
    alg_kwargs.pop("policy_kwargs", None)
    model = load_sb3_model(str(zip_path), algorithm="ppo", env=vn, device="cpu", **alg_kwargs)
    model = enable_mirror_kl(model, sym, MirrorMap.load(mid, current_plant_identity(mid)))
    assert isinstance(model, MirrorKLPPO)
    if blind:
        from stable_baselines3.common.torch_layers import MlpExtractor

        assert type(model.policy.mlp_extractor) is MlpExtractor
        model.policy.mlp_extractor.__class__ = H.critic_blind_class(MlpExtractor)
    vn.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return cfg, vn, model


def env_measures(u):
    """At the current state of one env (before its step): tail_tip clearance, tail_3/4/5 floor distances, pelvis."""
    import mujoco

    tip = float(u._clearance(u.data.site_xpos[u.tail_tip_site_id]))
    d = [float(mujoco.mj_geomDistance(u.model, u.data, int(g), int(u.floor_geom_id), 10.0, None))
         for g in (u.tail_3_geom_id, u.tail_4_geom_id, u.tail_5_geom_id)]
    pel = float(u._clearance(u.data.xpos[u.pelvis_id]))
    return [tip, d[0], d[1], d[2], pel]


def rollout(vn, model, n_per_env, max_vec_steps=None, log=print, return_open=False):
    """This probe's loop: collect_rollouts' action path, recording per decision state."""
    import numpy as np
    import torch

    policy = model.policy
    policy.set_training_mode(False)
    low, high = model.action_space.low, model.action_space.high
    nE = vn.num_envs
    envs = [vn.venv.envs[i].unwrapped for i in range(nE)]
    obs = vn.reset()
    reward_keys = None
    cur = [dict(rows=[], k=0) for _ in range(nE)]
    done_eps = [[] for _ in range(nE)]
    vec_steps = 0
    t0 = time.time()
    while min(len(e) for e in done_eps) < n_per_env:
        if max_vec_steps is not None and vec_steps >= max_vec_steps:
            break
        raw = vn.get_original_obs().copy()
        phys = [env_measures(u) for u in envs]
        for i, u in enumerate(envs):
            if int(u._step_count) != cur[i]["k"]:
                raise SystemExit(f"env {i}: _step_count {u._step_count} != k {cur[i]['k']}")
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs)
            actions, values, _logp = policy(obs_tensor)
            mean = policy.action_dist.distribution.mean.cpu().numpy().copy()
        actions = actions.cpu().numpy()
        clipped = np.clip(actions, low, high)
        new_obs, rewards, dones, infos = vn.step(clipped)
        raw_rew = vn.get_original_reward().copy()
        vec_steps += 1
        if reward_keys is None:
            reward_keys = sorted(k for k in infos[0] if k.startswith("reward_"))
        for i in range(nE):
            info = infos[i]
            row = dict(k=cur[i]["k"], raw=raw[i].astype(np.float32), z=np.asarray(obs[i], dtype=np.float32),
                       a=actions[i].astype(np.float32), mu=mean[i].astype(np.float32),
                       V=float(values[i, 0]), rn=float(rewards[i]), rr=float(raw_rew[i]),
                       terms=np.array([float(info.get(key, np.nan)) for key in reward_keys], dtype=np.float32),
                       extra=np.array([float(info.get(key, np.nan)) for key in EXTRA_INFO], dtype=np.float32),
                       phys=np.array(phys[i], dtype=np.float32))
            cur[i]["rows"].append(row)
            if dones[i]:
                trunc = bool(info.get("TimeLimit.truncated", False))
                reason = str(info.get("termination_reason", "")) if not trunc else ""
                vterm = float("nan")
                if trunc and info.get("terminal_observation") is not None:
                    term_obs = policy.obs_to_tensor(info["terminal_observation"])[0]
                    with torch.no_grad():
                        vterm = float(policy.predict_values(term_obs)[0, 0])
                ep = dict(env=i, rows=cur[i]["rows"], trunc=trunc, term=not trunc, reason=reason, vterm=vterm,
                          monitor_r=float(info.get("episode", {}).get("r", np.nan)),
                          monitor_l=int(info.get("episode", {}).get("l", -1)))
                done_eps[i].append(ep)
                cur[i] = dict(rows=[], k=0)
            else:
                cur[i]["k"] += 1
        obs = new_obs
        if vec_steps % 2000 == 0:
            log(f"  vec step {vec_steps}: episodes done per env {[len(e) for e in done_eps]}, "
                f"{vec_steps * nE / (time.time() - t0):.0f} env steps/s")
    if return_open:
        return done_eps, reward_keys, vec_steps, time.time() - t0, cur
    return done_eps, reward_keys, vec_steps, time.time() - t0


def assemble(done_eps, reward_keys, n_per_env):
    import numpy as np

    eps = []
    for i in range(len(done_eps)):
        eps.extend(done_eps[i][:n_per_env])
    eps.sort(key=lambda e: (e["env"], id(e)))
    out = {k: [] for k in ("ep", "k", "raw", "z", "a", "mu", "V", "rn", "rr", "terms", "extra", "phys")}
    epi = dict(env=[], len=[], trunc=[], term=[], reason=[], vterm=[], monitor_r=[], monitor_l=[])
    for j, e in enumerate(eps):
        for r in e["rows"]:
            out["ep"].append(j)
            for key in ("k", "raw", "z", "a", "mu", "V", "rn", "rr", "terms", "extra", "phys"):
                out[key].append(r[key])
        epi["env"].append(e["env"])
        epi["len"].append(len(e["rows"]))
        epi["trunc"].append(e["trunc"])
        epi["term"].append(e["term"])
        epi["reason"].append(TERM_CODES.get(e["reason"], TERM_CODES["other"]) if e["term"] else 0)
        epi["vterm"].append(e["vterm"])
        epi["monitor_r"].append(e["monitor_r"])
        epi["monitor_l"].append(e["monitor_l"])
    arr = {
        "ep": np.asarray(out["ep"], dtype=np.int32), "k": np.asarray(out["k"], dtype=np.int16),
        "raw_obs": np.stack(out["raw"]), "norm_obs": np.stack(out["z"]), "action": np.stack(out["a"]),
        "mu": np.stack(out["mu"]), "V": np.asarray(out["V"], dtype=np.float64),
        "r_norm": np.asarray(out["rn"], dtype=np.float64), "r_raw": np.asarray(out["rr"], dtype=np.float64),
        "terms": np.stack(out["terms"]), "extra": np.stack(out["extra"]), "phys": np.stack(out["phys"]),
        "ep_env": np.asarray(epi["env"], dtype=np.int16), "ep_len": np.asarray(epi["len"], dtype=np.int32),
        "ep_trunc": np.asarray(epi["trunc"], dtype=bool), "ep_term": np.asarray(epi["term"], dtype=bool),
        "ep_reason": np.asarray(epi["reason"], dtype=np.int16), "ep_vterm": np.asarray(epi["vterm"]),
        "ep_monitor_r": np.asarray(epi["monitor_r"]), "ep_monitor_l": np.asarray(epi["monitor_l"]),
        "reward_keys": np.asarray(reward_keys), "extra_keys": np.asarray(EXTRA_INFO),
        "phys_keys": np.asarray(["tail_tip_z", "tail3_floor_dist", "tail4_floor_dist", "tail5_floor_dist",
                                 "pelvis_z"]),
    }
    return arr


def critic_slopes(policy, vn, Z, qidx, blind_raw=False, H=None):
    """Finite-difference (raw q +- FD_DEG, re-normalised, clipped) and autograd slopes per degree, batched."""
    import numpy as np
    import torch
    from stable_baselines3.common.torch_layers import MlpExtractor

    stored_cls = type(policy.mlp_extractor)
    if blind_raw:
        policy.mlp_extractor.__class__ = MlpExtractor
    try:
        sd = math.sqrt(float(vn.obs_rms.var[qidx]) + float(vn.epsilon))
        clip = float(vn.clip_obs)
        dz = math.radians(FD_DEG) / sd
        V0, fd, ag = [], [], []
        for s in range(0, len(Z), 8192):
            zb = torch.as_tensor(np.asarray(Z[s:s + 8192], dtype=np.float32))
            zp, zm = zb.clone(), zb.clone()
            zp[:, qidx] = torch.clamp(zp[:, qidx] + dz, -clip, clip)
            zm[:, qidx] = torch.clamp(zm[:, qidx] - dz, -clip, clip)
            with torch.no_grad():
                v0 = policy.predict_values(zb)[:, 0].numpy().astype(np.float64)
                vp = policy.predict_values(zp)[:, 0].numpy().astype(np.float64)
                vm = policy.predict_values(zm)[:, 0].numpy().astype(np.float64)
            zg = zb.clone().requires_grad_(True)
            g = torch.autograd.grad(policy.predict_values(zg).sum(), zg)[0][:, qidx].numpy().astype(np.float64)
            V0.append(v0)
            fd.append((vp - vm) / (2 * FD_DEG))
            ag.append(g / sd * math.pi / 180.0)
        return np.concatenate(V0), np.concatenate(fd), np.concatenate(ag)
    finally:
        policy.mlp_extractor.__class__ = stored_cls


def equivalence(args, H, log):
    """This probe's loop vs SB3's own collect_rollouts: same build, same seeds, the first N vector steps (sampled
    actions, values, normalised rewards before the timeout bootstrap)."""
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger

    n = int(args.equivalence)
    _cfg, vn, model = build(args.zip, args.pkl, args.seed, args.blind, H)
    done_eps, _keys, _vs, _t, cur = rollout(vn, model, 10 ** 9, max_vec_steps=n, log=log, return_open=True)
    vn.close()
    per_env = []
    for i in range(4):
        rows = [r for e in done_eps[i] for r in e["rows"]] + cur[i]["rows"]
        per_env.append(rows[:n])
    mine_a = np.stack([np.stack([per_env[i][t]["a"] for i in range(4)]) for t in range(n)])
    mine_v = np.array([[per_env[i][t]["V"] for i in range(4)] for t in range(n)])
    mine_r = np.array([[per_env[i][t]["rn"] for i in range(4)] for t in range(n)])
    _cfg, vn2, model2 = build(args.zip, args.pkl, args.seed, args.blind, H)
    model2.set_logger(Logger(folder=None, output_formats=[]))
    model2._last_obs = None

    class Rec(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.a, self.v, self.r = [], [], []

        def _on_step(self):
            self.a.append(np.asarray(self.locals["actions"]).copy())
            self.v.append(self.locals["values"].detach().numpy().copy())
            self.r.append(np.asarray(self.locals["rewards"]).copy())
            return len(self.a) < n

    rec = Rec()
    _t2, cb = model2._setup_learn(model2.n_steps * model2.n_envs, rec, False, "pp_equiv", False)
    model2.collect_rollouts(model2.env, cb, model2.rollout_buffer, n_rollout_steps=model2.n_steps)
    vn2.close()
    A = np.stack(rec.a)[:n]
    Vv = np.stack(rec.v)[:n, :, 0]
    R = np.stack(rec.r)[:n]
    return {"n_vec_steps": n, "max_abs_diff_actions": float(np.max(np.abs(mine_a - A))),
            "max_abs_diff_values": float(np.max(np.abs(mine_v - Vv))),
            "max_abs_diff_rewards": float(np.max(np.abs(mine_r - R)))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--stage-config", required=True)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--episodes-per-env", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7700)
    ap.add_argument("--joint", default="tail_3_pitch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--equivalence", type=int, default=0)
    ap.add_argument("--max-vec-steps", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if not str(out).startswith(str(HERE) + os.sep):
        raise SystemExit(f"refusing to write outside {HERE}: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC [{args.label}] {msg}", flush=True)

    origin = setup(args.label)
    H = load_hooks()
    import numpy as np
    import stable_baselines3
    import torch

    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId

    meta = {"label": args.label, "zip": args.zip, "pkl": args.pkl, "stage_config": args.stage_config,
            "blind": bool(args.blind), "seed": args.seed, "episodes_per_env": args.episodes_per_env,
            "joint": args.joint, "environments_origin": origin, "python": sys.version.split()[0],
            "torch": torch.__version__, "numpy": np.__version__, "sb3": stable_baselines3.__version__,
            "zip_sha256": sha256_file(args.zip), "pkl_sha256": sha256_file(args.pkl),
            "bs_hooks_sha256": sha256_file(BS_HOOKS), "script_sha256": sha256_file(__file__)}
    mid = ModelId("tyrannosaurus_rex", "anatomical")
    diffs = check_env_kwargs(dict(load_stage_config(mid, "stance")["env_kwargs"]), args.stage_config)
    meta["env_kwargs_diffs"] = [list(map(str, d)) for d in diffs]
    if diffs:
        log(f"REFUSED: env kwargs differ from the run's stage_config.json: {diffs}")
        return 2
    if args.equivalence:
        res = equivalence(args, H, log)
        meta["equivalence"] = res
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        log(f"equivalence: {res}")
        return 0
    cfg, vn, model = build(args.zip, args.pkl, args.seed, args.blind, H)
    u0 = vn.venv.envs[0].unwrapped
    import mujoco

    jid = mujoco.mj_name2id(u0.model, mujoco.mjtObj.mjOBJ_JOINT, args.joint)
    qidx = int(u0.model.jnt_qposadr[jid]) - 7
    if args.joint == "tail_3_pitch" and qidx != 6:
        raise SystemExit(f"tail_3_pitch is observation {qidx}, not 6")
    if u0.model.actuator(14).name != "tail_3_pitch_act":
        raise SystemExit("action 14 is not tail_3_pitch_act")
    meta.update(qidx=qidx, num_timesteps=int(model.num_timesteps), gamma=float(vn.gamma),
                sigma_r=math.sqrt(float(vn.ret_rms.var) + float(vn.epsilon)), ret_rms_var=float(vn.ret_rms.var),
                epsilon=float(vn.epsilon), clip_obs=float(vn.clip_obs), clip_reward=float(vn.clip_reward),
                obs_mean=np.asarray(vn.obs_rms.mean, dtype=np.float64).tolist(),
                obs_var=np.asarray(vn.obs_rms.var, dtype=np.float64).tolist(),
                norm_obs=bool(vn.norm_obs), norm_reward=bool(vn.norm_reward), training=bool(vn.training),
                mlp_extractor_class=type(model.policy.mlp_extractor).__name__,
                settle_steps=int(cfg["env_kwargs"]["stance_width_settle_steps"]),
                max_episode_steps=int(cfg["env_kwargs"]["max_episode_steps"]),
                log_std=model.policy.log_std.detach().numpy().tolist())
    log(f"built: num_timesteps {meta['num_timesteps']}, sigma_r {meta['sigma_r']:.4f}, qidx {qidx}, "
        f"extractor {meta['mlp_extractor_class']}")
    mvs = args.max_vec_steps or None
    done_eps, reward_keys, vec_steps, secs = rollout(vn, model, args.episodes_per_env, max_vec_steps=mvs, log=log)
    meta.update(vec_steps=vec_steps, rollout_seconds=round(secs, 1),
                env_steps_per_s=round(vec_steps * 4 / max(secs, 1e-9), 1),
                episodes_done_per_env=[len(e) for e in done_eps])
    if min(len(e) for e in done_eps) < args.episodes_per_env:
        log(f"stopped early: {meta['episodes_done_per_env']}")
        meta["complete"] = False
        n_keep = min(len(e) for e in done_eps)
    else:
        meta["complete"] = True
        n_keep = args.episodes_per_env
    if n_keep == 0:
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        log("no complete episode; wrote meta only")
        return 0
    arr = assemble(done_eps, reward_keys, n_keep)
    policy = model.policy
    t1 = time.time()
    V0, fd, ag = critic_slopes(policy, vn, arr["norm_obs"], qidx)
    arr["V_recomp"], arr["fd_slope"], arr["ag_slope"] = V0, fd, ag
    if args.blind:
        Vr, fdr, agr = critic_slopes(policy, vn, arr["norm_obs"], qidx, blind_raw=True, H=H)
        arr["V_raw"], arr["fd_slope_raw"], arr["ag_slope_raw"] = Vr, fdr, agr
    # the tail_3 slope as well when another joint is the probe's (N80)
    if qidx != 6:
        _v, fd6, ag6 = critic_slopes(policy, vn, arr["norm_obs"], 6)
        arr["fd_slope_tail3"], arr["ag_slope_tail3"] = fd6, ag6
    meta["slopes_seconds"] = round(time.time() - t1, 1)
    meta["max_abs_V_recomp_minus_V"] = float(np.max(np.abs(V0 - arr["V"])))
    meta["n_states"] = int(len(arr["k"]))
    meta["n_episodes"] = int(len(arr["ep_len"]))
    arr["meta_json"] = np.asarray(json.dumps(meta))
    tmp = out.with_name(out.name + ".partial.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    vn.close()
    log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {meta['n_episodes']} episodes, {meta['n_states']} states, "
        f"{meta['env_steps_per_s']} env steps/s, slopes {meta['slopes_seconds']} s, "
        f"max |V recomputed - V| {meta['max_abs_V_recomp_minus_V']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
