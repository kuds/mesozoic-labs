"""verify_why: fresh rollouts of one checkpoint pair (PREREG Part F).  No training: no optimizer is built or stepped,
no update, no continuation; checkpoint files are opened read only; nothing is written outside this folder.

    PY -I -B vw_roll.py --label C11E --mode stoch --seed 8810 --rollouts 6 --out runs/FR1_C11E_stoch.npz
    PY -I -B vw_roll.py --label C11E --mode det   --seed 8820 --episodes-per-env 6 --out runs/FR2_C11E_det.npz
    PY -I -B vw_roll.py ... --smoke                (timing and shapes only; prints nothing about outcomes)

STOCHASTIC: SB3's own MirrorKLPPO.collect_rollouts (sampled actions, clipped, the time-limit bootstrap, GAE into the
rollout buffer), called --rollouts times after one _setup_learn, with a callback that records each decision state.
DETERMINISTIC: my loop on the same build, the Gaussian mean clipped to the action box.
Build = read_bs.onpolicy_states' build (4-env DummyVecEnv + Monitor, the pair's VecNormalize, training False,
norm_reward True, load_sb3_model with the stage's [ppo] kwargs, enable_mirror_kl).
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
CODE = S2 + "/r10round/bias_screen/code"
COMMIT = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"
F2 = S1 + "/foot/stance6/fetch/F2/stance6_r10LTSB_s42/copies/01_stance"
BS2R = S2 + "/r10round/bias_screen2/runs"
PAIRS = {
    "C11E": (f"{BS2R}/control_2511/01_stance/models/stage1_final.zip",
             f"{BS2R}/control_2511/01_stance/models/stage1_final_vecnorm.pkl",
             f"{BS2R}/control_2511/01_stance/stage_config.json"),
    "P2": (f"{F2}/models/stage1_2097152_steps.zip.copy1", f"{F2}/models/stage1_vecnormalize_2097152_steps.pkl.copy1",
           f"{F2}/stage_config.json.copy1"),
}
INFO_KEYS = ("reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band",
             "reward_tail_home_pose", "reward_total", "lateral_lean_quality", "stance_stagger_quality",
             "stance_cop_band_quality", "r_foot_terms_supported", "l_foot_terms_supported", "stance_width",
             "stance_width_target")
OBS_T3, ACT_T3 = 6, 14
FD_DEG = 0.25


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def setup(tag):
    tmp = HERE / "tmp" / tag
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


def build(zip_path, pkl_path, seed):
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
    vn.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return cfg, vn, model


def phys(u):
    """Before the step: tail_tip site z, the lowest point of the tail_3/4/5 capsules (world z), pelvis z."""
    d, m = u.data, u.model
    low = []
    for g in (u.tail_3_geom_id, u.tail_4_geom_id, u.tail_5_geom_id):
        zc = float(d.geom_xpos[g][2])
        axz = float(d.geom_xmat[g].reshape(3, 3)[2, 2])
        r, hl = float(m.geom_size[g][0]), float(m.geom_size[g][1])
        low.append(zc - abs(hl * axz) - r)
    return [float(d.site_xpos[u.tail_tip_site_id][2]), min(low), float(d.xpos[u.pelvis_id][2])]


class Recorder:
    """Per-env sequences of decision states (every step, partial episodes included)."""

    def __init__(self, vn, policy):
        self.vn, self.policy = vn, policy
        self.envs = [vn.venv.envs[i].unwrapped for i in range(vn.num_envs)]
        self.seq = [[] for _ in range(vn.num_envs)]
        self.pre = None

    def snapshot(self):
        import numpy as np

        raw = self.vn.get_original_obs().copy()
        return [dict(k=int(u._step_count), raw=raw[i].astype(np.float32), phys=phys(u))
                for i, u in enumerate(self.envs)]

    def record(self, z, actions, clipped, values, rn, infos, dones):
        import numpy as np
        import torch

        rr = self.vn.get_original_reward().copy()
        for i in range(len(self.envs)):
            info = infos[i]
            row = dict(self.pre[i])
            row.update(z=np.asarray(z[i], dtype=np.float32), a=np.asarray(actions[i], dtype=np.float32),
                       ca=np.asarray(clipped[i], dtype=np.float32), V=float(values[i]), rn=float(rn[i]),
                       rr=float(rr[i]), info=[float(info.get(key, np.nan)) for key in INFO_KEYS],
                       done=bool(dones[i]), trunc=False, reason="", vterm=float("nan"))
            if dones[i]:
                trunc = bool(info.get("TimeLimit.truncated", False))
                row["trunc"] = trunc
                row["reason"] = "" if trunc else str(info.get("termination_reason", "unknown"))
                if trunc and info.get("terminal_observation") is not None:
                    t_obs = self.policy.obs_to_tensor(info["terminal_observation"])[0]
                    with torch.no_grad():
                        row["vterm"] = float(self.policy.predict_values(t_obs)[0, 0])
            self.seq[i].append(row)
        self.pre = self.snapshot()


def run_stoch(vn, model, n_rollouts, log):
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger

    rec = Recorder(vn, model.policy)

    class CB(BaseCallback):
        def __init__(self):
            super().__init__(0)

        def _on_rollout_start(self):
            rec.pre = rec.snapshot()

        def _on_step(self):
            L = self.locals
            rec.record(L["obs_tensor"].cpu().numpy(), L["actions"], L["clipped_actions"],
                       L["values"].cpu().numpy()[:, 0], np.asarray(L["rewards"], dtype=np.float64).copy(),
                       L["infos"], L["dones"])
            return True

    model.set_logger(Logger(folder=None, output_formats=[]))
    model._last_obs = None
    cb_obj = CB()
    _t, cb = model._setup_learn(model.n_steps * model.n_envs * n_rollouts, cb_obj, False, "vw", False)
    buf = {k: [] for k in ("rewards", "values", "returns", "advantages", "episode_starts")}
    t0 = time.time()
    for r in range(n_rollouts):
        if not model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps):
            raise SystemExit("collect_rollouts stopped early")
        rb = model.rollout_buffer
        for k in buf:
            buf[k].append(np.asarray(getattr(rb, k), dtype=np.float64).copy())   # (n_steps, n_envs)
        log(f"rollout {r + 1}/{n_rollouts} done, {(r + 1) * model.n_steps * 4 / (time.time() - t0):.0f} env steps/s")
    for k in buf:
        buf[k] = np.concatenate(buf[k], 0).T.copy()   # (n_envs, n_rollouts * n_steps)
    return rec, buf, time.time() - t0


def run_det(vn, model, n_per_env, log):
    import numpy as np
    import torch

    policy = model.policy
    policy.set_training_mode(False)
    low, high = model.action_space.low, model.action_space.high
    rec = Recorder(vn, policy)
    obs = vn.reset()
    rec.pre = rec.snapshot()
    done_count = [0] * vn.num_envs
    t0 = time.time()
    steps = 0
    while min(done_count) < n_per_env:
        with torch.no_grad():
            ot = torch.as_tensor(obs)
            mu = policy.get_distribution(ot).distribution.mean.cpu().numpy()
            values = policy.predict_values(ot).cpu().numpy()[:, 0]
        clipped = np.clip(mu, low, high)
        new_obs, rewards, dones, infos = vn.step(clipped)
        rec.record(obs, mu, clipped, values, np.asarray(rewards, dtype=np.float64).copy(), infos, dones)
        for i in range(vn.num_envs):
            done_count[i] += int(dones[i])
        obs = new_obs
        steps += 1
        if steps % 2000 == 0:
            log(f"det step {steps}: episodes per env {done_count}, {steps * 4 / (time.time() - t0):.0f} env steps/s")
    return rec, None, time.time() - t0


def assemble(rec):
    """Complete episodes only, in (env, order) order; plus the full per-env sequences for the buffer checks."""
    import numpy as np

    keys = ("k", "raw", "z", "a", "ca", "V", "rn", "rr", "info", "phys")
    cols = {k: [] for k in keys}
    ep_idx, ep_meta = [], []
    full = {k: [] for k in ("env_rn", "env_V", "env_done", "env_trunc", "env_vterm", "env_k")}
    for i, seq in enumerate(rec.seq):
        start = 0
        for j, row in enumerate(seq):
            if row["done"]:
                rows = seq[start:j + 1]
                if rows[0]["k"] != 0:
                    raise SystemExit(f"env {i}: an episode does not start at k = 0")
                e = len(ep_meta)
                for r in rows:
                    for k in keys:
                        cols[k].append(r[k])
                    ep_idx.append(e)
                ep_meta.append(dict(env=i, len=len(rows), trunc=rows[-1]["trunc"], reason=rows[-1]["reason"],
                                    vterm=rows[-1]["vterm"], start=start))
                start = j + 1
        full["env_rn"].append([r["rn"] for r in seq])
        full["env_V"].append([r["V"] for r in seq])
        full["env_done"].append([r["done"] for r in seq])
        full["env_trunc"].append([r["trunc"] for r in seq])
        full["env_vterm"].append([r["vterm"] for r in seq])
        full["env_k"].append([r["k"] for r in seq])
    arr = {"ep": np.asarray(ep_idx, dtype=np.int32), "k": np.asarray(cols["k"], dtype=np.int16),
           "raw": np.stack(cols["raw"]).astype(np.float32), "z": np.stack(cols["z"]).astype(np.float32),
           "a": np.stack(cols["a"]).astype(np.float32), "ca": np.stack(cols["ca"]).astype(np.float32),
           "V": np.asarray(cols["V"]), "rn": np.asarray(cols["rn"]), "rr": np.asarray(cols["rr"]),
           "info": np.asarray(cols["info"], dtype=np.float64), "phys": np.asarray(cols["phys"], dtype=np.float32),
           "info_keys": np.asarray(INFO_KEYS), "phys_keys": np.asarray(["tail_tip_z", "tail_low_z", "pelvis_z"]),
           "ep_env": np.asarray([m["env"] for m in ep_meta], dtype=np.int16),
           "ep_len": np.asarray([m["len"] for m in ep_meta], dtype=np.int32),
           "ep_trunc": np.asarray([m["trunc"] for m in ep_meta], dtype=bool),
           "ep_reason": np.asarray([m["reason"] for m in ep_meta]),
           "ep_vterm": np.asarray([m["vterm"] for m in ep_meta], dtype=np.float64),
           "ep_start": np.asarray([m["start"] for m in ep_meta], dtype=np.int32)}
    n = min(len(x) for x in full["env_rn"])
    for k, v in full.items():
        arr[k] = np.asarray([x[:n] for x in v])
    return arr


def network_reads(policy, vn, Z):
    """Batched, after the rollout: Gaussian means (15), critic FD slope on raw q3 (+-0.25 deg, renormalised,
    clipped as VecNormalize clips), autograd dV/dz for all 63 inputs."""
    import numpy as np
    import torch

    sd = math.sqrt(float(vn.obs_rms.var[OBS_T3]) + float(vn.epsilon))
    clip = float(vn.clip_obs)
    dz = math.radians(FD_DEG) / sd
    MU, FD, GR = [], [], []
    stored = dict(vars(policy.action_dist))
    try:
        for s in range(0, len(Z), 8192):
            zb = torch.as_tensor(np.asarray(Z[s:s + 8192], dtype=np.float32))
            with torch.no_grad():
                MU.append(policy.get_distribution(zb).distribution.mean.numpy().astype(np.float32))
                zp, zm = zb.clone(), zb.clone()
                zp[:, OBS_T3] = torch.clamp(zp[:, OBS_T3] + dz, -clip, clip)
                zm[:, OBS_T3] = torch.clamp(zm[:, OBS_T3] - dz, -clip, clip)
                FD.append(((policy.predict_values(zp) - policy.predict_values(zm))[:, 0].numpy().astype(np.float64))
                          / (2 * FD_DEG))
            zg = zb.clone().requires_grad_(True)
            GR.append(torch.autograd.grad(policy.predict_values(zg).sum(), zg)[0].numpy().astype(np.float32))
    finally:
        vars(policy.action_dist).clear()
        vars(policy.action_dist).update(stored)
    return np.concatenate(MU), np.concatenate(FD), np.concatenate(GR)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=sorted(PAIRS))
    ap.add_argument("--mode", required=True, choices=("stoch", "det"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rollouts", type=int, default=6)
    ap.add_argument("--episodes-per-env", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    if HERE.resolve() not in out.parents:
        raise SystemExit(f"refusing to write outside {HERE}: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{a.label}_{a.mode}_{a.seed}"

    def log(msg):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC [{tag}] {msg}", flush=True)

    origin = setup(tag)
    import numpy as np
    import stable_baselines3
    import torch

    zp, pp, sc = PAIRS[a.label]
    cfg, vn, model = build(zp, pp, a.seed)
    u0 = vn.venv.envs[0].unwrapped
    assert u0.model.actuator(ACT_T3).name == "tail_3_pitch_act"
    import mujoco

    assert int(u0.model.jnt_qposadr[mujoco.mj_name2id(u0.model, mujoco.mjtObj.mjOBJ_JOINT, "tail_3_pitch")]) - 7 == OBS_T3
    rec_cfg = json.loads(Path(sc).read_text())["reward_weights"]
    ek = cfg["env_kwargs"]
    mismatch = {k: (ek[k], rec_cfg.get(k)) for k in ek if k in rec_cfg and ek[k] != rec_cfg[k]
                and not (isinstance(ek[k], (list, tuple)) and list(ek[k]) == rec_cfg[k])}
    meta = {"label": a.label, "mode": a.mode, "seed": a.seed, "zip": zp, "pkl": pp, "stage_config": sc,
            "zip_sha256": sha256_file(zp), "pkl_sha256": sha256_file(pp), "script_sha256": sha256_file(__file__),
            "origin": origin, "python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "sb3": stable_baselines3.__version__,
            "env_kwargs_vs_record_mismatch": {k: [str(x) for x in v] for k, v in mismatch.items()},
            "num_timesteps": int(model.num_timesteps), "gamma": float(vn.gamma), "model_gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda), "n_steps": int(model.n_steps),
            "sigma_r": math.sqrt(float(vn.ret_rms.var) + float(vn.epsilon)), "epsilon": float(vn.epsilon),
            "clip_obs": float(vn.clip_obs), "clip_reward": float(vn.clip_reward), "training": bool(vn.training),
            "norm_obs": bool(vn.norm_obs), "norm_reward": bool(vn.norm_reward),
            "vecnormalize_class": type(vn).__name__, "obs_rms_class": type(vn.obs_rms).__name__,
            "obs_mean": np.asarray(vn.obs_rms.mean, dtype=np.float64).tolist(),
            "obs_var": np.asarray(vn.obs_rms.var, dtype=np.float64).tolist(),
            "weights": {k: float(getattr(u0, k)) for k in ("stance_width_weight", "lateral_lean_weight",
                                                           "stance_stagger_weight", "stance_cop_band_weight")},
            "settle_steps": int(u0.stance_width_settle_steps), "max_episode_steps": int(ek["max_episode_steps"]),
            "log_std": model.policy.log_std.detach().numpy().tolist(),
            "mlp_extractor_class": type(model.policy.mlp_extractor).__name__}
    if mismatch:
        log(f"REFUSED: the code copy's stance env_kwargs differ from the run's record: {mismatch}")
        return 2
    log(f"built: num_timesteps {meta['num_timesteps']} sigma_r {meta['sigma_r']:.4f} gamma {meta['gamma']} "
        f"gae_lambda {meta['gae_lambda']} n_steps {meta['n_steps']} vecnorm {meta['vecnormalize_class']}")
    if a.smoke:
        if a.mode != "stoch":
            raise SystemExit("--smoke is for --mode stoch only")
        rec, buf, secs = run_stoch(vn, model, 1, log)
        log(f"smoke: rows per env {[len(s) for s in rec.seq]}, buffer keys {sorted(buf)} shape "
            f"{buf['rewards'].shape}, {secs:.1f} s; no outcome read, nothing saved")
        vn.close()
        return 0
    if a.mode == "stoch":
        rec, buf, secs = run_stoch(vn, model, a.rollouts, log)
    else:
        rec, buf, secs = run_det(vn, model, a.episodes_per_env, log)
    arr = assemble(rec)
    if buf is not None:
        for k, v in buf.items():
            arr["buf_" + k] = v
    t1 = time.time()
    mu, fd, gr = network_reads(model.policy, vn, arr["z"])
    arr["mu"], arr["fd6"], arr["grad"] = mu, fd, gr
    meta.update(rollout_seconds=round(secs, 1), reads_seconds=round(time.time() - t1, 1),
                n_states=int(len(arr["k"])), n_episodes=int(len(arr["ep_len"])),
                rows_per_env=[len(s) for s in rec.seq])
    arr["meta_json"] = np.asarray(json.dumps(meta))
    tmp = out.with_name(out.name + ".partial.npz")
    np.savez_compressed(tmp, **arr)
    os.replace(tmp, out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    vn.close()
    log(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB): {meta['n_episodes']} complete episodes, "
        f"{meta['n_states']} states, rollout {secs:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
