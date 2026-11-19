"""Gait analysis of the trike60-pipe1 policy (PLAN_GAIT_PIPE1.md). Simulation only; reported, not judged.

Usage (with MUJOCO_GL=osmesa for the stills):
  python gait_pipe1.py --ckpt-dir DIR [--out results] [--date 2026-11-19]
DIR holds trike60_2007040_steps.zip, trike60_vecnormalize_2007040_steps.pkl and run_record.json from the run folder.

Nothing stored in the checkpoint is executed: the policy weights load with torch.load(weights_only=True) into a policy
built here, and the normalisation statistics through an unpickler restricted to plain data.
"""
import argparse
import hashlib
import io
import json
import math
import pickle
import sys
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL = HERE.parent
sys.path.insert(0, str(RL))
from trike_env import CORNERS, TrikeEnv, TrikeEnvConfig  # noqa: E402

GEN = RL.parent / "sim" / "generated"
SEEDS = (100000, 100001, 100002, 100003, 100004)
DELAY = 10
WINDOW_S = 2.0
THERMAL = {"roll": 7.5, "pitch": 7.5, "front elbow": 10.8, "hind knee": 6.75}
LIMIT = {"roll": 21.0, "pitch": 21.0, "front elbow": 37.8, "hind knee": 18.9}
RPM_ALLOW = {"roll": 83.0, "pitch": 83.0, "front elbow": 108.0, "hind knee": 83.0}
SLIP_FLAG, CONTACT_SENS_FLAG = 0.25, 0.20


# ---------------------------------------------------------------- safe loading
class _Plain:
    """Stands in for every non-data class in the normalisation pickle; keeps only its attributes."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            self.__dict__.update(state[1])


def _factory(*a, **k):
    return _Plain()


class _RestrictedUnpickler(pickle.Unpickler):
    DATA = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype"), ("numpy._core.multiarray", "scalar"),
            ("numpy.core.multiarray", "scalar")}
    PLAIN = {("stable_baselines3.common.vec_env.vec_normalize", "VecNormalize"),
             ("stable_baselines3.common.running_mean_std", "RunningMeanStd"), ("gymnasium.spaces.box", "Box"),
             ("numpy.random._pcg64", "PCG64"), ("numpy.random.bit_generator", "SeedSequence")}
    FUNCS = {("numpy.random._pickle", "__generator_ctor"), ("numpy.random._pickle", "__bit_generator_ctor"),
             ("numpy.random.bit_generator", "__pyx_unpickle_SeedSequence")}

    def find_class(self, module, name):
        if (module, name) in self.DATA:
            return super().find_class(module, name)
        if (module, name) in self.PLAIN:
            return _Plain
        if (module, name) in self.FUNCS:
            return _factory
        raise pickle.UnpicklingError(f"refused to load {module}.{name}")


def load_obs_stats(pkl: Path):
    vn = _RestrictedUnpickler(io.BytesIO(pkl.read_bytes())).load()
    rms = vn.obs_rms
    return (np.asarray(rms.mean, dtype=np.float64), np.asarray(rms.var, dtype=np.float64),
            float(getattr(vn, "clip_obs", 10.0)), float(getattr(vn, "epsilon", 1e-8)))


def build_policy(env: TrikeEnv, zip_path: Path, net, log_std_init):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    model = PPO("MlpPolicy", DummyVecEnv([lambda: env]), device="cpu", seed=0,
                policy_kwargs=dict(net_arch=dict(pi=list(net), vf=list(net)), activation_fn=torch.nn.ELU,
                                   log_std_init=log_std_init))
    with zipfile.ZipFile(zip_path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), weights_only=True, map_location="cpu")
    model.policy.load_state_dict(sd, strict=True)
    model.policy.set_training_mode(False)
    return model.policy


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- one episode
def run_episode(env: TrikeEnv, policy, stats, seed, command, still_at=None, renderer=None):
    mean, var, clip, eps = stats
    obs, _ = env.reset(seed=seed, options={"command_speed": command, "delay_steps": DELAY})
    m, d = env.model, env.data
    idx = np.asarray(env._leg_act, dtype=np.int64)
    names = [m.actuator(int(i)).name for i in idx]
    log = {k: [] for k in ("t", "tau", "omega", "load", "sole_v", "pos", "rp")}
    vel6 = np.zeros(6)
    lim_ep = m.actuator_forcerange[idx, 1].copy()   # this episode's (randomised) limits

    def hook(e):
        log["t"].append(d.time)
        log["tau"].append(d.actuator_force[idx].copy())
        log["omega"].append(d.actuator_velocity[idx].copy())
        log["load"].append(d.sensordata[e._touch_adr].copy())
        sv = []
        for g in e._soles:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, int(g), vel6, 0)
            sv.append(math.hypot(vel6[3], vel6[4]))
        log["sole_v"].append(sv)
        log["pos"].append(d.xpos[e._chassis].copy())
        R = d.xmat[e._chassis].reshape(3, 3)
        log["rp"].append((math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(-max(-1, min(1, R[2, 0]))))))

    env.substep_hook = hook
    frames, fell, term = {}, False, ""
    heading0 = None
    try:
        while True:
            o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            with torch.no_grad():
                act, _, _ = policy(torch.as_tensor(o[None]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(act.numpy()[0])
            if heading0 is None:
                heading0 = env._heading
            if still_at and renderer is not None and any(abs(info["time_s"] - t) < 1e-9 for t in still_at):
                for cam, (az, el) in (("side", (90.0, -8.0)), ("overview", (135.0, -22.0))):
                    vc = mujoco.MjvCamera()   # follows the body: the fixed cameras frame the start position only
                    vc.type = mujoco.mjtCamera.mjCAMERA_FREE
                    vc.lookat[:] = d.xpos[env._chassis] + np.array([0.05, 0.0, 0.05])
                    vc.distance, vc.azimuth, vc.elevation = 1.55, az, el
                    renderer.update_scene(d, camera=vc)
                    frames[(info["time_s"], cam)] = renderer.render().copy()
            if terminated or truncated:
                fell, term = bool(terminated), info["termination"]
                break
    finally:
        env.substep_hook = None
    arr = {k: np.asarray(v) for k, v in log.items()}
    arr["names"], arr["limit_ep"], arr["heading0"], arr["mass"] = names, lim_ep, heading0, float(m.body_mass.sum())
    arr["fell"], arr["termination"], arr["contact_n"] = fell, term, env.cfg.contact_threshold_n
    return arr, frames


def group_of(name):
    if name.endswith("_roll"):
        return "roll"
    if name.endswith("_pitch"):
        return "pitch"
    return "front elbow" if "_fore_" in name else "hind knee"


def measures(a, command):
    t = a["t"]
    w = t >= WINDOW_S
    dt = float(np.median(np.diff(t)))
    T = float(t[w][-1] - t[w][0] + dt)
    h = np.array([math.cos(a["heading0"]), math.sin(a["heading0"])])
    p = a["pos"][w]
    disp = p[-1, :2] - p[0, :2]
    fwd = float(disp @ h)
    out = {"command_mps": command, "fell": a["fell"], "termination": a["termination"],
           "forward_speed_mps": fwd / T, "sideways_drift_m": float(disp @ np.array([-h[1], h[0]])),
           "height_range_mm": float(1000 * (p[:, 2].max() - p[:, 2].min()))}
    load = a["load"][w]
    down = load > a["contact_n"]
    out["duty_factor"] = {c: float(down[:, i].mean()) for i, c in enumerate(CORNERS)}
    n = down.sum(1)
    LF, RF, LH, RH = (down[:, CORNERS.index(c)] for c in ("left_fore", "right_fore", "left_hind", "right_hind"))
    two = n == 2
    out["feet_down_fraction"] = {
        "4": float((n == 4).mean()), "3": float((n == 3).mean()),
        "2 diagonal": float((two & ((LF & RH) | (RF & LH))).mean()),
        "2 same side": float((two & ((LF & LH) | (RF & RH))).mean()),
        "2 front": float((two & LF & RF).mean()), "2 hind": float((two & LH & RH).mean()),
        "1": float((n == 1).mean()), "0": float((n == 0).mean())}
    td = (down[1:] & ~down[:-1]).sum(0)
    out["touchdowns_per_s"] = {c: float(td[i] / T) for i, c in enumerate(CORNERS)}
    sv = a["sole_v"][w]
    slip = {}
    for i, c in enumerate(CORNERS):
        s = sv[down[:, i], i]
        slip[c] = {"mean_mps": float(s.mean()) if len(s) else None,
                   "p95_mps": float(np.percentile(s, 95)) if len(s) else None,
                   "slip_ratio": float(s.sum() * dt / fwd) if len(s) and fwd > 0.05 else None}
    out["stance_slip"] = slip
    tau, om = a["tau"][w], a["omega"][w]
    groups = {}
    for gname in THERMAL:
        cols = [i for i, nm in enumerate(a["names"]) if group_of(nm) == gname]
        rms = np.sqrt((tau[:, cols] ** 2).mean(0))
        sat = (np.abs(tau[:, cols]) >= 0.99 * a["limit_ep"][cols]).mean()
        groups[gname] = {"rms_worst_nm": float(rms.max()), "rms_mean_nm": float(rms.mean()),
                         "rms_over_thermal": float(rms.max() / THERMAL[gname]),
                         "peak_nm": float(np.abs(tau[:, cols]).max()),
                         "peak_over_limit": float(np.abs(tau[:, cols]).max() / LIMIT[gname]),
                         "saturated_fraction": float(sat),
                         "peak_rpm": float(np.abs(om[:, cols]).max() * 60 / (2 * math.pi)),
                         "rpm_over_allowance": float(np.abs(om[:, cols]).max() * 60 / (2 * math.pi) / RPM_ALLOW[gname])}
    out["drives"] = groups
    power = float(np.abs(tau * om).sum(1).mean())
    out["power_w"] = power
    out["cost_of_transport"] = power / (a["mass"] * 9.81 * (fwd / T)) if fwd / T > 0.02 else None
    rp = a["rp"][w]
    out["roll_deg"] = {"rms": float(np.sqrt((rp[:, 0] ** 2).mean())), "peak": float(np.abs(rp[:, 0]).max())}
    out["pitch_deg"] = {"rms": float(np.sqrt((rp[:, 1] ** 2).mean())), "peak": float(np.abs(rp[:, 1]).max())}
    return out


def summarise(eps):
    """Across episodes: means of scalars; worst case for the flagged drive and slip figures."""
    s = {"episodes": len(eps), "falls": sum(e["fell"] for e in eps),
         "forward_speed_mps": float(np.mean([e["forward_speed_mps"] for e in eps])),
         "power_w": float(np.mean([e["power_w"] for e in eps])),
         "cost_of_transport": float(np.mean([e["cost_of_transport"] for e in eps if e["cost_of_transport"]])) if any(
             e["cost_of_transport"] for e in eps) else None,
         "duty_factor": {c: float(np.mean([e["duty_factor"][c] for e in eps])) for c in CORNERS},
         "feet_down_fraction": {k: float(np.mean([e["feet_down_fraction"][k] for e in eps]))
                                for k in eps[0]["feet_down_fraction"]},
         "touchdowns_per_s": {c: float(np.mean([e["touchdowns_per_s"][c] for e in eps])) for c in CORNERS},
         "slip_ratio_worst": {c: max((e["stance_slip"][c]["slip_ratio"] or 0) for e in eps) for c in CORNERS},
         "slip_ratio_mean": {c: float(np.mean([(e["stance_slip"][c]["slip_ratio"] or 0) for e in eps])) for c in CORNERS},
         "stance_slip_p95_mps": {c: max((e["stance_slip"][c]["p95_mps"] or 0) for e in eps) for c in CORNERS},
         "roll_deg_rms": float(np.mean([e["roll_deg"]["rms"] for e in eps])),
         "pitch_deg_rms": float(np.mean([e["pitch_deg"]["rms"] for e in eps])),
         "drives": {}}
    for g in THERMAL:
        s["drives"][g] = {k: float(max(e["drives"][g][k] for e in eps)) for k in eps[0]["drives"][g]}
    flags = []
    for g, v in s["drives"].items():
        if v["rms_over_thermal"] > 1.0:
            flags.append(f"heat: {g} RMS {v['rms_worst_nm']:.2f} N·m over its {THERMAL[g]} N·m thermal rating")
        if v["saturated_fraction"] > 0.01:
            flags.append(f"saturation: {g} at >= 99 % of its limit in {100 * v['saturated_fraction']:.1f} % of steps")
        if v["rpm_over_allowance"] > 1.0:
            flags.append(f"speed: {g} peak {v['peak_rpm']:.0f} rpm over its {RPM_ALLOW[g]:.0f} rpm allowance")
    for c, r in s["slip_ratio_worst"].items():
        if r > SLIP_FLAG:
            flags.append(f"slip: {c} slip ratio {r:.2f} (flag above {SLIP_FLAG})")
    s["flags"] = flags
    return s


def caption(img, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    W, H = im.size
    band = 56
    out = Image.new("RGB", (W, H + band), (20, 22, 26))
    out.paste(im, (0, 0))
    dr = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        f = f2 = ImageFont.load_default()
    dr.text((16, H + 6), "SIMULATION", fill=(255, 196, 0), font=f)
    dr.text((170, H + 10), text, fill=(230, 230, 230), font=f2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--date", default="2026-11-19")
    a = ap.parse_args()
    cdir, out = Path(a.ckpt_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zp, pk = cdir / "trike60_2007040_steps.zip", cdir / "trike60_vecnormalize_2007040_steps.pkl"
    record = json.loads((cdir / "run_record.json").read_text())
    st = record["settings"]
    ec = dict(st["env_config"])

    def make(model_file, **changes):
        cfg = TrikeEnvConfig.from_dict({**ec, "model_path": str(GEN / model_file)}).replace(**changes)
        return TrikeEnv(cfg)

    env = make("trike60_C_closed.xml")
    mine = json.loads(json.dumps(env.cfg.to_dict()))
    diffs = {k: (mine.get(k), v) for k, v in ec.items() if k != "model_path" and mine.get(k) != v}
    assert not diffs, f"environment config differs from the run record: {diffs}"
    stats = load_obs_stats(pk)
    assert stats[0].shape == env.observation_space.shape, (stats[0].shape, env.observation_space.shape)
    policy = build_policy(env, zp, st["net"], st["log_std_init"])
    first = policy.mlp_extractor.policy_net[0].in_features
    assert first == env.observation_space.shape[0], (first, env.observation_space.shape)

    renderer = None
    try:
        renderer = mujoco.Renderer(env.model, 540, 960)
    except Exception as exc:  # no GL: the stills are skipped, the numbers are not affected
        print("no renderer:", exc)

    results = {"policy_zip_sha256": sha256(zp), "vecnormalize_sha256": sha256(pk), "run_commit": None,
               "delay_ms": DELAY, "seeds": SEEDS, "window_from_s": WINDOW_S, "cases": {}}
    results["run_commit"] = record["sessions"][0]["git"]["commit"]
    cases = [("target 0.25 m/s", "trike60_C_closed.xml", {}, 0.25, None),
             ("0.15 m/s", "trike60_C_closed.xml", {}, 0.15, None),
             ("sensitivity: impratio 10, 0.25 m/s", "trike60_C_closed.xml", {}, 0.25, 10.0),
             ("sensitivity: roll-ankle model, 0.25 m/s", "trike60_C_closed_ankleroll.xml", {}, 0.25, None),
             ("sensitivity: nominal model, no randomisation or noise, 0.25 m/s", "trike60_C_closed.xml",
              {"domain_randomization": False, "observation_noise": False}, 0.25, None)]
    frames_all = {}
    for label, mf, changes, cmd, impr in cases:
        e = env if (mf == "trike60_C_closed.xml" and not changes) else make(mf, **changes)
        imp0 = e.model.opt.impratio
        if impr:
            e.model.opt.impratio = impr
        eps = []
        for s in SEEDS:
            still = (6.0, 6.4) if (label.startswith("target") and s == SEEDS[0]) else None
            arr, fr = run_episode(e, policy, stats, s, cmd, still_at=still, renderer=renderer if still else None)
            frames_all.update(fr)
            eps.append(measures(arr, cmd))
        e.model.opt.impratio = imp0
        results["cases"][label] = {"model": mf, "changes": changes, "impratio": impr or float(imp0),
                                   "summary": summarise(eps), "episodes": eps}
        sm = results["cases"][label]["summary"]
        print(f"{label}: speed {sm['forward_speed_mps']:.3f} m/s, falls {sm['falls']}, flags {sm['flags']}")
    base = results["cases"]["target 0.25 m/s"]["summary"]
    sens = results["cases"]["sensitivity: impratio 10, 0.25 m/s"]["summary"]
    results["contact_sensitivity_flag"] = bool(
        sens["falls"] > 0 or sens["forward_speed_mps"] < (1 - CONTACT_SENS_FLAG) * base["forward_speed_mps"])
    (out / f"gait_pipe1_{a.date}.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    for (tt, cam), img in sorted(frames_all.items()):
        im = caption(img, f"learned policy, pipeline test (2 M steps), 0.25 m/s command, 10 ms delay, t = {tt:.1f} s")
        p = out / f"gait_pipe1_{cam}_t{tt:.1f}.jpg"
        im.save(p, quality=85)
        assert p.stat().st_size < 500 * 1024, p
    print("wrote", out)


if __name__ == "__main__":
    main()
