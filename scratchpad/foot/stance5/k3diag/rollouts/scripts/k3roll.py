"""K3 diagnosis rollouts (REPORT ONLY): re-roll the judge's joint-space noise test with per-step recording, plus
per-actuator-group noise ablations on the same draws.

Run through tools/iso.py with ISO_WT=<the arm's clone> (LTSB: fix/scratch_copband; LTS: fix/screen_s4_LTS), nice -n 19:

    python3 iso.py k3roll.py --arm LTSB --run stance5_LTSB_s42 --steps final --conds full noise0 legs tail neckhead notail \
        --out-dir OUT

The loop is judge_stance{4,5}.noise_matched (space "joint", sigma = frozen_params criteria.H4.control_sigma[seed].vector),
copied line for line: one env per condition, SubstepContactRecorder attached, spawns 3042..3061, rng =
numpy default_rng(episode seed), obs normalised with the checkpoint's VecNormalize stats, mu from
action_net(mlp_extractor.forward_actor(extract_features(obs))), cmd = clip(mu), a = clip(cmd + sigma z), env.step(a),
rec.end_step(a, r).  The judge's own Ctx, load_checkpoint, checkpoint_paths and run_bindings are imported from the judge
file (never edited).  The only differences are read-only: the recorder is built with balance=True and per_geom=True (extra
read-only capture), and the loop reads mjData after each step.  An ablation keeps the same z draw at every step and
multiplies sigma by a 0/1 mask (mask 1 reproduces sigma * z bit for bit).  "noise0" is the mask of zeros (the
deterministic command).  "t3hold0" conditions (counterfactual) set the tail_3 command to 0 before the noise is added.

Verification (condition "full"): every episode's length, termination reason, reward (to 1e-6) and
max_sole_cop_fore_aft / max_sole_tilt_deg must equal the judge file's noise_matched_control_joint.episodes_detail;
the result is printed and stored; on a mismatch the script stops (nothing else is rolled for that checkpoint).

Output: OUT/<run>__<steps>__<cond>.npz (per-step float16 arrays padded with NaN to 1000 steps, episode tables).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
JUDGES = {"LTSB": SCRATCH + "/foot/stance5/judge/judge_stance5.py", "LTS": SCRATCH + "/foot/stance4/judge/judge_stance4.py"}
ROOTS = {"LTSB": SCRATCH + "/foot/stance5", "LTS": SCRATCH + "/foot/stance4"}
SEED0, EPISODES, T = 3042, 20, 1000

ACT = ["neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act", "r_knee_act",
       "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act", "tail_1_pitch_act",
       "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act"]
GROUPS = {
    "full": list(range(15)), "noise0": [],
    "legs": [3, 4, 5, 6, 7, 8, 9, 10], "tail": [11, 12, 13, 14], "neckhead": [0, 1, 2], "notail": list(range(11)),
    "hip_pitch": [3, 7], "hip_roll": [4, 8], "knee": [5, 9], "ankle": [6, 10],
    "legs_no_ankle": [3, 4, 5, 7, 8, 9], "legs_no_hiproll": [3, 5, 6, 7, 9, 10], "legs_no_hippitch": [4, 5, 6, 8, 9, 10],
    "legs_no_knee": [3, 4, 6, 7, 8, 10], "legs_tail": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "legs_neckhead": list(range(11)),
    "t3hold0_full": list(range(15)), "t3hold0_noise0": [],
    # counterfactual hip-pitch modes (hip pitch noise only): the left draw replaced by the right one (common mode,
    # both hips pitch together) or by its negative (differential mode, one forward and one back); same stream otherwise
    "hp_sym": [3, 7], "hp_anti": [3, 7],
    # deterministic probes (no noise): a 0.2 s hip-pitch command offset of +/-0.3 at steps 400-419, differential
    # (right +0.3, left -0.3) or common (both +0.3); the response of the stagger / posture is the recovery readout
    "pulse_anti": [], "pulse_sym": [], "pulse_anti10": [], "pulse_anti20": [],
}
PULSE = {"pulse_anti": (0.3, -0.3), "pulse_sym": (0.3, 0.3), "pulse_anti10": (0.1, -0.1), "pulse_anti20": (0.2, -0.2)}
PULSE_STEPS = (400, 420)
FULL_SET = {"full", "noise0", "t3hold0_full", "t3hold0_noise0", "pulse_anti", "pulse_sym", "pulse_anti10", "pulse_anti20"}  # conditions that also keep the per-actuator arrays
INFO_KEYS = ("tilt_angle", "forward_z", "reward_stance_cop_band", "com_lateral_offset_m", "foot_stagger_m")
CONTACT_CATS = ("tail", "head", "torso", "arms", "thigh", "tibia", "metatarsus")
CORE = (["pitch_deg", "roll_deg", "pelvis_z", "com_fa", "com_lat", "com_vfa", "com_vlat", "xcom_fa", "xcom_lat",
         "toe_line_fa", "heel_line_fa", "head_tip_z", "tail_tip_z", "reward", "t3_mu", "t3_act"]
        + [f"{s}_{k}" for k in ("wcop", "pcop_fa", "pcop_lat", "load", "down", "tilt", "lift", "slip", "ankle_ff",
                                 "toe_fa", "heel_fa", "pitch_s", "roll_s") for s in ("r", "l")]
        + [f"c_{c}" for c in CONTACT_CATS] + [f"i_{k}" for k in INFO_KEYS])
PER_ACT = ("mu", "q", "ff")  # z is regenerable: default_rng(seed).standard_normal(15) per step
LITE = ["pitch_deg", "roll_deg", "pelvis_z", "com_fa", "com_lat", "com_vfa", "xcom_fa", "xcom_lat", "toe_line_fa",
        "heel_line_fa", "r_wcop", "l_wcop", "r_pcop_fa", "l_pcop_fa", "r_load", "l_load", "r_ankle_ff", "l_ankle_ff",
        "t3_mu", "c_tail", "c_head", "c_tibia", "r_down", "l_down", "r_slip", "l_slip"]


def load_judge(arm):
    name = "judge_stance5" if arm == "LTSB" else "judge_stance4"
    spec = importlib.util.spec_from_file_location(name, JUDGES[arm])
    J = importlib.util.module_from_spec(spec)
    sys.modules[name] = J
    spec.loader.exec_module(J)
    return J


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def category(name: str) -> str | None:
    if name.startswith("tail_"):
        return "tail"
    if name in ("skull_upper", "snout", "head_bite"):
        return "head"
    if name in ("torso", "ribcage", "belly"):
        return "torso"
    if "arm" in name or "forearm" in name or name.endswith("_hand"):
        return "arms"
    for c in ("thigh", "tibia", "metatarsus"):
        if f"_{c}_" in name:
            return c
    return None  # plantar pads and digits: the feet


def quat_frames(q):
    """(T,4) wxyz -> forward (T,2) heading unit vector, pitch_deg (+ nose up), roll_deg (+ left side up)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    f = np.stack([r00, r10], 1)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f, np.degrees(np.arcsin(np.clip(r20, -1, 1))), np.degrees(np.arcsin(np.clip(r21, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("LTSB", "LTS"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", nargs="+", required=True)
    ap.add_argument("--conds", nargs="+", required=True, choices=sorted(GROUPS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    a = ap.parse_args()
    for c in a.conds:
        assert c in GROUPS
    J = load_judge(a.arm)
    import mujoco
    import torch
    from stable_baselines3.common.policies import BaseModel

    torch.set_num_threads(1)
    spec = J.ARMS[a.arm]
    clone = spec["clone"]
    modules = J.isolate(clone)
    git = J.git_state(clone)
    assert git["head"] == spec["head"] and not git["status"], git
    ctx = J.Ctx(a.arm)
    from environments.shared.gait.morphology import Morphology
    from environments.shared.gait.recorder import SubstepContactRecorder
    from environments.shared.gait.stance_metrics import episode_stance_metrics, whole_foot_cop
    from environments.shared.task_fingerprint import stage_task_fingerprint

    panel_task = stage_task_fingerprint(ctx.mid, ctx.stage, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity)["task_sha256"]
    seed = int(a.run.rsplit("_s", 1)[1])
    stage_dir = os.path.join(ROOTS[a.arm], "runs", a.run, "01_stance")
    with open(J.FROZEN_PARAMS, "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == J.FROZEN_PARAMS_SHA256
    frozen = J.read_json(J.FROZEN_PARAMS)
    ctrl_sigma = np.asarray(frozen["criteria"]["H4"]["control_sigma"][str(seed)]["vector"], dtype=np.float64)
    os.makedirs(a.out_dir, exist_ok=True)

    for st in a.steps:
        steps = J.parse_steps(st)
        _, zp, vp = J.checkpoint_paths(os.path.join(stage_dir, "models"), steps)
        ns = types.SimpleNamespace(seed=seed, steps=steps, no_symmetry_check=False, task_binding="require",
                                   allow_missing_records=False, allow_seed_mismatch=False, allow_commit_mismatch=False)
        checks = J.run_bindings(ctx, ns, stage_dir, zp, panel_task, git)
        if checks["hard_failures"]:
            raise SystemExit("binding refused: " + " | ".join(checks["hard_failures"]))
        judge_file = os.path.join(ROOTS[a.arm], "judged", a.run, f"judge_{st}.json")
        jd = J.read_json(judge_file)
        jn = jd["noise_matched_control_joint"]
        assert jd["model_sha256"] == sha256_file(zp) and jd["vecnorm_sha256"] == sha256_file(vp), "judged file is another checkpoint"
        assert np.array_equal(np.asarray(jn["sigma_vector"]), ctrl_sigma), "sigma differs from the judge file's"
        model, normalizer = J.load_checkpoint(ctx, zp, vp)
        pol = model.policy
        own = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64))
        vmean, vvar = np.asarray(normalizer.obs_rms.mean), np.asarray(normalizer.obs_rms.var)
        eps, clip_obs = normalizer.epsilon, normalizer.clip_obs
        verified = None
        for cond in a.conds:
            mask = np.zeros(15)
            mask[GROUPS[cond]] = 1.0
            sig_eff = ctrl_sigma * mask
            t3hold = cond.startswith("t3hold0")
            t0 = time.monotonic()
            env = ctx.env_class(**ctx.env_kwargs)
            u = env.unwrapped
            m = u.model
            low, high = env.action_space.low.astype(np.float64), env.action_space.high.astype(np.float64)
            act_qadr = np.array([int(m.jnt_qposadr[int(m.actuator_trnid[i, 0])]) for i in range(m.nu)])
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
            assert names == ACT, names
            flim = np.abs(m.actuator_forcerange[:, 1]).astype(np.float64)
            mtp3 = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_toe_d3_joint") for s in ("r", "l")])
            head_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "head_tip")
            tail_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tail_tip")
            morph = Morphology.from_env(env, ctx.mid)
            assert [f.label for f in morph.feet] == ["r", "l"]
            rec = SubstepContactRecorder(env, morph, per_geom=True, balance=True).attach()
            E = a.episodes
            core = np.full((E, T, len(CORE)), np.nan, dtype=np.float32)
            pa = np.full((E, T, len(PER_ACT), 15), np.nan, dtype=np.float32) if (cond in FULL_SET and cond not in ("pulse_anti", "pulse_sym", "pulse_anti10")) else None
            ep = {"seed": [], "length": [], "reward": [], "term": [], "max_sole_cop": [], "max_sole_tilt": [],
                  "max_foot_cop": []}
            try:
                for i in range(E):
                    s = SEED0 + i
                    rng = np.random.default_rng(s)
                    obs, _ = env.reset(seed=s)
                    nsteps, reward = 0, 0.0
                    zs, mus, acts, qs, anch, tips, infos = [], [], [], [], [], [], []
                    while True:
                        o = np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)
                        with torch.no_grad():
                            f = BaseModel.extract_features(pol, torch.as_tensor(o[None]), pol.pi_features_extractor)
                            mu = pol.action_net(pol.mlp_extractor.forward_actor(f)).numpy()[0].astype(np.float64)
                        cmd = np.clip(mu, low, high)
                        if t3hold:
                            cmd[14] = 0.0
                        z = rng.standard_normal(cmd.shape)
                        if cond == "hp_sym":
                            z[7] = z[3]
                        elif cond == "hp_anti":
                            z[7] = -z[3]
                        act = np.clip(cmd + sig_eff * z, low, high)
                        if cond in PULSE and PULSE_STEPS[0] <= nsteps < PULSE_STEPS[1]:
                            act[3] = np.clip(act[3] + PULSE[cond][0], low[3], high[3])
                            act[7] = np.clip(act[7] + PULSE[cond][1], low[7], high[7])
                        obs, r, term, trunc, info = env.step(act)
                        rec.end_step(act, float(r))
                        reward += float(r)
                        nsteps += 1
                        d = u.data
                        zs.append(z), mus.append(mu), acts.append(act), qs.append(d.qpos[act_qadr].copy())
                        anch.append(np.asarray(d.xanchor[mtp3][:, :2], dtype=np.float64).copy())
                        tips.append((float(d.site_xpos[head_site][2]), float(d.site_xpos[tail_site][2])))
                        infos.append([float(info.get(k, np.nan)) for k in INFO_KEYS])
                        if term or trunc:
                            tr = rec.end_episode(terminated=bool(term), truncated=bool(trunc),
                                                 termination_reason=str(info.get("termination_reason", "")))
                            break
                    met = episode_stance_metrics(tr, settle_steps=ctx.settle)
                    L = nsteps
                    assert int(tr.length) == L
                    ep["seed"].append(s), ep["length"].append(L), ep["reward"].append(reward)
                    ep["term"].append(str(getattr(tr, "termination_reason", "")))
                    ep["max_sole_cop"].append(J.finite_or_none(met.max_sole_cop_fore_aft))
                    ep["max_sole_tilt"].append(J.finite_or_none(met.max_sole_tilt_deg))
                    ep["max_foot_cop"].append(J.finite_or_none(getattr(met, "max_foot_cop_fore_aft", None)))
                    # ---- per-step derived arrays (float64 here, stored float32/16)
                    fwd, pitch, roll = quat_frames(np.asarray(tr.root_quat, dtype=np.float64))
                    left = np.stack([-fwd[:, 1], fwd[:, 0]], 1)
                    an = np.asarray(anch)                                   # (L,2,2)
                    amean = an.mean(axis=1)
                    com = np.asarray(tr.com_pos, dtype=np.float64)
                    rel = com[:, :2] - amean
                    com_fa = np.einsum("ij,ij->i", fwd, rel)
                    com_lat = np.einsum("ij,ij->i", left, rel)
                    dt = ctx.control_dt
                    vel = np.vstack([np.zeros((1, 3)), np.diff(com, axis=0) / dt])
                    vfa = np.einsum("ij,ij->i", fwd, vel[:, :2])
                    vlat = np.einsum("ij,ij->i", left, vel[:, :2])
                    w0 = np.sqrt(9.81 / np.maximum(com[:, 2], 0.1))
                    xfa, xlat = com_fa + vfa / w0, com_lat + vlat / w0
                    slp = np.asarray(tr.support_line_pos, dtype=np.float64)  # (L,F,2,3) heel, toe
                    heel_fa = np.einsum("ij,ikj->ik", fwd, slp[:, :, 0, :2] - amean[:, None, :])
                    toe_fa = np.einsum("ij,ikj->ik", fwd, slp[:, :, 1, :2] - amean[:, None, :])
                    _, wcop, loaded = whole_foot_cop(tr)
                    scop = np.asarray(tr.sole_cop, dtype=np.float64)
                    load = np.asarray(tr.leg_floor_mean, dtype=np.float64) / float(tr.body_weight_n)
                    fp = np.asarray(tr.foot_pos, dtype=np.float64)
                    slip = np.vstack([np.zeros((1, 2)), np.linalg.norm(np.diff(fp[:, :, :2], axis=0), axis=2) / dt])
                    af = np.asarray(tr.actuator_force, dtype=np.float64)
                    ff = af / flim
                    gnames = list(tr.geom_floor_names)
                    gfm = np.asarray(tr.geom_floor_mean, dtype=np.float64) if tr.geom_floor_mean is not None else np.zeros((L, 0))
                    cats = np.zeros((L, len(CONTACT_CATS)))
                    for gi, gname in enumerate(gnames):
                        c = category(gname)
                        if c is not None:
                            cats[:, CONTACT_CATS.index(c)] += gfm[:, gi]
                    mu_a, act_a = np.asarray(mus), np.asarray(acts)
                    cols = {"pitch_deg": pitch, "roll_deg": roll, "pelvis_z": np.asarray(tr.root_pos)[:, 2],
                            "com_fa": com_fa, "com_lat": com_lat, "com_vfa": vfa, "com_vlat": vlat,
                            "xcom_fa": xfa, "xcom_lat": xlat, "toe_line_fa": np.nanmax(toe_fa, axis=1),
                            "heel_line_fa": np.nanmin(heel_fa, axis=1),
                            "head_tip_z": np.asarray(tips)[:, 0], "tail_tip_z": np.asarray(tips)[:, 1],
                            "reward": np.asarray(tr.reward, dtype=np.float64), "t3_mu": mu_a[:, 14], "t3_act": act_a[:, 14]}
                    per_foot = {"wcop": np.where(loaded, wcop, np.nan), "pcop_fa": scop[:, :, 0], "pcop_lat": scop[:, :, 1],
                                "load": load, "down": np.asarray(tr.leg_down_frac, dtype=np.float64),
                                "tilt": np.asarray(tr.sole_tilt_deg, dtype=np.float64),
                                "lift": np.asarray(tr.sole_corner_lift, dtype=np.float64), "slip": slip,
                                "ankle_ff": ff[:, [6, 10]], "toe_fa": toe_fa, "heel_fa": heel_fa,
                                "pitch_s": np.asarray(tr.sole_pitch_deg, dtype=np.float64),
                                "roll_s": np.asarray(tr.sole_roll_deg, dtype=np.float64)}
                    for k, v in per_foot.items():
                        cols[f"r_{k}"], cols[f"l_{k}"] = v[:, 0], v[:, 1]
                    for ci, c in enumerate(CONTACT_CATS):
                        cols[f"c_{c}"] = cats[:, ci]
                    inf = np.asarray(infos)
                    for ki, k in enumerate(INFO_KEYS):
                        cols[f"i_{k}"] = inf[:, ki]
                    core[i, :L] = np.stack([cols[k] for k in CORE], axis=1)
                    if pa is not None:
                        pa[i, :L, 0] = mu_a
                        pa[i, :L, 1] = np.asarray(qs)
                        pa[i, :L, 2] = ff
            finally:
                rec.detach()
                env.close()
            secs = time.monotonic() - t0
            lengths = np.asarray(ep["length"])
            row = {"run": a.run, "steps": st, "cond": cond, "falls": int((lengths < ctx.horizon).sum()),
                   "fall_seeds": [int(x) for x, L in zip(ep["seed"], lengths) if L < ctx.horizon], "seconds": round(secs, 1)}
            if cond == "full":
                jl = [(r["seed"], r["length"], r["termination_reason"]) for r in jn["episodes_detail"]]
                ml = list(zip(ep["seed"], ep["length"], ep["term"]))
                rdiff = max(abs(r["reward"] - x) for r, x in zip(jn["episodes_detail"], ep["reward"]))
                cop_eq = all((r["max_sole_cop_fore_aft"] == x) for r, x in zip(jn["episodes_detail"], ep["max_sole_cop"]))
                tilt_eq = all((r["max_sole_tilt_deg"] == x) for r, x in zip(jn["episodes_detail"], ep["max_sole_tilt"]))
                verified = {"lengths_terms_equal": jl == ml, "reward_max_abs_diff": rdiff, "max_sole_cop_equal": cop_eq,
                            "max_sole_tilt_equal": tilt_eq, "judge_falls": jn["falls"], "judge_fall_seeds": jn["fall_seeds"],
                            "judge_file": judge_file}
                row["verified"] = verified
                ok = jl == ml and rdiff < 1e-6 and cop_eq and tilt_eq
                if not ok:
                    print(json.dumps(row, default=str), flush=True)
                    raise SystemExit(f"REPRODUCTION FAILED for {a.run} {st}: {verified}")
            meta = {"run": a.run, "arm": a.arm, "steps": st, "cond": cond, "mask": mask.tolist(), "t3hold0": t3hold,
                    "sigma": ctrl_sigma.tolist(), "own_sigma": own.tolist(), "seed": seed, "model": zp, "vecnorm": vp,
                    "model_sha256": jd["model_sha256"], "vecnorm_sha256": jd["vecnorm_sha256"], "git": git,
                    "modules_environments": modules["environments"], "task_sha256_rolled": panel_task,
                    "bindings_hard_failures": checks["hard_failures"], "verified": verified, "control_dt": ctx.control_dt,
                    "settle": ctx.settle, "horizon": ctx.horizon, "core_cols": CORE if cond in FULL_SET else LITE, "per_act": PER_ACT, "act_names": ACT,
                    "contact_cats": CONTACT_CATS, "info_keys": INFO_KEYS, "body_weight_n": float(tr.body_weight_n),
                    "force_limits": flim.tolist(), "seconds": secs, "geom_floor_names": gnames,
                    "recorder": "SubstepContactRecorder(per_geom=True, balance=True)", "tool": os.path.realpath(__file__)}
            out = os.path.join(a.out_dir, f"{a.run}__{st}__{cond}.npz")
            cols_kept = CORE if cond in FULL_SET else LITE
            arrays = {"core": core[:, :, [CORE.index(k) for k in cols_kept]].astype(np.float16), "seed": np.asarray(ep["seed"]), "length": lengths,
                      "reward": np.asarray(ep["reward"]), "term": np.asarray(ep["term"]),
                      "max_sole_cop": np.asarray([np.nan if x is None else x for x in ep["max_sole_cop"]]),
                      "max_foot_cop": np.asarray([np.nan if x is None else x for x in ep["max_foot_cop"]]),
                      "max_sole_tilt": np.asarray([np.nan if x is None else x for x in ep["max_sole_tilt"]]),
                      "meta_json": np.array(json.dumps(J.jsonable(meta)))}
            if pa is not None:
                arrays["per_act"] = pa.astype(np.float16)
            np.savez_compressed(out, **arrays)
            row["MB"] = round(os.path.getsize(out) / 1e6, 2)
            print(json.dumps(row, default=str), flush=True)
        del model, normalizer
    J.modules_under(clone)


if __name__ == "__main__":
    main()
