"""HOLD variant of roll_unclip.py: tail_3 command held at a fixed a (ctrl = 0.2094 a, ctrlrange widened as in unclip) every step; the policy's own tail_3 mu is recorded (not applied); the other 14 actuators = clip(mu).

    python -I roll_unclip.py CKPT_ZIP VECNORM_PKL LABEL OUT_PREFIX [HALF_RANGE]

Question: with tail_3_pitch_act given room (ctrlrange widened) and its raw deterministic Gaussian mean passed through
unclipped, where does the mean settle? q3 = 0 needs a = +1.073 on the plant map q3 = -0.0924 + 0.0863 a.

Env: the stance stage of the cf80ecb1 code copy in ./code, arm LTSB (judge_stance5.Ctx("LTSB")); the checkpoint loads
through judge_stance5.load_checkpoint (SB3 load + frozen VecNormalize, r9 plant identity).  Deterministic, seeds
3042..3061, env.reset(seed); window = step indices 200..L-1 (the judge's).

Conditions (same seeds, same env instance class, one fresh env per condition):
  base     : stock env, act = model.predict(frozen-normalised obs, deterministic=True)  (SB3 clip(mu)), as stance_report.
  wrapclip : the wrapper installed (instance overrides of _filter_action / _scale_action) with the tail_3 passthrough
             OFF and the original ctrlrange; act = clip(mu) computed from the policy's own mean.  Must equal base bitwise.
  unclip   : ctrlrange of tail_3_pitch_act widened to +/-HALF_RANGE (from a scratch MJCF copy, compiled and checked to
             differ from the r9 MJCF in that one entry only), tail_3 passthrough ON: act[t3] = raw mu[t3], other 14 =
             clip(mu).  The env's clip in _filter_action and _scale_action is skipped for tail_3 only; the low-pass
             filter runs on the raw value; ctrl[t3] = 0.2094 * filtered (original half-range, home 0), so ctrl = 0.2094*mu
             in steady state.  MuJoCo's ctrllimited clamp then sits at +/-HALF_RANGE.  The env's own reward bookkeeping
             (_policy_action, used by the action-price terms) is left as the env computes it (clipped).
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

HERE = os.path.dirname(os.path.realpath(__file__))
CODE = os.path.join(HERE, "code")
XML_REL = "environments/tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml"
SEED0, N_EP, WIN0 = 3042, 20, 200
T3NAME = "tail_3_pitch_act"
HOLDS = ("0.90", "1.00", "1.05", "1.073", "1.10", "1.20", "1.40", "1.60")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    zip_path, pkl_path, label, out_prefix = sys.argv[1:5]
    half = float(sys.argv[5]) if len(sys.argv) > 5 else 0.40
    spec = importlib.util.spec_from_file_location("judge_stance5", os.path.join(HERE, "judge_stance5.py"))
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge_stance5"] = J
    spec.loader.exec_module(J)
    modules = J.isolate(CODE)
    import mujoco
    import numpy as np
    import torch
    from environments.shared import base_env as SBE

    torch.set_num_threads(1)
    ctx = J.Ctx("LTSB")
    model, normalizer = J.load_checkpoint(ctx, zip_path, pkl_path)
    pol = model.policy
    assert not getattr(pol, "use_sde", False)
    obs_rms = normalizer.obs_rms
    vmean, vvar = np.asarray(obs_rms.mean, dtype=np.float64), np.asarray(obs_rms.var, dtype=np.float64)
    eps, clip_obs = float(normalizer.epsilon), float(normalizer.clip_obs)

    # ---- scratch MJCF: widen ONLY tail_3_pitch_act's ctrlrange; check the compiled diff is that one entry
    orig_xml = os.path.join(CODE, XML_REL)
    src = open(orig_xml).read()
    old = '<position name="tail_3_pitch_act" joint="tail_3_pitch" kp="60" forcerange="-50 50" ctrlrange="-0.2094 0.2094"/>'
    assert src.count(old) == 1
    new = old.replace('ctrlrange="-0.2094 0.2094"', f'ctrlrange="-{half:.4f} {half:.4f}"')
    scratch_xml = f"{out_prefix}_scratch_mjcf.xml"
    with open(scratch_xml, "w") as fh:
        fh.write(src.replace(old, new))
    m0 = mujoco.MjModel.from_xml_path(orig_xml)
    m1 = mujoco.MjModel.from_xml_path(scratch_xml)
    diffs = []
    for name in dir(m0):
        if name.startswith("_"):
            continue
        try:
            a0, a1 = getattr(m0, name), getattr(m1, name)
        except Exception:
            continue
        if isinstance(a0, np.ndarray) and a0.dtype != object:
            if a0.shape != a1.shape or not np.array_equal(a0, a1, equal_nan=True):
                diffs.append(name)
    names = [mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m0.nu)]
    t3 = names.index(T3NAME)
    cr_diff = np.argwhere(m0.actuator_ctrlrange != m1.actuator_ctrlrange)
    mjcf_check = {"orig_xml_sha256": sha(orig_xml), "scratch_xml": scratch_xml, "scratch_xml_sha256": sha(scratch_xml),
                  "differing_model_fields": diffs, "ctrlrange_diff_entries": cr_diff.tolist(),
                  "orig_t3_ctrlrange": m0.actuator_ctrlrange[t3].tolist(),
                  "scratch_t3_ctrlrange": m1.actuator_ctrlrange[t3].tolist(),
                  "t3_joint_range": m0.jnt_range[m0.joint("tail_3_pitch").id].tolist()}
    print("mjcf check", mjcf_check, flush=True)
    assert diffs == ["actuator_ctrlrange"], diffs
    assert {tuple(x) for x in cr_diff.tolist()} <= {(t3, 0), (t3, 1)}

    def make_env(mode):
        env = ctx.env_class(**ctx.env_kwargs)
        u = env.unwrapped
        assert np.array_equal(u.model.actuator_ctrlrange, m0.actuator_ctrlrange)
        assert float(u._home_ctrl[t3]) == 0.0, u._home_ctrl[t3]
        if mode == "base":
            return env
        orig_range = u.model.actuator_ctrlrange.copy()
        passthrough = mode != "base" and mode != "wrapclip"
        if passthrough:
            u.model.actuator_ctrlrange[t3] = m1.actuator_ctrlrange[t3]
            assert np.array_equal(u.model.actuator_ctrlrange, m1.actuator_ctrlrange)

        def _filter_action(self, action):
            a = np.asarray(action, dtype=np.float64)
            # the trex override's bookkeeping, unchanged (clipped command for the reward's price terms)
            self._policy_action = np.clip(a, -1.0, 1.0)
            self._policy_action_step = self._step_count + 1
            # the base class's filter, with tail_3 exempt from the clip in passthrough mode
            clipped = np.clip(a, -1.0, 1.0)
            if passthrough:
                clipped[t3] = a[t3]
            if self._step_count == 0 or self._action_filter_state is None:
                self._action_filter_state = clipped
            else:
                self._action_filter_state = SBE._apply_low_pass(self._action_filter_state, clipped,
                                                                self._action_filter_alpha)
            return self._action_filter_state.copy()

        def _scale_action(self, action):
            residual = np.clip(action, -1.0, 1.0)
            if passthrough:
                residual[t3] = np.asarray(action, dtype=np.float64)[t3]
            ctrl_min, ctrl_max = orig_range[:, 0], orig_range[:, 1]   # the r9 mapping: ctrl[t3] = 0.2094 * a
            below_home = residual * (self._home_ctrl - ctrl_min)
            above_home = residual * (ctrl_max - self._home_ctrl)
            return np.asarray(self._home_ctrl + np.where(residual < 0.0, below_home, above_home))

        u._filter_action = types.MethodType(_filter_action, u)
        u._scale_action = types.MethodType(_scale_action, u)
        return env

    def norm_obs(obs):
        return np.clip((obs - vmean) / np.sqrt(vvar + eps), -clip_obs, clip_obs).astype(np.float32)

    def raw_mean(o):
        with torch.no_grad():
            dist = pol.get_distribution(torch.as_tensor(o[None]))
            return dist.distribution.mean.numpy()[0]

    low_space = high_space = None

    def episode(env, seed, mode):
        nonlocal low_space, high_space
        u = env.unwrapped
        m, d = u.model, u.data
        q3adr = int(m.jnt_qposadr[m.joint("tail_3_pitch").id])
        q3dof = int(m.jnt_dofadr[m.joint("tail_3_pitch").id])
        obs, _ = env.reset(seed=seed)
        rec = {k: [] for k in ("mu", "act", "ctrl", "q3", "force", "jlim", "rew")}
        allcmd = []
        L, ret = 0, 0.0
        term = trunc = False
        info = {}
        mismatch = 0
        while True:
            o = norm_obs(obs)
            mu = raw_mean(o)
            if mode == "base":
                act = np.asarray(model.predict(o, deterministic=True)[0])
                if not np.array_equal(act, np.clip(mu, -1.0, 1.0)):
                    mismatch += 1
            else:
                act = np.clip(mu, -1.0, 1.0)
                if mode != "wrapclip":
                    act = act.astype(np.float64)
                    act[t3] = float(mu[t3]) if mode == "unclip" else float(mode.split("_")[1])
            obs, r, term, trunc, info = env.step(act)
            rec["mu"].append(float(mu[t3]))
            rec["act"].append(float(act[t3]))
            rec["ctrl"].append(float(d.ctrl[t3]))
            rec["q3"].append(float(d.qpos[q3adr]))
            rec["force"].append(float(d.actuator_force[t3]))
            rec["jlim"].append(float(d.qfrc_constraint[q3dof]))
            rec["rew"].append(float(r))
            allcmd.append(np.clip(np.asarray(act, dtype=np.float64), -1.0, 1.0))
            ret += float(r)
            L += 1
            if term or trunc:
                break
        arr = {k: np.asarray(v) for k, v in rec.items()}
        cmd = np.asarray(allcmd)
        w = slice(WIN0, L)
        fr99 = np.mean(np.abs(cmd[w]) >= 0.99, axis=0) if L > WIN0 else np.full(cmd.shape[1], np.nan)
        return arr, {"seed": seed, "length": L, "fell": bool(term) and L < ctx.horizon, "terminated": bool(term),
                     "truncated": bool(trunc), "termination_reason": str(info.get("termination_reason", "")),
                     "return": ret, "window_return": float(arr["rew"][w].sum()),
                     "frac99_all": fr99.tolist(), "base_predict_mismatch_steps": mismatch}

    out = {"label": label, "zip": zip_path, "zip_sha256": sha(zip_path), "vecnorm": pkl_path,
           "vecnorm_sha256": sha(pkl_path), "num_timesteps": int(model.num_timesteps), "modules": modules,
           "env_kwargs": J.jsonable(ctx.env_kwargs), "horizon": ctx.horizon, "actuators": names, "t3_index": t3,
           "half_range": half, "mjcf_check": mjcf_check, "window": [WIN0, "L-1"], "seeds": [SEED0, SEED0 + N_EP - 1],
           "sigma": np.exp(pol.log_std.detach().numpy()).tolist(), "runs": {}}
    npz = {}
    t0 = time.monotonic()
    for mode in [f"hold_{x}" for x in HOLDS]:
        env = make_env(mode)
        rows = []
        for i in range(N_EP):
            arr, row = episode(env, SEED0 + i, mode)
            rows.append(row)
            for k, v in arr.items():
                npz[f"{mode}_{i}_{k}"] = v
        env.close()
        out["runs"][mode] = rows
        f99 = [r["frac99_all"][t3] for r in rows if r["length"] > WIN0]
        print(f"{label} {mode}: falls {sum(r['fell'] for r in rows)} t3frac99(clipped cmd) {np.mean(f99):.4f} "
              f"ret {np.mean([r['return'] for r in rows]):.2f} [{time.monotonic() - t0:.0f}s]", flush=True)
    out["seconds"] = time.monotonic() - t0
    with open(f"{out_prefix}.json", "x") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(f"{out_prefix}.npz", **npz)


if __name__ == "__main__":
    main()
