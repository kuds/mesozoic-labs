"""TRACK TAIL (scratch): roll stance checkpoints like SP/foot/stance4/satdiag/rollout/roll.py and reduce the tail habit.

    S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
    T=$S/foot/stance5/tailexp
    env ISO_WT=$S/fix/scratch_copband nice -n 19 python3 $S/foot/stance3/tools/iso.py $T/measure.py \
        --models-dir DIR --label LABEL --steps STEPS [STEPS ...] [--k3] [--out-dir $T/measure]

Run under the band clone fix/scratch_copband at its pinned HEAD (clean; judge_stance5.py's arm LTSB): the environment
the arms train on is that clone's (scratch_tailfix changes only mirror_kl_ppo.py), and the checkpoints the arms save
are stock PPO zips.  Everything the judge does is reused from judge_stance5.py itself (imported from its path, never
edited): Ctx("LTSB") (the clone's stance stage and [env] as loaded, StanceV2Thresholds), checkpoint_paths (the stage1_*
names; "final" = stage1_final.zip), load_checkpoint (policy_loading.load_sb3_checkpoint: the frozen VecNormalize
statistics, plant identity bound, allow_unnormalized False), the floor-truth trace capture, and for --k3 noise_matched
(the H4 tests, arithmetic unchanged: joint space under frozen_params criteria.H4.control_sigma["42"], and the own
sigma in policy space; 20 spawns 3042-3061).  The rollout is stance_report.run_panel(floor_truth=True) at heading 0 on
seeds 3042-3061 (20 deterministic episodes), the judge's v2 panel path.  predict is stance_report._load_policy's
(normalise with the frozen statistics, model.predict(deterministic=True), SB3's clip to [-1, 1]); the raw Gaussian mean
(before the clip and the env's low-pass) is computed beside it from the same normalised observation (roll.py's method)
and never reaches env.step.

Per checkpoint, over each episode's window (recorder index steps 200 .. L-1, the judge's), from the recorder's action
(the clipped command the env received), averaged over the episodes that reach the window:
  * per actuator: the share of window steps at |cmd| >= 0.9 (frac90) and >= 0.99 (frac99, the gate's
    max_actuator_saturation_fraction statistic, reproduced to 1e-12 per episode), the mean command, the mean raw mean
    and the share of window steps with |raw mean| > 1;
  * tail_3_pitch_act: the raw mean's window mean, its mean on pinned steps (|cmd| >= 0.99), its max, and per episode;
  * falls (episodes shorter than 1000 steps), the mean episode reward, terminations;
  * the panel-style clean count: stance_gate_v2.classify_stance_episode on each episode against the statue reference of
    the judge's full panel (40 zero-action episodes, seeds 3042-3081, rolled once here and checked against the judged
    file's reference when one is given), i.e. the judge's per-episode clean flag for seeds 3042-3061; the flagged count
    (gate saturation > 0.10) and the per-rail failure counts;
  * the policy std exp(log_std), all 15 and tail_3's.
Writes OUT/<label>__<steps>.npz (per-episode arrays) and OUT/<label>__summary_<steps...>.json (never overwritten), and
tailexp/measure/statue_ref_3042_3081.json (once, wherever OUT is).  Report only.
"""
from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
JUDGE = SCRATCH + "/foot/stance5/judge/judge_stance5.py"
TAILEXP = SCRATCH + "/foot/stance5/tailexp"
SEED0, EPISODES = 3042, 20
STATUE_SEED, STATUE_EPISODES = 3042, 40  # the judge's full panel: its statue reference divides the share ratio
TAIL3_NAME = "tail_3_pitch_act"
RUN_SEED = 42

RAW: list = []   # raw Gaussian mean per predict call
PRED: list = []  # the command predict returned


def load_judge():
    spec = importlib.util.spec_from_file_location("judge_stance5", JUDGE)
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge_stance5"] = J
    spec.loader.exec_module(J)
    return J


def actuator_names(ctx) -> list[str]:
    import mujoco

    env = ctx.env_class(**ctx.env_kwargs)
    try:
        m = env.unwrapped.model
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    finally:
        env.close()


def statue_ref(J, ctx, out_dir: str, git: dict, judged_ref: str | None) -> dict:
    """The zero-action panel's StatueReference on the judge's full-panel seeds (cached in out_dir)."""
    from environments.shared.curriculum.stance_gate_v2 import statue_reference

    path = os.path.join(TAILEXP, "measure", f"statue_ref_{STATUE_SEED}_{STATUE_SEED + STATUE_EPISODES - 1}.json")
    if os.path.exists(path):
        with open(path) as fh:
            rec = json.load(fh)
        if rec["git_head"] != git["head"] or rec["env_kwargs"] != J.jsonable(ctx.env_kwargs):
            raise SystemExit(f"{path} was rolled under another clone or [env]; move it away to re-roll")
        return rec
    t0 = time.monotonic()
    J.TRACES.clear()
    res = ctx.SR.run_panel(ctx.mid, predict=ctx.SR._zero_action_predict(ctx.env_class, ctx.env_kwargs),
                           episodes=STATUE_EPISODES, seed=STATUE_SEED, settle_steps=ctx.settle, horizon=ctx.horizon,
                           env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity, control_dt=ctx.control_dt,
                           floor_truth=True, spawn_yaw=None)
    J.TRACES.clear()
    ref = statue_reference(res["stance_metrics"], horizon=ctx.horizon).as_dict()
    rec = {"reference": J.jsonable(ref), "seeds": [STATUE_SEED, STATUE_SEED + STATUE_EPISODES - 1],
           "git_head": git["head"], "env_kwargs": J.jsonable(ctx.env_kwargs), "seconds": round(time.monotonic() - t0, 1)}
    if judged_ref:
        with open(judged_ref) as fh:
            jref = json.load(fh)["panel"]["statue_on_same_seeds"]["reference"]
        rec["judged_file"] = judged_ref
        rec["judged_reference"] = jref
        rec["matches_judged"] = (jref["n_episodes"] == ref["n_episodes"] and jref["n_full_horizon"] == ref["n_full_horizon"]
                                 and abs(jref["mean_reward"] - ref["mean_reward"]) < 1e-9
                                 and all(abs(a - b) < 1e-12 for a, b in zip(jref["foot_load_share"], ref["foot_load_share"], strict=True)))
        if not rec["matches_judged"]:
            raise SystemExit(f"the statue reference {ref} differs from the judged file's {jref}")
    try:
        with open(path, "x") as fh:
            json.dump(rec, fh, indent=1)
    except FileExistsError:  # a concurrent measure.py wrote it first: it must agree
        with open(path) as fh:
            other = json.load(fh)
        if other["reference"] != rec["reference"]:
            raise SystemExit(f"{path} (concurrent) differs from this roll: {other['reference']} vs {rec['reference']}")
    return rec


def reduce(J, ctx, res, names, thresholds, statue, pol) -> tuple[dict, dict]:
    import numpy as np

    from environments.shared.curriculum.stance_gate_v2 import classify_stance_episode, reason_key

    traces = [tr for tr, _ in J.TRACES]
    sm = res["stance_metrics"]
    assert len(traces) == len(sm) == EPISODES, (len(traces), len(sm))
    nu = len(names)
    t3 = names.index(TAIL3_NAME)
    bar = float(thresholds.declared().get("max_actuator_saturation_fraction", 0.10))
    n = len(traces)
    ep = {k: np.full((n, nu), np.nan) for k in ("frac90", "frac99", "mean_cmd", "mean_raw", "raw_gt1")}
    ep.update({"length": np.zeros(n, int), "reward": np.zeros(n), "gate_max_sat": np.zeros(n), "clean": np.zeros(n, bool),
               "t3_raw_window_mean": np.full(n, np.nan), "t3_raw_pinned_mean": np.full(n, np.nan),
               "t3_raw_max": np.full(n, np.nan), "t3_pinned_steps": np.zeros(n, int)})
    terminations, rails, off = {}, {}, 0
    for e, (tr, m) in enumerate(zip(traces, sm, strict=True)):
        L = int(tr.length)
        act = np.asarray(tr.action, dtype=np.float64)
        raw = np.asarray(RAW[off:off + L], dtype=np.float64)
        pred = np.asarray(PRED[off:off + L], dtype=np.float64)
        off += L
        assert np.max(np.abs(act - pred)) < 1e-6, np.max(np.abs(act - pred))
        assert np.max(np.abs(np.clip(raw, -1, 1) - pred)) < 1e-6, np.max(np.abs(np.clip(raw, -1, 1) - pred))
        ep["length"][e] = L
        ep["reward"][e] = float(res["rewards"][e])
        ep["gate_max_sat"][e] = float(m.max_actuator_saturation_fraction)
        reasons = classify_stance_episode(m, thresholds, horizon=ctx.horizon, statue=statue)
        ep["clean"][e] = not reasons
        for r in {reason_key(x) for x in reasons}:
            rails[r] = rails.get(r, 0) + 1
        t = str(getattr(tr, "termination_reason", ""))
        terminations[t] = terminations.get(t, 0) + 1
        if L <= ctx.settle:
            continue
        w, rw = act[ctx.settle:L], raw[ctx.settle:L]
        f99 = (np.abs(w) >= 0.99).mean(axis=0)
        assert abs(float(f99.max()) - ep["gate_max_sat"][e]) < 1e-12, (float(f99.max()), ep["gate_max_sat"][e])
        ep["frac99"][e] = f99
        ep["frac90"][e] = (np.abs(w) >= 0.9).mean(axis=0)
        ep["mean_cmd"][e] = w.mean(axis=0)
        ep["mean_raw"][e] = rw.mean(axis=0)
        ep["raw_gt1"][e] = (np.abs(rw) > 1.0).mean(axis=0)
        pinned = np.abs(w[:, t3]) >= 0.99
        ep["t3_raw_window_mean"][e] = float(rw[:, t3].mean())
        ep["t3_raw_max"][e] = float(rw[:, t3].max())
        ep["t3_pinned_steps"][e] = int(pinned.sum())
        if pinned.any():
            ep["t3_raw_pinned_mean"][e] = float(rw[pinned, t3].mean())
    assert off == len(RAW) == len(PRED), (off, len(RAW), len(PRED))
    std = np.exp(pol.log_std.detach().cpu().numpy().astype(np.float64))
    inw = ~np.isnan(ep["frac90"][:, 0])

    def col(key):
        return ep[key][inw].mean(axis=0)

    per = [{"actuator": names[i], "frac90": float(col("frac90")[i]), "frac99": float(col("frac99")[i]),
            "mean_command": float(col("mean_cmd")[i]), "mean_raw": float(col("mean_raw")[i]),
            "raw_beyond1": float(col("raw_gt1")[i]), "episodes_frac90_ge_0.5": int(np.sum(ep["frac90"][inw, i] >= 0.5)),
            "policy_std": float(std[i])} for i in range(nu)]
    top = max(per, key=lambda r: r["frac90"])
    pinned_all = ep["t3_pinned_steps"].sum()
    t3_pinned_mean = (float(np.nansum(ep["t3_raw_pinned_mean"] * ep["t3_pinned_steps"]) / pinned_all) if pinned_all else None)
    falls = [SEED0 + i for i in range(n) if ep["length"][i] < ctx.horizon]
    summary = {
        "episodes": n, "episodes_in_window": int(inw.sum()), "falls": len(falls), "fall_seeds": falls,
        "fall_lengths": [int(ep["length"][i]) for i in range(n) if ep["length"][i] < ctx.horizon],
        "reward_mean": float(ep["reward"].mean()), "reward_mean_full_horizon": float(ep["reward"][ep["length"] >= ctx.horizon].mean())
        if (ep["length"] >= ctx.horizon).any() else None,
        "terminations": terminations,
        "clean": int(ep["clean"].sum()), "clean_of": n, "clean_seeds": [SEED0 + i for i in range(n) if ep["clean"][i]],
        "failing_rails": rails, "flagged_over_bar": int(np.sum(ep["gate_max_sat"] > bar)), "bar": bar,
        "tail3": {"frac90": per[t3]["frac90"], "frac99": per[t3]["frac99"], "mean_command": per[t3]["mean_command"],
                  "raw_mean_window": float(np.nanmean(ep["t3_raw_window_mean"])),
                  "raw_mean_on_pinned_steps": t3_pinned_mean, "pinned_steps": int(pinned_all),
                  "raw_max": float(np.nanmax(ep["t3_raw_max"])), "raw_beyond1": per[t3]["raw_beyond1"],
                  "policy_std": float(std[t3]), "log_std": float(np.log(std[t3])),
                  "raw_window_mean_per_episode": [None if math.isnan(x) else float(x) for x in ep["t3_raw_window_mean"]]},
        "named": None if top["frac90"] == 0.0 else {"actuator": top["actuator"], "frac90": top["frac90"],
                                                   "sign": int(np.sign(top["mean_command"]))},
        "policy_std_all": std.tolist(),
        "per_actuator": per,
    }
    ep["policy_std"] = std
    return summary, ep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models-dir", required=True, help="a directory of stage1_<N>_steps.zip / stage1_vecnormalize_<N>_steps.pkl")
    ap.add_argument("--label", required=True, help="names the outputs (e.g. start, control, p1)")
    ap.add_argument("--steps", nargs="+", required=True, help="checkpoint steps, or final")
    ap.add_argument("--out-dir", default=os.path.join(TAILEXP, "measure"))
    ap.add_argument("--k3", action="store_true", help="also the judge's two H4 noise tests (falls; 40 more episodes)")
    ap.add_argument("--judged-ref", default=SCRATCH + "/foot/stance5/judged/stance5_LTSB_s42/judge_4194304.json",
                    help="a judged file whose panel statue reference the rolled one must equal ('' to skip)")
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out_dir)
    if not out_dir.startswith(TAILEXP + "/"):
        raise SystemExit(f"outputs go under {TAILEXP}")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, f"{a.label}__summary_{'_'.join(a.steps)}.json")
    if os.path.exists(summary_path):
        raise SystemExit(f"refusing to overwrite {summary_path}")
    J = load_judge()
    spec = J.ARMS["LTSB"]
    clone = spec["clone"]
    if os.path.realpath(os.environ.get("ISO_WT", clone)) != os.path.realpath(clone):
        raise SystemExit(f"ISO_WT {os.environ.get('ISO_WT')} is not arm LTSB's clone {clone}")
    modules = J.isolate(clone)
    git = J.git_state(clone)
    if git["status"] or git["head"] != spec["head"]:
        raise SystemExit(f"measure.py runs the clean clone at {spec['head']} (got {git['head']}, dirty={bool(git['status'])})")
    import numpy as np
    import torch

    torch.set_num_threads(1)
    J.install_trace_capture()
    ctx = J.Ctx("LTSB")
    names = actuator_names(ctx)
    from environments.shared.curriculum.stance_gate_v2 import StatueReference
    from environments.shared.task_fingerprint import stage_task_fingerprint

    panel_task = stage_task_fingerprint(ctx.mid, ctx.stage, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity)["task_sha256"]
    sref = statue_ref(J, ctx, out_dir, git, a.judged_ref or None)
    statue = StatueReference.from_dict(sref["reference"])
    ctrl = None
    if a.k3:
        with open(J.FROZEN_PARAMS, "rb") as fh:
            import hashlib

            if hashlib.sha256(fh.read()).hexdigest() != J.FROZEN_PARAMS_SHA256:
                raise SystemExit("frozen_params.json is not the pinned one")
        ctrl = J.read_json(J.FROZEN_PARAMS)["criteria"]["H4"]["control_sigma"][str(RUN_SEED)]
    record = {"tool": os.path.realpath(__file__), "tool_sha256": J.sha256_file(os.path.realpath(__file__)),
              "judge": JUDGE, "judge_sha256": J.sha256_file(JUDGE), "report_only": True, "label": a.label,
              "models_dir": os.path.abspath(a.models_dir), "clone": clone, "git": git, "modules": modules,
              "task_sha256_rolled": panel_task, "seeds": [SEED0, SEED0 + EPISODES - 1], "window": [ctx.settle, ctx.horizon],
              "actuators": names, "statue_reference": sref, "checkpoints": {}}
    for st in a.steps:
        steps = J.parse_steps(st)
        t0 = time.monotonic()
        stem, zp, vp = J.checkpoint_paths(os.path.abspath(a.models_dir), steps)
        if not (os.path.exists(zp) and os.path.exists(vp)):
            record["checkpoints"][str(st)] = {"present": False}
            print(json.dumps({"steps": st, "present": False}), flush=True)
            continue
        model, normalizer = J.load_checkpoint(ctx, zp, vp)
        fp = getattr(model, "mesozoic_task_fingerprint", None) or {}
        if fp.get("task_sha256") != panel_task:
            raise SystemExit(f"{zp} records task {fp.get('task_sha256')}, the panel rolls {panel_task}")
        pol = model.policy
        mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float64)
        var = np.asarray(normalizer.obs_rms.var, dtype=np.float64)
        eps, clip = float(normalizer.epsilon), float(normalizer.clip_obs)

        def predict(obs):  # stance_report._load_policy's predict, plus the raw mean beside it
            normalized = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            act = np.asarray(model.predict(normalized, deterministic=True)[0])
            with torch.no_grad():
                mu = pol.get_distribution(torch.as_tensor(normalized[None])).distribution.mean.cpu().numpy()[0]
            RAW.append(np.asarray(mu, dtype=np.float64))
            PRED.append(np.asarray(act, dtype=np.float64).ravel())
            return act

        RAW.clear(), PRED.clear(), J.TRACES.clear()
        res = ctx.SR.run_panel(ctx.mid, predict=predict, episodes=EPISODES, seed=SEED0, settle_steps=ctx.settle,
                               horizon=ctx.horizon, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity,
                               control_dt=ctx.control_dt, floor_truth=True, spawn_yaw=None)
        summary, ep = reduce(J, ctx, res, names, ctx.thresholds, statue, pol)
        summary.update({"checkpoint": stem, "model": zp, "model_sha256": J.sha256_file(zp), "vecnorm_sha256": J.sha256_file(vp),
                        "num_timesteps": int(model.num_timesteps),
                        "mesozoic_symmetry": getattr(model, "mesozoic_symmetry", None),
                        "mesozoic_mean_bound": getattr(model, "mesozoic_mean_bound", None)})
        if a.k3:
            k3 = {}
            for key, kwargs in (("noise_matched_control_joint", dict(sigma_arg="control", sigma=ctrl["vector"],
                                                                     sigma_source=ctrl["source"], space="joint")),
                                ("noise_matched_own", dict(sigma_arg="own", sigma=None, sigma_source="own", space="policy"))):
                nm = J.noise_matched(ctx, model, normalizer, **kwargs)
                k3[key] = {"falls": nm["falls"], "fall_seeds": nm["fall_seeds"], "fall_lengths": nm["fall_lengths"],
                           "sigma_mean": nm["sigma_mean"], "seconds": round(nm["panel_seconds"], 1)}
            J.TRACES.clear()
            summary["k3"] = k3
        summary["seconds"] = round(time.monotonic() - t0, 1)
        np.savez_compressed(os.path.join(out_dir, f"{a.label}__{st}.npz"), actuators=np.array(names), **ep,
                            meta_json=np.array(json.dumps(J.jsonable({k: v for k, v in summary.items() if k != "per_actuator"}))))
        record["checkpoints"][str(st)] = summary
        t3 = summary["tail3"]
        print(json.dumps({"label": a.label, "steps": st, "num_timesteps": summary["num_timesteps"],
                          "t3_frac90": round(t3["frac90"], 4), "t3_frac99": round(t3["frac99"], 4),
                          "t3_cmd": round(t3["mean_command"], 4), "t3_raw": round(t3["raw_mean_window"], 4),
                          "t3_raw_pinned": None if t3["raw_mean_on_pinned_steps"] is None else round(t3["raw_mean_on_pinned_steps"], 4),
                          "t3_std": round(t3["policy_std"], 4), "falls": summary["falls"], "clean": summary["clean"],
                          "flagged": summary["flagged_over_bar"], "reward": round(summary["reward_mean"], 1),
                          "k3": {k: v["falls"] for k, v in summary.get("k3", {}).items()}, "seconds": summary["seconds"]}),
              flush=True)
        del model, normalizer
    record["modules_all_under_clone"] = J.modules_under(clone)
    with open(summary_path, "x") as fh:
        json.dump(J.jsonable(record), fh, indent=1)
    print(json.dumps({"wrote": summary_path}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
