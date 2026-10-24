"""Name the saturating actuator of a placed stance checkpoint (stance5 replication copy of SP/foot/stance5/tools/
satroll5.py: judge_stance5r.py in place of judge_stance5.py, and --seed also takes 70 and 80) (report only; criteria SP/foot/stance5/criteria.txt,
PRE-REGISTERED READOUTS, "Saturation watch", D-D40 5a).

    S=/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad
    env ISO_WT=<the arm's clone> python3 $S/foot/stance3/tools/iso.py satroll5r.py RUN_DIR --arm {LTSB,LTS} --seed N \
        --steps STEPS [STEPS ...] --out OUT.json [--episodes 20]

SP/foot/stance4/satdiag/rollout/roll.py's method, reduced to what the alarm needs: the deterministic policy of the
checkpoint on the panel seeds 3042-3061 (20 episodes, heading 0, stance_report.run_panel with floor_truth=True, the
judge's v2 panel path), the policy loaded by stance_report._load_policy (the frozen VecNormalize statistics,
model.predict deterministic, SB3's clip to [-1, 1]).  Per actuator, over each episode's window (recorder index steps
200 .. L-1, the judge's), from the recorder's action (the clipped command the env received): the fraction of window
steps at |command| >= 0.9 (frac90), at >= 0.99 (frac99, the gate's max_actuator_saturation_fraction statistic) and the
mean command; each averaged over the episodes.  The named actuator is the one with the largest mean frac90, with the
sign of its mean command (criteria: "the actuator with the largest fraction of window steps at |command| >= 0.9, and
the sign of its mean command").  The gate's own per-episode max_actuator_saturation_fraction is checked against frac99
(equal to 1e-12) and the episodes over the stage's bar are counted.

Everything else is judge_stance5.py's, imported from its path: the arm's clone and pinned HEAD (clean tree), Ctx (the
clone's stance stage and [env]), run_bindings (plant, seed, symmetry record, task and stage_config, refused on a hard
failure), checkpoint_paths and the trace capture.  Nothing is written but OUT (never overwritten).  Report only: an
alarm stops nothing and fails nothing (D-D40 5a).
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
import time  # noqa: E402
import types  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
JUDGE = SCRATCH + "/foot/stance5/replication/judge/judge_stance5r.py"
SEED0, EPISODES = 3042, 20
FRAC_BAR = 0.9
#: What a sign means, where a diagnosis has read it (SP/foot/stance4/satdiag): report text only.
SIGN_NOTES = {("tail_3_pitch_act", +1): "tail raised (stance4 s42 and stance3 Run 2 held it here)"}


def load_judge():
    spec = importlib.util.spec_from_file_location("judge_stance5r", JUDGE)
    J = importlib.util.module_from_spec(spec)
    sys.modules["judge_stance5r"] = J
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


def reduce_panel(J, ctx, res, names: list[str], bar: float) -> dict:
    import numpy as np

    traces = [tr for tr, _ in J.TRACES]
    sm = res["stance_metrics"]
    assert len(traces) == len(sm), (len(traces), len(sm))
    nu = len(names)
    f90, f99, mact, lengths, gate, flagged = [], [], [], [], [], 0
    for tr, m in zip(traces, sm, strict=True):
        L = int(tr.length)
        lengths.append(L)
        act = np.asarray(tr.action, dtype=np.float64)
        g = float(m.max_actuator_saturation_fraction)
        gate.append(g)
        flagged += int(g > bar)
        if L <= ctx.settle:
            continue
        w = act[ctx.settle:L]
        a99 = (np.abs(w) >= 0.99).mean(axis=0)
        assert abs(float(a99.max()) - g) < 1e-12, (float(a99.max()), g)   # the gate's statistic, reproduced
        f90.append((np.abs(w) >= FRAC_BAR).mean(axis=0))
        f99.append(a99)
        mact.append(w.mean(axis=0))
    if not f90:
        return {"episodes": len(traces), "lengths": lengths, "named": None, "why": "no episode reached the window"}
    f90, f99, mact = (np.asarray(x) for x in (f90, f99, mact))
    per = [{"actuator": names[i], "frac90": float(f90[:, i].mean()), "frac99": float(f99[:, i].mean()),
            "mean_command": float(mact[:, i].mean()), "episodes_frac90_ge_0.5": int(np.sum(f90[:, i] >= 0.5))}
           for i in range(nu)]
    order = sorted(range(nu), key=lambda i: -per[i]["frac90"])
    top = per[order[0]]
    sign = 0 if top["mean_command"] == 0 else (1 if top["mean_command"] > 0 else -1)
    named = None if top["frac90"] == 0.0 else {
        "actuator": top["actuator"], "sign": sign, "frac90": top["frac90"], "frac99": top["frac99"],
        "mean_command": top["mean_command"], "note": SIGN_NOTES.get((top["actuator"], sign))}
    return {"episodes": len(traces), "episodes_in_window": int(f90.shape[0]), "lengths": lengths,
            "falls": [SEED0 + i for i, L in enumerate(lengths) if L < ctx.horizon],
            "gate_max_actuator_saturation_fraction": gate, "flagged_over_bar": flagged, "bar": bar,
            "named": named, "runner_up": per[order[1]] if nu > 1 else None,
            "per_actuator": sorted(per, key=lambda r: -r["frac90"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", help="the placed run directory (holds 01_stance/models)")
    ap.add_argument("--arm", required=True, choices=("LTSB", "LTS"), help="LTSB (stance5) or LTS (stance4, validation)")
    ap.add_argument("--seed", type=int, required=True, choices=(42, 52, 60, 70, 80))
    ap.add_argument("--steps", nargs="+", required=True, help="checkpoint slots: a multiple of 524288, final, best, robust_best")
    ap.add_argument("--out", required=True, help="the JSON to write (never overwritten)")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--expect-head", default=None, help="default: the arm's pinned HEAD in judge_stance5r.py ARMS")
    ap.add_argument("--allow-commit-mismatch", action="store_true",
                    help="validation only: the run's stage_config git_commit may differ from the clone HEAD")
    a = ap.parse_args()
    out_path = os.path.abspath(a.out)
    if os.path.exists(out_path):
        raise SystemExit(f"refusing to overwrite {out_path}")
    J = load_judge()
    spec = J.ARMS[a.arm]
    clone = spec["clone"]
    expect = a.expect_head or spec["head"]
    if expect is None:
        raise SystemExit(f"arm {a.arm}: HEAD is a PLACEHOLDER in judge_stance5r.py; pass --expect-head")
    if os.path.realpath(os.environ.get("ISO_WT", clone)) != os.path.realpath(clone):
        raise SystemExit(f"ISO_WT {os.environ.get('ISO_WT')} is not arm {a.arm}'s clone {clone}")
    modules = J.isolate(clone)
    git = J.git_state(clone)
    if git["status"] or git["head"] != expect:
        raise SystemExit(f"satroll5 runs the clean clone at {expect} (got {git['head']}, dirty={bool(git['status'])})")
    import torch

    torch.set_num_threads(1)
    J.install_trace_capture()
    ctx = J.Ctx(a.arm)
    names = actuator_names(ctx)
    bar = float(ctx.thresholds.declared().get("max_actuator_saturation_fraction", 0.10))
    from environments.shared.task_fingerprint import stage_task_fingerprint

    panel_task = stage_task_fingerprint(ctx.mid, ctx.stage, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity)["task_sha256"]
    run = os.path.abspath(a.run_dir)
    stage_dir = os.path.join(run, "01_stance")
    record = {"tool": os.path.realpath(__file__), "tool_sha256": J.sha256_file(os.path.realpath(__file__)),
              "judge": JUDGE, "judge_sha256": J.sha256_file(JUDGE), "report_only": True, "arm": a.arm, "seed": a.seed,
              "run_dir": run, "clone": clone, "git": git, "modules_environments": modules["environments"],
              "task_sha256_rolled": panel_task, "seeds": [SEED0, SEED0 + a.episodes - 1], "frac_bar": FRAC_BAR,
              "window": [ctx.settle, ctx.horizon], "actuators": names, "checkpoints": {}}
    for st in a.steps:
        steps = J.parse_steps(st)
        t0 = time.monotonic()
        stem, zp, vp = J.checkpoint_paths(os.path.join(stage_dir, "models"), steps)
        if not (os.path.exists(zp) and os.path.exists(vp)):
            record["checkpoints"][str(st)] = {"placed": False}
            print(json.dumps({"steps": st, "placed": False}), flush=True)
            continue
        ns = types.SimpleNamespace(seed=a.seed, steps=steps, no_symmetry_check=False, task_binding="require",
                                   allow_missing_records=False, allow_seed_mismatch=False,
                                   allow_commit_mismatch=bool(a.allow_commit_mismatch))
        checks = J.run_bindings(ctx, ns, stage_dir, zp, panel_task, git)
        if checks["hard_failures"]:
            raise SystemExit("binding refused: " + " | ".join(checks["hard_failures"]))
        predict, note = ctx.SR._load_policy(zp, vp, lambda: ctx.env_class(**ctx.env_kwargs), plant_identity=ctx.plant_identity,
                                            allow_legacy_plant=False)
        J.TRACES.clear()
        res = ctx.SR.run_panel(ctx.mid, predict=predict, episodes=a.episodes, seed=SEED0, settle_steps=ctx.settle,
                               horizon=ctx.horizon, env_kwargs=ctx.env_kwargs, plant_identity=ctx.plant_identity,
                               control_dt=ctx.control_dt, floor_truth=True, spawn_yaw=None)
        red = reduce_panel(J, ctx, res, names, bar)
        red.update({"checkpoint": stem, "model": zp, "model_sha256": J.sha256_file(zp), "vecnorm_sha256": J.sha256_file(vp),
                    "policy_note": note, "checkpoint_num_timesteps": checks.get("checkpoint_num_timesteps"),
                    "seconds": round(time.monotonic() - t0, 1)})
        record["checkpoints"][str(st)] = red
        nm = red.get("named") or {}
        print(json.dumps({"steps": st, "actuator": nm.get("actuator"), "sign": nm.get("sign"),
                          "frac90": None if not nm else round(nm["frac90"], 4),
                          "mean_command": None if not nm else round(nm["mean_command"], 4),
                          "flagged_over_bar": red.get("flagged_over_bar"), "seconds": red["seconds"]}), flush=True)
    record["modules_all_under_clone"] = J.modules_under(clone)
    with open(out_path, "x") as fh:
        json.dump(J.jsonable(record), fh, indent=1)
    print(json.dumps({"wrote": out_path}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"[satroll5r] {exc.code}", file=sys.stderr, flush=True)
        raise
