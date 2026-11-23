"""Steering evaluation of a 60 cm policy trained under a steering command mode (decision D20). Simulation only.

Runs a checkpoint, deterministic actions, servo delay 10 ms, on the run's own env config reduced to the flat floor
(``trike_env.flat_floor``):
- ``schedule``: N episodes of the env's own seeded request schedule (the training distribution), seeds 500000+;
- ``courses``: scripted requests an operator would give (``COURSES``), applied through ``set_command_target`` at
  set times: straight, 90 deg turns left and right while walking, a stop with a turn on the spot, and a stand.
Each episode's measures come from ``telemetry/steering.py`` on the env's step info (the shared contract's names),
with the 60 cm tolerances (``TOLERANCES``). Writes ``steer_eval.json`` and, per course, a top view of the path with
the requested headings (``steer_course_<name>.png``), labelled SIMULATION. Nothing here judges.

Usage:
  python hardware/triceratops_build/rl/steer_eval.py --ckpt-dir DIR --out OUT [--episodes 5] [--no-charts]
DIR holds one SB3 checkpoint pair and the run's run_record.json. The weights load with torch.load(weights_only=True)
and the normalisation statistics through a restricted unpickler (``analysis/gait_pipe1.py``); nothing stored runs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE / "analysis", HERE / "analytics"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TOLERANCES = {"velocity_tolerance": 0.05, "yaw_rate_tolerance": 0.1, "heading_tolerance": math.radians(10.0)}
SCHEDULE_SEED0 = 500_000
COURSE_SEED = 600_000
# (time s, heading offset from the start heading in degrees, speed m/s): each row is an operator's request held
# until the next; the episode runs to the last row's time plus the hold.
COURSES = {
    "turns_walking": {"rows": [(0.0, 0, 0.25), (5.0, 90, 0.25), (14.0, 0, 0.25), (23.0, -90, 0.25)], "hold_s": 9.0},
    # 179 deg, not 180: a half turn sits on the angle wrap, where noise would choose the direction
    "turn_on_the_spot": {"rows": [(0.0, 0, 0.25), (5.0, 0, 0.0), (7.0, 179, 0.0), (16.0, 179, 0.25)],
                         "hold_s": 5.0},
    "stand": {"rows": [(0.0, 0, 0.0)], "hold_s": 10.0},
}


def make_env(env_config: dict, seconds: float | None = None):
    import trike_env as te

    cfg = te.flat_floor(te.TrikeEnvConfig.from_dict({**env_config,
                                                     "model_path": str(te.MODEL_DIR / Path(env_config["model_path"]).name)}))
    if cfg.command_mode == "none":
        raise SystemExit("this run did not train under a steering command_mode; nothing to evaluate")
    cfg = cfg.replace(delay_steps_fixed=10)
    if seconds:
        cfg = cfg.replace(episode_seconds=seconds)
    return te.TrikeEnv(cfg)


def run_episode(env, act, seed: int, course: dict | None = None, track: bool = False) -> dict:
    """One episode: the env's own schedule, or ``course`` (rows of time, heading offset deg, speed). Returns the
    steering measures, plus the path (x, y, heading, and the requested heading) when ``track``."""
    from telemetry.steering import episode_measures

    options = None
    rows = []
    if course is not None:
        rows = list(course["rows"])
        t0, off0, v0 = rows.pop(0)
        options = {"command_target": (math.radians(off0), v0)}
    obs, _ = env.reset(seed=seed, options=options)
    h0 = env._heading
    infos, path = [], []
    end = (course["rows"][-1][0] + course["hold_s"]) if course is not None else None
    while True:
        if rows and env._step_count * env.dt >= rows[0][0] - 1e-9:
            _, off, v = rows.pop(0)
            obs = env.set_command_target(h0 + math.radians(off), v)  # act on the observation carrying it
        obs, _, term, trunc, info = env.step(act(obs))
        infos.append(info)
        if track:
            x, y = env.data.xpos[env._chassis, :2]
            path.append((float(x), float(y), float(info["actual_heading"]), float(info["desired_heading"]),
                         float(info["desired_speed"]), float(info["time_s"])))
        if term or trunc or (end is not None and info["time_s"] >= end - 1e-9):
            break
    out = episode_measures(infos, **TOLERANCES)
    out["seed"] = seed
    if track:
        out["path"] = path
    return out


def load_policy(ckpt_dir: Path, env):
    import torch

    import gait_pipe1 as gp

    import gait_report

    zp, pk, _ = gait_report.find_pair(ckpt_dir)  # the newest matched pair, as the gait report and the sweep take
    st = json.loads((ckpt_dir / "run_record.json").read_text())["settings"]
    mean, var, clip, eps = gp.load_obs_stats(pk)
    policy = gp.build_policy(env, zp, st["net"], st["log_std_init"])

    def act(obs):
        o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        with torch.no_grad():
            a, _, _ = policy(torch.as_tensor(o[None]), deterministic=True)
        return a.numpy()[0]

    return act, zp.name


def course_chart(name: str, rec: dict, out: Path, label: str) -> str:
    """Two panels: the top view of the path with the requested and actual headings, and both headings against
    time (where a turn on the spot shows)."""
    import chart_style as cs

    cs.setup()
    plt = cs.plt
    p = np.array(rec["path"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.6), gridspec_kw={"width_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.06, right=0.97, top=0.8, bottom=0.11, wspace=0.22)
    ax.plot(p[:, 0], p[:, 1], color=cs.SERIES[0], lw=cs.LW, label="chassis path")
    for i in np.arange(0, len(p), 50):  # one pair of arrows per second
        x, y, h, hd = p[i, :4]
        ax.annotate("", xy=(x + 0.12 * math.cos(hd), y + 0.12 * math.sin(hd)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=cs.SERIES[1], lw=1.2))
        ax.annotate("", xy=(x + 0.08 * math.cos(h), y + 0.08 * math.sin(h)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=cs.SERIES[0], lw=1.0))
    ax.plot([], [], color=cs.SERIES[1], lw=1.2, label="requested heading (every 1 s)")
    ax.plot([], [], color=cs.SERIES[0], lw=1.0, label="actual heading")
    ax.scatter([p[0, 0]], [p[0, 1]], s=40, color=cs.INK2, zorder=5)
    ax.annotate("start", (p[0, 0], p[0, 1]), xytext=(6, -12), textcoords="offset points", fontsize=8, color=cs.INK2)
    # equal scale, at least 1.5 m across, so a path that barely moves is not magnified into noise
    cx, cy = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
    half = max(0.75, np.ptp(p[:, 0]) / 2 + 0.2, np.ptp(p[:, 1]) / 2 + 0.2)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x, m")
    ax.set_ylabel("y, m")
    ax.set_title("Top view")
    ax.legend(loc="upper left", fontsize=8)
    t = p[:, 5]
    unwrap = lambda a: np.degrees(np.unwrap(a))  # noqa: E731
    ax2.plot(t, unwrap(p[:, 3]), color=cs.SERIES[1], lw=cs.LW, label="requested heading")
    ax2.plot(t, unwrap(p[:, 2]), color=cs.SERIES[0], lw=cs.LW, label="actual heading")
    ax2.set_xlabel("time, s")
    ax2.set_ylabel("heading, deg (unwrapped)")
    ax2.set_title("Heading against time")
    ax2.legend(loc="best", fontsize=8)
    es, tr = rec["events_summary"], rec["tracking"]
    med = es["heading_error_at_settle_deg"]["median"]
    sub = (f"{'fell (' + rec['termination'] + ')' if rec['fell'] else 'no fall'}; speed error mean "
           f"{(tr['speed_error_mps']['mean'] or 0):.3f} m/s, turn-rate error mean "
           f"{(tr['yaw_rate_error_radps']['mean'] or 0):.3f} rad/s (both over {tr.get('average_s', 0):.1f} s means); "
           f"heading error 3 s after each request, median {med if med is not None else float('nan'):.1f} deg")
    cs.frame(fig, f"Steering course '{name}': {label}", sub,
             "SIMULATION. Deterministic policy, 10 ms delay, flat floor. Source: rl/steer_eval.py.")
    fname = f"steer_course_{name}.png"
    cs.save(fig, out / fname)
    return fname


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    from telemetry.steering import summarise

    cdir, out = Path(a.ckpt_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ec = json.loads((cdir / "run_record.json").read_text())["settings"]["env_config"]
    env = make_env(ec)
    act, ckpt = load_policy(cdir, env)
    label = a.label or cdir.name
    sched = [run_episode(env, act, SCHEDULE_SEED0 + i) for i in range(a.episodes)]
    courses, charts = {}, []
    for name, course in COURSES.items():
        cenv = make_env(ec, seconds=course["rows"][-1][0] + course["hold_s"] + 1.0)
        rec = run_episode(cenv, act, COURSE_SEED, course=course, track=True)
        courses[name] = rec
        if not a.no_charts:
            charts.append(course_chart(name, rec, out, label))
    res = {"simulation_only": True, "tool": "hardware/triceratops_build/rl/steer_eval.py", "checkpoint": ckpt,
           "tolerances": TOLERANCES, "schedule": {"summary": summarise(sched), "episodes": sched},
           "courses": {n: {k: v for k, v in r.items() if k != "path"} for n, r in courses.items()},
           "course_definitions": COURSES, "charts": charts}
    (out / "steer_eval.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    s = res["schedule"]["summary"]
    print(f"schedule ({s['episodes']} episodes): falls {s['falls']}; speed error {s['speed_error_mps_mean']}, "
          f"turn-rate error {s['yaw_rate_error_radps_mean']}; in tolerance {s['share_in_tolerance_mean']}; "
          f"turns settled {s['turns_settled_share']}")
    for n, r in courses.items():
        print(f"course {n}: {'fell ' + r['termination'] if r['fell'] else 'no fall'}; heading error at settle "
              f"median {r['events_summary']['heading_error_at_settle_deg']['median']} deg")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
