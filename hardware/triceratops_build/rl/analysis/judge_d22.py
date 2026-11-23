"""The foreleg comparison's rules (PREREG_FORELEG_D22.md; decisions D22, D23, D24), read from each run's record and
the tools' JSON: two lengths, four seeds each. Simulation only.

Per run (length x seed), first the checks (any failure makes the run NOT JUDGED, and then no choice is made):
- the run record: every session at the run commit with a clean tree, none allowed to change settings, the last one
  completed; the seed and the model; every other setting (PPO, schedules, envs, steps, the environment) equal to run
  3's (`results/run3_reference_settings.json`), steering off; its last checkpoint is the one the tools measured;
- the tools: the model's and trike_env.py's sha256 as each recorded them against the run commit's files; the run
  commit; the final pair (10,000,000-10,008,191 steps); the same pair in all three tools; the gait report's settings
  (seeds 100000-100004, 20 s, 10 ms, deterministic, measured from 2 s) and 5 episodes in each judged case; the
  sweep's 8 episodes per point on seeds 200000-200007; the foot-load tool's settings;
- across runs: each (length, seed) once, eight different checkpoint pairs.
Then run 3's rules F1, F2 and T1-T5 (`judge_run3.judge_seed`, unchanged) and the measures:
- L: the worst front foot's 99th percentile of the 10 ms mean normal force, mean of 5 episodes (`foot_loads.py`);
- H_all: F2's figure, the worst joint's RMS torque over its continuous rating in the gait report's target case;
- H_front: the same for the six front drives only (roll, pitch, elbow); the elbows alone are reported (D24);
- N: falls at run 3's judged sweep points plus the gait report's target, 0.15 m/s and nominal cases (not its
  "other foot" case, which swaps in another leg).

The choice (D23, D24), in this order:
1. A length "walks" when F1 and F2 pass on all four of its seeds. If exactly one walks, it is chosen. If neither,
   no choice.
2. If any run's L or H_front is undefined or not finite (a foot-load episode fell), no choice.
3. Otherwise 128 mm, unless 160 mm meets all of, on the four-seed means:
   - loads: mean L(160) <= 0.85 x mean L(128);
   - falls: mean N(160) <= mean N(128) + 7;
   - heat: mean H_all(160) <= mean H_all(128) + 0.035, and mean H_front(160) <= mean H_front(128) + 0.10.
   The tolerances are run 3's seed-to-seed differences with the same leg (N 25 / 18; H_all 0.836 / 0.801;
   H_front 0.823 / 0.722): "not clearly worse".

Usage:
  python hardware/triceratops_build/rl/analysis/judge_d22.py --run-commit SHA \\
      --run 128 0 RECORD GAIT SWEEP FOOT  (x 8: lengths 128 and 160, seeds 0-3)  [--out FILE.json]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_run3 as j3  # noqa: E402

LENGTHS = (128, 160)
SEEDS = (0, 1, 2, 3)
LOAD_FACTOR, FALLS_TOL, HEAT_ALL_TOL, HEAT_FRONT_TOL = 0.85, 7.0, 0.035, 0.10
JUDGED_POINTS = (j3.STEP[2], j3.STEP[3], j3.RAMP[5], j3.RAMP[8], j3.FRICTION_03, *j3.PUSH.values())
GAIT_CASES = ("target_0.25", "cmd_0.15", "nominal_0.25")
SWEEP_SEEDS = list(range(200000, 200008))
FOOT_SETTINGS = {"seeds": list(range(300000, 300005)), "seconds": 20.0, "from_s": 2.0, "command_mps": 0.25,
                 "delay_ms": 10, "floor": "flat", "randomisation": False, "observation_noise": True}
FRONT_DRIVES = tuple(f"{s}_fore_{j}" for s in ("left", "right") for j in ("roll", "pitch", "flexion"))
ELBOWS = ("left_fore_flexion", "right_fore_flexion")
GAIT_SETTINGS = {"seeds": list(range(100000, 100005)), "delay_ms": 10, "deterministic": True, "episode_s": 20.0,
                 "window_from_s": 2.0}
REFERENCE = Path(__file__).resolve().parent / "results" / "run3_reference_settings.json"
ARG_SKIP = ("out", "seed", "model", "exit_with_parent")  # per run or per machine; compared separately or not at all
PPO_KEYS = ("n_envs", "total_steps", "checkpoint_every", "n_steps", "batch_size", "n_epochs", "lr_start", "lr_end",
            "gamma", "gae_lambda", "clip_range", "ent_coef", "log_std_init", "net", "penalty_ramp_steps",
            "penalty_ramp_start", "late_ramp_from", "late_ramp_to", "terrain_ramp_from", "terrain_ramp_to",
            "eval_every", "eval_commands", "eval_seconds")


def model_of(length: int) -> str:
    return f"trike60_C_closed_capsule_fore{length}.xml"


def _git_show(run_commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(j3.REPO), "show", f"{run_commit}:{path}"], capture_output=True,
                          check=True).stdout


def checks(length: int, seed: int, run_commit: str, record: dict, gait: dict, sweep: dict, foot: dict) -> list[str]:
    import hashlib

    bad = []
    model = model_of(length)
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    model_sha = sha(_git_show(run_commit, f"hardware/triceratops_build/sim/generated/{model}"))
    env_sha = sha(_git_show(run_commit, "hardware/triceratops_build/rl/trike_env.py"))
    preset = json.loads(_git_show(run_commit, "hardware/triceratops_build/rl/configs/run3_terrain.json"))
    # the run record
    sessions = record.get("sessions") or []
    if not sessions or any(s["git"]["commit"] != run_commit or s["git"]["dirty"] for s in sessions):
        bad.append("run record: a session is not at the run commit with a clean tree")
    if not sessions or sessions[-1].get("status") != "completed":
        bad.append("run record: the last session did not complete")
    st = record.get("settings", {})
    ref_record = json.loads(REFERENCE.read_text())["record"]
    ref, ref_args = ref_record["settings"], ref_record["sessions"][0]["args"]
    if st.get("seed") != seed or any(s["args"].get("seed") != seed for s in sessions):
        bad.append("run record: another seed")
    if Path(st.get("model", "")).name != model or any(f"{s['args'].get('model')}.xml" != model for s in sessions):
        bad.append("run record: another model")
    if any(st.get(k) != ref[k] for k in PPO_KEYS) or st.get("total_steps") != j3.TOTAL_STEPS or st.get("n_envs") != 8:
        bad.append("run record: training settings differ from run 3's")
    if any(s["args"].get("allow_changes") or s.get("changed_settings") for s in sessions):
        bad.append("run record: a session changed its settings")
    if any(s["args"].get(k) != v for s in sessions for k, v in ref_args.items() if k not in ARG_SKIP):
        bad.append("run record: a session's arguments differ from run 3's")
    ec, rec3 = st.get("env_config", {}), ref["env_config"]
    if any(ec.get(k) != v for k, v in rec3.items() if k != "model_path") or ec.get("command_mode", "none") != "none":
        bad.append("run record: the environment differs from run 3's (or steering is on)")
    if any(ec.get(k) != v for k, v in preset.items()):
        bad.append("run record: the environment is not run3_terrain's")
    last = (sessions[-1].get("saves") or [None])[-1] if sessions else None
    if last != gait["policy"]["checkpoint"]:
        bad.append("run record: its last checkpoint is not the measured one")
    # the tools
    if gait["environment"]["model_sha256"].get(model) != model_sha:
        bad.append("gait report: model sha256 differs from the run commit's")
    if sweep["env"]["model"] != model or sweep["env"]["model_sha256"] != model_sha:
        bad.append("sweep: model differs from the run commit's")
    fp = foot["provenance"]
    if fp["model"] != model or fp["model_sha256"] != model_sha:
        bad.append("foot loads: model differs from the run commit's")
    if {gait["environment"]["trike_env_sha256"], sweep["env"]["trike_env_sha256"], fp["trike_env_sha256"]} != {env_sha}:
        bad.append("trike_env.py differs from the run commit's")
    if {gait["policy"]["run_commit"], sweep["run_commit"], fp["run_commit"]} != {run_commit}:
        bad.append("run commit differs")
    if not j3.TOTAL_STEPS <= gait["policy"]["steps"] < j3.TOTAL_STEPS + j3.BLOCK:
        bad.append(f"checkpoint at {gait['policy']['steps']} steps is not the run's final pair")
    pair = (gait["policy"]["zip_sha256"], gait["policy"]["vecnormalize_sha256"])
    if (sweep["checkpoint"]["zip_sha256"], sweep["checkpoint"]["vecnormalize_sha256"]) != pair or \
            (fp["zip_sha256"], fp["vecnormalize_sha256"]) != pair:
        bad.append("the tools measured different checkpoints")
    gs = gait.get("settings", {})
    if any(gs.get(k) != v for k, v in GAIT_SETTINGS.items()) or any(
            gait["cases"][c]["summary"].get("episodes") != 5 for c in GAIT_CASES):
        bad.append("gait report: settings or episode counts differ from run 3's (e.g. --quick)")
    if sweep.get("seeds_per_point") != 8 or list(sweep.get("seeds", [])) != SWEEP_SEEDS:
        bad.append("sweep: not 8 episodes per point on seeds 200000-200007")
    if foot.get("settings") != FOOT_SETTINGS:
        bad.append("foot loads: settings differ from the pre-registration")
    return bad


def measure(gait: dict, sweep: dict, foot: dict) -> dict:
    rules = j3.judge_seed(gait, sweep)
    per = (gait["cases"]["target_0.25"]["summary"].get("drives") or {}).get("per_joint") or {}
    front = [per[j]["rms_over_continuous"]["mean"] for j in FRONT_DRIVES if j in per]
    elbows = [per[j]["rms_over_continuous"]["mean"] for j in ELBOWS if j in per]
    falls = sum(sweep["points"][k]["summary"]["falls"] for k in JUDGED_POINTS) + sum(
        gait["cases"][c]["summary"]["falls"] for c in GAIT_CASES)  # KeyError if a count is missing: not judged
    return {"rules": {k: rules[k]["pass"] for k in j3.RULES}, "rule_values": rules,
            "L": foot.get("front_p99_window_n"), "L_hind": foot.get("hind_p99_window_n"),
            "H_all": rules["F2"]["worst"], "H_front": max(front) if len(front) == 6 else None,
            "H_elbow": max(elbows) if len(elbows) == 2 else None, "N": falls,
            "foot_falls": foot.get("falls"), "checkpoint_sha256": gait["policy"]["zip_sha256"]}


def _mean(xs):
    return sum(xs) / len(xs)


def choose(runs: dict) -> dict:
    """``runs[(length, seed)]``: a measure() dict, or None when not judged."""
    if any(runs.get((n, s)) is None for n in LENGTHS for s in SEEDS):
        return {"choice": None, "why": "not judged: a run is missing or failed its checks"}
    walks = {n: all(runs[(n, s)]["rules"]["F1"] and runs[(n, s)]["rules"]["F2"] for s in SEEDS) for n in LENGTHS}
    if walks[128] != walks[160]:
        n = 128 if walks[128] else 160
        return {"choice": n, "why": f"only {n} mm passes F1 and F2 on all four seeds", "walks": walks}
    if not any(walks.values()):
        return {"choice": None, "why": "neither length passes F1 and F2 on all four seeds", "walks": walks}
    def ok(x):
        return x is not None and math.isfinite(x)

    if any(not ok(runs[(n, s)]["L"]) or not ok(runs[(n, s)]["H_front"]) for n in LENGTHS for s in SEEDS):
        return {"choice": None, "why": "a foot load or front-drive figure is undefined", "walks": walks}
    m = {n: {k: _mean([runs[(n, s)][k] for s in SEEDS]) for k in ("L", "N", "H_all", "H_front")} for n in LENGTHS}
    for n in LENGTHS:  # reported, not judged (D24)
        el = [runs[(n, s)].get("H_elbow") for s in SEEDS]
        m[n]["H_elbow_reported"] = _mean(el) if all(ok(x) for x in el) else None
    tests = {"loads": m[160]["L"] <= LOAD_FACTOR * m[128]["L"],
             "falls": m[160]["N"] <= m[128]["N"] + FALLS_TOL,
             "heat": m[160]["H_all"] <= m[128]["H_all"] + HEAT_ALL_TOL
             and m[160]["H_front"] <= m[128]["H_front"] + HEAT_FRONT_TOL}
    if all(tests.values()):
        return {"choice": 160, "why": "160 mm beats the 128 mm default on loads, falls and heat", "walks": walks,
                "means": m, "tests": tests}
    return {"choice": 128, "why": "the default: 160 mm does not beat it on " +
            ", ".join(k for k, v in tests.items() if not v), "walks": walks, "means": m, "tests": tests}


def _fmt(x, nd):
    return "undefined" if x is None else f"{x:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-commit", required=True)
    ap.add_argument("--run", nargs=6, action="append", required=True,
                    metavar=("LENGTH", "SEED", "RECORD_JSON", "GAIT_JSON", "SWEEP_JSON", "FOOT_JSON"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs, problems = {}, {}
    for length, seed, r, g, w, f in a.run:
        try:
            key = (int(length), int(seed))
        except ValueError:
            print(f"NOT JUDGED: length {length!r} or seed {seed!r} is not a number")
            continue
        if key in problems:
            problems[key] = ["given twice"]
            runs[key] = None
            continue
        try:
            record, gait, sweep = json.loads(Path(r).read_text()), j3._load(g), j3._load(w)
            foot = json.loads(Path(f).read_text())
            problems[key] = checks(key[0], key[1], a.run_commit, record, gait, sweep, foot)
            runs[key] = None if problems[key] else measure(gait, sweep, foot)
        except (OSError, KeyError, TypeError, ValueError, AttributeError, subprocess.CalledProcessError) as e:
            problems[key] = [f"could not be read or checked: {type(e).__name__}: {e}"]
            runs[key] = None
    shas = [v["checkpoint_sha256"] for v in runs.values() if v]
    if len(shas) != len(set(shas)):
        for k in runs:
            problems[k] = (problems.get(k) or []) + ["the same checkpoint is given for two runs"]
            runs[k] = None
    res = choose(runs)
    for key in sorted(problems):
        r = runs.get(key)
        print(f"{key[0]} mm seed {key[1]}: " + ("NOT JUDGED: " + "; ".join(problems[key]) if r is None else
              f"rules {''.join('P' if v else 'F' for v in r['rules'].values())} (F1 F2 T1-T5), L {_fmt(r['L'], 1)} N "
              f"(hind {_fmt(r['L_hind'], 1)}), H_all {_fmt(r['H_all'], 3)}, H_front {_fmt(r['H_front'], 3)}, "
              f"N {r['N']}"))
    print("CHOICE:", res["choice"], "-", res["why"])
    out = {"simulation_only": True, "prereg": "hardware/triceratops_build/rl/analysis/PREREG_FORELEG_D22.md",
           "run_commit": a.run_commit, "problems": {f"{k[0]}_s{k[1]}": v for k, v in problems.items()},
           "runs": {f"{k[0]}_s{k[1]}": v for k, v in runs.items()}, "choice": res}
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
