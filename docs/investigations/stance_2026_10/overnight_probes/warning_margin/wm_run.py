"""MARGIN-TO-RAIL TEST, main run (design fixed in PREREG.txt, sha256 in PREREG.sha256).  No training.

    PYTHONPATH=<r10round/ecs/code>:<this folder> OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 15 \
        venv/bin/python -B wm_run.py [--only run:steps,...]

Per checkpoint (PREREG order) and seed 3042-3061: the unnudged episode (value 0); BIAS bisection on (0, 2.0] (2.0
parks by construction, not simulated), 7 halvings; OBS: d = 12 deg, then, if parked, 7 halvings on (0, 12]; D3: OBS
d = 1, 2, 4 deg on seeds 3042-3046.  Every episode is appended to episodes.jsonl (resume-safe: an episode already in
the file is not re-run; the bisection is deterministic, so a restart retraces the same values).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import wm_common as C

ORDER = [("s80", 12_001_280), ("s42", 3_145_728), ("s60", 7_340_032), ("s80", 11_534_336), ("s80", 10_485_760),
         ("s60", 6_291_456), ("s42", 2_097_152)]
SEEDS = list(range(3042, 3062))
CURVE_SEEDS = list(range(3042, 3047))
CURVE_D = (1.0, 2.0, 4.0)
N_HALV = 7
BIAS_HI = 2.0
OBS_HI = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    order = ORDER
    if a.only:
        want = {(w.split(":")[0], int(w.split(":")[1])) for w in a.only.split(",")}
        order = [c for c in ORDER if c in want]
    path = C.HERE / "episodes.jsonl"
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            cache[(r["run"], r["ckpt_steps"], r["kind"], round(r["value"], 10), r["seed"])] = r
    factory, _ = C.make_factory()
    env = factory()
    assert env.unwrapped.model.actuator(C.A_T3).name == "tail_3_pitch_act"
    fh = open(path, "a")
    t_all = time.time()
    for run, steps in order:
        pol = C.Policy(run, steps, factory)
        n_new = 0
        t_ck = time.time()

        def get(kind: str, value: float, seed: int) -> dict:
            nonlocal n_new
            key = (run, steps, kind, round(value, 10), seed)
            if key in cache:
                return cache[key]
            e = C.episode(env, pol, seed, kind, value)
            row = {"run": run, "ckpt_steps": steps, **e}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            cache[key] = row
            n_new += 1
            return row

        for seed in SEEDS:
            base = get("none", 0.0, seed)
            # BIAS
            if base["parked"]:
                tb = 0.0
            else:
                lo, hi = 0.0, BIAS_HI
                for _ in range(N_HALV):
                    mid = (lo + hi) / 2.0
                    if get("bias", mid, seed)["parked"]:
                        hi = mid
                    else:
                        lo = mid
                tb = hi
            # OBS
            if base["parked"]:
                to = 0.0
            else:
                top = get("obs", OBS_HI, seed)
                if not top["parked"]:
                    to = float("inf")
                else:
                    lo, hi = 0.0, OBS_HI
                    for _ in range(N_HALV):
                        mid = (lo + hi) / 2.0
                        if get("obs", mid, seed)["parked"]:
                            hi = mid
                        else:
                            lo = mid
                    to = hi
            print(f"{run} {steps} seed {seed}: base cmd {base['cmd_win']:+.3f} | T_bias {tb:.4f} | T_obs {to} | "
                  f"new eps {n_new} | {time.time() - t_ck:.0f} s", flush=True)
        for seed in CURVE_SEEDS:
            for d in CURVE_D:
                get("obs", d, seed)
        print(f"== {run} {steps} done: {n_new} new episodes, {time.time() - t_ck:.0f} s (total {time.time() - t_all:.0f} s)",
              flush=True)
    fh.close()
    env.close()
    (C.HERE / "wm_run.finished").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
