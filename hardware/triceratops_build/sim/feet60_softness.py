"""After the foot study's judged result (not pre-registered): how much the crawl's C5c (net sliding at the contacts)
depends on the contact softness, for the ball and the capsule. Simulation only.

The independent review of the foot study found that MuJoCo's soft contact lets a loaded sole creep in proportion to
1 / (impratio x contacts per sole): the ball has one contact, the capsule two, the flat box four. This runs the judged
crawl (feet60_study's C5c, the same windows and limit) on the frozen feet60 models at other impratio values, crawl
seeds 11 and 12. It changes no verdict of PREREG_FEET60.md; it says how close to the contact model's settings a verdict
sits.

Usage: python feet60_softness.py --out results/feet60_2026-11-22/softness_check.json [--workers 3]
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
POINTS = [(v, imp) for v, imps in (("ball_r30", (1, 2, 4)), ("capsule_r24", (0.5, 1, 2))) for imp in imps]
SEEDS = (11, 12)


def job(args):
    import mujoco
    import crawl
    import crawl_judged60 as cj
    import feet60_study as fs
    v, imp, seed = args
    src = (HERE / "feet60" / f"trike60_foot_{v}.xml").read_text()
    assert src.count("<option ") == 1 and "impratio" not in src
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"trike60_foot_{v}_impratio{imp}.xml"
        p.write_text(src if imp == 1 else src.replace("<option ", f'<option impratio="{imp}" ', 1))
        sl = fs.Slider(mujoco.MjModel.from_xml_path(str(p)))
        r = crawl.run(p, seed, cj.PARAMS60, step_hook=sl)
    rules, _ = crawl.judge(r)
    t, hist = np.array(sl.t), np.array(sl.hist)
    windows = {i: [(0.0, None)] for i in range(4)}
    for ph in r["phases"]:  # as feet60_study.crawl_job
        L = ph["leg_index"]
        windows[L][-1] = (windows[L][-1][0], ph["swing_start"])
        windows[L].append((ph["swing_end"], None))
    slides = []
    for i in range(4):
        windows[i][-1] = (windows[i][-1][0], t[-1])
        for td_, lo in windows[i]:
            a, b = np.searchsorted(t, td_ + 0.10), np.searchsorted(t, lo, side="right") - 1
            if b > a:
                slides.append(float(np.max(np.linalg.norm(hist[a:b + 1, i] - hist[a, i], axis=1))))
    rules = {k: bool(x) for k, x in rules.items()}
    rules["C5c"] = max(slides) <= fs.SLIDE_LIMIT
    return {"foot": v, "impratio": imp, "seed": seed, "slide_max_mm": round(1000 * max(slides), 2),
            "slide_median_mm": round(1000 * float(np.median(slides)), 2), "rules": rules}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    git = lambda *x: subprocess.run(["git", *x], cwd=HERE, capture_output=True, text=True).stdout.strip()  # noqa: E731
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(job, [(v, imp, s) for v, imp in POINTS for s in SEEDS])
    out = {"simulation_only": True, "tool": "sim/feet60_softness.py", "not_preregistered": True,
           "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
           "seeds": list(SEEDS), "wall_s": round(time.time() - t0, 1), "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
