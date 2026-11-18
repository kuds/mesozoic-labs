"""Report-only: joint torque peaks in the 60 cm crawl, without the start-up transient.

crawl.judge's joint_peak_nm includes t = 0, where each seed's ±0.02 rad joint offsets make the stiff position servos
snap back (the independent recheck found every quoted peak in the first 10 ms). This re-runs the judged crawl runs
(deterministic; the rule results must match results/crawl60_2026-11-18c/) and reports, per axis, the peak and RMS
joint torque over the samples after the first second of standing (t >= 1.0 s), and the peak in the first 0.2 s.

Usage: python crawl_peaks60.py [--out results/crawl60_2026-11-18c/peaks_after_startup.json]   Simulation only.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import crawl
from crawl_judged60 import MODELS, PARAMS60, SEEDS

HERE = Path(__file__).resolve().parent
GROUPS = {"roll": ("roll",), "pitch": ("pitch",), "front elbow": ("left_fore_flexion", "right_fore_flexion"),
          "hind knee": ("left_hind_flexion", "right_hind_flexion")}


def one(job):
    name, seed = job
    r = crawl.run(HERE / "generated" / f"{name}.xml", seed, PARAMS60)
    rules, _ = crawl.judge(r)
    t = np.array(r["log"]["t"])
    frc = np.abs(np.array(r["log"]["frc"]))
    names = [f"{c}_{a}" for c in crawl.CORNERS for a in crawl.AXES]
    out = {"model": name, "seed": seed, "rules_all_pass": all(rules.values()), "rules": rules}
    for label, keys in GROUPS.items():
        cols = [i for i, n in enumerate(names) if any(n.endswith(k) for k in keys)]
        late, early = frc[t >= 1.0][:, cols], frc[t < 0.2][:, cols]
        out[label] = {"peak_after_1s_nm": round(float(late.max()), 2),
                      "rms_after_1s_nm": round(float(np.sqrt((late ** 2).mean(0)).max()), 2),
                      "peak_first_0.2s_nm": round(float(early.max()), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "crawl60_2026-11-18c" / "peaks_after_startup.json"))
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(one, [(m, s) for m in MODELS for s in SEEDS]))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    for r in res:
        print(r["model"], r["seed"], "rules", "all pass" if r["rules_all_pass"] else
              [k for k, v in r["rules"].items() if not v],
              {k: (v["peak_after_1s_nm"], v["peak_first_0.2s_nm"]) for k, v in r.items() if isinstance(v, dict) and
               "peak_after_1s_nm" in v})


if __name__ == "__main__":
    main()
