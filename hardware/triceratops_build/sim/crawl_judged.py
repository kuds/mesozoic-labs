"""Judged crawl runs (PREREG_CRAWL.md, amendment 1): seeds 1 and 2 on four models; verdict per model.

Usage: python crawl_judged.py [--out results/crawl_2026-11-18]   Simulation only.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import crawl

HERE = Path(__file__).resolve().parent
MODELS = ("trike_T20_closed", "trike_T15_closed", "trike_T20_closed_ankleroll", "trike_T15_closed_ankleroll")
SEEDS = (1, 2)


def one(job):
    name, seed = job
    r = crawl.run(HERE / "generated" / f"{name}.xml", seed)
    rules, info = crawl.judge(r)
    slim = {k: v for k, v in r.items() if k != "log"}
    return name, seed, rules, info, slim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "crawl_2026-11-18"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s) for m in MODELS for s in SEEDS]
    rows, verdict = [], {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for name, seed, rules, info, slim in ex.map(one, jobs):
            (out / f"{name}_seed{seed}.json").write_text(json.dumps({**slim, "rules": rules, "info": info},
                                                                    indent=1, default=float) + "\n")
            rows.append((name, seed, rules, info))
            verdict[name] = verdict.get(name, True) and all(rules.values())
    lines = ["# Crawl, judged runs (simulation)", "",
             "Rules: `PREREG_CRAWL.md` (C1-C6 and amendment 1). Seeds 1 and 2. Margin on the ankle-pin basis.", "",
             "| Model | Seed | μ | C1 | C2 | C3 | C4 | C5 | C6 | Min margin mm | Max pin motion mm | Travel m | Max tilt ° | Speed m/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, seed, rules, info in rows:
        mu = json.loads((out / f"{name}_seed{seed}.json").read_text())["mu"]
        lines.append(f"| {name} | {seed} | {mu:.3f} | " + " | ".join("pass" if rules[k] else "**FAIL**" for k in
                     ("C1", "C2", "C3", "C4", "C5", "C6")) +
                     f" | {1000 * info['min_margin_measured_m']:.1f} | {1000 * info['max_planted_slip_m']:.1f} | "
                     f"{info['travel_m']:.3f} | {info['max_tilt_deg']:.2f} | {info['speed_mps']:.4f} |")
    lines += ["", "Verdicts:", ""] + [f"- {m}: **{'PASS' if v else 'FAIL'}**" for m, v in verdict.items()]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
