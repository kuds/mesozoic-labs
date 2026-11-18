"""Judged crawl runs on the 60 cm models (PREREG_CRAWL60.md): seeds 1 and 2 on two models, plus the impratio=10
sensitivity runs (reported, not judged). The controller and the rules are crawl.py's, unchanged; only the gait
parameters are scaled (PARAMS60).

Usage: python crawl_judged60.py [--out results/crawl60_2026-11-18]   Simulation only.
"""
import argparse
import json
import math
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import crawl

HERE = Path(__file__).resolve().parent
MODELS = ("trike60_C_closed", "trike60_C_closed_ankleroll")
SEEDS = (1, 2)
S = 0.8
LENGTHS = ("stride_m", "lift_m", "margin_plan_m", "body_y_max_m", "body_dx_max_m", "stance_press_m")
TIMES = ("shift_s", "swing_s")
PARAMS60 = {k: (v * S if k in LENGTHS else v * math.sqrt(S) if k in TIMES else v) for k, v in crawl.PARAMS.items()}


def model_path(name, impratio):
    path = HERE / "generated" / f"{name}.xml"
    if not impratio:
        return path
    text = path.read_text()
    assert text.count("<option ") == 1 and "impratio" not in text
    text = text.replace("<option ", '<option impratio="10" ', 1)
    tmp = Path(tempfile.mkdtemp()) / f"{name}_impratio10.xml"
    tmp.write_text(text)
    return tmp


def one(job):
    name, seed, impratio = job
    r = crawl.run(model_path(name, impratio), seed, PARAMS60)
    rules, info = crawl.judge(r)
    slim = {k: v for k, v in r.items() if k != "log"}
    return name, seed, impratio, rules, info, slim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "crawl60_2026-11-18"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s, imp) for imp in (False, True) for m in MODELS for s in SEEDS]
    rows, verdict, sens = [], {}, {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        for name, seed, imp, rules, info, slim in ex.map(one, jobs):
            tag = f"{name}{'_impratio10' if imp else ''}_seed{seed}"
            (out / f"{tag}.json").write_text(json.dumps({**slim, "rules": rules, "info": info, "impratio10": imp},
                                                        indent=1, default=float) + "\n")
            rows.append((name, seed, imp, rules, info, slim["mu"]))
            d = sens if imp else verdict
            d[name] = d.get(name, True) and all(rules.values())
    lines = ["# Crawl on the 60 cm models (simulation)", "",
             "Rules: `PREREG_CRAWL60.md` (C1-C6 of `PREREG_CRAWL.md`, thresholds unchanged; gait parameters scaled, "
             "no tuning). Seeds 1 and 2. Margin on the ankle-pin basis.", "",
             f"Parameters: {json.dumps({k: round(v, 4) for k, v in PARAMS60.items()})}", "",
             "| Model | Contact | Seed | μ | C1 | C2 | C3 | C4 | C5 | C6 | Min margin mm | Max pin motion mm | Travel m | Max tilt ° | Speed m/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, seed, imp, rules, info, mu in rows:
        lines.append(f"| {name} | {'impratio 10 (reported)' if imp else 'as built (judged)'} | {seed} | {mu:.3f} | " +
                     " | ".join("pass" if rules[k] else "**FAIL**" for k in ("C1", "C2", "C3", "C4", "C5", "C6")) +
                     f" | {1000 * info['min_margin_measured_m']:.1f} | {1000 * info['max_planted_slip_m']:.1f} | "
                     f"{info['travel_m']:.3f} | {info['max_tilt_deg']:.2f} | {info['speed_mps']:.4f} |")
    lines += ["", "Verdicts (judged, as built):", ""] + [f"- {m}: **{'PASS' if v else 'FAIL'}**" for m, v in verdict.items()]
    lines += ["", "Sensitivity, impratio 10 (reported, not judged):", ""]
    lines += [f"- {m}: {'all rules hold' if v else 'rules fail'}" for m, v in sens.items()]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
