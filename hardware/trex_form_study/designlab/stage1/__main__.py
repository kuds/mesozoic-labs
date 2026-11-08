"""Command line for Stage 1 (``STAGE1_PLAN.md`` section 7).

Run from hardware/trex_form_study:

  python -m designlab.stage1 prepare --body 60-N-T-10 [--body ...] --out stage1/runs/<run> [--no-gates]
  python -m designlab.stage1 gates   --body 60-N-T-10 --out stage1/runs/<run> [--only g1|g2]

``prepare`` writes, per body, ``<out>/<body>/plant.json`` (gains, envelopes, ranges, sensors, floors, hashes,
library versions, commit), ``refs.json`` with ``refs/<name>.npz`` (``refs``) and, unless ``--no-gates``,
``gates.json`` (``gates``: G1 and G2 of plan section 3).  ``gates`` reruns G1 and G2 on the saved references.
The exit status is 1 if a gate fails (plan section 3: stop).  ``tune``, ``train``, ``eval``, ``sweep``, ``report``
and ``all`` are later steps of the plan and are not implemented yet.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one core per process: the step-rate gate is per core

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import designlab  # noqa: E402,F401  (selects the offscreen GL backend before mujoco is imported)

from . import BODIES  # noqa: E402

LATER = {"tune": "plan section 3 step 6 (CMA-ES tuning of the tracker)",
         "train": "plan section 3 step 7 (residual PPO)",
         "eval": "plan section 3 step 8 (evaluation panels)",
         "sweep": "plan section 3 step 8 (margin sweeps)",
         "report": "plan section 3 step 9 (STAGE1_PROBE.md)",
         "all": "every step of plan section 3"}


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=float))


def plant_doc(body: str) -> dict:
    """``plant.json``: the hard-floor plant in full, and every floor's parameters, settle and step rate."""
    from . import plant

    doc = plant.describe(plant.build(body, "hard"))
    floors = {}
    for floor in plant.FLOORS:
        t0 = time.perf_counter()
        p = plant.build(body, floor)
        build_s = time.perf_counter() - t0
        p.reset()
        p.hold()
        t1 = time.perf_counter()
        p.step(2000)
        floors[floor] = dict(params=p.meta["floor_params"], info=p.meta["floor_info"], settle=p.meta["settle"],
                             build_s=build_s, steps_per_s_zero_command=2000 / (time.perf_counter() - t1))
    doc["floors"] = floors
    return doc


def run_gates(body: str, out: Path, only: str | None = None) -> dict:
    from . import gates

    path = out / "gates.json"
    doc = json.loads(path.read_text()) if path.exists() else dict(body=body)
    if only in (None, "g1"):
        doc["G1"] = gates.g1(body, out)
    if only in (None, "g2"):
        doc["G2"] = gates.g2(body, out)
    doc["pass"] = bool(doc.get("G1", {}).get("pass") and doc.get("G2", {}).get("pass"))
    _write(path, doc)
    return doc


def summary(doc: dict) -> str:
    lines = []
    g1, g2 = doc.get("G1"), doc.get("G2")
    if g1:
        st = ", ".join(f"{f} {'stands' if s['stands'] else 'FAILS'} (tilt max {s['tilt_from_home_max_deg']:.2f} deg)"
                       for f, s in g1["stand"].items())
        sr = g1["step_rate"]
        hashes = all(h["matches_record"] and h["matches_plan"] for h in g1["hashes"].values())
        lines.append(f"G1 {'pass' if g1['pass'] else 'FAIL'}: hashes {hashes}; {st}; tendon map "
                     f"err {max(g1['tendon_map']['max_err_joint_torque_Nm'], g1['tendon_map']['max_err_motor_speed_rad_s']):.1e}; "
                     f"repeat {g1['repeatability']['ok']}; {sr['single_process']:.0f} steps/s alone, "
                     f"{sr['per_process_mean']:.0f} per process x {sr['processes']}")
    if g2:
        lines.append(f"G2 {'pass' if g2['pass'] else 'FAIL'}: references match {g2['all_match']} (nominal "
                     f"{g2['nominal_speed_m_s']} m/s, CoP margin {g2['nominal_min_cop_margin_single_m']} m); nominal "
                     f"replay {g2['nominal_strides_with_envelope']:.1f} strides with the envelope, "
                     f"{g2['nominal_strides_without_envelope']:.1f} without (design study, screened MJCF: "
                     f"{g2['design_study_strides_screened_model']})")
        info = g2.get("reported_not_gated")
        if info:
            sp, alt = info["nominal_spread_with_envelope"], info["joint_space_ankle_gains"]
            lines.append(f"   reported, not gated: {sp['runs']} starts offset <= {1e3 * sp['max_offset_m_s']:g} mm/s: "
                         f"median {sp['median']:.1f} strides ({sp['min']:.1f}-{sp['max']:.1f}); joint-space ankle "
                         f"gains: {alt['nominal_with_envelope']['strides']:.1f} strides, median "
                         f"{alt['nominal_spread_with_envelope']['median']:.1f} "
                         f"({alt['nominal_spread_with_envelope']['min']:.1f}-"
                         f"{alt['nominal_spread_with_envelope']['max']:.1f})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "gates", *LATER):
        p = sub.add_parser(name)
        p.add_argument("--body", action="append", choices=sorted(BODIES), required=name != "all")
        p.add_argument("--out", required=True, help="run directory, e.g. stage1/runs/<run>")
        p.add_argument("--seed", type=int, default=None, help="tuning or training seed (later steps)")
        if name == "prepare":
            p.add_argument("--no-gates", action="store_true", help="skip G1 and G2")
            p.add_argument("--refs", default=None, help="comma-separated reference names (default: all)")
        if name == "gates":
            p.add_argument("--only", choices=("g1", "g2"), default=None)
        if name == "all":
            p.add_argument("--cap-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd in LATER:
        raise NotImplementedError(f"'{a.cmd}' is {LATER[a.cmd]}, not implemented yet; this package implements plan "
                                  "section 3 steps 2 and 3 (prepare: plant and references, gates G1 and G2)")
    status = 0
    for body in a.body:
        out = Path(a.out) / body
        t0 = time.perf_counter()
        if a.cmd == "prepare":
            from . import refs

            _write(out / "plant.json", plant_doc(body))
            print(f"{body}: plant.json written", flush=True)
            names = a.refs.split(",") if a.refs else None
            doc = refs.prepare(body, out, names)
            print(f"{body}: {len(doc['references'])} references, records match: {doc['matches_records']}", flush=True)
            if a.no_gates:
                continue
        doc = run_gates(body, out, getattr(a, "only", None))
        print(f"{body} ({time.perf_counter() - t0:.0f} s)\n{summary(doc)}", flush=True)
        status |= 0 if doc["pass"] else 1
    return status


if __name__ == "__main__":
    sys.exit(main())
