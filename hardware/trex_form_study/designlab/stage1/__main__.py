"""Command line for Stage 1 (``STAGE1_PLAN.md`` section 7).

Run from hardware/trex_form_study:

  python -m designlab.stage1 prepare --body 60-N-T-10 [--body ...] --out stage1/runs/<run> [--no-gates]
  python -m designlab.stage1 gates   --body 60-N-T-10 --out stage1/runs/<run> [--only g1|g2]

``prepare`` writes, per body, ``<out>/<body>/plant.json`` (gains, envelopes, ranges, sensors, floors, hashes,
library versions, commit), ``refs.json`` with ``refs/<name>.npz`` (``refs``) and, unless ``--no-gates``,
``gates.json`` (``gates``: G1 and G2 of plan section 3).  ``gates`` reruns G1 and G2 on the saved references.
``--ankle-gains``, ``--ankle-armature`` and ``--g2-rule`` set the open choices G2 depends on (``gates.open_decisions``;
the defaults until the owner decides); they are recorded in plant.json and gates.json.  The exit status is 1 if a
gate fails (plan section 3: stop).  ``tune``, ``train``, ``eval``, ``sweep``, ``report`` and ``all`` are later steps
of the plan and are not implemented yet.
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
#: PPO env steps per second at 50 Hz on 2 ms physics are at most a tenth of the physics rate (plan section 3, G6a).
ENV_SUBSTEPS = 10
G6A_ENV_STEPS_PER_S = 600
COARSE_PILE_GRID_M = 0.008  # the training pile of plan section 2


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=float))


def _floor_rate(body: str, floor: str, plant_kw: dict, pile_grid: float | None = None, steps: int = 2000) -> dict:
    from . import plant

    t0 = time.perf_counter()
    p = plant.build(body, floor, pile_grid=pile_grid, **plant_kw)
    build_s = time.perf_counter() - t0
    p.reset(allow_unsettled=True)
    p.hold()
    t1 = time.perf_counter()
    p.step(steps)
    rate = steps / (time.perf_counter() - t1)
    s = p.meta["settle"]
    return dict(params=p.meta["floor_params"], info=p.meta["floor_info"], settle=s, stands=bool(s["settled"]),
                stand_status="stands" if s["settled"] else f"does not stand: the zero-command stance {s['status']} "
                f"after {s['seconds']:.1f} s", build_s=build_s, steps_per_s_zero_command=rate,
                env_steps_per_s_bound_50hz=rate / ENV_SUBSTEPS,
                below_g6a_env_rate=bool(rate / ENV_SUBSTEPS < G6A_ENV_STEPS_PER_S))


def plant_doc(body: str, plant_kw: dict | None = None) -> dict:
    """``plant.json``: the hard-floor plant in full, and every floor's parameters, settle, stand status and step rate
    (also on the coarse training pile), with the floors whose physics alone caps PPO below G6a's 600 env steps/s."""
    from .. import carpet
    from . import plant

    plant_kw = plant_kw or {}
    doc = plant.describe(plant.build(body, "hard", **plant_kw))
    doc["floors"] = {floor: _floor_rate(body, floor, plant_kw) for floor in plant.FLOORS}
    doc["floors_coarse_pile"] = {floor: _floor_rate(body, floor, plant_kw, COARSE_PILE_GRID_M) for floor in carpet.PILES}
    doc["throughput"] = dict(
        rule=f"physics steps/s with zero command / {ENV_SUBSTEPS} (50 Hz env steps on 2 ms physics) is an upper bound "
             f"on the PPO env rate; G6a asks for {G6A_ENV_STEPS_PER_S} per job",
        below_g6a={f"{floor} ({grid})": round(v["env_steps_per_s_bound_50hz"])
                   for grid, floors in (("4 mm probes", doc["floors"]), ("8 mm probes", doc["floors_coarse_pile"]))
                   for floor, v in floors.items() if v["below_g6a_env_rate"]})
    return doc


def run_gates(body: str, out: Path, only: str | None = None, plant_kw: dict | None = None,
              rule: str = "single") -> dict:
    from . import gates

    plant_kw = plant_kw or {}
    path = out / "gates.json"
    doc = json.loads(path.read_text()) if path.exists() else dict(body=body)
    if only in (None, "g1"):
        doc["G1"] = gates.g1(body, out, **plant_kw)
    if only in (None, "g2"):
        doc["G2"] = gates.g2(body, out, rule=rule, **plant_kw)
    same = (doc.get("G1", {}).get("plant_options") is not None
            and doc.get("G1", {}).get("plant_options") == doc.get("G2", {}).get("plant_options"))
    doc["same_plant_options"] = same
    doc["pass"] = bool(same and doc.get("G1", {}).get("pass") and doc.get("G2", {}).get("pass"))
    _write(path, doc)
    return doc


def summary(doc: dict) -> str:
    lines = []
    g1, g2 = doc.get("G1"), doc.get("G2")
    if g1:
        st = ", ".join(f"{f} {'stands' if s['stands'] else 'FAILS'}"
                       + (f" (tilt max {s['tilt_from_home_max_deg']:.2f} deg)" if "tilt_from_home_max_deg" in s else "")
                       for f, s in g1["stand"].items())
        sr = g1["step_rate"]
        hashes = all(h["matches_record"] and h["matches_plan"] for h in g1["hashes"].values())
        tm = g1["tendon_map"]
        lines.append(f"G1 {'pass' if g1['pass'] else 'FAIL'}: hashes {hashes}; {st}; tendon map err "
                     f"{max(tm['max_err_joint_torque_Nm'], tm['max_err_motor_speed_rad_s']):.1e}; "
                     f"repeat {g1['repeatability']['ok']}; {sr['single_process']:.0f} steps/s alone, "
                     f"{sr['per_process_mean']:.0f} per process x {sr['processes']}")
    if g2:
        o = g2["plant_options"]
        lines.append(f"G2 {'pass' if g2['pass'] else 'FAIL'} (rule {g2['rule']}; ankle gains {o['ankle_gains']}, "
                     f"armature {o['ankle_armature']}): references match {g2['all_match']} (nominal "
                     f"{g2['nominal_speed_m_s']} m/s, CoP margin {g2['nominal_min_cop_margin_single_m']} m); nominal "
                     f"replay {g2['nominal_strides_with_envelope']:.1f} strides with the envelope, "
                     f"{g2['nominal_strides_without_envelope']:.1f} without (design study, screened MJCF: "
                     f"{g2['design_study_strides_screened_model']}); panel median "
                     f"{g2['nominal_panel_with_envelope']['median']:.1f} "
                     f"({g2['nominal_panel_with_envelope']['runs_at_least_min_strides']}/"
                     f"{g2['nominal_panel_with_envelope']['runs']} at least {g2['min_strides']:g})")
        others = ", ".join(f"{name} {v['with_envelope']['strides']:.1f}" for name, v in g2["replay"].items()
                           if name != "nominal")
        lines.append(f"   other replays with the envelope (strides): {others}")
        info = g2["reported_not_gated"]
        ch = "; ".join(f"{name}: {c['nominal_strides']:.1f}, panel median {c['panel']['median']:.1f} "
                       f"({c['panel']['runs_at_least_min_strides']}/{c['panel']['runs']})"
                       for name, c in info["plant_choices"].items())
        lines.append(f"   plant choices (nominal strides unperturbed, panel): {ch}")
        lines.append("   open decisions: " + "; ".join(f"({d['id']}) {d['choice']} = {d['taken_until_decided']} until "
                                                       "decided" for d in g2["open_decisions"]))
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
        if name in ("prepare", "gates"):
            p.add_argument("--ankle-gains", choices=("motor", "joint"), default="motor",
                           help="ankle hold-servo gain: per motor (default) or joint space (open decision a)")
            p.add_argument("--ankle-armature", choices=("linkage", "screened"), default="linkage",
                           help="ankle armature: on the motor tendons (default) or as screened (open decision b)")
            p.add_argument("--g2-rule", choices=("single", "panel"), default="single",
                           help="G2 replay criterion: one unperturbed run (default, the plan's) or the seeded panel "
                                "(open decision c)")
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
    plant_kw = dict(ankle_gains=a.ankle_gains, ankle_armature=a.ankle_armature)
    status = 0
    for body in a.body:
        out = Path(a.out) / body
        t0 = time.perf_counter()
        if a.cmd == "prepare":
            from . import refs

            _write(out / "plant.json", plant_doc(body, plant_kw))
            print(f"{body}: plant.json written", flush=True)
            names = a.refs.split(",") if a.refs else None
            doc = refs.prepare(body, out, names)
            print(f"{body}: {len(doc['references'])} references, records match: {doc['matches_records']}", flush=True)
            if a.no_gates:
                continue
        doc = run_gates(body, out, getattr(a, "only", None), plant_kw, a.g2_rule)
        print(f"{body} ({time.perf_counter() - t0:.0f} s)\n{summary(doc)}", flush=True)
        status |= 0 if doc["pass"] else 1
    return status


if __name__ == "__main__":
    sys.exit(main())
