"""Command line for Stage 1 (``STAGE1_PLAN.md`` section 7).

Run from hardware/trex_form_study:

  python -m designlab.stage1 prepare --body 60-N-T-10 [--body ...] --out stage1/runs/<run> [--no-gates]
                                    [--parts plant,refs,transitions,stops] [--transitions start,stop,...]
  python -m designlab.stage1 gates   --body 60-N-T-10 --out stage1/runs/<run> [--only g1|g2|g3|g4]

``prepare`` writes, per body, ``<out>/<body>/plant.json`` (gains, envelopes, ranges, sensors, floors, hashes,
library versions, commit), ``refs.json`` with ``refs/<name>.npz`` (``refs``), ``transitions.json`` with
``transitions/<name>.npz`` (``transitions``: starts, stops, speed changes and turn entries and exits) and, unless
``--no-gates``, ``gates.json`` (``gates``: G1-G4 of plan section 3; G4 runs the tracker R0-R2 of ``control``).
``--parts`` limits ``prepare`` to some of its products (the others must exist for the later ones); ``stops`` re-reads
the saved stop transitions' stops under ``STAGE1_PLAN.md`` decision 18 (``transitions.reread_stops``) without
regenerating them.  ``gates`` reruns the gates (or one) on the saved products.  Every gate record carries the
provenance of the code that made it (``plant.provenance``: commit, dirty flag, the hash of the designlab/stage1
sources).  G4 is judged at the tracker's tuning start with the panel rule (decisions 16 and 17); a G4 record made at
the build's earlier defaults is kept in ``G4.evidence.earlier_defaults``.
``--ankle-gains``, ``--ankle-armature`` and ``--g2-rule`` set the choices G2 depends on; the defaults are the ones
``STAGE1_PLAN.md`` decided during the build (decisions 11-13: joint-space ankle gains, the linkage's armature, the
20-start panel), and the others stay available for comparison.  They are recorded in plant.json and gates.json
(``G2.decided`` says whether a run follows the decisions).  The exit status is 1 if a gate fails (plan section 3:
stop).  ``tune``, ``train``, ``eval``, ``sweep``, ``report`` and ``all`` are later steps of the plan (6-9) and are not
implemented yet.
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


GATES = ("g1", "g2", "g3", "g4")


def earlier_g4_evidence(prev: dict | None, gates_path: Path) -> dict | None:
    """The G4 record of the build's earlier defaults, kept as evidence (``STAGE1_PLAN.md`` decision 17): carried over
    from a record that already keeps it, or taken from a record made before decisions 15-20 (no ``judged_at``) with
    a note of the code it was committed with (that record names none)."""
    from . import plant

    if not prev:
        return None
    if prev.get("judged_at"):
        return (prev.get("evidence") or {}).get("earlier_defaults")
    commit = None
    try:
        import subprocess

        commit = subprocess.run(["git", "-C", str(plant.STUDY), "log", "-1", "--format=%H", "--",
                                 str(Path(gates_path).resolve())], capture_output=True, text=True,
                                check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        pass
    return dict(note="G4 as the build first ran it, before decisions 15-20: the tracker at its earlier (untuned) "
                     "defaults (control.EARLIER_DEFAULTS: heading 0.5 / 0.1, phase rate 2.0 / 1.0), no ankle cap, "
                     "the inward step clamped at the turn gap less 1 mm, the estimator's stance odometry at the foot's "
                     "centre of mass, every fall labelled by the episode-level counterfactual, judged on the "
                     "unperturbed runs; kept as evidence (decision 17)",
                record_names_code=False, committed_in=commit,
                committed_sources_sha256=plant.sources_sha256_at(commit) if commit else None,
                sources_rule=plant.SOURCES_RULE, record=prev)


def _provenanced(fn, *args, **kw) -> dict:
    """``fn(*args, **kw)`` with the provenance of the code that ran it (``plant.provenance``, taken before the run),
    and whether the Stage 1 sources were still the same when it finished (``unchanged_during_run``)."""
    from . import plant

    prov = plant.provenance()
    out = fn(*args, **kw)
    prov["stage1_sources"]["unchanged_during_run"] = plant.sources_sha256() == prov["stage1_sources"]["sha256"]
    return dict(out, provenance=prov)


def run_gates(body: str, out: Path, only: str | None = None, plant_kw: dict | None = None,
              rule: str | None = None) -> dict:
    """G1-G4 (or one of them) into ``<out>/gates.json``; ``rule`` is G2's (default: ``gates.RULES[0]``, the plan's
    decision 13).  G3 and G4 run on the decided plant (``plant.build``'s defaults).  Each gate record gets the
    provenance of the code that ran it (``_provenanced``).  ``pass`` needs every gate recorded so far to pass (and G1
    and G2 on the same plant)."""
    from . import gates

    plant_kw = plant_kw or {}
    rule = rule or gates.RULES[0]
    path = out / "gates.json"
    doc = json.loads(path.read_text()) if path.exists() else dict(body=body)
    if only in (None, "g1"):
        doc["G1"] = _provenanced(gates.g1, body, out, **plant_kw)
    if only in (None, "g2"):
        doc["G2"] = _provenanced(gates.g2, body, out, rule=rule, **plant_kw)
    if only in (None, "g3"):
        doc["G3"] = _provenanced(gates.g3, body, out)
    if only in (None, "g4"):
        evidence = earlier_g4_evidence(doc.get("G4"), path)
        doc["G4"] = dict(_provenanced(gates.g4, body, out), evidence=dict(earlier_defaults=evidence))
    same = (doc.get("G1", {}).get("plant_options") is not None
            and doc.get("G1", {}).get("plant_options") == doc.get("G2", {}).get("plant_options"))
    doc["same_plant_options"] = same
    passes = [doc[g]["pass" if g in ("G1", "G2", "G4") else "pass_"] for g in ("G1", "G2", "G3", "G4") if g in doc]
    plant_gates = "G1" in doc or "G2" in doc  # G1 and G2 must then both be there, on the same plant
    doc["pass"] = bool(passes and all(passes) and (same or not plant_gates))
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
        per = sr.get("per_process_all_median", sr.get("per_process_mean"))
        lines.append(f"G1 {'pass' if g1['pass'] else 'FAIL'}: hashes {hashes}; {st}; tendon map err "
                     f"{max(tm['max_err_joint_torque_Nm'], tm['max_err_motor_speed_rad_s']):.1e}; "
                     f"repeat {g1['repeatability']['ok']}; step rate (median of {sr.get('samples_per_process', 1)} x "
                     f"{sr.get('steps_per_sample', sr.get('steps'))} steps) {sr['single_process']:.0f} steps/s alone, "
                     f"{per:.0f} per process x {sr['processes']} (lowest sample "
                     f"{sr.get('lowest_sample', min(sr.get('per_process', [per]))):.0f})")
    if g2:
        o = g2["plant_options"]
        pnl = g2["nominal_panel_with_envelope"]
        lines.append(f"G2 {'pass' if g2['pass'] else 'FAIL'} (rule {g2['rule']}; ankle gains {o['ankle_gains']}, "
                     f"armature {o['ankle_armature']}): references match {g2['all_match']} (nominal "
                     f"{g2['nominal_speed_m_s']} m/s, CoP margin {g2['nominal_min_cop_margin_single_m']} m); nominal "
                     f"panel median {pnl['median']:.1f} strides ({pnl['runs_at_least_min_strides']}/{pnl['runs']} at "
                     f"least {g2['min_strides']:g}, range {pnl['min']:.1f}-{pnl['max']:.1f}); unperturbed run "
                     f"{g2['nominal_strides_with_envelope']:.1f} strides with the envelope, "
                     f"{g2['nominal_strides_without_envelope']:.1f} without (design study, screened MJCF: "
                     f"{g2['design_study_strides_screened_model']})")
        others = ", ".join(f"{name} {v['with_envelope']['strides']:.1f}" for name, v in g2["replay"].items()
                           if name != "nominal")
        lines.append(f"   other replays with the envelope (strides): {others}")
        info = g2["reported_not_gated"]
        ch = "; ".join(f"{name}: {c['nominal_strides']:.1f}, panel median {c['panel']['median']:.1f} "
                       f"({c['panel']['runs_at_least_min_strides']}/{c['panel']['runs']})"
                       for name, c in info["plant_choices"].items())
        lines.append(f"   plant choices (nominal strides unperturbed, panel): {ch}")
        dec = g2.get("decided")
        if dec:
            lines.append("   decided (STAGE1_PLAN.md): " + "; ".join(
                f"({d['decision']}) {d['choice']} = {d['taken']}"
                + ("" if d["follows"] else f", NOT FOLLOWED: this run used {d['used']}") for d in dec["decisions"]))
    g3 = doc.get("G3")
    if g3:
        lines.append(f"G3 {'pass' if g3['pass_'] else 'FAIL'}: {len(g3['transitions'])} transitions; worst CoM "
                     f"{g3['worst_com']['mm']:.3f} mm ({g3['worst_com']['transition']}), worst ZMP "
                     f"{g3['worst_zmp']['mm']:.3f} mm ({g3['worst_zmp']['transition']}); failed: "
                     f"{', '.join(g3['failed']) or 'none'} (fall back to amplitude ramps: {', '.join(g3['fallback']) or 'none'})")
        st = g3.get("stops", {}).get("stops", {})
        if st:
            lines.append("   stops (decision 18, strides from the command): " + "; ".join(
                f"{n} forward {'-' if s.get('forward_stopped_strides') is None else format(s['forward_stopped_strides'], '.3f')}"
                f" (within {s.get('within')}), rest {s.get('at_rest_strides', float('nan')):.3f}" for n, s in st.items()))
    g4 = doc.get("G4")
    if g4 and g4.get("judged_at"):
        parts = []
        for key, g in g4["gated"].items():
            u, pnl = g["unperturbed"], g["panel"]
            parts.append(f"{key} panel median {pnl['median']:.1f} (>= {g['threshold_strides']:g}: {g['panel_pass']}; "
                         f"{pnl['runs_at_least_threshold']}/{pnl['runs']} reach it, {pnl['survived']} survive); "
                         f"unperturbed {u['strides']:.1f}")
        lines.append(f"G4 {'pass' if g4['pass'] else 'FAIL'} (panel rule, at the tuning start): " + "; ".join(parts))
        for key, a in (g4.get("attribution") or {}).items():
            tests = ", ".join(f"{r} p {t['sign_test']['p']:.3g} ({t['survived']} survive)" for r, t in a["lifted"].items())
            lines.append(f"   {key}: {a['label']}; candidates {', '.join(a['candidates']) or 'none'}; lifted: "
                         f"{tests or 'nothing relaxable'}; null p {a['null']['sign_test']['p']:.3g}")
        lines.append("   R0: " + ", ".join(f"{n} {r['strides']:.1f}" for n, r in g4["R0"].items()))
        for mode, rows in g4["reported"].items():
            lines.append(f"   {mode}: " + ", ".join(f"{n} {r['strides']:.1f}" for n, r in rows.items()))
        for mode, r in g4["stance_start_stop"].items():
            stop = (r.get("stops") or [{}])[0]
            lines.append(f"   {mode} start-walk-stop: {r['strides']:.1f} strides, fell {r['fell']}; stop "
                         + (f"forward {stop.get('forward_stopped_strides')} strides (within {stop.get('within')}), rest "
                            f"{stop.get('at_rest_strides')}" if stop.get("reached") else "not reached"))
    return "\n".join(lines)


def main(argv=None) -> int:
    from . import gates, plant

    ap = argparse.ArgumentParser(prog="python -m designlab.stage1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "gates", *LATER):
        p = sub.add_parser(name)
        p.add_argument("--body", action="append", choices=sorted(BODIES), required=name != "all")
        p.add_argument("--out", required=True, help="run directory, e.g. stage1/runs/<run>")
        p.add_argument("--seed", type=int, default=None, help="tuning or training seed (later steps)")
        if name in ("prepare", "gates"):
            p.add_argument("--ankle-gains", choices=plant.ANKLE_GAINS, default=plant.ANKLE_GAINS[0],
                           help="ankle hold-servo gain: joint space (default, STAGE1_PLAN.md decision 11) or per "
                                "motor (for comparison)")
            p.add_argument("--ankle-armature", choices=plant.ANKLE_ARMATURE, default=plant.ANKLE_ARMATURE[0],
                           help="ankle armature: the linkage's, on the motor tendons (default, decision 12) or as "
                                "screened (for comparison)")
            p.add_argument("--g2-rule", choices=gates.RULES, default=gates.RULES[0],
                           help="G2 replay criterion: the median of the 20-start panel (default, decision 13) or the "
                                "one unperturbed run (the plan's original wording)")
        if name == "prepare":
            p.add_argument("--no-gates", action="store_true", help="skip the gates")
            p.add_argument("--refs", default=None, help="comma-separated reference names (default: all)")
            p.add_argument("--parts", default="plant,refs,transitions",
                           help="comma-separated products to make: plant, refs, transitions (default), and stops "
                                "(re-read the saved stops, decision 18)")
            p.add_argument("--transitions", default=None, help="comma-separated transition names (default: all)")
        if name == "gates":
            p.add_argument("--only", choices=GATES, default=None)
        if name == "all":
            p.add_argument("--cap-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd in LATER:
        raise NotImplementedError(f"'{a.cmd}' is {LATER[a.cmd]}, not implemented yet; this package implements plan "
                                  "section 3 steps 2-5 (prepare: plant, references and transitions; gates G1-G4)")
    plant_kw = dict(ankle_gains=a.ankle_gains, ankle_armature=a.ankle_armature)
    status = 0
    for body in a.body:
        out = Path(a.out) / body
        t0 = time.perf_counter()
        if a.cmd == "prepare":
            from . import refs, transitions

            parts = set(a.parts.split(","))
            if parts - {"plant", "refs", "transitions", "stops"}:
                raise SystemExit(f"--parts: unknown {sorted(parts - {'plant', 'refs', 'transitions', 'stops'})}")
            if "plant" in parts:
                _write(out / "plant.json", plant_doc(body, plant_kw))
                print(f"{body}: plant.json written", flush=True)
            if "refs" in parts:
                names = a.refs.split(",") if a.refs else None
                doc = refs.prepare(body, out, names)
                print(f"{body}: {len(doc['references'])} references, records match: {doc['matches_records']}", flush=True)
            if "transitions" in parts:
                names = a.transitions.split(",") if a.transitions else None
                doc = transitions.prepare(body, out, names)
                print(f"{body}: {len(doc['transitions'])} transitions", flush=True)
            if "stops" in parts:
                doc = transitions.reread_stops(out)
                print(f"{body}: stops re-read (decision 18): " + ", ".join(
                    f"{n} {e['stop']['forward_stopped_strides']}" for n, e in doc["transitions"].items()
                    if e["kind"] == "stop"), flush=True)
            if a.no_gates:
                continue
        doc = run_gates(body, out, getattr(a, "only", None), plant_kw, a.g2_rule)
        print(f"{body} ({time.perf_counter() - t0:.0f} s)\n{summary(doc)}", flush=True)
        status |= 0 if doc["pass"] else 1
    return status


if __name__ == "__main__":
    sys.exit(main())
