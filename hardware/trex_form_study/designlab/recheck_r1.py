"""Round one's sized models at round two's walk accuracy (decision log F14).

Round one's reference walks stopped the per-frame centre-of-mass correction at 2e-4 m and differenced the
stride boundary with an approximate root-x unwrap, so the multibody-ZMP correction did not converge.  This
rebuilds each round-one candidate exactly from its Stage A record (SHA-256 of the MJCF is not re-checked here;
``round1.py`` does that), recomputes ``sizing.requirements`` with ``gait.set_accuracy("r2")`` and reports,
against the record:

* the single-support CoP margin of both walks and the walk-feasibility flag (CoP within 10 mm of the pad);
* the final multibody-ZMP error of both walks;
* each group's requirement ratio (round-two accuracy / round one) for the continuous, peak and speed
  requirements;
* the leg drives ``sizing.select`` would pick from the converged requirements at the record's mass, layout
  and bus preference: one re-selection, not a re-sizing (a changed part would change the mass and could
  change other choices on the next iteration).

Run from hardware/trex_form_study:
    python -m designlab.recheck_r1 --out round2/round1_recheck.jsonl [ids...]
    python -m designlab.recheck_r1 --summarize round2/round1_recheck.jsonl --json round2/round1_recheck.json
The first form appends one JSON line per candidate and skips candidates already in the file.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
#: Stage A's walk-infeasible candidates and those whose nominal walk left the pad by less than 10 mm
#: (SCREENING.md section 3), checked first.
FIRST = ("60-M-H-20", "75-M-S-10-belly", "60-W-S-00", "60-N-S-00", "60-M-H-00", "100-M-S-10-compact", "75-N-S-00",
         "120-W-S-10", "120-W-S-00", "60-W-S-20", "60-W-T-10", "75-N-H-00")


def recheck_one(cid: str, models: Path, stage_a: Path = STUDY / "round1" / "stage_a") -> dict:
    from . import gait, round1, sizing
    from .params import parse
    from .pipeline import build_candidate

    gait.set_accuracy("r2")
    t0 = time.time()
    rec = json.loads((Path(stage_a) / "candidates" / f"{cid}.json").read_text())
    if not rec.get("ok", True):
        return dict(id=cid, skipped="failed in round one")
    try:
        sel = dict(rec["selection"])
        sel["mass_guess"] = round1._final_mass_guess(rec)
        res = build_candidate(parse(cid), Path(models), selection=sel, B=rec["B"], write=False)
        M, q0, H, lay = res["model"], res["q_home"], res["design"].sk.H, res["design"].cand.leg_layout
        q = sizing.requirements(M, q0, H, lay)
        pick = sizing.select(q, mass=M.mass(), layout=lay)
        r1sel = {g: rec["margins"][g]["actuator"] for g in rec["margins"]}
        ratio = {g: {k: q["groups"][g][k] / rec["margins"][g][k2]
                     for k, k2 in (("cont", "cont_req"), ("peak_req", "peak_req"), ("speed_req", "speed_req"))
                     if rec["margins"][g].get(k2)}
                 for g in sizing.GROUPS if g in rec["margins"]}
        walks = ("nominal", "brisk")
        return dict(id=cid, walk_feasible_r1=rec["walk_feasible"], walk_feasible_r2=q["walk_feasible"],
                    cop_r1={k: rec["gaits"][k]["min_cop_margin_single"] for k in walks},
                    cop_r2={k: q["gaits"][k]["min_cop_margin_single"] for k in walks},
                    zmp_r1={k: rec["gaits"][k]["gait"]["zmp_error_history_m"][-1] for k in walks},
                    zmp_r2={k: q["gaits"][k]["gait"]["zmp_error_history_m"][-1] for k in walks},
                    friction_r1={k: rec["gaits"][k]["max_friction_ratio"] for k in walks},
                    friction_r2={k: q["gaits"][k]["max_friction_ratio"] for k in walks},
                    ik_ok_r2={k: q["gaits"][k]["ik_ok"] for k in walks},
                    ratio=ratio, r1_selection=r1sel, r2_pick={g: pick["choice"].get(g) for g in sizing.GROUPS},
                    r2_bus=pick["bus"], r1_bus=rec["selection"].get("bus"),
                    changed=[g for g in sizing.GROUPS if pick["choice"].get(g) != r1sel.get(g)],
                    seconds=time.time() - t0)
    except Exception as e:  # keep failures on record
        return dict(id=cid, error=repr(e))


def summarize(rows: list[dict]) -> dict:
    """Counts and ranges over the rechecked candidates (the numbers decision log F14 quotes)."""
    ok = [r for r in rows if "error" not in r and "skipped" not in r]
    groups = sorted({g for r in ok for g in r["ratio"]})

    def rng(vals):
        vals = [v for v in vals if v is not None]
        return [min(vals), max(vals)] if vals else None

    ratio = {g: {k: rng([r["ratio"][g].get(k) for r in ok if g in r["ratio"]]) for k in ("cont", "peak_req", "speed_req")}
             for g in groups}
    changed = [r for r in ok if r["changed"]]
    by_group = {}
    for r in changed:
        for g in r["changed"]:
            by_group[g] = by_group.get(g, 0) + 1
    return dict(
        n=len(rows), n_checked=len(ok), errors=[r["id"] for r in rows if "error" in r],
        walk_infeasible_r1=sorted(r["id"] for r in ok if r["walk_feasible_r1"] is False),
        walk_infeasible_r2=sorted(r["id"] for r in ok if r["walk_feasible_r2"] is False),
        nominal_cop_margin_mm_r1=[1e3 * v for v in rng([r["cop_r1"]["nominal"] for r in ok])],
        nominal_cop_margin_mm_r2=[1e3 * v for v in rng([r["cop_r2"]["nominal"] for r in ok])],
        brisk_cop_margin_mm_r1=[1e3 * v for v in rng([r["cop_r1"]["brisk"] for r in ok])],
        brisk_cop_margin_mm_r2=[1e3 * v for v in rng([r["cop_r2"]["brisk"] for r in ok])],
        brisk_cop_off_pad_r1=sum(1 for r in ok if r["cop_r1"]["brisk"] < 0),
        brisk_cop_off_pad_r2=sum(1 for r in ok if r["cop_r2"]["brisk"] < 0),
        zmp_error_mm_r1=[1e3 * v for v in rng([max(r["zmp_r1"].values()) for r in ok])],
        zmp_error_mm_r2=[1e3 * v for v in rng([max(r["zmp_r2"].values()) for r in ok])],
        ik_ok_r2_all=all(all(r["ik_ok_r2"].values()) for r in ok),
        requirement_ratio_r2_over_r1=ratio,
        reselection_changes=len(changed), reselection_changes_by_group=by_group,
        reselection_changed_ids={r["id"]: r["changed"] for r in changed},
        bus_changes=sorted(r["id"] for r in ok if r["r2_bus"] != r["r1_bus"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--out", default="round2/round1_recheck.jsonl")
    ap.add_argument("--models", default="/tmp/recheck_r1_models")
    ap.add_argument("--summarize", default=None, help="a JSON-lines file to summarize instead of running")
    ap.add_argument("--json", default="round2/round1_recheck.json")
    a = ap.parse_args()
    if a.summarize:
        rows = [json.loads(line) for line in Path(a.summarize).read_text().splitlines() if line.strip()]
        rows = sorted({r["id"]: r for r in rows}.values(), key=lambda r: r["id"])
        Path(a.json).write_text(json.dumps(dict(summary=summarize(rows), rows=rows), indent=1))
        print(json.dumps(summarize(rows), indent=1))
        return
    import designlab  # noqa: F401

    out = Path(a.out)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()} if out.exists() else set()
    stage = STUDY / "round1" / "stage_a" / "candidates"
    ids = a.ids or (list(FIRST) + sorted(Path(f).stem for f in glob.glob(str(stage / "*.json")) if Path(f).stem not in FIRST))
    for cid in ids:
        if cid in done:
            continue
        doc = recheck_one(cid, Path(a.models))
        with out.open("a") as f:
            f.write(json.dumps(doc) + "\n")
        print(cid, doc.get("walk_feasible_r1"), doc.get("walk_feasible_r2"), doc.get("changed"), flush=True)


if __name__ == "__main__":
    main()
