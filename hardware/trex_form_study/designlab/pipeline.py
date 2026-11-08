"""Build one candidate end to end: model, stance keyframe, ledger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import catalog, cost, mjcf, power, sizing, statics
from . import hardware as hw
from .build import Design, solve_B
from .geometry import quat_from_matrix
from .params import Candidate


def design_qpos(design: Design, M: statics.Model) -> np.ndarray:
    q = np.zeros(M.m.nq)
    q[0:3] = design.sk.origin
    q[3:7] = quat_from_matrix(design.sk.R_p)
    return q


def build_candidate(cand: Candidate, out_dir: Path, selection: dict | None = None, B: float | None = None,
                    write: bool = True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    B = B if B is not None else solve_B(cand)
    design = Design(cand, B, selection)
    rows = mjcf.ledger(design)
    xml = out_dir / f"{cand.id}.xml"
    mjcf.write_mjcf(design, xml, list(rows))
    M = statics.Model.load(str(xml))
    q0 = design_qpos(design, M)
    q_home, ok, dx = statics.solve_stance(M, q0, 0.5 * design.hip["foot_spacing"])
    rows2 = mjcf.ledger(design)
    info = mjcf.write_mjcf(design, xml, rows2, home_qpos=q_home)
    M = statics.Model.load(str(xml))
    M.forward(q_home)
    crown = crown_height(M)
    summary = dict(id=cand.id, B=B, H=design.sk.H, mass=M.mass(), crown=crown, stance_ok=bool(ok), stance_dx=dx,
                   xml=str(xml), xml_sha256=info["sha256"])
    if write:
        mjcf.write_ledger(design, rows2, out_dir / f"{cand.id}.json", extra=dict(summary=summary))
    return dict(design=design, rows=rows2, model=M, q_home=q_home, summary=summary)


def crown_height(M: statics.Model) -> float:
    """Highest point of the skull covering (vertex-exact) at the current pose."""
    import mujoco

    m, d = M.m, M.d
    g = statics.gid(m, "cover_skull")
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = m.mesh_vert[a : a + n]
    R = d.geom_xmat[g].reshape(3, 3)
    w = d.geom_xpos[g] + v @ R.T
    del mujoco
    return float(w[:, 2].max())


def dump(obj, path: Path):
    path.write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))


#: Oscillation between buses (24 V <-> 48 V): settle on the higher bus and re-select (True), instead of the
#: round-one rule, which kept the stronger part of each group and the larger pack whatever its voltage and so could
#: pair a 12S pack with a 24 V bus (two round-one records, 100-M-S-10-serial and 120-N-T-00; decision log F17).
#: False reproduces round one; the round-two runners set True.
BUS_CONSISTENT_OSCILLATION = False


def size_candidate(cand: Candidate, out_dir: Path, max_iter: int = 8, selection: dict | None = None,
                   verbose: bool = False) -> dict:
    """Build, compute loads, re-select actuators and battery, rebuild until the selection is stable.

    Stopping rules (recorded in ``sizing_history``):
    * converged: same selection twice and installed mass within 5 %;
    * oscillation: a selection recurs (A -> B -> A); every group then keeps the stronger of the parts
      it alternated between (higher sustained torque), which is conservative;
    * mass spiral: installed mass grew more than 8 % per iteration for three iterations running, or
      reached twice the first build's mass.  The candidate is then reported as not converged;
    * otherwise the last selection after ``max_iter`` iterations, flagged as not converged.
    """
    sel = dict(selection or hw.default_selection(cand.height))
    B = solve_B(cand)
    history = []
    seen = []
    for it in range(max_iter):
        res = build_candidate(cand, out_dir, selection=sel, B=B, write=False)
        M, q_home, design = res["model"], res["q_home"], res["design"]
        req = sizing.requirements(M, q_home, design.sk.H, cand.leg_layout)
        new = sizing.select(req, mass=res["summary"]["mass"], layout=cand.leg_layout)
        nom = req["runs"]["nominal"]["idr"]
        ms = sizing._motor_space(M.names, nom["tau"], nom["omega"], cand.leg_layout)
        trial = dict(sel)
        trial["joints"] = {g: new["choice"][g] for g in hw.JOINT_GROUPS}
        trial["neck"], trial["tail"], trial["bus"] = new["choice"]["neck"], new["choice"]["tail"], new["bus"]
        trial["mass_guess"] = res["summary"]["mass"]
        d_tmp = Design(cand, B, trial) if trial != sel else design
        pw = power.walking_power(d_tmp, req, ms)
        bat = power.select_battery(new["bus"], pw["total_w"], pw["peak_drives_w"] + pw["aux"]["input_w"])
        trial["battery"] = bat["pack"]
        key = (tuple(sorted(trial["joints"].items())), trial["neck"], trial["tail"], trial["bus"], trial["battery"])
        prev = (tuple(sorted(sel.get("joints", {}).items())), sel.get("neck"), sel.get("tail"), sel.get("bus"),
                sel.get("battery"))
        history.append(dict(iteration=it, mass=res["summary"]["mass"], selection=dict(joints=trial["joints"],
                            neck=trial["neck"], tail=trial["tail"], bus=trial["bus"], battery=trial["battery"]),
                            walk_w=pw["total_w"], notes=new["notes"]))
        if verbose:
            print(cand.id, "iter", it, "mass %.2f" % res["summary"]["mass"], trial["joints"], trial["battery"],
                  "%.0f W" % pw["total_w"])
        converged = key == prev and abs(trial["mass_guess"] - sel.get("mass_guess", 0)) < 0.05 * trial["mass_guess"]
        masses = [h["mass"] for h in history]
        growth = [b / a - 1 for a, b in zip(masses, masses[1:], strict=False)]
        spiral = (len(growth) >= 3 and all(g > 0.08 for g in growth[-3:])) or masses[-1] > 2.0 * masses[0]
        if spiral:
            history[-1]["notes"] = history[-1]["notes"] + [
                "sizing stopped: mass spiral (installed mass kept growing as heavier actuators raised the loads); "
                "not converged"]
            break
        if key != prev and key in seen[:-1]:
            # A -> B -> A: keep the stronger part of each alternating group
            cyc = [h["selection"] for h in history[seen.index(key):]]
            acts = catalog.actuators()
            buses = {c["bus"] for c in cyc}
            if len(buses) > 1 and BUS_CONSISTENT_OSCILLATION:
                # the selection alternated between buses: settle on the higher bus, re-select every part for it
                # from the current loads, and take the battery for that bus (parts and pack must suit one bus)
                bus = max(buses)
                pick = sizing.select(req, mass=res["summary"]["mass"], layout=cand.leg_layout, prefer_voltage=(bus,))
                strong = dict(trial, joints={g: pick["choice"][g] for g in hw.JOINT_GROUPS}, neck=pick["choice"]["neck"],
                              tail=pick["choice"]["tail"], bus=pick["bus"])
                d_s = Design(cand, B, strong)
                pw_s = power.walking_power(d_s, req, ms)
                strong["battery"] = power.select_battery(pick["bus"], pw_s["total_w"],
                                                         pw_s["peak_drives_w"] + pw_s["aux"]["input_w"])["pack"]
                history[-1]["notes"] = history[-1]["notes"] + [
                    f"selection oscillated between buses; settled on {pick['bus']:.0f} V and re-selected every part"]
                sel = strong
                break
            joints = {}
            for g in hw.JOINT_GROUPS:
                opts = {c["joints"][g] for c in cyc}
                joints[g] = max(opts, key=lambda a: (acts[a].sustained, acts[a].peak))
            strong = dict(trial, joints=joints)
            for g in ("neck", "tail"):
                opts = {c[g] for c in cyc}
                strong[g] = max(opts, key=lambda a: (acts[a].sustained, acts[a].peak))
            strong["battery"] = max({c["battery"] for c in cyc}, key=lambda b: power.catalog.part(b).capacity_wh)
            history[-1]["notes"] = history[-1]["notes"] + [
                "selection oscillated; each group keeps the stronger of the alternating parts"]
            sel = strong
            break
        seen.append(key)
        sel = trial
        if converged:
            break
    else:
        history[-1]["notes"] = history[-1]["notes"] + [f"not converged after {max_iter} iterations"]
    # final build and evaluation; if the final loads exceed a part's planning limits, step that group up
    for repair in range(3):
        res = build_candidate(cand, out_dir, selection=sel, B=B, write=False)
        M, q_home, design = res["model"], res["q_home"], res["design"]
        req = sizing.requirements(M, q_home, design.sk.H, cand.leg_layout)
        bumped = False
        for g, r in req["groups"].items():
            a = design.act(g)
            bus = sel["bus"] if a.box is None else None
            if sizing.meets(a, r, bus, g):
                continue
            kind = "servo" if a.box is not None else "leg"
            pool = sizing.pool(kind, sel["bus"] if kind == "leg" else None)
            better = [b for b in pool if sizing.meets(b, r, sel["bus"] if kind == "leg" else None, g)]
            if better:
                if g in hw.JOINT_GROUPS:
                    sel["joints"] = dict(sel["joints"], **{g: better[0].id})
                else:
                    sel[g] = better[0].id
                bumped = True
        if not bumped:
            break
        history.append(dict(iteration=f"repair{repair}", selection=dict(joints=sel["joints"], neck=sel["neck"],
                                                                      tail=sel["tail"]), notes=["final-load repair"]))
    nom = req["runs"]["nominal"]["idr"]
    ms = sizing._motor_space(M.names, nom["tau"], nom["omega"], cand.leg_layout)
    pw = power.walking_power(design, req, ms)
    pack = power.catalog.part(sel["battery"])
    rt = power.runtime(pack, pw["total_w"])
    margins = {}
    for g, r in req["groups"].items():
        a = design.act(g)
        spd = a.speed_at(sel["bus"] if a.box is None else None)
        margins[g] = dict(actuator=a.id, sustained=a.sustained, peak=a.peak, speed=spd,
                          cont_req=r["cont"], peak_req=r["peak_req"], speed_req=r["speed_req"],
                          cont_ratio=r["cont"] / a.sustained, peak_ratio=r["peak_req"] / a.peak,
                          speed_ratio=r["speed_req"] / max(spd, 1e-6), rms=r["rms"], hold=r["hold"],
                          double_support=r["double_support"])
    evaluation = dict(requirements={g: r for g, r in req["groups"].items()}, holds=req["holds"], gaits=req["gaits"],
                      walk_feasible=req["walk_feasible"],
                      margins=margins, power=pw, runtime=rt, sizing_history=history,
                      cost=cost.estimate(design, res["rows"], sel))
    mjcf.write_ledger(design, res["rows"], out_dir / f"{cand.id}.json", extra=dict(summary=res["summary"],
                                                                                    evaluation=evaluation))
    return dict(res, requirements=req, evaluation=evaluation, selection=sel)
