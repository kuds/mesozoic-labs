"""Load requirements per joint group and actuator selection.

Requirements are computed on the *built* model:
* static double-support torques at the nominal stance;
* the quasi-static single-support hold (``statics.single_support_hold``), both sides;
* a nominal reference walk (Fr 0.012) for RMS torque and speed, and a brisk walk (Fr 0.025) for
  peaks.  Ankle requirements are mapped to the two motors of the parallel linkage.

Selection rule (per group, same part on left and right):
    sustained planning torque  >= max(nominal-walk RMS, single-support hold)   (continuous)
    peak planning torque       >= 1.25 x brisk-walk peak                       (transient)
    usable speed on the bus    >= 1.15 x brisk-walk peak speed
Among parts that meet all three, the lowest score wins:
    score = mass_kg + COST_WEIGHT x price_usd / 1000 + packaging term
* COST_WEIGHT = 0.8 kg per $1000 (a $1000 saving is worth 0.8 kg); only parts with a sourced price.
* Packaging term (hip groups only): PACK_K x M x (dy_leg / d dimension) x dimension.  The hip-roll
  motor diameter, the thigh-twist motor diameter and the hip-pitch motor length set the hip-pitch
  spacing (``build.Design._build``), and each metre of half-spacing y_leg adds about
  m_leg g y_leg to the single-support roll and twist loads of the stance hip.  Sized at a sustained
  torque density of ~13 N·m/kg (48 V class), with m_leg ~ 0.22 M, that costs
  2 x 0.22 x 9.81 / 13 ~ 0.33 kg of actuator per metre of y_leg per kg of robot mass M: PACK_K = 0.33/m.
  It is a first-order estimate that leaves out the loop gain; the sizing loop then checks the result.
Bus: all leg drives share one bus.  A 6S (24 V class, 21.0-25.2 V over the usable charge) bus is tried
first, then 12S (48 V class, 42.0-50.4 V).  A drive is eligible if its sourced allowable supply window
covers the bus swing (``catalog.Actuator.accepts``); below its nominal voltage its usable speed is
derated by bus/nominal (``catalog.Actuator.speed_at``), and no credit is taken above it.
"""

from __future__ import annotations

import numpy as np

from . import catalog, gait, statics

GROUPS = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b", "neck", "tail")
JOINT_OF = {"hip_roll": "hip_roll", "hip_pitch": "hip_pitch", "hip_yaw": "hip_yaw", "knee": "knee"}
CONT_MARGIN = 1.0
COST_WEIGHT = 0.8  # kg per $1000
PEAK_MARGIN = 1.25
SPEED_MARGIN = 1.15

NOMINAL = gait.GaitSpec(froude=0.012, stride_rel=0.55, duty=0.60)
BRISK = gait.GaitSpec(froude=0.025, stride_rel=0.62, duty=0.58, lift_frac=0.11)
#: Extra prescribed walks whose peaks and speeds join the requirements, as the brisk walk's do (name -> GaitSpec).
#: Empty in rounds one and two.  Round two's what-if "sized for the animal-like walk" sets {"fr05": envelope.FAST}
#: (``python -m designlab.round2 --whatif fr05``).
EXTRA_WALKS: dict = {}


def _motor_space(names, tau, omega, layout: str):
    """Per-group motor torque/speed time series (both sides stacked)."""
    idx = {n: i for i, n in enumerate(names)}
    out = {}
    for g, j in JOINT_OF.items():
        cols = [idx[f"{s}_{j}"] for s in statics.SIDES]
        out[g] = (tau[:, cols].ravel(), omega[:, cols].ravel())
    ta, wa, tb, wb = [], [], [], []
    for s in statics.SIDES:
        tp, tr = tau[:, idx[f"{s}_ankle_pitch"]], tau[:, idx[f"{s}_ankle_roll"]]
        wp, wr = omega[:, idx[f"{s}_ankle_pitch"]], omega[:, idx[f"{s}_ankle_roll"]]
        if layout in ("tandem", "side"):  # parallel linkage, unit crank ratios
            ta.append(0.5 * (tp + tr)), tb.append(0.5 * (tp - tr))
            wa.append(wp + wr), wb.append(wp - wr)
        else:
            ta.append(tp), tb.append(tr), wa.append(wp), wb.append(wr)
    out["ankle_a"] = (np.concatenate(ta), np.concatenate(wa))
    out["ankle_b"] = (np.concatenate(tb), np.concatenate(wb))
    out["neck"] = (tau[:, idx["neck_pitch"]], omega[:, idx["neck_pitch"]])
    out["tail"] = (tau[:, idx["tail_yaw"]], omega[:, idx["tail_yaw"]])
    return out


def requirements(M: statics.Model, q_home: np.ndarray, H: float, layout: str) -> dict:
    names = M.names
    tau_ds, w_ds = statics.contact_torques(M, q_home, support=("left", "right"))
    holds = {}
    for s in statics.SIDES:
        best, cases = statics.single_support_hold(M, q_home, s, lift=0.05 * H)
        if best is not None:
            holds[s] = dict(best=True, tau=np.array(best["tau"]), margin=best["margin"], roll_deg=best["roll_deg"],
                            ik_ok=True)
        else:  # no reachable, balanced single-support pose: a finding, not a load case
            holds[s] = dict(best=False, tau=np.zeros(len(names)), margin=float("nan"), roll_deg=None, ik_ok=False)
    runs = {}
    for name, spec in (("nominal", NOMINAL), ("brisk", BRISK)) + tuple(EXTRA_WALKS.items()):
        tr = gait.generate(M, q_home, H, spec)
        idr = gait.inverse_dynamics(M, tr)
        runs[name] = dict(traj=tr, idr=idr, summary=gait.summarize(M, tr, idr))
    zeros = np.zeros((1, len(names)))
    ms_nom = _motor_space(names, runs["nominal"]["idr"]["tau"], runs["nominal"]["idr"]["omega"], layout)
    ms_brk = _motor_space(names, runs["brisk"]["idr"]["tau"], runs["brisk"]["idr"]["omega"], layout)
    ms_ext = [_motor_space(names, runs[k]["idr"]["tau"], runs[k]["idr"]["omega"], layout) for k in EXTRA_WALKS]
    ms_ds = _motor_space(names, tau_ds[None, :], zeros, layout)
    hold_tau = np.vstack([holds[s]["tau"] for s in statics.SIDES])
    ms_hold = _motor_space(names, hold_tau, np.zeros_like(hold_tau), layout)
    req = {}
    for g in GROUPS:
        t_nom, w_nom = ms_nom[g]
        t_brk, w_brk = ms_brk[g]
        rms = float(np.sqrt(np.mean(t_nom**2)))
        hold = float(np.max(np.abs(ms_hold[g][0])))
        ds = float(np.max(np.abs(ms_ds[g][0])))
        peak = float(max(np.max(np.abs(t_brk)), np.max(np.abs(t_nom)), hold, *(np.max(np.abs(x[g][0])) for x in ms_ext)))
        speed = float(max(np.max(np.abs(w_brk)), np.max(np.abs(w_nom)), *(np.max(np.abs(x[g][1])) for x in ms_ext)))
        req[g] = dict(rms=rms, hold=hold, double_support=ds, peak=peak, speed=speed,
                      cont=max(rms, hold) * CONT_MARGIN, peak_req=peak * PEAK_MARGIN, speed_req=speed * SPEED_MARGIN)
    walk_ok = runs["nominal"]["summary"]["min_cop_margin_single"] > -0.010
    return dict(groups=req, walk_feasible=bool(walk_ok),
                holds={s: dict(margin=holds[s]["margin"], feasible=holds[s]["best"],
                               roll_deg=holds[s]["roll_deg"]) for s in statics.SIDES},
                gaits={k: v["summary"] for k, v in runs.items()}, runs=runs)


BUSES = {24.0: (21.0, 25.2), 48.0: (42.0, 50.4)}  # nominal -> (low, high) over the usable charge
PACK_K = 0.33  # kg of actuator per metre of hip half-spacing per kg of robot (see module docstring)
#: d(y_leg)/d(dimension) per layout for the dimension of each hip group that sets the spacing:
#: hip_roll -> diameter, hip_yaw -> diameter, hip_pitch -> axial length (build.Design._build).
PACK_GEOMETRY = {"tandem": {"hip_roll": 0.5, "hip_yaw": 0.5, "hip_pitch": 0.5},
                 "side": {"hip_roll": 1.0, "hip_yaw": 0.5, "hip_pitch": 0.5},
                 "serial": {"hip_roll": 0.5, "hip_yaw": 0.5, "hip_pitch": 0.5}}


def pool(kind: str, voltage: float | None = None):
    """Selectable parts, cheapest-lightest first.  ``voltage`` is a bus nominal (24.0 or 48.0)."""
    acts = catalog.actuators().values()
    if kind == "servo":
        return sorted([a for a in acts if a.box is not None and a.price],
                      key=lambda a: (a.mass + COST_WEIGHT * a.price / 1000.0, a.mass))
    sel = [a for a in acts if a.box is None and a.price]
    if voltage is not None:
        lo, hi = BUSES[voltage]
        sel = [a for a in sel if a.accepts(lo, hi)]
    return sorted(sel, key=lambda a: (a.mass + COST_WEIGHT * a.price / 1000.0, a.mass))


#: Fraction of each planning limit a selected part may use (1.0 in rounds one and two).  The size study's
#: headroom rule sets 0.80: every requirement at most 0.80 of the part's sustained, peak and speed limit, the
#: headroom the 60 cm bodies have without it.
HEADROOM = 1.0


def meets(a: catalog.Actuator, r: dict, bus: float | None = None) -> bool:
    if HEADROOM == 1.0:
        return a.sustained >= r["cont"] and a.peak >= r["peak_req"] and a.speed_at(bus) >= r["speed_req"]
    return (HEADROOM * a.sustained >= r["cont"] and HEADROOM * a.peak >= r["peak_req"]
            and HEADROOM * a.speed_at(bus) >= r["speed_req"])


def score(a: catalog.Actuator, group: str, mass: float, layout: str) -> float:
    s = a.mass + COST_WEIGHT * a.price / 1000.0
    k = PACK_GEOMETRY.get(layout, PACK_GEOMETRY["tandem"]).get(group)
    if k:
        D, L = a.envelope
        s += PACK_K * mass * k * (L if group == "hip_pitch" else D)
    return s


def select(req: dict, mass: float = 15.0, layout: str = "tandem", prefer_voltage=(24.0, 48.0)) -> dict:
    groups = req["groups"]
    choice, notes = {}, []
    bus = None
    for v in prefer_voltage:
        ok = True
        tmp = {}
        for g in ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b"):
            cands = sorted([a for a in pool("leg", v) if meets(a, groups[g], v)],
                           key=lambda a: (score(a, g, mass, layout), a.mass))
            if not cands:
                ok = False
                break
            tmp[g] = cands[0].id
        if ok:
            choice.update(tmp)
            bus = v
            break
    if bus is None:
        # nothing meets every requirement: take the strongest 48 V part for the failing groups and flag it
        bus = 48.0
        for g in ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b"):
            cands = sorted([a for a in pool("leg", bus) if meets(a, groups[g], bus)],
                           key=lambda a: (score(a, g, mass, layout), a.mass))
            if cands:
                choice[g] = cands[0].id
            else:
                best = max(pool("leg", bus), key=lambda a: a.sustained)
                choice[g] = best.id
                notes.append(f"{g}: no catalog part meets cont {groups[g]['cont']:.1f} N·m / peak "
                             f"{groups[g]['peak_req']:.1f} N·m / {groups[g]['speed_req']:.1f} rad/s; strongest used")
    for g in ("neck", "tail"):
        cands = [a for a in pool("servo") if meets(a, groups[g])]
        if cands:
            choice[g] = cands[0].id
        else:
            leg = [a for a in pool("leg", bus) if meets(a, groups[g], bus)]
            if leg:
                choice[g] = leg[0].id
                notes.append(f"{g}: no servo meets the load; QDD part {leg[0].id} used")
            else:
                choice[g] = max(pool("servo"), key=lambda a: a.sustained).id
                notes.append(f"{g}: no part meets the load")
    return dict(choice=choice, bus=bus, notes=notes)


