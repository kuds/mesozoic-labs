"""Prototype cost: low / nominal / high, one physical prototype, outsourced fabrication.

Structure mirrors the 16 lines of PR #662's procurement model
(hardware/tyrannosaurus_rex_60cm/platform/engineering/assumptions.json, "budget"): priced parts
use the sourced catalog prices; lots (frame, covers, integration, bench, spares, contingency) start
from PR #662's 60 cm values and scale with stated drivers.  Included: outsourced fabrication,
bounded subcontract integration, bench/tether supply and charger, spares (two leg drives and a pack),
freight/taxes inside contingency.  Excluded: engineering salaries, facilities, firm quotes.
Every scaled line is an estimate until quoted from drawings.
"""

from __future__ import annotations

from . import catalog

# PR #662 60 cm reference lots (low, target, high) and the driver used to scale them.
REF = dict(
    frame=(1800, 2750, 4000),  # structure, bearings, hubs, soles, fasteners: scales with structure mass^0.6
    covers=(350, 600, 1050),  # SLS/MJF skins incl. inserts/finish: scales with cover volume
    integration=(1350, 2200, 3200),  # subcontract fit-up: scales with drive count and size
    bench=(550, 850, 1300),  # tether supply, charger, cradle/catch frame, instrumentation
    harness=(150, 240, 350),
    foot_sensing=(100, 180, 300),
    control=(125, 180, 270),
    power24=(275, 420, 650),
)
REF_STRUCTURE_KG = 1.79  # PR #662 frames, bearings and feet
REF_COVER_CM3 = 0.30e3 * 1.6  # PR #662: ~0.30 m^2 of 1.6 mm skins (estimate of the reference lot's volume)
CONTINGENCY = (0.053, 0.072, 0.084)  # PR #662 contingency as a fraction of the other lines


def _scale(t, f):
    return tuple(v * f for v in t)


def estimate(design, rows: list[dict], selection: dict) -> dict:
    acts = {g: design.act(g) for g in ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b")}
    leg = sum(2 * a.price for a in acts.values())
    expr = design.act("neck").price + design.act("tail").price
    for g in getattr(design, "head_x_groups", ()):  # Stage 2's expressive head: its two servos (none before)
        expr += design.act(g).price
    parts = catalog.parts()
    comp = parts[selection["compute"]]
    cam = parts[selection["camera"]]
    tof = parts[selection["tof"]]
    bat = parts[selection["battery"]]
    lid = parts[selection["lidar"]] if selection.get("lidar") else None
    bus48 = selection.get("bus", 24) > 30
    structure = sum(r["mass_kg"] for r in rows if r["category"] in ("structure", "bearings", "linkage", "foot"))
    cover_cm3 = sum(r["mass_kg"] for r in rows if r["category"] == "cover") / 1.01e-3
    # material experiments (overnight refinement): the frame lot is priced on the aluminium-equivalent part size
    # with the plate material's cost factor; the cover lot on the cover's own volume with its cost per volume
    frame_factor_mat, cover_factor_mat = 1.0, 1.0
    if getattr(design, "struct_conv_al", 0.0) > 0.0:
        from . import hardware as _hw

        al_equiv = structure + design.struct_al_equiv_extra
        cf = _hw.STRUCT_MATERIALS[design.struct_mat or design.bracket_mat]["cost_factor"]
        frame_factor_mat = (design.struct_conv_al * cf + (al_equiv - design.struct_conv_al)) / al_equiv
        structure = al_equiv
    cmat = getattr(design, "cover_mat_override", None)
    if cmat:
        from .build import COVER_MATERIALS

        m = COVER_MATERIALS[cmat]
        cover_cm3 = sum(r["mass_kg"] for r in rows if r["category"] == "cover") / (m["density"] * 1e-6)
        cover_factor_mat = (m["cost_per_kg"] * m["density"]) / (180.0 * 1010.0)
    tether = getattr(design, "power_mode", "battery") == "tether"
    size = design.sk.H / 0.39
    lines = {}
    lines["leg_actuators"] = (leg, leg, leg * 1.05)
    lines["expression_actuators"] = (expr, expr, expr * 1.05)
    lines["compute"] = (comp.price + 60, comp.price + 120, comp.price + 200)
    lines["rgbd_camera"] = (cam.price + 15, cam.price + 35, cam.price + 80)
    lines["tof"] = (2 * tof.price, 2 * tof.price + 10, 2 * tof.price + 30)
    if lid:
        lines["lidar"] = (lid.price + 15, lid.price + 30, lid.price + 60)
    lines["battery"] = (bat.price + 30, bat.price + 60, bat.price + 100) if not tether else (120.0, 180.0, 280.0)
    lines["control"] = _scale(REF["control"], 1.3 if bus48 else 1.0)
    lines["foot_sensing"] = _scale(REF["foot_sensing"], max(1.0, size**0.5))
    lines["power"] = (450, 650, 950) if bus48 else REF["power24"]
    lines["harness"] = _scale(REF["harness"], max(1.0, size) * (1.3 if bus48 else 1.0))
    f_frame = max(1.0, (structure / REF_STRUCTURE_KG) ** 0.6)
    lines["frame"] = _scale(REF["frame"], f_frame * frame_factor_mat if frame_factor_mat != 1.0 else f_frame)
    f_cov = max(0.6, cover_cm3 / REF_COVER_CM3) ** 0.85
    lines["covers"] = _scale(REF["covers"], f_cov * cover_factor_mat if cover_factor_mat != 1.0 else f_cov)
    lines["bench"] = (900, 1400, 2200) if design.sk.H > 0.6 else REF["bench"]
    most = max(a.price for a in acts.values())
    bp = 0.0 if tether else bat.price
    lines["spares"] = (2 * most + bp + 150, 2 * most + bp + 350, 2 * most + bp + 650)
    f_int = max(1.0, size**0.7)
    lines["integration"] = _scale(REF["integration"], f_int)
    sub = [sum(v[i] for v in lines.values()) for i in range(3)]
    lines["contingency"] = tuple(sub[i] * CONTINGENCY[i] for i in range(3))
    total = [sum(v[i] for v in lines.values()) for i in range(3)]
    return dict(lines={k: [round(x, 2) for x in v] for k, v in lines.items()}, low=round(total[0], 0),
                nominal=round(total[1], 0), high=round(total[2], 0),
                drivers=dict(structure_kg=structure, cover_cm3=cover_cm3, frame_factor=f_frame, cover_factor=f_cov,
                             integration_factor=f_int, bus_48v=bus48))
