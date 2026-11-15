"""The tables of ``stage2/j/JETSON_BODY.md``, written from the committed records only: the Jetson case's candidate
record and screens (``round2/refine/jetson_75``) beside the pick's (``round2/refine/feet_75``), the two carpet-lift
passes, and the run's records (``stage2/j/75-M-H-10-J``).  Nothing is computed here but differences, sums and
formats, so every figure in the block is a record's; ``tests/test_stage2_j.py`` checks that the document's block is
this script's output.

Each row carries its kind of evidence: (a) a model result of this study (a prescribed-walk or static reading of the
simulated body), (c) a sourced fact, with its catalog id or document, (d) an estimate or assumption.

Run from hardware/trex_form_study, with PYTHONPATH=. :

    python stage2/j/j_tables.py            # print the block
    python stage2/j/j_tables.py --write    # rewrite the block in JETSON_BODY.md between its two markers
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

RUN = Path(__file__).resolve().parent
STUDY = RUN.parents[1]
BODY, CASE, CID, BASE = "75-M-H-10-J", "jetson_75", "75-M-H-10", "feet_75"
DOC = RUN / "JETSON_BODY.md"
BEGIN, END = "<!-- j_tables.py: begin -->\n", "<!-- j_tables.py: end -->"
SCREENS = ("clearance", "sensors", "turning", "carpet", "envelope")


def _json(*parts) -> dict:
    return json.loads(STUDY.joinpath(*parts).read_text())


def load() -> dict:
    """Every record the tables read."""
    refine = ("round2", "refine")
    out = dict(j=_json(*refine, CASE, "candidates", f"{CID}.json"), p=_json(*refine, BASE, "candidates", f"{CID}.json"),
               ij=_json(*refine, CASE, "review", "review_index.json")[CID],
               ip=_json(*refine, BASE, "review", "review_index.json")[CID])
    for key, case in (("cj", CASE), ("cp", BASE)):
        out[key] = _json(*refine, case, "review", "screening", f"{CID}_clearance.json")
    out["lj"] = next(r for r in _json(*refine, "carpet_lift_jetson_75_0.175_0.180_0.190.json") if r["id"] == CID)
    out["lp"] = next(r for r in _json(*refine, "carpet_lift_feet_75_0.175_0.180_0.190.json") if r["id"] == CID)
    out["gates"] = _json("stage2", "j", BODY, "gates.json")
    return out


def _table(head: list, rows: list) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _para(text: str) -> str:
    """A paragraph wrapped to the documents' 120 columns."""
    return "\n" + textwrap.fill(text, width=120, break_long_words=False, break_on_hyphens=False) + "\n"


def _mm(v, nd=1) -> str:
    return f"{v:.{nd}f} mm"


def _yes(v) -> str:
    return "pass" if v is True else "**fail**" if v is False else "not read"


def _is(v) -> str:
    return "yes" if v else "no"


def _signed(v: float, nd: int, unit: str = "") -> str:
    """A signed figure to ``nd`` decimals, a zero written without a sign."""
    text = f"{v:+.{nd}f}"
    return (text[1:] if float(text) == 0.0 else text) + unit


def _dims(d) -> str:
    return " × ".join(f"{1e3 * v:g}" for v in d) + " mm"


def bodies(D: dict) -> str:
    j, p = D["j"], D["p"]
    jr = j["jetson"]
    kit, pi, reg = jr["compute"], jr["compute"]["replaces"], jr["regulator"]
    rg = reg["geometry"]
    opt = reg["options"][reg["placed"]]
    sizes = jr["chest"]
    same = sorted(g for g in j["margins"] if j["margins"][g]["actuator"] == p["margins"][g]["actuator"])
    rows = [
        ["Model `xml_sha256`", f"`{p['xml_sha256'][:8]}…`", f"`{j['xml_sha256'][:8]}…`", "a"],
        ["Refine case and record", f"`{BASE}` (sized)", f"`{CASE}` (built once from that record, not re-sized)", "a"],
        ["Compute part", f"`{pi['part']}`: {pi['mass_kg']:.3f} kg, {_dims(pi['dims_m'])}, ${pi['price_usd']:.0f}",
         f"`{kit['part']}`: {kit['mass_kg']:.3f} kg, {_dims(kit['dims_m'])}, ${kit['price_usd']:.0f}", "c (catalog)"],
        ["Compute tray's envelope, in the chest", _dims([v / 1e3 for v in sizes["base"]["size_mm"]["compute_tray"]]),
         _dims([v / 1e3 for v in sizes["jetson"]["size_mm"]["compute_tray"]]),
         "a (the board plus 17 × 20 × 30 mm, `designlab/build.py`)"],
        ["Compute tray's allowance", "0.120 kg", "0.120 kg", "d (PR #662's allowance, any compute part)"],
        ["6S-to-12 V regulator", "none placed",
         f"`{reg['placed']}`: {opt['mass_kg'] * 1e3:.0f} g, {_dims(opt['dims_m'])}, ${opt['price_usd']:.2f}; on the power "
         f"box, {rg['standoff_m'] * 1e3:.0f} mm above it", "c (catalog); its place: a"],
        ["Regulator's standoffs, 12 V lead and plug", "none", f"{rg['leads_and_mount_kg'] * 1e3:.0f} g", "d"],
        ["Control stack (Teensy 4.1, CAN × 2, IMU)", "0.180 kg", "0.180 kg, kept", "d (PR #662's allowance)"],
        ["Drives, pack, bus, structure, covers", f"{len(p['margins'])} drive groups, `{p['selection']['battery']}`, "
         f"{p['selection']['bus']:g} V", f"the same ({len(same)} of {len(j['margins'])} drive groups the pick's, "
         f"`{j['selection']['battery']}`, {j['selection']['bus']:g} V)", "a"],
        ["Hip height (the body unit B)", f"{p['H']:.4f} m", f"{j['H']:.4f} m", "a"],
    ]
    return _table(["", f"`{CID}` (the pick, Pi 5 set)", f"`{BODY}`", "Kind"], rows)


def parts(D: dict) -> str:
    jr = D["j"]["jetson"]
    kind = {"compute": "c (catalog masses)", "compute_tray": "a (its box grows with the board; the mass is the "
            "allowance)", "compute_regulator": "c (15 g) + d (10 g)", "integration_reserve": "d (6 % of installed mass)"}
    rows = []
    for r in jr["parts"]:
        c0, c1 = r["com_body"]
        at = c1 if c1 is not None else c0
        rows.append([f"`{r['name']}`", r["status"], f"{r['mass_kg'][0] * 1e3:.1f} g", f"{r['mass_kg'][1] * 1e3:.1f} g",
                     f"{r['mass_change_kg'] * 1e3:+.1f} g", ", ".join(f"{1e3 * v:.1f}" for v in at), kind[r["name"]]])
    lm, mm = jr["ledger_mass_kg"], jr["mass_kg"]
    rows.append(["**Sum of the rows**", "", "", "", f"**{jr['parts_mass_change_kg'] * 1e3:+.1f} g**", "", "sum"])
    rows.append(["**The two models' masses**", "", f"{mm['base']:.3f} kg", f"{mm['jetson']:.3f} kg",
                 f"**{mm['change'] * 1e3:+.1f} g**", "", "a"])
    com = jr["com_at_pick_pose_m"]
    text = _table(["Ledger row", "", "Pi body", "Jetson body", "Change", "Centre in the trunk frame, mm (x, y, z)", "Kind"],
                  rows)
    text += _para("Centre of mass at the pick's home pose, x forward, y left, z up: "
                  f"({', '.join(_signed(1e3 * v, 3) for v in com['change'])}) mm in the model, "
                  f"({', '.join(_signed(1e3 * v, 3) for v in com['from_parts_change'])}) mm from the rows above (a). "
                  f"The ledgers' own masses differ by {lm['change'] * 1e3:+.3f} g; the models' masses are written to "
                  "six digits.")
    return text


def against(D: dict) -> str:
    j, p, ij, ip = D["j"], D["p"], D["ij"], D["ip"]
    jr = j["jetson"]
    ch, ab, at = jr["com_ahead_of_hip_m"], jr["against_base"], jr["against_base_at_pick_pose"]
    gj, gp = jr["chest"]["jetson"]["gaps_mm"], jr["chest"]["base"]["gaps_mm"]
    pmj, pmp = jr["chest"]["jetson"]["trunk_profile_margin_mm"], jr["chest"]["base"]["trunk_profile_margin_mm"]
    hp = ab["hip_pitch"]

    def gov(m):
        g, k, v = max(((g, k, x[f"{k}_ratio"]) for g, x in m.items() for k in ("cont", "peak", "speed")),
                      key=lambda t: t[2])
        return f"{g.replace('_', ' ')} {k} {v:.3f}"

    def gap(a, b):
        key = f"{a}|{b}"
        return ("–" if key not in gp else _mm(gp[key]), _mm(gj[key]))

    def hip(a):
        front, rear = gap(a, "left_hip_roll_motor"), gap(a, "right_hip_roll_motor")
        return (f"{front[0]} / {rear[0]}", f"{front[1]} / {rear[1]}")

    def margin(m, name):
        return "–" if name not in m else " / ".join(f"{m[name][k]:.1f}" for k in ("dorsal", "ventral", "half_width")) + " mm"

    cj, cp = ij["clearance"], ip["clearance"]
    sj, sp = ij["sensors"], ip["sensors"]
    bj = D["cj"]["battery"]["directions"]["-z"]
    bp = D["cp"]["battery"]["directions"]["-z"]
    ext = jr["exterior_m"]
    home = jr["home_pose"]
    rows = [
        ["Mass", f"{jr['mass_kg']['base']:.3f} kg", f"{jr['mass_kg']['jetson']:.3f} kg",
         f"{jr['mass_kg']['change'] * 1e3:+.1f} g", "a"],
        ["Centre of mass ahead of the hip axis, each body at its own home", _mm(1e3 * ch["base"], 2),
         _mm(1e3 * ch["jetson"], 2), f"{1e3 * ch['change']:+.2f} mm", "a"],
        ["The same, both at the pick's home (the changed parts alone)", _mm(1e3 * ch["base"], 2),
         _mm(1e3 * ch["jetson_at_pick_pose"], 2), f"{1e3 * ch['change_at_pick_pose']:+.2f} mm", "a"],
        ["Centre of mass height", _mm(1e3 * jr["com_height_m"]["base"], 2), _mm(1e3 * jr["com_height_m"]["jetson"], 2),
         f"{1e3 * jr['com_height_m']['change']:+.2f} mm", "a"],
        ["Home pose: largest joint change (the stance is re-solved)", "–",
         f"{home['largest_joint_change'][0].replace('_', ' ')} {home['largest_joint_change'][1]:+.4f} rad",
         f"trunk pose change {home['trunk_pose_change_max']:g}", "a"],
        ["Crown height; overall width", f"{1e3 * ext['crown']['base']:.2f} mm; {1e3 * ext['max_width']['base']:.2f} mm",
         f"{1e3 * ext['crown']['jetson']:.2f} mm; {1e3 * ext['max_width']['jetson']:.2f} mm",
         f"{_signed(1e3 * ext['crown']['change'], 3, ' mm')}; {_signed(1e3 * ext['max_width']['change'], 3, ' mm')}",
         "a"],
        ["Hip-pitch hold (share of the drive's sustained torque)",
         f"{hp['base_hold_Nm']:.3f} N·m ({hp['base_hold_over_sustained']:.3f})",
         f"{hp['hold_Nm']:.3f} N·m ({hp['hold_over_sustained']:.3f})", f"{hp['hold_change_Nm']:+.3f} N·m", "a"],
        ["The same at the pick's home", f"{hp['base_hold_Nm']:.3f} N·m",
         f"{at['hip_pitch']['hold_Nm']:.3f} N·m ({at['hip_pitch']['hold_over_sustained']:.3f})",
         f"{at['hip_pitch']['hold_change_Nm']:+.3f} N·m", "a"],
        ["Knee hold", f"{ab['knee']['base_hold_Nm']:.3f} N·m", f"{ab['knee']['hold_Nm']:.3f} N·m",
         f"{ab['knee']['hold_change_Nm']:+.3f} N·m", "a"],
        ["Hip-pitch ratios to the drive's planning limits: continuous / peak / speed",
         " / ".join(f"{hp['base_ratios'][k]:.3f}" for k in ("cont", "peak", "speed")),
         " / ".join(f"{hp['ratios'][k]:.3f}" for k in ("cont", "peak", "speed")), "", "a"],
        ["Governing ratio of any drive (the sizing walks, Froude 0.05 included)", gov(p["margins"]), gov(j["margins"]),
         "", "a"],
        ["Drive groups over a planning limit; groups one selection step would change",
         "none (sized)", f"{len(jr['groups_over_planning_limit'])}; {len(jr['selection_if_resized']['differs'])}",
         "", "a"],
        ["Dev kit's tray to the power box", *gap("compute_tray", "power_distribution"), "", "a"],
        ["Compute board to the power box", *gap("compute", "power_distribution"), "", "a"],
        ["Tray to the hip-roll motors, front / rear", *hip("compute_tray"), "", "a"],
        ["Power box to the hip-roll motors, front / rear", *hip("power_distribution"), "", "a"],
        ["Tray to the spine beam", *gap("compute_tray", "spine_beam"), "", "a"],
        ["Tray to the neck servo", *gap("compute_tray", "neck_motor"), "", "a"],
        ["Tray to a ToF sensor", *gap("compute_tray", "tof_l"), "", "a"],
        ["Regulator to the power box (its standoffs)", *gap("compute_regulator", "power_distribution"), "", "a"],
        ["Regulator to the tray; to the compute board", "–",
         f"{gap('compute_regulator', 'compute_tray')[1]}; {gap('compute_regulator', 'compute')[1]}", "", "a"],
        ["Regulator to the hip-roll motors, front / rear", *hip("compute_regulator"), "", "a"],
        ["Regulator to the spine beam; to a pelvis plate; to the control stack", "–",
         "; ".join(gap("compute_regulator", b)[1] for b in ("spine_beam", "pelvis_plate_l", "control_stack")), "", "a"],
        ["Compute board inside the trunk profile: dorsal / ventral / half-width", margin(pmp, "compute"),
         margin(pmj, "compute"), "", "a"],
        ["Regulator inside the trunk profile", "–", margin(pmj, "compute_regulator"), "", "a"],
        ["Component overlaps on one body at home (the screen's gate)", len(cp["same_body_component_overlaps"]),
         len(cj["same_body_component_overlaps"]), "", "a"],
        ["Battery removal: clear directions; least clearance on the way out",
         f"{', '.join(cp['battery_clear_directions'])}; {bp['min_clearance_mm']} mm",
         f"{', '.join(cj['battery_clear_directions'])}; {bj['min_clearance_mm']} mm", "", "a"],
        ["ToF floor window ahead of the toes (each sensor; 64 of 64 zones on the floor)",
         " to ".join(f"{v:.4f}" for v in sp["tof"]["tof_l"]["floor_ahead_of_toes_m"]) + " m",
         " to ".join(f"{v:.4f}" for v in sj["tof"]["tof_l"]["floor_ahead_of_toes_m"]) + " m", "", "a"],
        ["ToF zones on the robot itself", sum(len(v["self_zones"]) for v in sp["tof"].values()),
         sum(len(v["self_zones"]) for v in sj["tof"].values()), "", "a"],
        ["Foot path covered by the two ToF sensors", f"{sp['near_strip_coverage']['foot_path']['tof_union']:.4f}",
         f"{sj['near_strip_coverage']['foot_path']['tof_union']:.4f}", "", "a"],
        ["Camera view blocked by the robot at home; nearest floor it sees ahead of the toes",
         f"{sp['camera_blocked']['home']:.4f}; {sp['centre_column_nearest_visible_floor_ahead_of_toes_m']:.4f} m",
         f"{sj['camera_blocked']['home']:.4f}; {sj['centre_column_nearest_visible_floor_ahead_of_toes_m']:.4f} m", "", "a"],
        ["Cost model's nominal total (the compute line at the catalog price; no regulator line)",
         f"${p['cost']['nominal']:,.0f}", f"${j['cost']['nominal']:,.0f}",
         f"{j['cost']['nominal'] - p['cost']['nominal']:+,.0f} $", "d (cost model, A8)"],
    ]
    return _table(["Reading", "Pi body", "Jetson body", "Change", "Kind"], rows)


def screens(D: dict) -> str:
    ij, ip = D["ij"], D["ip"]
    cj, cp = ij["clearance"], ip["clearance"]

    def tight(c):
        v, k = min((min(m["margin_min"], m["margin_max"]), n) for n, m in c["joint_range_margins"].items())
        return f"{k.replace('_', ' ')} {v:.4f} rad"

    def swing(c):
        return " / ".join(f"{c['swing_clearance'][w]['mid_swing_min_mm']:.2f}" for w in ("nominal", "brisk")) + " mm"

    names = dict(interference_free_home="No interference at home",
                 interference_free_nominal_gait="No interference over the nominal walk",
                 interference_free_brisk_gait="No interference over the brisk walk",
                 component_overlap_free="No component overlap on one body",
                 joint_range_margin="Joint-range margin at least 0.10 rad", swing_clearance="Swing clearance at least 0.04 H",
                 tail_clearance="Tail at least 0.25 H above the floor", battery_path="Battery has a clear way out")
    reading = dict(joint_range_margin=(tight(cp), tight(cj)), swing_clearance=(swing(cp), swing(cj)),
                   tail_clearance=(f"{cp['tail_min_height']['frac_H']:.4f} H", f"{cj['tail_min_height']['frac_H']:.4f} H"),
                   battery_path=(", ".join(cp["battery_clear_directions"]), ", ".join(cj["battery_clear_directions"])),
                   interference_free_home=(cp["counts"]["home"]["interference_pairs"], cj["counts"]["home"]["interference_pairs"]),
                   interference_free_nominal_gait=(cp["counts"]["gait_nominal"]["interference_pairs"],
                                                   cj["counts"]["gait_nominal"]["interference_pairs"]),
                   interference_free_brisk_gait=(cp["counts"]["gait_brisk"]["interference_pairs"],
                                                 cj["counts"]["gait_brisk"]["interference_pairs"]),
                   component_overlap_free=(len(cp["same_body_component_overlaps"]), len(cj["same_body_component_overlaps"])))
    walking = ("interference_free_home", "interference_free_nominal_gait", "interference_free_brisk_gait",
               "component_overlap_free", "joint_range_margin")
    rows = []
    for g in walking + ("swing_clearance", "tail_clearance", "battery_path"):
        rows.append(["clearance" + (" (walking gate)" if g in walking else ""), names[g], reading[g][0], reading[g][1],
                     _yes(cj["pass"][g])])
    rows.append(["clearance", "Every gate (`pass.all`)", _yes(cp["pass"]["all"]), _yes(cj["pass"]["all"]),
                 _yes(cj["pass"]["all"])])
    sj, sp = ij["sensors"], ip["sensors"]
    rows.append(["sensors", "Readings, no gate of its own: a person's torso in view at home, straight ahead and at ±30°",
                 _is(sp["torso_follow"]["home"]["ok"]), _is(sj["torso_follow"]["home"]["ok"]),
                 "readings (more in table 3)"])
    tj, tp = ij["turning"], ip["turning"]
    rows.append(["turning", "Target turns (0.35 rad/s walking, 0.5 rad/s in place): within limits, no interference, IK",
                 f"worst ratio {tp['worst_ratio_targets']:.3f}; thigh twist {tp['hip_yaw_range_used_max']:.3f} of its "
                 f"range; grip needed {tp['mu_needed_max']:.3f}",
                 f"worst ratio {tj['worst_ratio_targets']:.3f}; thigh twist {tj['hip_yaw_range_used_max']:.3f} of its "
                 f"range; grip needed {tj['mu_needed_max']:.3f}", _yes(tj["targets_pass"])])
    rows.append(["turning", "Closest approach in the target turns, walking / in place",
                 f"{tp['min_gap_mm']['walk_0.35']:.2f} / {tp['min_gap_mm']['in_place_0.50']:.2f} mm",
                 f"{tj['min_gap_mm']['walk_0.35']:.2f} / {tj['min_gap_mm']['in_place_0.50']:.2f} mm",
                 _yes(tj["targets_interference_free"])])
    rows.append(["turning", "Stretch turn (0.5 rad/s walking)", _yes(tp["stretch_pass"]), _yes(tj["stretch_pass"]),
                 _yes(tj["stretch_pass"])])
    kj, kp = ij["carpet"], ip["carpet"]
    rows.append(["carpet", "Friction below the carpet band's lower edge and every pile top cleared in mid-swing",
                 f"toe margin {kp['toe_clearance_margin_mm']:.2f} mm; friction margin {kp['friction_margin']:.3f}",
                 f"toe margin {kj['toe_clearance_margin_mm']:.2f} mm; friction margin {kj['friction_margin']:.3f}",
                 _yes(kj["pass"])])
    rows.append(["carpet", "Stands, servo-held, on the hard floor and four carpets",
                 f"{sum(v['upright'] for v in kp['standing'].values())} of {len(kp['standing'])}",
                 f"{sum(v['upright'] for v in kj['standing'].values())} of {len(kj['standing'])}",
                 _yes(all(kj["checks"]["servo_held_upright"].values()))])
    lj, lp = D["lj"]["lifts"], D["lp"]["lifts"]

    def lift(r, k="0.18"):
        v = r[k]
        return (f"drives {'within' if v['drives_within_limits'] else 'over'} limits ({v['worst_group'].replace('_', ' ')} "
                f"{v['worst_ratio']:.3f}); {v['min_above_pile_top_mm']:.2f} mm above a pile top")

    def lift_ok(r, k="0.18"):
        v = r[k]
        return v["drives_within_limits"] and v["walking_gates"] and v["meets_target_all_piles"]

    rows.append(["carpet-lift pass", "Carpet swing at 0.18 H: target met on every pile, drives within limits, walking gates",
                 lift(lp), lift(lj), _yes(lift_ok(lj))])
    def low(r, k="0.175"):
        v = r[k]
        return (f"target met on every pile: {_is(v['meets_target_all_piles'])}; drives within limits: "
                f"{_is(v['drives_within_limits'])} ({v['worst_group'].replace('_', ' ')} {v['worst_ratio']:.4f})")

    rows.append(["carpet-lift pass", "One step lower, 0.175 H", low(lp), low(lj),
                 "as the pick's: 0.18 H is the rule's lift" if lift_ok(lj, "0.175") == lift_ok(lp, "0.175") else "differs"])
    rows.append(["carpet-lift pass", "One step higher, 0.19 H (not used)", low(lp, "0.19") + f"; walking gates: "
                 f"{_is(lp['0.19']['walking_gates'])}", low(lj, "0.19") + f"; walking gates: "
                 f"{_is(lj['0.19']['walking_gates'])}", "not the rule's lift on either body"])
    ej, ep = ij["envelope"], ip["envelope"]
    rows.append(["envelope", "Froude 0.05 walk: drives within limits, reference CoP on the pad, IK",
                 f"{ep['fast_walk_worst'][0]} {ep['fast_walk_worst'][1]:.3f}", f"{ej['fast_walk_worst'][0]} "
                 f"{ej['fast_walk_worst'][1]:.3f}",
                 _yes(ej["fast_walk_drives_ok"] and ej["fast_walk_reference_cop_on_pad"] and ej["fast_walk_ik_ok"])])
    rows.append(["envelope", "Readings, not gated: one-foot stop; one-step push bound; session's worst drive",
                 f"{ep['stop_max_one_foot_decel_ms2']:.3f} m/s²; {ep['push_max_dv_ms']:.4f} m/s; {ep['session_worst'][0]} "
                 f"{ep['session_worst'][1]:.3f}",
                 f"{ej['stop_max_one_foot_decel_ms2']:.3f} m/s²; {ej['push_max_dv_ms']:.4f} m/s; {ej['session_worst'][0]} "
                 f"{ej['session_worst'][1]:.3f}", "readings"])
    head = (f"The screened model is the sized one: {_yes(ij['matches_sized'])} (`{ij['xml_sha256'][:8]}…`). All rows are "
            "kind (a).\n\n")
    return head + _table(["Screen", "Gate or reading", "Pi body", "Jetson body", "Jetson body's verdict"], rows)


def power_rows(D: dict) -> str:
    j, p = D["j"], D["p"]
    pm = j["jetson"]["power_modes"]
    aux = p["power"]["aux"]["loads"]

    def row(label, rail, drives, total, rt, kind, change=None):
        return [label, f"{rail:.1f} W", f"{drives:.2f} W", f"{total:.2f} W", f"{rt:.1f} min",
                "–" if change is None else f"{change:+.1f} min", kind]

    b = pm["base"]
    rows = [row(f"Pi body, `{b['compute']}`: 8.0 W + 1.5 W SSD (A14)", b["compute_rail_w"], b["drives_w"], b["total_w"],
                b["runtime_min"], "d (survey's typical draw, unmeasured)")]
    for mode, label in (("15w", "Jetson body, 15 W mode: 15 W cap + 3 W carrier, storage, cooling"),
                        ("25w", "Jetson body, 25 W mode: 25 W cap + 3 W")):
        m = pm["modes"][mode]
        rows.append(row(label, m["compute_rail_w"], m["drives_w"], m["total_w"], m["runtime"]["minutes"],
                        "c (the caps) + d (the 3 W); draft A19", m["runtime_change_against_base_min"]))
    rows.append(row("Jetson body, the record's own `power`: 25 W + 6 W (A13, round one's allowance)",
                    pm["record_default"]["compute_rail_w"], j["power"]["drives_w"], j["power"]["total_w"],
                    j["runtime"]["minutes"], "d (A13)", j["runtime"]["minutes"] - b["runtime_min"]))
    t = pm["survey_typical_15w"]
    rows.append(row("For comparison only: the 15 W mode at the survey's typical draw, 12 W + 3 W", t["compute_rail_w"],
                    j["power"]["drives_w"], t["total_w"], t["runtime_min"], "d (survey; like for like with the Pi row)",
                    t["runtime_change_against_base_min"]))
    text = _table(["Compute row", "At the compute rail", "Drives (nominal walk)", "Walking power at the pack", "Runtime",
                   "Against the Pi body", "Kind"], rows)
    r = j["runtime"]
    text += _para(f"Every row adds the microcontroller's {aux['mcu_comms']:g} W and the sensors' {aux['sensors']:g} W, "
                  f"passes the sum through the 0.88 converter (A6) and uses 75 % of the `{r['pack']}` pack's "
                  f"{r['energy_wh']:g} Wh (A6): `designlab/power.py`, as every round-two record. The drives' power is "
                  "the model's (a); the runtime inherits the estimates (d). Both modes meet the 20 minute target: "
                  f"{'yes' if all(m['meets_target'] for m in pm['modes'].values()) else 'no'}.")
    return text


def stage1_gates(D: dict) -> str:
    sys.path.insert(0, str(STUDY))
    from designlab.stage1.__main__ import summary

    doc = D["gates"]
    lines = summary(doc).splitlines()
    g1, g2, g3 = doc["G1"], doc["G2"], doc["G3"]
    sr = g1["step_rate"]
    pnl = g2["nominal_panel_with_envelope"]
    verdict = {True: "pass", False: "**fail**", None: "provisional (not judged)"}
    rows = [
        ["G1 plant", verdict[g1["pass"]],
         f"stands 20 s on hard floor and low pile (tilt max {g1['stand']['hard']['tilt_from_home_max_deg']:.2f} / "
         f"{g1['stand']['low']['tilt_from_home_max_deg']:.2f} deg); hashes, motor map and repeatability pass; step rate "
         f"{sr['single_process']:,.0f} steps/s alone, {sr['per_process_all_median']:,.0f} per process on "
         f"{sr['processes']} of {sr['machine']['cores']} cores (lowest sample {sr['lowest_sample']:,.0f}); the rule asks "
         f"{sr['threshold']:,} with one process per core, and on these {sr['processes']} processes the samples "
         + ("meet it" if sr.get("ok_on_these_processes", sr["ok"]) else "are **below it**")],
        ["G2 references", verdict[g2["pass"]],
         f"every reference matches its records ({g2['all_match']}); nominal panel median {pnl['median']:.1f} strides "
         f"({pnl['runs_at_least_min_strides']}/{pnl['runs']} runs of at least {g2['min_strides']:g}); unperturbed "
         f"{g2['nominal_strides_with_envelope']:.1f} strides with the envelope, "
         f"{g2['nominal_strides_without_envelope']:.1f} without"],
        ["G3 transitions", verdict[g3["pass_"]],
         f"{len(g3['transitions']) - len(g3['failed']) - len(g3['missing'])}/{len(g3['transitions'])} pass; worst CoM "
         f"{g3['worst_com']['mm']:.3f} mm ({g3['worst_com']['transition']}), worst ZMP {g3['worst_zmp']['mm']:.3f} mm "
         f"({g3['worst_zmp']['transition']})"],
        ["The record's `pass`", verdict[doc["pass"]], "; ".join(f"{k}: {v}" for k, v in doc.get("provisional", {}).items())
         or "–"],
    ]
    text = _table(["Gate", "Verdict in `gates.json`", "Readings"], rows)
    text += "\nThe CLI's own summary of that record (`python -m designlab.stage1 gates`), first words of each line:\n\n"
    text += "".join(f"- `{ln.strip()[:150]}{'…' if len(ln.strip()) > 150 else ''}`\n" for ln in lines
                    if ln.startswith(("G1", "G2", "G3")))
    return text


def not_placed(D: dict) -> str:
    n = D["j"]["jetson"]["not_placed"]
    a = n["usb_can_adapter"]
    rows = [
        [f"USB-CAN adapter, `{a['part']}`, one per CAN bus", f"{a['mass_kg'] * 1e3:.0f} g each (c, catalog); "
         f"${a['price_usd']:.0f} each (c)",
         f"{a['would_add_kg']['1'] * 1e3:.0f} g for one bus, {a['would_add_kg']['2'] * 1e3:.0f} g for the study's two",
         f"{a['with_reserve_kg']['1'] * 1e3:.1f} g, {a['with_reserve_kg']['2'] * 1e3:.1f} g", "d (estimate; not placed)"],
        ["Supervisor", "no catalog part", "no figure", "no figure", "–"],
    ]
    return _table(["Part of the Jetson-direct option", "Catalog", "Would add", "With the model's 6 % reserve", "Kind"], rows)


def tables() -> str:
    D = load()
    sections = (("Table 1. The two bodies", bodies), ("Table 2. The ledger rows that differ", parts),
                ("Table 3. Against the Pi body", against), ("Table 4. The round-two screens, gate by gate", screens),
                ("Table 5. Power rows and runtime", power_rows), ("Table 6. Stage 1 gates", stage1_gates),
                ("Table 7. Not placed: the Jetson-direct option's parts", not_placed))
    return "\n".join(f"**{title}**\n\n{fn(D)}" for title, fn in sections)


def main(argv: list[str]) -> int:
    block = tables()
    if argv == ["--write"]:
        text = DOC.read_text()
        if text.count(BEGIN) != 1 or text.count(END) != 1:
            print(f"{DOC.name}: the two markers must each be there once")
            return 1
        DOC.write_text(text.split(BEGIN)[0] + BEGIN + block + END + text.split(END)[1])
        print(f"{DOC.relative_to(STUDY)}: tables rewritten")
        return 0
    if argv:
        print(__doc__.split("\n\n")[-1])
        return 2
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
