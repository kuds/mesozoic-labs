"""Convert the sourced research records into the generator's catalogs.

Inputs  : research/actuators_2026-11-07.json, research/parts_2026-11-07.json (each value carries
          its source URL; see those files).
Outputs : actuators.json, parts.json (SI units, planning fields, per-field basis labels).

Rules (stated here so every derived number can be traced):
* Leg actuators must use CAN and be currently listed; RS-485-only (all Unitree, MyActuator -R)
  and discontinued parts are excluded from selection but remain in the research record.
* Phase resistance is the Y-equivalent per-phase value (line-to-line / 2) as recorded.  CubeMars
  torque constants are motor-side and are multiplied by the gear ratio to get output-side Kt.
* Usable joint speed = max(vendor rated speed, 0.6 x no-load speed) (vendor rated speeds are quoted
  at rated torque; most walking is at part load).
* Reflected rotor inertia (MJCF armature) is an estimate: J_rotor = 2.5e-6 kg m^2 x (D / 55 mm)^4,
  times gear ratio squared.  PR #662 used 0.002 kg m^2 for the X4-36 (this rule gives 0.0032).
* Box servos (ROBOTIS, Feetech) publish stall torque; following PR #662 their planning sustained and
  peak torques are 0.167 and 0.293 of 12 V stall.  Their 'rated_torque' field below is set so the
  generic SUSTAINED_FRACTION gives that value.
* Allowable supply voltage = the first "a-b V" range in the record's ``voltage_range`` text (for
  CubeMars, the driver's "allowable" range when one is quoted).  Records that state only a nominal
  input ("48 V input; operating range not stated") are limited to that nominal voltage.
Run:  python -m designlab.data.build_catalog   (from hardware/trex_form_study)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXCLUDE = {
    "unitree_go_m8010_6": "RS-485 only", "unitree_im6014": "RS-485 only", "unitree_a1_motor": "RS-485 only",
    "unitree_b1_16": "RS-485 only, unavailable", "unitree_j288_servo": "RS-485 only",
    "myactuator_rmd_x8_p9_32_r": "RS-485 only", "myactuator_rmd_x6_p36_40": "discontinued",
    "myactuator_rmd_x8_p9_25": "discontinued", "myactuator_rmd_x10_p35_100": "discontinued",
}
BOX = {  # W x H x D mm from the research envelope notes
    "robotis_xm430_w210_t": (46.5, 28.5, 34.0), "robotis_xm430_w350_t": (46.5, 28.5, 34.0),
    "robotis_xm540_w150_t": (58.5, 33.5, 44.0), "robotis_xm540_w270_t": (58.5, 33.5, 44.0),
    "robotis_xh540_w150_t": (58.5, 33.5, 44.0), "feetech_sts3215_c018": (45.2, 24.7, 35.0),
    "feetech_sts3250": (45.22, 24.72, 35.0),
}
SUSTAINED_FRACTION = 0.57  # must match catalog.SUSTAINED_FRACTION


def voltage_window(r: dict, nominal: float) -> tuple[float, float, str]:
    """(min, max, basis) allowable supply voltage from the research record's text."""
    txt = r.get("voltage_range") or ""
    allow = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*V allowable", txt)
    m = allow or re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*V", txt)
    if m:
        return float(m.group(1)), float(m.group(2)), "sourced (voltage_range text)"
    return nominal, nominal, "nominal only (no operating range stated)"


def convert_actuators():
    src = json.loads((HERE / "research" / "actuators_2026-11-07.json").read_text())
    out, skipped = [], []
    for r in src["actuators"]:
        aid = r["id"]
        basis = {}
        if aid in EXCLUDE:
            skipped.append((aid, EXCLUDE[aid]))
            continue
        servo = aid in BOX
        if servo:
            stall = r["peak_torque_nm"]
            rated = 0.167 * stall / SUSTAINED_FRACTION
            peak = 0.293 * stall / 0.70
            box = [v / 1000 for v in BOX[aid]]
            d, L = box[0], box[2]
            basis.update(rated_torque="derived (PR #662 servo planning rule from stall)", dims="sourced")
        else:
            if r.get("diameter_mm") is None or r.get("axial_length_mm") is None or r["rated_torque_nm"] is None:
                skipped.append((aid, "incomplete dimensions or rating"))
                continue
            rated, peak = r["rated_torque_nm"], r["peak_torque_nm"]
            d, L = r["diameter_mm"] / 1000, r["axial_length_mm"] / 1000
            box = None
            basis.update(rated_torque="sourced (vendor definition in notes)", dims="sourced")
        gr = r.get("gear_ratio") or 1.0
        kt = r.get("torque_constant_nm_per_a")
        if kt is not None and aid.startswith("cubemars"):
            kt = kt * gr
            basis["kt"] = "derived (motor-side Kt x gear ratio)"
        elif kt is not None:
            basis["kt"] = "sourced"
        R = r.get("phase_resistance_ohm")
        basis["phase_resistance"] = "sourced (per phase, Y-equivalent)" if R is not None else "missing"
        nl = r.get("no_load_speed_rpm") or 0.0
        rs = r.get("rated_speed_rpm") or 0.0
        usable = max(rs, 0.6 * nl)
        D_ref = d if not servo else 0.03
        arm = 2.5e-6 * (D_ref / 0.055) ** 4 * gr**2 if not servo else 2.0e-4 * (gr / 270.0) ** 0.5
        price = r.get("price_usd")
        basis["price"] = "sourced" if price else "missing"
        vnom = r.get("nominal_voltage_v") or (12 if servo else 48)
        vmin, vmax, basis["voltage_window"] = voltage_window(r, vnom)
        out.append(dict(
            id=aid, vendor=r["vendor"], model=r["model"], mass_kg=r["mass_g"] / 1000, diameter_m=d, length_m=L,
            box_m=box, rated_torque_nm=rated, peak_torque_nm=peak, rated_speed_rpm=usable, no_load_speed_rpm=nl,
            nominal_voltage_v=vnom, voltage_min_v=vmin, voltage_max_v=vmax, gear_ratio=gr, kt_nm_per_a=kt,
            phase_resistance_ohm=R, price_usd=price, armature_kg_m2=arm, interface=r.get("interface", ""),
            basis=dict(mass="sourced", **basis, armature="estimate (rule in build_catalog.py)",
                       speed="derived (max of rated, 0.6 x no-load)"),
            sources=r.get("sources", []), notes=r.get("notes", "")[:600],
            vendor_rated_definition=r.get("rated_torque_definition", "")[:400],
        ))
    (HERE / "actuators.json").write_text(json.dumps(dict(
        generated_by="designlab/data/build_catalog.py", research="research/actuators_2026-11-07.json",
        excluded=skipped, actuators=out), indent=1))
    return out, skipped


PART_MAP = {
    # catalog id used by the generator -> research id, with installation notes
    "nvidia_orin_nano_super_devkit": "jetson_orin_nano_super_devkit",
    "luxonis_oak_d_lite": "luxonis_oak_d_lite",
    "luxonis_oak_d_pro_w": "luxonis_oak_d_pro_w",
    "st_vl53l5cx_carrier": "pololu_vl53l5cx_3417",
    "pololu_vl53l8cx": "pololu_vl53l8cx_3419",
    "slamtec_rplidar_c1": "slamtec_rplidar_c1",
    "ldrobot_stl19p": "ldrobot_stl19p",
    "lipo_6s_3300": "gensace_gtech_3300_6s_60c_ec5",
    "lipo_6s_5000": "gensace_gtech_5000_6s_60c_ec5",
    "lipo_6s_10000": "tattu_plus_10000_6s_25c",
    "lipo_12s_5000": "gensace_5000_12s_60c",
    "lipo_2x6s_3300_series": "gensace_3300_6s_x2_series",
    "liion_6s2p_p45b": "flywoo_explorer_p45b_6s2p",
    "liion_6s3p_50s": "dantech_tn6s3p_50s",
    "pololu_d42v110f12": "pololu_d42v110f12",
    "pololu_d36v50f12": "pololu_d36v50f12",
    "meanwell_ddr_120c_24": "meanwell_ddr_120c_24",
    "te_lev100": "te_kilovac_lev100",
    "pcan_usb_fd": "peak_pcan_usb_fd",
}
DIM_FALLBACK = {  # packs whose research record has no dimensions: estimated from cell layout (mm)
    "flywoo_explorer_p45b_6s2p": (150.0, 46.0, 72.0),  # 12 x 21700 in 6x2, plus wrap
}


def convert_parts():
    src = {p["id"]: p for p in json.loads((HERE / "research" / "parts_2026-11-07.json").read_text())["parts"]}
    out = []
    for cid, rid in PART_MAP.items():
        r = src[rid]
        dims = list(r.get("dims_mm") or [])
        basis = "sourced" if r["verification"] == "sourced" else "partially_sourced"
        if len(dims) < 3:
            if rid in DIM_FALLBACK:
                dims = list(DIM_FALLBACK[rid])
                basis += "; dims estimated from cell layout"
            else:
                dims = (dims + [10.0, 10.0, 10.0])[:3]
                basis += "; missing dims padded"
        mass = r.get("mass_g")
        out.append(dict(
            id=cid, research_id=rid, category=r["category"], vendor=r["vendor"], model=r["model"],
            dims_m=[v / 1000 for v in dims], mass_kg=(mass or 0) / 1000,
            power_typ_w=r.get("power_typ_w") or (r.get("power_max_w") or 0) * 0.6, power_max_w=r.get("power_max_w") or 0,
            capacity_wh=r.get("capacity_wh") or 0, voltage_v=_volt(r.get("voltage", "")),
            max_current_a=r.get("max_continuous_current_a"), price_usd=r.get("price_usd") or 0, fov=r.get("fov_deg", ""),
            basis=basis, sources=r.get("sources", []), notes=r.get("notes", "")[:500],
        ))
    # Derived series packs for a 12S (48 V class) bus: two identical 6S packs in series, cassette
    # holds both side by side.  Energy and mass double; current rating is that of one pack.
    by = {p["id"]: p for p in out}
    for cid, base, label in (("liion_12s2p_p45b", "liion_6s2p_p45b", "2 x Flywoo 6S2P P45B in series"),
                             ("liion_12s3p_50s", "liion_6s3p_50s", "2 x Dan-Tech 6S3P-50S in series")):
        b = by[base]
        L, W, Hh = b["dims_m"]
        out.append(dict(b, id=cid, research_id=b["research_id"] + " x2", model=label,
                        dims_m=[L, 2 * W + 0.004, Hh], mass_kg=2 * b["mass_kg"], capacity_wh=2 * b["capacity_wh"],
                        voltage_v=2 * b["voltage_v"], price_usd=2 * b["price_usd"] if b["price_usd"] else 0,
                        basis="derived (two sourced packs in series)", notes=label + "; " + b["notes"][:300]))
    out.append(_compute_from_survey())
    (HERE / "parts.json").write_text(json.dumps(dict(
        generated_by="designlab/data/build_catalog.py",
        research="research/parts_2026-11-07.json; research/compute_2026-11-07.json (round-two compute, P14)",
        parts=out), indent=1))
    return out


def _compute_from_survey() -> dict:
    """Round-two compute (owner choice P14): the survey's architecture C, a Raspberry Pi 5 8 GB with the
    official Active Cooler, the AI HAT+ 13 TOPS (Hailo-8L) and a USB 3 SSD.  Envelope, typical and
    worst-case power, mass and list price are the survey's synthesized values; the mass is its estimate
    (no measured Pi, cooler or HAT masses were found), so the basis is partially sourced."""
    survey = json.loads((HERE / "research" / "compute_2026-11-07.json").read_text())
    arch = {a["id"]: a for a in survey["synthesis"]["architectures"]}["C_pi5_hailo8l_oak_depth"]
    keys = ("raspberry-pi-5-product-brief", "ai-hat-plus-product-brief", "cnx-software.com/2023/11/05",
            "sixfab.com/blog/raspberry-pi-5-ai-hat")
    sources = [f"{s['url']} | accessed 2026-11-07 | {s['what']}" for s in survey["synthesis"]["key_sources"]
               if any(k in s["url"] for k in keys)]
    return dict(
        id="rpi5_8gb_ai_hat_plus_13tops", research_id="C_pi5_hailo8l_oak_depth (research/compute_2026-11-07.json)",
        category="compute", vendor="Raspberry Pi",
        model="Raspberry Pi 5 8 GB + Active Cooler + AI HAT+ 13 TOPS (Hailo-8L) + USB 3 SSD",
        dims_m=[0.085, 0.056, 0.030], mass_kg=arch["mass_g"] / 1000, power_typ_w=float(arch["power_w_typical"]),
        power_max_w=float(arch["power_w_max"]), capacity_wh=0, voltage_v=5.0, max_current_a=5.0,
        price_usd=float(arch["price_usd"]), fov="",
        basis="partially_sourced; envelope, power and price from the survey, mass a survey estimate",
        sources=sources, notes=arch["volume_note"][:500])


def _volt(s: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*V", s or "")
    return float(m.group(1)) if m else 0.0


if __name__ == "__main__":
    a, sk = convert_actuators()
    p = convert_parts()
    print(f"{len(a)} actuators ({len(sk)} excluded), {len(p)} parts")
