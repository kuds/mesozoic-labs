"""Pushrod sizing for the adopted 60 cm model's 1:1 elbow and knee linkages (DESK calculation; nothing is measured).

Geometry from the adopted model ``sim/generated/trike60_C_closed_capsule_fore128.xml`` through
``sim/check_packaging60.py`` (crank radius, pin-to-pin length, transmission angle over the flexion range, read from the
model's rod and cranks); drive torques from ``sim/data/drives.json`` (X6-60 ``peak_nm`` 60 N m and no ``stall_nm``;
X4-36-L ``peak_nm`` 30 N m and ``stall_nm`` 36 N m; the model's limit is 0.70 x peak at the drive, x 0.90 transmission
efficiency at the joint); walking torques from the four
128 mm policies' gait reports (``rl/analytics/results/fore128-s*/gait_report.json``, simulation, target case).

Rod force = drive torque / (crank radius x sin(transmission angle)). The rod pulls or pushes with the sign of the
torque, so tension and compression are both checked at the same force.

Load cases at the rod (each drive's worst transmission angle, 43 deg at -47 deg flexion):
- ``static``: the static design case, a drive without a firmware limit, or a fault: the largest torque drives.json
  gives for the drive, which can be reached stalled against a load. For the X4-36-L that is its ``stall_nm``, 36 N m,
  above its ``peak_nm`` of 30; for the X6-60 it is ``peak_nm``, 60 N m, because drives.json has no stall figure for it
  (``stall_nm`` null). ``x6_stall_sensitivity`` gives the front at 1.2 x peak, the X4's stall-to-peak ratio, as an
  assumption, not a record.
- ``peak``: the maker's peak torque at the drive.
- ``limit``: the model's 0.70 x peak at the drive (the limit every simulation and the bench plan use).
- ``joint_limit``: the model's joint-side limit (0.70 x peak x 0.90), as the 21 Nov 2026 review computed it.
Fatigue cases: ``F1`` fully reversed +-limit every cycle (a bound no gait reaches); ``F2`` 0 to limit every cycle;
``F3`` fully reversed at sqrt(2) x the worst RMS drive torque of the four 128 mm policies (simulation), at 43 deg.

Checks per candidate (rod material, thread, body diameter, rod end), safety factor 2.0 on static strength and on
buckling against ``static``, 1.5 on fatigue (Goodman; Soderberg would be stricter):
- tension and compression at the thread's tensile stress area (yield; proof stress for class 8.8);
- buckling of the body over the pin-to-pin length, pinned at both ends: Euler, and Johnson's parabola where the
  slenderness is below the transition. It is for most candidates here, so Johnson governs; Euler governs for 7075-T6
  at M5 and M6 (front and hind) and, at the hind's 98 mm, for 6061-T6 M5 and the class 8.8 M5 stud (each
  candidate's ``buckling.governs`` in the output);
- fatigue at the thread root: endurance 0.5 x ultimate (steel) or the 5e8-cycle reversed strength (aluminium), x 0.70
  for axial load, / Kf for the thread;
- the rod end's catalog static rating C0 against ``static``; its dynamic rating C against ``limit`` (a ratio only:
  plain rod-end life needs the maker's method);
- the crank pin, cantilevered from the crank (bending at the crank face) or in a clevis (the rod end's ball between
  two plates): the clevis pin's shear and its bending across the clevis, M = F/2 x (t/2 + g + b/4) for a ball of
  width b between plates of thickness t with a gap g each side (load spread over the ball, reactions over the
  plates), their combined (von Mises) stress taken as if both peaks met at one section (they do not: a bound), and
  the pins' fatigue at the limit force (Goodman; endurance 0.5 x ultimate x the Marin surface factor for a machined
  surface; size factor taken as 1 for 6-10 mm, no stress raiser at the clevis pin's mid-span).

Material and part figures are web-search summaries read 27 Nov 2026 (pages not opened; vendor pages were outside the
session's network), typical values, not design allowables: see ``SOURCES``. Usage: python pushrod60.py
(prints the tables; writes pushrod60_out.json next to this file).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "sim"))
import check_packaging60 as cp  # noqa: E402

SF_STATIC = 2.0
SF_FATIGUE = 1.5
SOURCES = {
    "1018": "AISI 1018 cold drawn: ultimate 440 MPa, yield 370 MPa, E 200 GPa (MatWeb datasheet, "
            "https://matweb.com/search/datasheettext.aspx?matid=6817, web search 27 Nov 2026)",
    "8.8": "property class 8.8 (ISO 898-1): nominal tensile 800 MPa, proof stress 580 MPa (TR Fastenings, "
           "https://www.trfastenings.com/knowledge-base/metric-screws/screws-and-bolts-mechanical-properties, "
           "web search 27 Nov 2026); E 200 GPa taken as for 1018",
    "6061": "6061-T6: ultimate 310 MPa, yield 276 MPa, fatigue 96.5 MPa at 5e8 cycles reversed (R.R. Moore), E 68.9 "
            "GPa, typical, 'not for design' (https://www.aerospacemetals.com/wp-content/uploads/2023/06/"
            "Aluminum-6061-T6-6061-T651.pdf, web search 27 Nov 2026)",
    "7075": "7075-T6: ultimate 572 MPa, yield 503 MPa, fatigue 159 MPa at 5e8 cycles reversed, E 71.7 GPa, typical "
            "(https://quickparts.com/wp-content/uploads/2024/05/Aluminum-7075.pdf, web search 27 Nov 2026; "
            "Wikipedia's T6 range is lower: 510-540 / 430-480 MPa)",
    "threads": "tensile stress areas M5 14.2, M6 20.1, M8 36.6 mm2; minor-diameter basis 0.7854((d2+d3)/2)^2 "
               "(ISO 898-1; https://www.engineeringtoolbox.com/metric-bolts-minimum-ultimate-tensile-proof-loads-d_2026."
               "html, https://roymech.org/Useful_Tables/Screws/Thread_Calcs.html, web search 27 Nov 2026)",
    "Kf": "thread fatigue factor: rolled 2.2 / cut 2.8 for class 5.8 and below, rolled 3.0 / cut 3.8 for 8.8 and "
          "above (https://danotes.mech.uwa.edu.au/threads/fatigue/fatigue.html, web search 27 Nov 2026); aluminium "
          "threads: the steel <= 5.8 cut value, an assumption",
    "endurance": "steel endurance 0.5 x ultimate below 1400 MPa; axial load factor 0.70 "
                 "(https://www.engineering.iastate.edu/~gkstarns/me325/lecture24.pdf, web search 27 Nov 2026)",
    "rod_ends": "steel/steel plain rod ends, DIN ISO 12240-4: M6 C 3.6 kN, C0 8.2 kN (KIPP K0722 via "
                "https://www.marek.eu/data/katalogove-listy/kipp/K0722.pdf and Misumi "
                "https://uk.misumi-ec.com/vona2/detail/221303940723); M8 C 5.8 kN, C0 12.9 kN (norelem 27631-08, "
                "https://norelem.de/en/Product-overview/Systems-and-components-for-machine-and-plant-construction/"
                "27000/Rod-ends/Rod-ends-with-plain-bearing-external-thread-narrow-version-DIN-ISO-12240-4/"
                "Rod-End-with-plain-bearing-steel-galvanised/p/27631-08, figures from a web-search summary of 27 Nov "
                "2026; a second search that day could not read them off the page, which was outside the session's "
                "network; the lowest M8 steel static rating a second search found, 7.9 kN, RS PRO "
                "https://uk.rs-online.com/web/p/rod-ends/1989083, bearing pairing not stated, is carried as "
                "ROD_END_LOW_C0_KN); M5: five listings disagree (dynamic 2.5-7.5 kN, static 3.4-6.2 kN; Halder, Fluro, LDK, RS "
                "PRO on https://uk.rs-online.com), the lowest taken. All web search 27 Nov 2026; ball-bearing and "
                "polymer rod ends excluded (static ratings near 1 kN; the kickoff list asks for steel, "
                "hardware/triceratops_kickoff/PARTS_PREMIUM.md:256)",
    "marin": "Marin surface factor ka = a x Su^b, machined or cold drawn a 4.51 MPa, b -0.265; ground a 1.58, b -0.085 "
             "(https://engineersedge.com/calculators/marin_endurance_limit_15637.htm, "
             "https://fchartsoftware.com/ees/mechanical_design/endurance/hs101.htm, web search 27 Nov 2026)",
}
MATERIALS = {  # Su, Sy (or proof), E (MPa), reversed fatigue strength (MPa) or None for steel (0.5 Su), thread Kf
    "1018 CD steel, cut threads": dict(su=440, sy=370, e=200e3, fat=None, kf=2.8, src="1018"),
    "class 8.8 steel stud, rolled threads": dict(su=800, sy=580, e=200e3, fat=None, kf=3.0, src="8.8"),
    "6061-T6, cut threads": dict(su=310, sy=276, e=68.9e3, fat=96.5, kf=2.8, src="6061"),
    "7075-T6, cut threads": dict(su=572, sy=503, e=71.7e3, fat=159.0, kf=2.8, src="7075"),
}
THREAD_AS = {"M5": 14.2, "M6": 20.1, "M8": 36.6}        # mm2
ROD_END = {"M5": dict(c=2.5, c0=3.4), "M6": dict(c=3.6, c0=8.2), "M8": dict(c=5.8, c0=12.9)}   # kN
ROD_END_LOW_C0_KN = {"M8": 7.9}   # the lowest M8 steel static rating found (pairing unstated): a sensitivity
# Clevis pin: the rod end's ball width b taken as the pin diameter (an 8 mm ball on an 8 mm pin; no maker's sheet in
# the records), plates 4 mm thick, the gap each side 0 or 1 mm (a washer). Assumptions, not a design.
CLEVIS_PLATE_MM = 4.0
CLEVIS_GAPS_MM = (0.0, 1.0)
PIN_SU, PIN_PROOF = 800.0, 580.0   # class 8.8 pin (SOURCES["8.8"])
# Rod: (thread, body diameter mm); a fully threaded stud's body is its stress-area diameter
GEOMS = [("M5", 5.0), ("M6", 6.0), ("M8", 8.0), ("M8", 10.0)]
DRIVE_SIDE = {"fore": "X6-60", "hind": "X4-36-L"}
X6_STALL_OVER_PEAK = 36 / 30   # the X4-36-L's stall_nm / peak_nm in drives.json: an assumption for the X6-60


def drive_torques(kind):
    drv = json.loads((ROOT / "sim" / "data" / "drives.json").read_text())
    d = drv["leg_drives"][DRIVE_SIDE[kind]]
    eta = drv["transmission_efficiency"]
    stall = d.get("stall_nm")
    static = max(d["peak_nm"], stall) if stall is not None else d["peak_nm"]
    basis = (f"stall_nm {stall:g} N m (above peak_nm {d['peak_nm']:g})" if stall is not None and stall > d["peak_nm"]
             else f"peak_nm {d['peak_nm']:g} N m" + (" (no stall_nm in drives.json)" if stall is None else ""))
    return {"static": static, "peak": d["peak_nm"], "stall": stall, "limit": 0.70 * d["peak_nm"],
            "joint_limit": 0.70 * d["peak_nm"] * eta, "eta": eta, "drive": DRIVE_SIDE[kind],
            "static_basis": f"sim/data/drives.json leg_drives[{DRIVE_SIDE[kind]!r}] {basis}"}


def walking_rms():
    """Worst RMS joint torque of the four 128 mm policies' gait reports (target case; simulation), per kind."""
    out = {"fore": (0.0, None), "hind": (0.0, None)}
    for s in range(4):
        f = ROOT / "rl" / "analytics" / "results" / f"fore128-s{s}" / "gait_report.json"
        pj = json.loads(f.read_text())["cases"]["target_0.25"]["summary"]["drives"]["per_joint"]
        for j in ("left_fore_flexion", "right_fore_flexion", "left_hind_flexion", "right_hind_flexion"):
            k = "fore" if "fore" in j else "hind"
            v = pj[j]["rms_nm"]["worst"]
            if v > out[k][0]:
                out[k] = (v, f"fore128-s{s} {j}")
    return out


def geometry():
    """Crank radius, pin-to-pin length and transmission angle against flexion, from the adopted model."""
    m = cp.with_probe_rods(cp.MODEL)
    d = cp.mujoco.MjData(m)
    out = {}
    for c in ("left_fore", "left_hind"):
        lk = cp.linkage(m, c)
        j = m.joint(f"{c}_flexion").id
        rows = []
        for q in np.linspace(*m.jnt_range[j], 167):
            cp.set_pose(m, d, [j], [q])
            ang, L = cp.place_rod(m, d, c, lk)
            rows.append((math.degrees(q), ang, L))
        worst = min(rows, key=lambda r: r[1])
        out["fore" if "fore" in c else "hind"] = {
            "crank_radius_m": lk["crank_radius_m"], "pin_to_pin_m": lk["rod_length_m"],
            "flexion_range_deg": lk["range_deg"], "min_transmission_angle_deg": worst[1], "at_flexion_deg": worst[0],
            "transmission_angle_at_range_ends_deg": [rows[0][1], rows[-1][1]],
            "transmission_angle_at_neutral_deg": min(rows, key=lambda r: abs(r[0]))[1]}
    return out


def johnson_euler(e, sy, dia, L):
    A = math.pi * dia ** 2 / 4
    I = math.pi * dia ** 4 / 64
    k = dia / 4
    slender = L / k
    cc = math.sqrt(2 * math.pi ** 2 * e / sy)
    euler = math.pi ** 2 * e * I / L ** 2
    johnson = A * sy * (1 - sy * slender ** 2 / (4 * math.pi ** 2 * e))
    return {"euler_n": euler, "johnson_n": johnson, "slenderness": slender, "transition": cc,
            "governing_n": johnson if slender < cc else euler, "governs": "Johnson" if slender < cc else "Euler"}


def endurance(mat):
    base = 0.5 * mat["su"] if mat["fat"] is None else mat["fat"]
    return base * 0.70 / mat["kf"]


def goodman_n(sa, sm, se, su):
    return 1.0 / (sa / se + sm / su) if (sa > 0 or sm > 0) else math.inf


def candidate(kind, mat_name, thread, body, forces, L_mm):
    mat = MATERIALS[mat_name]
    As = THREAD_AS[thread]
    stud = "stud" in mat_name
    dia = math.sqrt(4 * As / math.pi) if stud else body
    if stud and body != {"M5": 5.0, "M6": 6.0, "M8": 8.0}[thread]:
        return None                                  # a stud has no separate body
    tens = mat["sy"] * As                            # N, yield (proof) at the thread
    buck = johnson_euler(mat["e"], mat["sy"], dia, L_mm)
    comp = min(tens, buck["governing_n"])
    se = endurance(mat)
    fl = forces["limit"]
    f3 = forces["walk_amp"]
    fat = {"F1 reversed +-limit": goodman_n(fl / As, 0.0, se, mat["su"]),
           "F2 0 to limit": goodman_n(fl / 2 / As, fl / 2 / As, se, mat["su"]),
           "F3 walking, reversed": goodman_n(f3 / As, 0.0, se, mat["su"])}
    re = ROD_END[thread]
    fs = forces["static"]
    res = {"kind": kind, "material": mat_name, "thread": thread, "body_mm": round(dia, 2),
           "tension_capacity_n": tens, "sf_tension_static": tens / fs,
           "buckling": buck, "sf_compression_static": comp / fs,
           "endurance_at_thread_mpa": se, "fatigue_n": fat,
           "rod_end_c0_kn": re["c0"], "sf_rod_end_c0_static": 1000 * re["c0"] / fs,
           "rod_end_limit_over_c": fl / (1000 * re["c"])}
    res["passes"] = (res["sf_tension_static"] >= SF_STATIC and res["sf_compression_static"] >= SF_STATIC
                     and res["sf_rod_end_c0_static"] >= SF_STATIC)
    res["passes_fatigue_bound"] = fat["F1 reversed +-limit"] >= SF_FATIGUE
    res["passes_fatigue_walking"] = fat["F3 walking, reversed"] >= SF_FATIGUE and fat["F2 0 to limit"] >= SF_FATIGUE
    return res


def pin_endurance(su=PIN_SU):
    """Bending endurance of a plain machined pin: 0.5 x Su x ka (machined), size factor 1, no stress raiser."""
    return 0.5 * su * 4.51 * su ** -0.265


def crank_pin(force, pin_d_mm, lever_mm, limit_force, proof_mpa=PIN_PROOF, su=PIN_SU):
    """A pin at ``force`` (the static case) and its fatigue at ``limit_force``: cantilevered from the crank (bending at
    the crank face over ``lever_mm``), or in a clevis (double shear, plus bending across the clevis, see the module
    docstring). Fatigue: F1 fully reversed +-limit, F2 0 to limit (Goodman). The cantilever's figure has no stress
    raiser either; a press fit or a thread at the crank face, where its moment peaks, lowers it further."""
    A = math.pi * pin_d_mm ** 2 / 4
    Z = math.pi * pin_d_mm ** 3 / 32
    se = pin_endurance(su)

    def fatigue(sigma_at_limit):
        return {"F1 reversed +-limit": goodman_n(sigma_at_limit, 0.0, se, su),
                "F2 0 to limit": goodman_n(sigma_at_limit / 2, sigma_at_limit / 2, se, su)}

    bend = force * lever_mm / Z
    out = {"pin_mm": pin_d_mm, "lever_mm": lever_mm, "endurance_mpa": se,
           "cantilever_bending_mpa": bend, "cantilever_sf_proof": proof_mpa / bend,
           "cantilever_fatigue_n": fatigue(limit_force * lever_mm / Z),
           "double_shear_mpa": force / (2 * A), "double_shear_sf_proof_x0.577": 0.577 * proof_mpa / (force / (2 * A)),
           "clevis": []}
    b, t = pin_d_mm, CLEVIS_PLATE_MM
    for g in CLEVIS_GAPS_MM:
        M = force / 2 * (t / 2 + g + b / 4)                    # N mm
        sb, tau = M / Z, force / (2 * A)
        vm = math.sqrt(sb ** 2 + 3 * tau ** 2)
        out["clevis"].append({"ball_width_mm": b, "plate_mm": t, "gap_each_side_mm": g, "moment_nm": M / 1000,
                              "bending_mpa": sb, "shear_mpa": tau, "combined_von_mises_mpa": vm,
                              "sf_proof_combined": proof_mpa / vm,
                              "plate_bearing_pressure_mpa": force / (2 * t * pin_d_mm),
                              "fatigue_n": fatigue(limit_force / force * sb)})
    return out


def main(out_json=None):
    out_json = Path(out_json) if out_json else HERE / "pushrod60_out.json"
    geo = geometry()
    rms = walking_rms()
    out = {"evidence": "desk calculation on the simulation model's geometry; walking torques are simulation",
           "safety_factors": {"static": SF_STATIC, "fatigue": SF_FATIGUE}, "geometry": geo, "sources": SOURCES,
           "cases": {}, "candidates": [], "crank_pins": {}}
    for kind in ("fore", "hind"):
        g = geo[kind]
        rc = g["crank_radius_m"]
        s_worst = math.sin(math.radians(g["min_transmission_angle_deg"]))
        t = drive_torques(kind)
        walk_drive_rms = rms[kind][0] / t["eta"]          # joint-side RMS to the drive side (motoring)
        forces = {"static": t["static"] / (rc * s_worst), "peak": t["peak"] / (rc * s_worst),
                  "limit": t["limit"] / (rc * s_worst),
                  "joint_limit": t["joint_limit"] / (rc * s_worst),
                  "peak_at_90deg": t["peak"] / rc, "limit_at_90deg": t["limit"] / rc,
                  "walk_amp": math.sqrt(2) * walk_drive_rms / (rc * s_worst)}
        out["cases"][kind] = {"drive": t["drive"], "static_basis": t["static_basis"],
                              "torques_nm": {k: t[k] for k in ("static", "peak", "stall", "limit", "joint_limit")},
                              "walking_rms_joint_nm": rms[kind][0], "walking_rms_from": rms[kind][1],
                              "forces_n": forces}
        for mat in MATERIALS:
            for thread, body in GEOMS:
                r = candidate(kind, mat, thread, body, forces, 1000 * g["pin_to_pin_m"])
                if r:
                    out["candidates"].append(r)
        out["crank_pins"][kind] = [crank_pin(forces["static"], dpin, lev, forces["limit"])
                                   for dpin in (6.0, 8.0, 10.0) for lev in (6.0, 10.0)]
        if kind == "fore":
            out["rod_end_sensitivity"] = {f"{th} at C0 {c0} kN, front static case": 1000 * c0 / forces["static"]
                                          for th, c0 in ROD_END_LOW_C0_KN.items()}
            # the X6-60 has no stall figure: the front's recommended parts at 1.2 x its peak (an assumption)
            fx = X6_STALL_OVER_PEAK * t["peak"] / (rc * s_worst)
            rec = candidate(kind, "class 8.8 steel stud, rolled threads", "M8", 8.0, dict(forces, static=fx),
                            1000 * g["pin_to_pin_m"])
            pin8 = crank_pin(fx, 8.0, 6.0, forces["limit"])
            out["x6_stall_sensitivity"] = {
                "assumed_static_nm": X6_STALL_OVER_PEAK * t["peak"], "basis": "1.2 x peak_nm, the X4-36-L's "
                "stall_nm / peak_nm (36 / 30) in drives.json: an assumption, the X6-60 has none", "force_n": fx,
                "class 8.8 M8 rod": {k: rec[k] for k in ("sf_tension_static", "sf_compression_static",
                                                         "sf_rod_end_c0_static", "passes")},
                "rod_end_low_c0_sf": {th: 1000 * c0 / fx for th, c0 in ROD_END_LOW_C0_KN.items()},
                "clevis_pin_8mm_sf_proof_combined": [c["sf_proof_combined"] for c in pin8["clevis"]]}
    # print
    for kind in ("fore", "hind"):
        g, cs = geo[kind], out["cases"][kind]
        print(f"== {kind}: {cs['drive']}, crank {1000 * g['crank_radius_m']:.2f} mm, pin to pin "
              f"{1000 * g['pin_to_pin_m']:.2f} mm, flexion {g['flexion_range_deg'][0]:.0f} to "
              f"{g['flexion_range_deg'][1]:.0f} deg, transmission angle {g['transmission_angle_at_neutral_deg']:.1f} deg "
              f"at neutral, {g['transmission_angle_at_range_ends_deg'][0]:.1f} / "
              f"{g['transmission_angle_at_range_ends_deg'][1]:.1f} deg at the range ends, least "
              f"{g['min_transmission_angle_deg']:.2f} deg at {g['at_flexion_deg']:.1f} deg")
        f = cs["forces_n"]
        print(f"   torques at the drive: static case {cs['torques_nm']['static']:.1f} ({cs['static_basis']}), peak "
              f"{cs['torques_nm']['peak']:.1f}, limit {cs['torques_nm']['limit']:.1f}, joint-side limit "
              f"{cs['torques_nm']['joint_limit']:.2f} N m; walking RMS (worst, joint side) "
              f"{cs['walking_rms_joint_nm']:.2f} N m ({cs['walking_rms_from']})")
        print(f"   rod force at the least angle: static case {f['static']:,.0f} N, peak {f['peak']:,.0f} N, limit "
              f"{f['limit']:,.0f} N, joint-side limit {f['joint_limit']:,.0f} N; at 90 deg: peak {f['peak_at_90deg']:,.0f} "
              f"N, limit {f['limit_at_90deg']:,.0f} N; walking amplitude (F3) {f['walk_amp']:,.0f} N")
        print(f"   {'candidate':40s} {'thr':3s} {'body':>5s} | {'SF tens':>7s} {'SF comp':>7s} {'(gov.)':>8s} "
              f"{'SF C0':>6s} {'lim/C':>6s} | {'n F1':>5s} {'n F2':>5s} {'n F3':>5s} | static fat.bound fat.walk")
        for r in out["candidates"]:
            if r["kind"] != kind:
                continue
            fn = r["fatigue_n"]
            print(f"   {r['material']:40s} {r['thread']:3s} {r['body_mm']:5.2f} | {r['sf_tension_static']:7.2f} "
                  f"{r['sf_compression_static']:7.2f} {r['buckling']['governs']:>8s} {r['sf_rod_end_c0_static']:6.2f} "
                  f"{r['rod_end_limit_over_c']:6.2f} | {fn['F1 reversed +-limit']:5.2f} {fn['F2 0 to limit']:5.2f} "
                  f"{fn['F3 walking, reversed']:5.2f} | {'pass' if r['passes'] else 'FAIL':6s} "
                  f"{'pass' if r['passes_fatigue_bound'] else 'FAIL':9s} {'pass' if r['passes_fatigue_walking'] else 'FAIL'}")
        print("   crank pin at the static-case force (class 8.8 proof 580 MPa; fatigue n F1 / F2 at the limit force, "
              f"endurance {pin_endurance():.0f} MPa):")
        for p in out["crank_pins"][kind]:
            fc = p["cantilever_fatigue_n"]
            print(f"     d {p['pin_mm']:.0f} mm, lever {p['lever_mm']:.0f} mm: cantilever {p['cantilever_bending_mpa']:.0f} "
                  f"MPa (SF {p['cantilever_sf_proof']:.2f}; n {fc['F1 reversed +-limit']:.2f} / "
                  f"{fc['F2 0 to limit']:.2f})")
            if p["lever_mm"] == 6.0:
                for c in p["clevis"]:
                    fn = c["fatigue_n"]
                    print(f"       clevis, ball {c['ball_width_mm']:.0f} mm, plates {c['plate_mm']:.0f} mm, gap "
                          f"{c['gap_each_side_mm']:.0f} mm: shear {c['shear_mpa']:.0f}, bending {c['bending_mpa']:.0f}, "
                          f"combined {c['combined_von_mises_mpa']:.0f} MPa (SF {c['sf_proof_combined']:.2f}); plate "
                          f"bearing {c['plate_bearing_pressure_mpa']:.0f} MPa; n {fn['F1 reversed +-limit']:.2f} / "
                          f"{fn['F2 0 to limit']:.2f}")
    print("   rod-end static rating sensitivity (SF against the front static case):",
          {k: round(v, 2) for k, v in out["rod_end_sensitivity"].items()})
    xs = out["x6_stall_sensitivity"]
    print(f"   front at {xs['assumed_static_nm']:.0f} N m ({xs['basis']}): rod force {xs['force_n']:,.0f} N; class 8.8 M8 "
          f"rod SF tension {xs['class 8.8 M8 rod']['sf_tension_static']:.2f}, compression "
          f"{xs['class 8.8 M8 rod']['sf_compression_static']:.2f}, rod end C0 {xs['class 8.8 M8 rod']['sf_rod_end_c0_static']:.2f}"
          f" ({'pass' if xs['class 8.8 M8 rod']['passes'] else 'FAIL'}); lowest M8 rod end "
          f"{list(xs['rod_end_low_c0_sf'].values())[0]:.2f}; 8 mm clevis pin, combined, SF "
          + " / ".join(f"{v:.2f}" for v in xs["clevis_pin_8mm_sf_proof_combined"]))
    out_json.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {out_json.name}")
    return out


if __name__ == "__main__":
    main()
