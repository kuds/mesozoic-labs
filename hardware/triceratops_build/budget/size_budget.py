"""Rough size, performance and budget options for the Triceratops robot (owner's question, 18 Nov 2026).

Planning estimates only: nothing here is quoted, bought or measured. Every rule is stated below; prices carry a
source tag: PL = hardware/triceratops_kickoff/PARTS_PREMIUM.md (its own sources, read 7-18 Nov 2026); RC = the repo
catalog (hardware/trex_form_study/designlab/data/*.json, maker pages read 7 Nov 2026); MD = docs/hardware/
RAPTOR_SCALING_AND_ALTERNATIVES.md (24 Jul 2026); WS = web-search result summaries read 18 Nov 2026 (listing pages
not opened); EST = this script's estimate (scaling or allowance), to be quoted.

Run: python size_budget.py   (prints the tables used in SIZE_BUDGET_OPTIONS.md)
"""

import math

# ---- Configurations: geometric scale s = crown height / 0.75 m, PR #669's 75 cm robot as the reference --------------
# (name, s, bus, drive set) ; drive set = list of (count, drive, axis, ratio)
CONFIGS = [
    ("A: 50 cm, 6S, 12 x X4", 0.50 / 0.75, "6S", [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0),
                                                  (4, "X4-36-L", "elbow", 1.0)]),
    ("B: 60 cm, 6S, 12 x X4", 0.60 / 0.75, "6S", [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0),
                                                  (4, "X4-36-L", "elbow", 1.0)]),
    ("C: 60 cm, 12S, 10 x X4 + 2 x X6 (the T. rex's drive set)", 0.60 / 0.75, "12S",
     [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0), (2, "X6-60", "elbow", 1.0),
      (2, "X4-36-L", "knee", 1.0)]),
    ("D: 75 cm, 12S, 8 x X4 + 4 x X6 at 1.3:1 (the parts list's T15)", 1.0, "12S",
     [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0), (4, "X6-60", "elbow", 1.3)]),
]
BUS_MIN_V = {"6S": 21.0, "12S": 42.0}
HIP_75 = 0.40          # m, PR #669 shoulder and hip height
LEN_75 = 1.50          # m, overall length

# ---- Drives (catalog, per unit) -------------------------------------------------------------------------------------
DRIVES = {  # peak N·m, thermal-equilibrium N·m (None = no table), kg, $, source, rated V, no-load rpm at rated V
    "X4-36-L": dict(peak=30, therm=7.5, kg=0.36, usd=269, src="RC/PL", v=24, rpm=111),
    "X6-60": dict(peak=60, therm=12.0, kg=0.82, usd=428, src="RC/PL", v=48, rpm=176),
    "RobStride 02": dict(peak=17, therm=None, kg=0.405, usd=145, src="RC", v=48, rpm=410),
}

# ---- Leg loads at the reference: the parts list's T15 (15 kg, 0.40 m hips, 55 % front), joint side, N·m ----------
# Section 2.3 and Appendix A. Walk = the 0.30 m/s museum gait (x1.5 dynamic factor); held = one forefoot held up ("paw").
REF = dict(mass=15.0, elbow_walk=24.0, elbow_held=12.2, pitch_walk=14.0, pitch_cont=5.5, roll_walk=14.6,
           roll_held=6.5, pitch_rpm=59.9, flex_rpm=50.1, v=0.30)
ETA = 0.90             # flexion transmission efficiency (parts list)

# ---- Mass model (kg) for a lean build; reference values from the parts list's T15 column (section 6.2) -----------
def mass_estimate(s, drive_kg, n_head_servos_kg, battery_kg):
    structure = 2.85 * s ** 3 + 4 * 0.05 * s ** 2       # T15 frame/links/feet less foot cells, plus linkages (EST)
    head = 0.835 * s ** 3 + n_head_servos_kg               # PR head structure scaled, plus the chosen servos
    tail = 0.325 * s ** 3                                  # tail shell and core (its servo counted with the head's)
    shells = 0.85 * s ** 2                                 # MJF body shells scale with area (EST)
    compute = 0.55                                         # Jetson, NVMe, Teensy carrier, camera, IMU, ToF (EST)
    power = 0.75 + 0.30 * s ** 2                           # wired safety chain, power board, rails, harness (EST)
    listed = drive_kg + structure + head + tail + shells + compute + battery_kg + power
    return listed * 1.10                                   # 10 % reserve


def froude_speed(s):
    """Walking speed dynamically similar to 0.30 m/s at 75 cm: v ~ sqrt(leg length)."""
    return REF["v"] * math.sqrt(s)


def drive_check(s, mass, drives, bus_v):
    """Use of each axis against the parts list's rules at the Froude-similar walk (<= 1.00 passes)."""
    k = (mass / REF["mass"]) * s          # torque scales with weight x leg length
    w = 1 / math.sqrt(s)                   # joint speeds at a Froude-similar walk scale as v / L
    out = {}
    for axis, d, ratio in drives:
        D = DRIVES[d]
        allow_rpm = (0.70 * D["rpm"] * bus_v / D["v"]) if bus_v <= D["v"] else 83.0 if d == "X4-36-L" else \
            0.70 * D["rpm"] * bus_v / D["v"]
        if axis in ("elbow", "knee"):
            share = 1.0 if axis == "elbow" else 45 / 55       # hind legs carry 45 % against the front's 55 %
            pk = share * k * REF["elbow_walk"] / (ratio * ETA)
            cont = share * k * REF["elbow_held"] / (ratio * ETA)
            rpm = w * REF["flex_rpm"] * ratio
        elif axis == "pitch":
            pk, cont, rpm = k * REF["pitch_walk"], k * REF["pitch_cont"], w * REF["pitch_rpm"]
        else:
            pk, cont, rpm = k * REF["roll_walk"], k * REF["roll_held"], 30.0 * w
        out[axis] = dict(drive=d, ratio=ratio, peak_use=pk / (0.70 * D["peak"]),
                         heat_use=(cont / D["therm"]) if D["therm"] else None, speed_use=rpm / allow_rpm,
                         rpm_need=rpm, rpm_allow=allow_rpm)
    return out


# ---- Bills of materials (nominal $) ---------------------------------------------------------------------------------
def scale_cost(c75, s, power=2):
    return round(c75 * s ** power, -1)


def bom(cfg):
    """Lean but legitimate core build: catalog drives, industrial wired stop chain, named sensors."""
    name, s, bus, dset = cfg
    small = s < 0.9
    lines = []
    add = lambda cat, item, usd, src: lines.append((cat, item, round(usd, 2), src))
    n_x4 = sum(n for n, d, _, _ in dset if d == "X4-36-L")
    n_x6 = sum(n for n, d, _, _ in dset if d == "X6-60")
    add("Leg drives", f"{n_x4} x MyActuator RMD-X4-P36-36-E-L" + (f" + {n_x6} x RMD-X6-P20-60-E" if n_x6 else ""),
        n_x4 * 269 + n_x6 * 428, "RC/PL $269 / $428")
    add("Leg drives", "4 pushrod linkages (cranks, rod ends, bearings)", 4 * scale_cost(120, s), "PL $120 EST, scaled")
    add("Spares", "1 spare X4" + (" + 1 spare X6" if n_x6 else "") + " + rod ends, fuses, connectors",
        269 + (428 if n_x6 else 0) + scale_cost(250, s), "RC + PL EST")
    add("Compute", "Jetson Orin Nano Super kit + NVMe", 399 + 65, "WS 18 Nov / PL")
    add("Compute", "Teensy 4.1 CAN carrier x 2 (1 spare)", 300, "PL EST")
    add("Sensors", "BNO085 IMU, OAK-D Lite camera + locking cable, 2 x VL53L8CX", 24.95 + 269 + 30 + 49.90, "MD/RC")
    add("Safety (wired)", "Pilz PNOZ s5 relay, 2 x 2-NC e-stops + pendant, contactor, fuse, watchdog, 24 V rail",
        611.99 + 100 + 40 + 54.45 + 20 + 80 + 35, "PL (WS/MD/EST)")
    if bus == "6S":
        add("Power (6S)", "2 x 6S 3300 packs, pack monitor, power board, 12 V rails, 24 V tether supply + lead, "
            "charger, LiPo bags", 214.38 + 124.45 + 200 + 119.90 + 200 + 90 + 447.89 + 60, "PL/MD; tether supply EST")
    elif small:
        add("Power (12S)", "the parts list's T15 power block less cooling, with its 4 packs (2 sets)", 2024 - 90, "PL")
    else:
        add("Power (12S)", "the parts list's T15 power block", 2024, "PL")
    add("Wiring", "harness, limb quick-disconnects", scale_cost(550, s), "PL EST scaled")
    add("Frame", "outsourced CNC 6061 frame and links, bearings, cast soles, fasteners, belly skid",
        scale_cost(3100 + 220 + 300 + 150 + 80, s), "PL EST scaled by area")
    add("Shells", "MJF PA11 shells + soft tips, painted by the team (materials)",
        scale_cost(1750 + 150, s) + 150, "PL EST scaled; paint materials EST")
    add("Bench (minimum)", "60 V bench supply, simple torque lever rig, PCAN-USB FD", 400 + 400 + 379, "PL")
    add("Tools", "crimpers, torque drivers, meter", 350, "PL EST")
    return lines


# Expressiveness packages per size: (servo list, extras). Servo prices: XC430-W240 ~$132 (WS: $100-197),
# XL430-W250 ~$55 (WS: $45-70), XM430-W350-R $333.39 (PL), XM540-W270-R $494.39 (PL), XL330 $27.49 (PL).
SERVO = {"XC430-W240": (132, 0.065, "1.9 N·m stall @12 V"), "XL430-W250": (55, 0.057, "1.5 N·m stall @12 V"),
         "XM430-W350-R": (333.39, 0.082, "rated 1.2 N·m"), "XM540-W270-R": (494.39, 0.165, "rated 3.1 N·m"),
         "XL330-M288": (27.49, 0.018, "eyelids")}


def expression(level, s):
    """Head/tail packages. 'core': neck nod + turn, jaw/beak, tail sway. 'full': + blink, tail lift, frill lights."""
    big = s > 0.9
    neck = "XM540-W270-R" if big else "XC430-W240"
    other = "XM430-W350-R" if big else "XC430-W240"
    servos = [("neck pitch (nod, spring-assisted)", neck), ("neck yaw (look at visitors)", other),
              ("jaw / beak open-close", other), ("tail sway (yaw)", "XL430-W250" if not big else "XM430-W350-R")]
    extras = [("servo bus adapter + power hub (U2D2)", 55), ("neck yaw bearing + counterbalance spring", 115),
              ("eye LEDs + speaker, amp and microphone", 125)]
    if level == "full":
        servos += [("eyelids (blink) x 2", "XL330-M288"), ("eyelid 2", "XL330-M288"),
                   ("tail lift (pitch)", "XL430-W250" if not big else "XM430-W350-R")]
        extras += [("frill blush lighting", 100), ("5 V rail for the eyelid servos", 20)]
    usd = sum(SERVO[sv][0] for _, sv in servos) + sum(c for _, c in extras)
    kg = sum(SERVO[sv][1] for _, sv in servos)
    return servos, extras, usd, kg


ADDONS = [  # (item, nominal $, note)
    ("Wireless safety stop, PL d (handler-held), with machine-side receiver", 1900 + 800,
     "PL (Cattron $1,900 WS; receiver EST $0-1,500). Strongly advised before public demos"),
    ("Foot load cells x 4 + per-leg I2C links (21-23 week lead)", 174.28 + 80, "PL"),
    ("Xsens MTi-3 IMU instead of BNO085", 417.06 - 24.95, "PL"),
    ("OAK-D Pro W instead of OAK-D Lite (dim halls)", 529 - 269, "PL/RC"),
    ("Professional prop painter instead of team painting", 1200 - 150, "PL EST"),
    ("Padded road case for museum travel", 800, "PL EST"),
    ("Fuller bench: thermal camera, catch gantry", 250 + 800, "PL EST"),
]


def overheads(hw, drives_usd):
    trex_way = hw * 1.072
    freight, duty, tax = 0.05 * hw, 0.25 * drives_usd, 0.07 * hw
    landed = (hw + freight + duty + tax) * 1.07
    return trex_way, landed


def max_walk(s, chk):
    """Fastest walk with every axis's speed use <= 0.85, capped at the Froude-similar speed."""
    worst = max(r["speed_use"] for r in chk.values())
    return min(froude_speed(s), froude_speed(s) * 0.85 / worst)


def main():
    rows = []
    for cfg in CONFIGS:
        name, s, bus, dset = cfg
        drive_kg = sum(n * DRIVES[d]["kg"] for n, d, _, _ in dset)
        _, _, _, head_kg = expression("core", s)
        batt = 1.012 if bus == "12S" else 0.506
        m = 15.0 if s > 0.9 else mass_estimate(s, drive_kg, head_kg, batt)  # 75 cm: the parts list's closed T15
        chk = drive_check(s, m, [(ax, d, r) for _, d, ax, r in dset], BUS_MIN_V[bus])
        v = max_walk(s, chk)
        lines = bom(cfg)
        hw_core = sum(l[2] for l in lines)
        drives_usd = sum(l[2] for l in lines if l[0] == "Leg drives" and "linkage" not in l[1])  # duty base
        print(f"== {name}")
        print(f"   crown {0.75 * s:.2f} m, length {LEN_75 * s:.2f} m, hips {HIP_75 * s:.3f} m, mass ~{m:.1f} kg; "
              f"Froude-similar walk {froude_speed(s):.2f} m/s; fastest walk with 15 % speed margin {v:.2f} m/s "
              f"({v / (LEN_75 * s):.2f} body lengths/s)")
        for axis, r in chk.items():
            heat = f"{r['heat_use']:.2f}" if r["heat_use"] is not None else "n/a"
            print(f"   {axis:5s} {r['drive']:8s} {r['ratio']}:1  peak {r['peak_use']:.2f}  heat {heat}  "
                  f"speed at the Froude walk {r['speed_use']:.2f} ({r['rpm_need']:.0f} of {r['rpm_allow']:.0f} rpm)")
        cats = {}
        for cat, item, usd, src in lines:
            cats[cat] = cats.get(cat, 0) + usd
        print("   core hardware by category:", {k: round(x) for k, x in cats.items()})
        for lvl in ("core", "full"):
            servos, extras, e_usd, _ = expression(lvl, s)
            for wl, extra in (("wired stop only", 0), ("+ wireless stop", 2700)):
                hw = hw_core + e_usd + extra
                trex, landed = overheads(hw, drives_usd)
                print(f"   {lvl:4s} expression (${e_usd:,.0f}), {wl:16s}: hardware ${hw:,.0f}; T. rex way ${trex:,.0f}; "
                      f"landed with contingency ${landed:,.0f}")
    print("\n== Add-ons (nominal $, hardware before overheads) ==")
    for item, usd, note in ADDONS:
        print(f"   +${usd:,.0f}  {item}  [{note}]")
    print("\n== Expression packages ==")
    for cfg in CONFIGS:
        name, s = cfg[0], cfg[1]
        for lvl in ("core", "full"):
            servos, extras, usd, kg = expression(lvl, s)
            print(f"{name.split(',')[0]} {lvl}: ${usd:,.0f}, servos {kg:.2f} kg: " +
                  "; ".join(f"{a} {b}" for a, b in servos))


if __name__ == "__main__":
    main()
