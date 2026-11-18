"""Rough size, performance and budget options for the Triceratops robot (owner's question, 18 Nov 2026).

Planning estimates only: nothing here is quoted, bought or measured. Every rule is stated below; prices carry a
source tag: PL = hardware/triceratops_kickoff/PARTS_PREMIUM.md (its own sources, read 7-18 Nov 2026); RC = the repo
catalog (hardware/trex_form_study/designlab/data/*.json, maker pages read 7 Nov 2026); MD = docs/hardware/
RAPTOR_SCALING_AND_ALTERNATIVES.md (24 Jul 2026); WS = web-search result summaries read 18 Nov 2026 (listing pages
not opened); EST = this script's estimate, to be quoted. Revised 18 Nov 2026 after an independent check.

Run: python size_budget.py   (prints the tables used in SIZE_BUDGET_OPTIONS.md)
"""

import math

HIP_75, LEN_75 = 0.40, 1.50     # m, PR #669's 75 cm robot: shoulder/hip height, overall length

# Drives (catalog, per unit): peak N·m, thermal-equilibrium N·m, kg, $, rated V, no-load rpm at rated V
DRIVES = {"X4-36-L": dict(peak=30, therm=7.5, kg=0.36, usd=269, v=24, rpm=111),
          "X6-60": dict(peak=60, therm=12.0, kg=0.82, usd=428, v=48, rpm=176)}

# Configurations: (name, scale s = crown / 0.75 m, bus, [(count, drive, axis, ratio)])
CONFIGS = [
    ("A: 50 cm, 6S, 12 x X4", 0.50 / 0.75, "6S",
     [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0), (2, "X4-36-L", "elbow", 1.0), (2, "X4-36-L", "knee", 1.0)]),
    ("B: 60 cm, 6S, 12 x X4", 0.60 / 0.75, "6S",
     [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0), (2, "X4-36-L", "elbow", 1.0), (2, "X4-36-L", "knee", 1.0)]),
    ("C: 60 cm, 12S, 10 x X4 + 2 x X6 front elbows", 0.60 / 0.75, "12S",
     [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0), (2, "X6-60", "elbow", 1.0), (2, "X4-36-L", "knee", 1.0)]),
    ("D: 75 cm, 12S, 8 x X4 + 4 x X6 at 1.3:1 (the parts list's T15)", 1.0, "12S",
     [(4, "X4-36-L", "roll", 1.0), (4, "X4-36-L", "pitch", 1.0), (2, "X6-60", "elbow", 1.3), (2, "X6-60", "knee", 1.3)]),
]
BUS_MIN_V = {"6S": 21.0, "12S": 42.0}
PACK_WH = {"6S": 73.3, "12S": 146.5}   # one 6S 3300 pack; two in series (PL)

# Loads at the reference, the parts list's T15 (15 kg, 0.40 m hips, 55 % front), joint side (section 2.3, Appendix A).
REF = dict(mass=15.0, v_walk=0.30, v_trot=0.60,
           walk=dict(elbow=24.0, pitch=14.0, roll=14.6), held=dict(elbow=12.2, pitch=5.5, roll=6.5),
           trot=dict(elbow=26.9, pitch=13.8, roll=19.1), trot_rms=dict(elbow=11.3, pitch=5.5, roll=7.6),
           rpm_walk=dict(pitch=59.9, flex=50.1, roll=30.0), rpm_trot=dict(pitch=45.3, flex=47.0, roll=30.0),
           rpm_walk_pr669=dict(pitch=69.0, flex=48.1, roll=30.0))
ETA = 0.90


def mass_estimate(s, drive_kg, head_servo_kg, battery_kg, frame_power=3):
    structure = 2.85 * s ** frame_power + 4 * 0.05 * s ** 2   # T15 frame/links/feet less foot cells, + linkages (EST)
    head = 0.835 * s ** 3 + head_servo_kg
    tail = 0.325 * s ** 3
    shells = 0.85 * s ** 2
    compute = 0.55
    power = 0.75 + 0.30 * s ** 2
    return 1.10 * (drive_kg + structure + head + tail + shells + compute + battery_kg + power)


def check(s, mass, dset, bus, gait="walk", profile="rpm_walk"):
    """Use of each axis (<= 1.00 passes) at the Froude-similar speed for the gait."""
    k = (mass / REF["mass"]) * s
    w = 1 / math.sqrt(s)
    out = {}
    for n, d, axis, ratio in dset:
        D = DRIVES[d]
        allow = 83.0 if (d == "X4-36-L" and BUS_MIN_V[bus] > D["v"]) else 0.70 * D["rpm"] * BUS_MIN_V[bus] / D["v"]
        base = "elbow" if axis in ("elbow", "knee") else axis
        share = 45 / 55 if axis == "knee" else 1.0
        tr = (ratio * ETA) if base == "elbow" else 1.0
        if gait == "walk":
            pk = share * k * REF["walk"][base] / tr
            cont = share * k * REF["held"][base] / tr
            rpmref = REF[profile]["flex" if base == "elbow" else base]
        else:
            pk = share * k * REF["trot"][base] / tr
            cont = share * k * REF["trot_rms"][base] / tr
            rpmref = REF["rpm_trot"]["flex" if base == "elbow" else base]
        rpm = w * rpmref * (ratio if base == "elbow" else 1.0)
        out[axis] = dict(drive=d, ratio=ratio, peak=pk / (0.70 * D["peak"]), heat=cont / D["therm"],
                         speed=rpm / allow, rpm=rpm, allow=allow)
    return out


def walk_speed(s, chk, limit):
    v = REF["v_walk"] * math.sqrt(s)
    return min(v, v * limit / max(r["speed"] for r in chk.values()))


def scale_cost(c75, s, fixed=0.0):
    """Cost of a part at scale s: a fixed share plus the rest scaling with area (EST)."""
    return round(c75 * (fixed + (1 - fixed) * s ** 2), -1)


def bom(cfg, frame_fixed):
    name, s, bus, dset = cfg
    lines = []
    add = lambda cat, item, usd, duty=False: lines.append((cat, item, round(usd, 2), duty))
    n4 = sum(n for n, d, _, _ in dset if d == "X4-36-L")
    n6 = sum(n for n, d, _, _ in dset if d == "X6-60")
    add("Leg drives", f"{n4} x RMD-X4-P36-36-E-L" + (f" + {n6} x RMD-X6-P20-60-E" if n6 else ""), n4 * 269 + n6 * 428, True)
    add("Leg drives", "4 pushrod linkages", 4 * scale_cost(120, s), False)
    add("Spares", "spare X4" + (" + spare X6" if n6 else ""), 269 + (428 if n6 else 0), True)
    add("Spares", "rod ends, fuses, connectors", scale_cost(250, s), False)
    add("Compute", "Jetson Orin Nano Super kit + NVMe; 2 Teensy 4.1 CAN carriers", 399 + 65 + 300)
    add("Sensors", "BNO085 IMU, OAK-D Lite + locking cable, 2 x VL53L8CX", 24.95 + 269 + 30 + 49.90)
    add("Safety (wired)", "Pilz PNOZ s5, 2 x 2-NC e-stops + pendant, contactor, fuse, watchdog, 24 V rail",
        611.99 + 100 + 40 + 54.45 + 20 + 80 + 35)
    if bus == "6S":
        add("Power", "2 x 6S 3300, pack monitor, power board, 2 x 12 V rails, 24 V tether supply + lead, charger, LiPo bags",
            214.38 + 124.45 + 200 + 119.90 + 200 + 90 + 447.89 + 60)
    else:
        add("Power", "the parts list's T15 power block, less its 24 V rail (in the safety line) and cooling", 2024 - 35 - 90)
    add("Cooling", "drive-bay blowers, ducts, filters (PL F11)", 90)
    add("Wiring", "harness, limb quick-disconnects", scale_cost(550, s))
    add("Frame", "outsourced CNC frame and links, bearings, cast soles, fasteners, belly skid",
        scale_cost(3100 + 220 + 300 + 150 + 80, s, frame_fixed))
    add("Shells", "MJF PA11 shells + soft tips, team-painted (materials)", scale_cost(1750 + 150, s) + 150)
    add("Bench", "60 V supply, torque lever rig, PCAN-USB FD, thermal camera, overhead catch gantry", 400 + 400 + 379 + 250 + 800)
    add("Tools", "crimpers, torque drivers, meter", 350)
    return lines


# $, kg, continuous limit N·m. XM parts: the repo catalog's rated torque (RC). XC430 / XL430: one third of the stall
# torque found by web search (1.9 / 1.5 N·m at 12 V; WS), an estimate. XC430 and XL430 "-T" are TTL-bus parts: keep the
# bus adapter in the head or add an RS-485 converter for the long neck run (the parts list chose RS-485 servos).
SERVO = {"XC430-W240-T": (132, 0.065, 1.9 / 3), "XL430-W250-T": (55, 0.057, 1.5 / 3), "XM430-W350-R": (333.39, 0.082, 1.20),
         "XM540-W270-R": (494.39, 0.165, 3.11), "XL330-M288-T": (27.49, 0.018, None)}


def expression(level, s):
    big = s > 0.9
    neck, other = ("XM540-W270-R", "XM430-W350-R") if big else ("XC430-W240-T", "XC430-W240-T")
    tail = "XM430-W350-R" if big else "XL430-W250-T"
    servos = [("neck pitch", neck), ("neck yaw", other), ("jaw / beak", other), ("tail sway", tail)]
    extras = [("U2D2 bus adapter + hub", 55), ("neck yaw bearing + counterbalance spring", 115),
              ("eye LEDs, speaker, amplifier, microphone", 125)]
    if level == "full":
        servos += [("eyelid L", "XL330-M288-T"), ("eyelid R", "XL330-M288-T"), ("tail lift", tail)]
        extras += [("frill blush lighting", 100), ("5 V rail for the eyelids", 20)]
    usd = sum(SERVO[x][0] for _, x in servos) + sum(c for _, c in extras)
    return servos, usd, sum(SERVO[x][1] for _, x in servos)


def neck_check(s, level):
    """Neck pitch hold torque, scaled from the T20 head (1.51 kg; 34.9 mm horizontal lever in the model, 110 mm in the
    parts list's conservative sizing), against the servo's continuous limit (see SERVO)."""
    servos, _, kg = expression(level, s)
    head_kg = 0.835 * s ** 3 + sum(SERVO[x][1] for n, x in servos if n.startswith(("neck", "jaw", "eyelid")))
    limit = SERVO[servos[0][1]][2]
    out = {}
    for label, lever in (("model lever", 0.0349 * s), ("parts-list lever", 0.110 * s)):
        tq = head_kg * 9.81 * lever
        out[label] = (tq, tq * 0.30, limit)   # no spring, spring cancelling 70 %, continuous limit
    return head_kg, out


ADDONS = [("Certified wireless safety stop, PL d, handler-held, with receiver (only if a museum requires it; D15)",
           1900 + 800 - 250, "PL: Cattron $1,900 (WS); receiver EST $0-1,500; quote; net of the radio stop it replaces"),
          ("Foot load cells x 4 + per-leg links (21-23 week lead)", 174.28 + 80, "PL"),
          ("Xsens MTi-3 IMU instead of BNO085", 417.06 - 24.95, "PL"),
          ("OAK-D Pro W instead of OAK-D Lite", 529 - 269, "PL/RC"),
          ("Professional prop painter instead of team painting", 1200 - 150, "PL EST"),
          ("Padded road case", 800, "PL EST")]


def overheads(hw, duty_base, duty=0.25, brokerage=0.0):
    """T. rex way: hardware + 7.2 %. Landed: freight 5 %, duty on the drives, brokerage, sales tax 7 %, contingency 7 %."""
    trex = hw * 1.072
    landed = (hw + 0.05 * hw + duty * duty_base + brokerage + 0.07 * hw) * 1.07
    return trex, landed


# Duty scenarios for the MyActuator drives (owner-reported 18 Nov 2026, to confirm per order): items in stock at the
# US store's Fremont warehouse have cleared customs (no duty); units dispatched from the factory in China pay about
# 41.5 % in stacked US tariffs (HTS 8501.31.40.00) plus brokerage (no de minimis exemption since August 2025).
DUTY_SCENARIOS = [("Fremont stock, duty paid (0 %)", 0.0, 0.0), ("planning nominal so far (25 %)", 0.25, 0.0),
                  ("factory dispatch from China (41.5 % + brokerage, $150 EST)", 0.415, 150.0)]


def main():
    for cfg in CONFIGS:
        name, s, bus, dset = cfg
        drive_kg = sum(n * DRIVES[d]["kg"] for n, d, _, _ in dset)
        _, _, hs = expression("core", s)
        batt = 1.012 if bus == "12S" else 0.506
        m3 = 15.0 if s > 0.9 else mass_estimate(s, drive_kg, hs, batt, 3)
        m2 = 15.0 if s > 0.9 else mass_estimate(s, drive_kg, hs, batt, 2)
        print(f"== {name}")
        print(f"   crown {0.75 * s:.2f} m, length {LEN_75 * s:.2f} m, hips {HIP_75 * s:.3f} m; mass ~{m3:.1f} kg "
              f"({m2:.1f} kg if frame mass scales with area)")
        walk = check(s, m3, dset, bus)
        v_hi = walk_speed(s, walk, 1.00)
        v_mid = walk_speed(s, walk, 0.85)
        v_lo = walk_speed(s, check(s, m3, dset, bus, profile="rpm_walk_pr669"), 0.85)
        print(f"   walk speed: {v_lo:.2f}-{v_hi:.2f} m/s (PR #669 swing profile with 15 % margin / min-jerk with 15 % "
              f"margin {v_mid:.2f} / min-jerk at the parts list's rule); Froude-similar {REF['v_walk'] * math.sqrt(s):.2f}")
        for axis, r in walk.items():
            print(f"   walk  {axis:5s} {r['drive']:8s} {r['ratio']}:1  peak {r['peak']:.2f}  heat {r['heat']:.2f}  "
                  f"speed {r['speed']:.2f} ({r['rpm']:.0f} of {r['allow']:.0f} rpm)")
        if s < 0.9:
            w2 = check(s, m2, dset, bus)
            print("   heat if frame mass scales with area: " + ", ".join(f"{a} {r['heat']:.2f}" for a, r in w2.items()))
        trot = check(s, m3, dset, bus, gait="trot")
        print(f"   trot at {REF['v_trot'] * math.sqrt(s):.2f} m/s (Froude-similar to 0.60 m/s): " +
              ", ".join(f"{a} pk {r['peak']:.2f} heat {r['heat']:.2f} spd {r['speed']:.2f}" for a, r in trot.items()))
        p_lo, p_hi = 150 * m3 / 15, 210 * m3 / 15
        print(f"   run time on one battery set ({PACK_WH[bus]} Wh, 80 % usable, {p_lo:.0f}-{p_hi:.0f} W by the parts "
              f"list's rule scaled by mass): {0.8 * PACK_WH[bus] / p_hi * 60:.0f}-{0.8 * PACK_WH[bus] / p_lo * 60:.0f} min")
        for lvl in ("core", "full"):
            head_kg, nk = neck_check(s, lvl)
            print(f"   neck ({lvl}): head ~{head_kg:.2f} kg; " + "; ".join(
                f"{k}: {a:.2f} N·m no spring, {b:.2f} with spring, limit {c:.2f}" for k, (a, b, c) in nk.items()))
        for ff, tag in ((0.0, "frame cost by area"), (0.5, "half the frame cost fixed")):
            lines = bom(cfg, ff)
            hw0 = sum(l[2] for l in lines)
            duty = sum(l[2] for l in lines if l[3])
            for lvl in ("core", "full"):
                _, e, _ = expression(lvl, s)
                for wl, extra in (("wired stop", 0), ("museum-ready (+ wireless stop)", 2700)):
                    hw = hw0 + e + extra
                    trex, landed = overheads(hw, duty)
                    print(f"   [{tag}] {lvl:4s} expr ${e:,.0f}, {wl:31s}: hardware ${hw:,.0f}; T. rex way ${trex:,.0f}; "
                          f"landed ${landed:,.0f}")
        cats = {}
        for cat, item, usd, _ in bom(cfg, 0.5):
            cats[cat] = cats.get(cat, 0) + usd
        print("   hardware by category (half frame fixed, before expression):", {k: round(x) for k, x in cats.items()})
    print("\n== Duty scenarios, option C (60 cm), full expression, museum-ready ==")
    cfg = CONFIGS[2]
    for ff, tag in ((0.0, "frame cost by area"), (0.5, "half the frame cost fixed")):
        lines = bom(cfg, ff)
        hw = sum(l[2] for l in lines) + expression("full", cfg[1])[1] + 2700
        duty_base = sum(l[2] for l in lines if l[3])
        for label, rate, brok in DUTY_SCENARIOS:
            _, landed = overheads(hw, duty_base, rate, brok)
            print(f"   [{tag}] {label:58s}: landed ${landed:,.0f}")
    drives = sum(l[2] for l in bom(cfg, 0.0) if l[3])
    print(f"   if the store's 20 % educational / research discount applied to the drives (${drives:,.0f}): "
          f"-${0.2 * drives:,.0f} of hardware (eligibility not known)")
    print("\n== Chosen build (owner, 18 Nov 2026): option C, full expression, fail-safe radio stop (~$250 EST) ==")
    for ff, tag in ((0.0, "frame cost by area"), (0.5, "half the frame cost fixed")):
        lines = bom(cfg, ff)
        hw = sum(l[2] for l in lines) + expression("full", cfg[1])[1] + 250
        duty_base = sum(l[2] for l in lines if l[3])
        for label, rate, brok in DUTY_SCENARIOS:
            trex, landed = overheads(hw, duty_base, rate, brok)
            print(f"   [{tag}] hardware ${hw:,.0f}; T. rex way ${trex:,.0f}; {label:58s}: landed ${landed:,.0f}")
        _, fremont = overheads(hw, duty_base, 0.0, 0.0)
        saved = (2229 / 2 + 697) * 1.12 * 1.07
        print(f"   [{tag}] Fremont stock, half the bench gear counted against the T. rex, spares bought later: "
              f"landed ${fremont - saved:,.0f}")
    print("\n== Optional add-ons (hardware, before overheads) ==")
    for item, usd, note in ADDONS:
        print(f"   +${usd:,.0f}  {item}  [{note}]")


if __name__ == "__main__":
    main()
