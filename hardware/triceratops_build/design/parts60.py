"""Write PARTS_60CM.md, the parts list of the adopted 60 cm design, from its sources so they cannot drift apart:
- masses and positions: the adopted model's mass ledger (sim/generated/mass_ledger_trike60_C_closed_capsule_fore128.json,
  written by sim/build_model60.py --capsule-foot --foreleg 128: option C, the capsule sole, the 128 mm upper foreleg;
  D25, D41). The adopted model is the closed variant (10 % reserve); the "listed" column is derived from it by taking
  the reserve growth back out (the factor in each part's basis text), a derivation this script checks by reproducing
  the 188 mm listed ledger from its closed one;
- prices: the budget script (budget/size_budget.py, option C with the full expression package), plus the lines the
  owner's later decisions add here: the radio stop (D15) and the automatic battery cut-off (D47);
- the previous parts list's figures (the 188 mm box-sole ledger, mass_ledger_trike60_C_closed.json, and the same
  budget lines without D47) for the change table.

Planning estimates only: nothing here is quoted, bought, weighed or measured.
Usage: python parts60.py   (writes PARTS_60CM.md next to this file)
"""

import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "budget"))
import size_budget as sb  # noqa: E402

LEDGER = "mass_ledger_trike60_C_closed_capsule_fore128.json"     # the adopted model (D25, D41)
PREVIOUS = "mass_ledger_trike60_C_closed.json"                    # the parts list of 18 Nov 2026 (188 mm, box sole)
RADIO_STOP_USD = 250   # fail-safe radio stop into the watchdog and Pilz relay (D15; EST, as in size_budget.py)
# D47 (27 Nov 2026): an automatic battery cut-off wired into the stop chain. Prices from the kickoff parts list.
BMS_LINES = [
    ("Safety (battery)", "automatic cut-off (D47): Orion BMS Jr 2, 16-cell, CAN; its discharge-enable output into the "
     "watchdog, so a trip runs the sit-then-cut sequence (PARTS_PREMIUM.md:303); maker's list price, web search 18 Nov "
     "2026 (PARTS_PREMIUM.md:392)", 440.0),
    ("Safety (battery)", "its current sensor, dual-range Hall, which the maker calls optional but strongly recommended "
     "(EST, no price found, PARTS_PREMIUM.md:401)", 100.0),
]
# Parts priced here that the simulation model does not carry (listed mass, EST; to weigh). Being estimates, they are
# grown by the closed variant's reserve factor in the closed column, as the model's own estimated masses are.
NOT_IN_MODEL_KG = [
    ("Orion BMS Jr 2 with its cell-tap harness, and the Hall current sensor (D47)", 0.25,
     "EST: 0.20 kg BMS and harness + 0.05 kg sensor; the maker's mass is unpublished in the records "
     "(PARTS_PREMIUM.md:726 lists it as unchecked) and its specification page was outside this session's network "
     "(27 Nov 2026)"),
]


def family(name):
    n = re.sub(r"^(left|right)_(fore|hind)_", "", name)
    n = re.sub(r"^(left|right)_", "", n)
    n = re.sub(r"_\d+_\d+$", "", n)            # PA11_rib_<ring>_<segment>
    n = re.sub(r"_-?\d+\.\d+_-?1$", "", n)     # tray_stay_<x>_<side>
    n = re.sub(r"_[12]$", "", n)               # gtech_6s_3300_<k>
    return n


def mass_table(ledger):
    """One row per (category, family, basis): where the front and hind parts of a family have different bases (the
    128 mm foreleg's parts are the 188 mm parts' masses times a length ratio; the hind parts are unchanged), they get
    a row each, labelled fore and hind."""
    fam = OrderedDict()
    for r in ledger["parts"]:
        f = family(r["geom"])
        e = fam.setdefault((r["category"], f, r["basis"]), {"n": 0, "kg": 0.0, "pos": [], "ends": set()})
        e["n"] += 1
        e["kg"] += r["mass_kg"]
        e["pos"].append(r["world_pos_m"])
        m = re.match(r"^(?:left|right)_(fore|hind)_", r["geom"])
        e["ends"].add(m.group(1) if m else None)
    split = {(c, f) for (c, f, b) in fam if sum(1 for (c2, f2, _) in fam if (c2, f2) == (c, f)) > 1}
    rows = []
    for (cat, f, basis), e in sorted(fam.items(), key=lambda kv: (kv[0][0], -kv[1]["kg"])):
        xs = [p[0] for p in e["pos"]]
        zs = [p[2] for p in e["pos"]]
        span = f"{1000 * min(xs):+.0f}" if len(set(round(x, 3) for x in xs)) == 1 else \
            f"{1000 * min(xs):+.0f} to {1000 * max(xs):+.0f}"
        label = f
        if (cat, f) in split:
            assert len(e["ends"]) == 1 and None not in e["ends"], (cat, f, e["ends"])
            label = f"{f}, {e['ends'].pop()}"
        rows.append((cat, label, e["n"], e["kg"], span, 1000 * sum(zs) / len(zs), basis))
    return rows


GROWTH = re.compile(r"x (\d\.\d+) reserve growth")


def growth_factor(ledger):
    fs = {float(m.group(1)) for r in ledger["parts"] for m in [GROWTH.search(r["basis"])] if m}
    assert len(fs) == 1, fs
    return fs.pop()


def listed_from_closed(ledger):
    """The listed variant's category masses, derived from a closed ledger: parts grown by the reserve factor (their
    basis says so, or they are the 188 mm part's grown mass times a length ratio) divided by it, the reserve block
    left out, every other part as it is."""
    f = growth_factor(ledger)
    cats = {}
    for r in ledger["parts"]:
        if r["category"] == "reserve":
            continue
        grown = GROWTH.search(r["basis"]) or "the 188 mm part's mass" in r["basis"]
        cats[r["category"]] = cats.get(r["category"], 0.0) + (r["mass_kg"] / f if grown else r["mass_kg"])
    return cats


def costs(cfg, ff, with_bms):
    lines = sb.bom(cfg, frame_fixed=ff)
    _, expr_usd, _ = sb.expression("full", cfg[1])
    hw = sum(l[2] for l in lines) + expr_usd + RADIO_STOP_USD + (sum(u for _, _, u in BMS_LINES) if with_bms else 0)
    duty_base = sum(l[2] for l in lines if l[3])
    landed = {label: sb.overheads(hw, duty_base, rate, brok)[1] for label, rate, brok in sb.DUTY_SCENARIOS}
    return hw, landed


def main(out_path=None):
    out_path = Path(out_path) if out_path else HERE / "PARTS_60CM.md"
    gen = ROOT / "sim" / "generated"
    led = json.loads((gen / LEDGER).read_text())
    prev = json.loads((gen / PREVIOUS).read_text())
    prev_listed_file = json.loads((gen / "mass_ledger_trike60_C_listed.json").read_text())
    # the derivation must reproduce the 188 mm listed ledger from its closed one
    check = listed_from_closed(prev)
    assert abs(sum(check.values()) - prev_listed_file["total_mass_kg"]) < 2e-3, (sum(check.values()))
    for c, kg in prev_listed_file["category_kg"].items():
        assert abs(check.get(c, 0.0) - kg) < 2e-3, (c, check.get(c), kg)
    listed = listed_from_closed(led)
    cfg = next(c for c in sb.CONFIGS if c[0].startswith("C:"))
    lines = sb.bom(cfg, frame_fixed=0.0)
    servos, expr_usd, _ = sb.expression("full", cfg[1])
    grow = growth_factor(led)
    extra_kg = sum(kg for _, kg, _ in NOT_IN_MODEL_KG)
    extra_closed_kg = grow * extra_kg
    total_closed = led["total_mass_kg"]
    pr697 = sum(r["mass_kg"] for r in led["parts"] if r["basis"].startswith("PR #697 part"))
    out = ["# Parts list, 60 cm Triceratops (the adopted design)", "",
           "Generated by `design/parts60.py` on the date of its commit (regenerated 27 Nov 2026 from the adopted model). "
           "**Planning estimates: nothing here is quoted, bought, weighed or measured.** Masses and positions come from "
           "the adopted simulation model's mass ledger (`sim/generated/" + LEDGER + "`, written by "
           "`sim/build_model60.py --capsule-foot --foreleg 128`); prices come from `budget/size_budget.py` (option C, "
           "full expression), whose header gives each source tag: PL = the kickoff parts list (`hardware/"
           "triceratops_kickoff/PARTS_PREMIUM.md`, read 7-18 Nov 2026), RC = the repo catalog (maker pages read 7 Nov "
           "2026), WS = web-search summaries read 18 Nov 2026 (listing pages not opened), EST = estimate; plus the radio "
           "stop (D15) and the battery cut-off (D47) added here.", "",
           "The design (`DECISION_LOG.md`): 60 cm option C, 10 X4 and 2 X6 drives on 12S (D2, D4), the capsule sole "
           "(D19), the 128 mm upper foreleg (D25), adopted for building as the model `sim/generated/"
           "trike60_C_closed_capsule_fore128.xml` (D41); full expression (D14), fail-safe radio stop (D15), an "
           "automatic battery cut-off wired into the stop chain (D47), budget counted landed and all-in (D12). Adopting "
           "the design orders nothing (D41). Open: compute (D3; the Jetson Orin Nano Super is assumed) and the rest of "
           "the chassis (D5).", "",
           "## Mass budget", "",
           "| | Listed (derived) | Closed (10 % reserve): the adopted model |", "|---|---|---|"]
    cats = sorted(set(listed) | set(led["category_kg"]))
    for c in cats:
        out.append(f"| {c} | {listed.get(c, 0):.3f} kg | {led['category_kg'].get(c, 0):.3f} kg |")
    out += [f"| **Total, the model** | **{sum(listed.values()):.3f} kg** | **{total_closed:.3f} kg** |",
            f"| Centre of mass, x (forward of mid-body) | | {1000 * led['com_world_m'][0]:+.1f} mm |",
            f"| Centre of mass, height | | {1000 * led['com_world_m'][2]:.0f} mm |",
            f"| Front share, static estimate from the centre of mass (desk, not a simulation) | | "
            f"{100 * led['front_share_static_estimate']:.1f} % |"]
    for item, kg, why in NOT_IN_MODEL_KG:
        out.append(f"| Not in the model: {item} | +{kg:.3f} kg | +{grow * kg:.3f} kg ({why}; the closed column "
                   f"grows the estimate by x {grow:.4f}, as the model's estimated masses are) |")
    out += [f"| **Total with the parts not in the model** | **{sum(listed.values()) + extra_kg:.3f} kg** | "
            f"**{total_closed + extra_closed_kg:.3f} kg** |", "",
            "The listed column takes the closed variant's reserve back out: every part whose mass is an estimate was "
            f"grown by x {growth_factor(led):.4f}, and the closed variant adds a {led['category_kg'].get('reserve', 0):.3f} "
            "kg block of small parts (connectors, brackets, fasteners) on the chassis; PR #697's added parts were not "
            "grown (A-D22a). The same derivation reproduces the 188 mm listed ledger "
            f"({prev_listed_file['total_mass_kg']:.3f} kg) from its closed one to within 2 g.", "",
            "**Against the budget.** The adopted model is "
            f"{total_closed:.3f} kg, {total_closed - 12.2:+.2f} kg against the budget's 11.8-12.2 kg for option C "
            f"({total_closed + extra_closed_kg - 12.2:+.2f} kg with the battery cut-off's estimated "
            f"{extra_closed_kg:.2f} kg). The 188 mm parts list's "
            "breakdown of the overrun still holds (`sim/REPORT_60CM.md`, \"Mass against the budget\"): limb covers, the "
            "full expression package, the safety block, compute and audio. The battery cut-off is not in the simulation "
            "model; adding it there changes the training model, which the step sheets pin, so that is for the lead and "
            "the owner to decide.", "",
            "## What changed from the parts list of 18 Nov 2026", "",
            "The 18 Nov 2026 list was generated from the 188 mm model with PR #669's box sole "
            f"(`sim/generated/{PREVIOUS}`). Mass, closed variant:", "",
            "| Category | 18 Nov 2026 (188 mm, box sole) | Adopted (128 mm, capsule sole) | Change |", "|---|---|---|---|"]
    for c in sorted(set(prev["category_kg"]) | set(led["category_kg"])):
        a, b = prev["category_kg"].get(c, 0.0), led["category_kg"].get(c, 0.0)
        out.append(f"| {c} | {a:.3f} kg | {b:.3f} kg | {b - a:+.3f} kg |")
    out += [f"| **Total** | **{prev['total_mass_kg']:.3f} kg** | **{total_closed:.3f} kg** | "
            f"**{total_closed - prev['total_mass_kg']:+.3f} kg** |", ""]
    P = {r["geom"]: r for r in prev["parts"]}
    L = {r["geom"]: r for r in led["parts"]}
    fam_delta = OrderedDict()
    for g in sorted(set(P) | set(L)):
        a = P[g]["mass_kg"] if g in P else 0.0
        b = L[g]["mass_kg"] if g in L else 0.0
        if abs(b - a) > 1e-6:
            f = family(g)
            fam_delta[f] = fam_delta.get(f, 0.0) + b - a
    out += ["By part (both sides together): " + "; ".join(f"{f} {d * 1000:+.1f} g" for f, d in fam_delta.items()) + ".",
            "",
            "- **The 128 mm upper foreleg** (D25; PR #697's repackaging, F18): the upper strut, upper limb cover and "
            "pushrod are the 188 mm parts' masses times their length ratio (0.681, 0.681, 0.832; A-D22a), the elbow "
            "axle 45 mm long instead of 40 (x 1.125), and PR #697 adds an offset motor plate and a standoff for each X6, "
            "a drop bracket under each front roll drive and two arms for the ToF sensors, "
            f"{1000 * pr697:.0f} g together at the PR's own estimates (not grown by the reserve factor); "
            "the reserve block is 12 g lighter (0.150 to 0.138 kg). The forelegs' drives and their masses are unchanged; "
            "the shoulder sits 40 mm lower.",
            "- **The capsule sole** (D19): the model gives it the box sole's mass (0.069 kg each); it changes the shape "
            "only. The simulated capsule is a full shape; the part to make is a moulded rubber half-capsule rocker "
            "(`sim/REPORT_FEET60.md`), which has no design and no separate price: it stays inside the frame line's "
            "\"cast soles\".",
            "- **The battery cut-off** (D47): new below, with its estimated mass.", "",
            "## Parts by mass (closed variant, the adopted model)", "",
            "Position: x along the body (+ forward, from mid-body), height z above the floor, at the neutral standing "
            "pose; a range is given where the parts are spread along the body. Where the front and hind parts of one "
            "kind have different mass bases (the 128 mm foreleg's), they have a row each.", "",
            "| Category | Part | Qty | Mass, total | x, mm | z, mm | Basis |", "|---|---|---|---|---|---|---|"]
    for cat, f, n, kg, span, z, basis in mass_table(led):
        out.append(f"| {cat} | {f} | {n} | {kg:.3f} kg | {span} | {z:.0f} | {basis} |")
    out += ["", "## Drives and servos", "", "| Joint | Drive | Ratio | Joint-side limit in the model | Thermal-equilibrium torque at the joint |",
            "|---|---|---|---|---|"]
    seen = set()
    for j, d in led["drives"].items():
        key = re.sub(r"^(left|right)_(fore|hind)_", lambda m: m.group(2) + " ", j)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"| {key} | {d['drive']} | {d.get('ratio', 1.0):.1f}:1 | {d['joint_peak_limit_nm']:.2f} N·m | "
                   f"{d.get('joint_thermal_eq_nm', float('nan')):.2f} N·m |" if "joint_thermal_eq_nm" in d else
                   f"| {key} | {d['drive']} | servo | {d['joint_peak_limit_nm']:.2f} N·m | {sb.SERVO[d['drive']][2]:.2f} N·m (one third of stall, EST) |")
    out += ["", "Leg-drive limits are 0.70 × the maker's peak torque (× 0.90 transmission efficiency on the flexion "
            "joints); servo limits are 0.70 × stall at 12 V. Eyelids (XL330-M288-T) and the beak servo are in the model "
            "as mass only; their joints are not modelled yet.", "",
            "## Cost (hardware, before overheads)", "", "| Category | Item | Price | Duty applies |", "|---|---|---|---|"]
    total = 0.0
    for cat, item, usd, duty in lines:
        out.append(f"| {cat} | {item} | ${usd:,.0f} | {'yes' if duty else ''} |")
        total += usd
    out.append(f"| Expression (full) | {', '.join(f'{n}: {x}' for n, x in servos)}; bus adapter, neck bearing and "
               f"spring, eye LEDs, speaker, microphone, frill lighting, 5 V rail | ${expr_usd:,.0f} | |")
    total += expr_usd
    out.append(f"| Safety | fail-safe radio stop into the watchdog and Pilz relay (D15, EST) | ${RADIO_STOP_USD:,} | |")
    total += RADIO_STOP_USD
    for cat, item, usd in BMS_LINES:
        out.append(f"| {cat} | {item} | ${usd:,.0f} | |")
        total += usd
    out += [f"| **Total** | | **${total:,.0f}** | |", ""]
    # landed cost before and after D47, both frame-cost assumptions, the budget's three duty scenarios
    out += ["Landed cost (`size_budget.py`'s rule: freight 5 %, duty on the drives, brokerage, sales tax 7 %, "
            "contingency 7 %; the owner's duty scenarios, E4). Before = the 18 Nov 2026 list (the budget's chosen "
            "build); after = with the battery cut-off (D47):", "",
            "| Frame cost | Duty scenario | Hardware before | Landed before | Hardware after | Landed after | Change, landed |",
            "|---|---|---|---|---|---|---|"]
    for ff, tag in ((0.0, "by area"), (0.5, "half fixed")):
        hw0, l0 = costs(cfg, ff, False)
        hw1, l1 = costs(cfg, ff, True)
        for label, _, _ in sb.DUTY_SCENARIOS:
            out.append(f"| {tag} | {label} | ${hw0:,.0f} | ${l0[label]:,.0f} | ${hw1:,.0f} | ${l1[label]:,.0f} | "
                       f"+${l1[label] - l0[label]:,.0f} |")
    hw_a, l_a = costs(cfg, 0.0, True)
    out += ["", f"With the frame cost scaled by area and the drives from Fremont stock: hardware ${hw_a:,.0f}, about "
            f"${l_a[sb.DUTY_SCENARIOS[0][0]]:,.0f} landed (the 18 Nov 2026 list: $16,776 and $20,105, "
            "`budget/size_budget_out.txt`, \"Chosen build\"). `size_budget.py` itself does not carry the battery "
            "cut-off yet. Bench gear and tools are in the total; half the bench gear can be shared with the T. rex. "
            "The 128 mm foreleg's added brackets and the capsule rocker are inside the frame line's estimate, not priced "
            "on their own (EST).", "",
            "## Known gaps (to close before ordering)", "",
            "- **Packaging.** The model's chassis is schematic: both crossmembers and the rails pass through drives, "
            "the hind knee drive sits on its own strut's centreline (38.7 mm overlap), the tail pitch servo meets the "
            "pelvic crossmember, and the bay's tightest gaps are 0.5-1.0 mm (`design/PACKAGING_PUSHROD_60CM.md`, "
            "`sim/check_packaging60.py`). One trial layout there (shortened, inset rails with both crossmembers on "
            "their ends, two bay blocks moved 20 mm and the tray stays re-angled to the rails, layout C) puts every "
            "part it moves clear of every solid part over the legs', tail's and head's ranges, by 0.60 mm or more; the "
            "parts it leaves in place keep today's gaps, and it does not fix the hind knees. It adds drive brackets and needs a "
            "new tail mount, inside the frame line's estimate (EST). Chassis CAD under D5. The legs also meet each other "
            "within their ranges: the fore and hind legs on one side under the body, at some poses with the fore leg "
            "pitched aft and the hind leg forward (soles 48 mm, the whole capsule; lower struts and ankle hubs 19.2 mm), "
            "and the two hind soles at the -18 deg inward roll stop (8.49 mm); either needs a joint stop or a coupled "
            "limit, which changes the training model's ranges (the report's items 9 and 10).",
            "- **The pushrods** carry up to about 3.1 kN at the front (the X6's peak at the 43 deg transmission angle): "
            "sized in `design/PACKAGING_PUSHROD_60CM.md` (steel M8 rod ends and rod; crank pins in a clevis, where bending governs); the "
            "\"4 pushrod linkages\" line above is the budget's estimate, not re-priced for that sizing.",
            "- **The foot.** The capsule sole is chosen (D19); the part to make is a moulded rubber half-capsule rocker "
            "with no design yet (`sim/REPORT_FEET60.md`); a bench test of one belongs in the bench plan.",
            "- **The battery cut-off** (D47): wire the BMS's discharge-enable into the watchdog so a trip runs the "
            "sit-then-cut sequence (PARTS_PREMIUM.md:303); the bench plan's stop test (B8) has no BMS trip input yet. "
            "The Chargery BS12 cell alarm in the power block overlaps the BMS's cell monitoring and could go "
            "(not decided; kept in the total).",
            "- **Mass and the hind knees.** The model is over the budget's mass (above). The budget's rule put the hind "
            "X4 knees at 1.00 of their continuous rating at 12.69 kg (F12, an estimate); the four 128 mm policies' "
            "gait reports put them at 0.55-0.90 of it (RMS over the continuous rating, mean of episodes; worst episode "
            "0.95; `rl/analytics/results/fore128-s*/gait_report.json`, simulation), and the worst drive of any kind at "
            "0.83-0.95 in the judged case (F20). Fallbacks: "
            "a 1.2:1 crank ratio, X6 hind knees (+$318 of hardware, +0.92 kg) or a lighter build.",
            "- **Masses are estimates** from PR #669's parts scaled (structure × 0.8^2.5, head and tail × 0.8^3; the "
            "body shell at the parts list's 0.85 kg × 0.8^2), PR #697's estimates for its added parts, plus catalog "
            "masses for bought parts. Weigh the first parts as they arrive.",
            "- **X6 variant.** The US store lists the X6 at 0.78 kg (the X6-50-L); the model uses the X6-60 catalog "
            "mass, 0.82 kg (F9). Question M4 asks which variant ships.",
            "- **TTL servos.** XC430 / XL430 are TTL-bus parts; keep the bus adapter in the head or add an RS-485 "
            "converter for the neck run.",
            "- **Compute** (D3) is assumed, not decided.", ""]
    out_path.write_text("\n".join(out))
    print(f"wrote {out_path.name}: {len(out)} lines; model {total_closed:.3f} kg closed ({sum(listed.values()):.3f} kg "
          f"listed, derived), +{extra_kg:.3f} kg listed / +{extra_closed_kg:.3f} kg closed not in the model; hardware total ${total:,.0f}; landed (Fremont, "
          f"frame by area) ${l_a[sb.DUTY_SCENARIOS[0][0]]:,.0f}")


if __name__ == "__main__":
    main()
