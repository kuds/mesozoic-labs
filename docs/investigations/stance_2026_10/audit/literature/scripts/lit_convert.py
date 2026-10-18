"""Convert published T. rex / bird / human foot numbers into the model's units.

Reads (all under the literature/ directory given as argv[1]):
  out/model_frame.json                 kinematics-only model measurement (lit_model_frame.py)
  out/brochu_fig_*_digitised.json      digitised Brochu 2003 figures (lit_digitise_fig.py)
  scripts/published_inputs.json        hand-entered published values with locators
  dl/persons_currie2016/PMC4728391.xml Persons & Currie 2016 (Table 2 parsed)
  dl/hutchinson2011/PMC3192160.xml     Hutchinson et al. 2011 (Tables 3, 4, 5, 7 parsed)
  dl/mccrea2014/PMC4108409.xml         McCrea et al. 2014 (Table 1 parsed)
Writes out/brochu_pes_summary.json, out/audit_rows.json, out/load_centre_frame.json,
out/literature_crosscheck.md.  No MuJoCo, no simulation.

Usage: python -I lit_convert.py <literature dir>
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

D = Path(sys.argv[1])
OUT = D / "out"
mf = json.loads((OUT / "model_frame.json").read_text())
P = json.loads((D / "scripts/published_inputs.json").read_text())
R = mf["legs_home"]["r"]


def f(x, n=2):
    return f"{x:.{n}f}"


def rng(a, b, n=2, unit=""):
    lo, hi = min(a, b), max(a, b)
    return f"{lo:.{n}f}-{hi:.{n}f}{unit}" if round(lo, n) != round(hi, n) else f"{lo:.{n}f}{unit}"


# ---------------------------------------------------------------- model frame
PAD_C = R["pad_centre_ahead_of_mtp3"]
HALF = R["pad_half_length"]
PAD_R = R["pad_rear_ahead_of_mtp3"]
PAD_F = R["pad_front_ahead_of_mtp3"]
FP = R["footprint_pad_rear_to_d3_tip"]
D3 = R["digit_d3_len_mtp_to_tip_with_cap"]
D3N = R["digit_d3_len_mtp_to_tip_nocap"]
MT = R["metatarsus_len_ankle_to_mtp3"]
FEM = R["femur_len"]
TIB = R["tibia_len"]
HIP_H = R["hip_height_above_floor"]


def stat(x):
    return (x - PAD_C) / HALF


def frame(x):
    return {"x_ahead_of_mtp3_m": x, "pad_stat_signed": stat(x), "frac_pad_from_rear": (x - PAD_R) / (PAD_F - PAD_R),
            "frac_footprint_from_rear": (x - PAD_R) / FP, "frac_digit3_from_mtp": x / D3}


def x_from_fp(fr):
    return PAD_R + fr * FP


def x_from_stat(s):
    return PAD_C + s * HALF


# ---------------------------------------------------------------- table parsing
def tables(xml):
    r = ET.parse(xml).getroot()
    out = {}
    for tw in r.iter("table-wrap"):
        lab = tw.find("label")
        rows = [["".join(c.itertext()).strip() for c in row] for row in tw.iter("tr")]
        out[lab.text if lab is not None else "?"] = rows
    return out


def num(s):
    m = re.search(r"[-−–]?\d+(\.\d+)?", s.replace("−", "-").replace("–", "-"))
    return float(m.group(0).replace("−", "-")) if m else None


# Persons & Currie 2016 Table 2
pc = tables(D / "dl/persons_currie2016/PMC4728391.xml")["Table 2"]
pc_rex = {}
on = False
for row in pc:
    if len(row) == 1 or (len(row) >= 1 and row[0] and all(c == "" for c in row[1:])):
        on = row[0].startswith("Tyrannosaurus rex")
        continue
    if on and len(row) == 5:
        pc_rex[row[0]] = {"femur": num(row[1]), "tibia": num(row[2]), "mt3": num(row[3])}
adult_ids = [k for k in pc_rex if not k.startswith("LACM")]
juv_id = [k for k in pc_rex if k.startswith("LACM")][0]
pc_ratio = {k: {"mt_f": v["mt3"] / v["femur"], "t_f": v["tibia"] / v["femur"], "leg_f": (v["femur"] + v["tibia"] + v["mt3"]) / v["femur"]} for k, v in pc_rex.items()}

# Hutchinson 2011 tables
ht = tables(D / "dl/hutchinson2011/PMC3192160.xml")
h3 = {row[0]: {"femur": num(row[1]), "leg": num(row[4])} for row in ht["Table 3"][1:] if len(row) >= 5 and num(row[1])}
seg = {}
for tab in ("Table 4", "Table 5"):
    for row in ht[tab][1:]:
        if len(row) >= 5 and num(row[4]) is not None:
            seg.setdefault((row[1], row[2]), {})[row[0]] = num(row[4])
h7 = {}
cur = None
for row in ht["Table 7"]:
    if len(row) >= 12 and row[0] and all(c == "" for c in row[1:]):
        cur = row[0]
        continue
    if cur and len(row) >= 12 and num(row[1]) is not None:
        h7.setdefault(cur, {})[row[0]] = {"x_pct_fem": num(row[4]), "y_pct_fem": num(row[5]), "x_pct_leg": num(row[10]), "y_pct_leg": num(row[11]),
                                         "x_m": num(row[2]), "y_m": num(row[3])}
h6 = {}
for row in ht["Table 6"][1:]:
    if len(row) >= 7 and num(row[1]):
        h6[row[0]] = {"min_x": num(row[2]), "min_y": num(row[3]), "max_x": num(row[5]), "max_y": num(row[6])}
adults = ["Carnegie", "Sue", "Stan", "MOR"]

# McCrea 2014 Table 1
mt1 = tables(D / "dl/mccrea2014/PMC4108409.xml")["Table 1"]
prints = []
tw = None
for row in mt1[2:]:
    if len(row) < 27:
        continue
    tw = row[0] or tw
    if row[1] in ("X",) or row[2] in ("-", ""):
        continue
    def g(i):
        return num(row[i]) if row[i] not in ("-", "") else None
    prints.append({"trackway": tw, "print": row[1], "FL": g(2), "DL": [g(5), g(6), g(7)], "FDL": [g(8), g(9), g(10)], "FDL_flag": ["*" in row[8], "*" in row[9], "*" in row[10]],
                   "DIV": [g(14), g(15), g(16)], "DD": [g(17), g(18), g(19)], "MPD": g(20)})

# Brochu digitised figures
def comps(fid):
    return json.loads((OUT / f"brochu_fig_{fid}_digitised.json").read_text())["components"]


def big(cs, mn=5.0):
    return [c for c in cs if c.get("h_cm", 0) > mn]


dors = big(comps(3740187))
vent = big(comps(3740189))
# dorsal view (Fig. 105): columns digit IV | digit III | digit II | digit I ungual (right pes, distal down)
def col(cs, x_lo, x_hi):
    return sorted([c for c in cs if x_lo <= c["x0"] < x_hi], key=lambda c: c["y0"])
d_IV, d_III, d_II, d_I = col(dors, 0, 450), col(dors, 450, 1000), col(dors, 1000, 1478), col(dors, 1478, 9999)
# ventral view (Fig. 106) is mirrored: digit I ungual | digit II | digit III | digit IV
v_I, v_II, v_III, v_IV = col(vent, 0, 152), col(vent, 152, 690), col(vent, 690, 1240), col(vent, 1240, 9999)
assert [len(d_IV), len(d_III), len(d_II), len(d_I)] == [3, 4, 3, 1], [len(d_IV), len(d_III), len(d_II), len(d_I)]
assert [len(v_I), len(v_II), len(v_III), len(v_IV)] == [1, 3, 4, 3], [len(v_I), len(v_II), len(v_III), len(v_IV)]
mt3_views = [c["h_cm"] for c in big(comps(3740035), 30)]
mt2_views = [c["h_cm"] for c in big(comps(3740183), 30)]
mt4_views = [c["h_cm"] for c in big(comps(3740185), 30)]
br = {"source": "Brochu 2003 Figs. 99-101, 105-106 (Zenodo), digitised with lit_digitise_fig.py against each plate's 10 cm scale bar; vertical bounding-box extent, approximate (+-~1 cm)",
      "dorsal_cm": {"III": [c["h_cm"] for c in d_III], "II": [c["h_cm"] for c in d_II], "IV_preserved": [c["h_cm"] for c in d_IV], "I_ungual": [c["h_cm"] for c in d_I]},
      "ventral_cm": {"III": [c["h_cm"] for c in v_III], "II": [c["h_cm"] for c in v_II], "IV_preserved": [c["h_cm"] for c in v_IV], "I_ungual": [c["h_cm"] for c in v_I]},
      "MTIII_views_cm": mt3_views, "MTII_views_cm": mt2_views, "MTIV_views_cm": mt4_views}
sumIII = [sum(br["dorsal_cm"]["III"]), sum(br["ventral_cm"]["III"])]
sumII = [sum(br["dorsal_cm"]["II"]), sum(br["ventral_cm"]["II"])]
III1_frac = [br["dorsal_cm"]["III"][0] / sumIII[0], br["ventral_cm"]["III"][0] / sumIII[1]]
III12_frac = [sum(br["dorsal_cm"]["III"][:2]) / sumIII[0], sum(br["ventral_cm"]["III"][:2]) / sumIII[1]]
ung3_frac = [br["dorsal_cm"]["III"][3] / sumIII[0], br["ventral_cm"]["III"][3] / sumIII[1]]
II1_frac = [br["dorsal_cm"]["II"][0] / sumII[0], br["ventral_cm"]["II"][0] / sumII[1]]
mt3 = sum(mt3_views) / len(mt3_views)
sue_fem_cm = P["brochu2003"]["femur_m_from_hutchinson2011_table3"] * 100
br.update({"sum_III_cm": sumIII, "sum_II_cm": sumII, "III1_frac": III1_frac, "III1_2_frac": III12_frac, "III_ungual_frac": ung3_frac, "II1_frac": II1_frac,
           "MTIII_mean_cm": mt3, "MTIII_range_cm": [min(mt3_views), max(mt3_views)],
           "MTIII_over_femur": [min(mt3_views) / sue_fem_cm, max(mt3_views) / sue_fem_cm],
           "sumIII_over_MTIII": [min(sumIII) / mt3, max(sumIII) / mt3],
           "FLproxy_over_MTIII": [1.05 * min(sumIII) / mt3, 1.05 * max(sumIII) / mt3],
           "FLproxy_over_femur": [1.05 * min(sumIII) / sue_fem_cm, 1.05 * max(sumIII) / sue_fem_cm],
           "sumII_over_sumIII": [sumII[0] / sumIII[0], sumII[1] / sumIII[1]]})
dup = [(a["trackway"] + a["print"], b["trackway"] + b["print"]) for i, a in enumerate(prints) for b in prints[i + 1:] if a["DD"] == b["DD"] and a["MPD"] == b["MPD"]]
br_note_dup = dup
(OUT / "brochu_pes_summary.json").write_text(json.dumps(br, indent=1))

# Hutchinson 2005 pes segment
h05 = P["hutchinson2005"]
pes = h05["pes_segment_m"]["proximal_III1"] + h05["pes_segment_m"]["distal"]
mor_mt3 = pc_rex["MOR 555"]["mt3"] / 1000
mor_fem = pc_rex["MOR 555"]["femur"] / 1000
h05_III1_frac = h05["pes_segment_m"]["proximal_III1"] / pes

# Boeye 2D
bt = P["boeye2026"]["table1_2D_m"]
boeye_pes = {k: v["complete_leg"] - v["FTM"] for k, v in bt.items() if not k.startswith("_")}

# ---------------------------------------------------------------- audit rows
rows = []


def row(q, model, pub, reading, ev):
    rows.append({"quantity": q, "model": model, "published": pub, "reading": reading, "evidence": ev})


MF = "out/model_frame.json"
mtf = [pc_ratio[k]["mt_f"] for k in adult_ids]
tf = [pc_ratio[k]["t_f"] for k in adult_ids]
row("Metatarsus / femur (model: ankle-to-MTP III joint distance; published: MT III bone length / femur)",
    f"{f(mf['ratios']['metatarsus_over_femur'],3)} ({f(MT,4)} / {f(FEM,4)} m)",
    f"adults {rng(min(mtf), max(mtf), 3)} (MOR 555 {f(pc_ratio['MOR 555']['mt_f'],3)}, CM 9380 {f(pc_ratio['CM 9380 (cast of AMNH 973)']['mt_f'],3)}, RTMP 81.12.1 {f(pc_ratio['RTMP 81.12.1 (cast of NMC 9950)']['mt_f'],3)}, BHI 6230 {f(pc_ratio['BHI 6230']['mt_f'],3)}); juvenile LACM 23845 {f(pc_ratio[juv_id]['mt_f'],3)}; Sue from Brochu Fig. 100 (digitised MT III {rng(*br['MTIII_range_cm'],1)} cm / femur {f(sue_fem_cm,1)} cm) {rng(*br['MTIII_over_femur'],3)} [figure estimate]",
    f"Model sits at the juvenile LACM 23845 value and above all adults. Even with Hutchinson 2005's soft-tissue padding (metatarsus +10%, femur +7.5%), adults reach {rng(min(mtf)*1.10/1.075, max(mtf)*1.10/1.075, 3)}, so the model is at the top of that range (BHI 6230) [padding arithmetic is inference].",
    f"{MF}; Persons & Currie 2016 Table 2 (doi:10.1038/srep19828, parsed from dl/persons_currie2016/PMC4728391.xml); Brochu 2003 Fig. 100 (zenodo 3740035) via out/brochu_pes_summary.json; Hutchinson 2011 Table 3 femur")
row("Tibia / femur",
    f"{f(mf['ratios']['tibia_over_femur'],3)}",
    f"adults {rng(min(tf), max(tf), 3)}; juvenile LACM 23845 {f(pc_ratio[juv_id]['t_f'],3)}",
    f"Model = juvenile proportion; adults 0.90-0.93 ({rng(min(tf)*1.05/1.075, max(tf)*1.05/1.075, 3)} with Hutchinson 2005 padding). Confirmed.",
    f"{MF}; Persons & Currie 2016 Table 2")
legf_pc = [pc_ratio[k]["leg_f"] for k in adult_ids]
legf_h = [h3[k]["leg"] / h3[k]["femur"] for k in adults]
row("Hindlimb length (femur+tibia+metatarsus) / femur",
    f"{f(mf['ratios']['limb_len_sum_over_femur'],3)}",
    f"adults {rng(min(legf_pc), max(legf_pc), 2)} (Persons & Currie sums); Hutchinson 2011 Table 3 'leg length'/femur {rng(min(legf_h), max(legf_h), 2)}; juvenile LACM 23845 {f(pc_ratio[juv_id]['leg_f'],2)}",
    "Model limb is long for its femur, again juvenile-like (driven by tibia 1.00 and metatarsus 0.61).",
    f"{MF}; Persons & Currie 2016 Table 2; Hutchinson 2011 Table 3 (dl/hutchinson2011/PMC3192160.xml)")
row("Digit III length / metatarsus (model: MTP III joint to capsule tip)",
    f"{f(D3/MT,2)} with the capsule end-cap ({f(D3,3)} m), {f(D3N/MT,2)} without ({f(D3N,3)} m)",
    f"MOR 555 model pes segment {f(pes,3)} m (Hutchinson 2005 App. 2) / MT III {f(mor_mt3,3)} m = {f(pes/mor_mt3,2)}; Sue summed digit III phalanges {rng(*sumIII,1)} cm / MT III {f(mt3,1)} cm = {rng(*br['sumIII_over_MTIII'],2)} (summing overstates the articulated digit) [figure estimate]; Boeye 2026 Table 1 'complete leg minus femur+tibia+tarsometatarsus' / Persons & Currie MT III: MOR 555 {f(boeye_pes['MOR 555']/mor_mt3,2)}, LACM 23845 {f(boeye_pes['LACM 23845']/(pc_rex[juv_id]['mt3']/1000),2)} [2D reconstructions, indirect]",
    "Inside the published spread (0.75-0.98) without the cap and slightly above it with the cap. The digits are not over-long relative to the metatarsus.",
    f"{MF}; Hutchinson 2005 Appendix 2 (dl/hutchinson2005/h2005.txt); out/brochu_pes_summary.json; Boeye 2026 Table 1")
dl2 = [p["DL"][0] / p["DL"][1] for p in prints if p["DL"][0] and p["DL"][1] and not p["FDL_flag"][0]]
dl4 = [p["DL"][2] / p["DL"][1] for p in prints if p["DL"][2] and p["DL"][1]]
m_dl2 = (R["tip_d2_ahead_of_mtp3"] - PAD_R) / FP
m_dl4 = (R["tip_d4_ahead_of_mtp3"] - PAD_R) / FP
row("Digit II and IV length relative to digit III",
    f"from the pad rear (track convention): II {f(m_dl2,2)}, IV {f(m_dl4,2)} of III; from MTP joints: II {f(mf['ratios']['digit2_over_digit3_with_cap'],2)}, IV {f(mf['ratios']['digit4_over_digit3_with_cap'],2)}",
    f"Bellatoripes digit lengths from the track rear (prints without a flagged digit II): II/III {rng(min(dl2), max(dl2),2)}, IV/III {rng(min(dl4), max(dl4),2)} (McCrea 2014 Table 1); Sue summed phalanges II/III {rng(*br['sumII_over_sumIII'],2)} (digit IV incomplete) [figure estimate]",
    "Within or near the track ranges; digit II slightly short against Sue's skeleton. Not a fore-aft concern.",
    f"{MF}; McCrea 2014 Table 1; out/brochu_pes_summary.json")
row("Proximal phalanx (III-1) share of digit III, which locates the 'proximal phalanges' load zone",
    f"no phalanges in the model (one rigid capsule); the pad centre sits {f(PAD_C/D3*100,1)}% of the way along digit III",
    f"MOR 555: III-1 = {f(h05['pes_segment_m']['proximal_III1'],3)} of {f(pes,3)} m = {f(h05_III1_frac,3)} (Hutchinson 2005 App. 2); Sue: III-1 / summed digit III = {rng(*III1_frac,3)}, III-1+III-2 = {rng(*III12_frac,2)}, ungual III-4 = {rng(*ung3_frac,2)} [figure estimate]",
    f"On the model's digit III ({f(D3,3)} m) the III-1 zone spans 0 to {f(min(III1_frac)*D3,3)}-{f(h05_III1_frac*D3,3)} m ahead of MTP III, i.e. pad statistic -0.86 to {f(stat(min(III1_frac)*D3),2)}..{f(stat(h05_III1_frac*D3),2)}. The pad centre (0) lies inside it [inference].",
    f"{MF}; Hutchinson 2005 App. 2; Brochu 2003 Figs. 105-106 via out/brochu_pes_summary.json")
xs = [h7[s]["Minimal"]["x_pct_fem"] for s in adults] + [h7[s]["Maximal"]["x_pct_fem"] for s in adults]
xs_leg = [h7[s]["Minimal"]["x_pct_leg"] for s in adults] + [h7[s]["Maximal"]["x_pct_leg"] for s in adults]
sue6 = [h6["Sue"]["min_x"] / h3["Sue"]["femur"] * 100, h6["Sue"]["max_x"] / h3["Sue"]["femur"] * 100]
b09 = P["bates2009"]["best_estimate_com_rel_hip_m"]
b_pct = [b09["MOR 555"][0] / (pc_rex["MOR 555"]["femur"] / 1000) * 100, b09["BHI 3033"][0] / h3["Stan"]["femur"] * 100]
row("COM ahead of hip",
    f"{f(mf['com']['com_ahead_of_hip_pct_femur'],1)}% femur ({f(mf['com']['com_ahead_of_hip_m'],3)} m); {f(mf['com']['com_ahead_of_hip_m']/(FEM+TIB+MT)*100,1)}% of femur+tibia+metatarsus",
    f"'most plausible' 30-45% femur, Sue 44-61% (Hutchinson 2011 text) = {f(0.30*FEM,3)}-{f(0.45*FEM,3)} m on the model femur; Table 7 adult minimal/maximal models {rng(min(xs), max(xs), 1)}% femur, {rng(min(xs_leg), max(xs_leg), 1)}% leg length; Bates 2009 best estimates {rng(*b_pct, 1)}% femur (MOR 555 0.468 m, BHI 3033 0.572 m)",
    f"Inside the band. Note: Hutchinson 2011 Table 7 lists Sue at 66.8/46.0% femur, but Table 6 x / Table 3 femur gives {rng(*sue6,1)}%, which matches the text's 44-61%.",
    f"{MF}; Hutchinson 2011 Tables 3, 6, 7 + text; Bates 2009 Table 8 (dl/bates2009/PMC2639725.xml)")
ys = [h7[s]["Minimal"]["y_pct_fem"] for s in adults] + [h7[s]["Maximal"]["y_pct_fem"] for s in adults]
ys_leg = [h7[s]["Minimal"]["y_pct_leg"] for s in adults] + [h7[s]["Maximal"]["y_pct_leg"] for s in adults]
bv = [-b09["MOR 555"][1] / (pc_rex["MOR 555"]["femur"] / 1000) * 100, -b09["BHI 3033"][1] / h3["Stan"]["femur"] * 100]
row("COM below hip",
    f"{f(mf['com']['com_below_hip_pct_femur'],1)}% femur, i.e. {f(-mf['com']['com_below_hip_m'],4)} m ABOVE the hip",
    f"22-35% (Carnegie, MOR), 26-28% (Sue), 38-50% (Stan) femur below (Hutchinson 2011 text) = {f(0.22*FEM,3)}-{f(0.50*FEM,3)} m below on the model femur; Table 7 adult min/max {rng(-max(ys), -min(ys), 1)}% femur, {rng(-max(ys_leg), -min(ys_leg), 1)}% leg length below; Bates 2009 {rng(*bv,1)}% femur below",
    f"Mismatch: model COM is {f(0.22*FEM - mf['com']['com_below_hip_m'],3)}-{f(0.50*FEM - mf['com']['com_below_hip_m'],3)} m higher relative to the hip than any published model, which raises mgh and the toppling rate.",
    f"{MF}; Hutchinson 2011 text + Table 7; Bates 2009 Table 8")
mp = mf["segment_mass_pct"]
def segr(s, m):
    v = [seg[(s, m)][k] for k in adults]
    return min(v) * 100, max(v) * 100
th, sh, fo = segr("Thigh", "min"), segr("Shank", "min"), segr("Foot", "min")
he = segr("Head", "all"); ne = segr("Neck", "min"); ta = segr("Tail", "min")
limbs_model = 2 * (mp["thigh_one"] + mp["shank_one"] + mp["foot_one_incl_digits"])
limbs_pub = [2 * (seg[("Thigh", "min")][k] + seg[("Shank", "min")][k] + seg[("Foot", "min")][k]) * 100 for k in adults]
row("Segment mass shares (% body mass; limb segments per single limb)",
    f"head {f(mp['head'],1)}, neck {f(mp['neck'],1)}, tail {f(mp['tail'],1)}; per limb thigh {f(mp['thigh_one'],1)}, shank {f(mp['shank_one'],1)}, foot {f(mp['foot_one_incl_digits'],1)}; both hindlimbs {f(limbs_model,1)}",
    f"adult minimal models: head {rng(*he,1)}, neck {rng(*ne,1)}, tail {rng(*ta,1)}; per limb thigh {rng(*th,1)}, shank {rng(*sh,1)}, foot {rng(*fo,1)}; both hindlimbs {rng(min(limbs_pub), max(limbs_pub),1)} (Table 10 gives 35-65 for left+right, upper-end estimates)",
    f"Per-limb reading confirmed by Table 10 footnote ('for left and right limbs'). Model thigh is {f(th[0]/mp['thigh_one'],1)}-{f(th[1]/mp['thigh_one'],1)}x too light and the hindlimbs carry about half the published share; shank slightly light; head, neck, tail, foot inside.",
    f"{MF}; Hutchinson 2011 Tables 4, 5, 10 (parsed)")
bel = P["mccrea2014_text"]["hip_height_m"]
fl_hol = 62.0
fls = [p["FL"] for p in prints if p["FL"]]
row("Footprint length / hip height",
    f"{f(mf['ratios']['footprint_over_hip_height'],3)} ({f(FP,3)} / {f(HIP_H,4)} m)",
    f"Bellatoripes holotype FL {f(fl_hol,1)} cm (prints {rng(min(fls), max(fls),1)} cm) with hip height {bel['flexed']}-{bel['tyrannosaurid_eq']} m: {rng(fl_hol/100/bel['tyrannosaurid_eq'], fl_hol/100/bel['flexed'],3)}; Alexander's h ~ 4 FL gives {f(1/P['mccrea2014_text']['alexander_h_over_fl'],2)}; Sue FL proxy (1.05 x summed digit III) / femur {rng(*br['FLproxy_over_femur'],2)} vs model footprint/femur {f(FP/FEM,2)}",
    "Inside the trackway-derived band. Footprint/femur runs high in the model, but the model femur is short.",
    f"{MF}; McCrea 2014 Table 1 + Discussion (dl/mccrea2014/PMC4108409.xml); out/brochu_pes_summary.json")
mtp4 = R["mtp_d4_ahead_of_mtp3"]
row("Rear contact (track 'heel')",
    f"midline box pad, rear edge {f(PAD_R,3)} m (statistic {f(stat(PAD_R),2)}) under MTP III; digit IV MTP joint at {f(mtp4,3)} m fore-aft, {f(abs(R['mtp_d4_y_minus_mtp3_y']),3)} m lateral",
    f"digit IV MTP pad commonly impressed, II-III MTP pads held clear (Lallensack 2022); Bellatoripes 'metatarsal pad impression present with a wide caudal margin' (McCrea 2014). In the model frame the track's rear edge lies at or behind the MTP joint: statistic <= {f(stat(0.0),2)}, footprint fraction <= {f(frame(0.0)['frac_footprint_from_rear'],3)} [inference]",
    "Fore-aft placement of the rear edge agrees (within 1 cm). The side is wrong: the published rear contact is the lateral digit IV pad, not a midline pad under III.",
    f"{MF}; Lallensack 2022 (dl/lallensack2022/l2022.pdf); McCrea 2014 diagnosis")
dd_ratio = [p["MPD"] / p["DD"][1] for p in prints if p["MPD"] and p["DD"][1]]
row("Load centre (where weight sits)",
    f"pad centre {f(PAD_C,3)} m ahead of MTP III = {f(PAD_C/D3*100,1)}% of digit III (statistic 0); home COM projection statistic {f(mf['com']['com_projection_signed_pad_stat'],3)}",
    f"tracks: depressions 'around the proximal pedal phalanges' (Boeye 2026) = statistic -0.86 to {f(stat(min(III1_frac)*D3),2)}..{f(stat(h05_III1_frac*D3),2)}; Bellatoripes metatarsal-pad depth = {rng(min(dd_ratio), max(dd_ratio),2)} of digit III depth (McCrea 2014 Table 1); ostrich digit III 77-84% of load, claw 1.5-2.4% at toe-off (Zhang 2017)",
    "Plausible: the pad centre and the home COM projection fall inside the proximal-phalanx zone. The rear pad carries real load in tracks.",
    f"{MF}; Boeye 2026 (dl/boeye2026/boeye.txt l.101); McCrea 2014 Table 1; Zhang 2017; Hutchinson 2005 App. 2; out/brochu_pes_summary.json")
div23 = [p["DIV"][0] for p in prints if p["DIV"][0]]
div34 = [p["DIV"][1] for p in prints if p["DIV"][1]]
divt = [p["DIV"][2] for p in prints if p["DIV"][2]]
row("Digit spread (included angles between digit axes, ground plane)",
    f"II-III {f(R['spread_d2_d3_deg'],1)} deg, III-IV {f(R['spread_d3_d4_deg'],1)} deg, total {f(R['spread_total_d2_d4_deg'],1)} deg",
    f"Bellatoripes prints: II-III {rng(min(div23), max(div23),0)} deg, III-IV {rng(min(div34), max(div34),0)} deg, total {rng(min(divt), max(divt),0)} deg ('generally no greater than 80', III-IV > II-III)",
    f"Model is {f(min(divt)/R['spread_total_d2_d4_deg'],1)}-{f(max(divt)/R['spread_total_d2_d4_deg'],1)}x too narrow and its order is reversed (II-III > III-IV). This affects roll support, not fore-aft.",
    f"{MF}; McCrea 2014 Table 1")
rr = mf["ranges"]
a = h05["table1_ranges_deg"]
row("Ankle (intertarsal) range, included angle (180 = straight)",
    f"{f(rr['r_ankle']['ankle_included_at_lo'],1)}-{f(rr['r_ankle']['ankle_included_at_hi'],1)} deg (width {f(rr['r_ankle']['range_width_deg'],0)}), home {f(rr['r_ankle']['ankle_included_at_home'],1)}",
    f"T. rex MOR 555 model (Hutchinson 2005 Table 1): {180+a['ankle_flexion_extension'][0]}-{180+a['ankle_flexion_extension'][1]} deg (width {a['ankle_flexion_extension'][1]-a['ankle_flexion_extension'][0]}); flexed T. rex hip-height pose ankle 140 deg (McCrea 2014); ostrich stance >= 160-164, full extension 168, snap-through 115 (Schaller 2009; convention assumed included); guineafowl knee/ankle FE amount 162-176 deg (Kambic 2017)",
    f"Flexion end matches Hutchinson 2005 (91 vs 90). The model allows {f(rr['r_ankle']['ankle_included_at_hi']-180,1)} deg of hyperextension past straight, which Hutchinson 2005 does not. Home 135.7 is near the 140 used for a flexed T. rex.",
    f"{MF}; Hutchinson 2005 Table 1 (signs restored); McCrea 2014; Schaller 2009; Kambic 2017")
m_ = a["mtp_flexion_extension"]
row("Toe (MTP III) range, included metatarsus-digit angle",
    f"{f(rr['r_toe_d3_joint']['mtp_d3_included_at_lo'],1)}-{f(rr['r_toe_d3_joint']['mtp_d3_included_at_hi'],1)} deg (width 75), home {f(rr['r_toe_d3_joint']['mtp_d3_included_at_home'],1)}; digit III can tilt {f(rr['r_toe_d3_joint']['digit3_below_horizontal_at_hi'],1)} deg below horizontal; passive 40 N m/rad, damping 15",
    f"T. rex MOR 555 model (Hutchinson 2005 Table 1, 0 = digits at 90 deg to metatarsus): {90+m_[0]}-{90+m_[1]} deg included (width {m_[1]-m_[0]}); guineafowl toe reached 30 deg below horizontal in soft substrate (Falkingham & Gatesy 2014)",
    "The model range lies inside the published model range, narrower at both ends. The report's 'no published T. rex MTP/ankle range' needs qualifying: Hutchinson 2005 publishes model-allowed ranges, though not measured anatomical limits.",
    f"{MF}; Hutchinson 2005 Table 1 + App. 2; Falkingham & Gatesy 2014 abstract")
k_ = a["knee_extension_flexion"]; hp = a["hip_flexion_extension"]
row("Knee and hip ranges",
    f"knee included {f(rr['r_knee']['knee_included_at_hi'],1)}-{f(rr['r_knee']['knee_included_at_lo'],1)}; femur from vertical {f(rr['r_hip_pitch']['femur_from_vertical_at_hi'],1)} (back) to +{f(rr['r_hip_pitch']['femur_from_vertical_at_lo'],1)} (forward), home +{f(rr['r_hip_pitch']['femur_from_vertical_at_home'],1)}",
    f"Hutchinson 2005: knee included {180-k_[1]}-{180-k_[0]}; hip {hp[0]} (flexion) to +{hp[1]} (extension) from the columnar pose (femur at 90 deg to pelvis); hip-extensor moment arms ~94% of peak at 15 deg flexion (Fig. 6)",
    f"Knee matches within 5 deg. Model hip flexes {f(rr['r_hip_pitch']['femur_from_vertical_at_lo']+hp[0],1)} deg further [assumes the pelvis is level in both]. Home femur at 22.5 deg sits near the 15 deg moment-arm optimum.",
    f"{MF}; Hutchinson 2005 Table 1, Fig. 6")
bh = P["boeye2026"]["hip_height_MOR555_m"]
row("Standing posture",
    f"femur {f(R['angles_home']['femur_from_vertical'],1)} deg from vertical, knee {f(R['angles_home']['knee_included'],1)}, ankle {f(R['angles_home']['ankle_included'],1)}, metatarsus {f(R['angles_home']['metatarsus_from_horizontal'],1)} deg above horizontal; hip height / femur {f(HIP_H/FEM,2)}",
    f"subvertical femur, knee near the vertical GRF line (Bishop 2018; Gatesy 2009 via Bishop); flexed T. rex pose knee 110 / ankle 140 (McCrea 2014); MOR 555 hip height {bh['crouch75']}-{bh['crouch95']} m = {f(bh['crouch75']/mor_fem,2)}-{f(bh['crouch95']/mor_fem,2)} femur lengths (Boeye 2026)",
    "Model knee is less flexed than McCrea's flexed case. Model hip height per femur sits at the top of the MOR 555 range because of its long juvenile-like shank. Knee-to-CoP offset is not measured here (needs posture telemetry).",
    f"{MF}; Bishop 2018 (dl/bishop2018); McCrea 2014; Boeye 2026 Sec. 2.3")
row("Hallux and claws",
    "no hallux; digit capsules carry contact to the tip, with no separate ungual",
    "MT I attached posteriorly on MT II, digit I 'not reversed', narrow curved ungual (Brochu 2003); hallux impression on one Bellatoripes print only (McCrea 2014); unguals II-III flat-bottomed (Brochu); ostrich claw only at push-off (Schaller 2011), 1.5-2.4% of load (Zhang 2017); ungual = "
    f"{rng(*ung3_frac,2)} of summed digit III = statistic {f(stat((1-max(ung3_frac))*D3),2)} to {f(stat(D3),2)} on the model [figure estimate]",
    "Omitting the hallux is consistent with the evidence. Load at the claw zone (statistic > ~+1.5) would be a push-off pattern, not a standing one.",
    f"Brochu 2003 text (dl/plazi_brochu2003); McCrea 2014; Schaller 2011; Zhang 2017; out/brochu_pes_summary.json")
st = mf["stiffness"]
row("Passive stiffness / mgh",
    f"six digit springs {f(st['six_digit_springs_over_mgh'],3)}; two ankle servos kp {f(st['two_ankle_kp_over_mgh'],2)} (mgh {f(st['mgh_floor_home_Nm_per_rad'],1)} N m/rad at home)",
    "human intrinsic ankle stiffness 91 +- 23% (tiny perturbations; Loram & Lakie 2002), 64 +- 8% (1 deg; Casadio 2005)",
    "Rough index only. Digit springs are about half the human passive fraction; the servos are active stiffness, not passive.",
    f"{MF}; Loram & Lakie 2002 + Casadio 2005 abstracts (dl/abstracts)")
[r_ for r_ in rows if r_["quantity"].startswith("Load centre")][0]["_mccrea_duplicate_depth_rows"] = br_note_dup
(OUT / "audit_rows.json").write_text(json.dumps(rows, indent=1))

# ---------------------------------------------------------------- load-centre frame
L = []


def add(name, x_lo, x_hi, src, kind, note=""):
    lo, hi = frame(x_lo), frame(x_hi)
    L.append({"position": name, "kind": kind, "source": src, "note": note,
              "x_m": [x_lo, x_hi], "pad_stat_signed": [lo["pad_stat_signed"], hi["pad_stat_signed"]],
              "frac_pad_from_rear": [lo["frac_pad_from_rear"], hi["frac_pad_from_rear"]],
              "frac_footprint_from_rear": [lo["frac_footprint_from_rear"], hi["frac_footprint_from_rear"]],
              "frac_digit3": [lo["frac_digit3_from_mtp"], hi["frac_digit3_from_mtp"]]})


ct = P["birds"]["chang_ting2017"]
add("Flamingo quiet one-leg stance: CoP 'directly under the TMP joint'", 0.0, 0.0, "Chang & Ting 2017", "bird, standing, measured")
add("  ...with its 95% sway radius 3.2 mm = 5-10% of foot length (taken as model digit III length)", -0.10 * D3, 0.10 * D3, "Chang & Ting 2017", "bird, standing, measured; mapping is inference",
    f"inner band +-{f(0.05*D3,4)} m; foot length is not defined in the paper (implied 32-64 mm)")
add("Emu rising (after heel-off): rear edge of support = caudal edge of TMP III", 0.0, 0.0, "Lin et al. 2024", "bird, sit-to-stand, measured (qualitative)", "upper bound for the rear edge of the support base")
add("Track rear edge: digit IV MTP pad (II-III MTP pads clear)", 0.0, 0.0, "Lallensack 2022; McCrea 2014", "theropod track, qualitative", "at or behind the MTP joint")
add("Tyrannosaur tracks: deepest 'around the proximal pedal phalanges' (III-1 zone, Sue proportions)", 0.0, min(III1_frac) * D3, "Boeye 2026; Brochu 2003 figs (digitised)", "theropod track depth, qualitative; zone mapping is inference")
add("  ...III-1 zone with MOR 555 proportions (0.224 of 0.584 m)", 0.0, h05_III1_frac * D3, "Boeye 2026; Hutchinson 2005 App. 2", "theropod track depth; mapping is inference")
add("  ...midpoint of the III-1 zone (Sue / MOR 555 proportions)", 0.5 * min(III1_frac) * D3, 0.5 * h05_III1_frac * D3, "Boeye 2026; Brochu figs; Hutchinson 2005", "theropod track depth; mapping is inference",
    "the unsigned project statistic would read the absolute value")
add("Chicken: CoM 'vertically aligned with the middle of the foot' (assumed; taken as footprint midpoint)", x_from_fp(0.5), x_from_fp(0.5), "Grossi et al. 2014", "bird, standing, ASSUMPTION in source")
add("Ostrich running on sand: CoP originates at the middle of digit III", 0.5 * D3, 0.5 * D3, "Zhang et al. 2017", "bird, locomotion (not standing), measured")
add("Ostrich/claw zone (ungual III-4, Sue proportions): load only at push-off", (1 - max(ung3_frac)) * D3, D3, "Schaller 2011; Zhang 2017; Brochu figs", "bird push-off; mapping is inference")
fj = P["humans"]["fujiwara2003"]["qs_pct_fl"] / 100
add("Human quiet stance mean 45.7% foot length from heel (mapped onto footprint)", x_from_fp(fj), x_from_fp(fj), "Fujiwara et al. 2003", "human, standing, measured; mapping is analogy")
add("Human individual quiet-stance means 31-58% FL", x_from_fp(0.31), x_from_fp(0.58), "Fujiwara et al. 2010", "human, standing, measured; mapping is analogy")
add("Human holdable range 20-80% FL", x_from_fp(0.20), x_from_fp(0.80), "Fujiwara et al. 2003/2010", "human, standing, measured; mapping is analogy")
hy = P["unpublished_hypotheses_from_notes"]
add("Anatomy-note hypothesis 0.30-0.55 of track length from rear", x_from_fp(0.30), x_from_fp(0.55), "research note (unpublished)", "hypothesis")
add("Balance-note hypothesis 40-55% of contact length (contact = footprint)", x_from_fp(0.40), x_from_fp(0.55), "research note (unpublished)", "hypothesis")
add("Balance-note hypothesis 40-55% of contact length (contact = pad only)", PAD_R + 0.40 * (PAD_F - PAD_R), PAD_R + 0.55 * (PAD_F - PAD_R), "research note (unpublished)", "hypothesis")
add("Model pad centre", PAD_C, PAD_C, MF, "model")
add("Model home COM projection", x_from_stat(mf["com"]["com_projection_signed_pad_stat"]), x_from_stat(mf["com"]["com_projection_signed_pad_stat"]), MF, "model")
pj = P["project"]
add("Project r9 diagnostic, furthest lean held by >=5/8 spawns", x_from_stat(pj["r9_diag_stat"][0]), x_from_stat(pj["r9_diag_stat"][1]), "report", "project")
add("Project stance gate bar (0.80)", x_from_stat(pj["gate_bar_stat"]), x_from_stat(pj["gate_bar_stat"]), "report", "project")
add("Drifting seeds' stance (~1.0)", x_from_stat(pj["drifted_stance_stat"]), x_from_stat(pj["drifted_stance_stat"]), "report", "project")
(OUT / "load_centre_frame.json").write_text(json.dumps({"frame": {"pad_centre_m": PAD_C, "pad_half_m": HALF, "pad_rear_m": PAD_R, "pad_front_m": PAD_F,
                                                                  "footprint_m": FP, "digit3_m": D3, "x_origin": "digit III MTP joint, + forward"}, "positions": L}, indent=1))

# ---------------------------------------------------------------- markdown
md = ["# Literature cross-check: T. rex foot audit (generated by lit_convert.py)", "",
      f"Model frame (main 2e734e29, physics r9, home keyframe; {MF}): pad {f(PAD_R,3)} to {f(PAD_F,3)} m about MTP III, centre {f(PAD_C,3)} m, half-length {f(HALF,3)} m; "
      f"footprint (pad rear to digit III tip) {f(FP,3)} m; digit III {f(D3,3)} m (MTP to capsule tip).", "",
      "## Audit rows (published value restated in the model's units)", "",
      "| Quantity | Model | Published, in model units | Reading | Evidence |", "|---|---|---|---|---|"]
for r_ in rows:
    md.append("| " + " | ".join(r_[k].replace("|", "/") for k in ("quantity", "model", "published", "reading", "evidence")) + " |")
md += ["", "## Standing load centre in the project's CoP frame", "",
       "| Position | Kind | Signed pad statistic | Fraction of pad from rear | Fraction of footprint from rear | m ahead of MTP III | Source |", "|---|---|---|---|---|---|---|"]
for p in L:
    md.append(f"| {p['position']} | {p['kind']} | {rng(*p['pad_stat_signed'],2)} | {rng(*p['frac_pad_from_rear'],2)} | {rng(*p['frac_footprint_from_rear'],3)} | {rng(*p['x_m'],3)} | {p['source']} |")
md += ["", "## Brochu 2003 pes (FMNH PR 2081), digitised from the published figures", "",
       f"Digit III phalanges, dorsal / ventral view (cm): {[round(v,1) for v in br['dorsal_cm']['III']]} / {[round(v,1) for v in br['ventral_cm']['III']]}; sum {rng(*sumIII,1)}.",
       f"Digit II: {[round(v,1) for v in br['dorsal_cm']['II']]} / {[round(v,1) for v in br['ventral_cm']['II']]}; digit IV (IV-1..IV-3 only): {[round(v,1) for v in br['dorsal_cm']['IV_preserved']]} / {[round(v,1) for v in br['ventral_cm']['IV_preserved']]}; digit I ungual {[round(v,1) for v in br['dorsal_cm']['I_ungual']]} / {[round(v,1) for v in br['ventral_cm']['I_ungual']]}.",
       f"MT III views {[round(v,1) for v in mt3_views]} (mean {f(mt3,1)}); MT II {[round(v,1) for v in mt2_views]}; MT IV {[round(v,1) for v in mt4_views]}.",
       f"III-1 share {rng(*III1_frac,3)}; III-1+III-2 {rng(*III12_frac,3)}; ungual {rng(*ung3_frac,3)}; II-1 share of digit II {rng(*II1_frac,3)}; digit II/III sums {rng(*br['sumII_over_sumIII'],2)}.", ""]
(OUT / "literature_crosscheck.md").write_text("\n".join(md))
print("\n".join(md))
