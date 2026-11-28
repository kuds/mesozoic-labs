"""E4 forced air (design/PREREG_CANDIDATES_2026_11_28.md, section 2 "E4", section 3 item 12, section 4 "E4").

DESK search only. The search was done on 28 Nov 2026 with the session's web-search tool (generic engineering terms;
nobody asked, nothing requested). No page could be opened: a fetch of arxiv.org failed with "getaddrinfo ENOTFOUND"
(the host is outside this session's network), so, as the pre-registration's section 5 already notes, every figure
below is the search tool's summary of a page, read on 28 Nov 2026, not the page itself.

This script holds the search log and each source's figures as read, applies item 12's test to every source
(``qualifies``), and writes:
  sim/results/candidates_2026-11-28/E4/e4_sources.json   (with candidate_checks.provenance)
  sim/results/candidates_2026-11-28/E4/e4_sources.csv
It calls ``candidate_checks.forced_air_c1`` only if a source qualifies; if one does, it stops instead and says so,
since the C1 run then needs the four fore128 demo courses on both seed blocks (section 4 "E4").
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

BUILD = Path("/home/user/mesozoic-labs/hardware/triceratops_build")
sys.path.insert(0, str(BUILD / "sim"))
import candidate_checks as cc  # noqa: E402

OUT = BUILD / "sim" / "results" / "candidates_2026-11-28" / "E4"
SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E4")
READ = "2026-11-28"
ACCESS = "web-search summary (page not opened)"

# ------------------------------------------------------------------------------------------------ the queries run
# "P": the pre-registered queries (section 4 "E4"), in the order run. "the same with 'planetary gear actuator'" was
# read as each query with that phrase added or substituted. P01b was added after P07-P12 had returned (changed after
# a result). "F": follow-up queries chosen after the pre-registered results, to read more of the hits' summaries
# (changed after a result: an addition to the query list; item 12's test is unchanged). The F queries name paper
# titles, their public authors and makers' product names; none names this project, its owner or anyone tied to it.
QUERIES = [
    ("P01", "BLDC actuator thermal resistance forced air", "standard"),
    ("P02", "robot joint actuator fan cooling thermal resistance", "standard"),
    ("P03", "frameless BLDC motor thermal resistance airflow", "standard"),
    ("P04", "quasi-direct-drive actuator cooling winding temperature fan", "standard"),
    ("P05", "outrunner motor thermal resistance air velocity", "standard"),
    ("P06", "servo motor continuous torque with fan cooling", "standard"),
    ("P07", "planetary gear actuator thermal resistance forced air", "standard"),
    ("P08", "robot joint planetary gear actuator fan cooling thermal resistance", "standard"),
    ("P09", "frameless BLDC motor planetary gear actuator thermal resistance airflow", "standard"),
    ("P10", "quasi-direct-drive planetary gear actuator cooling winding temperature fan", "standard"),
    ("P11", "outrunner planetary gear actuator thermal resistance air velocity", "standard"),
    ("P12", "planetary gear actuator servo motor continuous torque with fan cooling", "standard"),
    ("P01b", "BLDC planetary gear actuator thermal resistance forced air", "standard"),
    ("F01", '"3D Printed, Open-Source Actuators for Legged Locomotion" thermal resistance fans off fans on °C/W',
     "extended"),
    ("F02", "Urs Adu Rouse Moore 3D printed actuator legged locomotion Table 3 thermal parameters fans on fans off "
            "thermal capacitance J/°C", "extended"),
    ("F03", "2202.12395 open-source 3D printed actuator legged locomotion motor model mass kg centrifugal fan "
            "airflow CFM continuous torque", "extended"),
    ("F04", '"Open-Source Actuators for Legged Locomotion" two auxiliary centrifugal fans ducted motor housing fan '
            'model', "extended"),
    ("F05", "Urs Enninful Adu Rouse Moore quasi-direct-drive 3D printed actuator brushless motor used diameter mass "
            "gear ratio peak torque", "extended"),
    ("F06", "osti 1369514 motor thermal forced air 79% increase heat transfer pumped water 107% robotic actuator "
            "lumped heat transfer coefficient", "extended"),
    ("F07", '"Improving Robotic Actuator Torque Density and Efficiency Through Enhanced Heat Transfer" motor '
            'frameless Allied Motion fins fan airflow', "extended"),
    ("F08", "Sandia DSCC2016-9738 robotic actuator enhanced heat transfer motor housing fins fan power W/K heat "
            "transfer coefficient motor size", "extended"),
    ("F09", "Mazumdar Spencer Buerger 2016 enhanced heat transfer motor housing STEPPR frameless motor diameter fans "
            "CFM", "extended"),
    ("F10", '"enhanced heat transfer" robotic actuator Sandia 2016 brushless motor tested "W/K" forced air fans '
            'nominal design motor model', "standard"),
    ("F11", 'Mazumdar 2016 "Megaflux" MF0127 OR MF0095 OR MF0076 OR MF0150 frameless motor housing heat transfer '
            'fans airflow', "extended"),
    ("F12", '"Heat Transfer Models and Measurements of Brushless DC Motors for Small UASs" thermal resistance '
            'freestream velocity m/s motor diameter winding temperature', "extended"),
    ("F13", "maxon high efficiency joint HEJ 70 HEJ 90 fan cooling continuous torque thermal resistance winding "
            "ambient", "extended"),
    ("F14", "legged robot actuator winding to ambient thermal resistance K/W with fan without fan measured",
     "extended"),
    ("F15", '"From Temperature-Limited to Battery-Limited Legged Locomotion" actuator cooling fan thermal resistance',
     "extended"),
    ("F16", "maxon HEJ 70-48-60 datasheet thermal dissipation active cooling forced air W thermal resistance housing "
            "ambient", "extended"),
    ("F17", "ACSELL Actuation and Control System for Efficient Legged Locomotion Sandia motor thermal resistance no "
            "fan K/W fan 1.4 W 2.6 W 4.2 W continuous torque", "extended"),
    ("F18", "STEPPR Sandia legged robot actuator Allied Motion MF0127 frameless motor fan cooled housing", "standard"),
]
FETCHES = [{"url": "https://arxiv.org/abs/2202.12395", "date": READ,
            "result": "failed: getaddrinfo ENOTFOUND arxiv.org (outside this session's allowed network)",
            "note": "the tool's message says to stop trying more hosts once one fails this way; none was tried"}]

# ------------------------------------------------------------------------------------------------ the sources read
# Per source: what the summaries gave, as read. Item 12's test fields:
#   rth_pair_given: True if an R_th (K/W, C/W; a conductance in W/K read as its inverse) is given both without and
#                   with forced air for one drive, False if the source gives no such pair, None if the summaries do
#                   not establish it;
#   flow_stated:    the flow as stated (m/s, m^3/h, CFM, l/min) or None; a fan's size, price or electrical power is
#                   not a flow;
#   heat_path:      "winding_to_ambient", "total", or the path the source gives ("housing_to_ambient", ...) or None;
#   diameter_mm / mass_kg of the drive measured, None if not found.
SOURCES = [
    {"id": "S01", "found_by": ["P02", "P04", "P10", "F01-F05", "F14"],
     "title": "Design and Characterization of 3D Printed, Open-Source Actuators for Legged Locomotion",
     "who_when": "K. Urs, C. Enninful Adu, E. J. Rouse, T. Y. Moore; arXiv 2202.12395 (Feb 2022), IROS 2022 "
                 "(DOI 10.1109/IROS47612.2022.9981940)",
     "urls": ["https://arxiv.org/pdf/2202.12395", "https://arxiv.org/html/2202.12395",
              "https://www.researchgate.net/publication/358899190_Design_and_Characterization_of_3D_Printed_Open-"
              "Source_Actuators_for_Legged_Locomotion", "https://ieeexplore.ieee.org/abstract/document/9981940/"],
     "drive": "two quasi-direct-drive actuators for 8-15 kg robots, 3D printed, under 200 USD each; motor, "
              "diameter and mass not in any summary",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["total thermal resistance R_th 2.27 C/W from a transfer-function fit (its Table 3)",
                 "a second value, 1.23 C/W, in the same table under 'Fan' / 'No Fan' columns: which column is which "
                 "is not in any summary",
                 "'the cooling fans provide an approximately 3.5x cooling advantage' (random-input thermal testing); "
                 "'nearly doubles' the continuous current; thermal capacitances 'very similar' fans on or off",
                 "summaries disagree: one calls 2.27 C/W the fans-on value; another gives 'from 2.27 to roughly "
                 "0.65 K/W', which looks like the search tool's own 2.27 / 3.5 (0.65 is the ACSELL deck's figure, "
                 "S03); 2.27 / 1.23 = 1.85, not 3.5"],
     "fan": "two auxiliary centrifugal fans, 30 x 10 mm, 5 USD each, ducted through housing cavities over the "
            "windings and rotor; no flow stated",
     "rth_pair_given": None, "flow_stated": None, "heat_path": "total",
     "notes": "the closest match found: a legged-robot actuator with and without fans. Not usable as read: no "
              "stated flow, the pair's mapping unconfirmed and inconsistent with the stated 3.5x, the drive's size "
              "and mass not found"},
    {"id": "S02", "found_by": ["P01", "P03", "P07", "P09", "P01b", "F06-F11"],
     "title": "Improving Robotic Actuator Torque Density and Efficiency Through Enhanced Heat Transfer",
     "who_when": "A. Mazumdar, S. Spencer, C. Hobart, S. Kuehl, G. Brunson, N. Coleman, S. Buerger (Sandia); ASME "
                 "DSCC 2016, DSCC2016-9738; OSTI 1369514",
     "urls": ["https://www.osti.gov/servlets/purl/1369514", "https://www.osti.gov/biblio/1369514",
              "https://www.sandia.gov/research/publications/details/improving-robotic-actuator-torque-density-and-"
              "efficiency-through-enhanced-h-2016-07-01/",
              "https://asmedigitalcollection.asme.org/DSCC/proceedings-abstract/DSCC2016/50701/V002T26A004/231106"],
     "drive": "Allied Motion Megaflux series frameless three-phase BLDC motor in the authors' housings; which size "
              "is not in any summary (the series' datasheets found run MF0076 to MF0310; the MF0127 is listed at "
              "127 mm outside diameter, S08)",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["H_m, a lumped motor heat-transfer coefficient in W/K 'which combines all the thermal resistances "
                 "between the copper coils and the ambient air'",
                 "housing design +50 % over a nominal design; forced air +79 % and pumped water +107 % over the "
                 "nominal design",
                 "water: H_m 2.09 W/K, '107 % larger than' nominal and '15 % better than the best performance with "
                 "forced air'",
                 "one summary: the enhanced design raises H_m 'from 1.25 to 1.51 W/K when no fans are used'; another: "
                 "'a single heat transfer coefficient of 1.77 W/K fits the data' (configuration cut off)",
                 "'further increasing the fan power only marginally increases the heat transfer'",
                 "DESK arithmetic on the summaries, not used: 2.09 / 2.07 = 1.01 W/K nominal, which disagrees with "
                 "the 1.25 W/K of the other summary"],
     "fan": "fan power mentioned; no flow in any summary",
     "rth_pair_given": None, "flow_stated": None, "heat_path": "winding_to_ambient",
     "notes": "the path is right (coil to ambient). Not usable as read: no stated flow, the motor's size not found "
              "(and a frameless motor in a lab housing, not a geared actuator), the nominal figure inconsistent "
              "between summaries"},
    {"id": "S03", "found_by": ["F15", "F17"],
     "title": "ACSELL: Actuation and Control System for Efficient Legged Locomotion (slide deck)",
     "who_when": "Sandia team; OSTI 1242124; date not in the summaries",
     "urls": ["https://www.osti.gov/servlets/purl/1242124"],
     "drive": "the STEPPR robot's actuator (per the summary); motor and size not in any summary",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["fan of 1.4 W: 0.65 K/W, 8.12 N m continuous; 2.6 W: 0.63 K/W, 8.25 N m; 4.2 W: 0.62 K/W, "
                 "8.32 N m",
                 "'Air cooling: 32% increase in cont. T with 4.2 W'; 'Liquid cooling: 80% increase in cont. T with "
                 "2.5 W'",
                 "the no-fan row is not in any summary"],
     "fan": "given by electrical power (1.4 / 2.6 / 4.2 W), not by flow",
     "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "forced-air R_th only (no passive R_th read); the 32 % is a continuous-rating ratio, reported and not "
              "converted (item 12); path and drive size not stated"},
    {"id": "S04", "found_by": ["P10", "F13", "F16"],
     "title": "maxon High Efficiency Joint data sheets: HEJ 70-48-60 (rev 1.0, preliminary), HEJ 70-48-50, "
              "HEJ 90-48-140 (March 2024, rev 1.3)",
     "who_when": "maxon (maker data sheets)",
     "urls": ["https://global.maxongroup.com/hubfs/Team_HQ_Global/10_Robotics/Datasheet_HEJ_70_48_60_rev1.0.pdf",
              "https://www.maxongroup.com/assets/public/caas/v1/media/271230/data/918cdb4e2af1177d5e7cd608fa170828/"
              "robotics-robotic-joints-hej-70-data-sheet.pdf",
              "https://global.maxongroup.com/high-efficiency-joints"],
     "drive": "planetary quasi-direct-drive robot joints; HEJ 70 and HEJ 90 (diameter and mass not in any summary)",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["HEJ 70-48-60: nominal joint torque +/-27 N m 'maintained indefinitely without external forced air "
                 "cooling'; passive convective dissipation 'up to ca. 50 W'; R_th winding-housing 0.38 K/W; winding "
                 "time constant 12 s; integrated heat sinks for forced-air cooling; data at 48 V, 25 C",
                 "HEJ 70-48-50 (another variant): nominal +/-19 N m; an active-cooling dissipation of 140 W "
                 "(flow not in the summary)",
                 "HEJ 90: 75 N m continuous; 'active thermal cooling if necessary'; its figures cut off"],
     "fan": "external forced air named; no flow in any summary",
     "rth_pair_given": False, "flow_stated": None, "heat_path": "winding_to_housing",
     "notes": "a dissipation pair across two variants (50 W passive on one, 140 W active on another) and a "
              "winding-to-housing R_th: neither is a winding-to-ambient R_th pair (item 12)"},
    {"id": "S05", "found_by": ["P06", "P12"],
     "title": "Moog AML servo motors (fan-cooled) against the naturally cooled AM series",
     "who_when": "Moog product page",
     "urls": ["https://moog.com/products/motors-servomotors/servo-motors/other-servo-motors/AML-Servo-Motors.html"],
     "drive": "industrial servo motors, continuous stall 1.72-194 N m (the line); per-model size not read",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["continuous torque and power 'up to 1.5 times greater' than the naturally cooled AM series, at "
                 "'almost the same' size"],
     "fan": "fan cooled; no flow stated",
     "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "a continuous-rating ratio, reported and not converted (item 12)"},
    {"id": "S06", "found_by": ["P06"],
     "title": "Rockwell Kinetix VPC continuous-duty motors (optional fan)",
     "who_when": "Rockwell Automation product page",
     "urls": ["https://www.rockwellautomation.com/en-ca/products/hardware/motion-control/kinetix-vpc-continuous-"
              "duty.html"],
     "drive": "industrial servo motors, 17.6-191.1 N m, 4.0-30.0 kW",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["an optional field-replaceable fan; no with / without figures in the summary"],
     "fan": "optional fan; no flow", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "no pair of any kind"},
    {"id": "S07", "found_by": ["P05", "P11", "F12"],
     "title": "Heat Transfer Models and Measurements of Brushless DC Motors for Small UASs",
     "who_when": "Aerospace 11(5):401, May 2024 (MDPI)",
     "urls": ["https://doi.org/10.3390/aerospace11050401", "https://mdpi.com/2226-4310/11/5/401"],
     "drive": "outrunner BLDC motors of small drones (sizes not in the summaries)",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["external heat-transfer coefficient against freestream velocity (wind tunnel, pitot tube); axial "
                 "Reynolds number 20,000-40,000; a Nusselt correlation; Etemad's rotating-cylinder correlation "
                 "'significantly overestimates' motor temperature"],
     "fan": "wind-tunnel freestream, not a fan", "rth_pair_given": False, "flow_stated": None,
     "heat_path": "housing_to_ambient",
     "notes": "a surface-to-air coefficient: the path item 12 refuses (the winding-to-housing part does not change "
              "with the airflow)"},
    {"id": "S08", "found_by": ["F11"],
     "title": "Allied Motion Megaflux frameless torque motors, MF0127 data sheet (R7e) and the series",
     "who_when": "Allied Motion (now Allient) data sheets",
     "urls": ["https://www.alliedmotion.com/wp-content/uploads/documents/Megaflux_Frameless_Torque_Motors_MF0127_"
              "R7e.pdf", "https://actuatorlist.com/actuators/allied-motion-megaflux-mf0127"],
     "drive": "MF0127: 127 mm outside diameter, 1.6-8.4 N m continuous stall (distributor page)",
     "diameter_mm": 127.0, "mass_kg": None,
     "figures": ["thermal resistance for housed motors on a 254 mm square, 6.35 mm aluminium plate in still air; "
                 "130 C maximum"],
     "fan": "none (still air)", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "passive only; the MF0127 is outside the 40-100 mm range"},
    {"id": "S09", "found_by": ["P03"],
     "title": "maxon EC frameless motor data (EC 45 / 60 / 90 frameless)",
     "who_when": "maxon product pages",
     "urls": ["https://www.maxongroup.com/maxon/view/product/motor/ecmotor/ecframeless/EC90Frameless/543673"],
     "drive": "frameless BLDC motors 45-90 mm class (per model names)",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["winding-housing 1.4-4.1 K/W and housing-ambient 1.4-4.0 K/W across the models; e.g. one EC 90 "
                 "frameless 1.82 / 1.78 K/W; the airflow behind them not stated"],
     "fan": "none stated", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "one condition only, no forced-air figure"},
    {"id": "S10", "found_by": ["P08", "P09", "P10", "P11", "P01b"],
     "title": "BEAR proprioceptive actuator with liquid cooling (single-stage planetary)",
     "who_when": "T. Zhu, J. Hooks, D. Hong (UCLA); PDF on ucla.box.com",
     "urls": ["https://ucla.box.com/shared/static/ywg13p17yu3e02ivk3ndcgcezslp7wi4.pdf"],
     "drive": "a commercial BLDC motor with a single-stage planetary, 10:1 (size not in the summaries)",
     "diameter_mm": None, "mass_kg": None,
     "figures": ["continuous torque 7.8 N m air cooled, about 21 N m liquid cooled"],
     "fan": "liquid, not forced air", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "a continuous-rating pair for liquid cooling: not forced air, not converted"},
    {"id": "S11", "found_by": ["F14"],
     "title": "a legged-robot actuator study with liquid and immersion cooling (the giving page not identified "
              "among the results)",
     "who_when": "not identified", "urls": [],
     "drive": "not identified", "diameter_mm": None, "mass_kg": None,
     "figures": ["winding-to-ambient 0.473 K/W with indirect housing liquid cooling, 0.127 K/W with immersion"],
     "fan": "liquid, not forced air", "rth_pair_given": False, "flow_stated": None,
     "heat_path": "winding_to_ambient",
     "notes": "liquid cooling, no forced-air or passive row read; source not identified"},
    {"id": "S12", "found_by": ["P02", "P08"],
     "title": "PatSnap blog on humanoid joint actuator weight",
     "who_when": "vendor blog",
     "urls": ["https://www.patsnap.com/resources/blog/articles/cut-humanoid-robot-joint-actuator-weight-by-40-50/"],
     "drive": "none named", "diameter_mm": None, "mass_kg": None,
     "figures": ["immersion cooling about 3.7x lower thermal resistance than a liquid jacket and about 10x lower "
                 "than forced air (no values or conditions)"],
     "fan": "not stated", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "marketing ratios without values or conditions"},
    {"id": "S13", "found_by": ["F14", "F15"],
     "title": "From Temperature-Limited to Battery-Limited Legged Locomotion",
     "who_when": "Segas et al., arXiv 2611.22231 (2026)",
     "urls": ["https://arxiv.org/pdf/2611.22231", "https://arxiv.org/html/2611.22231"],
     "drive": "Unitree Go2 actuators", "diameter_mm": None, "mass_kg": None,
     "figures": ["heat-aware RL policies that do not observe temperature; 174 min instead of 19 at 1 m/s with 5 kg "
                 "payload, 14 C headroom (Go2); per-actuator thermal parameters with a resistance entry but no fan "
                 "case in the summaries; 'hardware cooling pushes the thermal envelope at the price of mass and "
                 "power'"],
     "fan": "none", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "no forced-air pair; context only (a heat-aware policy, not forced air)"},
    {"id": "S14", "found_by": ["P01", "P02", "P04", "P05", "P07", "P11", "P01b"],
     "title": "other hits without a forced-air R_th pair",
     "who_when": "various",
     "urls": ["https://www.automation.com/article/enhancing-motor-performance-rotor-designs",
              "https://arxiv.org/pdf/2002.00798", "https://arxiv.org/pdf/2011.06217",
              "https://arxiv.org/pdf/2202.12365",
              "https://industrialtechnology.co.uk/thermal-management-considerations-to-optimise-bldc-flat-motor-"
              "performance", "https://www.techscience.com/fhmt/online/detail/28895",
              "https://www.jpl.nasa.gov/site/research/media/posters/2021/R21119p.pdf",
              "https://endless-sphere.com/sphere/threads/outrunner-self-cooling-testing.20298/latest"],
     "drive": "various", "diameter_mm": None, "mass_kg": None,
     "figures": ["rotor designs: R_th against rotor speed (self-ventilation), no fan", "a miniature DC-motor resonant "
                 "actuator with a fin: heat-sink resistances in K/W", "a series elastic actuator: a thermal network, "
                 "130 C winding limit", "QDD motor selection: uneven winding heating", "a vendor note on flat BLDC "
                 "cooling (qualitative)", "a 2026 review of robot-actuator cooling (qualitative)", "JPL: heat pipes "
                 "in a planetary actuator (not forced air)", "a forum thread: forcing air through an outrunner "
                 "'didn't seem to make a difference' (anecdote)"],
     "fan": "-", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
     "notes": "none gives R_th with and without forced air"},
]

DIAM_MM = (40.0, 100.0)   # section 4 "E4": a comparable drive is 40-100 mm across and 0.2-1.5 kg
MASS_KG = (0.2, 1.5)


def comparable(s: dict):
    d, m = s["diameter_mm"], s["mass_kg"]
    if d is not None and not (DIAM_MM[0] <= d <= DIAM_MM[1]):
        return False
    if m is not None and not (MASS_KG[0] <= m <= MASS_KG[1]):
        return False
    if d is None or m is None:
        return None      # not established
    return True


def qualifies(s: dict) -> dict:
    """Item 12: an R_th given for a comparable drive both without and with forced air at a stated flow, on the
    winding-to-ambient (or total) path. Every check must be True; None (not established) fails."""
    checks = {"rth_pair_given": s["rth_pair_given"],
              "flow_stated": s["flow_stated"] is not None,
              "heat_path_accepted": s["heat_path"] in cc.FORCED_AIR_PATHS,
              "comparable_drive": comparable(s)}
    return {"checks": checks, "usable": all(v is True for v in checks.values())}


def main() -> int:
    t0 = time.time()
    rows = []
    for s in SOURCES:
        q = qualifies(s)
        rows.append({**s, "read_date": READ, "access": ACCESS, **q})
    usable = [r["id"] for r in rows if r["usable"]]
    if usable:
        print("A source qualifies under item 12:", usable, "- compute C1 per section 4 'E4' (not done here).")
        return 2
    result = {
        "experiment": "E4 forced air", "date": READ, "evidence": "DESK (web search; no page opened)",
        "prereg": "design/PREREG_CANDIDATES_2026_11_28.md: section 2 'E4', section 3 item 12, section 4 'E4'",
        "rule": ("Compute C1 at 35 C with a forced-air R_th ONLY if a source gives one for a comparable drive (an R_th "
                 "both without and with forced air at a stated flow, winding to ambient or total; a BLDC motor or "
                 "geared actuator 40-100 mm across and 0.2-1.5 kg); otherwise report that no usable figure was found"),
        "accepted_heat_paths": list(cc.FORCED_AIR_PATHS), "comparable_box": {"diameter_mm": DIAM_MM, "mass_kg": MASS_KG},
        "queries": [{"id": i, "query": q, "mode": m, "date": READ,
                     "pre_registered": i.startswith("P") and i != "P01b",
                     "label": None if (i.startswith("P") and i != "P01b") else "changed after a result (added query)"}
                    for i, q, m in QUERIES],
        "fetches": FETCHES,
        "sources": rows,
        "usable_sources": usable,
        "outcome": "no usable figure was found",
        "c1_computed": False,
        "forced_air_c1_called": False,
        "seeds": None,
        "wall_time_s": None,
        "provenance": cc.provenance(runner=Path(__file__)),
    }
    result["wall_time_s"] = round(time.time() - t0, 3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e4_sources.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    with open(OUT / "e4_sources.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "title", "who_when", "drive", "diameter_mm", "mass_kg", "rth_pair_given", "flow_stated",
                    "heat_path", "comparable_drive", "usable", "read_date", "access", "first_url", "notes"])
        for r in rows:
            c = r["checks"]
            w.writerow([r["id"], r["title"], r["who_when"], r["drive"], r["diameter_mm"], r["mass_kg"],
                        c["rth_pair_given"], r["flow_stated"], r["heat_path"], c["comparable_drive"], r["usable"],
                        r["read_date"], r["access"], (r["urls"] or [""])[0], r["notes"]])
    SCR.mkdir(parents=True, exist_ok=True)
    (SCR / "run_e4.log").write_text(
        f"E4 run {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(QUERIES)} queries, {len(SOURCES)} source records, "
        f"usable {usable}; outcome: {result['outcome']}\n")
    for r in rows:
        print(r["id"], "usable" if r["usable"] else "not usable", r["checks"])
    print("usable sources:", usable, "| outcome:", result["outcome"], "| c1_computed:", result["c1_computed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
