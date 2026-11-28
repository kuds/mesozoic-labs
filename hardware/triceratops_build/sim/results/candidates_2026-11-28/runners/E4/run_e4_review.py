"""E4 forced air: fixes after the independent review of 28 Nov 2026 (findings E4-R1 to E4-R6).

Everything here was decided after E4's result was seen and is labelled "changed after a result". Item 12's test
(design/PREREG_CANDIDATES_2026_11_28.md, section 3 item 12; section 4 "E4") is NOT changed: the same ``qualifies``
function of run_e4.py (sha256 checked below) is applied to the corrected and added records.

Writes only new files:
  sim/results/candidates_2026-11-28/E4/e4_review_addendum.json
  sim/results/candidates_2026-11-28/E4/e4_sources_reviewed.csv
It edits nothing (run_e4.py, e4_sources.json, e4_sources.csv and E4_SUMMARY.md are left as they are).
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

SCR = Path("/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/E4")
BUILD = Path("/home/user/mesozoic-labs/hardware/triceratops_build")
OUT = BUILD / "sim" / "results" / "candidates_2026-11-28" / "E4"
READ = "2026-11-28"
LABEL = "changed after a result (E4 review fixes, 28 Nov 2026)"
RUN_E4_SHA = "59514dc3a3da039230d8359f4c4d5d50ff83ac8b7df7c2c09c910bdcf4b1b8c0"   # run_e4.sha256_before_run


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------- load the original runner (unchanged, checked)
assert sha(SCR / "run_e4.py") == RUN_E4_SHA, "run_e4.py changed since E4 ran"
_spec = importlib.util.spec_from_file_location("run_e4", SCR / "run_e4.py")
run_e4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_e4)          # main() is guarded; nothing is written by the import
cc = run_e4.cc

# ------------------------------------------------------------------- E4-R1: audit of every query's terms
# Hand classification of each query against the hard rule "Web searches use generic part and engineering terms only".
# Categories: personal names (paper authors), makers or products, institutions or named projects, report or paper
# identifiers, quoted paper titles. None of the 31 names this project, its owner or anyone tied to it.
NON_GENERIC = {
    "F01": ["quoted paper title"],
    "F02": ["personal names (paper authors)"],
    "F03": ["paper identifier (arXiv number)"],
    "F04": ["quoted paper title (part)"],
    "F05": ["personal names (paper authors)"],
    "F06": ["report identifier (OSTI number)"],
    "F07": ["quoted paper title", "maker"],
    "F08": ["institution", "paper identifier (conference paper number)"],
    "F09": ["personal names (paper authors)", "institution's named robot project"],
    "F10": ["institution"],
    "F11": ["personal name (paper author)", "maker's product line and model numbers"],
    "F12": ["quoted paper title"],
    "F13": ["maker", "product names"],
    "F15": ["quoted paper title"],
    "F16": ["maker", "product model number"],
    "F17": ["named project and its full title", "institution"],
    "F18": ["named robot project", "institution", "maker", "product model number"],
}

# ------------------------------------------------------------------- the review searches (generic terms only)
RAW = SCR / "review_searches_raw.txt"
REVIEW_QUERIES = [
    {"id": "R01", "query": "modular robot actuator forced air cooling thermal equivalent circuit maximum continuous "
                           "current rated current", "mode": "standard", "date": READ,
     "why": "E4-R2: look for the missed forced-air actuator source with generic terms", "label": LABEL},
    {"id": "R02", "query": "BLDC robot joint actuator winding to ambient thermal resistance K/W natural convection "
                           "versus fan cooled airflow m/s measured", "mode": "standard", "date": READ,
     "why": "E4-R3: the reviewer's own generic query, rerun", "label": LABEL},
    {"id": "R03", "query": "3D printed quasi-direct-drive legged robot actuator total thermal resistance with cooling "
                           "fans continuous current", "mode": "standard", "date": READ,
     "why": "E4-R3: try to reproduce the S01 wording the reviewer quoted, with generic terms", "label": LABEL},
]
REVIEW_FETCHES = "none: per the E4 fetch failure's tool message no other host was tried; a full text needs the owner " \
                 "to allow the host in the session's network settings"

# ------------------------------------------------------------------- the corrected and added records
src = {s["id"]: copy.deepcopy(s) for s in run_e4.SOURCES}

# E4-R3: S01, the fresh wording. The check stays None (not established): the reviewer's quoted sentence was not
# reproduced by R03, and the E4 summaries disagree on which value is fans-on. It changes no outcome: S01 fails on
# flow_stated and comparable_drive regardless.
src["S01"]["review"] = {
    "finding": "E4-R3", "label": LABEL,
    "reviewer_reading": ("the reviewer's search of 28 Nov 2026 (standard mode, the query rerun here as R02) quoted "
                         "arXiv 2202.12395: 'the total thermal resistance R_th was 2.27 °C/W. The 3.5× reduction in "
                         "R_th with the fans nearly doubles the amount of continuous current that can be safely "
                         "dissipated.' No airflow given."),
    "this_session_reading": ("R02 did not return S01; R03 returned it without the R_th sentence: 'The snippets I got "
                             "don't include the thermal resistance value or the continuous current rating'; 'The "
                             "abstract says the thermal solution nearly doubles the thermally driven torque limits of "
                             "the plastic design'; 'Two small centrifugal fans sit on the side of each actuator and "
                             "push air through ducts over the motor windings and rotor.'"),
    "if_reviewer_reading_holds": {"stated": "an R_th ratio of 3.5 (a stated R_th ratio, not a continuous-rating ratio)",
                                  "rho": round(1 / 3.5, 4),
                                  "consistency_desk": ("sqrt(3.5) = 1.871: with one node and a fixed winding rise, "
                                                       "continuous current scales as R_th^-0.5, which agrees with "
                                                       "'nearly doubles' (desk arithmetic, supporting)")},
    "rth_pair_given_kept": None,
    "why_kept_none": ("the quoted sentence was not reproduced here, the raw tool text behind it is not archived, and "
                      "E4's summaries conflict on which value goes with the fans; S01 fails flow_stated and "
                      "comparable_drive either way"),
    "would_qualify_if": ("the full text gives the fan flow (m/s, m^3/h or CFM) and the motor's size and mass within "
                         "40-100 mm and 0.2-1.5 kg; then rho of about 0.29 would be usable and C1 at 35 C for the four "
                         "fore128 policies on both seed blocks would have to be computed under item 12 "
                         "(forced_air_c1, heat_path 'total'). Not computed now: no source qualifies."),
}

# E4-R5: S11's two figures have no identified source and no URL. Withdrawn: unsourced, not used.
src["S11"]["figures_withdrawn"] = ("the record's two liquid / immersion winding-to-ambient values: no identified "
                                   "source and no URL, so unsourced, not used, and not to be quoted as sourced values")
src["S11"]["figures"] = []
src["S11"]["notes"] = "unsourced, not used (E4-R5); liquid cooling in any case, not forced air"
src["S11"]["review"] = {"finding": "E4-R5", "label": LABEL}

# E4-R2: the missed source.
src["S15"] = {
    "id": "S15", "found_by": ["R01", "R02"],
    "title": "Design of a Concentrically Stacked Modular Actuator with Forced Air Cooling for Multi-DOF Robotic Systems",
    "who_when": "Energies 11(11):2947, 2018 (MDPI); authors not in the summaries read",
    "urls": ["https://www.mdpi.com/1996-1073/11/11/2947", "https://www.mdpi.com/1996-1073/11/11/2947/xml",
             "https://www.mdpi.com/1996-1073/11/11/2947/review_report",
             "https://libmast.utm.my/Record/doaj.art-929c7fc97d57409f8e42aac672f57134"],
    "drive": "a concentrically stacked modular actuator for multi-DOF robots, forced-air cooled; motor, size and mass "
             "not in any summary",
    "diameter_mm": None, "mass_kg": None,
    "figures": ["'the maximum continuous current that can be applied to the CoSMo actuator is up to about three times "
                "greater than the rated current in a forced air-cooling environment' (R01)",
                "a 'Simplified thermal equivalent circuit' (STEC) and an experimental thermal section; no R_th value "
                "in any summary (R01, R02: 'I did not retrieve its measured K/W values or airflow speeds')",
                "'Adjacent actuator modules have twice the thermal capacitance and radiating area owing to the "
                "physical contact of the housings' (R01)"],
    "quoted_tool_text": {"R01": ("Its prototype experiments found that the maximum continuous current that can be "
                                 "applied to the CoSMo actuator is up to about three times greater than the rated "
                                 "current in a forced air-cooling environment."),
                         "R02": ("This paper on a forced-air-cooled actuator for multi-DOF robots says the benefit of "
                                 "forced cooling depends on the thermal resistance and thermal capacitance of the "
                                 "winding and housing. I did not retrieve its measured K/W values or airflow speeds.")},
    "fan": "forced air named; no flow in any summary",
    "rth_pair_given": None, "flow_stated": None, "heat_path": None,
    "notes": ("missed by E4 (in none of S01-S14). On the summaries it does not qualify: no R_th values, no flow, no "
              "drive size. Its 'about 3x rated current' is a continuous-rating ratio: reported, not converted (item 12). "
              "A full text to read beside S01 and S02 if mdpi.com is allowed"),
    "review": {"finding": "E4-R2", "label": LABEL},
}

# Other hits of the review searches (R02), recorded for completeness; none is a drive's pair.
src["S16"] = {
    "id": "S16", "found_by": ["R02"],
    "title": "other hits of the review searches without a forced-air R_th pair",
    "who_when": "various (a motor thermal calculator page; a design blog)",
    "urls": ["https://www.firgelliauto.com/es/blogs/calculators/motor-thermal-model-calculator",
             "https://dev.to/peter_hong_44601bad413835/bldc-motor-torque-density-optimization-for-robot-joints-a-"
             "practical-design-guide-1fo9"],
    "drive": "none (typical ranges); one unnamed robot-joint prototype",
    "diameter_mm": None, "mass_kg": None,
    "figures": ["calculator page: typical winding-to-ambient 2-5 C/W for smaller motors and 0.5-2 C/W for larger motors "
                "with forced cooling; still air h 5-25 W/(m^2 K), forced air 25-250 W/(m^2 K) (generic typical "
                "ranges, no drive, no flow)",
                "design blog: housing-to-ambient 1.2 C/W for a robot-joint prototype, no airflow or natural-convection "
                "comparison, source unverified"],
    "fan": "-", "rth_pair_given": False, "flow_stated": None, "heat_path": None,
    "notes": "typical ranges and a single unverified value: no drive's pair with and without forced air",
    "review": {"finding": "E4-R2 (search)", "label": LABEL},
}

REPORTED_NOT_CONVERTED = [
    {"source": "S05", "what": "Moog AML (fan) against AM: continuous torque and power 'up to 1.5 times'"},
    {"source": "S03", "what": "ACSELL deck: +32 % continuous torque with a 4.2 W fan"},
    {"source": "S04", "what": "maxon HEJ 70: about 50 W passive against 140 W 'active', on two different variants"},
    {"source": "S10", "what": "BEAR: 7.8 N m air-cooled against about 21 N m liquid-cooled (liquid, not forced air)"},
    {"source": "S15", "what": "about 3x the rated current as maximum continuous current with forced air (added by "
                              "E4-R2, " + LABEL + ")"},
]

FULL_TEXTS_TO_READ = [
    {"source": "S01", "host": "arxiv.org", "needs": "fan flow; motor size and mass; which R_th goes with the fans"},
    {"source": "S02", "host": "osti.gov", "needs": "fan flow; motor size and mass; the nominal H_m"},
    {"source": "S15", "host": "mdpi.com", "needs": "R_th values without and with forced air; flow; drive size and mass"},
]

# E4-R4: the "What it means" sentence, reworded against E2's later result (E2/E2_summary.md, written 07:11).
R4_WORDING = (
    "A's heat case cannot lean on forced air (E4: no usable figure). Under E2(a)'s pre-registered worst-of-four rule "
    "no reduction passes (0 of 45): at hip pitch, r = 1.2-1.3, C1 and C2 pass but the speed check fails on "
    "fore128-s3's right fore hip pitch (peak 99.2-111.3 rpm over both seed blocks, 102.7-111.3 on the worse one, "
    "against 91.3 rpm). Per policy (supporting, not the verdict), a 1.2-1.3:1 hip-pitch reduction passes C1, C2 and "
    "the speed check for s0, s1 and s2 on both blocks with every transmission (18 of 18 judged rows each; s3 0 of 18). "
    "E2(b) is a separate, simulation-only reading (r = 1.25, 83 rpm cap: rod, belt and cable compatible with today's "
    "four policies); it does not override E2(a). So A's heat case rests on a policy chosen or trained for heat (E3: "
    "fore128-s0 and s1 meet the rule on their own at 35 C) or on the bench heat test; a reduction would want a "
    "policy other than s3 or one trained with the reduction and the cap.")

R1_WORDING = (
    "E4's follow-up queries F01-F18 broke the hard rule 'Web searches use generic part and engineering terms only' "
    "(and PREREG section 4 'E4': 'web search only, generic engineering terms'). 17 of the 18 (all but F14) named "
    "paper authors (F02, F05, F09, F11), makers or products (F07, F11, F13, F16, F18), institutions or named projects "
    "(F08, F09, F10, F17, F18), report or paper identifiers (F03, F06, F08), or quoted paper titles (F01, F04, F07, "
    "F12, F15). None named this project or anyone tied to it. This is a breach of a hard rule, not only a departure "
    "from the pre-registration. It changed no outcome: no source qualifies under item 12, so none of these queries "
    "produced a usable figure. Later E4 work stays with generic engineering terms; reading S01, S02 or S15 in full "
    "needs the owner to allow arxiv.org, osti.gov or mdpi.com, not more targeted queries.")


def main() -> int:
    t0 = time.time()
    rows = []
    for sid in sorted(src, key=lambda k: int(k[1:])):
        s = src[sid]
        q = run_e4.qualifies(s)
        rows.append({**s, "read_date": READ, "access": run_e4.ACCESS, **q})
    usable = [r["id"] for r in rows if r["usable"]]
    if usable:
        print("A source qualifies under item 12:", usable, "- compute C1 per section 4 'E4' (not done here).")
        return 2

    audit = []
    for i, qtext, mode in run_e4.QUERIES:
        cats = NON_GENERIC.get(i, [])
        audit.append({"id": i, "query": qtext, "mode": mode, "generic_terms_only": not cats, "non_generic": cats})
    for q in REVIEW_QUERIES:
        audit.append({"id": q["id"], "query": q["query"], "mode": q["mode"], "generic_terms_only": True,
                      "non_generic": []})
    breaches = [a["id"] for a in audit if not a["generic_terms_only"]]

    sizes = {n: (OUT / n).stat().st_size for n in ("e4_sources.json", "e4_sources.csv", "E4_SUMMARY.md")}
    raw_text = RAW.read_text()
    result = {
        "experiment": "E4 forced air: review fixes", "date": READ, "label": LABEL,
        "evidence": "DESK (web search summaries; no page opened)",
        "prereg": "design/PREREG_CANDIDATES_2026_11_28.md: section 2 'E4', section 3 item 12, section 4 'E4'",
        "item_12_test": "unchanged: run_e4.qualifies (run_e4.py sha256 checked)",
        "review_findings": {
            "E4-R1": {"disposition": "accepted", "breach_queries": breaches,
                      "breach_count": f"{len([b for b in breaches if b.startswith('F')])} of 18 F queries",
                      "generic_F": [a["id"] for a in audit if a["id"].startswith("F") and a["generic_terms_only"]],
                      "outcome_changed": False, "wording_for_final_report": R1_WORDING},
            "E4-R2": {"disposition": "accepted", "added_record": "S15", "also_added": "S16 (other review hits)"},
            "E4-R3": {"disposition": ("accepted as a note; S01's check kept at None (reviewer's sentence not "
                                      "reproduced here), outcome unchanged"), "record": "S01.review"},
            "E4-R4": {"disposition": "accepted", "wording_for_final_report": R4_WORDING,
                      "e2_basis": "E2/E2_summary.md: 'Verdict: no judged (transmission, r, placement) PASSES; 0 of 45.'"},
            "E4-R5": {"disposition": "accepted", "record": "S11: figures withdrawn, unsourced, not used"},
            "E4-R6": {"disposition": "accepted", "bytes": sizes,
                      "kib": {k: round(v / 1024, 1) for k, v in sizes.items()}},
        },
        "query_audit": audit,
        "review_queries": REVIEW_QUERIES,
        "review_fetches": REVIEW_FETCHES,
        "raw_tool_text_file": str(RAW), "raw_tool_text_sha256": sha(RAW), "raw_tool_text": raw_text,
        "sources": rows,
        "usable_sources": usable,
        "outcome": "no usable figure was found",
        "c1_computed": False,
        "forced_air_c1_called": False,
        "reported_not_converted": REPORTED_NOT_CONVERTED,
        "full_texts_to_read_if_hosts_allowed": FULL_TEXTS_TO_READ,
        "files_left_unchanged": {n: sha(OUT / n) for n in ("e4_sources.json", "e4_sources.csv", "E4_SUMMARY.md")},
        "provenance": cc.provenance(runner=Path(__file__), extra={"runner:run_e4.py": SCR / "run_e4.py",
                                                                   "raw_tool_text": RAW}),
    }
    result["wall_time_s"] = round(time.time() - t0, 3)
    (OUT / "e4_review_addendum.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    with open(OUT / "e4_sources_reviewed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "title", "who_when", "drive", "diameter_mm", "mass_kg", "rth_pair_given", "flow_stated",
                    "heat_path", "comparable_drive", "usable", "read_date", "access", "first_url", "review_change",
                    "notes"])
        for r in rows:
            c = r["checks"]
            rv = r.get("review")
            w.writerow([r["id"], r["title"], r["who_when"], r["drive"], r["diameter_mm"], r["mass_kg"],
                        c["rth_pair_given"], r["flow_stated"], r["heat_path"], c["comparable_drive"], r["usable"],
                        r["read_date"], r["access"], (r["urls"] or [""])[0],
                        (rv["finding"] + "; " + rv["label"]) if rv else "", r["notes"]])
    (SCR / "run_e4_review.log").write_text(
        f"E4 review run {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(rows)} source records, usable {usable}; "
        f"non-generic queries {breaches}; outcome: {result['outcome']}\n")
    for r in rows:
        print(r["id"], "usable" if r["usable"] else "not usable", r["checks"])
    print("non-generic queries:", breaches, "|", result["review_findings"]["E4-R1"]["breach_count"])
    print("sizes (bytes):", sizes)
    print("usable sources:", usable, "| outcome:", result["outcome"], "| c1_computed:", result["c1_computed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
