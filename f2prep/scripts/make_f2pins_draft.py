"""Draft (never pin) <PCR>/F2_PINS.json per criteria HOW THE COORDINATOR PINS 5: tool_sha256 = every ST/tools/pc file
score_pc.py's tool_hashes lists (r10round_common.tool_hashes(), computed from the files now), each compared with
pins_pc.json's tool_sha256 and APPROVED's pins; validations {"VP1", "VP4"} set to "PASS" only where every log this
builder reads shows the criteria's expected result (else the failing parts are printed and the value is "NOT PASS");
validation_logs_sha256 (the logs, paths relative to ST/f2prep).  Writes compact JSON to --out (never into round/).

    PY -I -B make_f2pins_draft.py --f2prep DIR --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ST = Path("/home/user/work/pc_study")
T = ST / "tools/pc"
spec = importlib.util.spec_from_file_location("r10round_common", T / "r10round_common.py")
C = importlib.util.module_from_spec(spec)
sys.modules["r10round_common"] = C
spec.loader.exec_module(C)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def last_line(p: Path) -> str:
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f2prep", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F = Path(a.f2prep)
    out = Path(a.out).resolve()
    if (ST / "round") in out.parents:
        print("REFUSED: the draft is never written into round/")
        return 2
    pins = json.loads((T / "pins_pc.json").read_text())
    ts_pins = pins.get("tool_sha256") or {}
    ap_pins = json.loads((ST / "round/APPROVED").read_text())["pins"]["tools_pc"]
    tool = C.tool_hashes()
    print(f"tool_hashes() lists {len(tool)} files: {sorted(tool)}")
    flags = []
    for n, h in sorted(tool.items()):
        e1, e2 = ts_pins.get(n) == h, ap_pins.get(n) == h
        print(f"  {n} {h} = pins_pc.json tool_sha256: {e1}; = APPROVED pins: {e2}")
        if not (e1 and e2):
            flags.append(n)
    # ---- VP1
    vp1_parts = {}
    t = (F / "vp1/vp1_m1_compare.out").read_text()
    vp1_parts["M1 HEAD copy = round panel (s80 final), per-step CSV sha256 and reductions"] = \
        "VP1 (a) M1 round panel (8737cd3d copy) vs HEAD copy, stock s80 final: EQUAL" in t
    vp1_parts["M1 widened pair = M1 stock pair"] = "VP1 (c) M1 HEAD copy stock vs widened (zero phase columns): EQUAL" in t
    m3 = F / "vp1/vp1_m3_compare.out"
    t3 = m3.read_text() if m3.is_file() else ""
    vp1_parts["M3 HEAD copy certificate = round's s80 final certificate"] = \
        "VP1 (b) M3 round panel (8737cd3d copy) vs HEAD copy, stock s80 final: CERTIFICATE EQUAL" in t3
    vp1_parts["M3 widened zip certificate = round's s80 final certificate"] = \
        "VP1 (d) M3 HEAD copy stock vs widened zip (zero phase columns): CERTIFICATE EQUAL" in t3
    # ---- VP4
    vp4_parts = {}
    for arm in ("pc", "off"):
        s = (F / f"vp4/standin_{arm}.out").read_text()
        vp4_parts[f"stand-in --as {arm}: 25 of 25 class keys per STANCE6 run, ALL EQUAL"] = (
            s.count("25 of 25 class keys equal") == 3 and "ALL EQUAL: True" in s)
    vp4_parts["TD = DESIGN/comparators.out"] = last_line(F / "vp4/vp4_td_check.out") == "VP4 TD part: PASS"
    w = (F / "vp4/vp4_widen_s80.out").read_text()
    vp4_parts["manipulation check NOT IN FORCE on the widened s80 final, failing (i) only; V and actor bitwise"] = (
        "WIDENED (zero phase columns): verdict NOT IN FORCE, failing ['(i)']" in w
        and w.count("bitwise equal True") == 3)
    for arm, want in (("pc", "IN FORCE"), ("off", "NOT IN FORCE")):
        run = "stance7_r10LTSBpc_s42" if arm == "pc" else "stance7_r10LTSBoff_s42"
        cj = json.loads((F / f"vp4/f2dry_{arm}/complete.json").read_text())
        comp = cj.get("completeness") or {}
        c4p = any(re.match(r"^\s+PASS\s+C4 \(p\) ", ln) for ln in (F / f"vp4/f2dry_{arm}.out").read_text().splitlines())
        mc = json.loads((F / f"vp4/f2dry_{arm}/panels/{run}/final/c8/manip_check.json").read_text())
        vp4_parts[f"f2dry.py --arm {arm}: C8 PASS ({want}), C4 (p) PASS"] = (
            (comp.get("C8") or {}).get("ok") is True and mc.get("verdict") == want and c4p)
    vp4_parts["test_score_pc.py: 65 passed"] = "65 passed" in last_line(F / "vp4/pytest_score_pc.log")
    vp4_parts["test_manip_pc.py: 7 passed (smoke pair IN FORCE, widened STANCE6 pair NOT IN FORCE)"] = \
        "7 passed" in last_line(F / "vp4/pytest_manip_pc.log")
    mlog = F / "vp4/mutate_pc.log"
    vp4_parts["mutate_pc.py: KILLED 49 of 49"] = mlog.is_file() and last_line(mlog) == "KILLED 49 of 49"
    for name, parts in (("VP1", vp1_parts), ("VP4", vp4_parts)):
        print(f"{name}:")
        for k, v in parts.items():
            print(f"  {'PASS' if v else 'NOT PASS'}  {k}")
    vp1 = "PASS" if all(vp1_parts.values()) else "NOT PASS"
    vp4 = "PASS" if all(vp4_parts.values()) else "NOT PASS"
    logs = ["vp1/vp1_m1_compare.out", "vp1/vp1_m3_compare.out", "vp1/m1_stock/m1_summary.json",
            "vp1/m1_widened/m1_summary.json", "vp1/m3_stock/m3_summary.json", "vp1/m3_widened/m3_summary.json", "vp1/evidence_sha256.txt",
            "vp4/standin_pc.out", "vp4/standin_off.out", "vp4/vp4_td_check.out", "vp4/vp4_widen_s80.out",
            "vp4/f2dry_pc.out", "vp4/f2dry_off.out", "vp4/pytest_score_pc.log", "vp4/pytest_manip_pc.log",
            "vp4/mutate_pc.log", "vp4/evidence_sha256.txt"]
    vlogs = {}
    for rel in logs:
        p = F / rel
        if p.is_file():
            vlogs[rel] = sha(p)
    draft = {"tool_sha256": dict(sorted(tool.items())), "validations": {"VP1": vp1, "VP4": vp4},
             "validation_logs_sha256": {"VP1": {k: v for k, v in vlogs.items() if k.startswith("vp1/")},
                                        "VP4": {k: v for k, v in vlogs.items() if k.startswith("vp4/")}},
             "measuring_copy": {"commit": json.loads((F / "build_copy.out").read_text())["commit"],
                                "archive_sha256": json.loads((F / "build_copy.out").read_text())["archive_sha256"],
                                "tree_digest": json.loads((F / "build_copy.out").read_text())["tree_digest"]},
             "draft": "ST/f2prep builder B, not pinned; the coordinator writes <PCR>/F2_PINS.json once"}
    out.write_text(json.dumps(draft, separators=(",", ":")), encoding="utf-8")
    print(f"tool pins differing from pins_pc.json or APPROVED: {flags or 'none'}")
    print(f"validations: VP1 {vp1}, VP4 {vp4}")
    print(f"wrote {out.name} sha256 {sha(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
