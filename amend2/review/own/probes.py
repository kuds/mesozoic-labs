"""Reviewer's refusal and containment probes of score_pc_dev.py on synthetic roots (default expectation unless said).

    cd review/own && PY -I -B probes.py <workdir>
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import own_roots as O  # noqa: E402

C = O.C
PCID, OFFID = O.PC, O.OFF
PH = {60: 0.11, 42: 0.23, 80: 0.17}
OF = {60: 0.41, 42: 0.36, 80: 0.52}


def snap(hhmm, run, day=1, note="report-only look (reviewer probe)"):
    return f"2026-11-{day:02d} {hhmm} UTC  STANCE7 SNAPSHOT {run} note=\"{note}\""


def line(hhmm, kind, run, rest=""):
    return f"2026-11-01 {hhmm} UTC  STANCE7 {kind} {run}{(' ' + rest) if rest else ''}"


PROBES = [
    # (name, kwargs for build, extra tool args, expect)  expect: "refuse" or "run"
    ("P1_extra_early_snapshot_between_off_launches", dict(lines=(O.L902, snap("18:30", PCID[80]))), (), "refuse"),
    ("P2a_line902_trailing_space", dict(lines=(O.L902 + " ",)), (), "refuse"),
    ("P2b_line902_one_space_less_after_UTC", dict(lines=(O.L902.replace(" UTC  STANCE7", " UTC STANCE7", 1),)), (),
     "refuse"),
    ("P2c_line902_run_id_changed_to_s42", dict(lines=(O.L902.replace(PCID[60], PCID[42], 1),)), (), "refuse"),
    ("P3_line902_missing", dict(lines=()), (), "refuse"),
    ("P4a_report_only_snapshot_after_off_launches_kept", dict(lines=(O.L902, snap("19:30", PCID[42]))), (), "run"),
    ("P4b_off_launches_before_17_44_so_902_is_report_only",
     dict(lines=(O.L902,), off_launch={60: (17, 0), 80: (17, 10), 42: (17, 20)}), (), "refuse"),
    ("P5a_off_relaunch_after_a_report_only_snapshot",
     dict(lines=(O.L902, snap("19:30", PCID[42])),
          extra_notes=(line("20:00", "RELAUNCH", OFFID[60] + "_b", "seed=60"),)), (), "refuse"),
    ("P5b_off_relaunch_alone", dict(lines=(O.L902,), extra_notes=(line("20:00", "RELAUNCH", OFFID[60] + "_b",
                                                                       "seed=60"),)), (), "run"),
    ("P6_line902_twice", dict(lines=(O.L902, O.L902)), (), "refuse"),
    ("P7_snapshot_of_suffixed_phase_id", dict(lines=(O.L902, snap("18:00", PCID[42] + "_b"))), (), "refuse"),
    ("P8_snapshot_same_minute_as_last_off_launch_seconds",
     dict(lines=(O.L902, snap("18:56:59", PCID[80]))), (), "refuse_or_run"),
    ("P9_phase_F2START_before_last_off_launch", dict(lines=(O.L902, line("18:30", "F2-START", PCID[80]))), (), "run"),
    ("P10_expect_dropped_override_with_extra_snapshot",
     dict(lines=(O.L902, snap("18:30", PCID[80]))), ("--expect-dropped", O.L902_SHA, "--expect-dropped",
                                                      "SHA_OF_EXTRA"), "run"),
]


def main(argv):
    work = Path(argv[1]).resolve()
    for name, kw, extra, expect in PROBES:
        root = work / name
        if root.exists():
            shutil.rmtree(root)
        rnd = O.build(root, PH, OF, **kw)
        if "SHA_OF_EXTRA" in extra:
            import hashlib
            ex = snap("18:30", PCID[80])
            extra = tuple(hashlib.sha256(ex.encode()).hexdigest() if x == "SHA_OF_EXTRA" else x for x in extra)
        out = root / "out"
        rc, text, err, changed = O.run_dev(rnd, out, *extra)
        made = sorted(p.name for p in out.iterdir()) if out.exists() else "absent"
        ref = [x for x in text.splitlines() if x.startswith("REFUSED")]
        verdict = ("refused" if rc == 3 else f"ran (exit {rc})")
        ok = (expect == "refuse" and rc == 3) or (expect == "run" and rc in (0, 10, 11)) or expect == "refuse_or_run"
        print(f"== {name}: {verdict}; expected {expect}: {'OK' if ok else 'NOT AS EXPECTED'}; stderr {len(err)} B; "
              f"out folder: {made}; changed outside --out: {changed or 'none'}")
        for x in ref:
            print("   " + x[:400])
        if rc in (0, 10, 11):
            dl = [x for x in text.splitlines() if x.startswith("    dropped (content removed")]
            print("   " + (dl[0][:300] if dl else "(no dropped line)"))
            for reading in ("pinned", "deviation"):
                j = json.loads((out / f"{reading}.json").read_text())
                ot = (j.get("reading") or {}).get("one_test") or {}
                print(f"   {reading}: not_blind {j['blindness']['not_blind']}; first opening "
                      f"{j['blindness']['first_opening'] and (j['blindness']['first_opening']['kind'], j['blindness']['first_opening']['time'])}; "
                      f"one test {ot.get('text')}")
            hdr = [x for x in text.splitlines() if x.startswith("DEVIATION READING (flagged")]
            print("   header printed: " + (hdr[0][:140] if hdr else "none"))
            last = [x for x in text.splitlines() if "differ;" in x]
            print("   " + (last[0].strip()[:200] if last else ""))
            print("   " + " | ".join(x.strip()[:120] for x in text.splitlines() if x.startswith("  [deviation] OUTCOME")))
    # --out probes
    base = work / "P1_extra_early_snapshot_between_off_launches"
    rnd_root = base / "r"
    print("== out-folder probes (P3 RUNLOG has no 902 line is irrelevant: the out check comes first)")
    rnd = O.build(work / "Pout", PH, OF)
    sep = work / "Pout_runlog_dir"
    sep.mkdir(exist_ok=True)
    rl = sep / "run_log.txt"
    rl.write_bytes(rnd.runlog.read_bytes())
    link = work / "Pout_link_into_round"
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(rnd.root / "newsub", link)
    for nm, out, ex in (("inside the RUNLOG folder", sep / "o", ("--runlog", rl)),
                        ("a symlink resolving into the round", link, ()),
                        ("the RUNLOG folder itself (non-empty)", sep, ("--runlog", rl))):
        rc, text, err, changed = O.run_dev(rnd, out, *ex)
        ref = [x for x in text.splitlines() if x.startswith("REFUSED")]
        print(f"   --out {nm}: exit {rc}; {ref[0][:200] if ref else 'no refusal'}; changed: {changed or 'none'}; "
              f"round/newsub exists: {(rnd.root / 'newsub').exists()}")
    del rnd_root
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main(sys.argv))
