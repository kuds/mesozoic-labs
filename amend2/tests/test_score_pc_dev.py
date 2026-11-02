"""AMENDMENT 2: tests of score_pc_dev.py (the flagged DEVIATION reading of THE ONE TEST) on the synthetic six-run roots
that ST/tools/tests/synth_pc.py builds (imported read only; nothing there is changed).  Every deviation run is the tool
run as a subprocess (PY -I -B), as the round would run it; the pinned scorer is run unchanged by the tool.

    cd ST/amend2/tests && PY -I -B -m pytest -q -p no:cacheprovider --basetemp=ST/amend2/dry/pytest_tmp
    (AMEND2_DEV_TOOL=<path> points the tests at another copy of the tool: the mutation runner's mutants)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DEV = Path(os.environ.get("AMEND2_DEV_TOOL") or HERE.parent / "tools" / "score_pc_dev.py")
TOOLS_TESTS = Path("/home/user/work/pc_study/tools/tests")
PINNED_TOOLS = Path("/home/user/work/pc_study/tools/pc")
sys.path.insert(0, str(TOOLS_TESTS))
import synth_pc as S  # noqa: E402

C, SP = S.C, S.SP
_spec = importlib.util.spec_from_file_location("score_pc_dev_under_test", DEV)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)

UTC = timezone.utc
#: RUNLOG line 902 byte for byte (the 17:44 UTC opening; its sha256 is the tool's PINNED_SNAPSHOT_LINE_SHA256)
PINNED_LINE = ('2026-11-01 17:44 UTC  STANCE7 SNAPSHOT stance7_r10LTSBpc_s60 note="THE BLIND PERIOD opening (criteria '
               "lines 362-373): the maintainer pasted into chat the s60 PHASE tab's notebook output beyond the "
               'launch-check rows, read by the maintainer and the coordinator: the in-training evaluations at 11,796,480 '
               'and 11,927,552 (episode_reward and length), the final-model save and its evaluation (mean_reward, length, '
               'forward velocity, distance, success rate), the best-model evaluation step (6,815,744), the selected '
               'checkpoint (robust_best_model) and the selected-model 30-episode evaluation. No tail command, TD or '
               'ledger figure was shown. No OFF run has launched; under the pinned criteria every OFF run launched after '
               'this line is NOT BLIND and enters THE ONE TEST as missing (lines 297-302, 359-361). Noted at once; '
               'nothing else opened; the maintainer is asked how to proceed."')
#: a non-note line naming a RUN_ID after the opening, as RUNLOG's NOTE lines are: the scorer prints it with its line
#: number, so a deviation copy that renumbered lines would show differences that are not the deviation
NOTE_LINE = '2026-11-01 18:16 UTC  STANCE7 NOTE the maintainer: "kicked off stance7_r10LTSBoff_s60" (synthetic)'
OFF_LAUNCH = {60: (18, 12), 80: (18, 20), 42: (18, 56)}          # RUNLOG lines 907, 911, 923 (UTC, 2026-11-01)
OFF_IDS = [f"{C.ARM_PREFIX['off']}{s}" for s in C.SEEDS]
LOWER = "LOWER THAN EVERY CONCURRENT SWITCH-OFF RUN (exact permutation p = 1/20)"
ONE_TEST = "    THE ONE TEST (phase vs the concurrent switch-off runs): "
EXPECTED_DIFF_PREFIXES = ("OUTCOME BLINDNESS", "    s60: phase", "    s42: phase", "    s80: phase", ONE_TEST)


def snap(hhmm, run=f"{C.RUN_ID_PREFIX}42"):
    return f"2026-11-01 {hhmm} UTC  STANCE7 SNAPSHOT {run} note=\"report-only look (synthetic)\""


def build(root, phase_cmd=0.05, off_cmd=0.5, off_per_seed=None, off_seeds=(60, 42, 80), lines=(PINNED_LINE,),
          phase_per_seed=None):
    """Six runs as test_score_pc's _round makes them, the switch-off runs launched at RUNLOG's times (after 17:44),
    the given RUNLOG lines added (default: the pinned opening line), written as synth_pc's Round.score writes them."""
    rnd = S.Round(Path(root) / "r")
    for s in C.SEEDS:
        kw = dict(ledger=S.ledger_with_cmd(phase_cmd))
        kw.update((phase_per_seed or {}).get(s, {}))
        S.standard_run(rnd, s, **kw)
    for s in off_seeds:
        kw = dict(ledger=S.ledger_with_cmd(off_cmd))
        kw.update((off_per_seed or {}).get(s, {}))
        h, m = OFF_LAUNCH[s]
        S.standard_run(rnd, s, arm="off", launch=datetime(2026, 11, 1, h, m, tzinfo=UTC), **kw)
    rnd.notes.append(NOTE_LINE)
    rnd.notes.extend(lines)
    rnd.pins["stage_config_view_sha256"] = S.view_pin_of(S.Run(f"{C.RUN_ID_PREFIX}60", S.T0, rnd.pins))
    rnd.pins["off_arm"]["stage_config_view_sha256"] = S.view_pin_of(S.Run(f"{C.ARM_PREFIX['off']}60", S.T0, rnd.pins))
    rnd.write_meta()
    S.make_trigger_stores(rnd.root, rnd.runlog)
    S.bind_panel_records(rnd.root)
    return rnd


def run_tool(cwd, *args):
    """The tool as the round runs it (PY -I -B, a clean environment), with exactly these arguments."""
    tmp = Path(cwd)
    tmp.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-I", "-B", str(DEV), *map(str, args)], capture_output=True, text=True,
                       cwd=str(tmp), env={"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp), "LANG": "C.UTF-8"})
    assert p.stderr == "", p.stderr
    return p.returncode, p.stdout


def run_dev(rnd, out, *extra):
    # cwd: the test's own folder, never next to a refused --out
    return run_tool(Path(rnd.root).parent / "cwd", "--out", out, "--round", rnd.root, "--runlog", rnd.runlog, "--pins",
                    rnd.pins_path, *extra)


def real_runlog_lines():
    """RUNLOG (read only), as lines without their line feeds."""
    return Path(C.RUNLOG).read_bytes().decode("utf-8").splitlines()


def decision_line():
    """RUNLOG line 949, the maintainer decision "Pin both readings", byte for byte (read from RUNLOG)."""
    dec = [x for x in real_runlog_lines() if "STANCE7 NOTE MAINTAINER DECISION" in x and '"Pin both readings"' in x]
    assert len(dec) == 1
    return dec[0]


def sections(text):
    L = text.splitlines()
    a = next(i for i, x in enumerate(L) if " PINNED READING (the round's reading of record" in x)
    b = next(i for i, x in enumerate(L) if " END OF THE PINNED READING " in x)
    h = L.index(D.HEADER)
    d = next(i for i, x in enumerate(L) if x.startswith("DIFFERENCES between the pinned run"))
    pinned = L[a + 2:b]
    dev = [x[len("  [deviation] "):] for x in L[h + 1:d] if x.startswith("  [deviation] ")]
    diffs = L[d + 1:]
    assert a < b < h < d
    return pinned, dev, diffs


def diff_pairs(diffs):
    pin = [x[len("        pinned:    "):] for x in diffs if x.startswith("        pinned:    ")]
    dev = [x[len("        deviation: "):] for x in diffs if x.startswith("        deviation: ")]
    return pin, dev


def one_test(out, name):
    return json.loads((Path(out) / f"{name}.json").read_text())["reading"]["one_test"]


def assert_only_blindness_differs(text, c1_listing=1, p1_copies=0):
    """Every differing output line is a blindness line, THE ONE TEST line or (once F2 is read) C1's printed listing of
    the run's note lines minus the dropped line (and, for a LOST run, PRESENCE p1's printed copies of it): checked here
    on the lines themselves and on the tool's own kinds."""
    _p, _d, diffs = sections(text)
    pin, dev = diff_pairs(diffs)
    assert len(pin) == len(dev) == 5 + c1_listing + p1_copies, diffs
    for x, y in zip(pin, dev):
        if x.startswith(EXPECTED_DIFF_PREFIXES):
            assert y.startswith(x[:16]), (x, y)
        else:
            assert x.startswith(('        value: {"launches": ', '      p1: ')), x
            assert '"SNAPSHOT"' in x and '"SNAPSHOT"' not in y, x
            assert any(x.replace(a, "", 1) == y for a in (f', "{i}": "SNAPSHOT"' for i in range(1, 200))), (x, y)
    heads = [x for x in diffs if x.startswith("    output line ")]
    kinds = [x[x.index(" [") + 2:x.rindex("]")] for x in heads]
    assert kinds.count("blindness: the first opening and the NOT BLIND runs") == 1, kinds
    assert kinds.count("blindness: the seed's OFF run blind or not") == 3 and kinds.count("THE ONE TEST") == 1, kinds
    c1 = [k for k in kinds if k.startswith("C1's listing of the run's note lines without dropped line")]
    assert len(c1) == c1_listing + p1_copies, kinds
    assert sum(", as PRESENCE p1 prints it" in k for k in c1) == p1_copies, kinds
    jd = [x for x in diffs if x.startswith("    JSON ")]
    assert jd and not [x for x in jd if "[UNEXPECTED]" in x], jd
    pc60 = f"    JSON runs.{C.RUN_ID_PREFIX}60."
    assert all(x.startswith(("    JSON blindness.", "    JSON reading.one_test.", pc60 + "completeness.C1.value.lines."))
               or (p1_copies and x.startswith((pc60 + "presence[", pc60 + "d16.presence[")) and ".p1.value.lines." in x
                   and x.endswith(': pinned "SNAPSHOT" | deviation "(absent)"')) for x in jd), jd
    assert ("every one is a blindness line, THE ONE TEST or C1's listing of a dropped line (C1's own, or its PRESENCE "
            "p1 / D-16 copies)") in text
    assert "UNEXPECTED" not in "\n".join(diffs)


# ================================================================================================ the constants
def test_pinned_line_constant_is_the_lines_sha256():
    assert hashlib.sha256(PINNED_LINE.encode("utf-8")).hexdigest() == D.PINNED_SNAPSHOT_LINE_SHA256
    assert len(PINNED_LINE.encode("utf-8")) == 873
    assert PINNED_LINE.startswith("2026-11-01 17:44 UTC  STANCE7 SNAPSHOT stance7_r10LTSBpc_s60 note=")


def test_runlog_line_902_is_the_pinned_line():
    """RUNLOG (read only) still holds the pinned line as line 902; the decision line's sha256 is the tool's."""
    lines = real_runlog_lines()
    assert lines[901] == PINNED_LINE
    dec = decision_line()
    assert lines[948] == dec and dec.startswith(D.DECISION_TIME)
    assert hashlib.sha256(dec.encode("utf-8")).hexdigest() == D.DECISION_LINE_SHA256


# ================================================================================================ (a) complete separation
def test_a_snapshot_before_every_off_launch_complete_separation(tmp_path):
    rnd = build(tmp_path)
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    pinned, dev, _diffs = sections(text)
    # the pinned reading: the three OFF runs NOT BLIND, missing at -infinity: never LOWER (p = 20/20 here)
    p_ot = next(x for x in pinned if x.startswith(ONE_TEST))
    assert "NOT SEPARATED (p = 20/20)" in p_ot and f"not blind {sorted(OFF_IDS)}" in p_ot
    assert one_test(out, "pinned")["k"] == 20 and one_test(out, "pinned")["off_missing"] == [True] * 3
    # the deviation reading: as if blind: complete separation reads LOWER at p = 1/20
    d_ot = next(x for x in dev if x.startswith(ONE_TEST))
    assert d_ot.startswith(ONE_TEST + LOWER) and "not blind []" in d_ot
    ot = one_test(out, "deviation")
    assert ot["k"] == 1 and ot["lower"] is True and ot["off_missing"] == [False] * 3 and ot["p"] == "1/20"
    assert any(x.startswith("OUTCOME BLINDNESS") and "not blind (enter THE ONE TEST as missing) []" in x for x in dev)
    assert any(x.startswith("READING: line 4") for x in dev) and any(x == "STATUS: FINAL" for x in dev)
    assert [x for x in dev if x.startswith("  pc s")] and [x for x in dev if x.startswith("  off s")]
    assert_only_blindness_differs(text)
    assert "maintainer decision line: NOT FOUND in the RUNLOG read" in text
    assert "expected: the pinned line (default)" in text


def test_a_deviation_copy_blanks_only_the_pinned_line(tmp_path):
    rnd = build(tmp_path)
    out = tmp_path / "out"
    rc, _text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL
    a = (out / "runlog_asread.txt").read_bytes()
    assert a == rnd.runlog.read_bytes()
    al, bl = a.decode().splitlines(keepends=True), (out / "runlog_deviation.txt").read_bytes().decode().splitlines(
        keepends=True)
    assert len(al) == len(bl)
    changed = [i for i, (x, y) in enumerate(zip(al, bl)) if x != y]
    assert len(changed) == 1 and al[changed[0]] == PINNED_LINE + "\n" and bl[changed[0]] == "\n"


# ================================================================================================ (b), (c) identical
def test_b_no_snapshot_outputs_byte_identical(tmp_path):
    rnd = build(tmp_path, lines=())
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out, "--expect-dropped", "none")
    assert rc == SP.EXIT_FINAL, text
    assert (out / "pinned.out").read_bytes() == (out / "deviation.out").read_bytes()
    assert (out / "pinned.json").read_bytes() == (out / "deviation.json").read_bytes()
    assert (out / "runlog_asread.txt").read_bytes() == (out / "runlog_deviation.txt").read_bytes()
    assert "(no output line differs: the two outputs are byte-identical)" in text and "(no JSON value differs)" in text
    assert "NON-DEFAULT, given by --expect-dropped" in text
    assert one_test(out, "pinned")["k"] == 1
    # the default expectation (exactly the pinned line) refuses a RUNLOG without it
    rc2, text2 = run_dev(rnd, tmp_path / "out2")
    assert rc2 == D.EXIT_REFUSED and "REFUSED: the dropped set [] " in text2
    assert not (tmp_path / "out2" / "pinned.out").exists()


def test_c_snapshot_only_after_every_off_launch_identical(tmp_path):
    rnd = build(tmp_path, lines=(snap("19:30"),))
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out, "--expect-dropped", "none")
    assert rc == SP.EXIT_FINAL, text
    assert (out / "pinned.out").read_bytes() == (out / "deviation.out").read_bytes()
    assert (out / "pinned.json").read_bytes() == (out / "deviation.json").read_bytes()
    assert one_test(out, "pinned")["k"] == 1 and one_test(out, "pinned")["not_blind_entered_as_missing"] == []


def test_c_later_report_only_snapshot_is_kept_and_changes_nothing(tmp_path):
    """The pinned line and a report-only SNAPSHOT after every OFF launch: only the pinned line is dropped; the later
    line stays in the deviation copy and the deviation reads as with no SNAPSHOT at all."""
    late = snap("19:30")
    rnd = build(tmp_path, lines=(PINNED_LINE, late))
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    dev_lines = (out / "runlog_deviation.txt").read_text().splitlines()
    assert late in dev_lines and PINNED_LINE not in dev_lines
    assert one_test(out, "deviation")["k"] == 1 and one_test(out, "pinned")["k"] == 20
    assert_only_blindness_differs(text)


def test_status_passes_through_provisional_in_both(tmp_path):
    """The tool's exit code is the pinned run's; both runs print the scorer's own STATUS (here PROVISIONAL: no
    F2_PINS.json in the round)."""
    rnd = build(tmp_path, lines=())
    (rnd.root / SP.F2_PINS_NAME).unlink()
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out, "--expect-dropped", "none")
    assert rc == SP.EXIT_PROVISIONAL, text
    pinned, dev, _diffs = sections(text)
    for sec in (pinned, dev):
        assert "STATUS: PROVISIONAL (read, printed, not acted on)" in sec
        assert any(x.startswith("    PROVISIONAL: F2_PINS.json absent or unreadable") for x in sec), sec
    assert "exit codes: pinned 11, deviation 11" in text


# ================================================================================================ (d) LOST / never launched
def test_d_off_never_launched_missing_in_both(tmp_path):
    rnd = build(tmp_path, off_seeds=(60, 80))
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    p, d = one_test(out, "pinned"), one_test(out, "deviation")
    assert p["off_missing"] == [True, True, True] and p["k"] == 20
    assert d["off_missing"] == [False, True, False] and d["k"] == 11 and not d["lower"]
    pinned, dev, _diffs = sections(text)
    for sec in (pinned, dev):
        assert any(x.startswith("  off s42 None: NEVER LAUNCHED (no class; THE ONE TEST: missing, -infinity)")
                   for x in sec), sec


def test_d_off_lost_missing_in_both(tmp_path):
    rnd = build(tmp_path, off_per_seed={80: dict(c8={"final": "IN FORCE"})})
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    p, d = one_test(out, "pinned"), one_test(out, "deviation")
    assert p["off_missing"] == [True, True, True]
    assert d["off_missing"] == [False, False, True] and d["k"] == 11 and not d["lower"]
    pinned, dev, _diffs = sections(text)
    for sec in (pinned, dev):
        assert any(x.startswith(f"  off s80 {OFF_IDS[2]}: LOST") for x in sec), sec


# ================================================================================================ (e) partial separation
def test_e_partial_separation_no_snapshot_same_p(tmp_path):
    rnd = build(tmp_path, off_per_seed={42: dict(ledger=S.ledger_with_cmd(0.01))}, lines=())
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out, "--expect-dropped", "none")
    assert rc == SP.EXIT_FINAL, text
    p, d = one_test(out, "pinned"), one_test(out, "deviation")
    assert p["k"] == d["k"] == 4 and p["p"] == d["p"] == "4/20" and p["text"] == d["text"] == "NOT SEPARATED (p = 4/20)"


def test_e_partial_separation_deviation_p_is_the_pinned_scorers_without_the_snapshot(tmp_path):
    rnd = build(tmp_path, off_per_seed={42: dict(ledger=S.ledger_with_cmd(0.01))})
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    assert one_test(out, "pinned")["k"] == 20
    d = one_test(out, "deviation")
    # the pinned scorer, in process, on a RUNLOG copy with the line deleted (not blanked)
    cut = tmp_path / "runlog_without_snapshot.txt"
    cut.write_text("".join(x for x in rnd.runlog.read_text().splitlines(keepends=True) if x != PINNED_LINE + "\n"))
    ref, _st = SP.score(rnd.root, cut, rnd.pins_path, None, comparators=S.comparators_cached())
    r = json.loads(json.dumps(S.SC._jsonable(ref["reading"]["one_test"]), default=str))
    assert d["k"] == r["k"] == 4 and d["p"] == r["p"] and d["text"] == r["text"]
    assert d["phase"] == r["phase"] and d["off"] == r["off"] and d["off_missing"] == r["off_missing"]


# ================================================================================================ (f) refusals
def _tools_copy(tmp_path):
    t = tmp_path / "tools_copy"
    shutil.copytree(PINNED_TOOLS, t)
    return t


def test_f_refuses_changed_scorer_bytes(tmp_path):
    rnd = build(tmp_path)
    t = _tools_copy(tmp_path)
    rc0, text0 = run_dev(rnd, tmp_path / "out0", "--tools", t)          # the unchanged copy is accepted
    assert rc0 == SP.EXIT_FINAL and "REFUSED" not in text0
    b = (t / "score_pc.py").read_bytes()
    assert b"score_pc pc-3" in b
    (t / "score_pc.py").write_bytes(b.replace(b"score_pc pc-3", b"score_pc pc-4", 1))
    rc, text = run_dev(rnd, tmp_path / "out", "--tools", t)
    assert rc == D.EXIT_REFUSED and "REFUSED:" in text and "score_pc.py sha256" in text, text
    assert not (tmp_path / "out").exists()


def test_f_refuses_changed_imported_tool_even_with_pins_refilled(tmp_path):
    rnd = build(tmp_path)
    t = _tools_copy(tmp_path)
    f = t / "pc_common.py"
    f.write_bytes(f.read_bytes() + b"\n")
    pins = json.loads((t / "pins_pc.json").read_text())
    pins["tool_sha256"]["pc_common.py"] = hashlib.sha256(f.read_bytes()).hexdigest()
    (t / "pins_pc.json").write_text(json.dumps(pins))
    rc, text = run_dev(rnd, tmp_path / "out", "--tools", t)
    assert rc == D.EXIT_REFUSED and "pins_pc.json: sha256 is not APPROVED's" in text, text
    (t / "pins_pc.json").write_bytes((PINNED_TOOLS / "pins_pc.json").read_bytes())
    rc2, text2 = run_dev(rnd, tmp_path / "out2", "--tools", t)
    assert rc2 == D.EXIT_REFUSED and "pc_common.py sha256" in text2, text2


@pytest.mark.parametrize("lines,why", [
    ((PINNED_LINE, snap("17:50")), "two early SNAPSHOT lines"),
    ((PINNED_LINE.replace("11,796,480", "11,796,481"),), "the pinned line edited by one byte"),
    ((PINNED_LINE, snap("18:56")), "a SNAPSHOT in the last OFF launch's minute (the pinned rule reads it NOT BLIND)"),
    ((PINNED_LINE.replace("17:44", "18:57"),), "the pinned line's time moved after every OFF launch"),
])
def test_f_refuses_a_dropped_set_other_than_the_pinned_line(tmp_path, lines, why):
    rnd = build(tmp_path, lines=lines)
    rc, text = run_dev(rnd, tmp_path / "out")
    assert rc == D.EXIT_REFUSED and "REFUSED: the dropped set" in text, (why, text)
    assert not (tmp_path / "out" / "pinned.out").exists() and not (tmp_path / "out" / "runlog_deviation.txt").exists()


def test_f_refuses_an_out_folder_inside_the_round_or_not_empty(tmp_path):
    rnd = build(tmp_path)
    rc, text = run_dev(rnd, rnd.root / "dev_out")
    assert rc == D.EXIT_REFUSED and "is inside" in text and not (rnd.root / "dev_out").exists()
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    rc2, text2 = run_dev(rnd, full)
    assert rc2 == D.EXIT_REFUSED and "exists and is not an empty folder" in text2
    # inside --tools: a copy of the pinned tools stands in for them (no test ever names a path under the pinned
    # tools folder as --out, so a mutant that skipped the check could not write there)
    t = _tools_copy(tmp_path)
    rc3, text3 = run_dev(rnd, t / "dev_out", "--tools", t)
    assert rc3 == D.EXIT_REFUSED and "is inside" in text3 and not (t / "dev_out").exists()


# ================================================================================================ (g) containment
def _state(root, skip=None):
    st = {}
    for dp, _dn, fn in os.walk(root):
        if skip and Path(dp).resolve().is_relative_to(skip):
            continue
        for f in fn:
            p = Path(dp) / f
            s = p.lstat()
            st[str(p)] = (s.st_size, s.st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "")
    return st


def test_g_the_deviation_run_writes_only_into_its_out_folder(tmp_path):
    rnd = build(tmp_path)
    out = (tmp_path / "out").resolve()
    watched = [tmp_path, PINNED_TOOLS.parent, HERE.parent / "tools", Path(DEV).parent]
    before = {str(w): _state(w, skip=out) for w in watched}
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    after = {str(w): _state(w, skip=out) for w in watched}
    for w in before:
        new = {k: v for k, v in after[w].items() if before[w].get(k) != v}
        gone = [k for k in before[w] if k not in after[w]]
        assert not gone and not new, (w, new, gone)
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["runlog_asread.txt", "runlog_deviation.txt", "pinned.out", "pinned.err", "pinned.json", "deviation.out",
         "deviation.err", "deviation.json"])
    assert (out / "pinned.err").read_bytes() == b"" and (out / "deviation.err").read_bytes() == b""
    assert f"CONTAINMENT: {rnd.root.resolve()}: unchanged" in text
    assert "CONTAINMENT: the scorer's cwd and TMPDIR (under --out): left empty, removed" in text


# ================================================================================================ review fixes
# ---- M1: --expect-dropped is refused on the round's RUNLOG, the pinned round and a copy of the round's RUNLOG
def test_m1_expect_dropped_refused_on_the_rounds_runlog_and_the_pinned_round(tmp_path):
    """The guard runs before the --out check; --out here is a non-empty folder, so a run past the guard is refused by
    the --out check (another message) and nothing of the real round is read or listed."""
    rnd = build(tmp_path, lines=())
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    link = tmp_path / "link_to_runlog.txt"
    link.symlink_to(Path(C.RUNLOG))
    cwd = tmp_path / "cwd"
    base = ["--out", full, "--pins", rnd.pins_path]
    for runlog, rnd_dir, why in ((C.RUNLOG, rnd.root, "--runlog"), (link, rnd.root, "--runlog"),
                                 (rnd.runlog, D.PINNED_ROUND, "--round")):
        rc, text = run_tool(cwd, *base, "--runlog", runlog, "--round", rnd_dir, "--expect-dropped", "none")
        assert rc == D.EXIT_REFUSED and f"REFUSED: --expect-dropped is for synthetic tests only: {why} " in text, text
        assert "exists and is not an empty folder" not in text
    # controls: the default expectation on the round's RUNLOG, and --expect-dropped on a synthetic RUNLOG and round,
    # pass the guard (and are refused by the --out check)
    for extra, runlog in (((), C.RUNLOG), (("--expect-dropped", "none"), rnd.runlog)):
        rc, text = run_tool(cwd, *base, "--runlog", runlog, "--round", rnd.root, *extra)
        assert rc == D.EXIT_REFUSED and "exists and is not an empty folder" in text and "synthetic tests only" not in text


def test_m1_expect_dropped_refused_on_a_runlog_holding_the_decision_line(tmp_path):
    rnd = build(tmp_path, lines=(PINNED_LINE, decision_line()))
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out, "--expect-dropped", D.PINNED_SNAPSHOT_LINE_SHA256)
    assert rc == D.EXIT_REFUSED and "REFUSED: --expect-dropped is for synthetic tests only: the RUNLOG read holds the " \
                                    "maintainer decision line" in text, text
    assert sorted(p.name for p in out.iterdir()) == ["runlog_asread.txt"]


# ---- M2: the deviation must read every OFF run blind, or no deviation reading is printed
def test_m2_refuses_when_the_deviation_run_still_reads_an_off_run_not_blind(tmp_path):
    """A PHASE F2-START at 18:30, before OFF s42's 18:56 launch, is never dropped: the deviation run still reads OFF
    s42 NOT BLIND, so its header ("as if the three OFF runs were blind") would be untrue: refused."""
    f2 = S.note(datetime(2026, 11, 1, 18, 30, tzinfo=UTC), "F2-START", f"{C.RUN_ID_PREFIX}80")
    rnd = build(tmp_path, lines=(PINNED_LINE, f2))
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    off42 = f"{C.ARM_PREFIX['off']}42"
    assert rc == D.EXIT_REFUSED, text
    assert f"REFUSED: the deviation run still reads OFF runs NOT BLIND: ['{off42}']" in text, text
    assert D.HEADER not in text and "PINNED READING" not in text and "DIFFERENCES" not in text
    assert json.loads((out / "deviation.json").read_text())["blindness"]["not_blind"] == [off42]


# ---- M3: PRESENCE p1 and D-16 copies of C1's listing (a LOST PHASE run with a re-fetch; the reviewer's S4)
def test_m3_lost_phase_run_with_refetch_presence_copies_of_c1_listing(tmp_path):
    rnd = build(tmp_path, phase_per_seed={60: dict(cut_at=40 * C.GRID, refetch={})})
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    p, d = one_test(out, "pinned"), one_test(out, "deviation")
    assert p["k"] == 20 and p["phase_missing"] == [True, False, False] and p["off_missing"] == [True] * 3
    assert d["k"] == 11 and d["phase_missing"] == [True, False, False] and d["off_missing"] == [False] * 3
    pinned, dev, diffs = sections(text)
    for sec in (pinned, dev):
        assert any(x.startswith(f"  pc s60 {C.RUN_ID_PREFIX}60: LOST") for x in sec), sec
    assert_only_blindness_differs(text, c1_listing=1, p1_copies=2)
    pc60 = f"runs.{C.RUN_ID_PREFIX}60"
    for k in (f"{pc60}.presence[0]", f"{pc60}.presence[1]", f"{pc60}.d16.presence[0]", f"{pc60}.d16.presence[1]"):
        assert any(x.startswith(f"    JSON {k}.p1.value.lines.") for x in diffs), (k, diffs)


# ---- M5: containment enforced
@pytest.mark.parametrize("where", ["round", "tools"])
def test_m5_refuses_when_a_file_under_round_or_tools_changes_during_the_runs(tmp_path, where):
    """A file is added under --round (or --tools) while the two scorer runs run (the test waits for the tool's
    scorer folder under --out, which it makes after its before-listing): the tool refuses and prints no reading."""
    rnd = build(tmp_path)
    t = _tools_copy(tmp_path)
    out = tmp_path / "out"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    cmd = [sys.executable, "-I", "-B", str(DEV), "--out", str(out), "--round", str(rnd.root), "--runlog",
           str(rnd.runlog), "--pins", str(rnd.pins_path), "--tools", str(t)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(cwd),
                         env={"PATH": "/usr/bin:/bin", "TMPDIR": str(cwd), "LANG": "C.UTF-8"})
    deadline = time.monotonic() + 120
    while not (out / "work").exists() and p.poll() is None and time.monotonic() < deadline:
        time.sleep(0.002)
    target = (rnd.root if where == "round" else t) / "written_during_the_runs.txt"
    target.write_text("x")
    so, se = p.communicate(timeout=600)
    assert se == "", se
    assert p.returncode == D.EXIT_REFUSED, so
    assert "CHANGED DURING THE RUNS" in so and "written_during_the_runs.txt" in so
    assert "REFUSED: files under" in so and "changed while the two scorer runs ran" in so
    assert D.HEADER not in so and "PINNED READING" not in so


# ---- R1: an OFF RELAUNCH is an OFF launch for the dropped set
def test_r1_an_off_relaunch_counts_as_the_last_off_launch(tmp_path):
    """A PHASE SNAPSHOT at 19:30 is after every OFF LAUNCH but before an OFF RELAUNCH at 20:00: it is before the last
    OFF launch, so it joins the dropped set, which is then not the pinned line alone: refused."""
    relaunch = S.note(datetime(2026, 11, 1, 20, 0, tzinfo=UTC), "RELAUNCH", f"{C.ARM_PREFIX['off']}60_b", seed=60)
    rnd = build(tmp_path, lines=(PINNED_LINE, snap("19:30"), relaunch))
    rc, text = run_dev(rnd, tmp_path / "out")
    assert rc == D.EXIT_REFUSED and "REFUSED: the dropped set" in text, text
    ref = next(x for x in text.splitlines() if x.startswith("REFUSED: the dropped set"))
    assert "'2026-11-01 17:44:00+00:00'" in ref and "'2026-11-01 19:30:00+00:00'" in ref, ref


# ---- R8: only PHASE SNAPSHOT lines are dropped; an OFF run's SNAPSHOT is not an opening and stays
def test_r8_an_off_runs_snapshot_is_kept(tmp_path):
    off_snap = snap("18:30", run=f"{C.ARM_PREFIX['off']}60")
    rnd = build(tmp_path, lines=(PINNED_LINE, off_snap))
    out = tmp_path / "out"
    rc, text = run_dev(rnd, out)
    assert rc == SP.EXIT_FINAL, text
    dev_lines = (out / "runlog_deviation.txt").read_text().splitlines()
    assert off_snap in dev_lines and PINNED_LINE not in dev_lines
    assert one_test(out, "deviation")["k"] == 1 and one_test(out, "pinned")["k"] == 20
    assert_only_blindness_differs(text)


# ---- R2, R3: the difference classifier (unit tests of the tool's own functions)
C1X = ('        value: {"launches": [4], "launch_check_stops": [], "actions": [], "lines": {"4": "LAUNCH", "7": '
       '"SNAPSHOT", "10": "TRIGGER-a"}}')
P1X = ('      p1: PASS  {"launches": [4], "launch_check_stops": [], "actions": [], "lines": {"7": "SNAPSHOT", "4": '
       '"LAUNCH"}} [C1 reads the RUN_ID\'s LAUNCH/RELAUNCH lines]')


def test_r3_output_line_kinds():
    k = D.kind_of_line_difference
    c1y, p1y = C1X.replace(', "7": "SNAPSHOT"', ""), P1X.replace('"7": "SNAPSHOT", ', "")
    assert k(C1X, c1y, {7}) == ("C1's listing of the run's note lines without dropped line 7 (C1's verdict reads only "
                                "launches, launch-check stops and actions)")
    assert k(P1X, p1y, {7}).startswith("C1's listing of the run's note lines without dropped line 7, as PRESENCE p1 "
                                       "prints it")
    assert k("OUTCOME BLINDNESS (x): a", "OUTCOME BLINDNESS (x): b", {7}).startswith("blindness")
    assert k("    THE ONE TEST (phase vs the concurrent switch-off runs): a",
             "    THE ONE TEST (phase vs the concurrent switch-off runs): b", {7}) == "THE ONE TEST"
    for x, y, dl in ((C1X, c1y, {8}),                                          # not a dropped line
                     (C1X, c1y.replace("TRIGGER-a", "TRIGGER-b"), {7}),        # another change too
                     (P1X, p1y.replace("p1: PASS", "p1: FAIL"), {7}),          # p1's status changed too
                     (C1X, C1X.replace('"4": "LAUNCH", ', ""), {4, 7}),        # a LAUNCH entry removed
                     (C1X, p1y, {7}),                                          # C1 line against a p1 line
                     ("  pc s60 x: CLASS A", "  pc s60 x: CLASS B", {7}),
                     ("RUN x: COMPLETE", None, {7}), (None, "RUN x: COMPLETE", {7}),
                     ("    s60: phase a blind False", "    s60: phase b blind True", {7})):
        assert k(x, y, dl) == "UNEXPECTED (not a blindness line)", (x, y)


def test_r2_json_value_kinds():
    k, A, pc = D.kind_of_json_difference, D.ABSENT, f"runs.{C.RUN_ID_PREFIX}60"
    assert k(f"{pc}.completeness.C1.value.lines.7", "SNAPSHOT", A, {7}) == "C1's listing of a dropped line"
    assert k(f"{pc}.presence[0].p1.value.lines.7", "SNAPSHOT", A, {7}) == \
        "C1's listing of a dropped line (PRESENCE p1 copy)"
    assert k(f"{pc}.d16.presence[1].p1.value.lines.7", "SNAPSHOT", A, {7}) == \
        "C1's listing of a dropped line (D-16 PRESENCE p1 copy)"
    assert k("blindness.not_blind[0]", "x", A, {7}) == "blindness"
    assert k("reading.one_test.k", 20, 1, {7}) == "THE ONE TEST"
    for key, x, y in ((f"{pc}.completeness.C1.value.lines.8", "SNAPSHOT", A),      # not a dropped line
                      (f"{pc}.completeness.C1.value.lines.7", "LAUNCH", A),        # not the SNAPSHOT entry
                      (f"{pc}.presence[0].p1.value.lines.7", "SNAPSHOT", "TRIGGER-a"),
                      (f"{pc}.d16.presence[0].p1.value.lines.7", A, "SNAPSHOT"),   # added, not removed
                      (f"{pc}.presence[0].p2.value.lines.7", "SNAPSHOT", A),       # not a copy of C1
                      (f"{pc}.presence[0].p1.ok", True, False), (f"{pc}.verdict", "COMPLETE", "LOST"),
                      ("(unreadable)", "ValueError: x", "")):
        assert k(key, x, y, {7}) == "UNEXPECTED", key
    assert json.dumps(A, default=str) == '"(absent)"'


def _main_in_process(rnd, out, capsys, *extra):
    rc = D.main(["--out", str(out), "--round", str(rnd.root), "--runlog", str(rnd.runlog), "--pins",
                 str(rnd.pins_path), *extra])
    return rc, capsys.readouterr().out


def test_r2_r3_unexpected_differences_are_flagged_and_counted(tmp_path, monkeypatch, capsys):
    """main() with the two scorer runs replaced by fixed outputs that differ in a class line and a class value."""
    rnd = build(tmp_path)

    def fake_run_both(tools, out, a, runlogs):
        res = dict(left_in_work_tmp=[])
        for name, cls in (("pinned", "A"), ("deviation", "B")):
            (out / f"{name}.json").write_text(json.dumps({"blindness": {"not_blind": []}, "runs": {"x": {"class": cls}}}))
            res[name] = dict(cmd=["score_pc.py", name], rc=0, stdout=f"STATUS: FINAL\n  pc s60 x: CLASS {cls}\n",
                             stderr="")
        return res

    monkeypatch.setattr(D, "run_both", fake_run_both)
    rc, text = _main_in_process(rnd, tmp_path / "out", capsys)
    assert rc == 0, text
    assert "    output line 2/2 [UNEXPECTED (not a blindness line)]:" in text
    assert '    JSON runs.x.class [UNEXPECTED]: pinned "A" | deviation "B"' in text
    assert ("1 output lines and 1 JSON values differ; UNEXPECTED (not blindness, THE ONE TEST or C1's listing of a "
            "dropped line): 2") in text


# ---- R5, R6, R7: the closure check, the decision line's check, the copy's self-check
def test_r5_the_scorers_load_closure_must_be_pinned(tmp_path, monkeypatch):
    """An APPROVED (a test copy) that pins every tool file but pc_common.py, which score_pc.py loads: refused."""
    t = _tools_copy(tmp_path)
    pins = json.loads((t / "pins_pc.json").read_text())
    pins["tool_sha256"].pop("pc_common.py", None)
    (t / "pins_pc.json").write_text(json.dumps(pins))
    ap = json.loads(D.APPROVED_FILE.read_bytes())
    assert "pc_common.py" in ap["pins"]["tools_pc"]
    ap["pins"]["tools_pc"].pop("pc_common.py")
    ap["pins"]["tools_pc"]["pins_pc.json"] = hashlib.sha256((t / "pins_pc.json").read_bytes()).hexdigest()
    fake = tmp_path / "APPROVED_test_copy"
    fake.write_text(json.dumps(ap))
    monkeypatch.setattr(D, "APPROVED_FILE", fake)
    monkeypatch.setattr(D, "APPROVED_SHA256", hashlib.sha256(fake.read_bytes()).hexdigest())
    with pytest.raises(D.Refused, match="score_pc.py loads pc_common.py, which is not a pinned tool file"):
        D.verify_tools(t)


def test_r6_the_decision_lines_check():
    dec = decision_line()
    r = D.decision_record(["2026-10-31 18:00 UTC  an unrelated line\n", dec + "\n"])
    assert r.startswith("RUNLOG line 2, ") and "sha256 equals the pinned decision line's" in r, r
    assert "before every F2-START / REFETCH-START line: yes (0 such lines)" in r, r
    bad = dec.replace("13:52 UTC", "13:53 UTC", 1)
    assert bad != dec and "sha256 IS NOT the pinned decision line's" in D.decision_record([bad + "\n"])
    f2 = f"2026-11-02 12:00 UTC  STANCE7 F2-START {C.RUN_ID_PREFIX}60\n"
    assert "before every F2-START / REFETCH-START line: NO (1 such lines)" in D.decision_record([f2, dec + "\n"])
    assert D.decision_record(["x\n"]).startswith("NOT FOUND")


def test_r7_the_copy_self_check_refuses_a_faulty_copy(tmp_path, monkeypatch, capsys):
    """A faulty copy builder (the line not blanked) is refused by the copy's self-check before any scorer run."""
    rnd = build(tmp_path)
    monkeypatch.setattr(D, "build_deviation_copy", lambda lines, drop_lines: "".join(lines))

    def no_run(*a, **k):
        raise AssertionError("the scorer ran on a faulty deviation copy")

    monkeypatch.setattr(D, "run_both", no_run)
    rc, text = _main_in_process(rnd, tmp_path / "out", capsys)
    assert rc == D.EXIT_REFUSED and "REFUSED: deviation copy line" in text, text


def test_json_differences_tell_an_absent_key_from_null(tmp_path):
    """A key present with null in one run's JSON and absent in the other's is a difference (the dry run's
    blindness.first_opening: a dict in the pinned run, null in the deviation run)."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"x": {"y": 1}, "z": None, "same": [1]}))
    b.write_text(json.dumps({"x": None, "same": [1]}))
    got = {k: (x, y) for k, x, y in D.json_differences(a, b)}
    assert got == {"x": (D.ABSENT, None), "x.y": (1, D.ABSENT), "z": (None, D.ABSENT)}, got
