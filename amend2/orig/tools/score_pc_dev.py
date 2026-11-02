"""PC study, AMENDMENT 2 (ST/amend2/AMENDMENT2.txt): the DEVIATION reading of THE ONE TEST, flagged, beside the pinned
reading.  Report only; never trains; changes nothing outside --out.

    PY -I -B score_pc_dev.py --out DIR [--round PCR] [--runlog P] [--pins P] [--approved P] [--panels-format F]
                             [--tools DIR] [--expect-dropped SHA256|none ...]

What it does, in order (each step refuses, exit 3, on any doubt):
  1. --out must be new or empty and outside --round, --tools, the pinned tools folder, the pinned round folder and the
     RUNLOG's folder; everything this tool and the two scorer runs write goes under it.
  2. The pinned scorer's bytes: <PCR>/APPROVED (the pinned round's, at its sha256 below) names the sha256 of every file
     of the pinned tools folder (pins.tools_pc); the pins file pins_pc.json must be at APPROVED's sha256, and every
     file named in APPROVED's tools_pc or in pins_pc.json's tool_sha256 must be at both sha256s; score_pc.py and every
     tool it loads (its _load("...") closure) must be among them.
  3. RUNLOG is read once (strict UTF-8); a byte-exact copy (runlog_asread.txt) is the pinned run's input.  The notes
     are parsed by the pinned r10round_common.parse_runlog (the scorer's own parser).  The DROPPED set is every
     STANCE7 SNAPSHOT note of a PHASE RUN_ID whose time is not later than the last switch-off (OFF) LAUNCH or RELAUNCH
     note (at or before it: the pinned rule reads an OFF launch in the opening's minute as NOT BLIND, so a same-minute
     SNAPSHOT is in the set too).  The sha256 of each dropped line's exact bytes (UTF-8, without its line feed) must
     equal the expected set: by default exactly PINNED_SNAPSHOT_LINE_SHA256 (RUNLOG line 902, the 17:44 UTC opening);
     any other set refuses.  --expect-dropped replaces the expected set (synthetic tests only; the output then says so).
  4. The deviation copy (runlog_deviation.txt) is the copy as read with each dropped line's content removed and its
     line feed kept, so every other line keeps its bytes and its line number (the scorer prints "RUNLOG line <i>" for
     every non-note line naming a RUN_ID; deleting the line would renumber those and add differences that are not the
     deviation).  It is re-read and checked: same line count, every other line byte-identical, the notes equal the
     original notes minus the dropped ones.
  5. The pinned score_pc.py, unchanged, runs twice as subprocesses (PY -I -B, cwd and TMPDIR inside --out): on
     runlog_asread.txt (--json pinned.json) and on runlog_deviation.txt (--json deviation.json); --round, --pins,
     --approved and --panels-format are passed to both alike.  score_pc.py writes only its --json file and stdout
     [measured: every write call of score_pc.py and the five tools it loads, NOTES.txt]; both --json paths are under
     --out.  The files under --round and --tools are listed (size, mtime) before and after and must be unchanged.
  6. Printed: the pinned run's whole output first (the round's reading of record); then the header "DEVIATION READING
     (flagged; not the pinned reading): ..."; the deviation run's OUTCOME BLINDNESS, class, READING / THE ONE TEST and
     STATUS lines; then every output line, and every JSON value, that differs between the two runs, each with its
     kind: a blindness line, THE ONE TEST, C1's printed listing of a run's note lines without a dropped line (C1's
     verdict reads only launches, launch-check stops and actions), or UNEXPECTED (counted on the last line).
Exit codes: the pinned run's (0 final, 10 incomplete, 11 provisional); 2 a scorer input error; 3 refused.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

VERSION = "score_pc_dev 1 (PC AMENDMENT 2, 2026-11-02: the DEVIATION reading of THE ONE TEST, flagged)"
EXIT_INPUT, EXIT_REFUSED = 2, 3
SCORER_EXITS = (0, 10, 11)

PINNED_TOOLS = Path("/home/user/work/pc_study/tools/pc")
PINNED_ROUND = Path("/home/user/work/pc_study/round")
APPROVED_FILE = PINNED_ROUND / "APPROVED"
#: sha256 of <PCR>/APPROVED (mode 0444, pinned 2026-11-01T03:07:17Z; its sha256 stands in RUNLOG line 880)
#: [measured 2026-11-02 by sha256sum].  Its pins.tools_pc names the sha256 of every pinned tool file.
APPROVED_SHA256 = "ce8865205a218541a742d7acf0cee479e87d63c7b53cbc52ae47a8892569873a"
#: sha256 of the exact bytes (UTF-8, without the line feed; 873 bytes) of RUNLOG line 902,
#: "2026-11-01 17:44 UTC  STANCE7 SNAPSHOT stance7_r10LTSBpc_s60 note="THE BLIND PERIOD opening ...": the one line the
#: deviation drops [measured 2026-11-02 from RUNLOG (950 lines, sha256 7a713156...) before any F2 file existed].
PINNED_SNAPSHOT_LINE_SHA256 = "0e30b26ed0a346dd0bdacd8bfd18f2ff9d5f6f58e492e1cc6e29d29763fbce12"
#: The maintainer's decision "Pin both readings": RUNLOG line 949 (its time and the sha256 of its exact bytes)
#: [measured 2026-11-02 from RUNLOG; no F2-START or REFETCH-START line stood in RUNLOG then].
DECISION_TIME = "2026-11-02 13:52 UTC"
DECISION_LINE_SHA256 = "f7b7278a60b3b2e9ecab7eaffae3eb010c862bf1b90fab9c8409a1b8bf3c79fc"
HEADER = ("DEVIATION READING (flagged; not the pinned reading): THE ONE TEST computed as if the three OFF runs were "
          f"blind; maintainer decision {DECISION_TIME}, recorded before F2; the pinned reading above stands")
LOAD_RE = re.compile(r"""_load\(\s*["']([A-Za-z0-9_]+)["']\s*\)""")


class Refused(Exception):
    """A check failed: nothing is read as a deviation reading."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def _inside(p: Path, folder: Path) -> bool:
    p, folder = Path(p).resolve(), Path(folder).resolve()
    return p == folder or folder in p.parents


# ------------------------------------------------------------------------------------------------ 1. the out folder
def check_out(out: Path, forbidden: list) -> None:
    for f in forbidden:
        if f is not None and _inside(out, f):
            raise Refused(f"--out {out} is inside {f}: the deviation run writes only into its own folder")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refused(f"--out {out} exists and is not an empty folder")


# ------------------------------------------------------------------------------------------------ 2. pinned bytes
def verify_tools(tools: Path) -> dict:
    """The pinned scorer and every tool file pinned beside it at APPROVED's and pins_pc.json's sha256."""
    try:
        ab = APPROVED_FILE.read_bytes()
    except OSError as exc:
        raise Refused(f"cannot read {APPROVED_FILE}: {exc}") from exc
    if sha256_bytes(ab) != APPROVED_SHA256:
        raise Refused(f"{APPROVED_FILE} sha256 {sha256_bytes(ab)} is not the pinned {APPROVED_SHA256}")
    tp = json.loads(ab.decode("utf-8"))["pins"]["tools_pc"]
    pins_file = tools / "pins_pc.json"
    if not pins_file.is_file() or sha256_file(pins_file) != tp.get("pins_pc.json"):
        raise Refused(f"{pins_file}: sha256 is not APPROVED's {tp.get('pins_pc.json')}")
    th = json.loads(pins_file.read_text(encoding="utf-8"))["tool_sha256"]
    got = {}
    for name in sorted(set(tp) | set(th)):
        f = tools / name
        if not f.is_file():
            raise Refused(f"pinned tool file {f} is missing")
        h = sha256_file(f)
        for src, pins in (("APPROVED pins.tools_pc", tp), ("pins_pc.json tool_sha256", th)):
            if name in pins and pins[name] != h:
                raise Refused(f"{f} sha256 {h} is not its pin in {src} ({pins[name]})")
        got[name] = h
    closure, todo = [], ["score_pc"]
    while todo:
        m = todo.pop()
        if m in closure:
            continue
        if f"{m}.py" not in got:
            raise Refused(f"score_pc.py loads {m}.py, which is not a pinned tool file")
        closure.append(m)
        todo += LOAD_RE.findall((tools / f"{m}.py").read_text(encoding="utf-8"))
    return dict(hashes=got, closure=sorted(closure))


def load_common(tools: Path):
    """The pinned r10round_common (bytes verified above): the scorer's own RUNLOG parser."""
    spec = importlib.util.spec_from_file_location("r10round_common_pinned_for_dev", tools / "r10round_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ 3-4. RUNLOG copies
def read_lines(path: Path) -> tuple:
    data = Path(path).read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refused(f"RUNLOG {path} is not UTF-8 ({exc}): a line's exact bytes cannot be identified") from exc
    lines = text.splitlines(keepends=True)
    if "".join(lines) != text:
        raise Refused("RUNLOG lines do not rejoin to its text")
    return data, lines


def line_body(line: str) -> str:
    return line.splitlines()[0] if line.splitlines() else ""


def dropped_set(C, notes) -> tuple:
    """The SNAPSHOT notes of a PHASE RUN_ID at or before the last OFF LAUNCH / RELAUNCH note (time order)."""
    off = [n["time"] for n in notes if n["kind"] in ("LAUNCH", "RELAUNCH") and C.arm_of(n["run"]) == "off"]
    last_off = max(off) if off else None
    drop = [n for n in notes if n["kind"] == "SNAPSHOT" and C.arm_of(n["run"]) == "pc"
            and last_off is not None and n["time"] <= last_off]
    return drop, last_off


def build_deviation_copy(lines: list, drop_lines: set) -> str:
    out = []
    for i, ln in enumerate(lines, 1):
        out.append(ln[len(line_body(ln)):] if i in drop_lines else ln)   # content removed, line feed kept
    return "".join(out)


def check_deviation_copy(C, asread: Path, dev: Path, drop_lines: set, notes) -> None:
    _d, a = read_lines(asread)
    _d, b = read_lines(dev)
    if len(a) != len(b):
        raise Refused(f"deviation copy has {len(b)} lines, RUNLOG {len(a)}")
    for i, (x, y) in enumerate(zip(a, b), 1):
        if i in drop_lines:
            if line_body(y) != "" or y != x[len(line_body(x)):]:
                raise Refused(f"deviation copy line {i} is not the dropped line's line feed alone")
        elif x != y:
            raise Refused(f"deviation copy line {i} differs from RUNLOG")
    dn, _o = C.parse_runlog(dev, C.RUN_IDS)
    key = [(n["line"], n["kind"], n["run"], n["time"], n["text"]) for n in notes if n["line"] not in drop_lines]
    if [(n["line"], n["kind"], n["run"], n["time"], n["text"]) for n in dn] != key:
        raise Refused("the deviation copy's notes are not RUNLOG's notes minus the dropped lines")
    rest, _l = dropped_set(C, dn)
    if rest:
        raise Refused(f"the deviation copy still holds an early PHASE SNAPSHOT line: {[n['line'] for n in rest]}")


def decision_record(lines: list) -> str:
    for i, ln in enumerate(lines, 1):
        body = line_body(ln)
        if "STANCE7 NOTE MAINTAINER DECISION" in body and '"Pin both readings"' in body:
            f2 = [j for j, x in enumerate(lines, 1) if re.search(r" UTC\s+STANCE7\s+(F2-START|REFETCH-START)\s", x)]
            before = all(j > i for j in f2)
            ok = sha256_bytes(body.encode("utf-8")) == DECISION_LINE_SHA256
            return (f"RUNLOG line {i}, {body[:20]}; sha256 {'equals' if ok else 'IS NOT'} the pinned decision line's; "
                    f"before every F2-START / REFETCH-START line: {'yes' if before else 'NO'} ({len(f2)} such lines)")
    return "NOT FOUND in the RUNLOG read (a test RUNLOG, or the line changed)"


# ------------------------------------------------------------------------------------------------ 5. the two runs
def tree_state(folder: Path) -> dict:
    st = {}
    for dp, _dn, fn in os.walk(folder):
        for f in fn:
            p = Path(dp) / f
            try:
                s = p.lstat()
            except OSError:
                continue
            st[str(p.relative_to(folder))] = (s.st_size, s.st_mtime_ns)
    return st


def run_both(tools: Path, out: Path, a, runlogs: dict) -> dict:
    work, tmp = out / "work", out / "tmp"
    work.mkdir()
    tmp.mkdir()
    env = {"PATH": "/usr/bin:/bin", "TMPDIR": str(tmp), "HOME": str(work), "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    procs = {}
    for name, rl in runlogs.items():
        cmd = [sys.executable, "-I", "-B", str(tools / "score_pc.py"), "--round", str(a.round), "--runlog", str(rl),
               "--json", str(out / f"{name}.json")]
        for opt, v in (("--pins", a.pins), ("--approved", a.approved), ("--panels-format", a.panels_format)):
            if v is not None:
                cmd += [opt, str(v)]
        fo, fe = open(out / f"{name}.out", "wb"), open(out / f"{name}.err", "wb")
        procs[name] = (cmd, subprocess.Popen(cmd, cwd=str(work), env=env, stdout=fo, stderr=fe), fo, fe)
    res = {}
    for name, (cmd, p, fo, fe) in procs.items():
        rc = p.wait()
        fo.close()
        fe.close()
        res[name] = dict(cmd=cmd, rc=rc, stdout=(out / f"{name}.out").read_text(encoding="utf-8", errors="replace"),
                         stderr=(out / f"{name}.err").read_text(encoding="utf-8", errors="replace"))
    left = [str(p.relative_to(out)) for d in (work, tmp) for p in d.rglob("*")]
    for d in (tmp, work):
        if not any(d.iterdir()):
            d.rmdir()
    res["left_in_work_tmp"] = left
    return res


# ------------------------------------------------------------------------------------------------ 6. printing
def deviation_lines(text: str) -> list:
    """The deviation run's OUTCOME BLINDNESS block, each run's class line, the READING block (THE ONE TEST in it) and
    STATUS with its PROVISIONAL items, as the scorer printed them."""
    out, in_bl, in_rd = [], False, False
    for ln in text.splitlines():
        if ln.startswith("OUTCOME BLINDNESS"):
            in_bl = True
            out.append(ln)
            continue
        if in_bl and re.match(r"    s\d+: phase ", ln):
            out.append(ln)
            continue
        in_bl = False
        if re.match(r"  (pc|off) s\d+ ", ln):
            out.append(ln)
            continue
        if ln.startswith("READING:"):
            in_rd = True
        if ln.startswith("STATUS:"):
            in_rd = False
            out.append(ln)
            continue
        if in_rd or ln.startswith("    PROVISIONAL:"):
            out.append(ln)
    return out


def flatten(x, prefix="") -> dict:
    if isinstance(x, dict):
        d = {}
        for k, v in x.items():
            d.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return d
    if isinstance(x, list):
        d = {}
        for i, v in enumerate(x):
            d.update(flatten(v, f"{prefix}[{i}]"))
        return d if x else {prefix: []}
    return {prefix: x}


def differences(pin_text: str, dev_text: str) -> list:
    a, b = pin_text.splitlines(), dev_text.splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for k in range(max(i2 - i1, j2 - j1)):
            pa = a[i1 + k] if i1 + k < i2 else None
            pb = b[j1 + k] if j1 + k < j2 else None
            out.append((i1 + k + 1 if pa is not None else None, pa, j1 + k + 1 if pb is not None else None, pb))
    return out


def kind_of_line_difference(x, y, dropped_lines) -> str:
    """Each differing output line's kind: the blindness lines and THE ONE TEST line, C1's printed listing of a run's
    note lines that held a dropped line (C1's verdict reads only launches, launch-check stops and actions), else
    UNEXPECTED."""
    both = [s for s in (x, y) if s is not None]
    if x is not None and y is not None:
        if all(s.startswith("OUTCOME BLINDNESS (") for s in both):
            return "blindness: the first opening and the NOT BLIND runs"
        if all(re.match(r"    s\d+: phase ", s) for s in both) and x.split(" blind ")[0] == y.split(" blind ")[0]:
            return "blindness: the seed's OFF run blind or not"
        if all(s.startswith("    THE ONE TEST (phase vs the concurrent switch-off runs): ") for s in both):
            return "THE ONE TEST"
        if all(re.match(r"\s+value: \{\"launches\": ", s) for s in both):
            for ln in dropped_lines:
                for a in (f', "{ln}": "SNAPSHOT"', f'"{ln}": "SNAPSHOT", '):
                    if a in x and x.replace(a, "", 1) == y:
                        return (f"C1's listing of the run's note lines without dropped line {ln} (C1's verdict reads "
                                "only launches, launch-check stops and actions)")
    return "UNEXPECTED (not a blindness line)"


def kind_of_json_difference(k, dropped_lines) -> str:
    if k.startswith("blindness."):
        return "blindness"
    if k.startswith("reading.one_test."):
        return "THE ONE TEST"
    m = re.match(r"runs\.[^.]+\.completeness\.C1\.value\.lines\.(\d+)$", k)
    if m and int(m.group(1)) in dropped_lines:
        return "C1's listing of a dropped line"
    return "UNEXPECTED"


def json_differences(pj: Path, dj: Path) -> list:
    try:
        a = flatten(json.loads(pj.read_text(encoding="utf-8")))
        b = flatten(json.loads(dj.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        return [("(unreadable)", f"{type(exc).__name__}: {exc}", "")]
    return [(k, a.get(k, "(absent)"), b.get(k, "(absent)")) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="a new or empty folder; every file written goes under it")
    ap.add_argument("--round", default=str(PINNED_ROUND))
    ap.add_argument("--runlog", default=None, help="default: the pinned tools' RUNLOG path")
    ap.add_argument("--pins", default=None)
    ap.add_argument("--approved", default=None)
    ap.add_argument("--panels-format", default=None)
    ap.add_argument("--tools", default=str(PINNED_TOOLS), help="the pinned tools folder (its bytes are verified)")
    ap.add_argument("--expect-dropped", action="append", default=None,
                    help="TESTS ONLY: the expected dropped lines' sha256 (repeat), or 'none'; default the pinned line")
    a = ap.parse_args(argv)
    out, tools = Path(a.out).resolve(), Path(a.tools).resolve()
    L = [VERSION]
    try:
        verified = verify_tools(tools)
        C = load_common(tools)
        runlog = Path(a.runlog or C.RUNLOG).resolve()
        a.round = Path(a.round).resolve()
        check_out(out, [a.round, tools, PINNED_TOOLS, PINNED_ROUND, runlog.parent])
        out.mkdir(parents=True, exist_ok=True)
        L.append(f"PINNED SCORER: {tools / 'score_pc.py'} sha256 {verified['hashes']['score_pc.py']}; it loads "
                 f"{verified['closure']}; all {len(verified['hashes'])} pinned tool files at APPROVED's and "
                 f"pins_pc.json's sha256 (APPROVED {APPROVED_SHA256[:16]}...)")
        data, lines = read_lines(runlog)
        asread = out / "runlog_asread.txt"
        asread.write_bytes(data)
        if sha256_file(asread) != sha256_bytes(data):
            raise Refused("the RUNLOG copy as read is not byte-identical")
        L.append(f"RUNLOG {runlog}: {len(lines)} lines, sha256 {sha256_bytes(data)}; read once, copied byte for byte "
                 f"to {asread}")
        notes, _o = C.parse_runlog(asread, C.RUN_IDS)
        drop, last_off = dropped_set(C, notes)
        got = sorted(sha256_bytes(line_body(lines[n["line"] - 1]).encode("utf-8")) for n in drop)
        if a.expect_dropped is None:
            want, how = [PINNED_SNAPSHOT_LINE_SHA256], "the pinned line (default)"
        else:
            want = sorted(x for x in a.expect_dropped if x != "none")
            how = "NON-DEFAULT, given by --expect-dropped (synthetic tests only; not the round's deviation reading)"
        if got != sorted(want):
            raise Refused(f"the dropped set {[(n['line'], n['kind'], n['run'], str(n['time'])) for n in drop]} "
                          f"(sha256 {got}) is not the expected set {sorted(want)} [{how}]")
        drop_lines = {n["line"] for n in drop}
        dev = out / "runlog_deviation.txt"
        dev.write_bytes(build_deviation_copy(lines, drop_lines).encode("utf-8"))
        check_deviation_copy(C, asread, dev, drop_lines, notes)
        before = {str(d): tree_state(d) for d in (a.round, tools)}
        res = run_both(tools, out, a, {"pinned": asread, "deviation": dev})
        after = {str(d): tree_state(d) for d in (a.round, tools)}
    except Refused as exc:
        L.append(f"REFUSED: {exc}")
        print("\n".join(L))
        return EXIT_REFUSED
    pin, dv = res["pinned"], res["deviation"]
    for name, r in (("pinned", pin), ("deviation", dv)):
        if r["rc"] not in SCORER_EXITS:
            L.append(f"SCORER ({name} run) exit {r['rc']}: {r['stdout'][-2000:]} {r['stderr'][-2000:]}")
            print("\n".join(L))
            return EXIT_INPUT
    L.append("=" * 24 + " PINNED READING (the round's reading of record, printed first): score_pc.py unchanged on "
             "RUNLOG as read " + "=" * 24)
    L.append("    command: " + " ".join(pin["cmd"]))
    L += pin["stdout"].rstrip("\n").splitlines()
    L.append("=" * 24 + " END OF THE PINNED READING " + "=" * 24)
    L.append("")
    L.append(HEADER)
    L.append(f"    maintainer decision line: {decision_record(lines)}")
    L.append(f"    dropped (content removed, line feed kept; every other line byte-identical): "
             f"{[(n['line'], n['kind'], n['run'], n['time'].strftime('%Y-%m-%d %H:%M UTC')) for n in drop]} at or "
             f"before the last OFF launch {last_off and last_off.strftime('%Y-%m-%d %H:%M UTC')}; dropped-line sha256 "
             f"{got}; expected: {how}")
    L.append(f"    deviation RUNLOG copy: {dev} sha256 {sha256_file(dev)}")
    L.append("    command: " + " ".join(dv["cmd"]))
    sel = deviation_lines(dv["stdout"])
    if not any("THE ONE TEST (phase vs" in x for x in sel):
        sel.append("    THE ONE TEST (deviation): not computed by the scorer in this run (READING incomplete; the open "
                   "items are listed above)")
    L += ["  [deviation] " + x for x in sel]
    L.append("DIFFERENCES between the pinned run and the deviation run (every output line that differs; then every "
             "JSON value):")
    diffs = differences(pin["stdout"], dv["stdout"])
    unexpected = 0
    for i, x, j, y in diffs:
        kind = kind_of_line_difference(x, y, drop_lines)
        unexpected += kind.startswith("UNEXPECTED")
        L.append(f"    output line {i}/{j} [{kind}]:")
        L.append(f"        pinned:    {x if x is not None else '(no line)'}")
        L.append(f"        deviation: {y if y is not None else '(no line)'}")
    if not diffs:
        L.append("    (no output line differs: the two outputs are byte-identical)" if pin["stdout"] == dv["stdout"]
                 else "    (no output line differs)")
    jd = json_differences(out / "pinned.json", out / "deviation.json")
    for k, x, y in jd:
        kind = kind_of_json_difference(k, drop_lines)
        unexpected += kind.startswith("UNEXPECTED")
        L.append(f"    JSON {k} [{kind}]: pinned {json.dumps(x, default=str)} | deviation "
                 f"{json.dumps(y, default=str)}")
    if not jd:
        L.append("    (no JSON value differs)")
    L.append(f"    {len(diffs)} output lines and {len(jd)} JSON values differ; "
             + (f"UNEXPECTED (not blindness, THE ONE TEST or C1's listing of a dropped line): {unexpected}"
                if unexpected else "every one is a blindness line, THE ONE TEST or C1's listing of a dropped line"))
    L.append(f"    exit codes: pinned {pin['rc']}, deviation {dv['rc']}; stderr bytes: pinned {len(pin['stderr'])}, "
             f"deviation {len(dv['stderr'])}")
    changed = {d: sorted(set(before[d].items()) ^ set(after[d].items())) for d in before}
    for d, ch in changed.items():
        L.append(f"CONTAINMENT: {d}: " + ("unchanged (every file's size and mtime as before)" if not ch
                                          else f"CHANGED DURING THE RUNS: {ch[:20]}"))
    L.append("CONTAINMENT: the scorer's cwd and TMPDIR (under --out): "
             + ("left empty, removed" if not res["left_in_work_tmp"] else f"left {res['left_in_work_tmp'][:20]}"))
    L.append(f"OUTPUT FILES (all under {out}): runlog_asread.txt, runlog_deviation.txt, pinned.out/.err/.json, "
             "deviation.out/.err/.json")
    print("\n".join(L))
    return pin["rc"]


if __name__ == "__main__":
    sys.exit(main())
