"""Reviewer's own synthetic six-run roots for score_pc_dev.py (independent of builder A's tests: own TD values, own
launch times, own oracle).  Uses the pinned tools' synthetic-round builder (tools/tests/synth_pc.py, imported read
only) to write the round files; the expected p comes from an exact 3v3 permutation oracle written here.

    cd review/own && PY -I -B own_roots.py <workdir> [scenario ...]
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "/home/user/work/pc_study/tools/tests")
import synth_pc as S  # noqa: E402

C = S.C
UTC = timezone.utc
DEV = Path("/home/user/work/pc_study/amend2/tools/score_pc_dev.py")
RUNLOG = Path(C.RUNLOG)
PY = sys.executable

# RUNLOG line 902 read from RUNLOG itself (not from the builder's test constant); only its sha256 is printed
_RL = RUNLOG.read_bytes().decode("utf-8").splitlines()
L902 = _RL[901]
L902_SHA = hashlib.sha256(L902.encode("utf-8")).hexdigest()
PC = {s: f"{C.ARM_PREFIX['pc']}{s}" for s in C.SEEDS}
OFF = {s: f"{C.ARM_PREFIX['off']}{s}" for s in C.SEEDS}
PHASE_LAUNCH = {60: (3, 26), 42: (3, 27), 80: (3, 28)}       # RUNLOG lines 885-887
OFF_LAUNCH = {60: (18, 12), 80: (18, 20), 42: (18, 56)}       # RUNLOG lines 907, 911, 923


def oracle(phase, off):
    """Exact one-sided 3v3 permutation p (criteria lines 291-302): phase missing +inf, off missing -inf; a 3-sum is
    ordered by (#+inf - #-inf, exact rational sum of its finite values); k = splits with 3-sum <= the phase 3-sum."""
    vals = list(phase) + list(off)

    def key(xs):
        n = sum(1 for v in xs if v == math.inf) - sum(1 for v in xs if v == -math.inf)
        return (n, sum((Fraction(v) for v in xs if math.isfinite(v)), Fraction(0)))

    obs = key(vals[:3])
    k = sum(1 for idx in itertools.combinations(range(6), 3) if key([vals[i] for i in idx]) <= obs)
    return k


def t(h, m, day=1):
    return datetime(2026, 11, day, h, m, tzinfo=UTC)


def build(root: Path, phase_td: dict, off_td: dict, lines=(L902,), lost=(), off_launch=None, extra_notes=(),
          never=()):
    """phase_td / off_td: seed -> constant tail-3 command (TD = its absolute value); lost: (arm, seed) cut runs;
    never: (arm, seed) never launched."""
    rnd = S.Round(root / "r")
    for arm, td, launch in (("pc", phase_td, PHASE_LAUNCH), ("off", off_td, off_launch or OFF_LAUNCH)):
        for s in C.SEEDS:
            if (arm, s) in never:
                continue
            kw = dict(ledger=S.ledger_with_cmd(td[s]))
            if (arm, s) in lost:
                kw["cut_at"] = 40 * C.GRID
                kw["refetch"] = {}                          # the re-fetch the rule calls for (C2/C3 fail at F2)
            S.standard_run(rnd, s, arm=arm, launch=t(*launch[s]), **kw)
    rnd.notes.extend(lines)
    rnd.notes.extend(extra_notes)
    rnd.pins["stage_config_view_sha256"] = S.view_pin_of(S.Run(PC[60], S.T0, rnd.pins))
    rnd.pins["off_arm"]["stage_config_view_sha256"] = S.view_pin_of(S.Run(OFF[60], S.T0, rnd.pins))
    rnd.write_meta()
    S.make_trigger_stores(rnd.root, rnd.runlog)
    S.bind_panel_records(rnd.root)
    return rnd


def tree(folder: Path) -> dict:
    st = {}
    for p in sorted(Path(folder).rglob("*")):
        s = p.lstat()
        st[str(p)] = (p.is_dir(), s.st_size, s.st_mtime_ns,
                      hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "")
    return st


def run_dev(rnd, out: Path, *extra):
    cwd = out.parent / (out.name + "_cwd")
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-I", "-B", str(DEV), "--out", str(out), "--round", str(rnd.root), "--runlog", str(rnd.runlog),
           "--pins", str(rnd.pins_path), *map(str, extra)]
    watched = [rnd.root, Path("/home/user/work/pc_study/tools"), DEV.parent]
    before = {str(w): tree(w) for w in watched}
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd),
                       env={"PATH": "/usr/bin:/bin", "TMPDIR": str(cwd), "LANG": "C.UTF-8"})
    after = {str(w): tree(w) for w in watched}
    changed = {w: sorted(set(before[w].items()) ^ set(after[w].items())) for w in before}
    (out.parent / (out.name + ".stdout")).write_text(p.stdout)
    return p.returncode, p.stdout, p.stderr, {w: c for w, c in changed.items() if c}


def one_test(out, name):
    return json.loads((out / f"{name}.json").read_text())["reading"]["one_test"]


def summary_of_diffs(text):
    L = text.splitlines()
    d = next((i for i, x in enumerate(L) if x.startswith("DIFFERENCES between the pinned run")), None)
    if d is None:
        return []
    return [x for x in L[d + 1:] if x.startswith("    output line ") or x.startswith("    JSON ")
            or x.startswith("    ") and "differ;" in x]


SCEN = {
    # complete separation: every phase TD below every off TD
    "S1_separation": dict(phase_td={60: 0.11, 42: 0.23, 80: 0.17}, off_td={60: 0.41, 42: 0.36, 80: 0.52}),
    # one overlap and an exact cross-arm tie (dyadic values: TD exact): ties count against H1
    "S2_overlap_tie": dict(phase_td={60: 0.125, 42: 0.25, 80: 0.375}, off_td={60: 0.25, 42: 0.625, 80: 0.75}),
    # one OFF run LOST (cut): -infinity in both readings
    "S3_lost_off": dict(phase_td={60: 0.20, 42: 0.15, 80: 0.10}, off_td={60: 0.60, 42: 0.55, 80: 0.70},
                        lost=(("off", 42),)),
    # the run named by the dropped SNAPSHOT is itself LOST (+infinity): which output lines differ?
    "S4_lost_pc60": dict(phase_td={60: 0.05, 42: 0.20, 80: 0.30}, off_td={60: 0.50, 42: 0.60, 80: 0.70},
                         lost=(("pc", 60),)),
}


def expected(sc, nb_off):
    lost = sc.get("lost", ())
    ph = [math.inf if ("pc", s) in lost else abs(sc["phase_td"][s]) for s in C.SEEDS]
    of = [-math.inf if (("off", s) in lost or nb_off) else abs(sc["off_td"][s]) for s in C.SEEDS]
    return oracle(ph, of), ph, of


def main(argv):
    work = Path(argv[1]).resolve()
    names = argv[2:] or list(SCEN)
    print(f"RUNLOG line 902 sha256 {L902_SHA}; tool constant equal: "
          f"{L902_SHA == '0e30b26ed0a346dd0bdacd8bfd18f2ff9d5f6f58e492e1cc6e29d29763fbce12'}")
    bad = 0
    for name in names:
        sc = SCEN[name]
        root = work / name
        rnd = build(root, sc["phase_td"], sc["off_td"], lost=sc.get("lost", ()))
        out = root / "out"
        rc, text, err, changed = run_dev(rnd, out)
        print(f"== {name}: tool exit {rc}; stderr {len(err)} bytes; files changed outside --out: {changed or 'none'}")
        if rc not in (0, 10, 11):
            print(text[-3000:], err[-2000:])
            bad += 1
            continue
        for reading, nb in (("pinned", True), ("deviation", False)):
            ot = one_test(out, reading)
            k_or, ph, of = expected(sc, nb)
            got_ph = [math.inf if m else v for v, m in zip(ot["phase"], ot["phase_missing"])]
            got_of = [-math.inf if m else v for v, m in zip(ot["off"], ot["off_missing"])]
            td_ok = all((a == b) or (math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-12)
                        for a, b in zip(ph + of, got_ph + got_of))
            k_from_reported = oracle(got_ph, got_of)
            ok = ot["k"] == k_or == k_from_reported and td_ok and ot["p"] == f"{k_or}/20" \
                and ot["lower"] == (k_or == 1)
            bad += not ok
            print(f"   {reading:9s}: scorer k={ot['k']} p={ot['p']} text={ot['text']!r}; oracle k={k_or} "
                  f"(on reported values {k_from_reported}); TDs as intended {td_ok}; off_missing {ot['off_missing']} "
                  f"phase_missing {ot['phase_missing']} not_blind {ot['not_blind_entered_as_missing']} -> "
                  f"{'AGREE' if ok else 'DISAGREE'}")
        for x in summary_of_diffs(text):
            print("   " + x[:260])
    print(f"scenarios with a disagreement: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    sys.exit(main(sys.argv))
