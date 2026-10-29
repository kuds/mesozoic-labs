"""Lane R tests: score_ecs.py (build list v2 item 9, the scorer's part).  Arithmetic and synthetic records only."""
import importlib.util
import itertools
import json
import re
from fractions import Fraction
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("test_read_score_fixtures",
                                               Path(__file__).resolve().parent / "test_read_score_fixtures.py")
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
E, R, S = F.E, F.R, F.S
TE_OUT = Path(E.ECS) / "v1" / "table_e_ecs.out"
TE_OUT_SHA256 = "07d32255b904a62985996599ee57ea9ff5ea039322279bdb58ce41bbee4bb542"      # v1/MANIFEST.sha256
CHK_OUT = Path(E.ECS) / "v2" / "check_v2.out"
CHK_OUT_SHA256 = "a8c05b9e09e85e49b3ea2aa0875b3979f3e05c47162cdda2cb34e17d0cee6448"     # v2/MANIFEST.sha256
SYM = {"B LOWER": "L", "B HIGHER": "H", "NO DIFFERENCE DETECTED": "."}


def te_lines():
    assert E.sha256_file(TE_OUT) == TE_OUT_SHA256
    return TE_OUT.read_text(encoding="utf-8").splitlines()


def te_section(lines, head):
    i = next(k for k, x in enumerate(lines) if x.startswith(head))
    out = []
    for x in lines[i + 1:]:
        if re.match(r"^\d\. ", x):
            break
        out.append(x)
    return out


def grid_of(sec):
    g = {}
    for x in sec:
        m = re.match(r"^   a=(\d)   (.*)$", x)
        if m:
            g[int(m.group(1))] = m.group(2).split()
    assert sorted(g) == list(range(7)) and all(len(v) == 7 for v in g.values())
    return g


# ================================================================================================ batteries
def battery_te(Sm):
    """Every cell of TE sections 1-2 (the grids and every printed p-value) and section 3's examples."""
    lines = te_lines()
    s1, s2, s3 = (te_section(lines, h) for h in ("1. THE SAME", "2. ONE START", "3. MIXED"))
    g1, g2 = grid_of(s1), grid_of(s2)
    n_checked = 0
    for a in range(7):
        for b in range(7):
            assert SYM[Sm.reading_of(((a, 6, b, 6), (a, 6, b, 6)))[0]] == g1[a][b], ("TE 1", a, b)
            assert SYM[Sm.reading_of(((a, 6, b, 6), (6, 6, 6, 6)))[0]] == g2[a][b], ("TE 2", a, b)
            n_checked += 2
    for x in s1:
        m = re.match(r"^\s+(\d)/6 vs (\d)/6 at both starts: (.+?)\s+P_UP ([\d.]+)  P_DOWN ([\d.]+)$", x)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            r, up, down = Sm.reading_of(((a, 6, b, 6), (a, 6, b, 6)))
            assert (r, "%.4f" % float(up), "%.4f" % float(down)) == (m.group(3), m.group(4), m.group(5)), x
            n_checked += 1
    for x in s2:
        m = re.match(r"^\s+(\d)/6 vs (\d)/6 at one start, 0/6 vs 0/6 at the other: (.+?)\s+P_UP ([\d.]+)$", x)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            r, up, _down = Sm.reading_of(((a, 6, b, 6), (0, 6, 0, 6)))
            assert (r, "%.4f" % float(up)) == (m.group(3), m.group(4)), x
            n_checked += 1
    for x in s3:
        m = re.match(r"^\s+(\d)/(\d) vs (\d)/(\d) ; (\d)/(\d) vs (\d)/(\d): (.+?)\s+P_UP ([\d.]+)  P_DOWN ([\d.]+)$", x)
        m6 = re.match(r"^\s+(\d)/6 vs (\d)/6 ; (\d)/6 vs (\d)/6: (.+?)\s+P_UP ([\d.]+)  P_DOWN ([\d.]+)$", x)
        if m:
            v = [int(m.group(k)) for k in range(1, 9)]
            cells = ((v[0], v[1], v[2], v[3]), (v[4], v[5], v[6], v[7]))
            want = (m.group(9), m.group(10), m.group(11))
        elif m6:
            v = [int(m6.group(k)) for k in range(1, 5)]
            cells = ((v[0], 6, v[1], 6), (v[2], 6, v[3], 6))
            want = (m6.group(5), m6.group(6), m6.group(7))
        else:
            continue
        r, up, down = Sm.reading_of(cells)
        assert (r, "%.4f" % float(up), "%.4f" % float(down)) == want, x
        n_checked += 1
    assert n_checked == 98 + 18 + 6 + 11, n_checked
    return True


def classes_of(spec):
    """Class dicts as score_ecs.classify returns them.  spec {start: {arm: (pos, neg, lost, censored)}}."""
    out = []
    for st in ("s42", "s80"):
        for arm in ("A", "B"):
            pos, neg, lost, cen = spec[st][arm]
            assert pos + neg + lost + cen == 6
            kinds = ["P"] * pos + ["N"] * neg + ["L"] * lost + ["C"] * cen
            for seed, k in zip(E.SEEDS[(st, arm)], kinds, strict=True):
                c = dict(name=E.run_name(st, arm, seed), start=st, arm=arm, seed=seed, T=E.window(st)[1], read_ok=True,
                         report=None, cause=None)
                if k == "P":
                    c.update(cls="COUNTED", outcome="POSITIVE")
                elif k == "N":
                    c.update(cls="COUNTED", outcome="NEGATIVE")
                elif k == "L":
                    c.update(cls="LOST", outcome=None, cause="crashes")
                else:
                    c.update(cls="STOPPED", outcome="CENSORED", cause="collapse")
                out.append(c)
    return out


def chk8_examples():
    assert E.sha256_file(CHK_OUT) == CHK_OUT_SHA256
    rx = re.compile(r"^\s+(\d)/(\d)(\(-1\))? vs (\d)/(\d)(\(-1\))? ; (\d)/(\d)(\(-1\))? vs (\d)/(\d)(\(-1\))?: "
                    r"as counted P_UP ([\d.]+) \(([L.])\); filled P_UP ([\d.]+) \(([L.])\)$")
    ex = [m for m in (rx.match(x) for x in CHK_OUT.read_text(encoding="utf-8").splitlines()) if m]
    assert len(ex) == 5
    return ex


def battery_chk8(Sm, censored=False):
    """CHK 8, worked from classes: a '(-1)' continuation is LOST (or censored), filled against B LOWER."""
    for m in chk8_examples():
        g = m.groups()
        spec = {}
        for st, off in (("s42", 0), ("s80", 6)):
            arms = {}
            for arm, o in (("A", off), ("B", off + 3)):
                pos, n, miss = int(g[o]), int(g[o + 1]), 1 if g[o + 2] else 0
                arms[arm] = (pos, n - pos, 0 if censored else miss, miss if censored else 0)
            spec[st] = arms
        tabs = Sm.tables(classes_of(spec))
        d = Sm.decide(tabs)
        assert "%.4f" % float(d["counted_p_up"]) == g[12] and "%.4f" % float(d["p_up"]) == g[14], m.group(0)
        assert ("L" if d["counted_p_up"] <= Sm.SIDE else ".") == g[13]
        assert ("L" if d["p_up"] <= Sm.SIDE else ".") == g[15]
        assert all(tabs[s]["up"][1] == 6 and tabs[s]["up"][3] == 6 for s in tabs)
    return True


def battery_fill(Sm):
    """THE FILL: against B LOWER arm A's missing are NEGATIVE and arm B's POSITIVE; against B HIGHER the reverse."""
    spec = {"s42": {"A": (3, 1, 1, 1), "B": (2, 2, 2, 0)}, "s80": {"A": (6, 0, 0, 0), "B": (1, 4, 0, 1)}}
    tabs = Sm.tables(classes_of(spec))
    assert tabs["s42"]["counted"] == (3, 4, 2, 4) and tabs["s42"]["missing"] == {"A": 2, "B": 2}
    assert tabs["s42"]["up"] == (3, 6, 4, 6) and tabs["s42"]["down"] == (5, 6, 2, 6)
    assert tabs["s80"]["counted"] == (6, 6, 1, 5) and tabs["s80"]["up"] == (6, 6, 2, 6)
    assert tabs["s80"]["down"] == (6, 6, 1, 6)
    d = Sm.decide(tabs)
    assert d["p_up"] == Sm.pvals(((3, 6, 4, 6), (6, 6, 2, 6)))[0]
    assert d["p_down"] == Sm.pvals(((5, 6, 2, 6), (6, 6, 1, 6)))[1]
    assert d["n_missing"] == 5
    # P_DOWN on its own table: a B-HIGHER case with a missing arm-B continuation
    spec = {"s42": {"A": (0, 6, 0, 0), "B": (5, 0, 1, 0)}, "s80": {"A": (1, 5, 0, 0), "B": (4, 2, 0, 0)}}
    d = Sm.decide(Sm.tables(classes_of(spec)))
    assert d["p_down"] == Sm.pvals(((0, 6, 5, 6), (1, 6, 4, 6)))[1] and d["line"] == 2
    assert d["p_up"] == Sm.pvals(((0, 6, 6, 6), (1, 6, 4, 6)))[0]
    return True


def battery_lines(Sm):
    def run(spec):
        return Sm.decide(Sm.tables(classes_of(spec)))

    def full(a1, b1, a2, b2):
        return {"s42": {"A": (a1, 6 - a1, 0, 0), "B": (b1, 6 - b1, 0, 0)},
                "s80": {"A": (a2, 6 - a2, 0, 0), "B": (b2, 6 - b2, 0, 0)}}
    d = run(full(6, 3, 6, 3))
    assert d["line"] == 1 and d["text"] == Sm.LINE_LOWER and d["p_up"] == Fraction(1, 121) and not d["additions"]
    d = run(full(0, 4, 1, 4))
    assert d["line"] == 2 and d["text"] == Sm.LINE_HIGHER
    d = run(full(6, 4, 6, 4))
    assert d["line"] == 3 and d["text"] == Sm.LINE_NONE
    assert d["notes"]["s42"] == ["ARM A AT 6 OF 6: no room for B HIGHER at this start"]
    d = run(full(2, 2, 1, 1))
    assert d["line"] == 3 and d["text"] == Sm.LINE_NOT_INFORMATIVE
    assert d["notes"]["s42"] == ["LOW ARM-A RATE (2 of 6): little power for B LOWER at this start"]
    d = run(full(3, 3, 2, 2))
    assert d["text"] == Sm.LINE_NONE and d["notes"]["s42"] == [], "3 of 6 is half, not fewer than half"
    d = run(full(5, 2, 4, 2))
    assert d["line"] == 3 and d["text"] == Sm.LINE_NONE
    # holds only as counted: s42 A 4 of 5 (one LOST) vs B 0 of 6; s80 0/6 vs 0/6
    d = run({"s42": {"A": (4, 1, 1, 0), "B": (0, 6, 0, 0)}, "s80": {"A": (0, 6, 0, 0), "B": (0, 6, 0, 0)}})
    assert d["line"] == 3 and d["counted_p_up"] <= Sm.SIDE < d["p_up"]
    assert d["additions"] == ["(B LOWER holds only without the 1 LOST or censored continuations)"]
    assert d["text"] == Sm.LINE_NONE, "s42 (4 of 5) carries no LOW note"
    d = run({"s42": {"A": (0, 6, 0, 0), "B": (4, 1, 0, 1)}, "s80": {"A": (0, 6, 0, 0), "B": (0, 6, 0, 0)}})
    assert d["line"] == 3 and d["text"] == Sm.LINE_NOT_INFORMATIVE
    assert d["additions"] == ["(B HIGHER holds only without the 1 LOST or censored continuations)"]
    d = run({"s42": {"A": (0, 0, 4, 2), "B": (0, 6, 0, 0)}, "s80": {"A": (5, 1, 0, 0), "B": (1, 5, 0, 0)}})
    assert d["notes"]["s42"] == ["NO ARM-A CONTINUATION COUNTED"] and d["n_missing"] == 6
    # review RR-4: a NO ARM-A note at one start and a LOW ARM-A RATE at the other read NOT INFORMATIVE
    d = run({"s42": {"A": (0, 0, 6, 0), "B": (3, 3, 0, 0)}, "s80": {"A": (2, 4, 0, 0), "B": (2, 4, 0, 0)}})
    assert d["line"] == 3 and d["notes"]["s42"] == ["NO ARM-A CONTINUATION COUNTED"]
    assert d["notes"]["s80"] == ["LOW ARM-A RATE (2 of 6): little power for B LOWER at this start"]
    assert d["text"] == Sm.LINE_NOT_INFORMATIVE and "or none was counted" in d["text"]
    return True


def battery_brute(Sm):
    """P_UP and P_DOWN against a brute-force relabelling: every way of choosing which nA of a start's nA + nB
    continuations are arm A, the outcomes held fixed, independently per start."""
    def counts(nA, nB, m):
        c = {}
        for lab in itertools.combinations(range(nA + nB), nA):
            x = sum(1 for i in lab if i < m)                  # positions 0..m-1 are the POSITIVE ones
            c[x] = c.get(x, 0) + 1
        return c
    cache = {}

    def brute(cells):
        dists = []
        for a, nA, b, nB in cells:
            key = (nA, nB, a + b)
            if key not in cache:
                cache[key] = counts(*key)
            dists.append(cache[key])
        x = sum(c[0] for c in cells)
        tot = up = down = 0
        for x1, c1 in dists[0].items():
            for x2, c2 in dists[1].items():
                w = c1 * c2
                tot += w
                up += w * (x1 + x2 >= x)
                down += w * (x1 + x2 <= x)
        return Fraction(up, tot), Fraction(down, tot)
    nL = nH = 0
    for a1, b1, a2, b2 in itertools.product(range(7), repeat=4):
        cells = ((a1, 6, b1, 6), (a2, 6, b2, 6))
        assert Sm.pvals(cells) == brute(cells), cells
        r = Sm.reading_of(cells)[0]
        nL += r == "B LOWER"
        nH += r == "B HIGHER"
    assert (nL, nH) == (242, 242), (nL, nH)                   # RS: 242 L and 242 H of 2,401
    for nA1, nB1 in ((5, 6), (6, 5), (4, 6), (6, 4)):
        for a1, b1, a2, b2 in itertools.product(range(nA1 + 1), range(nB1 + 1), range(7), range(7)):
            if (a1 + b1 + a2 + b2) % 3:
                continue
            cells = ((a1, nA1, b1, nB1), (a2, 6, b2, 6))
            assert Sm.pvals(cells) == brute(cells), cells
    return True


def test_te_sections_reproduced():
    assert battery_te(S)


def test_chk8_reproduced_lost_and_censored():
    assert battery_chk8(S) and battery_chk8(S, censored=True)


def test_fill():
    assert battery_fill(S)


def test_lines_and_notes():
    assert battery_lines(S)


def test_brute_force_relabelling():
    assert battery_brute(S)


def test_full_enumeration_of_label_pairs():
    """Two tables by enumerating all 924 x 924 joint relabellings directly (no per-start shortcut)."""
    for cells in (((5, 6, 2, 6), (4, 6, 1, 6)), ((2, 6, 5, 6), (3, 6, 3, 6))):
        labs = list(itertools.combinations(range(12), 6))
        m1, m2 = cells[0][0] + cells[0][2], cells[1][0] + cells[1][2]
        x = cells[0][0] + cells[1][0]
        a1s = [sum(1 for i in lab if i < m1) for lab in labs]
        a2s = [sum(1 for i in lab if i < m2) for lab in labs]
        up = down = 0
        for u in a1s:
            for v in a2s:
                up += u + v >= x
                down += u + v <= x
        assert S.pvals(cells) == (Fraction(up, 924 * 924), Fraction(down, 924 * 924))


def test_side_and_exactness():
    assert S.SIDE == Fraction(1, 40)
    # a value on the line decides exactly: P_UP of 0.025 or less is B LOWER, never by float rounding
    for a1, b1, a2, b2 in itertools.product(range(7), repeat=4):
        up, down = S.pvals(((a1, 6, b1, 6), (a2, 6, b2, 6)))
        assert isinstance(up, Fraction) and isinstance(down, Fraction)
        assert up + down >= 1                                  # both sides never fire together


def test_fisher_one_sided():
    a_gt, b_gt = S.fisher_one_sided(6, 6, 0, 6)
    assert a_gt == Fraction(1, 924) and b_gt == 1
    a_gt, b_gt = S.fisher_one_sided(3, 6, 3, 6)
    assert a_gt == b_gt and a_gt > Fraction(1, 2)


def test_power_lines_match_te():
    lines = S.power_lines()
    assert "      0.90     1.000  0.993  0.897  0.642  0.428" in lines
    assert "      0.60     0.943  0.642  0.294  0.097  0.037" in lines
    assert any("A 0.90 0.004; A 0.75 0.158; A 0.60 0.562" in x for x in lines)


# ================================================================================================ classification
def _cls(name, check=None, read=None, lost=None):
    if check is not None:
        check.setdefault("_sha256", "c" * 64)
    return S.classify(name, check, read, lost)


def _check(name, **over):
    p = E.parse_run_name(name)
    S_, E_ = E.window(p["start"])
    d = dict(name=name, start=p["start"], arm=p["arm"], seed=p["seed"], S=S_, E=E_, T=E_, mode="continuation",
             passed=True, stop_reason="end", _sha256="c" * 64)
    d.update(over)
    return d


def _read(name, T=None, positive=True, c6_ok=True, **over):
    p = E.parse_run_name(name)
    S_, E_ = E.window(p["start"])
    d = dict(name=name, start=p["start"], arm=p["arm"], seed=p["seed"], S=S_, T=E_ if T is None else T,
             check_sha256="c" * 64, r1=dict(positive=positive, c6_ok=c6_ok,
                                           c6_counting_failures=[] if c6_ok else ["evaluation 1 absent"]))
    d.update(over)
    return d


def test_classify():
    n = "s42_A_1101"
    S_, E_ = E.window("s42")
    assert _cls(n)["cls"] == "UNDECIDED"
    assert _cls(n, _check(n, passed=False))["cls"] == "UNDECIDED"
    c = _cls(n, _check(n))
    assert c["cls"] == "COUNTED" and c["outcome"] is None and not c["read_ok"]
    assert _cls(n, _check(n), _read(n))["outcome"] == "POSITIVE"
    assert _cls(n, _check(n), _read(n, positive=False))["outcome"] == "NEGATIVE"
    T = S_ + 10 * F.GRID
    c = _cls(n, _check(n, T=T, stop_reason="collapse"), _read(n, T=T, positive=False))
    assert c["cls"] == "STOPPED" and c["outcome"] == "CENSORED"
    c = _cls(n, _check(n, T=T, stop_reason="collapse"), _read(n, T=T, positive=True))
    assert c["cls"] == "STOPPED" and c["outcome"] == "POSITIVE"
    c = _cls(n, _check(n), _read(n, c6_ok=False))
    assert c["cls"] == "LOST" and "C6" in c["cause"]
    assert _cls(n, _check(n), _read(n), lost={"cause": "crashes"})["cls"] == "LOST"
    for bad in (dict(check=_check(n, T=T)), dict(check=_check(n, stop_reason="other")),
                dict(check=_check(n, T=E_, stop_reason="collapse")), dict(check=_check(n, mode="validation")),
                dict(check=_check(n, seed=1111)), dict(check=_check(n), read=_read(n, check_sha256="d" * 64)),
                dict(check=_check(n), read=_read(n, T=T)), dict(check=_check(n), read=_read(n, arm="B"))):
        with pytest.raises(S.InputError):
            _cls(n, **bad)
    with pytest.raises(S.InputError):
        _cls("s42_A_1101_crashed1")


# ================================================================================================ the scorer on trees
def _score(root):
    res, lines, code = S.score(root, with_power=False)
    return res, "\n".join(lines), code


def test_score_incomplete_is_outcome_blind():
    root = F.work_dir("score_incomplete")
    F.outcomes_tree(root, {"s42": (6, 1), "s80": (5, 2)})
    (root / "runs" / "s80_B_1451" / R.CHECK_NAME).unlink()
    res, text, code = _score(root)
    assert code == 3 and res["status"] == "incomplete" and "s80_B_1451" in text
    for word in ("POSITIVE", "NEGATIVE", "CENSORED", "P_UP", "P_DOWN", "B LOWER", "LOWERED"):
        assert word not in text, word
    assert "outcome" not in json.dumps(res["classes"])
    root2 = F.work_dir("score_unread")
    F.outcomes_tree(root2, {"s42": (6, 1), "s80": (5, 2)})
    (R.read_dir(root2, "s42_B_1201") / R.READ_NAME).unlink()
    res, text, code = _score(root2)
    assert code == 3 and res["status"] == "incomplete" and "R1 not read yet" in text


def test_score_complete_provisional_line_one():
    root = F.work_dir("score_l1")
    F.outcomes_tree(root, {"s42": (6, 1), "s80": (5, 2)})
    res, text, code = _score(root)
    assert code == 3 and res["status"] == "PROVISIONAL" and res["line"] == 1 and res["line_text"] == S.LINE_LOWER
    assert any("APPROVED" in x for x in res["provisional"])
    assert Fraction(res["p_up"]) == S.pvals(((6, 6, 1, 6), (5, 6, 2, 6)))[0]
    assert "TABLES" in text and "P_UP   (filled against B LOWER)" in text


def test_score_lost_censored_and_c6():
    root = F.work_dir("score_missing")
    special = {"s42_A_1101": ("lost", "three crashes"), "s42_B_1251": ("censored", E.window("s42")[0] + 10 * F.GRID),
               "s80_A_1351": ("c6",), "s80_B_1401": ("stopped_pos", E.window("s80")[0] + 11 * F.GRID)}
    F.outcomes_tree(root, {"s42": (6, 1), "s80": (5, 2)}, special=special)
    res, text, code = _score(root)
    t = res["tables"]
    assert t["s42"]["counted"] == [5, 5, 1, 5] and t["s42"]["missing"] == {"A": 1, "B": 1}
    assert t["s42"]["up"] == [5, 6, 2, 6] and t["s42"]["down"] == [6, 6, 1, 6]
    assert t["s80"]["counted"] == [5, 5, 2, 6] and t["s80"]["up"] == [5, 6, 2, 6]
    assert res["n_missing"] == 3
    for n in ("s42_A_1101", "s42_B_1251", "s80_A_1351", "s80_B_1401"):
        assert n in text.split("STOPPED, CENSORED AND LOST CONTINUATIONS:")[1].split("REPORT ONLY")[0]
    # review RR-9: a C6 LOST cause is printed as the failure's text, not as a Python dict
    c6 = next(x for x in text.splitlines() if x.strip().startswith("s80_A_1351") and "C6 failing" in x)
    assert "first: evaluation 5767168 absent [grid, window]" in c6 and "{'failure'" not in text


def _final_root(tag, monkeypatch, table=None):
    """A complete, approved synthetic screen whose every PROVISIONAL guard is clear: criteria.txt, the tools, a code
    copy sealed and measured by M1's check_copy (its digest standing in for the real copy's), SHA256SUMS.txt and a
    JSON APPROVED with bare 64-hex values (ecs_common APPROVED_KEYS)."""
    root = F.work_dir(tag)
    crit = root / "criteria.txt"
    crit.write_text("criteria text\n")
    csha = E.sha256_file(crit)
    (root / "tools").mkdir()
    pins = {}
    for t in E.TOOL_FILES:
        (root / "tools" / t).write_text(f"tool {t}\n")
        pins[t] = E.sha256_file(root / "tools" / t)
    digest, n = F.sealed_code(root)
    monkeypatch.setattr(E, "TREE_DIGEST", digest)
    monkeypatch.setattr(E, "TREE_FILES", n)
    (root / "SHA256SUMS.txt").write_text("".join(f"{v}  tools/{k}\n" for k, v in pins.items()))
    appr = dict(criteria_sha256=csha, runlog_line=724, code_tree_digest=digest, tool_sha256=dict(pins))
    (root / "APPROVED").write_text(json.dumps(appr))
    F.outcomes_tree(root, table or {"s42": (6, 4), "s80": (6, 4)}, criteria_sha256=csha)
    assert _score(root)[0]["provisional"] == []
    return root, appr, pins, csha


def _prov(root):
    res, _text, code = _score(root)
    assert code == 3 and res["status"] == "PROVISIONAL", res["provisional"]
    return res["provisional"]


def test_score_final_with_pins_and_main(monkeypatch):
    root, appr, _pins, _csha = _final_root("score_final", monkeypatch)
    res, text, code = _score(root)
    assert code == 0 and res["status"] == "final" and res["line"] == 3 and res["line_text"] == S.LINE_NONE
    assert "s42: arm A 6 of 6, arm B 4 of 6" in text
    out = root / "score"
    assert S.main(["--root", str(root), "--no-power"]) == 0
    assert json.loads((out / "score.json").read_text())["status"] == "final" and (out / "score.txt").exists()
    # a pinned file changed after approval: PROVISIONAL, read and printed, not acted on
    (root / "tools" / "read_ecs.py").write_text("changed\n")
    assert any("read_ecs.py" in x for x in _prov(root))
    (root / "tools" / "read_ecs.py").write_text("tool read_ecs.py\n")
    appr2 = dict(appr, criteria_sha256="0" * 64)
    (root / "APPROVED").write_text(json.dumps(appr2))
    assert any("criteria.txt sha256" in x for x in _prov(root))


def test_score_provisional_guards(monkeypatch):
    """Review RR-2: each guard that keeps a reading from being acted on with changed tools, code or text."""
    root, appr, pins, csha = _final_root("score_guards", monkeypatch)
    # the M1 / M5 pins (mutant s_m1m5_pins_ignored)
    monkeypatch.setattr(E, "check_pins", lambda: ["/x/ledger_r10round.py: sha256 a != pinned b"])
    assert any(x.startswith("M1/M5 pin:") for x in _prov(root))
    monkeypatch.setattr(E, "check_pins", lambda: [])
    # the M5 recorded with an R1 read (mutant s_r1_m5_pin_ignored)
    rp = R.read_dir(root, "s42_B_1211") / R.READ_NAME
    keep = rp.read_text()
    d = json.loads(keep)
    d["r1"]["m5_sha256"] = "0" * 64
    rp.write_text(json.dumps(d))
    assert any("s42_B_1211: R1 ran M5" in x for x in _prov(root))
    rp.write_text(keep)
    # run_meta's criteria sha256 (mutant s_run_criteria_ignored)
    cp = root / "runs" / "s80_A_1321" / R.CHECK_NAME
    c = json.loads(cp.read_text())
    c["criteria_sha256"] = "1" * 64
    cp.write_text(json.dumps(c))
    r = json.loads((R.read_dir(root, "s80_A_1321") / R.READ_NAME).read_text())
    r["check_sha256"] = E.sha256_file(cp)
    (R.read_dir(root, "s80_A_1321") / R.READ_NAME).write_text(json.dumps(r))
    assert any("s80_A_1321: run_meta criteria sha256" in x for x in _prov(root))
    c["criteria_sha256"] = csha
    cp.write_text(json.dumps(c))
    r["check_sha256"] = E.sha256_file(cp)
    (R.read_dir(root, "s80_A_1321") / R.READ_NAME).write_text(json.dumps(r))
    assert _score(root)[2] == 0
    # a tool missing from APPROVED (mutant s_tool_pin_required_dropped), even with SHA256SUMS.txt pinning it
    (root / "APPROVED").write_text(json.dumps(dict(appr, tool_sha256={k: v for k, v in pins.items()
                                                                        if k != "score_ecs.py"})))
    assert "no pin for score_ecs.py in APPROVED" in _prov(root)
    (root / "APPROVED").write_text(json.dumps(appr))
    # the code copy measured, not its seal file read (mutant s_seal_ignored): a changed file under an unchanged seal
    x = root / "code" / "environments" / "x.py"
    x.write_text("x = 2\n")
    assert any(i.startswith("code copy:") and "changed" in i for i in _prov(root))
    x.write_text("x = 1\n")
    seal = json.loads((root / "code" / E.SEAL_NAME).read_text())
    (root / "code" / E.SEAL_NAME).write_text(json.dumps(dict(seal, tree_digest="2" * 64)))
    assert any(i.startswith("code copy") for i in _prov(root))
    (root / "code" / E.SEAL_NAME).write_text(json.dumps(seal))
    monkeypatch.setattr(E, "TREE_FILES", seal["files"] + 1)  # the pinned count is checked as well
    assert any("files) != pinned" in i for i in _prov(root))
    monkeypatch.setattr(E, "TREE_FILES", seal["files"])
    assert _score(root)[2] == 0


def test_score_approved_format_and_pin_agreement(monkeypatch):
    """Review RR-10: APPROVED carries bare 64-hex values (as run_ecs.py compares them); a prefixed or malformed value,
    a non-JSON APPROVED, and APPROVED and SHA256SUMS.txt pinning one file differently are each PROVISIONAL."""
    root, appr, pins, csha = _final_root("score_approved", monkeypatch)
    (root / "APPROVED").write_text(json.dumps(dict(appr, tool_sha256={k: f"sha256:{v}" for k, v in pins.items()})))
    assert any("is not a bare 64-hex sha256" in x for x in _prov(root))
    (root / "APPROVED").write_text(json.dumps(dict(appr, criteria_sha256="sha256:" + csha)))
    assert any("criteria_sha256" in x and "bare" in x for x in _prov(root))
    (root / "APPROVED").write_text("criteria " + csha + "\n")
    assert any("not a JSON object" in x for x in _prov(root))
    (root / "APPROVED").write_text(json.dumps(appr))
    sums = root / "SHA256SUMS.txt"
    sums.write_text(sums.read_text() + f"{'3' * 64}  tools/read_ecs.py\n")
    assert any("APPROVED pins" in x and "SHA256SUMS.txt pins" in x for x in _prov(root))


def test_score_input_errors():
    root = F.work_dir("score_bad")
    F.outcomes_tree(root, {"s42": (6, 1), "s80": (5, 2)})
    p = R.read_dir(root, "s42_A_1111") / R.READ_NAME
    d = json.loads(p.read_text())
    d["check_sha256"] = "0" * 64
    p.write_text(json.dumps(d))
    assert S.main(["--root", str(root), "--no-power"]) == 2
    root2 = F.work_dir("score_bad_lost")
    F.outcomes_tree(root2, {"s42": (6, 1), "s80": (5, 2)})
    (root2 / "lost.json").write_text(json.dumps({"schema": E.LOST_SCHEMA, "lost": {"s42_A_9999": {"cause": "x"}}}))
    assert S.main(["--root", str(root2), "--no-power"]) == 2
    assert S.main(["--root", str(root2), "--no-power", "--out", "/tmp/elsewhere_lane_r"]) == 2


def test_score_reads_r3_after_the_traces():
    """The runner makes the end traces after the R1/R2 reads, so the scorer reads R3 itself (report only)."""
    root = F.work_dir("score_r3")
    F.outcomes_tree(root, {"s42": (6, 1), "s80": (5, 2)})
    name = "s80_A_1311"
    T = E.window("s80")[1]
    z = R.end_zip(root / "runs" / name, "end", T)
    z.parent.mkdir(parents=True)
    z.write_bytes(b"end pair")
    F.write_trace(root / "traces" / name / R.TRACE_NAME, z, T)
    res, text, code = _score(root)
    line = next(x for x in text.splitlines() if x.strip().startswith(name))
    assert "R3 CMD tail_3 0.14, tail flags 1, falls 0" in line
    other = next(x for x in text.splitlines() if x.strip().startswith("s80_A_1301 "))
    assert "R3 missing" in other
    z.write_bytes(b"another file")
    _res, text, _code = _score(root)
    line = next(x for x in text.splitlines() if x.strip().startswith(name))
    assert "R3 not of the end pair" in line


def test_score_incomplete_names_void_and_crashed_folders():
    root = F.work_dir("score_void")
    F.outcomes_tree(root, {"s42": (6, 1), "s80": (5, 2)})
    run = root / "runs" / "s42_B_1221"
    run.rename(root / "runs" / "s42_B_1221_void1")
    (root / "runs" / "s80_A_1331_crashed1").mkdir()
    _res, text, code = _score(root)
    assert code == 3 and "s42_B_1221" in text and "VOID until fixed" in text and "s42_B_1221_void1" in text


def test_score_provisional_when_a_continuation_ran_with_other_tools(monkeypatch):
    root, _appr, pins, _csha = _final_root("score_run_tools", monkeypatch)
    assert _score(root)[2] == 0
    name = "s80_B_1421"
    cp = root / "runs" / name / R.CHECK_NAME
    d = json.loads(cp.read_text())
    d["meta"] = {"run_tool_sha256": dict(pins, **{"run_ecs.py": "0" * 64})}
    cp.write_text(json.dumps(d))
    rp = R.read_dir(root, name) / R.READ_NAME
    r = json.loads(rp.read_text())
    r["check_sha256"] = E.sha256_file(cp)
    rp.write_text(json.dumps(r))
    res, _text, code = _score(root)
    assert code == 3 and any(name in x and "run_ecs.py" in x for x in res["provisional"]), res["provisional"]
