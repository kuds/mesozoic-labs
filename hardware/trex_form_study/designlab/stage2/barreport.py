"""Machine-made tables from the bar's records (``designlab/stage2/bar.py``): ``bar_tables.json`` and ``BAR_TABLES.md``.

The bar writes one JSON record per set, body, seed block, row and condition.  This module reads every record under a
run directory and makes the tables a reader is given: nothing is simulated, nothing is judged again.  Counts, bounds
and passes are the records' own; a controller's verdict is ``bar.verdict`` on its two rows' records.

* ``load(out)``: every record under ``out``, indexed ``[set][body][block][row][condition]``, with what the checks
  need under the key ``"_run"`` (the ledger, today's source hashes, each set's latest freeze, the controls' records,
  the records kept beside a re-run).
* ``tables(doc)``: the JSON form of every table; ``markdown(tabs)``: the same as Markdown.  Both are pure functions:
  no time, no host; ``markdown`` reads nothing but its argument, so ``BAR_TABLES.md`` regenerates from
  ``bar_tables.json`` byte for byte.
* ``jps_table(doc)``: the seed-by-seed comparison with section 2.3 of the pull request's report (``JPS_ROWS``: its
  numbers, copied with file and line, printed only in columns labelled as its claims).
* ``write(out, dest)``: both files, written atomically.

Rules the tables keep (STAGE2_75_PLAN.md rule R5 and H1 to H6): readings in the plan's order (reading 1 "with the
look" is not defined for a set without look bands; reading 2, every limit; reading 3, every limit but for A1) and
never pooled; ``shown`` and ``provisional`` only at the judged delay, ``passes at <ms> ms`` at another; a candidate's
verdict with the replication word of each record that passed; a cell with no record reads "not run", or "cut by the
pre-registered order" where the ledger shows its unit was not started; a measured value never stands in a column
that holds a target or a limit; every line of the candidate's carries its margin label; record paths are relative to
the study directory; numbers are printed with ``%g`` or a fixed number of decimals per unit (``DECIMALS``).

Run from hardware/trex_form_study::

    python -m designlab.stage2.barreport --out RUN [--dest DIR] [--set S ...] [--body B]

This file is in no record key.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":  # before any numpy import (the bar's modules import it): one core per process
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from ..stage1 import panels  # noqa: E402
from . import bar, readings, s2io, setio  # noqa: E402

SCHEMA = "bar-tables-1"
#: The key of ``load``'s value that holds what is not a record (a set cannot take this name).
META = "_run"
TABLES_JSON, TABLES_MD = "bar_tables.json", "BAR_TABLES.md"
PLAN_FILE = "STAGE2_75_PLAN.md"
JPS_FILE = "stage2/jetson_plem/JETSON_PLEM_AND_SPEED.md"

# ------------------------------------------------------------------------------------------------
# Words
# ------------------------------------------------------------------------------------------------
NOT_RUN = "not run"
CUT = "cut by the pre-registered order"
READING_1 = "not defined for this set"
MARGIN_LABEL = "margin set after results on seeds 3042-3061"
RERUN = "re-run after a result"
REGISTERED = "registered record (the first finished one; its counts and verdict stand)"
CHANGE = "change after a result"
OTHER_SOURCES = "record made by other sources than today's"
NOT_IN_MANIFEST = "inputs not in the set's latest freeze"
NOT_CHECKED = "inputs not checked: the set's set.json was not found"
NO_VERDICT_R0 = "no verdict (an R0 row)"
READINGS = (("success", "counts", "reading 2 (every limit)"), ("but_A1", "counts_but_A1",
                                                               "reading 3 (every limit but for A1)"))

# ------------------------------------------------------------------------------------------------
# The registered order (STAGE2_75_PLAN.md H6 and B3): what a table lists even where no record exists
# ------------------------------------------------------------------------------------------------
PLAN_BODY = "75-M-H-10"
#: The sets in the tables' order: the candidate first, then the two controls.
SETS = ("F25F9cl", "F25F9", "V0")
R2_ROWS = panels.VERDICT_ROWS["R2"]
R0_ROW = "R0"
#: Task 3, tiers 1 and 3 to 5: the walk at 0.40 m/s on both grips, every set, both R2 rows and R0.
WALK_SETS, WALK_DELAYS = ("F25F9cl", "F25F9", "V0"), (10, 0, 2, 4, 6, 8)
#: Task 3, tiers 6 and 9: the speed ladder at mu 0.9 under both R2 rows.
LADDER_SETS, LADDER_DELAYS = ("F25F9cl", "V0"), (10, 0, 4, 6)
#: Task 3, tier 7: the pull request's remaining rows, under both R2 rows (set, condition, delay in ms).
PR_ROWS = (("F25F9", "walk/v0.5", 0), ("F25F9cl", "walk/v0.45_mu0.54", 0), ("F25F9cl", "walk/v0.5_mu0.54", 0),
           ("F25F9cl", "walk/v0.5", 2))
#: Task 4, tiers 1 and 3 to 5: the 14 turn, nudge and stop conditions, both R2 rows and R0; tier 6: the reported
#: ladders at 10 ms under both R2 rows.
REST_SETS, REST_DELAYS, REST_LADDER_DELAYS = ("F25F9cl", "V0"), (10, 0, 4, 6), (10,)
#: Task 3, tier 3: the omega control (control, set, condition, row, delays).
OMEGA_CONTROL = ("omega_s1", "F25F9cl", "walk/v0.4", "R2/42", (10, 0))
TASK_OF = {"walk": 3, "turn": 4, "nudge": 4, "stop": 4}
WALK_ONLY_NOTE = ("not run in this round: F25F9 is a control for the walk only (STAGE2_75_PLAN.md H3 and its list "
                  "\"Not run in this round\")")

# ------------------------------------------------------------------------------------------------
# Rules amended after a result (STAGE2_75_PLAN.md, with the plan's own labels): every table with a verdict cites them
# ------------------------------------------------------------------------------------------------
AMENDED_AFTER_RESULT = (
    dict(rule="A7", file=PLAN_FILE,
         label="[changed 15 Nov after a result: the review's scratch builds of V0's stop and quick turn (prescribed "
               "motion); before any set was frozen and before any closed-loop episode on a Stage 2 set]",
         text="The two causal limits each read their own frames."),
    dict(rule="C3", file=PLAN_FILE,
         label="[changed 15 Nov after a result: the review's reread of Stage 1's stock walks of this body under the "
               "reading above (`walk/nominal`, `walk/brisk`, `walk/fr05`, seeds 3042–3044, hard floor, no servo "
               "delay; smoke on Stage 1's own run directory, which R4 allows); before any set was frozen and before "
               "any closed-loop episode on a Stage 2 set]",
         text="The middle half is that of the **planned** swing, counted from the foot's own lift-off."),
)

# ------------------------------------------------------------------------------------------------
# JPS section 2.3: its numbers, copied (emphasis marks and backticks removed) with the file and the line
# ------------------------------------------------------------------------------------------------
JPS_HEADER = dict(file=JPS_FILE, line=90, condition="75-M-H-10 Condition (20 seeds each, R2/42 · R2/43)", delay="Delay",
                  floor_plant="Floor / Plant", stay_up="Stay Up (42 · 43)", meet_every_limit="Meet Every Limit (42 · 43)",
                  median_speed="Median Speed (m/s)", takeaway="Binding Limit / Takeaway")
JPS_ROWS = (
    dict(line=92, condition="Stage 1 V0 brisk (0.351 m/s)", delay="0 ms", floor_plant="mu = 0.90 (Pi 5)",
         stay_up="20/20 · 20/20", meet_every_limit="0/20 · 0/20", median_speed="0.360 · 0.361",
         takeaway="Fails only in-phase STS3215 tail A1 (1.05x)."),
    dict(line=93, condition="Stage 1 V0 animal pace (0.496 m/s)", delay="0 ms", floor_plant="mu = 0.90 (Pi 5)",
         stay_up="20/20 · 20/20", meet_every_limit="0/20 · 0/20", median_speed="0.506 · 0.490",
         takeaway="Fails tail A1 (1.18–1.31x), duty (0.546 / 0.518), and R2/43 ankle speed (1.17x)."),
    dict(line=94, condition="F25 + F9 (duty = 0.55, 0.496 m/s)", delay="0 ms", floor_plant="mu = 0.90 (Pi 5)",
         stay_up="20/20 · 20/20", meet_every_limit="20/20 · 0/20", median_speed="0.511 · 0.510",
         takeaway="Cures tail A1 (0.58x) & ankle speed (0.74x); R2/43 misses only duty = 0.542 < 0.55."),
    dict(line=95, condition="F25_F9_cl (duty = 0.59) @ 0.400 / 0.450 / 0.496 m/s", delay="0 ms",
         floor_plant="mu = 0.90 & 0.54 (Pi 5)", stay_up="20/20 · 20/20", meet_every_limit="20/20 · 20/20",
         median_speed="0.415 / 0.466 / 0.514",
         takeaway="Shown on both grips (18–20/20 also keep Stage 2 <= 0.25 m drift) with zero hardware change."),
    dict(line=96, condition="F25_F9_cl @ 0.550 m/s", delay="0 ms", floor_plant="mu = 0.90 (Pi 5)",
         stay_up="20/20 · 20/20", meet_every_limit="20/20 · 20/20", median_speed="0.570 · 0.568",
         takeaway="hip_pitch reaches 0.966x of 24 V speed limit (true 0 ms closed-loop speed ceiling)."),
    dict(line=97, condition="F25_F9_cl + Jetson +397 g trunk mass (0.496 m/s)", delay="0 / 2 ms",
         floor_plant="mu = 0.90 (Jetson)", stay_up="20/20 · 20/20", meet_every_limit="20/20 · 20/20",
         median_speed="0.515 · 0.516",
         takeaway="< 0.001 m/s difference vs. Pi 5 at equal delay: Jetson does not alter raw motor speed/torque."),
    dict(line=98, condition="F25_F9_cl @ 0.400 & 0.496 m/s (Partial PLEM 2 ms)", delay="2 ms", floor_plant="mu = 0.90",
         stay_up="20/20 · 20/20", meet_every_limit="20/20 · 20/20", median_speed="0.416 / 0.515",
         takeaway="Partial PLEM (CPU 5 + pcan_usb_fd) holds the 2 ms pass zone (toe = 22.8–23.6 mm)."),
    dict(line=99, condition="F25_F9_cl @ 0.496 m/s (4 ms vs 6–10 ms bridge lag)", delay="4 / 6–10 ms",
         floor_plant="mu = 0.90", stay_up="20·20 / 0·0", meet_every_limit="5·1 / 0·0", median_speed="0.512 / falls",
         takeaway="4 ms loses toe clearance (18.9–19.3 mm); 6–10 ms (S2-7) collapses 0/20 at both 0.40 & 0.50 m/s."),
    dict(line=100, condition="Stage 2 stop2 (<= 1.5 strides, <= 1.5 m/s²) @ 0.40 & 0.50 m/s", delay="2 ms",
         floor_plant="mu = 0.90", stay_up="20/20 · 20/20", meet_every_limit="20/20 · 20/20",
         median_speed="0.67 / 0.73 strides",
         takeaway="0.79 / 0.92 m/s² peak decel; 2-step stopping passes cleanly up to 0.50 m/s."),
    dict(line=101, condition="Stage 2 walking nudge (~22 N, forward / left) @ 0.40 vs 0.50 m/s", delay="2 ms",
         floor_plant="mu = 0.90", stay_up="20·20 (15·15 L) -> 20·17 (13·11 L)",
         meet_every_limit="18·15 -> 19·13 (9·7 L)", median_speed="0.416 / 0.515",
         takeaway="Mid-stride shoves (and turns) degrade above 0.40 m/s, anchoring Stage 2's 0.40 m/s all-around bar."),
)
JPS_CLAIM_KEYS = ("condition", "delay", "floor_plant", "stay_up", "meet_every_limit", "median_speed", "takeaway")
#: Which records stand beside each JPS row: (set, condition, delay in ms), or a cell this round has no condition for
#: (a string: it reads "not run").  JPS's 0.496 m/s rows stand beside the 0.50 m/s rung; its 2 ms stop and nudge rows
#: stand beside the delays this round runs on either side (0 and 4 ms), the 2 ms cell itself listed.
JPS_MAP = {
    92: (("V0", "walk/v0.35", 0),),
    93: (("V0", "walk/v0.5", 0),),
    94: (("F25F9", "walk/v0.5", 0),),
    95: tuple(("F25F9cl", f"walk/{w}{g}", 0) for w in ("v0.4", "v0.45", "v0.5") for g in ("", "_mu0.54")),
    96: ("the walk at 0.55 m/s: this round's speed ladder ends at 0.50 m/s",),
    97: ("the Jetson-mass row: this round has no plant with the added trunk mass",),
    98: (("F25F9cl", "walk/v0.4", 2), ("F25F9cl", "walk/v0.5", 2)),
    99: (("F25F9cl", "walk/v0.5", 4), ("F25F9cl", "walk/v0.5", 6), ("F25F9cl", "walk/v0.5", 8),
         ("F25F9cl", "walk/v0.5", 10), ("F25F9cl", "walk/v0.4", 6), ("F25F9cl", "walk/v0.4", 8),
         ("F25F9cl", "walk/v0.4", 10)),
    100: (("F25F9cl", "stop/two_step", 2), ("F25F9cl", "stop/two_step", 0), ("F25F9cl", "stop/two_step", 4),
          "the stop from 0.50 m/s: this round's stop is from the 0.40 m/s walk"),
    101: (("F25F9cl", "nudge/forward", 2), ("F25F9cl", "nudge/forward", 0), ("F25F9cl", "nudge/forward", 4),
          ("F25F9cl", "nudge/left", 2), ("F25F9cl", "nudge/left", 0), ("F25F9cl", "nudge/left", 4),
          "the nudges from 0.50 m/s: this round's nudges are given in the 0.40 m/s walk"),
}
JPS_STATEMENTS = (
    "JPS is `" + JPS_FILE + "`, the report of pull request kuds/mesozoic-labs#668; its section 2.3 has no code and no "
    "record (JETSON_PLEM_REVIEW.md section 1).  Its numbers stand only in the columns headed \"JPS's claim\"; they are "
    "claims of a pull request, not evidence.",
    "A JPS row at 0.496 m/s (JPS's claim) is set beside this round's 0.50 m/s rung (`walk/v0.5`).",
    "The stop and the turn of this round are commanded causally: the reference is the periodic walk, frame for "
    "frame, up to the command instant (STAGE2_75_PLAN.md A6; design D5).  JPS's were not: the causal construction is "
    "this round's (`designlab/stage2/moves.py`), and JPS ran no 90 degree turn.  A stopped time here is counted from "
    "that causal command.",
    "JPS's 2 ms stop and nudge rows: this round reads the turn, the nudges and the stop at 10, 0, 4 and 6 ms "
    "(STAGE2_75_PLAN.md B3); the 2 ms cell is listed, and the records at 0 and 4 ms stand beside it.",
    "Stage 1's reading (`measures.stage1.succeeded`: `tune.measure`'s terms restricted to the panels' list for that "
    "walk) is the reading JPS names; it gives no verdict here.  It is defined for walks only.",
)

# ------------------------------------------------------------------------------------------------
# Number formats: a fixed number of decimals per unit; everything else with %g
# ------------------------------------------------------------------------------------------------
DECIMALS = {"m": 3, "m/s": 3, "s": 2, "deg": 1, "mm": 1, "ratio": 3, "bound": 3, "m/s^2": 2, "strides": 2, "N": 1,
            "core-h": 3, "rad/s": 4}


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _num(x, unit: str | None = None) -> str:
    """A number as the tables print it: an int as it is, a float with its unit's decimals (``DECIMALS``), else %g."""
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    d = DECIMALS.get(unit or "")
    return f"{float(x):.{d}f}" if d is not None else f"{float(x):g}"


# ------------------------------------------------------------------------------------------------
# Reading a run directory
# ------------------------------------------------------------------------------------------------
def _read(path) -> dict | None:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _today_sources() -> dict:
    """Today's source hashes, as a record's ``key.sources`` holds them."""
    return dict(panels=panels.panel_sources()["sha256"], bar=bar.bar_sources()["sha256"])


def _rel(path, out: Path, study: Path) -> str:
    """A path as the tables cite it: relative to the study directory; for a run directory outside it, from the run
    directory's own name."""
    p, s = Path(os.path.abspath(path)), Path(os.path.abspath(study))
    if p == s or s in p.parents:
        return p.relative_to(s).as_posix()
    o = Path(os.path.abspath(out))
    return (Path(o.name) / os.path.relpath(p, o)).as_posix()


def _superseded(path: Path, out: Path, study: Path) -> list[dict]:
    """The records kept beside ``path`` (``<record>.superseded/``), oldest first: each with its path, the record
    (None: it cannot be read) and the reason file written beside it."""
    sdir = path.with_name(path.name + ".superseded")
    if not sdir.is_dir():
        return []
    reasons = {q.name[: -len(".reason.json")]: _read(q) for q in sorted(sdir.glob("*.reason.json"))}
    kept = []
    for q in sorted(sdir.glob("*.json")):
        if q.name.endswith(".reason.json"):
            continue
        why = reasons.get(q.stem) or {}
        kept.append(dict(path=_rel(q, out, study), name=q.name, record=_read(q), reason=why or None,
                         sequence=int(why.get("sequence") or 0)))
    return sorted(kept, key=lambda k: (k["sequence"], k["name"]))


def _index_root(root: Path, out: Path, study: Path, sets, body, meta: dict) -> dict:
    """Every record under ``root`` (``<root>/<set>/<body>/<block>/eval/<row>/<behavior>/<condition>.json``, the
    runner's ``bar.record_path``), indexed ``[set][body][block][row][condition]`` by the record's own fields."""
    index: dict = {}
    for edir in sorted(root.glob("*/*/*/eval")):
        files = [q for q in sorted(edir.rglob("*.json"))
                 if not (q.parent == edir and q.name == "summary.json")
                 and not q.relative_to(edir).parts[0].startswith("traces")
                 and not any(x.endswith(".superseded") for x in q.relative_to(edir).parts)]
        pending = [d for d in sorted(edir.rglob("*.json.superseded"))
                   if d.is_dir() and not d.with_name(d.name[: -len(".superseded")]).exists()]
        for q in files + pending:
            if q.is_dir():  # a re-run that has not finished: the registered record is the kept one
                kept = _superseded(q.with_name(q.name[: -len(".superseded")]), out, study)
                rec = dict(kept[0]["record"] or {}) if kept else {}
                rec.update(_rerun_pending=True, _superseded=kept)
                path = Path(kept[0]["path"]) if kept else q
                rel = kept[0]["path"] if kept else _rel(q, out, study)
            else:
                rec, path, rel = _read(q), q, _rel(q, out, study)
            if not rec or rec.get("schema") != bar.SCHEMA or not all(k in rec for k in ("set", "body", "block", "row",
                                                                                         "key_name", "status")):
                meta["unreadable"].append(rel)
                continue
            if (sets is not None and rec["set"] not in sets) or (body is not None and rec["body"] != body):
                continue
            rec["_path"] = rel
            if not q.is_dir():
                rec["_superseded"] = _superseded(q, out, study)
                want = bar.block_dir(root, rec["set"], rec["body"], rec["block"]) / "eval" / rec["row"] / (
                    rec["key_name"] + ".json")
                if want != path:
                    meta["misplaced"].append(rel)
            slot = index.setdefault(rec["set"], {}).setdefault(rec["body"], {}).setdefault(rec["block"], {}).setdefault(
                rec["row"], {})
            if rec["key_name"] in slot:
                meta["misplaced"].append(rel)
                continue
            slot[rec["key_name"]] = rec
    return index


def load(out, sets=None, body=None, *, study=None) -> dict:
    """Every record under the run directory ``out``, indexed ``[set][body][block][row][condition]`` (row
    ``"R2/42@10ms"``, condition ``"walk/v0.4"``); ``sets`` and ``body`` keep those only.  Each record carries
    ``_path`` (relative to ``study``, default the study directory) and ``_superseded`` (the records kept beside it by
    a re-run).  Under ``"_run"``: the run directory, ``budget.json``, ``COST.md``'s rates, today's source hashes, the
    latest freeze of every set a record names (``<study>/<inputs.dir>/set.json``), the controls' records
    (``controls[control][set]...``) and the files that are no record."""
    out = Path(out)
    study = Path(s2io.STUDY if study is None else study)
    sets = None if sets is None else [str(s) for s in sets]
    if not out.is_dir():
        raise s2io.S2Error(f"{out}: no such run directory")
    meta: dict = dict(out=_rel(out, out, study), sets=sets, body=body, unreadable=[], misplaced=[], controls={},
                      budget=_read(out / "budget.json"), cost=None, today=_today_sources(), manifests={})
    cost = out / "COST.md"
    if cost.is_file():
        meta["cost"] = bar.read_cost_text(cost.read_text())
    doc = _index_root(out, out, study, sets, body, meta)
    if META in doc:
        raise s2io.S2Error(f"{out}: a set named {META!r}")
    for cdir in sorted((out / "controls").glob("*")) if (out / "controls").is_dir() else []:
        if cdir.is_dir():
            meta["controls"][cdir.name] = _index_root(cdir, out, study, sets, body, meta)
    for rec in list(_records(doc)) + [r for c in meta["controls"].values() for r in _records(c)]:
        d = (rec.get("inputs") or {}).get("dir")
        if d and d not in meta["manifests"]:
            sd = _read(study / d / "set.json")
            fz = setio.frozen(sd) if sd else None
            meta["manifests"][d] = (dict(found=False) if sd is None else dict(found=True, frozen=False) if fz is None
                                    else dict(found=True, frozen=True, index=fz["index"],
                                              manifest_sha256=fz["manifest_sha256"], entries=fz["entries"]))
    doc[META] = meta
    return doc


def _records(index: dict):
    """Every record of an index, in sorted order of set, body, block, row and condition."""
    for s in sorted(k for k in index if k != META):
        for b in sorted(index[s]):
            for blk in sorted(index[s][b]):
                for row in sorted(index[s][b][blk]):
                    for key in sorted(index[s][b][blk][row]):
                        yield index[s][b][blk][row][key]


def _rec(index: dict, set_name: str, body: str, block: str, row: str, key: str) -> dict | None:
    try:
        return index[set_name][body][block][row][key]
    except KeyError:
        return None


# ------------------------------------------------------------------------------------------------
# Conditions, the registered order and the ledger
# ------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _conds() -> dict:
    """{key: description} of every condition the bar defines (``bar.conditions`` on a set with no entries)."""
    stub = {"set.json": {"set": ""}, "refs.json": {"references": {}}, "transitions.json": {"transitions": {}}}
    return {c.key: dict(c.describe(), groups=[g for g, f in bar.GROUPS.items() if f(c)])
            for c in bar.conditions(stub, every=True)}


def _group(name: str) -> list[str]:
    return [k for k, c in _conds().items() if name in c["groups"]]


@functools.lru_cache(maxsize=1)
def planned() -> dict:
    """The registered order as cells: ``{(set, condition, delay in ms, row name before its delay): task}``
    (STAGE2_75_PLAN.md H6; the replication and attribution units follow from these)."""
    out = {}
    for s in WALK_SETS:
        for k in _group("@walk"):
            for ms in WALK_DELAYS:
                for r in R2_ROWS + (R0_ROW,):
                    out[(s, k, ms, r)] = 3
    for s in LADDER_SETS:
        for k in _group("@speed-ladder"):
            for ms in LADDER_DELAYS:
                for r in R2_ROWS:
                    out[(s, k, ms, r)] = 3
    for s, k, ms in PR_ROWS:
        for r in R2_ROWS:
            out[(s, k, ms, r)] = 3
    for s in REST_SETS:
        for k in _group("@rest"):
            for ms in REST_DELAYS:
                for r in R2_ROWS + (R0_ROW,):
                    out[(s, k, ms, r)] = 4
        for k in _group("@ladders"):
            for ms in REST_LADDER_DELAYS:
                for r in R2_ROWS:
                    out[(s, k, ms, r)] = 4
    return out


def _split_row(row: str) -> tuple[str, int | None]:
    """A row's name before its delay and the delay in ms ("R2/42@10ms" -> ("R2/42", 10)); (name, None) for a row
    that names no delay."""
    ms = bar.delay_ms(row)
    return (row, None) if ms is None else (row.rsplit("@", 1)[0], int(ms))


def _run_path(set_name: str, body: str, block: str, row: str, key: str, control: str | None = None) -> str:
    """A record's path relative to the run directory, as the runner makes it (``bar.block_dir``) and as the ledger
    lists it (``detail.paths``)."""
    return (bar.block_dir(Path(), set_name, body, block, control) / "eval" / row / f"{key}.json").as_posix()


def _ledger(meta: dict) -> dict:
    """Per task of ``budget.json``: the paths of its started units' records and of its cut units' records, and
    whether a unit of it was cut."""
    out = {}
    for t, task in ((meta.get("budget") or {}).get("tasks") or {}).items():
        started = {str(q) for u in task.get("units") or [] for q in ((u.get("detail") or {}).get("paths") or [])}
        cut = {str(q) for u in task.get("cut") or [] for q in ((u.get("detail") or {}).get("paths") or [])}
        out[str(t)] = dict(started=started, cut=cut, any_cut=bool(task.get("cut")))
    return out


def absent_word(meta: dict, set_name: str, body: str, block: str, row: str, key: str, control: str | None = None,
                in_order: bool | None = None) -> str:
    """What a cell with no record reads: "cut by the pre-registered order" when ``budget.json`` lists its record
    under a cut unit, or when a unit of its task was cut and the cell, one of the registered order's
    (``in_order``; default: ``planned()``), is in no started unit (the first unit that does not fit is cut with every
    unit after it); else "not run"."""
    path = _run_path(set_name, body, block, row, key, control)
    ledger = _ledger(meta)
    if any(path in t["cut"] for t in ledger.values()):
        return CUT
    base, ms = _split_row(row)
    if in_order is None:
        in_order = (set_name, key, ms, base) in planned() and body == PLAN_BODY and control is None
    t = ledger.get(str(TASK_OF.get(key.split("/", 1)[0])))
    if in_order and t is not None and t["any_cut"] and path not in t["started"]:
        return CUT
    return NOT_RUN


# ------------------------------------------------------------------------------------------------
# One record, as the tables read it
# ------------------------------------------------------------------------------------------------
def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _median(values) -> dict:
    """``{median, n}`` of the numbers among ``values`` (the episodes that give the quantity)."""
    xs = [float(v) for v in values if _is_num(v)]
    return dict(median=statistics.median(xs) if xs else None, n=len(xs))


def _episodes(rec: dict) -> list[dict]:
    """The record's episodes that ran (an episode that failed to run is no result)."""
    return [e for e in rec.get("episodes") or [] if not e.get("error")]


def _one(values):
    """The one value a record's episodes share (a limit, a definition); a list if they differ; None if none has it."""
    seen = []
    for v in values:
        if v is not None and v not in seen:
            seen.append(v)
    return None if not seen else seen[0] if len(seen) == 1 else seen


def grip_of(definition: dict) -> str:
    mu = (definition or {}).get("friction")
    return "mu 0.9" if mu is None else f"mu {float(mu):g}"


def item_of(definition: dict) -> str:
    """A condition in plain words, from its description (``BarCondition.describe``)."""
    d = definition or {}
    b, name = d.get("behavior"), str(d.get("name"))
    if d.get("stock") or d.get("judge") not in bar.BAR_JUDGES:
        return f"Stage 1's condition {b}/{name}"
    if b == "walk":
        v = setio.parse_name(d["walk"]).get("speed_m_s")
        return "walk at the nominal speed" if v is None else f"walk at {v:g} m/s"
    if b == "turn":
        rate = d.get("case")
        quick = " (the quick turn)" if rate == setio.QUICK else ""
        label = "pi/4" if rate == setio.QUICK else str(rate)
        return f"turn {d.get('direction')}, {setio.TURN_ANGLE_DEG} deg at {label} rad/s{quick}"
    if b == "nudge":
        return f"nudge {d.get('direction')}, {float(d.get('frac')):g} x the push bound"
    if b == "stop":
        steps = setio.parse_name(d["move"]).get("steps") if d.get("move") else None
        return "stop, two touchdowns" if steps == 2 else "stop, Stage 1's one-step stop" if steps == 1 else f"stop {name}"
    return f"{b}/{name}"


def _flat(numbers, prefix: str = "", limit: int = 12) -> str:
    """A reference's own numbers on one line: ``key=value`` pairs, nested keys dotted, floats with %g."""
    items: list[str] = []

    def walk(x, name):
        if isinstance(x, dict):
            for k in x:
                walk(x[k], f"{name}.{k}" if name else str(k))
        elif isinstance(x, (list, tuple)):
            items.append((f"{name}=" if name else "") + "/".join(_num(v) if _is_num(v) else str(v) for v in x))
        elif x is not None:
            items.append((f"{name}=" if name else "") + (_num(x) if _is_num(x) else str(x)))

    walk(numbers, prefix)
    more = len(items) - limit
    return ", ".join(items[:limit]) + (f", ... ({more} more in the record)" if more > 0 else "")


def status_of(rec: dict) -> str:
    """"done", or "not run: <the record's reason> (<the missing reference's own numbers>)"."""
    if rec.get("status") == "done":
        return "done"
    text = f"{NOT_RUN}: {rec.get('reason') or 'no reason recorded'}"
    if rec.get("reference"):
        text += f" [{rec['reference']}]"
    nums = _flat(rec.get("numbers")) if rec.get("numbers") else ""
    return text + (f" ({nums})" if nums else "")


def pass_word(counts: dict | None, ms: int | None) -> str:
    """A record's pass in one reading, in words: at the judged delay "passes" or "does not pass"; at another "passes
    at <ms> ms"; "not read: a1_over"; "not judged" with why."""
    if counts is None:
        return "no such reading (Stage 1's one reading)"
    if counts.get("not_read"):
        return f"not read: {counts['not_read']}"
    if counts.get("pass_") is None:
        n, bad = int(counts.get("episodes") or 0), int(counts.get("errors") or 0)
        why = (f"{bad} episode{'' if bad == 1 else 's'} failed to run" if bad
               else f"{n} episode{'' if n == 1 else 's'}, not the block's {bar.RUNS}" if n != bar.RUNS
               else "its seeds are not a block's")
        return f"not judged ({why})"
    at = "" if ms == bar.JUDGED_MS or ms is None else f" at {ms} ms"
    return ("passes" if counts["pass_"] else "does not pass") + at


def _worst_drive(rec: dict, which: str) -> dict | None:
    """The drive with the largest sustained ratio (RMS over its sustained rating) over the record's episodes, on the
    span the item is judged on: ``{name, ratio, n, span}``; ``which``: "worst", "worst_leg" or "worst_a1"."""
    best, n, span = None, 0, None
    for e in _episodes(rec):
        d = _get(e, "measures", "bar", "drives")
        if isinstance(d, dict):
            w = _get(d, d.get("judged_span"), which)
            span = d.get("judged_span") or span
        else:  # a stock condition: Stage 1's episode value
            w = e.get("worst_sustained_ratio") if which == "worst" else None
            span = "episode (Stage 1's)" if w else span
        if isinstance(w, (list, tuple)) and len(w) == 2 and _is_num(w[1]):
            n += 1
            if best is None or w[1] > best[1]:
                best = (str(w[0]), float(w[1]))
    return None if best is None else dict(name=best[0], ratio=best[1], n=n, span=span)


def _failed(rec: dict, key: str) -> dict:
    """{term: episodes} over the record's episodes that ran, most frequent first (a fall counts under "survived")."""
    c = Counter(t for e in _episodes(rec) for t in (e.get(key) or ()))
    return {t: n for t, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))}


def flags_of(rec: dict, meta: dict) -> list[str]:
    """The checks a record fails, as labels: other sources than today's; inputs that the set's latest freeze does not
    list (or a set that cannot be checked)."""
    out = []
    src = _get(rec, "key", "sources") or {}
    differ = [k for k in ("panels", "bar") if src.get(k) != meta["today"].get(k)]
    if differ:
        out.append(f"{OTHER_SOURCES} ({', '.join(differ)})")
    why = manifest_differences(rec, meta)
    if why:
        out.append(NOT_CHECKED if why == [NOT_CHECKED] else NOT_IN_MANIFEST)
    return out


def manifest_differences(rec: dict, meta: dict) -> list[str] | None:
    """How a record's input hashes differ from its set's latest freeze: [] when every file is listed with its sha256
    and its entry's hashed subset; ``[NOT_CHECKED]`` when the set's ``set.json`` is not found or holds no freeze; None
    for a record without inputs (a "not run" record, a stock condition on Stage 1's run directory)."""
    files = _get(rec, "inputs", "files")
    if not files or rec.get("set") == bar.STOCK_SET:
        return None
    m = meta["manifests"].get(_get(rec, "inputs", "dir")) or {}
    if not m.get("found") or not m.get("frozen"):
        return [NOT_CHECKED]
    out = []
    for f in sorted(files):
        ent = m["entries"].get(Path(f).stem)
        if ent is None:
            out.append(f"{f}: not listed")
        elif ent.get("sha256") != files[f].get("sha256"):
            out.append(f"{f}: another sha256")
        elif ent.get("entry_sha256") != files[f].get("entry_sha256"):
            out.append(f"{f}: another entry_sha256")
    return out


COUNT_KEYS = ("episodes", "survived", "succeeded", "falls", "errors", "survival_lcb", "success_lcb", "judged", "pass_")


def counts_differences(rec: dict) -> list[str]:
    """Where a finished record's counts, bounds and passes differ from ``bar.counts`` on its own episodes and seeds
    (floats to 1e-12); reading 2 of a turn frozen with ``a1_over`` holds its pass under ``pass_if_read``."""
    out = []
    if rec.get("status") != "done" or rec.get("_brief"):
        return out
    for which, ck, name in READINGS:
        have = rec.get(ck)
        if have is None:
            continue
        want = bar.counts(rec.get("episodes") or [], rec.get("seeds") or [], which)
        for k in COUNT_KEYS:
            a = have.get("pass_if_read") if k == "pass_" and have.get("not_read") else have.get(k)
            b = want[k]
            same = abs(a - b) <= 1e-12 if _is_num(a) and _is_num(b) and not isinstance(b, int) else a == b
            if not same:
                out.append(f"{name}: {k} is {a!r} in the record, {b!r} from its episodes")
    return out


def labels_of(set_name: str, rec: dict | None = None, meta: dict | None = None) -> list[str]:
    """The labels a line carries: the candidate's margin label (every line of that set), a set version, and from the
    record: its reference's label, a re-run after a result with its reason, a change after a result with its
    reasons, a control, stand-in walks, and the checks it fails."""
    out = []
    base = bar.registered_name(set_name)
    if base == bar.CANDIDATE_SET:
        out.append(MARGIN_LABEL)
    if base is not None and base != set_name:
        out.append(f"a version of set {base}")
    if rec is None:
        return out
    rs = _get(rec, "rules", "set") or {}
    for k, what in (("label", "walk reference labelled"), ("move_label", "move reference labelled")):
        if rs.get(k):
            out.append(f"{what} {rs[k]}")
    if rec.get("control"):
        out.append(f"control: {rec['control']}")
    if rec.get("supersedes"):
        out.append(f"{RERUN}: {rec['supersedes'].get('reason') or 'no reason recorded'}")
    if rec.get("_rerun_pending"):
        out.append(f"{REGISTERED}; a {RERUN} was started and has no record yet")
    ch = rec.get("change_after_result")
    if ch:
        reasons = "; ".join(f"freeze {f.get('freeze')}: {f.get('reason')}" for f in ch.get("freezes") or [])
        out.append(f"{CHANGE}: {reasons or 'no reason recorded'}")
    if _get(rec, "set_info", "after_check_episodes"):
        out.append("a freeze made after check episodes (no change after a result)")
    if rec.get("stand_in_walks"):
        out.append("a fixture set whose walks stand in for rungs of another speed")
    conv = _get(rec, "command", "convention")
    if conv and not str(conv).startswith("causal"):
        out.append(f"command read by {conv}")
    judged_too = [t for t in rec.get("terms") or [] if t in panels.NOMINAL_ONLY_TERMS]
    if judged_too and rec.get("behavior") == "walk" and not _get(rec, "definition", "stock"):
        out.append(f"{', '.join(judged_too)} judged as Stage 1 judges them")
    if meta is not None:
        out += flags_of(rec, meta)
    return out


def digest(rec: dict, meta: dict | None = None) -> dict:
    """What the tables print of one record: its names, both readings' counts, bounds and passes (the record's own),
    Stage 1's reading, falls, failed terms, the worst drive, labels and its path.  Measured medians are added by
    ``measures_of``."""
    row = str(rec["row"])
    ms = bar.delay_ms(row)
    d = rec.get("definition") or {}
    done = rec.get("status") == "done"
    c2, c3 = rec.get("counts") or {}, rec.get("counts_but_A1")
    eps = _episodes(rec)
    stage1 = [bool(_get(e, "measures", "stage1", "succeeded")) for e in eps
              if isinstance(_get(e, "measures", "stage1"), dict)]
    falls = rec.get("falls") or {}
    att = rec.get("attribution") or {}
    wd, wl = _worst_drive(rec, "worst"), _worst_drive(rec, "worst_leg")

    def of(c, k):
        return None if not done or c is None or c.get(k) is None else dict(k=int(c[k]), n=int(c.get("episodes") or 0))

    return dict(
        set=rec["set"], body=rec["body"], block=rec["block"], row=row, controller=bar.controller_of(row), delay_ms=ms,
        behavior=rec.get("behavior"), condition=rec["key_name"], item=item_of(d), grip=grip_of(d),
        gated=rec.get("plan_gated"), status=status_of(rec), done=done, labels=labels_of(rec["set"], rec, meta),
        episodes=int(c2.get("episodes") or 0) if done else None, up=of(c2, "survived"),
        up_lcb=c2.get("survival_lcb") if done else None, errors=int(c2.get("errors") or 0) if done else None,
        reading_1=READING_1 if rec.get("counts_look") is None else "defined",
        r2=of(c2, "succeeded"), r2_lcb=c2.get("success_lcb") if done else None,
        r2_pass=pass_word(c2, ms) if done else None, r2_pass_=c2.get("pass_") if done else None,
        r3=of(c3, "succeeded"), r3_lcb=(c3 or {}).get("success_lcb") if done else None,
        r3_pass=pass_word(c3, ms) if done else None, r3_pass_=(c3 or {}).get("pass_") if done else None,
        stage1=dict(k=sum(stage1), n=len(stage1)) if stage1 else None,
        falls_before=falls.get("before_command") if done else None,
        falls_after=falls.get("after_command") if done else None,
        falls_no_command=falls.get("no_command") if done else None,
        failed_2=_failed(rec, "failed_terms") if done else None,
        failed_3=_failed(rec, "failed_terms_but_A1") if done and c3 is not None else None,
        worst_drive=None if wd is None else wd["name"], worst_ratio=None if wd is None else wd["ratio"],
        worst_leg_drive=None if wl is None else wl["name"], worst_leg_ratio=None if wl is None else wl["ratio"],
        drive_span=None if wd is None else wd["span"], ratio_limit=1.0 if wd is not None else None,
        attribution=(att.get("reported_as") or att.get("label")) if att else
        ((rec.get("failure") or {}).get("falls_label") if done else None),
        label_2=_get(rec, "failure", "label"), label_3=_get(rec, "failure_but_A1", "label"),
        omega=rec.get("omega"), crouch_frac=rec.get("crouch_frac"), control=rec.get("control"),
        replication_of=rec.get("replication_of"), path=rec.get("_path"))


def measures_of(rec: dict) -> dict:
    """The record's measured values as medians over the episodes that give each (``{median, n}``), beside the limits
    and definitions the episodes hold (never mixed: a key ending ``_limit`` or ``_defined`` is no measurement)."""
    eps = _episodes(rec)
    bars = [_get(e, "measures", "bar") or {} for e in eps]

    def med(*keys, scale: float = 1.0, source="bar"):
        vals = [_get(e, "measures", source, *keys) for e in eps]
        m = _median([v * scale for v in vals if _is_num(v)])
        return m

    def shared(*keys, scale: float = 1.0):
        v = _one(_get(b, *keys) for b in bars)
        return v * scale if _is_num(v) else v

    out: dict = {}
    b = rec.get("behavior")
    judge = _get(rec, "definition", "judge")
    if judge == "bar_walk" or (b == "walk" and judge not in bar.BAR_JUDGES):
        out.update(speed=med("speed_m_s", source="window"), along=med("along_commanded_heading_m"),
                   along_limit=shared("along_target_m"), lateral=med("lateral_from_line_max_m"),
                   lateral_limit=shared("line_max_m"))
    if judge in ("bar_walk", "bar_turn"):
        out.update(toe=med("toe_clearance_own_swing_m", scale=1e3), toe_limit=shared("toe", "limit_m", scale=1e3),
                   toe_first=med("toe_clearance_own_duration_m", scale=1e3),
                   toe_stage1=med("stage1_toe_clearance_m", scale=1e3))
    if judge == "bar_turn":
        out.update(heading_at=med("heading_error_at_deg"), heading_hold=med("heading_error_hold_max_deg"),
                   heading_limit=shared("tol_deg"), within=med("time_to_within_10deg_s"), at_defined=shared("at_s"),
                   hold_defined=shared("hold_s"), turned_own=med("turned_own_deg"))
    if judge == "bar_nudge":
        rec_ok = [bool(_get(e, "success", "recovered")) for e in eps]
        force = _one(None if not isinstance(f, list) else math.sqrt(sum(float(x) ** 2 for x in f))
                     for f in (_get(e, "measures", "push", "force_N") for e in eps))
        out.update(recovered=dict(k=sum(rec_ok), n=len(rec_ok)), back=med("after_s", source="recovery"),
                   back_limit=_one(_get(e, "measures", "recovery", "limit_s") for e in eps), force_defined=force)
    if judge == "bar_stop":
        n_td = Counter(str(v) for v in (x.get("touchdowns_after_command") for x in bars) if v is not None)
        ok = [bool(_get(e, "success", "stopped_by_end")) for e in eps]
        out.update(touchdowns={k: n_td[k] for k in sorted(n_td, key=lambda s: (len(s), s))},
                   touchdowns_limit=readings.STOP_TOUCHDOWNS_MAX, stopped=dict(k=sum(ok), n=len(ok)),
                   stopped_s=med("forward_stopped_s"), stopped_strides=med("forward_stopped_strides"),
                   accel=med("stop_acceleration_1hz_m_s2"), accel_limit=readings.ACCEL_MAX_M_S2,
                   decel_raw=med("deceleration_raw_peak_m_s2"), convention=shared("command_convention"))
    return out


# ------------------------------------------------------------------------------------------------
# Verdicts and replication (the runner's words: bar.verdict, bar.REPLICATION_WORDS)
# ------------------------------------------------------------------------------------------------
def registered(rec: dict | None) -> dict | None:
    """The registered record at a record's path: the first one finished there (STAGE2_75_PLAN.md rule R2).  A record
    that superseded none is its own; else the kept record its ``supersedes.registered`` names, or, where that file
    cannot be read, the counts ``supersedes.registered`` copied from it."""
    if rec is None or not rec.get("supersedes") or rec.get("_rerun_pending"):
        return rec
    reg = rec["supersedes"].get("registered") or {}
    kept = rec.get("_superseded") or []
    if reg.get("record"):
        pick = next((k for k in kept if k["name"] == reg["record"]), None)
    else:  # the later record names none (its reason file was lost): the oldest record kept beside it
        pick = kept[0] if kept else None
    if pick is not None and pick["record"]:
        return dict(pick["record"], _path=pick["path"], _registered_of=rec.get("_path"))
    return dict(schema=bar.SCHEMA, set=rec["set"], body=rec["body"], block=rec["block"], row=rec["row"],
                key_name=rec["key_name"], behavior=rec.get("behavior"), definition=rec.get("definition"),
                status=reg.get("status") or "done", counts=reg.get("counts") or {},
                counts_but_A1=reg.get("counts_but_A1"), episodes=[], _path=None, _registered_of=rec.get("_path"),
                _brief=True)


def _vcell(rec: dict | None) -> dict | None:
    """A record as ``bar.verdict`` reads it."""
    if rec is None:
        return None
    return dict(status=rec.get("status"), counts=rec.get("counts"), counts_but_A1=rec.get("counts_but_A1"),
                label=_get(rec, "failure", "label"), label_but_A1=_get(rec, "failure_but_A1", "label"))


def replication_word(doc: dict, set_name: str, body: str, row: str, key: str, which: str) -> str:
    """``bar.replication_word`` from the loaded records: what stands beside a passing headline record in one
    reading.  It reads the registered record of the second block (the first finished at its path)."""
    rec = registered(_rec(doc, set_name, body, bar.REPLICATION_BLOCK, row, key))
    if rec and rec.get("status") == "done":
        ok = (rec.get("counts" if which == "success" else "counts_but_A1") or {}).get("pass_")
        return (bar.REPLICATION_WORDS[0] if ok is True else bar.REPLICATION_WORDS[1] if ok is False
                else bar.REPLICATION_PENDING[1])
    if absent_word(doc[META], set_name, body, bar.REPLICATION_BLOCK, row, key, in_order=True) == CUT:
        return bar.REPLICATION_WORDS[2]
    return bar.REPLICATION_PENDING[0]


def verdict_of(doc: dict, set_name: str, body: str, key: str, ms: int, ctrl: str = "R2",
               block: str = bar.HEADLINE_BLOCK, latest: bool = False) -> dict:
    """A controller's verdict on one condition and delay, from its two rows' registered records (``latest``: from
    the records at the paths now, which differ where a record was re-run): ``{reading_2, reading_3, rows, paths}``,
    each reading ``{text, verdict, replication}``.  With no record under either row the text is the absent word."""
    names = [f"{r}@{ms}ms" for r in panels.VERDICT_ROWS[ctrl]]
    now = [_rec(doc, set_name, body, block, r, key) for r in names]
    recs = now if latest else [registered(r) for r in now]
    out = dict(rows=names, paths=[r.get("_path") if r else None for r in recs],
               rerun=any(bool(r and r.get("supersedes")) for r in now), none=all(r is None for r in recs))
    absent = [None if r is not None else absent_word(doc[META], set_name, body, block, n, key)
              for r, n in zip(recs, names, strict=True)]
    for rkey, which in (("reading_2", "success"), ("reading_3", "but_A1")):
        if all(r is None for r in recs):
            word = absent[0] if absent[0] == absent[1] else f"{names[0]}: {absent[0]}; {names[1]}: {absent[1]}"
            out[rkey] = dict(verdict=word, text=word, replication=None)
            continue
        v = bar.verdict([_vcell(r) for r in recs], ms, which)
        text, rep = v["verdict"], None
        if v["verdict"] == "incomplete" and any(r is None for r in recs):
            text += " (" + "; ".join(f"{n}: {a}" for n, a in zip(names, absent, strict=True) if a) + ")"
        if v["verdict"] == NOT_RUN:
            why = next((status_of(r) for r in recs if r is not None and r.get("status") != "done"), None)
            text = why or text
        if block == bar.HEADLINE_BLOCK and v["verdict"].startswith(("shown", "provisional")):
            if bar.registered_name(set_name) in (None, bar.CANDIDATE_SET):
                rep = {n: replication_word(doc, set_name, body, n, key, which)
                       for n, ok in zip(names, v["passes"], strict=True) if ok}
                text += " [" + "; ".join(f"{n}: {w}" for n, w in rep.items()) + "]"
            else:
                text += " [a control is not replicated]"
        out[rkey] = dict(verdict=v["verdict"], text=text, replication=rep)
    return out


# ------------------------------------------------------------------------------------------------
# Table parts
# ------------------------------------------------------------------------------------------------
def _col(key: str, title: str, kind: str = "text", unit: str | None = None) -> dict:
    """A column: ``kind`` says what it holds (text, code, labels, of, count, bound, word, measured, limit, defined,
    terms, claim, path, list) and how ``markdown`` prints it; ``unit`` picks the decimals (``DECIMALS``)."""
    return dict(key=key, title=title, kind=kind, unit=unit)


HEAD = [_col("set", "Set", "code"), _col("labels", "Labels", "labels"), _col("item", "Item"),
        _col("condition", "Condition", "code"), _col("grip", "Grip"), _col("delay", "Servo delay"),
        _col("row", "Row", "code")]
COUNT_COLUMNS = HEAD + [
    _col("status", "Status"), _col("episodes", "Episodes", "count"), _col("up", "Stayed up", "of"),
    _col("up_lcb", "Stayed up: one-sided 95 % lower bound (share of episodes)", "bound", "bound"),
    _col("reading_1", "Reading 1 (with the look)"),
    _col("r2", "Reading 2 (every limit): succeeded", "of"),
    _col("r2_lcb", "Reading 2: one-sided 95 % lower bound (share of episodes)", "bound", "bound"),
    _col("r2_pass", "Reading 2: pass", "word"),
    _col("r3", "Reading 3 (every limit but for A1): succeeded", "of"),
    _col("r3_lcb", "Reading 3: one-sided 95 % lower bound (share of episodes)", "bound", "bound"),
    _col("r3_pass", "Reading 3: pass", "word"),
    _col("stage1", "Stage 1's reading: succeeded (no verdict)", "of"), _col("path", "Record", "path")]
FAIL_COLUMNS = HEAD + [
    _col("falls_before", "Falls before the command", "count"), _col("falls_after", "Falls after the command", "count"),
    _col("falls_no_command", "Falls (an item without a command)", "count"),
    _col("failed_2", "Failed terms, reading 2 (term: episodes)", "terms"),
    _col("failed_3", "Failed terms, reading 3 (term: episodes)", "terms"),
    _col("worst_drive", "Worst drive", "code"),
    _col("worst_ratio", "Worst drive: largest RMS over its sustained rating, measured", "measured", "ratio"),
    _col("worst_leg_drive", "Worst leg drive", "code"),
    _col("worst_leg_ratio", "Worst leg drive: largest RMS over its sustained rating, measured", "measured", "ratio"),
    _col("ratio_limit", "Limit: at most (ratio of the sustained rating)", "limit"),
    _col("drive_span", "Span the drives are judged over"), _col("attribution", "Attribution label"),
    _col("label_2", "Failure label, reading 2"), _col("label_3", "Failure label, reading 3"), _col("path", "Record", "path")]
TOE_COLUMNS = [
    _col("toe", "Toe clearance, the judged term `toe_clearance_own_swing` (each episode's lowest swing, read over the "
                "planned swing from the own lift-off): median, measured", "measured", "mm"),
    _col("toe_limit", f"Limit: at least (mm; {readings.TOE_CLEARANCE_FRAC_H:g} H)", "limit"),
    _col("toe_first", "Toe clearance as first written, `toe_clearance_own_duration_m` (over the swing's own "
                      "duration), reported and never judged: median, measured", "measured", "mm"),
    _col("toe_stage1", "Toe clearance, Stage 1's reading (on the reference's clock), reported and never judged: "
                       "median, measured", "measured", "mm")]
WALK_COLUMNS = HEAD + [
    _col("speed", "Speed over the judged 10 s: median, measured", "measured", "m/s"),
    _col("along", "Distance along the commanded heading over the judged 10 s: median, measured", "measured", "m"),
    _col("along_limit", "Limit: at least (m)", "limit"),
    _col("lateral", "Largest distance from the line: median, measured", "measured", "m"),
    _col("lateral_limit", "Limit: at most (m)", "limit")] + TOE_COLUMNS + [_col("path", "Record", "path")]
TURN_COLUMNS = HEAD + [
    _col("at_defined", "The turn's time after its command, as defined (s)", "defined"),
    _col("heading_at", "Heading error at the turn's time: median, measured", "measured", "deg"),
    _col("heading_hold", "Largest heading error over the hold: median, measured", "measured", "deg"),
    _col("heading_limit", "Limit: within (deg)", "limit"),
    _col("within", "Time to within 10 deg: median, measured", "measured", "s"),
    _col("turned_own", "Turned from the trunk's own heading at the command, reported: median, measured", "measured",
         "deg")] + TOE_COLUMNS + [_col("path", "Record", "path")]
NUDGE_COLUMNS = HEAD + [
    _col("force_defined", "Pulse for 0.1 s, as defined (N)", "defined"),
    _col("recovered", "Recovered (back on the command and held)", "of"),
    _col("back", "Time back on the command after the pulse's end: median, measured", "measured", "s"),
    _col("back_limit", "Limit: at most (s)", "limit"), _col("path", "Record", "path")]
STOP_COLUMNS = HEAD + [
    _col("convention", "Command instant read by"), _col("stopped", "Stopped by the end", "of"),
    _col("touchdowns", "Touchdowns after the command (number of touchdowns: episodes)", "terms"),
    _col("touchdowns_limit", "Limit: at most (touchdowns)", "limit"),
    _col("stopped_s", "Stopped time after the command: median, measured", "measured", "s"),
    _col("stopped_strides", "Stopped time in commanded strides: median, measured", "measured", "strides"),
    _col("accel", "Forward acceleration at 1 Hz over two strides: median, measured", "measured", "m/s^2"),
    _col("accel_limit", "Limit: at most (m/s^2)", "limit"),
    _col("decel_raw", "Unfiltered deceleration peak, reported: median, measured", "measured", "m/s^2"),
    _col("path", "Record", "path")]
VERDICT_COLUMNS = [
    _col("set", "Set", "code"), _col("labels", "Labels", "labels"), _col("item", "Item"),
    _col("condition", "Condition", "code"), _col("grip", "Grip"), _col("delay", "Servo delay"),
    _col("controller", "Controller (its two rows)"), _col("which", "Read from"),
    _col("reading_1", "Reading 1 (with the look)"), _col("reading_2", "Reading 2 (every limit)", "word"),
    _col("reading_3", "Reading 3 (every limit but for A1)", "word"), _col("paths", "Records", "path")]
MEASURE_COLUMNS = {"bar_walk": WALK_COLUMNS, "bar_turn": TURN_COLUMNS, "bar_nudge": NUDGE_COLUMNS,
                   "bar_stop": STOP_COLUMNS}
MEASURE_TITLES = {"bar_walk": "measured values of the walk beside their limits",
                  "bar_turn": "measured values of the turns beside their limits",
                  "bar_nudge": "the nudges: recovered counts and measured times",
                  "bar_stop": "the stops: touchdown counts and measured times"}


def delay_word(ms: int | None) -> str:
    if ms is None:
        return "no delay named"
    return f"{ms} ms ({'judged' if ms == bar.JUDGED_MS else 'reported'})"


def _delays(found) -> list[int]:
    """Delays in the tables' order: the judged delay first, the others ascending."""
    ds = sorted({int(d) for d in found if d is not None})
    return [d for d in ds if d == bar.JUDGED_MS] + [d for d in ds if d != bar.JUDGED_MS]


def _set_order(doc: dict, wanted) -> list[str]:
    """The sets a table lists: ``wanted`` (the registered order's) and every set with records, the registered ones
    first in ``SETS``' order, each followed by its versions; a ``sets`` filter of ``load`` keeps those only."""
    names = set(wanted) | {s for s in doc if s != META}
    keep = doc[META].get("sets")
    if keep is not None:
        names = {s for s in names if s in keep}

    def rank(s):
        base = bar.registered_name(s)
        return (SETS.index(base), s != base, s) if base in SETS else (len(SETS), False, s)

    return sorted(names, key=rank)


def _bodies(doc: dict, set_name: str) -> list[str]:
    found = sorted(doc.get(set_name) or {})
    if found:
        return found
    return [doc[META].get("body") or PLAN_BODY]


def _found(doc: dict, set_name: str, body: str, block: str, keys) -> dict:
    """{(key, ms): {row name before its delay}} of the records of one set, body and block among ``keys``."""
    out: dict = {}
    for row, by_key in ((doc.get(set_name) or {}).get(body) or {}).get(block, {}).items():
        base, ms = _split_row(row)
        for key in by_key:
            if keys is None or key in keys:
                out.setdefault((key, ms), set()).add(base)
    return out


def _row_order(bases) -> list[str]:
    known = list(R2_ROWS) + [R0_ROW]
    return [r for r in known if r in bases] + sorted(r for r in bases if r not in known)


def _cells(doc: dict, sets, keys, plan_rows) -> list[dict]:
    """The cells of one table: for each set and body, every (condition, delay) of the registered order
    (``plan_rows``: ``{(set, key, ms): [rows]}``) and of the records found among ``keys``, in the tables' order,
    each with its rows."""
    order = {k: i for i, k in enumerate(_conds())}
    out = []
    for s in _set_order(doc, sets):
        base = bar.registered_name(s) or s
        for body in _bodies(doc, s):
            found = _found(doc, s, body, bar.HEADLINE_BLOCK, keys)
            want: dict = {}
            if body == PLAN_BODY and s == base:
                for (ps, key, ms), rows in plan_rows.items():
                    if ps == s:
                        want[(key, ms)] = set(rows)
            for cell, rows in found.items():
                if cell[1] is not None:  # a row without a delay in its name is none of the bar's rows
                    want.setdefault(cell, set()).update(rows)
            for ms in _delays(m for _, m in want):
                for key in sorted({k for k, m in want if m == ms}, key=lambda k: (order.get(k, len(order)), k)):
                    out.append(dict(set=s, body=body, key=key, ms=ms, rows=_row_order(want[(key, ms)]),
                                    any_record=(key, ms) in found))
    return out


def _head(set_name: str, key: str, ms: int | None, row: str, rec: dict | None, meta: dict) -> dict:
    d = (rec or {}).get("definition") or _conds().get(key) or {}
    return dict(set=set_name, labels=labels_of(set_name, rec, meta), item=item_of(d) if d else key, condition=key,
                grip=grip_of(d) if d else None, delay=delay_word(ms), row=row)


def _section(doc: dict, sets, keys, plan_rows, stage1: bool) -> dict:
    """The parts every record table shares: counts and passes; falls, failed terms and drives; measured values per
    kind of item; the controllers' verdicts.  The cells of one set and delay that have no record at all are one line
    per absent word."""
    meta = doc[META]
    counts, fails, verdicts = [], [], []
    measured: dict = {j: [] for j in MEASURE_COLUMNS}
    groups: dict = {}
    for c in _cells(doc, sets, keys, plan_rows):
        groups.setdefault((c["set"], c["body"], c["ms"]), []).append(c)
    for (s, body, ms), group in groups.items():
        if not any(c["any_record"] for c in group):
            by_word: dict = {}
            for c in group:
                for r in c["rows"]:
                    w = absent_word(meta, s, body, bar.HEADLINE_BLOCK, f"{r}@{ms}ms", c["key"])
                    by_word.setdefault(w, []).append((c["key"], r))
            for word, pairs in by_word.items():
                keys_ = list(dict.fromkeys(k for k, _ in pairs))
                rows = _row_order({r for _, r in pairs})
                first = _rel_run(meta, _run_path(s, body, bar.HEADLINE_BLOCK, f"{pairs[0][1]}@{ms}ms", pairs[0][0]))
                line = dict(set=s, labels=labels_of(s), item=f"{len(keys_)} conditions, {len(pairs)} records: none found",
                            condition=", ".join(keys_), grip=None, delay=delay_word(ms),
                            row=", ".join(f"{r}@{ms}ms" for r in rows), status=word, reading_1=READING_1,
                            path=[first] + ([f"and {len(pairs) - 1} more paths of the same form"]
                                            if len(pairs) > 1 else []))
                counts.append(line)
                if any(r in R2_ROWS for r in rows):
                    verdicts.append(dict(set=s, labels=labels_of(s), item=line["item"], condition=line["condition"],
                                         grip=None, delay=delay_word(ms), controller="R2 (R2/42, R2/43)",
                                         which="no record", reading_1=READING_1, reading_2=word, reading_3=word,
                                         paths=None))
            continue
        for c in group:
            key = c["key"]
            for base in c["rows"]:
                row = f"{base}@{ms}ms"
                rec = _rec(doc, s, body, bar.HEADLINE_BLOCK, row, key)
                if rec is None:
                    word = absent_word(meta, s, body, bar.HEADLINE_BLOCK, row, key)
                    counts.append(dict(_head(s, key, ms, row, None, meta), status=word, reading_1=READING_1,
                                       path=_rel_run(meta, _run_path(s, body, bar.HEADLINE_BLOCK, row, key))))
                    continue
                reg = registered(rec)
                for i, r in enumerate([rec] if reg is rec else [rec, reg]):
                    d = digest(r, meta)
                    head = _head(s, key, ms, row, r, meta)
                    if i == 1:  # the registered record stands beside its re-run
                        head["labels"] = labels_of(s) + [REGISTERED + (
                            "; its kept file cannot be read: counts from the later record's supersedes field"
                            if r.get("_brief") else "")]
                    counts.append(dict(head, **{k["key"]: d.get(k["key"]) for k in COUNT_COLUMNS[len(HEAD):]}))
                    if not d["done"] or r.get("_brief"):
                        continue
                    fails.append(dict(head, **{k["key"]: d.get(k["key"]) for k in FAIL_COLUMNS[len(HEAD):]}))
                    judge = _get(r, "definition", "judge")
                    if judge in measured:
                        m = dict(measures_of(r), path=d["path"])
                        measured[judge].append(
                            dict(head, **{k["key"]: m.get(k["key"]) for k in MEASURE_COLUMNS[judge][len(HEAD):]}))
            for ctrl in sorted({bar.controller_of(b) for b in c["rows"]} & set(panels.VERDICT_ROWS)):
                v = verdict_of(doc, s, body, key, ms, ctrl)
                d0 = _conds().get(key) or {}
                line = dict(set=s, labels=labels_of(s), item=item_of(d0) if d0 else key, condition=key,
                            grip=grip_of(d0) if d0 else None, delay=delay_word(ms),
                            controller=f"{ctrl} ({', '.join(panels.VERDICT_ROWS[ctrl])})", reading_1=READING_1)
                verdicts.append(dict(line, which="no record" if v["none"] else "registered records",
                                     reading_2=v["reading_2"]["text"],
                                     reading_3=v["reading_3"]["text"], paths=[p for p in v["paths"] if p]))
                if v["rerun"]:
                    w = verdict_of(doc, s, body, key, ms, ctrl, latest=True)
                    verdicts.append(dict(line, labels=labels_of(s) + [RERUN], which=RERUN,
                                         reading_2=w["reading_2"]["text"], reading_3=w["reading_3"]["text"],
                                         paths=[p for p in w["paths"] if p]))
    cols = [k for k in COUNT_COLUMNS if stage1 or k["key"] != "stage1"]
    parts = [dict(key="counts", title="counts, bounds and passes per record", columns=cols, lines=counts),
             dict(key="fails", title="falls, failed terms, drives and labels per record", columns=FAIL_COLUMNS,
                  lines=fails)]
    parts += [dict(key=j, title=MEASURE_TITLES[j], columns=MEASURE_COLUMNS[j], lines=measured[j])
              for j in MEASURE_COLUMNS if measured[j]]
    parts.append(dict(key="verdicts", title="the controller's verdict per condition and delay",
                      columns=VERDICT_COLUMNS, lines=verdicts, notes=verdict_notes()))
    return dict(parts=parts)


def _rel_run(meta: dict, run_path: str) -> str:
    """A path given relative to the run directory, as the tables cite it."""
    return (Path(meta["out"]) / run_path).as_posix()


def verdict_notes() -> list[str]:
    """What stands under every table that prints a verdict: the words' rule and the rules amended after a result."""
    return [
        "Verdict words (STAGE2_75_PLAN.md H2): at the judged delay (10 ms) `shown` if both R2 rows pass and "
        "`provisional` if one does, in reading 3 with `but for A1`; at another delay the cell reads `passes at <ms> "
        "ms` and is no verdict.  An R0 row gives no verdict and is not replicated.  A candidate's `shown` or "
        "`provisional` stands with the replication word of each row's record that passed.",
        "Rules amended after a result, with the plan's own labels (" + PLAN_FILE + "): "
        + "  ".join(f"{a['rule']} {a['label']} {a['text']}" for a in AMENDED_AFTER_RESULT),
    ]


# ------------------------------------------------------------------------------------------------
# The tables
# ------------------------------------------------------------------------------------------------
def _plan_rows(sets, keys, delays, rows) -> dict:
    return {(s, k, ms): list(rows) for s in sets for k in keys for ms in delays}


def walk_table(doc: dict) -> dict:
    """Table 1: the walk at 0.40 m/s (task 3)."""
    keys = _group("@walk")
    sec = _section(doc, WALK_SETS, set(keys), _plan_rows(WALK_SETS, keys, WALK_DELAYS, R2_ROWS + (R0_ROW,)), True)
    return dict(title="1. The walk at 0.40 m/s (task 3)", notes=[
        "Per set, grip and servo delay: the two fitted controllers' rows (R2/42, R2/43) and the open-loop row (R0).  "
        "The judged delay (10 ms) stands first; every other delay is reported.",
        "Stage 1's reading is `measures.stage1.succeeded` of the same episodes; it gives no verdict."], **sec)


def ladder_table(doc: dict) -> dict:
    """Table 2: the speed ladder (reported) and the highest rung that passes under both rows."""
    walk = set(_group("@walk"))
    keys = [k for k, c in _conds().items() if c["behavior"] == "walk" and k not in walk]
    plan_rows = _plan_rows(LADDER_SETS, _group("@speed-ladder"), LADDER_DELAYS, R2_ROWS)
    for s, k, ms in PR_ROWS:
        plan_rows[(s, k, ms)] = list(R2_ROWS)
    sec = _section(doc, LADDER_SETS, set(keys), plan_rows, True)
    lines = []
    rungs = [k for k, c in _conds().items() if c["behavior"] == "walk"]
    for s in _set_order(doc, LADDER_SETS):
        for body in _bodies(doc, s):
            found = _found(doc, s, body, bar.HEADLINE_BLOCK, set(rungs))
            delays = {m for _, m in found} | ({ms for (ps, _, ms) in plan_rows if ps == s} if body == PLAN_BODY else set())
            for grip in ("mu 0.9", "mu 0.54"):
                ks = sorted((k for k in rungs if grip_of(_conds()[k]) == grip),
                            key=lambda k: setio.parse_name(_conds()[k]["walk"]).get("speed_m_s") or 0.0)
                for ms in _delays(delays):
                    if grip != "mu 0.9" and not any((k, ms) in found for k in ks):
                        continue
                    line = dict(set=s, labels=labels_of(s), grip=grip, delay=delay_word(ms))
                    for rkey in ("reading_2", "reading_3"):
                        per, best = [], None
                        for k in ks:
                            v = verdict_of(doc, s, body, k, ms)[rkey]
                            per.append(f"{k}: {v['text']}")
                            if v["verdict"].startswith(("shown", "passes at")):
                                best = k
                        line[rkey + "_highest"] = ("none" if best is None else
                                                   f"{best} ({item_of(_conds()[best])})")
                        line[rkey + "_rungs"] = per
                    lines.append(line)
    what = "Highest rung at which both R2 rows pass (at 10 ms: shown)"
    sec["parts"].append(dict(
        key="highest", title="the highest rung that passes under both R2 rows, per set, grip and delay",
        columns=[_col("set", "Set", "code"), _col("labels", "Labels", "labels"), _col("grip", "Grip"),
                 _col("delay", "Servo delay"), _col("reading_2_highest", what + ", reading 2 (every limit)", "word"),
                 _col("reading_2_rungs", "Each rung, reading 2", "list"),
                 _col("reading_3_highest", what + ", reading 3 (every limit but for A1)", "word"),
                 _col("reading_3_rungs", "Each rung, reading 3", "list")],
        lines=lines, notes=["The rungs in ascending order: the nominal walk, then 0.3 to 0.5 m/s; the 0.40 m/s rung's "
                            "records are table 1's.  \"none\" says no rung passes under both rows, whether its records "
                            "fail, are not run or are cut: each rung's own cell says which."] + verdict_notes()))
    return dict(title="2. The speed ladder (reported)", notes=[
        "Reported conditions: the walks at the nominal speed and at 0.30 to 0.50 m/s other than the bar's 0.40 m/s, "
        "under both R2 rows, and the pull request's remaining rows (STAGE2_75_PLAN.md H6, task 3, tiers 6, 7 and 9).",
        "A rung's `along` limit is 10 s x its own target speed; `walk/nominal` is `start`, then `nominal` to the "
        "end."], **sec)


def rest_table(doc: dict) -> dict:
    """Table 3: the rest of the bar (task 4): the turns, the nudges and the stop, and their reported ladders."""
    keys = [k for k, c in _conds().items() if c["behavior"] != "walk"]
    plan_rows = _plan_rows(REST_SETS, _group("@rest"), REST_DELAYS, R2_ROWS + (R0_ROW,))
    plan_rows.update(_plan_rows(REST_SETS, _group("@ladders"), REST_LADDER_DELAYS, R2_ROWS))
    sec = _section(doc, REST_SETS, set(keys), plan_rows, False)
    return dict(title="3. The rest of the bar (task 4): turns, nudges and the stop", notes=[
        "The 14 gated conditions (the quick turn each way, a nudge from each of four sides, the two-touchdown stop, "
        "on both grips) and the reported ladders (turn rates, nudges at 0.375 and 0.5 x the push bound, Stage 1's "
        "one-step stop).  Every item starts with the walk's lead-in and takes the walk's seed slot: a fall before the "
        "command is a fall of the lead-in and shows in every item of that seed.",
        "A turn without a reference is \"not run\" with its reason and the reference's own numbers: that is no "
        "result of the robot.  It goes to the owner and has no fallback (STAGE2_75_PLAN.md A11).",
        "`stop/one_step` plays Stage 1's non-causal stop and is read from Stage 1's command; its lines say so."],
        **sec)


def glance_table(doc: dict) -> dict:
    """Table 4: the bar at a glance: per set and body the 16 gated conditions at the judged delay."""
    ms = bar.JUDGED_MS
    lines, totals = [], []
    for s in _set_order(doc, SETS):
        for body in _bodies(doc, s):
            tally = {"reading_2": Counter(), "reading_3": Counter()}
            for key in _group("@gated"):
                d = _conds()[key]
                v = verdict_of(doc, s, body, key, ms)
                r0 = registered(_rec(doc, s, body, bar.HEADLINE_BLOCK, f"{R0_ROW}@{ms}ms", key))
                note = None
                if bar.registered_name(s) == "F25F9" and d["behavior"] != "walk" and v["reading_2"]["verdict"] == NOT_RUN:
                    note = WALK_ONLY_NOTE
                r0d = digest(r0) if r0 is not None and r0.get("status") == "done" else None
                lines.append(dict(
                    set=s, body=body, labels=labels_of(s), item=item_of(d), condition=key, grip=grip_of(d),
                    reading_1=READING_1, reading_2=v["reading_2"]["text"], reading_3=v["reading_3"]["text"],
                    r0=NO_VERDICT_R0 if r0d else (status_of(r0) if r0 is not None else absent_word(
                        doc[META], s, body, bar.HEADLINE_BLOCK, f"{R0_ROW}@{ms}ms", key)),
                    r0_up=None if r0d is None else r0d["up"], r0_r2=None if r0d is None else r0d["r2"],
                    r0_r3=None if r0d is None else r0d["r3"], note=note, paths=[p for p in v["paths"] if p]))
                for rkey in tally:
                    tally[rkey][v[rkey]["verdict"]] += 1
            totals.append(dict(set=s, body=body, labels=labels_of(s),
                               reading_2={k: tally["reading_2"][k] for k in sorted(tally["reading_2"])},
                               reading_3={k: tally["reading_3"][k] for k in sorted(tally["reading_3"])}))
    cols = [_col("set", "Set", "code"), _col("body", "Body", "code"), _col("labels", "Labels", "labels"),
            _col("item", "Item"), _col("condition", "Condition", "code"), _col("grip", "Grip"),
            _col("reading_1", "Reading 1 (with the look)"),
            _col("reading_2", "Reading 2 (every limit): R2's verdict at 10 ms", "word"),
            _col("reading_3", "Reading 3 (every limit but for A1): R2's verdict at 10 ms", "word"),
            _col("r0", "The R0 row at 10 ms"), _col("r0_up", "R0: stayed up", "of"),
            _col("r0_r2", "R0: succeeded, reading 2", "of"), _col("r0_r3", "R0: succeeded, reading 3", "of"),
            _col("note", "Note"), _col("paths", "R2 records", "path")]
    tcols = [_col("set", "Set", "code"), _col("body", "Body", "code"), _col("labels", "Labels", "labels"),
             _col("reading_2", "Reading 2: gated conditions per verdict word (word: conditions, of 16)", "terms"),
             _col("reading_3", "Reading 3: gated conditions per verdict word (word: conditions, of 16)", "terms")]
    return dict(title="4. The bar at a glance: the 16 gated conditions at 10 ms", notes=[
        "Per set and body, the verdict of the controller R2 (rows R2/42@10ms and R2/43@10ms) on each gated "
        "condition, in each reading.  A condition without a reference reads \"not run\" with its reason; it is not "
        "a pass."],
        parts=[dict(key="glance", title="verdict per gated condition", columns=cols, lines=lines,
                    notes=verdict_notes()),
               dict(key="tally", title="conditions per verdict word", columns=tcols, lines=totals)])


def replication_table(doc: dict) -> dict:
    """Table 5: every record of the replication block beside the headline record it replicates."""
    meta = doc[META]
    lines = []
    seen = set()

    def side(rec):
        if rec is None:
            return dict(up=None, r2=None, r2_pass=None, r3=None, r3_pass=None, path=None)
        d = digest(rec, meta)
        return dict(up=d["up"], r2=d["r2"], r2_pass=d["r2_pass"] if d["done"] else d["status"], r3=d["r3"],
                    r3_pass=d["r3_pass"] if d["done"] else d["status"], path=d["path"])

    def add(s, body, row, key, first, second, rerun=False):
        rep = (second or {}).get("replication_of") or {}
        trig = rep.get("readings")
        if trig is None and first is not None:
            trig = [w for w, ck, _ in READINGS if (first.get(ck) or {}).get("pass_") is True]
        names = {w: n for w, _, n in READINGS}
        a, b = side(first), side(second)
        d = (first or second or {}).get("definition") or _conds().get(key) or {}
        labels = labels_of(s, second, meta) if second else labels_of(s)
        if second is not None and second.get("_registered_of"):
            labels = labels_of(s) + [REGISTERED]
        lines.append(dict(
            set=s, labels=labels, item=item_of(d) if d else key, condition=key, grip=grip_of(d) if d else None,
            delay=delay_word(bar.delay_ms(row)), row=row,
            triggered_by=[names[w] for w in trig or []] or [
                "no seeds3042 record found" if first is None else "no reading of the seeds3042 record passes"],
            a_up=a["up"], a_r2=a["r2"], a_r2_pass=a["r2_pass"], a_r3=a["r3"], a_r3_pass=a["r3_pass"],
            b_up=b["up"], b_r2=b["r2"], b_r2_pass=b["r2_pass"], b_r3=b["r3"], b_r3_pass=b["r3_pass"],
            words=[RERUN + ": the registered record's word stands"] if rerun else
            [f"{names[w]}: {replication_word(doc, s, body, row, key, w)}" for w in trig or []],
            a_path=a["path"], b_path=b["path"] or _rel_run(meta, _run_path(s, body, bar.REPLICATION_BLOCK, row, key))))

    for s in _set_order(doc, ()):
        for body in sorted(doc.get(s) or {}):
            blocks = doc[s][body]
            for row in sorted(blocks.get(bar.REPLICATION_BLOCK) or {}):
                for key in sorted(blocks[bar.REPLICATION_BLOCK][row]):
                    seen.add((s, body, row, key))
                    first = registered(_rec(doc, s, body, bar.HEADLINE_BLOCK, row, key))
                    second = blocks[bar.REPLICATION_BLOCK][row][key]
                    add(s, body, row, key, first, registered(second))
                    if registered(second) is not second:  # a re-run of the second block stands beside, labelled
                        add(s, body, row, key, first, second, rerun=True)
            base = bar.registered_name(s)  # the candidate, or a set outside the round that has a second block
            if base not in (None, bar.CANDIDATE_SET) or (base is None and bar.REPLICATION_BLOCK not in blocks):
                continue
            for row in sorted(blocks.get(bar.HEADLINE_BLOCK) or {}):
                if bar.controller_of(row) in bar.NO_VERDICT_CONTROLLERS or bar.delay_ms(row) is None:
                    continue
                for key in sorted(blocks[bar.HEADLINE_BLOCK][row]):
                    first = registered(blocks[bar.HEADLINE_BLOCK][row][key])
                    passed = any((first.get(ck) or {}).get("pass_") is True for _, ck, _ in READINGS)
                    if passed and (s, body, row, key) not in seen:
                        add(s, body, row, key, first, None)
    cols = [_col("set", "Set", "code"), _col("labels", "Labels", "labels"), _col("item", "Item"),
            _col("condition", "Condition", "code"), _col("grip", "Grip"), _col("delay", "Servo delay"),
            _col("row", "Row", "code"), _col("triggered_by", "Reading that triggered the replication", "list"),
            _col("a_up", "seeds3042: stayed up", "of"), _col("a_r2", "seeds3042: succeeded, reading 2", "of"),
            _col("a_r2_pass", "seeds3042: reading 2", "word"), _col("a_r3", "seeds3042: succeeded, reading 3", "of"),
            _col("a_r3_pass", "seeds3042: reading 3", "word"),
            _col("b_up", "seeds5042: stayed up", "of"), _col("b_r2", "seeds5042: succeeded, reading 2", "of"),
            _col("b_r2_pass", "seeds5042: reading 2", "word"), _col("b_r3", "seeds5042: succeeded, reading 3", "of"),
            _col("b_r3_pass", "seeds5042: reading 3", "word"),
            _col("words", "Replication, per triggering reading", "list"),
            _col("a_path", "seeds3042 record", "path"), _col("b_path", "seeds5042 record", "path")]
    return dict(title="5. Replication on fresh seeds (seeds 5042-5061)", notes=[
        "Every record of the block `seeds5042` beside the `seeds3042` record it replicates.  The two blocks are "
        "never pooled and never averaged: each stands with its own 20 seeds, and the headline stays `seeds3042`.",
        "Only the candidate is replicated, under the row that passed, in the reading that triggered it; a control and "
        "an R0 row are not (STAGE2_75_PLAN.md H4).  A passing record of the candidate with no `seeds5042` record is "
        "listed with its replication word.",
        bar.RULES["replication"]],
        parts=[dict(key="replication", title="the two blocks side by side", columns=cols, lines=lines,
                    notes=verdict_notes())])


def controls_table(doc: dict) -> dict:
    """The reported controls (``controls/<control>/``): each record beside the registered record of the same set,
    row, condition and delay.  A control gives no verdict."""
    meta = doc[META]
    lines = []
    cells = []
    control, s, key, base, delays = OMEGA_CONTROL
    for ms in delays:
        cells.append((control, s, PLAN_BODY, bar.HEADLINE_BLOCK, f"{base}@{ms}ms", key))
    for c in sorted(meta["controls"]):
        for rec in _records(meta["controls"][c]):
            cell = (c, rec["set"], rec["body"], rec["block"], rec["row"], rec["key_name"])
            if cell not in cells:
                cells.append(cell)
    keep = meta.get("sets")
    for c, s, body, block, row, key in cells:
        if keep is not None and s not in keep:
            continue
        rec = _rec(meta["controls"].get(c) or {}, s, body, block, row, key)
        ref = registered(_rec(doc, s, body, block, row, key))
        d = digest(rec, meta) if rec else None
        r = digest(ref, meta) if ref else None
        d0 = (rec or ref or {}).get("definition") or _conds().get(key) or {}
        lines.append(dict(
            control=((rec.get("control") or f"{c}: the record holds no control field") if rec
                     else bar.CONTROLS[c][1] if c in bar.CONTROLS else c), set=s, labels=labels_of(s, rec, meta),
            item=item_of(d0), condition=key, grip=grip_of(d0), delay=delay_word(bar.delay_ms(row)), row=row,
            status=d["status"] if d else absent_word(meta, s, body, block, row, key, control=c, in_order=True),
            omega=d["omega"] if d else None, crouch=d["crouch_frac"] if d else None, up=d["up"] if d else None,
            r2=d["r2"] if d else None, r3=d["r3"] if d else None, verdict="no verdict (a control)",
            ref_omega=r["omega"] if r else None, ref_crouch=r["crouch_frac"] if r else None,
            ref_up=r["up"] if r else None, ref_r2=r["r2"] if r else None, ref_r3=r["r3"] if r else None,
            path=d["path"] if d else _rel_run(meta, _run_path(s, body, block, row, key, c)),
            ref_path=r["path"] if r else None))
    cols = [_col("control", "Control"), _col("set", "Set", "code"), _col("labels", "Labels", "labels"),
            _col("item", "Item"), _col("condition", "Condition", "code"), _col("grip", "Grip"),
            _col("delay", "Servo delay"), _col("row", "Row", "code"), _col("status", "Status"),
            _col("omega", "Control: omega as run (rad/s)", "defined", "rad/s"),
            _col("crouch", "Control: crouch in omega (H)", "defined"), _col("up", "Control: stayed up", "of"),
            _col("r2", "Control: succeeded, reading 2", "of"), _col("r3", "Control: succeeded, reading 3", "of"),
            _col("verdict", "Verdict"),
            _col("ref_omega", "Registered record: omega as run (rad/s)", "defined", "rad/s"),
            _col("ref_crouch", "Registered record: crouch in omega (H)", "defined"),
            _col("ref_up", "Registered record: stayed up", "of"),
            _col("ref_r2", "Registered record: succeeded, reading 2", "of"),
            _col("ref_r3", "Registered record: succeeded, reading 3", "of"),
            _col("path", "Control record", "path"), _col("ref_path", "Registered record", "path")]
    return dict(title="7. Reported control: omega at Stage 1's 0.03 H crouch", notes=[
        "The candidate's walk at 0.40 m/s, mu 0.9, under R2/42 at 10 ms and at 0 ms with the track's pendulum "
        "frequency computed with Stage 1's 0.03 H crouch instead of the set's own (STAGE2_75_PLAN.md B5): 40 "
        "episodes on seeds 3042-3061.  It gives no verdict and is neither replicated nor attributed; its records "
        "live under `controls/omega_s1/`."],
        parts=[dict(key="controls", title="the control beside the registered record", columns=cols, lines=lines)])


# ------------------------------------------------------------------------------------------------
# The comparison with JPS section 2.3
# ------------------------------------------------------------------------------------------------
def _seed_flags(rec: dict | None) -> dict:
    """{seed: {up, stage1, r2, r3}} of a record's episodes ("failed to run" for an episode that is no result)."""
    out = {}
    for e in (rec or {}).get("episodes") or []:
        if e.get("error"):
            out[int(e["seed"])] = dict(up="failed to run", stage1="failed to run", r2="failed to run",
                                       r3="failed to run")
            continue
        s1 = _get(e, "measures", "stage1")
        out[int(e["seed"])] = dict(
            up=not e.get("fell"), stage1=bool(s1.get("succeeded")) if isinstance(s1, dict) else "not defined",
            r2=bool(e.get("succeeded")), r3=bool(e["succeeded_but_A1"]) if "succeeded_but_A1" in e else "not defined")
    return out


def jps_table(doc: dict) -> dict:
    """Table 6: one row per row of JPS section 2.3 with its claim (``JPS_ROWS``) beside this round's records
    (``JPS_MAP``), then seed by seed (the headline block's seeds) under R2/42 and R2/43: stayed up, Stage 1's
    reading, reading 2, reading 3."""
    meta = doc[META]
    keep = meta.get("sets")
    body = meta.get("body") or PLAN_BODY
    block = bar.HEADLINE_BLOCK
    rows, seeds_parts = [], []
    for claim in JPS_ROWS:
        matched, lines, paths = [], [], []
        for cell in JPS_MAP[claim["line"]]:
            if isinstance(cell, str):
                matched.append(cell)
                lines.append(f"{cell}: {NOT_RUN}")
                continue
            s, key, ms = cell
            d0 = _conds()[key]
            name = f"{s} {key} at {ms} ms"
            v = setio.parse_name(d0["walk"]).get("speed_m_s") if d0["behavior"] == "walk" else None
            matched.append(f"{name} ({item_of(d0)}, {grip_of(d0)}; rows {R2_ROWS[0]}@{ms}ms and {R2_ROWS[1]}@{ms}ms)"
                           + ("" if v is None else f"; the rung's target speed {v:g} m/s"))
            if keep is not None and s not in keep:
                lines.append(f"{name}: the set is not among the sets asked for")
                continue
            recs = {}
            for r in R2_ROWS:
                row = f"{r}@{ms}ms"
                rec = registered(_rec(doc, s, body, block, row, key))
                recs[row] = rec
                if rec is None:
                    lines.append(f"{name}, {row}: {absent_word(meta, s, body, block, row, key)}")
                    continue
                d = digest(rec, meta)
                paths.append(d["path"])
                if not d["done"]:
                    lines.append(f"{name}, {row}: {d['status']}")
                    continue
                m = measures_of(rec)
                text = (f"{name}, {row}: stayed up {_of(d['up'])}; Stage 1's reading "
                        + (_of(d["stage1"]) if d["stage1"] else "not defined for this item")
                        + f"; reading 2 {_of(d['r2'])} ({d['r2_pass']}); reading 3 "
                        + (f"{_of(d['r3'])} ({d['r3_pass']})" if d["r3"] else "not defined"))
                if m.get("speed"):
                    text += "; median speed measured " + _median_text(m["speed"], "m/s") + " m/s"
                if m.get("stopped_s"):
                    text += ("; median stopped time measured " + _median_text(m["stopped_s"], "s") + " s, "
                             + _median_text(m["stopped_strides"], "strides") + " commanded strides")
                if m.get("recovered"):
                    text += f"; recovered {_of(m['recovered'])}"
                if d["labels"]:
                    text += " [" + "; ".join(d["labels"]) + "]"
                lines.append(text)
            if any(r is not None for r in recs.values()):
                v = verdict_of(doc, s, body, key, ms)
                lines.append(f"{name}, the controller R2 from its two rows: reading 2: {v['reading_2']['text']}; "
                             f"reading 3: {v['reading_3']['text']}")
            if any(r is not None and r.get("status") == "done" for r in recs.values()):
                seeds_parts.append(_seed_part(claim["line"], name, s, recs))
        rows.append(dict(line=claim["line"], **{f"jps_claim_{k}": claim[k] for k in JPS_CLAIM_KEYS},
                         matched=matched, record=lines, paths=paths))
    cols = [_col("line", "JPS line", "count")] + [
        _col(f"jps_claim_{k}", f"JPS's claim: {JPS_HEADER[k]}", "claim") for k in JPS_CLAIM_KEYS] + [
        _col("matched", "This round's matching condition, rows and delay", "list"),
        _col("record", "This round's record (registered records; counts of the block's episodes; measured values)",
             "list"),
        _col("paths", "Records", "path")]
    return dict(title="6. The comparison with JPS section 2.3", notes=list(JPS_STATEMENTS) + [
        f"JPS's claims are copied from `{JPS_FILE}`, lines {JPS_ROWS[0]['line']} to {JPS_ROWS[-1]['line']} (emphasis "
        "marks removed).  A cell with no record reads \"not run\", or \"cut by the pre-registered order\" where "
        "`budget.json` shows its unit was not started."],
        parts=[dict(key="jps", title="JPS's claim beside this round's record", columns=cols, lines=rows,
                    notes=verdict_notes())] + seeds_parts)


def _of(v: dict | None) -> str:
    return "-" if v is None else f"{v['k']} of {v['n']}"


def _median_text(m: dict, unit: str) -> str:
    return "-" if m is None or m.get("median") is None else f"{_num(m['median'], unit)} ({m['n']})"


def _seed_part(line: int, name: str, set_name: str, recs: dict) -> dict:
    """One matched cell seed by seed: a line per seed of the records, the two rows side by side."""
    flags = {row: _seed_flags(rec) for row, rec in recs.items()}
    seeds = sorted({s for f in flags.values() for s in f})
    names = list(recs)
    cols = [_col("seed", "Seed", "count")]
    for i, row in enumerate(names):
        for k, title in (("up", "stayed up"), ("stage1", "Stage 1's reading"), ("r2", "reading 2"),
                         ("r3", "reading 3")):
            cols.append(_col(f"{k}_{i}", f"{row}: {title}", "flag"))
    lines = []
    for s in seeds:
        x = dict(seed=s)
        for i, row in enumerate(names):
            for k in ("up", "stage1", "r2", "r3"):
                x[f"{k}_{i}"] = (flags[row].get(s) or {}).get(k)
        lines.append(x)
    def tally(vals):
        if not vals:
            return None
        if len(set(map(str, vals))) == 1 and not isinstance(vals[0], bool):
            return str(vals[0])  # "not defined" or "failed to run" in every episode
        return f"{sum(1 for v in vals if v is True)} of {len(vals)}"

    total = dict(seed="episodes with yes, of those listed")
    for i, row in enumerate(names):
        for k in ("up", "stage1", "r2", "r3"):
            total[f"{k}_{i}"] = tally([f[k] for f in flags[row].values()])
        if recs[row] is None:
            total[f"up_{i}"] = "no record"
    lines.append(total)
    return dict(key="seeds " + name, title=f"JPS line {line}: {name}, seed by seed", columns=cols, lines=lines,
                notes=[MARGIN_LABEL] if bar.registered_name(set_name) == bar.CANDIDATE_SET else [],
                paths=[r.get("_path") for r in recs.values() if r is not None and r.get("_path")])


# ------------------------------------------------------------------------------------------------
# The checks printed at the top
# ------------------------------------------------------------------------------------------------
def checks(doc: dict) -> dict:
    """Records found; records made by other sources than today's; records whose input hashes are not in their
    set's latest freeze; records with errors or unjudged counts; records whose counts do not recompute
    (``bar.counts``); core-hours per task against its budget, with the units started after the first cut; the rates of
    ``COST.md``."""
    meta = doc[META]
    main = list(_records(doc))
    ctl = [r for c in sorted(meta["controls"]) for r in _records(meta["controls"][c])]
    by_set: dict = {}
    for r in main:
        x = by_set.setdefault(r["set"], dict(done=0, not_run=0))
        x["done" if r.get("status") == "done" else "not_run"] += 1
    other, missing, unchecked, unjudged, miscounted, second = [], [], [], [], [], []
    for r in main + ctl:
        if r["block"] == bar.REPLICATION_BLOCK:  # H4: the candidate only, under a row with a verdict; no control
            why = [w for w, bad in (
                ("an R0 row", bar.controller_of(r["row"]) in bar.NO_VERDICT_CONTROLLERS),
                ("a control set of the round", bar.registered_name(r["set"]) not in (None, bar.CANDIDATE_SET)),
                ("a reported control", bool(r.get("control"))),
                ("no replication_of field", r.get("status") == "done" and not r.get("replication_of"))) if bad]
            if why:
                second.append(dict(path=r["_path"], why=why))
        diff = counts_differences(r)
        if diff:
            miscounted.append(dict(path=r["_path"], differences=diff))
        src = _get(r, "key", "sources") or {}
        differ = [k for k in ("panels", "bar") if src.get(k) != meta["today"].get(k)]
        if differ:
            other.append(dict(path=r["_path"], differs=differ, panels=src.get("panels"), bar=src.get("bar")))
        why = manifest_differences(r, meta)
        if why == [NOT_CHECKED]:
            unchecked.append(dict(path=r["_path"], set_dir=_get(r, "inputs", "dir")))
        elif why:
            missing.append(dict(path=r["_path"], set_dir=_get(r, "inputs", "dir"), differences=why))
        if r.get("status") == "done":
            cs = [c for c in (r.get("counts"), r.get("counts_but_A1")) if c is not None]
            if any(c.get("errors") for c in cs) or any(not c.get("judged") for c in cs):
                c2 = r.get("counts") or {}
                unjudged.append(dict(path=r["_path"], episodes=c2.get("episodes"), errors=c2.get("errors"),
                                     judged=bool(c2.get("judged"))))
    budget = None
    if meta.get("budget"):
        budget = {}
        for t, task in sorted((meta["budget"].get("tasks") or {}).items()):
            units = [dict(unit=u.get("unit"), estimate_core_h=u.get("estimate_core_h"),
                          spent_before_core_h=u.get("spent_before_core_h"), started_at=u.get("started_at"),
                          used_core_h=float(sum((u.get("items") or {}).values())) / 3600.0)
                     for u in task.get("units") or []]
            cut = [dict(unit=c.get("unit"), estimate_core_h=c.get("estimate_core_h"),
                        spent_before_core_h=c.get("spent_before_core_h"), at=c.get("at"), reason=c.get("reason"),
                        label=c.get("label")) for c in task.get("cut") or []]
            reg = bar.BUDGET_CORE_H.get(int(t)) if str(t).isdigit() else None
            first_cut = min((str(c["at"]) for c in cut if c.get("at")), default=None)
            late = [u["unit"] for u in units if first_cut is not None and str(u.get("started_at") or "") > first_cut]
            budget[str(t)] = dict(cap_core_h=task.get("cap_core_h"), registered_cap_core_h=reg,
                                  spent_core_h=float(sum(u["used_core_h"] for u in units)), units=units, cut=cut,
                                  first_cut_at=first_cut, started_after_the_first_cut=late)
    cost = None
    if meta.get("cost"):
        c = meta["cost"]
        cost = dict(measured_on=c.get("measured_on"), load=c.get("load"),
                    rates={k: dict(row=v.get("row"), wall_per_sim_s=v.get("wall_per_sim_s"),
                                   full_length=v.get("full_length"), episodes=len(v.get("episodes") or []),
                                   warm_up_counted=_get(v, "warm_up", "counted"))
                           for k, v in (c.get("rates") or {}).items()})
    return dict(
        title="Checks", run=meta["out"], sets_asked=meta.get("sets"), body_asked=meta.get("body"),
        records=dict(found=len(main), done=sum(v["done"] for v in by_set.values()),
                     not_run=sum(v["not_run"] for v in by_set.values()), by_set={k: by_set[k] for k in sorted(by_set)},
                     controls=len(ctl), kept_beside_a_rerun=sum(len(r.get("_superseded") or []) for r in main + ctl),
                     not_a_record=list(meta["unreadable"]), misplaced=list(meta["misplaced"])),
        today_sources=dict(meta["today"]), other_sources=other, not_in_latest_freeze=missing, not_checked=unchecked,
        freezes={d: {k: v for k, v in m.items() if k != "entries"} for d, m in sorted(meta["manifests"].items())},
        errors_or_unjudged=unjudged, counts_not_recomputed=miscounted, second_block_outside_the_rule=second,
        budget=budget, cost=cost)


def other_records(doc: dict, tabs: dict) -> dict:
    """Records under the run directory that no table above lists (a stock condition of the pseudo-set, a body or a
    block the tables do not read), so that none is dropped silently."""
    cited = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            cited.add(x)

    walk(tabs)
    lines = []
    for r in _records(doc):
        if r["_path"] in cited:
            continue
        d = digest(r, doc[META])
        lines.append({k: d[k] for k in ("set", "labels", "item", "condition", "grip", "row", "status", "episodes", "up",
                                         "r2", "r2_pass", "r3", "r3_pass", "path")} | dict(block=r["block"], body=r["body"]))
    cols = [_col("set", "Set", "code"), _col("body", "Body", "code"), _col("block", "Block", "code"),
            _col("labels", "Labels", "labels"), _col("item", "Item"), _col("condition", "Condition", "code"),
            _col("grip", "Grip"), _col("row", "Row", "code"), _col("status", "Status"),
            _col("episodes", "Episodes", "count"), _col("up", "Stayed up", "of"),
            _col("r2", "Reading 2 (every limit): succeeded", "of"), _col("r2_pass", "Reading 2: pass", "word"),
            _col("r3", "Reading 3 (every limit but for A1): succeeded", "of"), _col("r3_pass", "Reading 3: pass", "word"),
            _col("path", "Record", "path")]
    return dict(title="8. Records in no table above", notes=[
        "Every record found under the run directory that tables 1 to 7 do not cite: a Stage 1 condition of the "
        "pseudo-set `S1` (one reading), or a row without a delay.  They give no verdict here."],
        parts=[dict(key="other", title="records not cited above", columns=cols, lines=lines)])


def tables(doc: dict) -> dict:
    """The JSON form of every table (``bar_tables.json``): the checks, tables 1 to 8, the rule texts they rest on.
    A pure function of ``doc``; the value is plain JSON (what the file holds after a round trip)."""
    meta = doc[META]
    tabs = dict(
        schema=SCHEMA, made_by="designlab.stage2.barreport", run=meta["out"],
        simulation="Every number in these tables comes from simulation.",
        conventions=[
            "The tables are made by code from the bar's records (one JSON record per set, body, seed block, row and "
            "condition); each line cites its record, with paths relative to the study directory "
            "(hardware/trex_form_study).",
            "Readings stand in the plan's order and are never pooled: reading 1 (with the look) is "
            f"\"{READING_1}\" (the records say: {bar.LOOK_NOT_DEFINED}); reading 2 is every limit; reading 3 is every "
            "limit but for A1 (the sustained-drive term read on the leg drives only).",
            "Counts, bounds and passes are the records' own.  A bound is the one-sided 95 % Clopper-Pearson lower "
            "bound of the share of the block's 20 episodes.",
            "A measured value never stands in a column that holds a target or a limit: a column headed \"measured\" "
            "holds measurements, a column headed \"Limit\" or \"as defined\" holds none.  A measured median is "
            "printed as \"median (n)\": n is the number of episodes that give the quantity.",
            "Numbers: counts as integers; "
            + ", ".join(f"{u} with {d} decimals" for u, d in DECIMALS.items()) + "; every other number with %g.",
            f"Every line of set {bar.CANDIDATE_SET} carries the label \"{MARGIN_LABEL}\": its duty and lift were set "
            "after the pull request's results on those seeds (STAGE2_75_PLAN.md A3).",
            f"A cell with no record reads \"{NOT_RUN}\", or \"{CUT}\" where `budget.json` lists its record under a "
            "cut unit, or where a unit of its task was cut and the cell, one of the registered order's, is in no "
            "started unit.",
            f"A record made later at the path of a finished one is labelled \"{RERUN}\" with its reason, and the "
            "registered record's counts and verdict stand beside it (STAGE2_75_PLAN.md rule R2)."],
        amended_after_result=[dict(a) for a in AMENDED_AFTER_RESULT],
        checks=checks(doc), walk=walk_table(doc), ladder=ladder_table(doc), rest=rest_table(doc),
        glance=glance_table(doc), replication=replication_table(doc), jps=jps_table(doc),
        controls=controls_table(doc),
        rules={k: bar.RULES[k] for k in ("readings", "pass_", "replication", "seeds", "budget", "no_overwrite")})
    tabs["other"] = other_records(doc, tabs)
    return json.loads(json.dumps(s2io.plain(tabs)))


# ------------------------------------------------------------------------------------------------
# Markdown
# ------------------------------------------------------------------------------------------------
ORDER = ("walk", "ladder", "rest", "glance", "replication", "jps", "controls", "other")


def _cell_text(v, col: dict) -> str:
    kind, unit = col.get("kind"), col.get("unit")
    if v is None or v == [] or v == {}:
        return "none" if kind == "terms" and v == {} else "-"
    if kind == "of" and isinstance(v, dict):
        return f"{v['k']} of {v['n']}"
    if kind == "measured" and isinstance(v, dict):
        return "-" if v.get("median") is None else f"{_num(v['median'], unit)} ({v['n']})"
    if kind == "terms" and isinstance(v, dict):
        return "; ".join(f"{k}: {n}" for k, n in v.items())
    if kind == "flag":
        return "yes" if v is True else "no" if v is False else str(v)
    if kind in ("labels", "list") and isinstance(v, list):
        return ("; " if kind == "labels" else "<br>").join(str(x) for x in v)
    if kind == "path":
        return "<br>".join(f"`{x}`" if "/" in str(x) and " " not in str(x) else str(x)
                           for x in (v if isinstance(v, list) else [v]))
    if kind == "code":
        return f"`{v}`"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if _is_num(v):
        return _num(v, unit)
    if isinstance(v, list):
        return ", ".join(_num(x) if _is_num(x) else str(x) for x in v)
    return str(v)


def _title(col: dict) -> str:
    unit = col.get("unit")
    return col["title"] + (f" ({unit})" if unit and unit != "bound" and col.get("kind") == "measured" else "")


def _md_table(part: dict) -> list[str]:
    cols = part["columns"]
    if not part["lines"]:
        return ["No line: no record of this kind was found.", ""]
    out = ["| " + " | ".join(_title(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for line in part["lines"]:
        out.append("| " + " | ".join(_cell_text(line.get(c["key"]), c).replace("|", "\\|").replace("\n", " ")
                                     for c in cols) + " |")
    return out + [""]


def _md_checks(c: dict) -> list[str]:
    r = c["records"]
    out = ["## Checks", "",
           f"- Run directory: `{c['run']}`"
           + (f"; sets asked for: {', '.join(c['sets_asked'])}" if c.get("sets_asked") else "")
           + (f"; body asked for: {c['body_asked']}" if c.get("body_asked") else "") + ".",
           f"- Records found: {r['found']} ({r['done']} done, {r['not_run']} \"not run\")"
           + ("" if not r["by_set"] else ": " + "; ".join(
               f"`{s}` {v['done']} done and {v['not_run']} \"not run\"" for s, v in r["by_set"].items()))
           + f".  Control records: {r['controls']}.  Records kept beside a re-run: {r['kept_beside_a_rerun']}.",
           f"- Files under `eval/` that are no record of the bar: {len(r['not_a_record'])}"
           + "".join(f"\n  - `{p}`" for p in r["not_a_record"]),
           f"- Records whose path is not the one the runner makes from their fields: {len(r['misplaced'])}"
           + "".join(f"\n  - `{p}`" for p in r["misplaced"]),
           f"- Today's sources: panels `{c['today_sources']['panels']}`, bar `{c['today_sources']['bar']}`.",
           f"- Records whose `key.sources` differ from today's: {len(c['other_sources'])} (they are tabulated, "
           "flagged in their labels)"
           + "".join(f"\n  - `{x['path']}`: {', '.join(x['differs'])} differ" for x in c["other_sources"]),
           f"- Records whose input hashes are not in their set's latest freeze: {len(c['not_in_latest_freeze'])}"
           + "".join(f"\n  - `{x['path']}` (`{x['set_dir']}`): {'; '.join(x['differences'])}"
                     for x in c["not_in_latest_freeze"]),
           f"- Records whose inputs could not be checked (no `set.json` with a freeze at the record's `inputs.dir` "
           f"under the study directory): {len(c['not_checked'])}"
           + "".join(f"\n  - `{x['path']}` (`{x['set_dir']}`)" for x in c["not_checked"]),
           "- Latest freeze of each set a record names:" + (" none" if not c["freezes"] else "".join(
               f"\n  - `{d}`: " + ("no `set.json` found" if not m.get("found") else "not frozen" if not m.get("frozen")
                                   else f"freeze {m['index']}, manifest `{m['manifest_sha256']}`")
               for d, m in c["freezes"].items())),
           f"- Records with episodes that failed to run or with unjudged counts: {len(c['errors_or_unjudged'])}"
           + "".join(f"\n  - `{x['path']}`: episodes {x['episodes']}, failed to run {x['errors']}, "
                     + ("judged" if x["judged"] else "not judged") for x in c["errors_or_unjudged"]),
           "- Records whose counts, bounds or passes do not recompute from their own episodes (`bar.counts`): "
           f"{len(c['counts_not_recomputed'])}"
           + "".join(f"\n  - `{x['path']}`: {'; '.join(x['differences'])}" for x in c["counts_not_recomputed"]),
           "- Records of the block `seeds5042` that the replication rule does not provide for (an R0 row, a control "
           f"set, a reported control, no `replication_of`): {len(c['second_block_outside_the_rule'])}"
           + "".join(f"\n  - `{x['path']}`: {', '.join(x['why'])}" for x in c["second_block_outside_the_rule"])]
    if c["budget"] is None:
        out.append("- Core-hours: no `budget.json` in the run directory (no task's unit has started there).")
    else:
        for t, b in c["budget"].items():
            cap = b["cap_core_h"]
            out.append(
                f"- Task {t}: {_num(b['spent_core_h'], 'core-h')} core-hours spent of {_num(float(cap))} "
                "(the sum of its episodes' wall seconds in `budget.json`)"
                + ("" if b["registered_cap_core_h"] in (None, cap) else
                   f"; the registered budget is {_num(float(b['registered_cap_core_h']))} core-hours")
                + f"; {len(b['units'])} units started, {len(b['cut'])} cut"
                + ("" if not b["cut"] else
                   f"; units started after the first cut ({b['first_cut_at']}): "
                   + (", ".join(b["started_after_the_first_cut"]) or "none (the cut units are the order's tail)"))
                + ".")
            if b["units"]:
                out += ["", "  | Unit, in the order started | Estimate (core-h) | Spent before it (core-h) | Started at "
                            "| Used (core-h) |", "  |---|---|---|---|---|"]
                out += [f"  | {u['unit']} | " + " | ".join(
                    "-" if u[k] is None else _num(float(u[k]), "core-h") for k in ("estimate_core_h",
                                                                                     "spent_before_core_h"))
                        + f" | {u['started_at']} | {_num(u['used_core_h'], 'core-h')} |" for u in b["units"]]
                out.append("")
            if b["cut"]:
                out += ["  | Unit cut by the pre-registered order | Estimate (core-h) | Spent before it (core-h) | "
                        "Reason |", "  |---|---|---|---|"]
                out += [f"  | {u['unit']} | " + " | ".join(
                    "-" if u[k] is None else _num(float(u[k]), "core-h") for k in ("estimate_core_h",
                                                                                     "spent_before_core_h"))
                        + f" | {u['reason']} |" for u in b["cut"]]
                out.append("")
    if c["cost"] is None:
        out.append("- `COST.md`: none in the run directory (it is measured and committed before task 3's first unit).")
    else:
        mo = c["cost"].get("measured_on") or {}
        load = _get(c["cost"], "load", "before", "load_average")
        out.append(
            "- `COST.md`'s rates (wall seconds per simulated second), measured on Stage 1's own condition "
            f"`{mo.get('condition')}` of `{mo.get('body')}`, seeds {mo.get('seeds')}: " + "; ".join(
                f"{k} {_num(float(v['wall_per_sim_s']), 'ratio')} (row `{v['row']}`, {v['episodes']} episodes, "
                + ("run their full length" if v["full_length"] else "not all run their full length")
                + (", after a warm-up episode that is not counted" if v["warm_up_counted"] is False else "") + ")"
                for k, v in c["cost"]["rates"].items())
            + (f"; the machine's load averages before the measurement: {', '.join(_num(float(x)) for x in load)}"
               if load else "") + ".")
    return out + [""]


def markdown(tabs: dict) -> str:
    """``BAR_TABLES.md`` from the JSON form of the tables, and from nothing else."""
    out = [f"# The bar's tables: `{tabs['run']}`", "",
           f"Made by `{tabs['made_by']}` from the records under `{tabs['run']}`.  {tabs['simulation']}", ""]
    out += _md_checks(tabs["checks"])
    out += ["## How to read the tables", ""] + [f"- {x}" for x in tabs["conventions"]] + [
        "- Rules amended after a result, with the plan's own labels (" + PLAN_FILE + "): "
        + "  ".join(f"{a['rule']} {a['label']} {a['text']}" for a in tabs["amended_after_result"]), ""]
    for name in ORDER:
        t = tabs[name]
        out += [f"## {t['title']}", ""] + [f"- {x}" for x in t.get("notes") or []] + [""]
        number = t["title"].split(".", 1)[0]
        for i, part in enumerate(t["parts"]):
            out += [f"### {number}.{i + 1} {part['title'][:1].upper()}{part['title'][1:]}", ""]
            out += _md_table(part)
            for x in part.get("paths") or []:
                out.append(f"Record: `{x}`")
            if part.get("paths"):
                out.append("")
            if part.get("notes"):
                out += [f"- {x}" for x in part["notes"]] + [""]
    out += ["## The runner's rules these tables rest on", ""] + [f"- `{k}`: {v}" for k, v in tabs["rules"].items()]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------------------------------------
def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write(out, dest=None, sets=None, body=None, *, study=None) -> dict:
    """Read the run directory ``out`` and write ``<dest>/bar_tables.json`` and ``<dest>/BAR_TABLES.md`` (``dest``
    default: ``out``), each atomically; the Markdown is made from the JSON as written.  Returns the tables."""
    dest = Path(out if dest is None else dest)
    tabs = tables(load(out, sets=sets, body=body, study=study))
    s2io.write_json(dest / TABLES_JSON, tabs)
    _write_text(dest / TABLES_MD, markdown(json.loads((dest / TABLES_JSON).read_text())))
    return tabs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.barreport", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the run directory the bar wrote its records under")
    ap.add_argument("--dest", default=None, help="where the two files go (default: the run directory)")
    ap.add_argument("--set", action="append", dest="sets", default=None, help="a set to tabulate (repeat; default: "
                    "the three sets of the round and every set found)")
    ap.add_argument("--body", default=None, help="a body to tabulate (default: every body found)")
    a = ap.parse_args(argv)
    try:
        tabs = write(a.out, a.dest, a.sets, a.body)
    except s2io.S2Error as exc:
        print(f"barreport: {exc}", file=sys.stderr)
        return 1
    dest = Path(a.out if a.dest is None else a.dest)
    c = tabs["checks"]
    print(f"{dest / TABLES_JSON}\n{dest / TABLES_MD}")
    print(f"records found: {c['records']['found']} ({c['records']['done']} done, {c['records']['not_run']} not run); "
          f"other sources: {len(c['other_sources'])}; not in the latest freeze: {len(c['not_in_latest_freeze'])}; "
          f"not checked: {len(c['not_checked'])}; errors or unjudged: {len(c['errors_or_unjudged'])}; counts that "
          f"do not recompute: {len(c['counts_not_recomputed'])}")
    return 0


__all__ = ["absent_word", "checks", "digest", "jps_table", "load", "main", "markdown", "measures_of", "planned",
           "registered", "replication_word", "tables", "verdict_of", "write"]

if __name__ == "__main__":
    from designlab.stage2 import barreport as _named  # the importable copy owns the module's state

    sys.exit(_named.main())
