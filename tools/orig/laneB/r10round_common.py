"""T. rex r10 round (criteria lean v6): shared constants and readers for the lane B tools.  Report only; never trains.

Imported by path (importlib) by ledger_r10round.py, complete_r10round.py, score_r10round.py, fetch_r10round.py,
k3panel_r10round.py and v9_bundle.py.  Standard library only (numpy is imported lazily where an .npz is read).

Every rule here is the criteria text's (S2/r10round/lean/v6/criteria_r10round_lean_v6.txt, sha256 e5bea5d3...), or a
definition it cites from criteria v12 (S2/r10round/v12/criteria_r10round_v12.txt, "THIS12") by line.  Where the text
leaves a point open the code implements nothing for it and says so (S2/r10round/lean/build/v5/REPORT.txt,
S2/r10round/lean/build/v6/REPORT.txt).

v5 (RUNLOG line 657; tool updater 2026-10-26): KIND ACTION (C1 reads it as an action); a STANCE6 note written after
its RUN_ID's F2-START line with an earlier time (a restatement after the deadline) is printed and read by no rule; branch_head is no pin (the branch HEAD is APPROVED's and the
LAUNCH lines'); R1 of a STOPPED run and every RESOLUTION checkpoint from the 262,144 grid (R-m1); CMD_a over the
episodes that last beyond step 200, not formed (reading as below 0.60 in either sign) when none does (O-6); an exact
sign tie counts a flag once in each sign (O-7); DRIVE-BACKED at any flagged checkpoint (O-8, score_r10round.py).

v6 (RUNLOG line 661, V5C-2 option A; amender 2026-10-27): MEAN_a (mean_a, mean_reaches) is (CMD_a(R1) + CMD_a(R2)) / 2
with both formed, the formed one's CMD_a with one not formed, and not formed (below 0.60 in either sign) with neither;
it always has a reading, so nothing about MEAN_a is held open any more (V5-1 / V5C-2 closed; V5C-3: a formed 0.60
beside one not formed is MEAN_a 0.60, which reaches 0.60).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import shlex
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

VERSION = ("r10round lane B tools 1.3 (criteria lean v6 sha256 e5bea5d314cb...; built 2026-10-26; fixer 2026-10-26; "
           "v5 tool update 2026-10-26; v6 MEAN_a 2026-10-27)")

# ------------------------------------------------------------------------------------------------- paths
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
SP = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
RUNLOG = SP + "/foot/stance4/live/run_log.txt"
ROUND = SP + "/foot/stance6"                      # <ROUND> (D16)
BR = S2 + "/r10round/runbranch/clone"             # read only
CRITERIA = S2 + "/r10round/lean/v6/criteria_r10round_lean_v6.txt"
THIS12 = S2 + "/r10round/v12/criteria_r10round_v12.txt"
EVID = S2 + "/r10round/v12/evidence_r10round_v12.txt"
TOOLS = Path(__file__).resolve().parent
PINS_FILE = TOOLS / "pins_r10round.json"

# The three texts' sha256 as the criteria text states them (STATUS / Short names); this text's own is its file's.
CRITERIA_SHA256 = "e5bea5d314cb3c6b6c1e2889452ad6dfa8ba1a90aec8dbee758262754bed2162"
THIS12_SHA256 = "0ab5edeba04b319a4233a4700ebf9abf7d7d7127c30afc7565a7fa3bc34369ac"
EVID_SHA256 = "353f4536ef3c0e06a1937714c44a22c7d52819be2e613cbc1ad23b47821f4e20"

# ------------------------------------------------------------------------------------------------- the round
SEEDS = (60, 42, 80)                                        # ARMS AND SEEDS (D2), in this order
RUN_IDS = {f"stance6_r10LTSB_s{s}": s for s in SEEDS}       # D16
SIGMA_KEYS = {60: "source_vectors.60", 42: "source_vectors.42", 80: "control_sigma.80"}
SIGMA_FILE_COLAB = "/content/mesozoic-labs/docs/investigations/stance_2026_10/stance5/replication/sigma_s70_s80.json"
RUN_LABELS = {60: "r10 LTSB s60", 42: "r10 LTSB s42", 80: "r10 LTSB s80"}
REPO_REF = "claude/screen-trex-r10-armltsb"

GRID = 131072                                               # evaluation every 131,072 steps (91, last 11,927,552)
N_EVALS = 91
LAST_EVAL = N_EVALS * GRID
GRID_STEPS = tuple(k * GRID for k in range(1, N_EVALS + 1))
CKPT = 262144                                               # periodic pair every 262,144 (45 pairs, D3 (a))
LAST_PERIODIC = 45 * CKPT                                   # 11,796,480
FINAL = 12001280
J_STEPS = (4194304, 5242880, 6291456, 8388608, 10485760)    # J = these and the final
R1_STEP = 10485760
REPORT_GRID_PAIRS = tuple(k * 1048576 for k in range(1, 12))   # F2 (6): the 11 pairs of the 1,048,576 grid
TAIL = (11, 12, 13, 14)                                     # TAIL ACTUATORS
PITCH = (11, 13, 14)                                        # TAIL PITCH ACTUATORS
BAR = 0.10
PARK_LINE = 0.60
FULL_HORIZON = 1000
WINDOW_START = 200
SAT_KEY = "max_actuator_saturation_fraction"
LAUNCH_WINDOW_MIN = 60.0                                    # C4 started_at, launch check
JUDGE_GAP_MIN = 120.0                                       # C5
TRIGGER_WINDOW_MIN = 120.0                                  # REPLACEMENT: trigger note within 2 h of the launch
RELAUNCH_WINDOW_MIN = 60.0                                  # REPLACEMENT: _b within 1 h after the trigger note
STATUE_MAIN = 3766.6                                        # launch check (4): not within 50 of main's statue
STATUE_BAND = 50.0
LOAD_KEYS = ("load_path", "load_mode", "parent_checkpoint_sha256", "resume_load_path", "resume_checkpoint_sha256")
# digest_snapshot.py UNPINNED_STAGE_CONFIG_KEYS (line 145) and the two blocks the harness does not pass (THIS12 O0)
VIEW_UNPINNED = ("species", "embodiment", "stage", "library_version", "git_commit", "run", "gpu",
                 "plant_identity", "task_fingerprint")
SAT640_COLUMNS = ("saturated_actuator", "saturated_actuator_fraction", "saturated_actuator_positive_share",
                  "saturated_actuator_ties", "actuators_saturated")
# Actuator order [measured, BR MJCF <actuator> block]: index -> name the code writes in saturated_actuator.
ACTUATOR_NAMES = ("neck_pitch_act", "neck_yaw_act", "head_pitch_act", "r_hip_pitch_act", "r_hip_roll_act",
                  "r_knee_act", "r_ankle_act", "l_hip_pitch_act", "l_hip_roll_act", "l_knee_act", "l_ankle_act",
                  "tail_1_pitch_act", "tail_1_yaw_act", "tail_2_pitch_act", "tail_3_pitch_act")
TRACE_SEEDS = tuple(range(3042, 3062))                      # M1: 20 episodes
CERT_SEEDS = tuple(range(3042, 3082))                       # M3: 40 per heading

# ------------------------------------------------------------------------------------------------- RUNLOG note form
KINDS = ("LAUNCH", "CHECK", "STOP", "TRIGGER-a", "TRIGGER-b", "TRIGGER-c", "TRIGGER-d", "LISTING", "DELETED",
         "FOUND-GONE", "RELAUNCH", "ACTION", "F2-START", "REFETCH-START")
#: Records (v5): "ACTION <RUN_ID> what=<resume|restart|relaunch|cell>", read by C1 as an action (any ACTION line is one;
#: a what= outside these four is printed beside it).
ACTION_WHATS = ("resume", "restart", "relaunch", "cell")
#: Records (v5): a line restated "before F2's start note (F2-START)" is read; "a line not restated by then is printed
#: and read by no rule".  A note of a RUN_ID that stands in RUNLOG after that RUN_ID's first F2-START line but carries a
#: time before that line's time (an event of before F2's start, written after it: a late restatement) is printed and
#: read by no rule; F2-START and REFETCH-START lines, and a note of an event at or after F2's start (its own time), are
#: read as they stand.
AFTER_F2_KINDS = ("F2-START", "REFETCH-START")
LATE_PREFIX = ("AFTER ITS RUN_ID'S F2-START LINE WITH AN EARLIER TIME (not restated before F2's start note; printed, "
               "read by no rule): ")
NOTE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?) UTC\s+STANCE6\s+(\S+)\s+(\S+)(?:\s+(.*))?$")
RUN_TOKEN_RE = re.compile(r"stance6_r10LTSB_s\d+\w*")


class InputError(Exception):
    """An input the tool cannot read as the text requires (printed; exit 2)."""


# ------------------------------------------------------------------------------------------------- small helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hexpart(value) -> str | None:
    """The hex after an entry's "sha256:" prefix, or the value itself when it has none (C3; V2: read the same)."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v.startswith("sha256:"):
        v = v[len("sha256:"):]
    return v.lower() if re.fullmatch(r"[0-9a-fA-F]{64}", v) else None


def num(cell):
    """A cell as a finite float, or None when it is NaN, blank or does not parse (THIS12 line 349)."""
    if cell is None:
        return None
    if isinstance(cell, bool):
        return None
    try:
        v = float(cell)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_utc(text) -> datetime | None:
    """An ISO time as written by the code ("...Z", "+00:00") or RUNLOG's "YYYY-MM-DD HH:MM[:SS]"; UTC."""
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip().replace("Z", "+00:00")
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.astimezone(timezone.utc)


def minutes(a: datetime, b: datetime) -> float:
    """b - a in minutes."""
    return (b - a).total_seconds() / 60.0


def load_json_bytes(b: bytes):
    """JSON from bytes, or None when it is not UTF-8 JSON."""
    try:
        return json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def canonical_json_sha256(value) -> str:
    """BR result_bundle/hashing.py canonical_json_sha256: sorted keys, compact separators, ensure_ascii False."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def view_sha256(stage_config: dict) -> str:
    """stage_config_view_sha256 as THIS12 O0 (lines 376-383) defines it: the canonical JSON sha256 of
    stage_config.json without UNPINNED_STAGE_CONFIG_KEYS and without plant_identity and task_fingerprint."""
    return canonical_json_sha256({k: v for k, v in stage_config.items() if k not in VIEW_UNPINNED})


def keys_at_any_depth(obj, names) -> list[str]:
    """Every dotted path in obj whose last key is one of names (C4: load keys at any depth)."""
    found = []

    def walk(o, prefix):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{prefix}.{k}" if prefix else str(k)
                if k in names:
                    found.append(p)
                walk(v, p)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{prefix}[{i}]")
    walk(obj, "")
    return found


def dig(obj, dotted: str):
    """obj at a dotted path, or None."""
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def zip_num_timesteps(raw: bytes):
    """(num_timesteps, problem): an SB3 zip's recorded num_timesteps from its JSON "data" member (never unpickled),
    after zipfile.testzip passes (F2: each zip passes testzip and has its num_timesteps noted)."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            bad = z.testzip()
            if bad is not None:
                return None, f"testzip failed at {bad}"
            data = json.loads(z.read("data").decode("utf-8"))
    except (zipfile.BadZipFile, KeyError, ValueError, UnicodeDecodeError, OSError) as exc:
        return None, f"not a readable SB3 zip ({type(exc).__name__}: {exc})"
    n = data.get("num_timesteps") if isinstance(data, dict) else None
    if not isinstance(n, int) or isinstance(n, bool):
        return None, f"no integer num_timesteps in the data member ({n!r})"
    return n, None


# ------------------------------------------------------------------------------------------------- pins
# v5 (VALIDATIONS, PINS): the branch HEAD (the header commit) is no pin: it is recorded in RUNLOG (the LAUNCH lines'
# head=) and <ROUND>/APPROVED, never inside a tool.  A pins file that still holds a "branch_head" key is not read for it.
PIN_KEYS = ("main_commit", "tel_columns", "port_commits", "d3_commit", "d19_commit",
            "task_sha256", "hyperparameters_sha256", "stage_config_view_sha256", "gate_sha256", "plant_identity",
            "collapse_peak_floor_reference", "m1_sha256", "m2_sha256", "m3_sha256", "judge_sha256", "tool_sha256",
            "sigma_file_sha256", "sigma_vector_sha256", "known25_sha256", "v7_result", "v7_method",
            "v0h_digests", "model_sha256", "model_out_sha256", "v10_result", "v7_failed_on_head")


def load_pins(path=None) -> dict:
    """The pins file (JSON).  A key whose value is null, missing or the string "UNFILLED" is unfilled."""
    p = Path(path) if path else PINS_FILE
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read the pins file {p}: {exc}") from exc
    if not isinstance(d, dict):
        raise InputError(f"pins file {p} is not a JSON object")
    return d


def pin(pins: dict, key: str):
    v = pins.get(key)
    return None if (v is None or v == "UNFILLED") else v


def unfilled_pins(pins: dict) -> list[str]:
    return [k for k in PIN_KEYS if pin(pins, k) is None]


# ------------------------------------------------------------------------------------------------- RUNLOG
def parse_note_fields(rest: str) -> dict:
    fields = {}
    if not rest:
        return fields
    try:
        toks = shlex.split(rest)
    except ValueError:
        toks = rest.split()
    for t in toks:
        if "=" in t:
            k, v = t.split("=", 1)
            fields[k] = v
        else:
            fields.setdefault("_words", []).append(t)
    return fields


def of_round(run: str, run_ids) -> bool:
    """A round RUN_ID or a suffixed one (_b, and a _c that REPLACEMENT then reads as not of the round)."""
    return any(run == r or run.startswith(r + "_") for r in run_ids)


def parse_runlog(path=RUNLOG, run_ids=None) -> tuple[list[dict], list[tuple[int, str]]]:
    """(notes, others): every line of the form "<date -u> STANCE6 <KIND> <RUN_ID> <fields>" (Records), parsed;
    and every other line that names a RUN_ID of the round (printed for the coordinator; read by no rule).  Fixer
    (R-m7): a STANCE6 note whose RUN_ID is not a round RUN_ID (a typo such as stance6_r10LTSB_s420) is not a note of
    the round: it goes to the printed lines, never silently dropped.
    v5 (Records, O-2): a line in another form is restated "with its original time ... before F2's start note
    (F2-START); a line not restated by then is printed and read by no rule".  RUNLOG is written in order, so a note of
    a RUN_ID (other than F2-START / REFETCH-START) that stands after that RUN_ID's first F2-START line and carries a
    time before that line's time was written after F2's start about an event before it: not restated by then, it goes
    to the printed lines with LATE_PREFIX.  A note standing there with its own time at or after F2's start (an event of
    after F2's start, e.g. an ACTION) is read as it stands [inferred: only a restatement carries an earlier time]."""
    notes, others = [], []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise InputError(f"cannot read RUNLOG {path}: {exc}") from exc
    for i, line in enumerate(text.splitlines(), 1):
        m = NOTE_RE.match(line.strip())
        if m and m.group(2) in KINDS and run_ids is not None and not of_round(m.group(3), run_ids):
            others.append((i, line.rstrip()))
            continue
        if m and m.group(2) in KINDS:
            t = parse_utc(m.group(1).replace(" ", "T") + ("" if m.group(1).count(":") == 2 else ":00"))
            notes.append(dict(line=i, time=t, kind=m.group(2), run=m.group(3),
                              fields=parse_note_fields(m.group(4) or ""), text=line.rstrip()))
        elif RUN_TOKEN_RE.search(line):
            names = set(RUN_TOKEN_RE.findall(line))
            if run_ids is None or any(any(n == r or n.startswith(r) for r in run_ids) for n in names):
                others.append((i, line.rstrip()))
    f2 = {}                                              # RUN_ID -> its first F2-START note (by line)
    for n in notes:
        if n["kind"] == "F2-START" and (n["run"] not in f2 or n["line"] < f2[n["run"]]["line"]):
            f2[n["run"]] = n
    late = [n for n in notes if n["kind"] not in AFTER_F2_KINDS and n["run"] in f2 and n["line"] > f2[n["run"]]["line"]
            and n["time"] is not None and f2[n["run"]]["time"] is not None and n["time"] < f2[n["run"]]["time"]]
    if late:
        notes = [n for n in notes if n not in late]
        others = sorted(others + [(n["line"], LATE_PREFIX + n["text"]) for n in late])
    return notes, others


def records_pairing(notes) -> list[str]:
    """v5 (Records, report only): "trigger (a) is noted as TRIGGER-a and FOUND-GONE at one time; a launch-check stop as
    STOP and TRIGGER-d at one time".  Each note of such a pair without its partner at the same time is listed (the
    rules read each line as it stands; nothing is inferred from a missing partner)."""
    out = []
    for a, b in (("TRIGGER-a", "FOUND-GONE"), ("STOP", "TRIGGER-d")):
        for n in notes:
            for x, y in ((a, b), (b, a)):
                if n["kind"] == x and not any(m["kind"] == y and m["run"] == n["run"] and m["time"] == n["time"]
                                              for m in notes):
                    out.append(f"RUNLOG line {n['line']}: {x} {n['run']} at {n['time']:%Y-%m-%d %H:%M} has no {y} "
                               "at the same time (Records: the two are noted at one time)")
    return out


def notes_for(notes, run_id, kinds=None):
    return [n for n in notes if n["run"] == run_id and (kinds is None or n["kind"] in kinds)]


def first_note(notes, run_id, kinds):
    xs = sorted(notes_for(notes, run_id, kinds), key=lambda n: (n["time"], n["line"]))
    return xs[0] if xs else None


# ------------------------------------------------------------------------------------------------- the ledger
def reason_keys(cell) -> set:
    """score_stance4.saturation_alarm's reading of screen_v2_reasons: ';'-separated, key before ':' (SC3.reason_key)."""
    return {x.split(":", 1)[0].strip() for x in (cell or "").split(";") if x.strip()}


def actuator_index(cell):
    """The actuator a saturated_actuator cell names: a model actuator name, or "actuator[<i>]" (THIS12 line 364)."""
    if cell is None:
        return None
    c = str(cell).strip()
    if not c:
        return None
    if c in ACTUATOR_NAMES:
        return ACTUATOR_NAMES.index(c)
    m = re.fullmatch(r"actuator\[(\d+)\]", c)
    if m and int(m.group(1)) < len(ACTUATOR_NAMES):
        return int(m.group(1))
    return None


def read_ledger_bytes(raw: bytes):
    """(header, rows) of eval_episodes.csv bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"ledger is not UTF-8: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    header = list(reader.fieldnames or [])
    return header, list(reader)


def read_ledger(path):
    with open(path, "rb") as fh:
        return read_ledger_bytes(fh.read())


def decode_row(r: dict, header) -> dict:
    """One ledger row's ANY and TAIL inputs as THIS12 lines 351-364 define them (M5 (a)-(c)).

    Returns any_status / tail_status ('measured' or 'missing'), any_over, tail, tail_act, sign ('+', '-', '?'),
    unres, over (fraction > 0.10), reason ('> 0.1' / 'unmeasured' / 'listed') and why (for a missing input)."""
    cols = set(header)
    clean = num(r.get("screen_v2_clean")) if "screen_v2_clean" in cols else None
    screen_measured = clean is not None
    lists = SAT_KEY in reason_keys(r.get("screen_v2_reasons")) if "screen_v2_reasons" in cols else False
    frac = num(r.get("saturated_actuator_fraction")) if "saturated_actuator_fraction" in cols else None
    length = num(r.get("length")) if "length" in cols else None
    act = actuator_index(r.get("saturated_actuator")) if "saturated_actuator" in cols else None
    ties = num(r.get("saturated_actuator_ties")) if "saturated_actuator_ties" in cols else None
    nsat = num(r.get("actuators_saturated")) if "actuators_saturated" in cols else None
    pos = num(r.get("saturated_actuator_positive_share")) if "saturated_actuator_positive_share" in cols else None
    out = dict(any_status="measured", any_over=False, tail_status="measured", tail=False, tail_act=None, sign="?",
               unres=False, over=False, reason=None, why=None, frac=frac, act=act, screen_measured=screen_measured,
               lists=lists)
    if screen_measured and not lists:
        # a measured screen that does not list the key: under the bar when the fraction is blank or <= 0.10;
        # over 0.10 the row contradicts its screen: ANY and TAIL inputs missing (THIS12 lines 356-358)
        if frac is not None and frac > BAR:
            out.update(any_status="missing", tail_status="missing", why="fraction over the bar under an unlisted "
                       "measured screen (contradiction)")
        return out
    # ANY input (THIS12 lines 351-355): a measured screen lists the key; an unmeasured one reads the fraction
    if screen_measured:
        any_over = True
    elif frac is not None:
        any_over = frac > BAR
    else:
        any_over = None
    # TAIL input (THIS12 lines 359-364)
    tail_missing, why = False, None
    if frac is None:
        if length is None or length > WINDOW_START:
            tail_missing, why = True, "fraction missing in an episode that outlived the window start"
    elif frac > BAR:
        out["over"] = True
        if act is None:
            tail_missing, why = True, "fraction over the bar and saturated_actuator blank or naming no actuator"
        elif act in TAIL:
            out["tail"] = True
            out["tail_act"] = act
            out["sign"] = "?" if pos is None else ("+" if pos >= 0.5 else "-")
        elif ties is None or nsat is None:
            tail_missing, why = True, "fraction over the bar on a non-tail actuator with ties/actuators_saturated missing"
        else:
            out["unres"] = ties >= 1 or nsat >= 2
    if out["tail"]:
        any_over = True                                      # a row over the bar on a tail actuator is any-over
    if any_over is None:
        out.update(any_status="missing", any_over=False, why=why or "screen unmeasured and fraction missing")
    else:
        out["any_over"] = bool(any_over)
        if any_over:
            out["reason"] = "> 0.1" if (frac is not None and frac > BAR) else ("unmeasured" if frac is None else "listed")
    if tail_missing:
        out.update(tail_status="missing", tail=False, tail_act=None, unres=False, why=why)
    return out


def grid_step(cell):
    """The grid step of a timesteps cell, or None when it is not a number on the grid 131,072 ... 11,927,552."""
    t = num(cell)
    if t is None or t != int(t):
        return None
    t = int(t)
    if t % GRID or not (1 <= t // GRID <= N_EVALS):
        return None
    return t


def episode_index(cell):
    e = num(cell)
    return int(e) if (e is not None and e == int(e)) else None


def m5(header, rows) -> dict:
    """M5 (a)-(c) on ONE ledger, rows as they stand: per evaluation the ANY, TAIL and TAIL-UNRESOLVED counts (a row
    with a missing input flags nothing; a measured ANY input stays as measured), flagged evaluations and alarms."""
    by, offgrid = {}, []
    for i, r in enumerate(rows):
        s = grid_step(r.get("timesteps"))
        if s is None:
            offgrid.append((i, r.get("timesteps")))
            continue
        by.setdefault(s, []).append(r)
    evals = {}
    for s in sorted(by):
        rs = by[s]
        d = dict(step=s, n=len(rs), episodes=[episode_index(r.get("episode")) for r in rs], any=0, tail=0, unres=0,
                 tail_by={}, any_by={}, reasons={}, any_missing=[], tail_missing=[], rail=[], rows=[])
        for r in rs:
            x = decode_row(r, header)
            d["rows"].append(x)
            ep = episode_index(r.get("episode"))
            if x["any_status"] == "missing":
                d["any_missing"].append((ep, x["why"]))
            if x["tail_status"] == "missing":
                d["tail_missing"].append((ep, x["why"]))
            if x["any_over"]:
                d["any"] += 1
                name = ACTUATOR_NAMES[x["act"]] if x["act"] is not None else "(unnamed)"
                d["any_by"][name] = d["any_by"].get(name, 0) + 1
                d["reasons"][x["reason"]] = d["reasons"].get(x["reason"], 0) + 1
            if x["tail"]:
                d["tail"] += 1
                key = f"{ACTUATOR_NAMES[x['tail_act']]}{x['sign']}"
                d["tail_by"][key] = d["tail_by"].get(key, 0) + 1
            if x["unres"]:
                d["unres"] += 1
            if "rail_actuators" in header and (r.get("rail_actuators") or "").strip():
                d["rail"].append(r.get("rail_actuators").strip())
        d["any_flagged"] = d["any"] >= 5
        d["tail_flagged"] = d["tail"] >= 5
        evals[s] = d
    any_alarms, tail_alarms, need = [], [], []
    for s in sorted(evals):
        p = s - GRID
        if p not in evals:
            continue
        e0, e1 = evals[p], evals[s]
        if e0["any_flagged"] and e1["any_flagged"]:
            any_alarms.append(s)
        if e0["tail_flagged"] and e1["tail_flagged"]:
            tail_alarms.append(s)
        elif (e0["any_flagged"] and e1["any_flagged"] and e0["tail"] + e0["unres"] >= 5
              and e1["tail"] + e1["unres"] >= 5):
            need.append(s)                                   # NEEDS RESOLUTION (CLASSES)
    return dict(evals=evals, offgrid=offgrid, any_alarms=any_alarms, tail_alarms=tail_alarms, needs_resolution=need,
                any_flagged=[s for s in sorted(evals) if evals[s]["any_flagged"]],
                tail_flagged=[s for s in sorted(evals) if evals[s]["tail_flagged"]])


def tail_alarm_detail(res: dict, step: int) -> dict:
    """Actuator(s) and sign(s) of a TAIL ALARM's two evaluations, with the flagged-evaluation count."""
    e0, e1 = res["evals"][step - GRID], res["evals"][step]
    names = {}
    for e in (e0, e1):
        for k, v in e["tail_by"].items():
            names[k] = names.get(k, 0) + v
    return dict(step=step, actuators=dict(sorted(names.items())), flagged_evaluations=len(res["tail_flagged"]))


def resolution_checkpoint(s1: int, periodic_listed, E=None):
    """RESOLUTION CHECKPOINT (v5, R-m1): "the pair at the second evaluation's step when that step is a multiple of
    262,144, else the pair at the next multiple of 262,144 at or before E, else stage1_final (C3 then fails when that
    pair is not in the manifest)".  The tools pass periodic_listed=None: the 262,144 grid up to min(E, 11,796,480)
    (whatever the manifest lists).  A set (a test's) restricts the grid to it."""
    if periodic_listed is None:
        top = LAST_PERIODIC if E is None else min(E, LAST_PERIODIC)
        periodic_listed = {k * CKPT for k in range(1, 46) if k * CKPT <= top}
    if s1 in periodic_listed:
        return s1
    later = sorted(p for p in periodic_listed if p > s1)
    return later[0] if later else "final"


# ------------------------------------------------------------------------------------------------- the manifest
def manifest_schema_problems(m) -> list[str]:
    """C2's "schema valid" as BR result_bundle/manifest.py verify_artifact_manifest checks it (lines 218-237)."""
    if not isinstance(m, dict) or m.get("schema_version") != 1:
        return [f"schema_version {None if not isinstance(m, dict) else m.get('schema_version')!r} != 1"]
    probs = []
    if not isinstance(m.get("run_id"), str) or not m["run_id"].strip():
        probs.append("run_id is not a non-empty string")
    if m.get("status") not in ("partial", "failed", "complete"):
        probs.append(f"status {m.get('status')!r} not partial, failed or complete")
    files = m.get("files")
    if not isinstance(files, list):
        return probs + ["files is not a list"]
    seen = set()
    for e in files:
        if not isinstance(e, dict):
            probs.append("an entry is not an object")
            continue
        p = e.get("path")
        if not isinstance(p, str) or not p or p in seen:
            probs.append(f"invalid or duplicate path {p!r}")
            continue
        pp = PurePosixPath(p)
        if pp.is_absolute() or any(part in ("", ".", "..") for part in pp.parts):   # as BR checks it (R-i2)
            probs.append(f"path not a normalized relative POSIX path: {p!r}")
        if "size_bytes" not in e or "sha256" not in e:
            probs.append(f"entry {p!r} lacks size_bytes or sha256")
        seen.add(p)
    return probs


def manifest_entries(m) -> dict:
    """path -> (size_bytes, hex) for the manifest's file entries."""
    out = {}
    for e in (m.get("files") or []) if isinstance(m, dict) else []:
        if isinstance(e, dict) and isinstance(e.get("path"), str):
            out[e["path"]] = (e.get("size_bytes"), hexpart(e.get("sha256")))
    return out


def r1_of(E):
    """R1 (v5, R-m1): the periodic pair 10,485,760; for a STOPPED run that ended before it, "the pair at the last
    multiple of 262,144 at or before E" (C3 then fails when that pair is not in the manifest).  None when E is unknown
    or no multiple of 262,144 lies at or before E (no R1 exists: C3 fails)."""
    if E is None:
        return None
    if E >= R1_STEP:
        return R1_STEP
    k = min(E, LAST_PERIODIC) // CKPT
    return k * CKPT if k >= 1 else None


def pair_paths(step) -> tuple[str, str]:
    if step == "final":
        return "01_stance/models/stage1_final.zip", "01_stance/models/stage1_final_vecnorm.pkl"
    return (f"01_stance/models/stage1_{step}_steps.zip", f"01_stance/models/stage1_vecnormalize_{step}_steps.pkl")


PERIODIC_ZIP_RE = re.compile(r"^stage1_(\d+)_steps\.zip$")


def periodic_steps_in(paths) -> set:
    """Periodic steps whose zip AND pkl a set of run-relative paths holds."""
    ps = set(paths)
    out = set()
    for p in ps:
        m = re.fullmatch(r"01_stance/models/stage1_(\d+)_steps\.zip", p)
        if m and f"01_stance/models/stage1_vecnormalize_{m.group(1)}_steps.pkl" in ps:
            out.add(int(m.group(1)))
    return out


# ------------------------------------------------------------------------------------------------- fetch store
FETCH_MANIFEST_COLUMNS = ("utc", "fetch", "run_id", "path", "drive_id", "listing_size", "transfer", "decoded_size",
                          "status", "copy", "sha256", "stored_as", "method", "note")


class FetchStore:
    """One run's files in one fetch, as fetch_r10round.py recorded them:
        <fetch_dir>/copies/<run-relative path>.copy<k>   every COPY (decoded size = the listing's)
        <fetch_dir>/listings/<folder key>.json           each listing, pages followed to the last
        <ROUND>/fetch/manifest.csv                        every transfer (COPY or TRANSFER-ERROR)
    Read only.  Each accessor states the rule it applies."""

    def __init__(self, fetch_dir, fetch_manifest=None, fetch_name=None, run_id=None):
        self.dir = Path(fetch_dir)
        self.fetch_manifest = Path(fetch_manifest) if fetch_manifest else None
        self.fetch_name = fetch_name
        self.run_id = run_id
        self._rows = None

    def copies(self, path) -> list[tuple[int, bytes, Path]]:
        base = self.dir / "copies" / path
        out = []
        if base.parent.is_dir():
            for f in sorted(base.parent.iterdir()):
                m = re.fullmatch(re.escape(base.name) + r"\.copy(\d+)", f.name)
                if m:
                    out.append((int(m.group(1)), f.read_bytes(), f))
        return sorted(out, key=lambda x: x[0])

    def agreed(self, path):
        """(bytes, how): two copies that agree byte for byte; on a mismatch a third, read when two of the three
        agree (PRESENCE, F0, C2's manifest, D-16).  (None, why) otherwise."""
        cs = self.copies(path)
        if not cs:
            return None, "absent from this fetch (no copy)"
        if len(cs) == 1:
            return None, "one copy only (two agreeing copies needed)"
        if cs[0][1] == cs[1][1]:
            return cs[0][1], f"copies 1 and 2 agree (sha256 {sha256_bytes(cs[0][1])[:12]})"
        if len(cs) < 3:
            return None, "copies 1 and 2 differ; no third copy"
        for i, j in ((0, 2), (1, 2)):
            if cs[i][1] == cs[j][1]:
                return cs[i][1], f"copies 1 and 2 differ; copy 3 agrees with copy {i + 1}"
        return None, "three copies, no two agree"

    def matching(self, path, want_hex, want_size=None):
        """(bytes, how): a copy whose sha256 equals the manifest's entry (and its size the entry's), among at most
        three copies (F2 (4)-(5)); (None, why) otherwise."""
        cs = self.copies(path)[:3]
        if not cs:
            return None, "not fetched"
        for k, b, _f in cs:
            if sha256_bytes(b) == want_hex and (want_size is None or len(b) == want_size):
                return b, f"copy {k} equals the manifest entry"
        return None, f"{len(cs)} cop{'y' if len(cs) == 1 else 'ies'}, none equal to the manifest entry"

    def listing(self, key):
        p = self.dir / "listings" / f"{key}.json"
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def manifest_rows(self):
        if self._rows is None:
            self._rows = []
            if self.fetch_manifest and self.fetch_manifest.is_file():
                with open(self.fetch_manifest, newline="", encoding="utf-8") as fh:
                    self._rows = [r for r in csv.DictReader(fh)
                                  if (self.fetch_name is None or r.get("fetch") == self.fetch_name)
                                  and (self.run_id is None or r.get("run_id") == self.run_id)]
        return self._rows

    def copy_recorded(self, path, raw: bytes) -> bool:
        """True when the fetch manifest records a COPY of path with this sha256 (PROVISIONAL otherwise)."""
        h = sha256_bytes(raw)
        return any(r.get("path") == path and r.get("status") == "COPY" and r.get("sha256") == h
                   for r in self.manifest_rows())


# ------------------------------------------------------------------------------------------------- M1 / M3 records
def read_csv_maybe_gz(path: Path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def find_record(label_dir: Path) -> dict:
    """The runner's record beside an M1/M3 output (any *.json in the label folder other than the tool's own
    reports, a refusal or a crash record, holding "physics_revision"): physics_revision, measuring commit,
    checkpoint sha256."""
    for f in sorted(label_dir.glob("*.json")):
        if f.name in ("episode_trace.json", "stance_heading_report.json", "stance_gate_report.json") or \
                f.name.endswith(("_refused.json", "_run_failed.json")):
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "physics_revision" in d:
            return dict(d, _file=str(f))
    return {}


def read_trace(label_dir) -> dict:
    """M1's own output (episode_trace.json and the per-episode CSVs, gzipped or not) as the classes read it:
    per episode its length and, per tail actuator, the window-mean command, the share at |command| >= 0.99 and the
    sign of most of its saturated window steps (from the CSV).  Raises InputError when unreadable."""
    d = Path(label_dir)
    j = d / "episode_trace.json"
    if not j.is_file():
        raise InputError(f"no episode_trace.json in {d}")
    rep = json.loads(j.read_text(encoding="utf-8"))
    rec = find_record(d)
    eps = []
    for ep in rep.get("episodes", []):
        seed, length = ep.get("seed"), ep.get("length")
        tail = {}
        for a in TAIL:
            name = ACTUATOR_NAMES[a]
            e = (ep.get("actuators") or {}).get(name)
            if e is None:
                raise InputError(f"{j}: episode {seed} has no {name} summary (trace every actuator)")
            tail[a] = dict(cmd=e.get("command_window_mean"), share99=e.get("saturated_fraction"), sign=None)
        flagged = [a for a in TAIL if (tail[a]["share99"] or 0.0) > BAR]
        if flagged:
            csvs = [d / f"episode_trace_seed{seed}.csv", d / f"episode_trace_seed{seed}.csv.gz"]
            src = next((c for c in csvs if c.is_file()), None)
            if src is None:
                raise InputError(f"{d}: episode {seed} is over the bar on a tail actuator but its per-step CSV is "
                                 "absent (the flag's sign cannot be read)")
            rows = read_csv_maybe_gz(src)
            for a in flagged:
                col = f"{ACTUATOR_NAMES[a]}_command"
                up = dn = 0
                for r in rows:
                    if str(r.get("in_window")) != "1":
                        continue
                    c = num(r.get(col))
                    if c is None:
                        continue
                    up += c >= 0.99
                    dn += c <= -0.99
                if up == 0 and dn == 0:
                    raise InputError(f"{src}: {ACTUATOR_NAMES[a]} is over the bar in episode {seed} but its CSV holds "
                                     "no window step at |command| >= 0.99 (the flag's sign cannot be read)")
                tail[a]["sign"] = 1 if up > dn else (-1 if dn > up else 0)      # 0: an exact tie (v5, O-7)
        if isinstance(length, (int, float)) and length > WINDOW_START:
            nowin = [ACTUATOR_NAMES[a] for a in TAIL if num(tail[a]["cmd"]) is None]
            if nowin:
                raise InputError(f"{j}: episode {seed} lasts {length} steps (beyond step {WINDOW_START}) but has no "
                                 f"window-mean command for {nowin}")
        eps.append(dict(seed=seed, length=length, full=(length is not None and length >= FULL_HORIZON), tail=tail))
    out = dict(label_dir=str(d), seeds=rep.get("seeds"), deterministic=rep.get("deterministic"),
               holds=rep.get("holds"), checkpoint=rep.get("checkpoint"), episodes=eps,
               physics_revision=rec.get("physics_revision"), commit=record_commit(rec), record=rec,
               binding=record_binding(rec))
    out["cross_check"] = m1_cross_check(out, rec)
    return out


def record_binding(rec: dict) -> dict:
    """Fixer (R-B4): what a runner's record says it measured: the pair's sha256, the checkpoint's recorded
    num_timesteps and seed, the validation mark and the RUN_ID/label it was filed under."""
    ck = rec.get("checkpoint") if isinstance(rec.get("checkpoint"), dict) else {}
    rr = rec.get("run_records") if isinstance(rec.get("run_records"), dict) else {}
    filed = rec.get("filed") if isinstance(rec.get("filed"), dict) else {}
    return dict(checkpoint_sha256=ck.get("sha256"), vecnorm_sha256=ck.get("vecnorm_sha256"),
                num_timesteps=ck.get("num_timesteps"), seed=ck.get("seed"),
                validation_no_run_records=rr.get("validation_no_run_records"), run_git_commit=rr.get("git_commit"),
                filed_run_id=filed.get("run_id"), filed_label=filed.get("label"))


def m1_cross_check(tr: dict, rec: dict):
    """Fixer (RA-8), report only: lane A's own reduction (m1_summary.json tail_flags and cmd_a) against this
    reader's (flags with their signs, CMD_a).  None when the record carries no reduction."""
    if not isinstance(rec.get("tail_flags"), list) or not isinstance(rec.get("cmd_a"), dict):
        return None
    sign = {"+": 1, "-": -1, "=": 0}
    a_flags = sorted((int(f["seed"]), int(f["actuator"]), sign.get(f.get("sign"))) for f in rec["tail_flags"])
    b_flags = sorted((int(seed), a, sg) for (a, sg, full, seed) in trace_flags(tr) if full)
    diffs = []
    if a_flags != b_flags:
        diffs.append(dict(tail_flags_lane_a=a_flags, tail_flags_here=b_flags))
    for a in TAIL:
        x = rec["cmd_a"].get(str(a))
        y = trace_cmd(tr, a)[0]
        if (x is None) != (y is None) or (x is not None and abs(x - y) > 1e-12):
            diffs.append(dict(actuator=ACTUATOR_NAMES[a], cmd_a_lane_a=x, cmd_a_here=y))
    return dict(equal=not diffs, differences=diffs)


def read_refusal(label_dir, kind: str):
    """Fixer (RA-5): the runner's record of the measuring tool's own refusal (m1_refused.json / m3_refused.json),
    or None."""
    f = Path(label_dir) / f"{kind}_refused.json"
    if not f.is_file():
        return None
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(message=f"{f.name} unreadable", record={}, file=str(f))
    return dict(message=d.get("message"), record=d, file=str(f), binding=record_binding(d))


def record_commit(rec: dict):
    """The measuring copy's commit a runner's record names (lane A: copy_seal.commit; else commit keys)."""
    seal = rec.get("copy_seal") if isinstance(rec.get("copy_seal"), dict) else {}
    return seal.get("commit") or rec.get("measuring_commit") or rec.get("commit")


def flag_signs(sg) -> tuple:
    """The signs a flag counts in (v5, O-7, lane A's form): its sign, or both on an exact tie (sign 0): "on an exact
    tie the flag counts once in each sign"."""
    return (1, -1) if sg == 0 else (sg,)


def trace_flags(tr) -> list[tuple[int, int, bool, int]]:
    """Every (actuator, sign, full_horizon, seed) with a tail actuator over the BAR (TAIL FLAG when full-horizon,
    FALL FLAG otherwise); sign 1, -1, or 0 for an exact tie (counted in each sign by flag_signs)."""
    out = []
    for ep in tr["episodes"]:
        for a in TAIL:
            t = ep["tail"][a]
            if (t["share99"] or 0.0) > BAR:
                out.append((a, t["sign"], ep["full"], ep["seed"]))
    return out


def trace_cmd(tr, a):
    """(CMD_a(X), n, others) (v5, O-6): "the mean, over those of M1's 20 episodes at X that last beyond step 200, of
    each one's window-mean command of tail actuator a (the others are printed); with none, CMD_a(X) is not formed and
    reads as below 0.60 in either sign".  n: the episodes averaged; others: the seeds of the episodes that ended by step
    200 (printed).  CMD_a is None when no episode lasts beyond step 200."""
    used, others = [], []
    for ep in tr["episodes"]:
        L = ep.get("length")
        if isinstance(L, (int, float)) and L > WINDOW_START:
            used.append(float(ep["tail"][a]["cmd"]))
        else:
            others.append(ep.get("seed"))
    if not used:
        return None, 0, others
    return sum(used) / len(used), len(used), others


def reaches(v, sign) -> bool:
    """v >= 0.60 in sign (+1 / -1); a CMD_a that is not formed (None) "reads as below 0.60 in either sign" (v5, O-6)."""
    return v is not None and sign * v >= PARK_LINE


def mean_a(x, y):
    """MEAN_a from x = CMD_a(R1) and y = CMD_a(R2) (None: not formed), criteria lean v6 lines 184-187 (RUNLOG line 661,
    V5C-2 option A): "MEAN_a = (CMD_a(R1) + CMD_a(R2)) / 2; when one of CMD_a(R1), CMD_a(R2) is not formed, MEAN_a is
    the formed one's CMD_a, so a park the readable checkpoint shows is never hidden; with neither formed, MEAN_a is not
    formed and reads as below 0.60 in either sign".  Returns the number, or None (not formed) with neither formed."""
    if x is not None and y is not None:
        return (x + y) / 2.0
    return x if x is not None else y


def mean_reaches(x, y, sign) -> bool:
    """Whether MEAN_a (mean_a) is >= 0.60 in sign (+1 / -1): True or False, never open (v6).  A MEAN_a that is not
    formed reads as below 0.60 in either sign (reaches)."""
    return reaches(mean_a(x, y), sign)


def read_certificate(label_dir) -> dict:
    """M3's own verdict (stance_heading_report.json "passed", "failures") with the runner's record."""
    d = Path(label_dir)
    j = d / "stance_heading_report.json"
    if not j.is_file():
        raise InputError(f"no stance_heading_report.json in {d}")
    rep = json.loads(j.read_text(encoding="utf-8"))
    rec = find_record(d)
    return dict(passed=bool(rep.get("passed")), failures=rep.get("failures"), seeds=rep.get("seeds"),
                physics_revision=rec.get("physics_revision"), commit=record_commit(rec), record=rec, label_dir=str(d),
                binding=record_binding(rec))


# ------------------------------------------------------------------------------------------------- statistic A
TABLE_E = {  # A of n_A: (Fisher p vs 2/5, P(r10 < r9), P(r10 < r9/2), one-sided 95% upper bound)  [criteria TABLE E]
    (0, 3): (0.36, 0.83, 0.59, 0.63), (1, 3): (0.71, 0.55, 0.22, 0.86), (2, 3): (0.93, 0.26, 0.05, 0.98),
    (3, 3): (1.00, 0.07, 0.004, 1.0), (0, 2): (0.48, 0.76, 0.50, 0.78), (1, 2): (0.86, 0.41, 0.13, 0.97),
    (2, 2): (1.00, 0.12, 0.015, 1.0), (0, 1): (0.67, 0.64, 0.38, 0.95), (1, 1): (1.00, 0.21, 0.05, 1.0),
}


def fisher_lower(a: int, n: int, ref_a: int = 2, ref_n: int = 5) -> float:
    """One-sided Fisher exact test (r10 lower): P(X <= a), X hypergeometric with the round's n of N = n + ref_n runs
    drawn, K = a + ref_a alarms in all."""
    N, K = n + ref_n, a + ref_a
    tot = math.comb(N, n)
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(0, a + 1) if n - x <= N - K) / tot


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tool_hashes() -> dict:
    """sha256 of every lane B tool file (PROVISIONAL compares them with their pins)."""
    names = ("r10round_common.py", "ledger_r10round.py", "complete_r10round.py", "score_r10round.py",
             "fetch_r10round.py", "k3panel_r10round.py", "v9_bundle.py")
    return {n: sha256_file(TOOLS / n) for n in names if (TOOLS / n).is_file()}


def add_minutes(t: datetime, m: float) -> datetime:
    return t + timedelta(minutes=m)
