"""Walking pilot 4's report-only noise and value readout (criteria_pilot4 draft v3, "Report only": "Noise and value").

Scratch, never merged: walking pilot 3's pilot3_noise.py (sha256 c191c72a...) adapted as the pilot-4 pre-registration's
TOOL PLAN says: the warm-start reference becomes each parent's exp(log_std) (mean, min, max per seed), no warm-up
section (WARMUP_STEPS 0), BLOCK 250,000, and the forward reference is pilot 3's own evaluation at 8M on R6's scale.
REPORT ONLY: never part of an outcome or the existence reading. Pinned by sha256 before launch. It reads a local copy
of one run (the RUN_ID folder pilot4_walk_s<seed> or pilot4_walk_s<seed>_b, paths as on Drive) and prints, each
against the text's references:
  * exp(log_std) mean/min/max at every periodic checkpoint 03_locomotion/models/stage2_<steps>_steps.zip (and the
    final model when present), from the zip's policy.pth (torch.load weights_only; nothing else unpickled), against
    the seed's parent at the warm start: its own pilot-3 8M zip's exp(log_std) (PARENT_STD, measured from that zip
    by this file's log_std_stats; pilot 4 keeps each walker's log_std, decision (26)). The seed is read from the run
    folder's RUN_ID; a folder that is not one of seeds 42, 52, 60 is compared with all three parents;
  * TensorBoard's train/std (every logged value), per 250,000-step block (mean, min, max); train/std,
    train/value_loss and train/explained_variance over the first 65,536 steps (FIRST_STEPS, the fresh ret_rms: the
    text's "Report only" list); no warm-up section (pilot 4 has no warm-up). The event files
    03_locomotion/tensorboard/*/events.out.tfevents.* are read by this file's own TFRecord reader (length, masked
    CRC-32C of length and data checked; a scalar's simple_value, or a scalar tensor's float or double value), so no
    TensorBoard install is needed; a damaged or cut record ends that file's reading and is printed;
  * the forward-reward term per in-training evaluation, gate_progress.npz term_forward (the evaluation's mean per
    episode; full_horizon_term_forward beside it), against pilot 3's own in-training evaluation at 8M (its last at or
    before 8,000,000: 7,995,392) on R6's scale (FORWARD_REF, measured from pilot 3's run copies; np.load with
    allow_pickle False).
A part whose input is missing is printed "not run" with the reason, never estimated; so is a part whose input cannot
be read (a gate_progress.npz cut short by a partial sync, an event file the reader cannot decode), and the other
parts are still printed (pilot 3's FIX3 rule). Exit 0 always (report only), except 2 for a RUN that is not a folder
and for an existing --json target, which is never overwritten (nothing is written).

    python3 -I TOOLS/pilot4_noise.py RUN [--json NOISE/s42.json]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import struct
import sys
import zipfile
from pathlib import Path

SCHEMA = "walking-pilot4-noise/v1"
SEEDS = (42, 52, 60)
#: Each seed's parent pair zip (the text's table, "Parent and entry") and its exp(log_std) at the warm start, measured
#: from that zip by log_std_stats below [measured: SP/work/pilot4/tools/out/adapt/logs/measure_refs.txt; the text's
#: 0.1467 / 0.1460 / 0.1451, min 0.1063 / 0.1061 / 0.0956, max 0.1817 / 0.1816 / 0.1898 are these, rounded].
PARENT_SHA256 = {
    42: "sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
    52: "sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
    60: "sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2",
}
PARENT_STD = {
    42: {"mean": 0.14665701759809271, "min": 0.10631248123897372, "max": 0.18167976718752318},
    52: {"mean": 0.1460313846708207, "min": 0.10611695806782012, "max": 0.18163541718965057},
    60: {"mean": 0.14507681139782455, "min": 0.09555350251971369, "max": 0.18984233446899185},
}
WARMUP_STEPS = 0  # pilot 4 has no warm-up (warmup_timesteps 0): no warm-up section
FIRST_STEPS = 65_536  # train/* over the first 65,536 steps (the fresh ret_rms; the text's "Report only")
BLOCK = 250_000
TAGS = ("train/std", "train/value_loss", "train/explained_variance")
PERIODIC_RE = re.compile(r"stage2_(\d+)_steps\.zip$")
RUN_ID_RE = re.compile(r"pilot4_walk_s(\d+)(_b)?")
#: Pilot 3's own in-training evaluation at 8M (the last at or before 8,000,000), R6's scale, per seed:
#: (steps, term_forward, full_horizon_term_forward) from SP/work/pilot3/runs/final/pilot3_walk_s<seed>/03_locomotion/
#: gate_progress.npz [measured: SP/work/pilot4/tools/out/adapt/logs/measure_refs.txt].
FORWARD_REF = {
    42: (7_995_392, 1943.4951736510022, 1943.4951736510025),
    52: (7_995_392, 1912.660498694539, 1912.6604986945397),
    60: (7_995_392, 1897.8457585330175, 1897.845758533018),
}


def run_seed(run: "str | Path") -> "int | None":
    """The seed a run folder's RUN_ID names (pilot4_walk_s<seed> or pilot4_walk_s<seed>_b), when it is a pilot seed."""
    m = RUN_ID_RE.fullmatch(Path(run).name)
    seed = int(m.group(1)) if m else None
    return seed if seed in SEEDS else None


def references(seed: "int | None") -> dict:
    """The parent and forward references printed for a run of `seed` (all three seeds' when it is None)."""
    seeds = [seed] if seed in SEEDS else list(SEEDS)
    return {"seed": seed, "parent_std": {str(s): PARENT_STD[s] for s in seeds},
            "parent_sha256": {str(s): PARENT_SHA256[s] for s in seeds},
            "forward_pilot3_8M": {str(s): {"steps": FORWARD_REF[s][0], "term_forward": FORWARD_REF[s][1],
                                           "full_horizon_term_forward": FORWARD_REF[s][2]} for s in seeds},
            "warmup_steps": WARMUP_STEPS, "first_steps": FIRST_STEPS, "block": BLOCK}


def sha256_file(path: "str | Path") -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------- CRC-32C and the TFRecord / Event reader

def _crc32c_table() -> list:
    poly, table = 0x82F63B78, []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly if c & 1 else c >> 1
        table.append(c)
    return table


_CRC_TABLE = _crc32c_table()


def crc32c(data: bytes) -> int:
    c = 0xFFFFFFFF
    t = _CRC_TABLE
    for b in data:
        c = t[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def masked_crc(data: bytes) -> int:
    c = crc32c(data)
    return ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


def tfrecords(blob: bytes) -> "tuple[list[bytes], str | None]":
    """(the records, None) or (the records before the first damaged or cut one, why it stopped)."""
    out, i, n = [], 0, len(blob)
    while i < n:
        if n - i < 12:
            return out, f"cut record header at byte {i}"
        (length,) = struct.unpack("<Q", blob[i:i + 8])
        (len_crc,) = struct.unpack("<I", blob[i + 8:i + 12])
        if masked_crc(blob[i:i + 8]) != len_crc:
            return out, f"length CRC mismatch at byte {i}"
        if n - i < 12 + length + 4:
            return out, f"cut record at byte {i} (length {length})"
        data = blob[i + 12:i + 12 + length]
        (data_crc,) = struct.unpack("<I", blob[i + 12 + length:i + 16 + length])
        if masked_crc(data) != data_crc:
            return out, f"data CRC mismatch at byte {i}"
        out.append(data)
        i += 16 + length
    return out, None


def _varint(buf: bytes, i: int) -> "tuple[int, int]":
    shift = result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """(field number, wire type, value) of a protobuf message: varint int, fixed64/fixed32 bytes, or bytes."""
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(buf, i)
        elif wire == 1:
            v, i = buf[i:i + 8], i + 8
        elif wire == 2:
            ln, i = _varint(buf, i)
            v, i = buf[i:i + ln], i + ln
        elif wire == 5:
            v, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, v


def _tensor_scalar(buf: bytes) -> "float | None":
    """A TensorProto's single float (field 5) or double (field 6) value, packed or not; else None."""
    vals = []
    for f, w, v in _fields(buf):
        if f == 5:
            vals += list(struct.unpack(f"<{len(v) // 4}f", v)) if w == 2 else [struct.unpack("<f", v)[0]]
        elif f == 6:
            vals += list(struct.unpack(f"<{len(v) // 8}d", v)) if w == 2 else [struct.unpack("<d", v)[0]]
    return float(vals[0]) if len(vals) == 1 else None


def event_scalars(record: bytes) -> list:
    """[(tag, step, value)] of one Event record's summary values that hold a scalar."""
    step, summary, out = 0, None, []
    for f, _w, v in _fields(record):
        if f == 2:
            step = v
        elif f == 5:
            summary = v
    if summary is None:
        return out
    for f, _w, v in _fields(summary):
        if f != 1:
            continue
        tag, value = None, None
        for g, w2, x in _fields(v):
            if g == 1:
                tag = x.decode("utf-8", "replace")
            elif g == 2 and w2 == 5:
                value = float(struct.unpack("<f", x)[0])
            elif g == 8 and w2 == 2 and value is None:
                value = _tensor_scalar(x)
        if tag is not None and value is not None:
            out.append((tag, int(step), value))
    return out


def read_events(paths: list) -> dict:
    """{"scalars": {tag: [(step, value, file)]}, "files": [{path, records, stopped}]} over the event files in order."""
    scalars: dict = {}
    files = []
    for p in paths:
        recs, stopped = tfrecords(Path(p).read_bytes())
        for r in recs:
            for tag, step, value in event_scalars(r):
                scalars.setdefault(tag, []).append((step, value, str(p)))
        files.append({"path": str(p), "records": len(recs), "stopped": stopped})
    return {"scalars": scalars, "files": files}


# ---------------------------------------------------------------- the parts

def log_std_stats(zip_path: "str | Path") -> dict:
    import numpy as np
    import torch

    with zipfile.ZipFile(zip_path) as zf:
        state = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True)
    std = np.exp(state["log_std"].detach().cpu().numpy().astype(np.float64))
    return {"mean": float(std.mean()), "min": float(std.min()), "max": float(std.max()), "n": int(std.size)}


def checkpoint_noise(models: Path) -> dict:
    if not models.is_dir():
        return {"not_run": f"no folder {models}"}
    rows = []
    for p in sorted(models.glob("stage2_*_steps.zip"), key=lambda q: int(PERIODIC_RE.match(q.name).group(1))
                    if PERIODIC_RE.match(q.name) else -1):
        m = PERIODIC_RE.match(p.name)
        if not m:
            continue
        try:
            rows.append({"steps": int(m.group(1)), "file": p.name, **log_std_stats(p)})
        except Exception as exc:  # report only: an unreadable checkpoint is printed, never estimated
            rows.append({"steps": int(m.group(1)), "file": p.name, "not_run": f"{type(exc).__name__}: {exc}"})
    final = models / "stage2_final.zip"
    fin = None
    if final.is_file():
        try:
            fin = {"file": final.name, **log_std_stats(final)}
        except Exception as exc:
            fin = {"file": final.name, "not_run": f"{type(exc).__name__}: {exc}"}
    if not rows and fin is None:
        return {"not_run": f"no stage2_<steps>_steps.zip or stage2_final.zip in {models}"}
    return {"periodic": rows, "final": fin}


def tensorboard_part(stage: Path) -> dict:
    paths = sorted(stage.glob("tensorboard/*/events.out.tfevents.*"))
    if not paths:
        return {"not_run": f"no event file under {stage / 'tensorboard'}"}
    try:  # FIX3: an unreadable input is "not run", never a crash of the whole readout
        ev = read_events(paths)
    except Exception as exc:
        return {"not_run": f"the event files could not be read ({type(exc).__name__}: {exc})"}
    out: dict = {"files": ev["files"], "tags_found": len(ev["scalars"])}
    for tag in TAGS:
        series = ev["scalars"].get(tag)
        if not series:
            out[tag] = {"not_run": f"no {tag} scalar in the event files"}
            continue
        early = [(s, v) for s, v, _f in series if s <= FIRST_STEPS]
        rec: dict = {"n": len(series), "first_steps": early, "first": series[0][:2], "last": series[-1][:2]}
        if tag == "train/std":
            blocks = {}
            for s, v, _f in series:
                blocks.setdefault((s - 1) // BLOCK if s > 0 else 0, []).append(v)
            rec["blocks"] = [{"from": b * BLOCK, "to": (b + 1) * BLOCK, "n": len(vs), "mean": sum(vs) / len(vs),
                              "min": min(vs), "max": max(vs)} for b, vs in sorted(blocks.items())]
            rec["all"] = [(s, v) for s, v, _f in series]
            steps = [s for s, _v, _f in series]
            rec["steps_not_increasing"] = any(b <= a for a, b in zip(steps, steps[1:], strict=False))
        out[tag] = rec
    return out


def forward_part(stage: Path) -> dict:
    path = stage / "gate_progress.npz"
    if not path.is_file():
        return {"not_run": f"no {path}"}
    import numpy as np

    try:  # FIX3: a cut or damaged file (a partial sync) is "not run", never a crash of the whole readout
        with np.load(path, allow_pickle=False) as d:
            if "timesteps" not in d.files or "term_forward" not in d.files:
                return {"not_run": f"{path.name} lacks timesteps or term_forward"}
            ts = [int(x) for x in d["timesteps"]]
            tf = [float(x) for x in d["term_forward"]]
            fh = ([float(x) for x in d["full_horizon_term_forward"]] if "full_horizon_term_forward" in d.files
                  else [None] * len(ts))
    except Exception as exc:
        return {"not_run": f"{path.name} could not be read ({type(exc).__name__}: {exc})"}
    if not (len(ts) == len(tf) == len(fh)):
        return {"not_run": f"{path.name}: timesteps, term_forward and full_horizon_term_forward differ in length"}
    return {"file": str(path), "sha256": sha256_file(path),
            "rows": [{"timesteps": s, "term_forward": a, "full_horizon_term_forward": b}
                     for s, a, b in zip(ts, tf, fh, strict=True)]}


def _g(x, fmt=".4f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def _parent_text(refs: dict) -> str:
    return "; ".join(f"s{s} parent {p['mean']:.4f}/{p['min']:.4f}/{p['max']:.4f}"
                     for s, p in refs["parent_std"].items())


def _forward_text(refs: dict) -> str:
    return "; ".join(f"s{s} {f['term_forward']:.3f} (full horizon {f['full_horizon_term_forward']:.3f}) at "
                     f"{f['steps']:,}" for s, f in refs["forward_pilot3_8M"].items())


def render(rec: dict) -> list:
    refs = rec["references"]
    lines = [f"Noise and value (report only), run {rec['run']}"
             + ("" if refs["seed"] is not None else " (not a pilot-4 RUN_ID of seed 42, 52 or 60: every parent shown)")]
    ck = rec["checkpoints"]
    lines.append(f"  exp(log_std) at the checkpoints (the warm start is the parent's own, mean/min/max: "
                 f"{_parent_text(refs)}):")
    if "not_run" in ck:
        lines.append(f"    not run: {ck['not_run']}")
    else:
        for r in ck["periodic"] + ([ck["final"]] if ck["final"] else []):
            where = f"{r['steps']:>10,}" if "steps" in r else f"{'final':>10}"
            lines.append(f"    {where}  " + (f"not run: {r['not_run']}" if "not_run" in r else
                                            f"mean {r['mean']:.4f}  min {r['min']:.4f}  max {r['max']:.4f}"))
    tb = rec["tensorboard"]
    if "not_run" in tb:
        lines.append(f"  TensorBoard: not run: {tb['not_run']}")
    else:
        for f in tb["files"]:
            lines.append(f"  TensorBoard file {f['path']}: {f['records']} records"
                         + (f"; reading stopped: {f['stopped']}" if f["stopped"] else ""))
        for tag in TAGS:
            t = tb[tag]
            if "not_run" in t:
                lines.append(f"  {tag}: not run: {t['not_run']}")
                continue
            lines.append(f"  {tag}: {t['n']} values, first {t['first'][1]:.4f} at {t['first'][0]:,}, last "
                         f"{t['last'][1]:.4f} at {t['last'][0]:,}; first {FIRST_STEPS:,} steps (the fresh ret_rms): "
                         + (", ".join(f"{s:,}: {v:.4f}" for s, v in t["first_steps"]) or "none logged"))
            for b in t.get("blocks", []):
                lines.append(f"    {b['from']:>10,}-{b['to']:<10,} n {b['n']:>3}  mean {b['mean']:.4f}  min "
                             f"{b['min']:.4f}  max {b['max']:.4f}")
            if t.get("steps_not_increasing"):
                lines.append(f"    note: {tag}'s steps do not increase throughout (a resume repeats steps)")
    fw = rec["forward"]
    if "not_run" in fw:
        lines.append(f"  forward-reward term per evaluation: not run: {fw['not_run']}")
    else:
        lines.append("  forward-reward term per evaluation (term_forward; full-horizon episodes beside; pilot 3's own "
                     f"evaluation at 8M, R6's scale: {_forward_text(refs)}):")
        for r in fw["rows"]:
            lines.append(f"    {r['timesteps']:>10,}  {_g(r['term_forward'], '.3f')}  "
                         f"(full horizon {_g(r['full_horizon_term_forward'], '.3f')})")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="a local copy of the run folder (RUN_ID), holding 03_locomotion/")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    if not run.is_dir():
        print(f"refusing: {run} is not a folder", flush=True)
        return 2
    if a.json and Path(a.json).exists():  # pilot 3's FIX3: never overwrite (a mistyped path could name a binding.json)
        print(f"refusing (nothing written): --json {a.json} exists (never overwritten)", flush=True)
        return 2
    stage = run / "03_locomotion"
    rec = {"schema": SCHEMA, "report_only": True, "tool_sha256": sha256_file(__file__), "run": str(run),
           "checkpoints": checkpoint_noise(stage / "models"), "tensorboard": tensorboard_part(stage),
           "forward": forward_part(stage), "references": references(run_seed(run))}
    print("\n".join(render(rec)), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
