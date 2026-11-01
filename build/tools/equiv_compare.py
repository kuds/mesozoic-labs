"""Compare two equiv_run.py outputs bitwise.  usage: python -I -B equiv_compare.py DIR_A DIR_B

1. dump.npz: every array (per-update rollout buffers, parameters, Adam, VecNormalize statistics, counters) with
   np.array_equal (equal_nan for float arrays: a NaN must sit at the same place).
2. The run directory: every file under run/ by relative path.  .npz files array by array; .zip (SB3 archives) member
   by member, policy.pth / pytorch_variables.pth / policy.optimizer.pth tensors by value and the "data" JSON with
   the keys that name a wall clock or a path dropped; .pkl and other files byte for byte, except files that record
   wall-clock times or the code tree's provenance (listed and skipped by name).
Prints one line per difference and a summary; exit 1 if anything differs.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

#: Files that record wall clocks, durations or the tree's own path / git state (the two trees differ there by
#: construction); their numbers are not training outputs.
SKIP_NAMES = {"stage_config.json", "stage_duration.json", "progress.json", "run_provenance.json"}
VOLATILE_DATA_KEYS = {"_last_obs", "_last_episode_starts", "_last_original_obs", "start_time", "_logger", "tensorboard_log"}
#: npz arrays that are wall-clock measurements ("descriptive only" in policy_probe.py).
VOLATILE_ARRAYS = {"bank_build_seconds"}


def _mask_addresses(value):
    """The data JSON writes a class's members as reprs carrying memory addresses: compare them with those masked."""
    import re

    return re.sub(r"0x[0-9a-f]+", "0xADDR", json.dumps(value, sort_keys=True))


def _episodes(value):
    """ep_info_buffer: the episodes' returns and lengths (r, l), not their wall-clock times (t)."""
    import base64
    import pickle

    episodes = pickle.loads(base64.b64decode(value[":serialized:"]))
    return [(episode["r"], episode["l"]) for episode in episodes]


def arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def compare_npz(a: Path, b: Path, label: str, problems: list[str]) -> int:
    with np.load(a, allow_pickle=False) as x, np.load(b, allow_pickle=False) as y:
        if set(x.files) != set(y.files):
            problems.append(f"{label}: keys differ {sorted(set(x.files) ^ set(y.files))}")
            return 0
        for key in x.files:
            if key in VOLATILE_ARRAYS:
                continue
            if not arrays_equal(x[key], y[key]):
                problems.append(f"{label}:{key} differs")
        return len(x.files)


def compare_zip(a: Path, b: Path, label: str, problems: list[str]) -> int:
    import torch

    n = 0
    with zipfile.ZipFile(a) as x, zipfile.ZipFile(b) as y:
        if sorted(x.namelist()) != sorted(y.namelist()):
            problems.append(f"{label}: members differ")
            return 0
        for name in x.namelist():
            pa, pb = x.read(name), y.read(name)
            if name.endswith(".pth"):
                ta = torch.load(io.BytesIO(pa), map_location="cpu", weights_only=False)
                tb = torch.load(io.BytesIO(pb), map_location="cpu", weights_only=False)
                flat_a, flat_b = _flatten(ta), _flatten(tb)
                if flat_a.keys() != flat_b.keys():
                    problems.append(f"{label}/{name}: keys differ")
                for key in flat_a:
                    n += 1
                    if not arrays_equal(flat_a[key], flat_b.get(key, np.array([]))):
                        problems.append(f"{label}/{name}:{key} differs")
            elif name == "data":
                da, db = json.loads(pa), json.loads(pb)
                for key in sorted(set(da) | set(db)):
                    if key in VOLATILE_DATA_KEYS:
                        continue
                    n += 1
                    va, vb = da.get(key), db.get(key)
                    if key == "ep_info_buffer":
                        same = _episodes(va) == _episodes(vb)
                    elif isinstance(va, dict) and ":serialized:" in va and isinstance(vb, dict):
                        # cloudpickle bytes exact; the readable member reprs with their addresses masked
                        same = va[":serialized:"] == vb.get(":serialized:") and _mask_addresses(va) == _mask_addresses(vb)
                    else:
                        same = va == vb
                    if not same:
                        problems.append(f"{label}/data:{key} differs")
            else:
                n += 1
                if pa != pb:
                    problems.append(f"{label}/{name} bytes differ")
    return n


def _flatten(obj, prefix=""):
    import torch

    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}{index}."))
    elif isinstance(obj, torch.Tensor):
        out[prefix] = obj.detach().cpu().numpy()
    else:
        out[prefix] = np.asarray(obj)
    return out


def main() -> int:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    problems: list[str] = []
    n_dump = compare_npz(a / "dump.npz", b / "dump.npz", "dump", problems)
    files_a = {p.relative_to(a / "run"): p for p in (a / "run").rglob("*") if p.is_file()}
    files_b = {p.relative_to(b / "run"): p for p in (b / "run").rglob("*") if p.is_file()}
    if set(files_a) != set(files_b):
        problems.append(f"run files differ: {sorted(map(str, set(files_a) ^ set(files_b)))}")
    compared, skipped = [], []
    for rel in sorted(set(files_a) & set(files_b)):
        pa, pb = files_a[rel], files_b[rel]
        if rel.name in SKIP_NAMES:
            skipped.append(str(rel))
            continue
        if rel.suffix == ".npz":
            compare_npz(pa, pb, str(rel), problems)
        elif rel.suffix == ".zip":
            compare_zip(pa, pb, str(rel), problems)
        elif pa.read_bytes() != pb.read_bytes():
            problems.append(f"{rel} bytes differ")
        compared.append(str(rel))
    for line in problems:
        print("DIFF", line)
    print(f"dump arrays compared: {n_dump}; run files compared: {len(compared)} {compared}; skipped (clocks/provenance): {skipped}")
    print("RESULT", "IDENTICAL" if not problems else f"{len(problems)} DIFFERENCES")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
