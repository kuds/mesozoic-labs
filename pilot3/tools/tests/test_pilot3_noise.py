"""Tests of pilot3_noise.py, the report-only noise and value readout (pytest; walking pilot 3 scratch; FIX2).

The TFRecord/Event reader is checked on records built here by hand (simple_value and tensor scalars, a cut file, a
flipped bit) and, when the notebook smoke's run is on this machine, on its real event file (the values the criteria
quote). The checkpoint part reads a synthetic zip's policy.pth; the forward part a synthetic gate_progress.npz.
(The reader was also compared with TensorBoard's own EventAccumulator on the smoke's three event files: FIX2
out/noise/validate_tb_reader.txt.) FIX3: a damaged input prints "not run" for its part alone; no --json overwrite.
"""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot3_noise as pn  # noqa: E402

SMOKE_RUN = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/"
                 "pilot3/smoke/clone/logs/tyrannosaurus_rex/anatomical/ppo_quick_test/pilot3_walk_s42")


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def field(num: int, wire: int, payload: bytes) -> bytes:
    key = varint((num << 3) | wire)
    return key + (varint(len(payload)) + payload if wire == 2 else payload)


def event(step: int, values: list) -> bytes:
    """An Event with wall_time, step and a Summary of (tag, float) simple values or ("tensor", tag, float) values."""
    vals = b""
    for v in values:
        if v[0] == "tensor":
            tensor = field(1, 0, varint(1)) + field(5, 2, struct.pack("<f", v[2]))
            vals += field(1, 2, field(1, 2, v[1].encode()) + field(8, 2, tensor))
        else:
            vals += field(1, 2, field(1, 2, v[0].encode()) + field(2, 5, struct.pack("<f", v[1])))
    return field(1, 1, struct.pack("<d", 1.0)) + field(2, 0, varint(step)) + field(5, 2, vals)


def record(data: bytes) -> bytes:
    n = struct.pack("<Q", len(data))
    return n + struct.pack("<I", pn.masked_crc(n)) + data + struct.pack("<I", pn.masked_crc(data))


def test_crc32c_known_vector():
    assert pn.crc32c(b"123456789") == 0xE3069283


def test_records_and_scalars_round_trip():
    blob = record(field(3, 2, b"brain.Event:2")) + record(event(16384, [("train/std", 0.25), ("x", 1.5)])) \
        + record(event(32768, [("tensor", "train/std", 0.5)]))
    recs, stopped = pn.tfrecords(blob)
    assert stopped is None and len(recs) == 3
    assert pn.event_scalars(recs[0]) == []
    assert pn.event_scalars(recs[1]) == [("train/std", 16384, 0.25), ("x", 16384, 1.5)]
    assert pn.event_scalars(recs[2]) == [("train/std", 32768, 0.5)]


def test_a_cut_or_damaged_file_keeps_the_records_before_the_damage():
    blob = record(event(1, [("a", 1.0)])) + record(event(2, [("a", 2.0)]))
    recs, stopped = pn.tfrecords(blob[:-3])
    assert len(recs) == 1 and stopped.startswith("cut record")
    bad = bytearray(blob)
    bad[-6] ^= 0x10
    recs, stopped = pn.tfrecords(bytes(bad))
    assert len(recs) == 1 and stopped.startswith("data CRC mismatch")
    recs, stopped = pn.tfrecords(blob[:5])
    assert recs == [] and stopped.startswith("cut record header")


def write_run(root: Path, *, std_steps=(16384, 32768, 300000, 316384, 600000), forward=True, zips=True) -> Path:
    stage = root / "pilot3_walk_s42" / "03_locomotion"
    (stage / "tensorboard" / "PPO_1").mkdir(parents=True)
    blob = record(field(3, 2, b"brain.Event:2"))
    for i, s in enumerate(std_steps):
        blob += record(event(s, [("train/std", 0.22 + 0.001 * i), ("train/value_loss", 1.0 / (i + 1)),
                                 ("train/explained_variance", 0.5 + 0.1 * i)]))
    (stage / "tensorboard" / "PPO_1" / "events.out.tfevents.1.host.1.0").write_bytes(blob)
    if zips:
        import torch

        models = stage / "models"
        models.mkdir()
        for steps, log_std in ((500000, [-1.5, -1.4, -1.6]), (1000000, [-2.0, -2.0, -2.0])):
            buf = io.BytesIO()
            torch.save({"log_std": torch.tensor(log_std, dtype=torch.float32)}, buf)
            with zipfile.ZipFile(models / f"stage2_{steps}_steps.zip", "w") as z:
                z.writestr("policy.pth", buf.getvalue())
                z.writestr("data", "{}")
    if forward:
        np.savez(stage / "gate_progress.npz", timesteps=np.array([131072, 262144]),
                 term_forward=np.array([0.5, 3.25]), full_horizon_term_forward=np.array([0.4, 3.0]))
    return stage.parent


def test_the_three_parts_on_a_synthetic_run(tmp_path):
    run = write_run(tmp_path)
    out = tmp_path / "n.json"
    assert pn.main([str(run), "--json", str(out)]) == 0
    rec = json.loads(out.read_text())
    p = rec["checkpoints"]["periodic"]
    assert [r["steps"] for r in p] == [500000, 1000000]
    want = np.exp(np.array([-1.5, -1.4, -1.6], dtype=np.float32).astype(np.float64))
    assert math.isclose(p[0]["mean"], want.mean()) and math.isclose(p[0]["min"], want.min())
    assert math.isclose(p[1]["max"], math.exp(-2.0), rel_tol=1e-6) and rec["checkpoints"]["final"] is None
    std = rec["tensorboard"]["train/std"]
    assert [s for s, _v in std["warmup"]] == [16384, 32768, 300000] and std["n"] == 5
    assert [(b["from"], b["n"]) for b in std["blocks"]] == [(0, 4), (500000, 1)]
    assert rec["tensorboard"]["train/value_loss"]["warmup"][0] == [16384, 1.0]
    assert rec["forward"]["rows"] == [{"timesteps": 131072, "term_forward": 0.5, "full_horizon_term_forward": 0.4},
                                      {"timesteps": 262144, "term_forward": 3.25, "full_horizon_term_forward": 3.0}]
    assert rec["references"]["warm_start_std"] == pytest.approx(0.1353, abs=5e-5)


def test_missing_parts_print_not_run(tmp_path, capsys):
    run = write_run(tmp_path, forward=False, zips=False)
    assert pn.main([str(run)]) == 0
    text = capsys.readouterr().out
    assert "exp(log_std) at the checkpoints" in text and "not run: no folder" in text
    assert "forward-reward term per evaluation: not run: no " in text
    empty = tmp_path / "empty"
    empty.mkdir()
    assert pn.main([str(empty)]) == 0
    assert "TensorBoard: not run: no event file" in capsys.readouterr().out
    assert pn.main([str(tmp_path / "nowhere")]) == 2


def test_a_damaged_input_prints_not_run_and_the_other_parts_still_print(tmp_path, capsys):
    """FIX3 (the third review's M4): a gate_progress.npz cut in half (a partial sync) and an event file whose record
    passes its CRCs but does not decode print "not run" for their part; the checkpoints still print; exit 0."""
    run = write_run(tmp_path)
    stage = run / "03_locomotion"
    npz = stage / "gate_progress.npz"
    npz.write_bytes(npz.read_bytes()[: npz.stat().st_size // 2])
    out = tmp_path / "n.json"
    assert pn.main([str(run), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "forward-reward term per evaluation: not run: gate_progress.npz could not be read (" in text
    rec = json.loads(out.read_text())
    assert [r["steps"] for r in rec["checkpoints"]["periodic"]] == [500000, 1000000]
    assert rec["tensorboard"]["train/std"]["n"] == 5
    bad = record(field(1, 1, struct.pack("<d", 1.0)) + bytes([(5 << 3) | 3]))  # wire type 3: not decodable
    (stage / "tensorboard" / "PPO_1" / "events.out.tfevents.2.host.1.0").write_bytes(bad)
    assert pn.main([str(run)]) == 0
    text = capsys.readouterr().out
    assert "TensorBoard: not run: the event files could not be read (ValueError: unsupported wire type 3)" in text
    assert "   500,000  mean 0.22" in text and "1,000,000  mean 0.1353" in text  # the checkpoints still print


def test_an_existing_json_target_is_never_overwritten(tmp_path, capsys):
    run = write_run(tmp_path, forward=False, zips=False)
    target = tmp_path / "binding.json"
    target.write_text('{"judged": true}')
    assert pn.main([str(run), "--json", str(target)]) == 2
    assert target.read_text() == '{"judged": true}' and "never overwritten" in capsys.readouterr().out


def test_a_resume_that_repeats_steps_is_noted(tmp_path, capsys):
    run = write_run(tmp_path, std_steps=(16384, 32768, 32768, 49152), forward=False, zips=False)
    pn.main([str(run)])
    assert "steps do not increase throughout" in capsys.readouterr().out


@pytest.mark.skipif(not (SMOKE_RUN / "03_locomotion" / "tensorboard").is_dir(), reason="the smoke's run is absent")
def test_the_smoke_run_reads_the_values_the_criteria_quote(tmp_path):
    out = tmp_path / "smoke.json"
    assert pn.main([str(SMOKE_RUN), "--json", str(out)]) == 0
    rec = json.loads(out.read_text())
    std = rec["tensorboard"]["train/std"]["all"]
    assert [(s, round(v, 4)) for s, v in std] == [(32768, 0.2245), (49152, 0.225), (65536, 0.2252)]
    vl = rec["tensorboard"]["train/value_loss"]["warmup"]
    assert round(vl[0][1], 2) == 1.05 and round(vl[-1][1], 2) == 0.16
    assert [round(r["mean"], 4) for r in rec["checkpoints"]["periodic"]] == [0.2245, 0.2252]
    assert [r["timesteps"] for r in rec["forward"]["rows"]] == [32768, 65536]
