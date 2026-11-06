"""Tests of pilot4_noise.py, the report-only noise and value readout (pytest; walking pilot 4 scratch, never merged).

Walking pilot 3's tests/test_pilot3_noise.py (sha256 3a315964...) adapted to pilot 4's TOOL PLAN: the warm-start
reference is each seed's parent, its own pilot-3 8M zip's exp(log_std) (mean, min, max); no warm-up section
(WARMUP_STEPS 0); train/std blocks of 250,000; train/* over the first 65,536 steps (the text's "Report only" list: the
fresh ret_rms); the forward reference is pilot 3's own evaluation at 8M on R6's scale. Pilot 3's cases are kept: the
TFRecord/Event reader on records built here by hand (simple_value and tensor scalars, a cut file, a flipped bit); the
checkpoint part on a synthetic zip's policy.pth; the forward part on a synthetic gate_progress.npz; a damaged input
printing "not run" for its part alone; no --json overwrite; and, when pilot 3's notebook smoke is on this machine, its
real event file (a check of the reader). New: the parent references are re-measured from the three verified pilot-3
8M zips (sha256 checked against the text's table) and the forward references from pilot 3's run copies' own
gate_progress.npz, when those copies are on this machine; the seed is read from the RUN_ID.
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
import pilot4_noise as pn  # noqa: E402

SP = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad")
SMOKE_RUN = SP / "work/pilot3/smoke/clone/logs/tyrannosaurus_rex/anatomical/ppo_quick_test/pilot3_walk_s42"
PILOT3_RUNS = SP / "work/pilot3/runs/final"


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


STD_STEPS = (16384, 32768, 49152, 65536, 81920, 250000, 266384, 500000, 516384)


def write_run(root: Path, *, name="pilot4_walk_s42", std_steps=STD_STEPS, forward=True, zips=True) -> Path:
    stage = root / name / "03_locomotion"
    (stage / "tensorboard" / "PPO_1").mkdir(parents=True)
    blob = record(field(3, 2, b"brain.Event:2"))
    for i, s in enumerate(std_steps):
        blob += record(event(s, [("train/std", 0.14 + 0.001 * i), ("train/value_loss", 1.0 / (i + 1)),
                                 ("train/explained_variance", 0.5 + 0.01 * i)]))
    (stage / "tensorboard" / "PPO_1" / "events.out.tfevents.1.host.1.0").write_bytes(blob)
    if zips:
        import torch

        models = stage / "models"
        models.mkdir()
        for steps, log_std in ((250000, [-1.9, -2.0, -1.8]), (500000, [-2.0, -2.0, -2.0])):
            buf = io.BytesIO()
            torch.save({"log_std": torch.tensor(log_std, dtype=torch.float32)}, buf)
            with zipfile.ZipFile(models / f"stage2_{steps}_steps.zip", "w") as z:
                z.writestr("policy.pth", buf.getvalue())
                z.writestr("data", "{}")
    if forward:
        np.savez(stage / "gate_progress.npz", timesteps=np.array([65536, 131072]),
                 term_forward=np.array([1700.5, 1850.25]), full_horizon_term_forward=np.array([1701.0, 1850.0]))
    return stage.parent


def test_the_tool_plan_s_constants():
    assert pn.WARMUP_STEPS == 0 and pn.BLOCK == 250_000 and pn.FIRST_STEPS == 65_536
    assert pn.SCHEMA == "walking-pilot4-noise/v1" and pn.SEEDS == (42, 52, 60)
    assert not hasattr(pn, "WARM_START_STD") and not hasattr(pn, "REF_FORWARD")
    for s in pn.SEEDS:
        p = pn.PARENT_STD[s]
        assert p["min"] < p["mean"] < p["max"] and pn.FORWARD_REF[s][0] == 7_995_392
    # the text's rounded values ("Parent and entry"): exp mean 0.1467 / 0.1460 / 0.1451, min 0.1063 / 0.1061 / 0.0956,
    # max 0.1817 / 0.1816 / 0.1898
    assert [round(pn.PARENT_STD[s]["mean"], 4) for s in pn.SEEDS] == [0.1467, 0.1460, 0.1451]
    assert [round(pn.PARENT_STD[s]["min"], 4) for s in pn.SEEDS] == [0.1063, 0.1061, 0.0956]
    assert [round(pn.PARENT_STD[s]["max"], 4) for s in pn.SEEDS] == [0.1817, 0.1816, 0.1898]
    assert pn.PARENT_SHA256 == {
        42: "sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
        52: "sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
        60: "sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2"}


@pytest.mark.parametrize("name, seed", [("pilot4_walk_s42", 42), ("pilot4_walk_s52_b", 52), ("pilot4_walk_s60", 60),
                                        ("pilot4_walk_s7", None), ("pilot3_walk_s42", None),
                                        ("pilot4_walk_s42_c", None), ("xpilot4_walk_s42", None), ("run", None)])
def test_the_seed_is_read_from_the_run_id(name, seed):
    assert pn.run_seed(Path("/copies") / name) == seed
    refs = pn.references(seed)
    assert list(refs["parent_std"]) == ([str(seed)] if seed else ["42", "52", "60"])
    assert list(refs["forward_pilot3_8M"]) == list(refs["parent_std"]) == list(refs["parent_sha256"])


def test_the_three_parts_on_a_synthetic_run(tmp_path, capsys):
    run = write_run(tmp_path)
    out = tmp_path / "n.json"
    assert pn.main([str(run), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    rec = json.loads(out.read_text())
    p = rec["checkpoints"]["periodic"]
    assert [r["steps"] for r in p] == [250000, 500000]
    want = np.exp(np.array([-1.9, -2.0, -1.8], dtype=np.float32).astype(np.float64))
    assert math.isclose(p[0]["mean"], want.mean()) and math.isclose(p[0]["min"], want.min())
    assert math.isclose(p[1]["max"], math.exp(-2.0), rel_tol=1e-6) and rec["checkpoints"]["final"] is None
    std = rec["tensorboard"]["train/std"]
    assert [s for s, _v in std["first_steps"]] == [16384, 32768, 49152, 65536] and std["n"] == 9
    assert "warmup" not in std
    assert [(b["from"], b["n"]) for b in std["blocks"]] == [(0, 6), (250000, 2), (500000, 1)]
    assert rec["tensorboard"]["train/value_loss"]["first_steps"][0] == [16384, 1.0]
    assert [s for s, _v in rec["tensorboard"]["train/explained_variance"]["first_steps"]] == [16384, 32768, 49152,
                                                                                             65536]
    assert rec["forward"]["rows"] == [
        {"timesteps": 65536, "term_forward": 1700.5, "full_horizon_term_forward": 1701.0},
        {"timesteps": 131072, "term_forward": 1850.25, "full_horizon_term_forward": 1850.0}]
    refs = rec["references"]
    assert refs["seed"] == 42 and refs["parent_std"] == {"42": pn.PARENT_STD[42]} and refs["warmup_steps"] == 0
    assert refs["forward_pilot3_8M"] == {"42": {"steps": 7_995_392, "term_forward": pn.FORWARD_REF[42][1],
                                                "full_horizon_term_forward": pn.FORWARD_REF[42][2]}}
    assert rec["schema"] == "walking-pilot4-noise/v1" and rec["report_only"] is True
    assert "the warm start is the parent's own, mean/min/max: s42 parent 0.1467/0.1063/0.1817" in text
    assert ("first 65,536 steps (the fresh ret_rms): 16,384: 0.1400, 32,768: 0.1410, 49,152: 0.1420, 65,536: "
            "0.1430") in text
    assert "warm-up" not in text and "warm start 0.1353" not in text and "0.057" not in text  # pilot 3's references
    assert "   250,000  mean 0.1501  min 0.1353  max 0.1653" in text
    assert "pilot 3's own evaluation at 8M, R6's scale: s42 1943.495 (full horizon 1943.495) at 7,995,392" in text
    assert "     0-250,000    n   6" in text and "250,000-500,000    n   2" in text


def test_a_run_that_is_not_a_pilot_4_run_id_is_compared_with_every_parent(tmp_path, capsys):
    run = write_run(tmp_path, name="pilot4_walk_s42_rtest", zips=False, forward=False)
    out = tmp_path / "n.json"
    assert pn.main([str(run), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    rec = json.loads(out.read_text())
    assert rec["references"]["seed"] is None and list(rec["references"]["parent_std"]) == ["42", "52", "60"]
    assert "every parent shown" in text and "s52 parent 0.1460/0.1061/0.1816" in text
    assert "s60 parent 0.1451/0.0956/0.1898" in text


def test_a_run_whose_first_65536_steps_logged_nothing_says_so(tmp_path, capsys):
    run = write_run(tmp_path, std_steps=(81920, 98304), zips=False, forward=False)
    assert pn.main([str(run)]) == 0
    assert "first 65,536 steps (the fresh ret_rms): none logged" in capsys.readouterr().out


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
    """Pilot 3's FIX3 case: a gate_progress.npz cut in half (a partial sync) and an event file whose record passes its
    CRCs but does not decode print "not run" for their part; the checkpoints still print; exit 0."""
    run = write_run(tmp_path)
    stage = run / "03_locomotion"
    npz = stage / "gate_progress.npz"
    npz.write_bytes(npz.read_bytes()[: npz.stat().st_size // 2])
    out = tmp_path / "n.json"
    assert pn.main([str(run), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "forward-reward term per evaluation: not run: gate_progress.npz could not be read (" in text
    rec = json.loads(out.read_text())
    assert [r["steps"] for r in rec["checkpoints"]["periodic"]] == [250000, 500000]
    assert rec["tensorboard"]["train/std"]["n"] == 9
    bad = record(field(1, 1, struct.pack("<d", 1.0)) + bytes([(5 << 3) | 3]))  # wire type 3: not decodable
    (stage / "tensorboard" / "PPO_1" / "events.out.tfevents.2.host.1.0").write_bytes(bad)
    assert pn.main([str(run)]) == 0
    text = capsys.readouterr().out
    assert "TensorBoard: not run: the event files could not be read (ValueError: unsupported wire type 3)" in text
    assert "   250,000  mean 0.1501" in text and "   500,000  mean 0.1353" in text  # the checkpoints still print


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


needs_pilot3_runs = pytest.mark.skipif(not all((PILOT3_RUNS / f"pilot3_walk_s{s}" / "03_locomotion").is_dir()
                                               for s in (42, 52, 60)), reason="pilot 3's run copies are absent")


@needs_pilot3_runs
@pytest.mark.parametrize("seed", [42, 52, 60])
def test_the_parent_references_are_the_parent_zips_own(seed):
    """Each seed's reference is its own pilot-3 8M zip's exp(log_std), read by this tool's log_std_stats (policy.pth
    through torch.load weights_only, nothing else unpickled), exactly; the zip is the text's table's (sha256)."""
    z = PILOT3_RUNS / f"pilot3_walk_s{seed}" / "03_locomotion" / "models" / "stage2_8000000_steps.zip"
    assert pn.sha256_file(z) == pn.PARENT_SHA256[seed]
    got = pn.log_std_stats(z)
    assert got == dict(pn.PARENT_STD[seed], n=15)


@needs_pilot3_runs
@pytest.mark.parametrize("seed", [42, 52, 60])
def test_the_forward_references_are_pilot_3_s_own_evaluation_at_8m(seed):
    """Pilot 3's last in-training evaluation at or before 8,000,000 (its 61st, 7,995,392), from its own
    gate_progress.npz through this tool's forward part (np.load, allow_pickle False)."""
    fw = pn.forward_part(PILOT3_RUNS / f"pilot3_walk_s{seed}" / "03_locomotion")
    rows = [r for r in fw["rows"] if r["timesteps"] <= 8_000_000]
    assert len(fw["rows"]) == 61 and rows[-1]["timesteps"] == 7_995_392
    assert (rows[-1]["timesteps"], rows[-1]["term_forward"], rows[-1]["full_horizon_term_forward"]) == \
        pn.FORWARD_REF[seed]


@pytest.mark.skipif(not (SMOKE_RUN / "03_locomotion" / "tensorboard").is_dir(), reason="the smoke's run is absent")
def test_the_reader_on_pilot_3_s_smoke_run(tmp_path, capsys):
    """Pilot 3's notebook smoke (FIX2, -1.5) as a check of the reader on real event files: the values pilot 3's
    criteria quote. Its folder name is not a pilot-4 RUN_ID, so every parent is printed."""
    out = tmp_path / "smoke.json"
    assert pn.main([str(SMOKE_RUN), "--json", str(out)]) == 0
    rec = json.loads(out.read_text())
    std = rec["tensorboard"]["train/std"]["all"]
    assert [(s, round(v, 4)) for s, v in std] == [(32768, 0.2245), (49152, 0.225), (65536, 0.2252)]
    vl = rec["tensorboard"]["train/value_loss"]["first_steps"]
    assert round(vl[0][1], 2) == 1.05 and round(vl[-1][1], 2) == 0.16
    assert [round(r["mean"], 4) for r in rec["checkpoints"]["periodic"]] == [0.2245, 0.2252]
    assert [r["timesteps"] for r in rec["forward"]["rows"]] == [32768, 65536]
    assert rec["references"]["seed"] is None
