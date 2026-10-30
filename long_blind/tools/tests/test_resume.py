"""Unit tests of bs_resume.py (the resume of a crashed continuation) on synthetic attempt folders: no simulation, no
torch, no code copy.  A small grid (S = 64, rollout 8, evaluation 16, save 32) keeps the files tiny; the functions take
the grid as arguments, and run_bs.py passes bs_common's.

    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <LB>/tools/tests/test_resume.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bs_resume", str(TOOLS / "bs_resume.py"))
RS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RS)

S, R, G, SAVE, E, EP = 64, 8, 16, 32, 64 + 8 * 40, 3          # E = S + 40 rollouts
SAVED = tuple(range(S + SAVE, E, SAVE))                          # 96, 128, ..., 352
STAGE = "01_stance"


def names(steps):
    return f"stage1_{steps}_steps.zip", f"stage1_vecnormalize_{steps}_steps.pkl"


def _zip(path, steps, members=("data", "policy.pth")):
    with zipfile.ZipFile(path, "w") as z:
        for m in members:
            z.writestr(m, json.dumps({"num_timesteps": steps}) if m == "data" else b"weights" * 50)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_attempt(root, *, probe_to, evidence_to, eval_to, train_to, pairs_to, name="critic_blind_tail3_9301_crashed1"):
    """A synthetic last attempt: pairs at S and every saved step <= pairs_to; the probe's rows (the parent's seeded ones
    <= S, then one per rollout) to probe_to; evidence lines (S, evidence_to]; EP ledger rows per evaluation to eval_to;
    evaluations.npz and gate_progress.npz to eval_to; training-ledger episodes ending every 5 steps to train_to."""
    run = Path(root) / name
    st = run / STAGE
    (st / "models").mkdir(parents=True)
    for s in (S, *[x for x in SAVED if x <= pairs_to]):
        z, p = names(s)
        _zip(st / "models" / z, s)
        (st / "models" / p).write_bytes(pickle.dumps({"obs_rms": s}))
    steps = np.arange(R, probe_to + 1, R, dtype=np.int64)
    n = len(steps)
    np.savez(st / RS.PROBE, update_timesteps=steps, ent_coef=np.linspace(0.005, 0.004, n).astype(np.float32),
             learning_rate=np.full(n, 3e-5, np.float32), obs_rms_mean=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
             obs_rms_var=np.ones((n, 3), np.float32), obs_rms_count=np.arange(n, dtype=np.float64),
             bank_action_now=np.zeros((n, 2, 2), np.float32))
    with open(run / RS.EVIDENCE, "w") as fh:
        for k in range(1, (evidence_to - S) // R + 1):
            fh.write(json.dumps({"num_timesteps": S + R * k, "dv": 0.0}) + "\n")
    with open(st / RS.EVAL_LEDGER, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["evaluation", "timesteps", "episode", "reward"])
        for i, t in enumerate(range(G, eval_to + 1, G)):
            for e in range(EP):
                w.writerow([i, t, e, f"{t * 0.5 + e:.3f}"])
    ts = np.arange(G, eval_to + 1, G, dtype=np.int64)
    np.savez(st / RS.EVALS_NPZ, timesteps=ts, results=np.ones((len(ts), EP)), ep_lengths=np.full((len(ts), EP), 9))
    np.savez(st / RS.GATE_NPZ, timesteps=ts, unsupported_duty=np.zeros(len(ts)))
    with open(st / RS.TRAIN_LEDGER, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["timesteps", "cause", "length"])
        for t in range(5, train_to + 1, 5):
            w.writerow([t, "truncated", 5])
    return run


def point(run, **kw):
    return RS.resume_point(run, S=S, E=E, saved_steps=SAVED, periodic_names=names,
                           start_sha256=(_sha(run / STAGE / "models" / names(S)[0]),
                                         _sha(run / STAGE / "models" / names(S)[1])),
                           stage_dir_name=STAGE, rollout=R, eval_freq=G, eval_episodes=EP, **kw)


# ------------------------------------------------------------------------------------------------ the resume point
def test_the_newest_complete_pair_is_chosen(tmp_path):
    """The code's flush clocks: the newest pair (224) is saved, its probe row and the training ledger after it are not
    on disk yet; the pair before it (192) has every row: P = 192, the newer one named with its reason."""
    run = make_attempt(tmp_path, probe_to=216, evidence_to=232, eval_to=224, train_to=224, pairs_to=224)
    pt = point(run)
    assert pt["P"] == 192 and pt["problems"] == []
    assert [c["steps"] for c in pt["candidates"]] == [224, 192]
    assert "policy_probe.npz" in pt["candidates"][0]["problem"] and "train_episodes" in pt["candidates"][0]["problem"]
    assert pt["zip_sha256"] == _sha(run / STAGE / "models" / names(192)[0])


def test_all_complete_takes_the_newest(tmp_path):
    run = make_attempt(tmp_path, probe_to=256, evidence_to=256, eval_to=256, train_to=260, pairs_to=256)
    assert point(run)["P"] == 256


@pytest.mark.parametrize("damage", ["truncated_zip", "no_pkl", "no_members", "bad_pkl", "wrong_steps", "no_zip"])
def test_a_damaged_newest_pair_falls_back(tmp_path, damage):
    run = make_attempt(tmp_path, probe_to=256, evidence_to=256, eval_to=256, train_to=260, pairs_to=256)
    z, p = (run / STAGE / "models" / n for n in names(256))
    if damage == "truncated_zip":
        z.write_bytes(z.read_bytes()[: len(z.read_bytes()) // 2])
    elif damage == "no_pkl":
        p.unlink()
    elif damage == "no_members":
        _zip(z, 256, members=("other",))
    elif damage == "bad_pkl":
        p.write_bytes(b"not a pickle")
    elif damage == "wrong_steps":
        _zip(z, 224)
    else:
        z.unlink()
    pt = point(run)
    assert pt["P"] == 224 and pt["candidates"][0]["steps"] == 256 and pt["candidates"][0]["problem"]


@pytest.mark.parametrize("damage", ["probe_gap", "probe_repeat", "evidence_cut", "evidence_order", "ledger_short",
                                    "ledger_repeat", "evals_npz_gap", "gate_gap", "train_before"])
def test_incomplete_records_are_refused_as_resume_points(tmp_path, damage):
    """Each record missing (or repeating) a row up to the newest pair (256) makes it no resume point; the attempt's
    older pair (224) still is when the damage lies after it."""
    run = make_attempt(tmp_path, probe_to=256, evidence_to=256, eval_to=256, train_to=260, pairs_to=256)
    st = run / STAGE
    if damage in ("probe_gap", "probe_repeat"):
        with np.load(st / RS.PROBE) as z:
            d = {k: np.asarray(z[k]) for k in z.files}
        i = int(np.flatnonzero(d["update_timesteps"] == 240)[0])
        keep = [j for j in range(len(d["update_timesteps"])) if j != i] if damage == "probe_gap" else \
            sorted([*range(len(d["update_timesteps"])), i])
        np.savez(st / RS.PROBE, **{k: v[keep] for k, v in d.items()})
    elif damage == "evidence_cut":
        lines = (run / RS.EVIDENCE).read_text().splitlines(keepends=True)
        (run / RS.EVIDENCE).write_text("".join(lines[:-1]) + lines[-1][:-5])
    elif damage == "evidence_order":
        lines = (run / RS.EVIDENCE).read_text().splitlines(keepends=True)
        lines[-2], lines[-1] = lines[-1], lines[-2]
        (run / RS.EVIDENCE).write_text("".join(lines))
    elif damage in ("ledger_short", "ledger_repeat"):
        rows = (st / RS.EVAL_LEDGER).read_text().splitlines(keepends=True)
        last = [i for i, r in enumerate(rows) if r.split(",")[1] == "240"]
        if damage == "ledger_short":
            del rows[last[-1]]
        else:
            rows.insert(last[-1], rows[last[-1]])
        (st / RS.EVAL_LEDGER).write_text("".join(rows))
    elif damage in ("evals_npz_gap", "gate_gap"):
        name = RS.EVALS_NPZ if damage == "evals_npz_gap" else RS.GATE_NPZ
        with np.load(st / name) as z:
            d = {k: np.asarray(z[k]) for k in z.files}
        keep = d["timesteps"] != 240
        np.savez(st / name, **{k: v[keep] for k, v in d.items()})
    else:
        rows = (st / RS.TRAIN_LEDGER).read_text().splitlines(keepends=True)
        kept = [r for r in rows if not r[:1].isdigit() or int(r.split(",")[0]) <= 255]
        (st / RS.TRAIN_LEDGER).write_text("".join(kept))
    pt = point(run)
    assert pt["candidates"][0]["steps"] == 256 and pt["candidates"][0]["problem"]
    assert pt["P"] == 224, (damage, pt["candidates"])


def test_no_resume_point_and_a_bad_start_copy(tmp_path):
    run = make_attempt(tmp_path, probe_to=88, evidence_to=104, eval_to=96, train_to=96, pairs_to=96)
    pt = point(run)                          # the only pair (96) is not complete yet: a fresh start from S
    assert pt["P"] is None and [c["steps"] for c in pt["candidates"]] == [96] and pt["problems"] == []
    run2 = make_attempt(tmp_path, probe_to=256, evidence_to=256, eval_to=256, train_to=260, pairs_to=256, name="x")
    pt2 = RS.resume_point(run2, S=S, E=E, saved_steps=SAVED, periodic_names=names, start_sha256=("0" * 64, "1" * 64),
                          stage_dir_name=STAGE, rollout=R, eval_freq=G, eval_episodes=EP)
    assert pt2["P"] is None and len(pt2["problems"]) == 2 and pt2["candidates"] == []


def test_an_earlier_pair_missing_refuses_the_later_one(tmp_path):
    run = make_attempt(tmp_path, probe_to=256, evidence_to=256, eval_to=256, train_to=260, pairs_to=256)
    (run / STAGE / "models" / names(160)[1]).unlink()
    pt = point(run)
    assert pt["P"] == 128 and [c["steps"] for c in pt["candidates"]] == [256, 224, 192, 160, 128]
    assert all("earlier saved pairs" in c["problem"] for c in pt["candidates"][:3])
    assert "sidecar" in pt["candidates"][3]["problem"] and pt["candidates"][4]["problem"] is None


# ------------------------------------------------------------------------------------------------ the new folder
def test_resume_inputs_and_stitching(tmp_path):
    """The new folder: the pairs S..P linked (same bytes, same inode), the records copied, the evidence lines up to P
    byte for byte; the stitched digests over rows <= P equal before and after the session appends rows after P and the
    code trims the rows after P (simulated as the code does: kept rows' text unchanged); a changed row <= P fails."""
    src = make_attempt(tmp_path, probe_to=216, evidence_to=232, eval_to=224, train_to=224, pairs_to=224)
    P = 192
    before = RS.stitch_digests(src / STAGE, src, P, S=S, rollout=R)
    new = tmp_path / "critic_blind_tail3_9301"
    new.mkdir()
    rec = RS.resume_inputs(new / STAGE, new, src, P, S=S, saved_steps=SAVED, periodic_names=names,
                           stage_dir_name=STAGE, records=(RS.EVAL_LEDGER, RS.EVALS_NPZ, RS.GATE_NPZ, RS.TRAIN_LEDGER,
                                                          RS.PROBE), rollout=R)
    assert set(rec["pairs"]) == {n for s in (S, *[x for x in SAVED if x <= P]) for n in names(s)}
    assert rec["links"] == len(rec["pairs"]) and rec["evidence_lines"] == (P - S) // R
    zp = new / STAGE / "models" / names(P)[0]
    assert zp.stat().st_ino == (src / STAGE / "models" / names(P)[0]).stat().st_ino
    assert not (new / STAGE / "models" / names(224)[0]).exists()
    src_lines = (src / RS.EVIDENCE).read_text().splitlines(keepends=True)
    assert (new / RS.EVIDENCE).read_text() == "".join(src_lines[: (P - S) // R])
    assert RS.stitch_problems(before, RS.stitch_digests(new / STAGE, new, P, S=S, rollout=R)) == []
    # the session: rows <= P kept as they were, its own rows after P appended
    st = new / STAGE
    with np.load(st / RS.PROBE) as z:
        d = {k: np.asarray(z[k]) for k in z.files}
    keep = d["update_timesteps"] <= P
    d = {k: v[keep] for k, v in d.items()}
    extra = np.arange(P + R, P + 6 * R, R, dtype=np.int64)
    d2 = {k: np.concatenate([v, np.repeat(v[-1:], len(extra), axis=0)]) for k, v in d.items()}
    d2["update_timesteps"] = np.concatenate([d["update_timesteps"], extra])
    np.savez(st / RS.PROBE, **d2)
    with open(new / RS.EVIDENCE, "a") as fh:
        for t in extra:
            fh.write(json.dumps({"num_timesteps": int(t), "dv": 1.0}) + "\n")
    rows = (st / RS.EVAL_LEDGER).read_text().splitlines(keepends=True)
    kept = [r for r in rows if not r[:1].isdigit() or int(r.split(",")[1]) <= P]
    (st / RS.EVAL_LEDGER).write_text("".join(kept) + "".join(f"9,{P + G},{e},1.0\n" for e in range(EP)))
    after = RS.stitch_digests(st, new, P, S=S, rollout=R,
                              headers={n: before[n]["header"] for n in (RS.EVAL_LEDGER, RS.TRAIN_LEDGER)})
    assert RS.stitch_problems(before, after) == []
    # a row <= P changed: CHECK fails
    rows = (st / RS.EVAL_LEDGER).read_text().splitlines(keepends=True)
    rows[1] = rows[1].replace(",0,", ",7,", 1)
    (st / RS.EVAL_LEDGER).write_text("".join(rows))
    bad = RS.stitch_digests(st, new, P, S=S, rollout=R)
    assert any(RS.EVAL_LEDGER in x for x in RS.stitch_problems(before, bad))
    assert RS.stitch_problems({}, after) and RS.stitch_problems(before, {})


def test_resume_inputs_refuses_an_incomplete_evidence_prefix(tmp_path):
    src = make_attempt(tmp_path, probe_to=216, evidence_to=120, eval_to=224, train_to=224, pairs_to=224)
    new = tmp_path / "new"
    new.mkdir()
    with pytest.raises(RS.ResumeError):
        RS.resume_inputs(new / STAGE, new, src, 192, S=S, saved_steps=SAVED, periodic_names=names,
                         stage_dir_name=STAGE, records=(RS.PROBE,), rollout=R)


# ------------------------------------------------------------------------------------------------ CHECK's grid
def test_evaluation_grid(tmp_path):
    run = make_attempt(tmp_path, probe_to=256, evidence_to=256, eval_to=256, train_to=260, pairs_to=256)
    st = run / STAGE
    p, det = RS.evaluation_grid_problems(st, 256, eval_freq=G, eval_episodes=EP)
    assert p == [] and det["evaluations_expected"] == 16 and det["ledger_repeated"] == 0
    assert RS.evaluation_grid_problems(st, 272, eval_freq=G, eval_episodes=EP)[0]        # one missing at 272
    assert RS.evaluation_grid_problems(st, 240, eval_freq=G, eval_episodes=EP)[0]        # rows after T
    rows = (st / RS.EVAL_LEDGER).read_text().splitlines(keepends=True)
    (st / RS.EVAL_LEDGER).write_text("".join(rows + [rows[5]]))
    assert any("repeated" in x for x in RS.evaluation_grid_problems(st, 256, eval_freq=G, eval_episodes=EP)[0])


def test_local_pair_problem_matches_the_code_rules(tmp_path):
    z, p = tmp_path / "a.zip", tmp_path / "a.pkl"
    _zip(z, 1)
    p.write_bytes(pickle.dumps(1))
    assert RS.local_pair_problem(z, p) is None
    assert "sidecar" in RS.local_pair_problem(z, tmp_path / "none.pkl")
    _zip(z, 1, members=("data",))
    assert "SB3 members" in RS.local_pair_problem(z, p)
    z.write_bytes(b"PK\x03\x04 truncated")
    assert "bad/truncated" in RS.local_pair_problem(z, p)
    assert RS.zip_num_timesteps(z) is None


# ------------------------------------------------------------------------------------------------ the seeded copies
def test_a_record_never_rewritten_by_the_attempt_is_not_its_own(tmp_path):
    """A record still byte-equal to the copy the attempt started from (the parent's F2 record, or the attempt before's)
    holds no row of the attempt after its load step: no pair after the load step is complete for it.  (The window
    between a pair's save and the probe's flush on the same call, or a flush that failed, would otherwise stitch
    foreign rows.)  Pairs at or before the load step (a resumed attempt's linked ones) stay complete."""
    run = make_attempt(tmp_path, probe_to=256, evidence_to=256, eval_to=256, train_to=260, pairs_to=256)
    st = run / STAGE
    fresh = {RS.PROBE: _sha(st / RS.PROBE), RS.TRAIN_LEDGER: _sha(st / RS.TRAIN_LEDGER)}
    pt = point(run, seeded=fresh, own_load=S)
    assert pt["P"] is None and all("never rewritten" in c["problem"] for c in pt["candidates"])
    assert RS.PROBE in pt["candidates"][0]["problem"] and RS.TRAIN_LEDGER in pt["candidates"][0]["problem"]
    pt = point(run, seeded=fresh, own_load=192)          # a resumed attempt that loaded 192: 192 and older still are
    assert pt["P"] == 192 and [c["steps"] for c in pt["candidates"]] == [256, 224, 192]
    other = {RS.PROBE: "0" * 64, RS.TRAIN_LEDGER: "1" * 64}  # rewritten by the attempt: judged by content as before
    assert point(run, seeded=other, own_load=S)["P"] == 256
    assert point(run, seeded=None, own_load=None)["P"] == 256
