"""``designlab.stage2.monitor2``: the monitor, G8b and the stop rules (round C design D15 and section 4's row
"Monitor, G8b, stop rules"; Track B rules M1-M5, M3b; the lead's rulings of 16 Nov 2026 on a check's attempts), and
``designlab.stage2.clip2`` (one clip from a monitor trace).

What is held here, on tables of episodes in ``lookbar.episodes``' form (``monitor2._episodes``, the module's one
indirection to ``lookbar``, is replaced by such a table; no episode is run):

* 18 distinct seeds, one episode each (6001-6018); a tiny run takes each condition's first; the row names;
* the gain rule: two more with the look; one second longer; against the best earlier check, N* and S* each taken
  separately and R0L's values staying in that best; the first check against R0L;
* a policy that stands (read, not succeeding), a policy whose clock lags (``unread``, reason ``clock_behind``) and
  a too-few-steps episode (read, its look terms failed): each counted as not succeeding, never an error, and the
  monitor asks for each episode exactly once;
* the survival credit: a clock held at 0.75 earns 0.75 of its time, R0L its full time, an error nothing;
* G8b's 2 of 3 (up is read or unread); the fix doubles the mapped tasks' weights only, each once; a second failure
  stops; three checks without a gain from 20M stop at the third; ``decide`` gives the same state from the records
  alone, however often and at whatever point of the history it is asked;
* a check counted without a record (every episode an error); what a record of a check is; a baseline that holds
  an error is kept beside as a failed attempt and is no baseline;
* S2-10's record with its earlier checks, its per-term table and M3b's route;
* the monitor refuses to run on other sources than the trainer's, and on a checkpoint that is not intact;
* ``monitor2``'s table of rule M4 is the task table's (``r5lib.TASKS``' ``monitor`` fields).

``clip2`` renders a real clip from a Stage 1 trace in a child process (OSMesa and ffmpeg; skipped without them)
and that process holds no torch; a clip that cannot be made is a record with its reason and exit code 0.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from designlab.stage2 import monitor2 as m2
from designlab.stage2 import s2io, tbenv

STUDY = Path(__file__).resolve().parents[1]
BODY = "75-M-H-10"
PLANNED = 12.0


# ---------------------------------------------------------------------------------------------------------------
# Tables of episodes
# ---------------------------------------------------------------------------------------------------------------
def ep(cond, seed, outcome="read", survival=PLANNED, look=False, ok=False, a1=False, a1_look=False, factor=1.0,
       terms=(), reason=None, planned=PLANNED) -> dict:
    """One companion episode as ``lookbar.episodes`` returns it."""
    fell = outcome == "fell"
    return dict(condition=cond, seed=seed, outcome=outcome, reason=reason, fell=fell, survival_s=survival,
                planned_s=planned, succeeded=ok, succeeded_but_A1=a1, succeeded_look=look,
                succeeded_but_A1_look=a1_look, failed_terms=list(terms), look=None,
                clock=None if outcome == "error" else dict(
                    mean_factor=factor, min_factor=factor, t_ref_reached=m2.RATE * factor * survival,
                    t_ref_planned=m2.RATE * planned), error="boom" if outcome == "error" else None)


def table(change=None, set_name="V6", tiny=False, **default) -> list:
    """Every pair's episode with ``default``'s fields; ``change`` maps a pair, a condition or a seed to others."""
    change = change or {}
    out = []
    for cond, seed in m2.pairs_for(set_name, tiny):
        kw = dict(default)
        for key in (cond, seed, (cond, seed)):
            kw.update(change.get(key, {}))
        out.append(ep(cond, seed, **kw))
    return out


def record(episodes, at_steps=None, set_name="V6") -> dict:
    """A record as ``decide`` reads it (the episodes with their credit, the step count, the fix table)."""
    eps = m2._checked(episodes, [(e["condition"], e["seed"]) for e in episodes])
    doc = dict(schema=m2.SCHEMA if at_steps is not None else m2.BASELINE_SCHEMA, episodes=eps,
               fix_tasks=m2.fix_tasks_for(set_name), **m2.summarize(eps))
    if at_steps is not None:
        doc["at_steps"] = at_steps
    return doc


WEIGHTS = dict(stand=1.0, walk_nominal=2.0, **{"walk_v0.4": 3.0, "walk_v0.3": 0.5}, turn_q90_left=1.5,
               turn_q90_right=1.5, stop=2.0, start_stop=1.0, inplace_left=0.5)
R0L = record(table(terms=("tail_servo",)))  # R0L walks: every episode read, none succeeding, S at the planned length
RULES = m2.rules(2_500_000)


def at(k: int) -> int:
    """The step count of the check of the multiple ``k`` x 2.5M: the first rollout start at or past it."""
    return -(-k * 2_500_000 // 8192) * 8192


def decide(history, baseline=R0L, rules=RULES):
    return m2.decide(history, baseline, dict(task_weights=WEIGHTS), rules)


# ---------------------------------------------------------------------------------------------------------------
# Episodes, seeds, rows
# ---------------------------------------------------------------------------------------------------------------
def test_eighteen_distinct_seeds_one_episode_each():
    assert m2.CONDITIONS == ("walk/nominal", "walk/v0.4", "turn/q90_left", "stop/two_step", "nudge/forward",
                             "nudge/left")
    assert m2.SEEDS["walk/nominal"] == (6001, 6002, 6003) and m2.SEEDS["nudge/left"] == (6016, 6017, 6018)
    pairs = m2.pairs_for("V6")
    assert len(pairs) == 18 and sorted(s for _, s in pairs) == list(range(6001, 6019))
    assert [c for c, _ in pairs] == [c for c in m2.CONDITIONS for _ in range(3)]
    for i, c in enumerate(m2.CONDITIONS):
        assert m2.SEEDS[c] == tuple(6001 + 3 * i + r for r in range(3)) == m2.seeds_for("V6")[c]
    assert m2.summarize(record(table())["episodes"])["distinct_start_seeds"] == 18
    # a tiny run: each condition's first seed; the Stage 1 view: its stock conditions
    assert m2.pairs_for("V6", tiny=True) == [(c, m2.SEEDS[c][0]) for c in m2.CONDITIONS]
    assert m2.pairs_for("S1", tiny=True) == [("walk/nominal", 6001), ("walk/brisk", 6004)]
    assert m2.S1_CONDITIONS == ("walk/nominal", "walk/brisk") == m2.conditions_for("S1")
    assert m2.row_name("R5", 42) == "R5/42@10ms" and m2.row_name("R6", 43, 0) == "R6/43@0ms"
    assert m2.row_name(None, None) == "R0L@10ms" and m2.K == 5


def test_the_fix_table_is_the_task_tables():
    from designlab.stage2 import bar, r5lib

    for table_, tasks in ((m2.FIX_TASKS, r5lib.TASKS), (m2.S1_FIX_TASKS, r5lib.S1_TASKS)):
        want: dict = {}
        for td in tasks:
            for cond in td.monitor:
                want.setdefault(cond, []).append(td.name)
        assert {c: list(t) for c, t in table_.items()} == want
    assert set(m2.FIX_TASKS) == set(m2.CONDITIONS)
    assert m2.STEP_MS == bar.STEP_MS and m2.RATE == bar.RATE


# ---------------------------------------------------------------------------------------------------------------
# Outcomes: not succeeding is no error
# ---------------------------------------------------------------------------------------------------------------
def test_standing_lagging_and_too_few_steps_count_as_not_succeeding(monkeypatch, tmp_path):
    """A policy that stands still (read; the bar's terms fail), a policy whose clock lags (unread, clock_behind)
    and a walk with too few own steps (read; bob, tail and toe-off failed): each is an outcome of the record, not
    succeeding in any reading, never an error, and no episode is asked for twice."""
    stand = dict(outcome="read", terms=("distance", "speed"))
    lag = dict(outcome="unread", reason="clock_behind", factor=0.75, ok=True, look=True)  # (flags a row may leave)
    few = dict(outcome="read", ok=True, a1=True, terms=("bob", "tail", "toe_off"))
    eps = table({"walk/nominal": stand, "walk/v0.4": lag, "turn/q90_left": few, 6010: dict(outcome="fell",
                                                                                          survival=3.0)})
    asked = []

    def episodes(body, set_name, row, pairs, policies=None, traces=None):
        asked.append((row, list(pairs), dict(policies or {}), traces))
        return [dict(e) for e in eps]

    monkeypatch.setattr(m2, "_episodes", episodes)
    monkeypatch.setattr(m2, "_sources", lambda given: dict(given=given, own=given))
    job = _job(tmp_path)
    doc = m2.run(job, 512, BODY, "V6", job / "monitor", sources_sha256="s" * 64)
    assert len(asked) == 1 and asked[0][0] == "R5/42@10ms" and len(asked[0][1]) == 18  # each episode once
    assert asked[0][2] == {"R5/42": dict(job_dir=str(job), steps=512)}
    assert asked[0][3] == job / "monitor" / "traces" / "512"
    assert doc == json.loads((job / "monitor" / "monitor_512.json").read_text())
    assert doc["outcomes"] == dict(read=14, fell=1, unread=3, error=0, not_run=0)
    assert doc["unread_reasons"] == dict(clock_behind=3)
    # the unread episodes do not succeed whatever flags came back; the too-few-steps walks succeed without the look
    assert doc["N"] == 0 and doc["N_reading2"] == 3 and doc["N_reading3"] == 3 and doc["N_but_A1_look"] == 0
    assert doc["failures_by_term"] == dict(bob=3, distance=3, speed=3, tail=3, toe_off=3)
    assert doc["up_by_condition"]["walk/v0.4"] == dict(up=3, of=3)  # unread is up (rule M4): it did not fall
    assert doc["up_by_condition"]["stop/two_step"] == dict(up=2, of=3)
    assert doc["mean_clock_factor"] == pytest.approx((15 * 1.0 + 3 * 0.75) / 18)
    assert doc["distinct_start_seeds"] == 18 and doc["sources"] == dict(given="s" * 64, own="s" * 64)
    assert doc["checkpoint_sha256"] == _hashes(job) and len(set(_hashes(job).values())) == 3
    assert doc["sha256"] == s2io.sha256_json({k: v for k, v in doc.items() if k != "sha256"})
    assert m2.is_record(doc, doc["checkpoint_sha256"], m2.pairs_for("V6"))
    lagging = [e for e in doc["episodes"] if e["outcome"] == "unread"]
    assert all(e["survival_credit"] == pytest.approx(0.75 * PLANNED) for e in lagging)
    assert m2.g8b(doc)["pass"] is True and m2.gain(doc, [], R0L)["gain"] is False


def test_the_monitor_also_runs_the_reported_delays(monkeypatch, tmp_path):
    """Rule M6: the same episodes at further delays, reported; they are no part of N and S."""
    rows = []

    def episodes(body, set_name, row, pairs, policies=None, traces=None):
        rows.append((row, traces))
        return table({"walk/nominal": dict(outcome="fell", survival=4.0)} if row.endswith("@10ms") else None)

    monkeypatch.setattr(m2, "_episodes", episodes)
    monkeypatch.setattr(m2, "_sources", lambda given: dict(given=given, own="x"))
    job = _job(tmp_path)
    doc = m2.run(job, 512, BODY, "V6", job / "monitor", also_k=(0, 5))
    assert [r for r, _ in rows] == ["R5/42@10ms", "R5/42@0ms"] and rows[1][1] is None
    assert set(doc["also"]) == {"0"} and doc["also"]["0"]["row"] == "R5/42@0ms"
    assert doc["up_by_condition"]["walk/nominal"]["up"] == 0 and doc["also"]["0"]["up_by_condition"][
        "walk/nominal"]["up"] == 3
    base = m2.baseline(BODY, "V6", job / "monitor", also_k=(0,), job_dir=job)
    assert base["row"] == "R0L@10ms" and base["also"]["0"]["row"] == "R0L@0ms" and base["valid"] is True


def _job(tmp_path, steps=512) -> Path:
    """A job directory with a ``run.json`` and one checkpoint whose three files are what its record says."""
    job = tmp_path / "jobs" / "R5-x-s42"
    ck = job / "checkpoints"
    ck.mkdir(parents=True)
    files = {}
    for kind, name, fill in (("model", f"model_{steps}.zip", b"m"), ("vecnormalize", f"vecnormalize_{steps}.pkl", b"v"),
                             ("fingerprint", f"fingerprint_{steps}.npz", b"f")):
        (ck / name).write_bytes(fill * 10)
        files[kind] = dict(name=name, bytes=10, sha256=s2io.sha256_file(ck / name))
    (job / "run.json").write_text(json.dumps(dict(job="R5-x-s42", run_uuid="u" * 32, run_id=None, mode="R5", seed=42,
                                                  settings_sha256="h" * 64, flags=dict(kl_cut=False))))
    (ck / f"checkpoint_{steps}.json").write_text(json.dumps(dict(
        run_uuid="u" * 32, settings_sha256="h" * 64, steps=steps, files=files)))
    return job


def _hashes(job, steps=512) -> dict:
    ck = job / "checkpoints"
    return dict(model=s2io.sha256_file(ck / f"model_{steps}.zip"),
                vecnormalize=s2io.sha256_file(ck / f"vecnormalize_{steps}.pkl"),
                fingerprint=s2io.sha256_file(ck / f"fingerprint_{steps}.npz"))


@pytest.fixture(autouse=True)
def _one_thread(monkeypatch):
    """The CLIs ask for one BLAS thread (``s2io.require_one_thread``)."""
    for v in tbenv.THREAD_VARS:
        monkeypatch.setenv(v, "1")


def test_the_monitor_refuses_other_sources_and_a_checkpoint_that_is_not_intact(monkeypatch, tmp_path):
    from designlab.stage2 import train2

    called = []
    monkeypatch.setattr(m2, "_episodes", lambda *a, **k: called.append(a) or table())
    job = _job(tmp_path)
    own = train2.sources()["sha256"]
    with pytest.raises(m2.MonitorError, match="the sources differ"):
        m2.run(job, 512, BODY, "V6", job / "monitor", sources_sha256="0" * 64)
    with pytest.raises(m2.MonitorError, match="the sources differ"):
        m2.baseline(BODY, "V6", job / "monitor", sources_sha256="0" * 64, job_dir=job)
    assert not called and not (job / "monitor").exists()
    doc = m2.run(job, 512, BODY, "V6", job / "monitor", sources_sha256=own)
    assert doc["sources"] == dict(given=own, own=own) and len(called) == 1
    (job / "checkpoints" / "model_512.zip").write_bytes(b"x" * 10)
    with pytest.raises(m2.MonitorError, match="not the file its record names"):
        m2.run(job, 512, BODY, "V6", job / "monitor", sources_sha256=own)
    with pytest.raises(m2.MonitorError, match="no checkpoint record"):
        m2.run(job, 1024, BODY, "V6", job / "monitor", sources_sha256=own)
    # episodes that are not the check's are no record
    monkeypatch.setattr(m2, "_episodes", lambda *a, **k: table()[:-1])
    with pytest.raises(m2.MonitorError, match="17 episodes came back for 18"):
        m2.baseline(BODY, "V6", job / "monitor", job_dir=job)
    monkeypatch.setattr(m2, "_episodes", lambda *a, **k: table({6001: dict(outcome="walked")}))
    with pytest.raises(m2.MonitorError, match="outcome 'walked'"):
        m2.baseline(BODY, "V6", job / "monitor", job_dir=job)
    # the CLI: exit code 1 and no record
    assert m2.main(["baseline", "--job", str(job), "--body", BODY, "--set", "V6"]) == 1
    assert not (job / "monitor" / "r0l.json").exists()


def test_a_baseline_that_holds_an_error_is_no_baseline(monkeypatch, tmp_path):
    """Rule M1: R0L's episodes must each end read or fell; the failed attempt is kept beside."""
    job = _job(tmp_path)
    monkeypatch.setattr(m2, "_sources", lambda given: dict(given=given, own="x"))
    monkeypatch.setattr(m2, "_episodes", lambda *a, **k: table({6007: dict(outcome="error", survival=0.0)}))
    for i in range(2):
        with pytest.raises(m2.MonitorError, match="neither read nor fell.*turn/q90_left seed 6007: error"):
            m2.baseline(BODY, "V6", job / "monitor", job_dir=job)
        kept = json.loads((job / "monitor" / f"r0l.failed.{i}.json").read_text())
        assert kept["valid"] is False and kept["outcomes"]["error"] == 1
    assert not (job / "monitor" / "r0l.json").exists()
    # an unreadable episode of a row without a policy is an error for the baseline too (rule C8 stands there)
    unread = record(table({6001: dict(outcome="unread", reason="unreadable")}))
    assert "walk/nominal seed 6001: unread" in m2.baseline_problem(unread)
    monkeypatch.setattr(m2, "_episodes", lambda *a, **k: table({6001: dict(outcome="fell", survival=5.0)}))
    doc = m2.baseline(BODY, "V6", job / "monitor", job_dir=job)
    assert doc["valid"] is True and m2.baseline_problem(doc) is None and doc["outcomes"]["fell"] == 1
    assert json.loads((job / "monitor" / "r0l.json").read_text())["sha256"] == doc["sha256"]


# ---------------------------------------------------------------------------------------------------------------
# The survival credit
# ---------------------------------------------------------------------------------------------------------------
def test_survival_credit():
    # R0L (a row that does not slow its clock) earns its full time; a clock held at 0.75 earns 0.75 of it
    assert m2.survival_credit(ep("walk/nominal", 6001, survival=12.0)) == pytest.approx(12.0)
    assert m2.survival_credit(ep("walk/nominal", 6001, outcome="fell", survival=4.2)) == pytest.approx(4.2)
    assert m2.survival_credit(ep("walk/nominal", 6001, survival=12.0, factor=0.75)) == pytest.approx(9.0)
    assert m2.survival_credit(ep("walk/nominal", 6001, outcome="unread", survival=8.0, factor=0.75)) == pytest.approx(6.0)
    # a fast clock earns no more than the time
    assert m2.survival_credit(ep("walk/nominal", 6001, survival=12.0, factor=1.25)) == pytest.approx(12.0)
    # an error earns nothing; a condition without a reference neither
    assert m2.survival_credit(ep("walk/nominal", 6001, outcome="error", survival=12.0)) == 0.0
    assert m2.survival_credit(dict(ep("walk/nominal", 6001), outcome="not_run")) == 0.0
    # a track that starts later on the reference clock: the covered time is counted from the track's start
    e = ep("walk/nominal", 6001, survival=10.0)
    e["clock"].update(t_ref_reached=3.0 + m2.RATE * 0.5 * 10.0, t_ref_planned=3.0 + m2.RATE * PLANNED)
    assert m2.survival_credit(e) == pytest.approx(5.0)
    # S is the mean credit; the plain mean survival is printed beside it
    rec = record(table({"walk/nominal": dict(factor=0.5)}))
    assert rec["S"] == pytest.approx((15 * 12.0 + 3 * 6.0) / 18) and rec["mean_survival_s"] == pytest.approx(12.0)
    assert rec["L_planned_s"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------------------------------------------
# The gain
# ---------------------------------------------------------------------------------------------------------------
def test_gain_rule():
    r0l = record(table({"walk/nominal": dict(outcome="fell", survival=6.0), 6004: dict(look=True, ok=True)}))
    assert (r0l["N"], r0l["S"]) == (1, pytest.approx((15 * 12 + 3 * 6) / 18))  # N 1, S 11.0
    same = record(table({"walk/nominal": dict(outcome="fell", survival=6.0), 6004: dict(look=True, ok=True)}), at(1))
    g = m2.gain(same, [], r0l)
    assert g == dict(N=1, S=pytest.approx(11.0), N_star=1, S_star=pytest.approx(11.0), by_N=False, by_S=False,
                     gain=False, against=dict(baseline=True, earlier_checks=0))
    # two more with the look: a gain; one more: none
    def more(n, **kw):
        ch = {"walk/nominal": dict(outcome="fell", survival=6.0)}
        ch.update({s: dict(look=True, ok=True) for s in range(6004, 6004 + n)})
        return record(table(ch, **kw), at(1))

    assert m2.gain(more(2), [], r0l)["gain"] is False and m2.gain(more(3), [], r0l)["by_N"] is True
    # one second longer: a gain at exactly + 1.0 s, none just under it
    longer = record(table({"walk/nominal": dict(outcome="fell", survival=12.0)}), at(1))
    g = m2.gain(longer, [], r0l)
    assert longer["S"] == 12.0 and (g["by_S"], g["by_N"], g["gain"]) == (True, False, True)
    under = record(table({"walk/nominal": dict(outcome="fell", survival=11.9)}), at(1))
    assert m2.gain(under, [], r0l)["gain"] is False
    # against the best earlier check, N* and S* each taken separately; R0L's values stay in that best
    high_n = record(table({s: dict(look=True, ok=True) for s in range(6004, 6009)}
                          | {"walk/nominal": dict(outcome="fell", survival=1.0)}), at(1))  # N 5, S 10.17
    high_s = record(table(), at(2))                                                         # N 0, S 12.0
    now = record(table({s: dict(look=True, ok=True) for s in range(6004, 6010)}), at(3))   # N 6, S 12.0
    g = m2.gain(now, [high_n, high_s], r0l)
    assert (g["N_star"], g["S_star"]) == (5, pytest.approx(12.0)) and g["gain"] is False
    assert m2.gain(now, [high_s], r0l)["by_N"] is True  # (without the check that held N 5: 6 >= 1 + 2)
    weak = record(table({"walk/nominal": dict(outcome="fell", survival=2.0)}), at(1))      # below R0L in both
    g = m2.gain(same, [weak], r0l)
    assert (g["N_star"], g["S_star"]) == (1, pytest.approx(11.0))  # R0L's, not the weaker check's
    # a count that was stored wrong is not read: N and S are recomputed from the episodes
    lied = dict(more(3), N=0, S=0.0)
    assert m2.gain(lied, [], r0l)["N"] == 3


# ---------------------------------------------------------------------------------------------------------------
# G8b, the fix, the stops
# ---------------------------------------------------------------------------------------------------------------
def test_g8b_two_of_three():
    two = record(table({6001: dict(outcome="fell", survival=3.0)}), at(2))
    one = record(table({6001: dict(outcome="fell", survival=3.0), 6002: dict(outcome="error", survival=0.0)}), at(2))
    assert m2.g8b(two) == {"condition": "walk/nominal", "up": 2, "of": 3, "need": 2, "pass": True}
    assert m2.g8b(one) == {"condition": "walk/nominal", "up": 1, "of": 3, "need": 2, "pass": False}
    # a policy that stands still or slows its clock passes (the plan's wording is "stays up")
    standing = record(table({"walk/nominal": dict(outcome="unread", reason="clock_behind", factor=0.5)}), at(2))
    assert m2.g8b(standing)["pass"] is True
    # a tiny run: one episode of one
    assert [m2.need_up(n) for n in (1, 2, 3, 6)] == [1, 2, 2, 4]
    tiny = record(table({6001: dict(outcome="fell", survival=3.0)}, tiny=True), 2048)
    assert m2.g8b(tiny) == {"condition": "walk/nominal", "up": 0, "of": 1, "need": 1, "pass": False}
    # a condition without a reference is not up
    absent = record(table({"walk/nominal": dict(outcome="read")}), at(2))
    for e in absent["episodes"][:3]:
        e["outcome"] = "not_run"
    assert m2.g8b(absent)["up"] == 0 and m2.summarize(absent["episodes"])["L_planned_s"] == pytest.approx(10.0)


def test_the_fix_doubles_the_mapped_tasks_only():
    down = {"walk/nominal": dict(outcome="fell", survival=3.0), 6007: dict(outcome="fell", survival=3.0),
            6008: dict(outcome="fell", survival=3.0), 6013: dict(outcome="fell", survival=3.0),
            6014: dict(outcome="error", survival=0.0), "nudge/left": dict(outcome="fell", survival=2.0)}
    mon = record(table(down), at(2))
    assert m2.conditions_down(mon) == ["walk/nominal", "turn/q90_left", "nudge/forward", "nudge/left"]
    new = m2.fix_weights(mon, WEIGHTS)
    # walk_v0.4 is named by two conditions that are down and is doubled once
    assert new == dict(WEIGHTS, walk_nominal=4.0, **{"walk_v0.4": 6.0}, turn_q90_left=3.0, turn_q90_right=3.0)
    assert WEIGHTS["walk_nominal"] == 2.0  # (a new dict)
    # a task the set does not hold is left out; nothing down changes nothing
    assert "turn_q90_right" not in m2.fix_weights(mon, {k: v for k, v in WEIGHTS.items() if k != "turn_q90_right"})
    assert m2.fix_weights(record(table(), at(2)), WEIGHTS) == WEIGHTS
    # the stop item's tasks
    stop = record(table({"stop/two_step": dict(outcome="fell", survival=4.0)}), at(2))
    assert m2.fix_weights(stop, WEIGHTS) == dict(WEIGHTS, stop=4.0, start_stop=2.0)


def test_decide_g8b_its_fix_and_the_second_failure():
    ok = [record(table(), at(k)) for k in (1, 2, 3, 4)]
    st = decide(ok)
    assert st["gates"]["g8b"]["at_steps"] == at(2) == 5_005_312 and st["gates"]["g8b"]["multiple"] == 5_000_000
    assert st["gates"]["g8b"]["pass"] is True and st["fix"] is None and st["gates"]["g8b_after_fix"] is None
    assert st["task_weights"] == WEIGHTS and st["stopped"] is None and st["checks_done"] == [at(k) for k in (1, 2, 3, 4)]
    # the reading is made at the check of the multiple 5M only: a fall at the first check decides nothing
    bad = record(table({"walk/nominal": dict(outcome="fell", survival=3.0)}), at(1))
    assert decide([bad])["gates"]["g8b"] is None and decide([bad] + ok[1:])["fix"] is None
    # G8b fails at 5M: the one fix; the second reading is the check of the multiple 10M (10,002,432 steps)
    bad2 = record(table({"walk/nominal": dict(outcome="fell", survival=3.0), "stop/two_step": dict(
        outcome="fell", survival=5.0)}), at(2))
    st = decide([ok[0], bad2, ok[2]])
    assert st["fix"]["at_steps"] == at(2) and st["fix"]["tasks"] == ["walk_nominal", "stop", "start_stop"]
    assert st["fix"]["weights_before"] == WEIGHTS and st["fix"]["conditions_down"] == ["walk/nominal",
                                                                                      "stop/two_step"]
    assert st["task_weights"] == st["fix"]["weights"] == dict(WEIGHTS, walk_nominal=4.0, stop=4.0, start_stop=2.0)
    assert st["gates"]["g8b_after_fix"] is None and st["stopped"] is None  # (7.5M: no reading)
    st = decide([ok[0], bad2, ok[2], ok[3]])
    assert st["gates"]["g8b_after_fix"]["at_steps"] == at(4) == 10_002_432 and st["gates"]["g8b_after_fix"]["pass"]
    assert st["stopped"] is None and st["task_weights"]["walk_nominal"] == 4.0  # doubled once, never again
    # a second failure stops the run
    bad4 = record(table({"walk/nominal": dict(outcome="fell", survival=3.0)}), at(4))
    st = decide([ok[0], bad2, ok[2], bad4])
    assert st["stopped"]["status"] == "stopped_g8b" and st["stopped"]["at_steps"] == at(4)
    assert st["stopped"]["multiple"] == 10_000_000 and st["task_weights"]["walk_nominal"] == 4.0
    with pytest.raises(ValueError, match="has no later check"):
        decide([ok[0], bad2, ok[2], bad4, record(table(), at(5))])
    # a tiny run reads at its first check and at the next
    tr = m2.rules(2048, tiny=True, set_name="S1")
    assert (tr["g8b"]["at"], tr["g8b"]["again_at"], tr["stop"]["start"]) == (2048, 4096, None)
    fell = record(table({6001: dict(outcome="fell", survival=3.0)}, set_name="S1", tiny=True), 2048, "S1")
    st = m2.decide([fell], record(table(set_name="S1", tiny=True)), dict(task_weights=dict(stand=1.0, walk=2.0,
                                                                                          brisk=2.0, stop=2.0)), tr)
    assert st["fix"]["tasks"] == ["walk"] and st["task_weights"] == dict(stand=1.0, walk=4.0, brisk=2.0, stop=2.0)


def test_three_checks_without_a_gain_from_20m_stop_at_the_third():
    def run_of(gains):
        """Checks 1..: a gain where asked, by two more episodes with the look than every check before."""
        hist, n = [], 0
        for k, g in enumerate(gains, start=1):
            n = n + 2 if g else n
            hist.append(record(table({s: dict(look=True, ok=True) for s in range(6001, 6001 + n)}), at(k)))
        return hist

    r0l = record(table())
    # no gain at all: nothing counts before the check of the multiple 20M (the 8th); the stop is at 25M
    hist = run_of([False] * 10)
    assert at(8) == 20_004_864 and at(10) == 25_001_984
    assert decide(hist[:9], r0l)["stopped"] is None and decide(hist[:9], r0l)["no_gain_count"] == 2
    st = decide(hist, r0l)
    assert st["stopped"]["status"] == "stopped_no_gain" and st["stopped"]["at_steps"] == at(10)
    assert [c["no_gain_count"] for c in st["checks"]] == [0] * 7 + [1, 2, 3]
    # a gain in between starts the count again: consecutive checks
    st = decide(run_of([False] * 8 + [True, False, False]), r0l)
    assert st["stopped"] is None and [c["no_gain_count"] for c in st["checks"]][7:] == [1, 0, 1, 2]
    st = decide(run_of([False] * 8 + [True, False, False, False]), r0l)
    assert st["stopped"]["at_steps"] == at(12) and st["stopped"]["status"] == "stopped_no_gain"
    # gains before 20M do not postpone it
    st = decide(run_of([True] * 7 + [False] * 3), r0l)
    assert st["stopped"]["at_steps"] == at(10)


def test_decide_gives_the_same_state_from_the_records_alone():
    """Live after each check, or once at a resume: the state after check i is a function of the records up to
    i, and asking again changes nothing (no argument is changed)."""
    r0l = record(table(outcome="fell", survival=4.0))
    hist = [record(table(outcome="fell", survival=4.0 + 0.4 * k), at(k)) for k in range(1, 5)]
    hist[1] = record(table({"walk/nominal": dict(outcome="fell", survival=3.0)}, outcome="read"), at(2))
    keep = json.dumps([hist, r0l, WEIGHTS], sort_keys=True)
    live = [decide(hist[: i + 1], r0l) for i in range(len(hist))]
    again = decide(hist, r0l)
    assert again == live[-1] == decide(json.loads(json.dumps(hist)), json.loads(json.dumps(r0l)))
    assert json.dumps([hist, r0l, WEIGHTS], sort_keys=True) == keep
    for i, st in enumerate(live):  # what was decided at check i is what the whole history says of check i
        assert st["checks"] == again["checks"][: i + 1]
    assert live[0]["fix"] is None and live[1]["fix"] == live[3]["fix"] and live[3]["fix"]["at_steps"] == at(2)
    with pytest.raises(ValueError, match="not in order"):
        decide([hist[1], hist[0]], r0l)


def test_a_check_counted_without_a_record():
    """Rule M1 and the lead's ruling of 16 Nov 2026: after two closed attempts that left no record every episode
    of the check is an error: without a gain, a failed G8b reading, every mapped task doubled."""
    none = m2.no_record(at(2), "V6")
    assert none["no_record"] is True and none["closed_attempts"] == 2 and len(none["episodes"]) == 18
    assert {e["outcome"] for e in none["episodes"]} == {"error"} and (none["N"], none["S"]) == (0, 0.0)
    assert m2.g8b(none) == {"condition": "walk/nominal", "up": 0, "of": 3, "need": 2, "pass": False}
    assert not m2.is_record(none, {}, m2.pairs_for("V6"))
    st = decide([record(table(), at(1)), none, record(table(), at(3))])
    assert st["checks"][1]["no_record"] is True and st["checks"][1]["gain"]["gain"] is False
    assert st["gates"]["g8b"]["pass"] is False
    assert st["fix"]["tasks"] == ["walk_nominal", "walk_v0.4", "turn_q90_left", "turn_q90_right", "stop",
                                  "start_stop"]
    assert m2.no_record(2048, "S1", tiny=True)["outcomes"]["error"] == 2
    # what a record of a check is: it names the checkpoint's hashes and holds an outcome for each of its episodes
    hashes = dict(model="a", vecnormalize="b", fingerprint="c")
    good = dict(record(table(), at(1)), checkpoint_sha256=dict(hashes))
    assert m2.is_record(good, hashes, m2.pairs_for("V6"))
    assert not m2.is_record(good, dict(hashes, model="z"), m2.pairs_for("V6"))        # another checkpoint's
    assert not m2.is_record(dict(good, episodes=good["episodes"][:-1]), hashes, m2.pairs_for("V6"))  # cut
    assert not m2.is_record(dict(good, schema="other"), hashes, m2.pairs_for("V6"))
    assert not m2.is_record(None, hashes, m2.pairs_for("V6"))
    assert not m2.is_record(good, hashes, m2.pairs_for("V6", tiny=True))


# ---------------------------------------------------------------------------------------------------------------
# S2-10
# ---------------------------------------------------------------------------------------------------------------
def test_s2_10_record():
    # R0L walks and stays up: S is at the planned length, so the survival route is closed (L - S < 1.0 s)
    r0l = record(table(terms=("tail_servo",), a1_look=False, a1=True))
    final = record(table({6001: dict(a1_look=True), 6004: dict(a1_look=True)}, terms=("tail_servo",), a1=True), 2_007_040)
    early = [record(table({"walk/nominal": dict(outcome="fell", survival=3.0)}), 500_000 + 3712),
             record(table(), 1_007_616)]
    doc = m2.s2_10(final, early, r0l)
    assert doc["schema"] == "s2-10-1" and doc["at_steps"] == 2_007_040
    assert doc["policy"]["N"] == 0 == doc["r0l"]["N"] and doc["policy"]["S"] == pytest.approx(12.0)
    assert doc["policy"]["mean_survival_s"] == pytest.approx(12.0) == doc["r0l"]["mean_survival_s"]
    assert doc["L_planned_s"] == pytest.approx(12.0) and doc["survival_route_closed"] is True
    assert doc["gain"]["gain"] is False and doc["m3b"] == dict(applies=True, N_but_A1_look=2, N_but_A1_look_r0l=0,
                                                              gain=True) and doc["go"] is True
    assert doc["policy"]["failures_by_term"] == dict(tail_servo=18) == doc["r0l"]["failures_by_term"]
    assert doc["policy"]["N_reading3"] == 18 and doc["policy"]["mean_clock_factor"] == 1.0
    # its earlier checkpoints are reported and decide nothing
    assert [e["at_steps"] for e in doc["earlier"]] == [503_712, 1_007_616]
    assert doc["earlier"][0]["gain_against_r0l"]["gain"] is False and doc["earlier"][0]["outcomes"]["fell"] == 3
    # the two counts and the two means recompute from its 36 episodes
    assert len(doc["episodes"]["policy"]) == len(doc["episodes"]["r0l"]) == 18
    assert m2.summarize(doc["episodes"]["policy"])["N"] == doc["policy"]["N"]
    assert m2.summarize(doc["episodes"]["r0l"])["S"] == doc["r0l"]["S"]
    # one more with A1 lifted is no gain; and the route does not apply while survival can still rise by a second
    one = record(table({6001: dict(a1_look=True)}, a1=True), 2_007_040)
    assert m2.s2_10(one, [], r0l)["go"] is False
    short = record(table({"walk/nominal": dict(outcome="fell", survival=6.0)}, a1=True))
    doc = m2.s2_10(final, [], short)
    assert doc["L_planned_s"] - doc["r0l"]["S"] == pytest.approx(1.0) and doc["survival_route_closed"] is False
    assert doc["m3b"]["gain"] is False and doc["gain"]["by_S"] is True and doc["go"] is True
    # read against R0L alone: an earlier check that was better does not hide the final one's gain
    better = record(table({s: dict(look=True, ok=True) for s in range(6001, 6007)}), 1_007_616)
    final2 = record(table({s: dict(look=True, ok=True) for s in range(6001, 6003)}), 2_007_040)
    assert m2.s2_10(final2, [better], r0l)["gain"]["by_N"] is True
    # R0L's baseline against the committed one, episode for episode
    same = m2.baseline_equal(r0l, json.loads(json.dumps(r0l)))
    assert same["equal"] is True and same["episodes"] == 18 and same["digests_compared"] is False
    moved = json.loads(json.dumps(r0l))
    moved["episodes"][4]["survival_s"] = 11.98
    assert m2.baseline_equal(r0l, moved) == dict(equal=False, episodes=18, differ=["walk/v0.4 seed 6005: survival_s"],
                                                 digests_compared=False)


def test_s2_10_from_a_jobs_records(monkeypatch, tmp_path):
    job = _job(tmp_path)
    monkeypatch.setattr(m2, "_sources", lambda given: dict(given=given, own="x"))
    monkeypatch.setattr(m2, "_episodes", lambda *a, **k: table())
    base = m2.baseline(BODY, "V6", job / "monitor", job_dir=job)
    monkeypatch.setattr(m2, "_episodes", lambda *a, **k: table({s: dict(look=True, ok=True) for s in (6001, 6002)}))
    mon = m2.run(job, 512, BODY, "V6", job / "monitor")
    d = json.loads((job / "run.json").read_text())
    d.update(monitor=[dict(at_steps=512, record="monitor/monitor_512.json")], num_timesteps=512, status="done")
    (job / "run.json").write_text(json.dumps(d))
    (tmp_path / "R0L_BASELINE.json").write_text(json.dumps(base))
    assert m2.main(["s2-10", "--job", str(job), "--committed-baseline", str(tmp_path / "R0L_BASELINE.json")]) == 0
    doc = json.loads((job / "S2_10.json").read_text())
    assert doc["go"] is True and doc["gain"]["by_N"] is True and doc["baseline_equal"]["equal"] is True
    assert doc["checkpoint_sha256"] == mon["checkpoint_sha256"] and doc["final_is_last_checkpoint"] is True


# ---------------------------------------------------------------------------------------------------------------
# The one indirection, and the stand-in on the Stage 1 view
# ---------------------------------------------------------------------------------------------------------------
def test_episodes_goes_through_lookbar_and_configures_the_rows(monkeypatch):
    """``_episodes`` is the module's one indirection: ``s2rows.configure`` with the checkpoint, ``lookbar.rows``
    for the one row, ``lookbar.episodes`` once with every pair."""
    import types

    calls = []
    lookbar = types.ModuleType("designlab.stage2.lookbar")
    lookbar.rows = lambda body, names: calls.append(("rows", body, names)) or {names[0]: "row"}
    lookbar.episodes = lambda body, set_name, rows, pairs, traces=None: calls.append(
        ("episodes", body, set_name, rows, pairs, traces)) or ["e"]
    s2rows = types.ModuleType("designlab.stage2.s2rows")
    s2rows.configure = lambda policies: calls.append(("configure", policies))
    monkeypatch.setitem(sys.modules, "designlab.stage2.lookbar", lookbar)
    monkeypatch.setitem(sys.modules, "designlab.stage2.s2rows", s2rows)
    pairs = m2.pairs_for("V6", tiny=True)
    out = m2._episodes(BODY, "V6", "R5/42@10ms", pairs, {"R5/42": dict(job_dir=Path("/j"), steps=512)}, "tr")
    assert out == ["e"] and calls == [
        ("configure", {"R5/42": dict(job_dir="/j", steps=512)}), ("rows", BODY, ["R5/42@10ms"]),
        ("episodes", BODY, "V6", {"R5/42@10ms": "row"}, pairs, "tr")]
    calls.clear()
    m2._episodes(BODY, "V6", "R0L@10ms", pairs)  # the baseline: no policy to configure
    assert [c[0] for c in calls] == ["rows", "episodes"]


def test_without_lookbar_only_the_stage_1_view_has_a_stand_in(monkeypatch):
    import importlib.util

    if importlib.util.find_spec("designlab.stage2.lookbar") is not None:
        pytest.skip("lookbar is merged: the stand-in is no longer reached")
    with pytest.raises(ModuleNotFoundError):
        m2._episodes(BODY, "V6", "R0L@10ms", m2.pairs_for("V6", tiny=True))
    seen = []
    monkeypatch.setattr(m2, "_stand_in", lambda body, row, pairs, policies: seen.append((body, row)) or [])
    assert m2._episodes(BODY, "S1", "R0L@10ms", m2.pairs_for("S1", tiny=True)) == [] and seen == [(BODY, "R0L@10ms")]
    assert "stand_in" in m2._episodes_by("S1") and m2._episodes_by("V6").startswith("none")


# ---------------------------------------------------------------------------------------------------------------
# clip2
# ---------------------------------------------------------------------------------------------------------------
def _clip(args, tmp_path) -> subprocess.CompletedProcess:
    env = tbenv.child_env("osmesa", base={**os.environ, "LP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
                                          "MESA_SHADER_CACHE_DISABLE": "true"})
    return subprocess.run([sys.executable, "-m", "designlab.stage2.clip2", *args], cwd=STUDY, env=env,
                          capture_output=True, text=True, timeout=600)


def test_a_clip_that_cannot_be_made_is_a_record_with_its_reason(tmp_path):
    out = tmp_path / "clip_512.mp4"
    r = _clip(["--trace", str(tmp_path / "none.npz"), "--body", BODY, "--set", "S1", "--out", str(out)], tmp_path)
    assert r.returncode == 0, r.stderr[-2000:]
    doc = json.loads((tmp_path / "clip_512.json").read_text())
    assert doc["ok"] is False and "no trace" in doc["reason"] and doc["trace"]["sha256"] is None
    assert doc["mujoco_gl"] == "osmesa" and doc["torch_imported"] is False and not out.exists()
    assert doc["sha256"] == s2io.sha256_json({k: v for k, v in doc.items() if k != "sha256"})
    r = _clip(["--trace", str(tmp_path / "none.npz"), "--body", BODY, "--set", "S1", "--out", str(out), "--strict"],
              tmp_path)
    assert r.returncode == 1 and "not made" in r.stdout


def test_clip2_renders_a_real_clip_and_holds_no_torch(tmp_path):
    """A 1 s clip from the trace of a Stage 1 stock episode (the open-loop row R0, cut to 2 s)."""
    run_dir = STUDY / "stage1" / "runs" / "2026-11-08" / BODY
    if not (run_dir / "refs" / "nominal.npz").is_file():
        pytest.skip("Stage 1's arrays are not restored")
    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg on PATH")
    from designlab.stage1 import panels
    from designlab.stage2 import setio

    _, docs = setio.stage1_view(run_dir.parent, BODY)
    ctx = panels.Context(BODY, run_dir, docs)
    cond = dataclasses.replace(next(c for c in panels.conditions() if c.key == "walk/nominal"), seconds=2.0)
    e = panels.tracker_rows(BODY, run_dir, ("R0",))["R0"].run(ctx, cond, 6001, trace_root=tmp_path / "traces")
    trace = tmp_path / "traces" / e["trace"]["path"]
    out = tmp_path / "monitor" / "clip_512.mp4"
    r = _clip(["--trace", str(trace), "--body", BODY, "--set", "S1", "--out", str(out), "--seconds", "1",
               "--strict"], tmp_path)
    doc = json.loads(out.with_suffix(".json").read_text())
    if not doc["ok"] and ("OSMesa" in doc["reason"] or "GL" in doc["reason"]):
        pytest.skip(f"no software renderer here: {doc['reason']}")
    assert r.returncode == 0 and doc["ok"] is True, (doc["reason"], r.stderr[-2000:])
    assert out.is_file() and doc["clip"]["sha256"] == s2io.sha256_file(out) and doc["clip"]["bytes"] == out.stat().st_size
    assert doc["trace"]["sha256"] == e["trace"]["sha256"] == s2io.sha256_file(trace)
    assert doc["clip"]["frames"] == 31 and doc["clip"]["fps"] == 30 and doc["span"]["seconds"] == pytest.approx(1.0)
    assert doc["span"]["t1_s"] == pytest.approx(2.0, abs=0.03) and doc["labels"]["top"][0].startswith("SIMULATION")
    assert doc["mujoco_gl"] == "osmesa" and doc["torch_imported"] is False
    assert sorted(doc["modules"]) == ["designlab/stage2/barviews.py", "designlab/stage2/clip2.py",
                                     "designlab/stage2/clips.py"]
    assert not list(tmp_path.rglob("*.part.mp4"))
