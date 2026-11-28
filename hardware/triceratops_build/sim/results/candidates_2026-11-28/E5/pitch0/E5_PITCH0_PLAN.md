# E5 pitch-0 branch: plan (written before any pitch-0 episode ran)

28 Nov 2026. SIMULATION. Nothing measured; nothing adopted.

**Changed after a result.** This plan was written after E5's pitched held-out results (`../E5_SUMMARY.md`) and
after the independent review's finding E5-R1. The whole branch carries that label. It replaces nothing: E5's
pitched result and this one are both kept.

## Why

Pre-registration (`design/PREREG_CANDIDATES_2026_11_28.md`) section 3 item 9: "if this is read as new controller
structure, body pitch is held at 0 and the same stage-1 points are run with that coordinate set to 0; that decision
must be made before the first E5 run". No ruling is on record (`DECISION_LOG.md` D1-D56, `design/*.md`), and E5 ran
with `PitchedTrot`. The lead's ruling is still open. This branch runs the alternative that item 9 names now, so a
result exists whichever way the lead rules. It does not make the ruling.

## What runs

Everything not listed here is as in prereg section 4 E5 and the E5 runner's readings R1-R6 (`run_e5.py`): the
models, plants, the 10 conditions, the 0.25 m/s command, the 10 ms delay, 20 s episodes, the slow-gait rule, the
screen score, `e5_success` and `e5_choose`.

1. **Stage 1.** The 33 settings in `plan.json` `E5.sizes.<size>.stage1`, checked against `plan.json`, with
   `body_pitch_deg` set to 0.0. Ids are `H<size>-p0-s1-NN`, where NN is the source point's index. Setting 00 is the
   frozen controller (its pitch was already 0), so its 20 episodes per size also serve as a determinism check
   against E5's stored `H<size>-s1-00` episodes. The runner refuses any nonzero pitch.
2. **Stage 2.** The 6 best by `e5_screen_score` (ties broken by index), on all 10 conditions, seeds 101-110.
3. **Stage 3,** only if no stage-2 setting meets `e5_success` on seeds 101-110. It takes `e5_neighbours` of
   `e5_choose`'s first stage-2 setting and drops the two body-pitch neighbours, so pitch stays 0. That leaves at
   most 8 settings, with ids `H<size>-p0-s3-NN`.
4. **Choice.** `e5_choose` runs over every setting run on seeds 101-110. Its result is written to
   `e5p0_choice_<size>.json`, with the sha256 kept in scratch, before any held-out seed of that size runs.
5. **Budget.** At most 47 setting runs per size (33 + 6 + 8), under the lead's 60. They are counted for this branch
   alone. Under a "structure" ruling the pitched search is void and this branch is E5's search. Under the other
   ruling this branch is supporting. `plan.json` fixes the stage-1 points, so nothing from the pitched search
   steers this one.
6. **Held-out, judged once.** The chosen setting runs on seeds 301-310, 401-410, 501-510 and 601-610, all 10
   conditions (400 episodes per size).
   - 301-310 and 401-410 are the lead's sets. They were run before with other settings (the pitched choice and the
     frozen controller), but not with this one. The exception is if the choice is setting 00: that setting's
     301-410 results are already in `../e5_heldout_<size>.json`, and the summary will say so.
   - 501-510 and 601-610 are prereg section 6's E5 rerun seeds. The review proposed them because 301-410 have been
     seen. They are labelled a rerun.
7. **Verdict, per size.** The pitch-0 retune SUCCEEDS only if `e5_success` passes on all four sets. That is stricter
   than either pair alone. Otherwise it FAILS. Each set's result is reported, with each pair's reading beside it.
8. **Supporting, not judged.** The frozen controller runs on 501-510 and 601-610. Its 301-410 results are already in
   `../e5_heldout_<size>.json`.
9. **Reported, not judged.** As in E5: drive speeds, slip, the 10-20 s pitch and "slow". After review E5-R2, also
   any held-out episode whose 2-8 s speed is under 0.225 m/s, or whose 2-8 s hind knee is at or over 0.9 of rating.
   This is context only.
10. **Run order and code.** 0.60 m first, then 0.50 m, on 2 workers. `sim/candidate_checks.py` stays unchanged
    (sha256 `008aa308debdf1b6...`). The runner is the session scratch `candidates/E5_pitch0/code/run_e5_pitch0.py`,
    generated from `run_e5.py` by literal text replacements only (listed in `make_runner.py` beside it). Every
    result file records its sha256 and this plan's.
11. **Outputs,** in this folder:
    - `e5p0_tuning_<size>.csv` and `e5p0_tuning_<size>.json` (each row with its seeds and episode wall time)
    - `e5p0_choice_<size>.json`
    - `e5p0_heldout_<size>.json`
    - `e5p0_run_record.json`
    - `e5p0_harness_checks.json`
    - `E5_PITCH0_SUMMARY.md`

## What each outcome means (fixed now)

- **If the lead rules body pitch is not new controller structure,** E5's pitched result stands as judged, and the
  condition on its headline is lifted. This branch is then supporting.
- **If the lead rules it is new controller structure,** this branch's verdict is E5's verdict at each size. It is
  labelled "changed after a result" and judged in part on rerun seeds 501-610. The pitched result is kept beside
  it as supporting.
- **If the pitch-0 retune SUCCEEDS at a size,** E5's outcome at that size is the same under either ruling, apart
  from this branch's label. **If it FAILS at a size,** E5's outcome at that size depends on the lead's ruling.
