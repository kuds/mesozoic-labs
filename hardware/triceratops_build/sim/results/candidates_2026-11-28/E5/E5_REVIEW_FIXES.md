# E5 retuned trot: response to the independent review (findings E5-R1 to E5-R4)

28 Nov 2026. SIMULATION. Nothing measured; nothing adopted. Every trot setting, and the body-pitch ruling, stay
with the owner and the lead.

`E5_SUMMARY.md` and the other E5 files stay as written, because the build's rules forbid editing an existing file.
Where this file and `E5_SUMMARY.md` word something differently, use this file's wording. It changes no E5 figure.

Supporting files:
- `e5_review_checks.json`: the numbers each finding rests on, recomputed from E5's stored files.
- `e5_tuning_seeds_walltime.json`: the provenance addendum for E5-R4.
- `pitch0/`: the pitch-0 branch for E5-R1, with its plan, results and summary.

## Corrected headline

**E5 (SIMULATION), as pre-registered:** with `PitchedTrot`'s body-pitch setpoint, the retune SUCCEEDS at 60 cm
and at 50 cm. **The lead's item-9 ruling is still open:** does the setpoint count as new controller structure?

**Changed after a result:** a pitch-0 branch was planned after E5's result and this review, then run. It also
SUCCEEDS at both sizes, on all four held-out sets: the lead's 301-310 and 401-410, and the rerun seeds 501-510 and
601-610. So E5's outcome is the same under either ruling. At 60 cm, though, the pitch-0 setting's margin is
narrower (see E5-R1).

## E5-R1 (medium): the headline ignored prereg item 9's open gate. Accepted

**Checked:**
- Prereg section 3 item 9 requires the ruling "before the first E5 run".
- No ruling is on record:
  - No line of `DECISION_LOG.md` (D1-D56) mentions `PitchedTrot`, body pitch or item 9.
  - The only `design/*.md` file that mentions `PitchedTrot` or body pitch is the pre-registration itself.
- The chosen settings use +1.08 deg (60 cm) and +0.82 deg (50 cm), nose down.
- Their held-out 10-20 s pitch had condition means of +0.86 to +1.12 deg (60 cm) and +0.65 to +0.85 deg (50 cm).
- On the frozen trot at 60 cm, the 127 held-out slow-gait episodes ran at a mean of -1.32 deg (range -1.40 to
  -1.08; nose up). The other 73 episodes averaged -0.08 deg. So the setpoint acts on the mechanism the rule targets.
- The reviewer's stage-1 points are confirmed (`e5_review_checks.json` R1):
  - s1-04 at +0.08 deg: 0 slow-gait, knee 0.748
  - s1-07 at -0.49 deg: knee 0.751
  - s1-02 at -1.14 deg: knee 0.738, and it met the rule on 101-110 in stage 2

**Fix 1: the headline is conditional** (above).

**Fix 2: the pitch-0 branch. Changed after a result.**

The review asked for this branch only if the lead rules that pitch is structure. It was run now as a contingency,
so that a result exists for either ruling. It does not make the ruling.

How it was run:
- **The plan:** `pitch0/E5_PITCH0_PLAN.md` was written and its sha256 recorded (`1eb4903d...`, 09:08:45Z) before
  any pitch-0 episode ran.
- **The search:** plan.json's 33 stage-1 points, with pitch set to 0, then E5's stage-2 rule and choice.
- **Judged once on four held-out sets.** The lead's 301-310 and 401-410 had never been run with this setting.
  501-510 and 601-610 are the review's proposal (prereg section 6's rerun seeds), labelled a rerun.
- **The verdict rule:** it SUCCEEDS only if all four sets pass. That is stricter than either pair alone.
- **The code:** `sim/candidate_checks.py` was unchanged (`008aa308...`). The runner is generated from `run_e5.py`
  by literal replacements.

| Size | Chosen (seeds 101-110 only) | 301-310 | 401-410 | 501-510 (rerun) | 601-610 (rerun) | Pitch-0 retune |
|---|---|---|---|---|---|---|
| 60 cm | H0.60-p0-s1-24: period 0.845 s, duty 0.628, pitch 0, lift 25.0 mm, reach 87.0 mm (E5's chosen point, pitch 0) | pass: J3 0.2556 m/s, knee 0.829 | pass: 0.2564, 0.759 | pass: 0.2509, 0.961 | pass: 0.2597, 0.680 | **SUCCEEDS** |
| 50 cm | H0.50-p0-s1-15: period 0.638 s, duty 0.532, pitch 0, lift 23.1 mm, reach 81.9 mm (E5's chosen point, pitch 0) | pass: 0.2491, 0.680 | pass: 0.2490, 0.709 | pass: 0.2479, 0.681 | pass: 0.2476, 0.707 | **SUCCEEDS** |

How to read the table:
- Every set had 0 falls in J1-J4 and 0 slow-gait episodes in every condition.
- J3 speed is the 2-20 s mean (rule: at least 0.225 m/s).
- Knee is the worst hind-knee RMS over the 6.75 N m continuous rating (rule: at most 1.0).
- Tuning used 39 setting runs per size; stage 3 was not needed at either size.
- Source: `pitch0/e5p0_heldout_<size>.json`.

**The narrower 60 cm margin** (reported, not judged; `pitch0/E5_PITCH0_SUMMARY.md`):
- On seeds 501-510 in J3, 2 of 10 pitch-0 episodes ran "slow": seed 505 at 0.2136 m/s with the hind knee at
  0.906, and seed 507 at 0.2084 m/s with the knee at 0.961.
- The slow-gait rule also needs the knee at 1.0 or more, so both count 0. Had both crossed 1.0, J3 would have
  counted 2 of 10 and that set would have failed.
- The pitched choice had no "slow" episode on 301-410.
- Pitch-0 worst hind knee on the lead's sets: 0.829 / 0.759, against 0.818 / 0.719 pitched.
- Pitch-0 X4 drive peak: 80.5 rpm, against 75.1 rpm pitched (cap 83 rpm).
- At 50 cm the two settings are close: knee 0.680 / 0.709 at pitch 0, against 0.670 / 0.678 pitched, with no
  "slow" episode either way.

**The frozen trot on the rerun seeds** (supporting, not judged):
- **60 cm:** it would fail both sets. J3 ran 0.2205 / 0.2121 m/s, slow gait reached 7-10 of 10 in J3 and after
  every push, and the hind knee reached 1.141 / 1.124.
- **50 cm:** it would pass both sets, with the hind knee at 0.911 / 0.913 and no slow-gait episode.

**Harness checks** (`pitch0/e5p0_harness_checks.json`), all passing:
- Setting 00 (the frozen controller) reproduces E5's stored stage-1 episodes: 0 differing fields over 20 per size.
- Stage 2 reran the 120 episodes it shares with stage 1: 0 mismatches.
- All 1,860 episode files per size are at pitch 0, with allowed seeds.
- The choice files and the plan are unchanged since written, and each choice was written before any held-out file.
- A bug in my own check script first reported the choice hashes as changed: it compared "hash and name" with the
  hash. I fixed it before writing the results file. No result changed.

**What each ruling means** (fixed in the plan before the branch ran):
- **If the lead rules that body pitch is not new structure:** E5's pitched result stands as judged, and this
  branch is supporting.
- **If the lead rules that it is new structure:** this branch's verdict is E5's verdict. It SUCCEEDS at both sizes,
  labelled "changed after a result" and judged in part on rerun seeds. The pitched result is kept beside it.

## E5-R2 (low): a transient slow phase on a held-out J3 seed was not mentioned. Accepted

**Checked:** episode H0.60-s1-24__J3__302 (seeds 301-310).
- Over 2-8 s it ran 0.21297 m/s, with the hind knee at 0.9655 of rating.
- Over 10-20 s it recovered to 0.25159 m/s and a knee of 0.6714. Its 2-20 s figures were 0.234 m/s and 0.818.
- It is the only episode, of 200 held-out at each size and of the chosen settings' stage-1 and stage-2 episodes,
  with a 2-8 s speed under 0.225 m/s or a 2-8 s knee of 0.9 or more.
- For comparison, the frozen trot had 5 such held-out episodes at 60 cm and 3 at 50 cm.

**Context line, to sit beside the verdict:**
- In 1 of 20 held-out J3 episodes at 60 cm (seed 302), the robot ran 0.213 m/s with the hind knee at 0.97 of rating
  over 2-8 s, then recovered.
- The rule reads only 10-20 s for the slow gait, so this episode is not judged, and the verdict is unchanged.
- "Avoids the slow gait in J1-J4" means no slow-gait episode in the rule's 10-20 s window.
- The same seed does the same at pitch 0: 0.2075 m/s, knee 0.991, over 2-8 s.

## E5-R3 (low): the 50 cm reading overstated what E5 shows. Accepted

**Checked** (`e5_review_checks.json` R3):
- On the held-out seeds, the frozen 50 cm trot had 1 slow-gait episode: J3, seed 304, on 301-310. That episode's
  hind knee of 1.0186 is its only rule failure there. On 401-410 it had none and would pass the whole rule.
- In stage 1 it had 0 slow-gait episodes in 20.
- The study's 50 cm counts came from its S2 plant: friction 0.8, a 0.228 m/s command and the as-built layout. That
  plant gave 3 of 10 seeds with rescaled gains and 2 of 10 with the adopted gains.
- The study's own L1 run had "no slow seed" (`FORM_FACTOR_STUDY_50_80CM.md` lines 423 and 444).

**Reworded reading for candidate B:**
- The pre-registered outcome label "SUCCEEDS at 50 cm" stands.
- The study's plan-is-wrong condition ("a retune does not remove the trot's slow gait at 50 cm") did not occur in
  these conditions: 0.25 m/s, adopted gains, L1 and the J1-J4 plants.
- The frozen trot showed at most 1 slow-gait episode per condition here.
- The retune's effect at 50 cm is on hind-knee load. The per-condition worst knee was:

  | Setting | Seeds | Worst knee per condition |
  |---|---|---|
  | Retune, pitched | 301-410 | 0.59-0.68 |
  | Retune, pitch 0 | all four sets | 0.60-0.71 |
  | Frozen trot | 301-410 | 0.76-1.02 |
  | Frozen trot | 501-610 | 0.77-0.91 |

- E5 does not test removing a common slow gait at 50 cm.

## E5-R4 (low): the tuning files lack seeds and wall time. Accepted

**Checked:**
- `e5_tuning_<size>.json` has neither field. Its keys are experiment, size, evidence, date, prereg, model, rows,
  choice_file, stage2_determinism_check, budget and provenance. Its rows hold stage, id, the five values, score,
  episodes and finished_utc.

**Where seeds and wall time are recorded:**
- **Seeds:**
  - the CSV column `seeds`
  - `e5_choice_<size>.json` `seeds_used`
  - `e5_run_record.json` `seeds`
- **Wall time:**
  - `e5_run_record.json` `stages[].pools[].wall_s`, the pool wall time per stage on 2 workers
  - `E5_SUMMARY.md`
  - `e5_heldout_<size>.json` `episode_wall_s_total`

**Fix:** a new file, `e5_tuning_seeds_walltime.json`.
- For every tuning row it gives the seeds, the conditions, the episode count, the summed per-episode wall time and
  the finish time.
- It also gives each stage's pool wall time.
- The pitch-0 branch's own tuning files carry these fields in every row.

## Rejected

None. One part of the R1 fix was not done here: putting the conditional verdict into `CANDIDATES_2026_11_28.md`.
That file does not exist yet. It is the synthesis across E1-E5, and under the no-edit rule whoever writes it gets
one pass, so it is left to its author.

Suggested wording for it:

> E5 (SIMULATION): the retune SUCCEEDS at 60 cm and 50 cm with `PitchedTrot`'s body-pitch setpoint. The lead's
> item-9 ruling is still open. A pitch-0 branch, planned and run after the result and labelled so, also SUCCEEDS at
> both sizes on 301-310, 401-410 and the rerun seeds 501-510 and 601-610. At 60 cm its margin is narrower: two
> slow J3 episodes, with hind knees at 0.91 and 0.96 of rating. Context, not judged: one of 20 held-out J3
> episodes at 60 cm ran 0.213 m/s with the hind knee at 0.97 of rating over 2-8 s, then recovered. At 50 cm the
> frozen trot showed at most 1 slow-gait episode per condition, so the retune's effect there is on hind-knee load,
> not on removing a common slow gait.

## What this means for candidates A and B (SIMULATION; nothing adopted)

**Candidate A, 60 cm:**
- The scripted fallback's retune meets E5's rule with or without the pitch setpoint. The margin is wider with the
  setpoint.
- One J3 seed runs slow and near the knee rating for its first seconds with either setting.
- At pitch 0 the fore hip pitch drive peaks at 80.5 rpm.
- Desk arithmetic, context only: at a 1.25 hip-pitch reduction that is 100.6 rpm at the drive, over E2's 91.3 rpm
  peak line. So the trot settings and a hip-pitch reduction still pull against each other.

**Candidate B, 50 cm:**
- The plan-is-wrong condition did not occur under E5's conditions. What E5 shows at 50 cm is a lower hind-knee
  load.

**Limits of both:**
- One controller: 0.25 m/s, straight, flat floor, ideal servo.
- No E1 stops in place, and no hardware.
