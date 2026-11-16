<!-- Working paper: the independent reviewer's report on section 13 of STAGE2_75_REPORT.md as committed at cc24a63, returned on 16 Nov 2026. Nothing here is a rule. Its fixes of wording are in the report's next version. -->

## Verdict

**Yes after listed fixes.** No number in section 13 is wrong: every table cell, count, range, median and ratio matches my independent recompute of the 91 records. The fixes are all wording. The headline sentences say more than the pre-registered words allow, and the status note claims this check before it happened.

Paths are relative to `/home/user/mesozoic-labs/hardware/trex_form_study/`. Add-on record paths are under `stage2/runs/2026-11-16-75cm-addon/A/V0/75-M-H-10/seeds3042/eval/`.

## Findings, most serious first

**1. Blocker. The two headline sentences use a phrase rule A-6 withheld, and give a verdict.**
- Section 13: "**In plain words: 10 ms is not a lone bad point, and no longer delay gives a usable walk.**"
- Section 1 pointer: "The 0.40 m/s walk does not come back at any longer delay, so 10 ms is not a lone bad point for the stand-in controller."

Evidence:
- **A-6, 0.40 m/s walk:** the pattern is "Neither", for which "the report prints the counts and uses neither phrase". "10 ms is not special" is attached only to "Delay simply gets worse".
- **A-6, slow walk:** its pattern is "A window", and "the two paragraphs are not merged".
- **A-5:** no verdict; "usable" is an undefined verdict word.
- **A-9 and B3:** "any longer delay" covers delays above 20 ms and between the 2 ms steps, which were not run.
- **A-8:** the pointer carries none of the labels.
- **Records the sentences hide:** fit 43 at 18 ms stays up in 18 and 19 of 20 (`R2/43@18ms/walk/v0.4.json`, `v0.4_mu0.54.json`). The slow walk under fit 43 at 18 ms has 18 up and all 18 cover the distance (`R2/43@18ms/walk/nominal.json`).

Replacement, section 13:
> **In plain words.** At the longer delays that were run (12 to 20 ms in 2 ms steps), no run under the stand-in controller met every limit. At 0.40 m/s no run covered the distance; at 18 ms the robot stayed up in 18 and 19 of 20 runs under one of the two fits and in 6 and 4 under the other. The slow walk stayed up again at 20 ms (19 and 20 of 20) without covering its distance. These are counts, not a verdict.

Replacement, section 1 pointer:
> **Added 16 Nov 2026: longer delays** (simulation; stand-in controller fitted with no delay; delays that are not the judged 10 ms; no verdict; 14, 16 and 20 ms are the delays the owner approved (P37), 12 and 18 ms are the lead's addition). Today's walk was then run at 12 to 20 ms in 2 ms steps (section 13). At 0.40 m/s no run covers the distance at any of these delays. Of 20 runs, 0 to 3 stay up at 12, 14, 16 and 20 ms; at 18 ms 6 and 4 stay up under one fit and 18 and 19 under the other. At 12 to 16 ms most falls come in the start from standing, before any walking at 0.40 m/s. The slow walk stays up again at 20 ms (19 and 20 of 20), short of its distance and outside its limits. Nothing above changes.

**2. Major. The status note claims a finished check and misattributes delays to the owner.**
Sentence: "**Added 16 Nov 2026, 02:15 UTC.** … Section 13 is new: today's walk at 12 to 20 ms of delay, which the owner asked for in that answer. … its text was checked once by an independent reviewer against the records."

Evidence:
- Commit `cc24a63` is stamped 02:02:13 UTC and its message says "awaiting its independent check". The 02:15 stamp is later than the commit.
- P37 names 14, 16 and 20 ms only. A-8 requires 12 and 18 ms and the slow walk at 8 ms to carry "the lead's addition, not among the delays the owner approved".

Replacement:
> **Added 16 Nov 2026, [time of the corrected version] UTC.** … Section 13 is new: today's walk at 12 to 20 ms of delay. The owner asked for 14, 16 and 20 ms in that answer; 12 and 18 ms and the slow walk at 8 ms are the lead's addition. Its rules were written and independently reviewed before its runs. Its text was checked once by an independent reviewer against the records: no number was wrong, and the changes of wording it asked for are in this version. …

**3. Major. "What this means for the plan" overclaims in all four bullets, even as labelled interpretation.**

(a) "Keeping the judged delay at 10 ms loses nothing. The stand-in controller fails from 8 ms up at 0.40 m/s, so there is no longer delay at which the hardware could be expected to be fine with it."
- 18 ms reads mixed, not down. Nothing above 20 ms was run. It speaks of hardware from simulation.
> These runs give no reason to judge at another delay. At 0.40 m/s the stand-in controller reads down at 8 to 16 ms and at 20 ms, and mixed at 18 ms; at none of them does a run cover the distance. Delays above 20 ms and between the 2 ms steps were not run.

(b) "Survival is not a steady slide either: runs stay up again near 18 to 20 ms."
- "near" reads between delays, and the two walks are merged.
> Survival does not simply fall with delay either. At 18 ms runs stay up again under fit 43 (18 and 19 of 20 at 0.40 m/s, 18 of 20 on the slow walk). At 20 ms the slow walk stays up under both fits (19 and 20 of 20) and the 0.40 m/s walk does not (1 of 20).

(c) "What fells the robot at these delays is the stand-in controller's feedback acting late, not the planned motion …"
- This states a mechanism. A-6 says "No mechanism is tested by these runs", and treats `R0` as a check on the runner.
- "two earlier attempts" hides that one was on the 60 cm bodies (F27) and the other was a learned add-on with three runs (F26). The F26 row does not itself say that add-on was trained with delay; `designlab/stage1/env.py:235` does (0 to 20 ms).
> The falls come with the feedback on: the same motion played with no feedback stays up in 18 to 20 of 20 runs at every delay that was run. The lead reads this as the feedback acting late; these runs do not test that. A controller trained or fitted with delay is the remedy the lead takes from it. It is not shown to work: a learned add-on trained with delay on this body (three runs) and a refit with delay on the 60 cm bodies both still fell at 10 ms (`DECISION_LOG.md` F26, F27).

(d) "With this controller, today's walk at 0.40 m/s stands at 6 ms and below and not at 8 ms and above."
- "stands" is undefined, and an owner would read it as "fine at 6 ms". At 6 ms all 20 stay up, but 20, 20, 19 and 20 fail toe clearance and 0, 0, 1 and 0 meet every limit but for A1 (registered `R2/42@6ms`, `R2/43@6ms`, both floors).
- Fit 43 at 18 ms stays up in 18 and 19.
> With this controller, today's walk at 0.40 m/s stays up in 20 of 20 runs at 0, 2, 4 and 6 ms and in none at 8 ms. Staying up is not passing: at 6 ms it fails toe clearance in 19 or 20 of 20 runs.

**4. Major. The bullet "The 0.40 m/s walk does not come back." omits the plan's own caveat.**
A-6 says "a 0.40 m/s record at a longer delay may say nothing about walking at 0.40 m/s". Of 80 runs per delay (two fits, two floors), by `segment_reached`:

| Delay | Reached the 0.40 m/s walk itself | Fell in it |
|---|---|---|
| 12 ms | 0 | 0 |
| 14 ms | 4 | 0 |
| 16 ms | 2 | 0 |
| 18 ms | 47 | 0 |
| 20 ms | 71 | 67 |

The grippy 0.40 m/s records and the slow-walk records are also one simulation up to the end of the first slow stride. At 12, 14 and 16 ms, 110 of the 113 runs that fall in both have the identical fall time, seed by seed. The two tables largely count the same falls.

Replace the heading with "**At 0.40 m/s no run covers the distance at any longer delay that was run.**" and add:
> At 12, 14 and 16 ms these records say almost nothing about walking at 0.40 m/s itself: of 80 runs, 0, 4 and 2 reach it. At 18 ms 47 reach it and none of them falls; at 20 ms 71 reach it and 67 fall in it. On the grippy floor the slow walk's runs and the 0.40 m/s walk's runs are the same simulation until the first slow stride ends, so at 12 to 16 ms the two tables largely count the same falls.

**5. Major (readability). The answer block cannot be read without the plan.**
"Neither phrase for the 0.40 m/s walk" never names the two phrases, and "A window" is not explained. The only readable answer is then the overclaiming plain-words line. Add before the bullets:
> Three answers were fixed in advance for each walk. "Delay simply gets worse": every delay run above 10 ms reads down. "A window": some delay above 10 ms reads up, so the falls around 10 ms sit between delays that stay up. "Neither": no delay above 10 ms reads up and at least one is mixed.

Also change "No other centre is fitted" to "No other point of symmetry is tried".

**6. Minor. The slow-walk bullet at 18 and 20 ms hides a difference between the fits.**
"That is staying up, not walking as asked" reads as covering 18 ms too. Under fit 43 at 18 ms all 18 runs that stay up cover the distance (median 0.245 m/s; the record's word is "up and covering the distance"). Under fit 42, none of 10 do, and the median toe clearance is -2.0 mm. Add:
> At 18 ms the 18 runs that stay up under fit 43 cover the distance and fail cadence, ground force, stride length and toe clearance; the 10 under fit 42 do not cover it.

Expectation 2 should also carry the second word: "reads up, short of the distance: met."

**7. Minor. "commanded distance" (three places) mixes the command with the target.**
- The limit is 10 s at the target speed: `measures.bar.along_target_m` = 2.4306 m, and `measures.stage1.speed_target_m_s` = 0.24306. So "0.243 m/s" is right.
- The command is 5 % faster (`controller.rate` 1.05, about 0.255 m/s).
- Write "the distance the limit asks (10 s at the target speed)".
- "The slow walk (0.24 m/s)" next to "0.240 to 0.241 m/s … asks 0.243" looks like a contradiction; write 0.243 there.
- State the size of the miss. At 20 ms the medians are 2.40 and 2.41 m of 2.43 m, and the nearest run is 2.418 m. Fit 43 at 18 ms on the 0.40 m/s walk covers 3.90 and 3.87 m of 4.0 m. A plain reader will otherwise think the robot barely moves.

**8. Minor. "With the balance feedback off, delay makes no difference."**
- Say instead "the count that stays up does not move with delay (18 to 20 of 20 at each delay run)".
- Add that on the 0.40 m/s walk this row meets every limit in no run at any delay. The duty limit and the tail servo's planning limit fail in every run that stays up (`R0@*ms/walk/v0.4*.json`).

**9. Minor. Terms a plain reader will trip on.**
- "add-on A1" in the Records column collides with the limit "A1" in the same cells.
- "(Stage 1 speed 20)" is never explained.
- "holds one step further" reads as a walking step; write "2 ms longer".
- "Records at 14, 16 and 20 ms: (P37: 14, 16 and 20 ms)." is not a sentence.

**10. Minor. Left out, worth one line each.**
- Peak ground force of the fit-43 runs that stay up at 18 ms on the 0.40 m/s walk: median 3.0 body weights, against about 1.5 with no delay (`measures.window.grf_peak_bw`).
- The runner's word under A-5 is "does not pass at <ms> ms" for every pair of fits at 12 to 20 ms, in both readings. It is not in section 13, and `DELAYS_TABLE.md` gives it for the 0.40 m/s walk only.

**11. Minor. `delays_table.py` against the rule text; none of these changes today's output.**
- An unread `R0` record, or an unread record of the other walk, would not remove the delay's word. A-6 says such a delay takes no word "in either walk's reading".
- If some delay above 10 ms read up while 8 and 10 ms did not both read down, the script prints "no pattern is named". The rule is silent on that case.
- A delay with no word counts as "differs" in the mirror result. The rule is silent.
- The table of where the falls come lists the two fits only, not the 13 `R0` falls (all in the 0.40 m/s walk itself).
- The words, the flag rule, "not read", the second word from `success.along` for both walks, and the 8 and 10 ms condition for the 0.40 m/s walk only, are as written.

## Checked and found right

- **Git.** The plan, `run_addon.sh` and `delays_table.py` are unchanged since `f4a8b7d` (01:48:21 UTC). The first unit started at 01:48:39 and the records were committed at `f35ed65`. No file under `designlab`, `tools`, `tests`, `stage1` or `stage2/sets` changed since `f4a8b7d`. Sections 1 to 12 are identical to the previous commit apart from the three added blocks.
- **Run.** Six units in the registered order, one session each, none cut or re-run. Spent-before equals the running sum. The estimates (0.12811 per delay, 0.02459 for A3) reproduce from the `COST.md` rates. The total is 0.4952 core-hours, equal to the sum of episode wall times.
- **Records.** 47 records, 940 runs; each has 20 episodes, seeds 3042 to 3061, no errors. `latency_steps` = ms / 2 in all 91 records. `key.sources` equal the two A-4 hashes, and `key.context`, row sources, inputs, gains and plans match the registered records.
- **Numbers.** Zero differences between my recompute, `delays_table.json` and the report, including both patterns, the mirror result (14, 16, 18 and 20 ms differ) and expectations 1 to 4.
- **Stage 1.** The 20 ms statement agrees with F26. In `R2/43@20ms/walk/nominal.json`, Stage 1's own reading fails speed, stride, ground force and drive speed in 20 of 20 and cadence in 18.
- **Regeneration.** `assemble.py` output is byte-identical to the committed report. `delays_table.py` rewrites both files identically and `git status` stays clean.

Two things in the history, neither affecting a record:
- The chain log has one failed start at 01:17:04 UTC, before any unit, with a message found in neither committed version of the chain script.
- `c16cc2c` (`BUY_GATE_DRAFT.md` only) was committed while the chain ran.

## Not checked

- No simulation, and not the no-physics test `tests/test_stage2_addon_delay.py`.
- The runner's dry run itself.
- The trace `.npz` files.
- Stage 1's own 20 ms records (only F26 and the plan's quotation of them).
- Whether every fix in `stage2/design/addon/review_A.md` was applied.
- The content of sections 1 to 12 and of `BUY_GATE_DRAFT.md`.
- Commit times are taken as git gives them.

My scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/review_sec13/` (`recompute.py`, `details.py`, `compare.py`, `more.py`).
