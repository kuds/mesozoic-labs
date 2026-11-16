# Independent review of the add-on design, C (16 Nov 2026)

Working paper: a reviewer's findings as returned. Nothing here is a rule.

**Verdict.** Do not register design C as written. I found no blocker, but 12 major points: rules that today's code or today's records cannot meet, guards that are missing, and choices left open. All are fixable in the text or in the small build. The code claims are mostly right (I checked about 40 against file and line; 3 are wrong or loose). The cost estimate is right: I recomputed 6.85 to 9.70 core-hours (design estimate) from the committed rates and Stage 1b's records. The most important reading gap: the stand-in also falls in 80 of 80 walk episodes at 8 ms (simulated, committed records), which the design does not state, and its "cured" sentence looks at 10 ms only. Hard limits: repository untouched (git status --porcelain: 0 lines after my last command); no episode, fit or training run. One deviation: my JSON-reading scripts and one --help call ran with python directly, not through tools/lowprio.sh (each under 1 s, read only). Commands that import the study's code ran through tools/lowprio.sh with the venv python, one thread, one at a time.

## 1. [major] Section 3.1 (import of r2lfit sets panels.VERDICT_ROWS["R2L-V0"]) and 3.4 (tests)

**Problem.** The import-time change to panels.VERDICT_ROWS breaks an existing test and leaks into every later use of the panels in that process. The design says no existing file changes and lists no such effect.

**Evidence.** tests/test_stage1_panels.py:302 asserts panels.VERDICT_ROWS == {"R2": ("R2/42", "R2/43"), "R3": ("R3/42", "R3/43")} exactly. Pytest imports every test module before it runs a test, so a new tests/test_stage2_r2lfit.py that imports r2lfit makes that assert fail. Stage 1's panels.summarize also writes verdict_rows into summary.json (panels.py:1984), so a Stage 1 summary written in such a process would differ from the committed one. barreport reads the same table (barreport.py:1486, 1780).

**Fix.** Do not change VERDICT_ROWS at import. Set it only inside the driver, in a context manager around bar.evaluate and bar.summarize that restores the old table. Add tests: after import and after a driver call panels.VERDICT_ROWS and panels.ATTRIBUTED equal today's; the full existing suite passes with the new files present; bar.summarize on a copy of the committed V0 block and barreport's BAR_TABLES.md and bar_tables.json of the 15 Nov run regenerate byte for byte with r2lfit imported. Also add a commit-level check: git diff --name-status from state A shows only added files under designlab/ and tests/. Check 1 of 3.3 (file hashes before and after an import) proves nothing: an import cannot change file bytes.

## 2. [major] CJ7 (labels every record carries)

**Problem.** A bar record cannot carry the seven new labels with today's code. The rule cannot be met without editing bar.py, which is in every record's key.

**Evidence.** bar.condition_record (bar.py:1903-1942) writes fixed fields. evaluate has no label argument; extra is filled only for a control (bar.py:2365). The record's rules are the 75 cm plan's texts (rules_of). The only free texts are the row's describe() (written to controller and to the key, bar.py:1926, 1702) and the unit name.

**Fix.** Reword CJ7: the labels are held in the row's gains_info (so in every record's controller field and key), in the fit and selection records, and in every table and sentence. Fix the label texts before tier 1, because they enter the key (a later change of wording would be a key mismatch under rule R2). Say that the records also carry the 75 cm plan's rule texts unchanged.

## 3. [major] CJ4 (five rows per condition and delay), CJ6 (R0@10ms beside every 10 ms count), CJ2 tiers 1 to 5

**Problem.** Several comparison rows do not exist, so the tables cannot be printed as the rule says and some new counts would stand alone.

**Evidence.** Registered V0 records (stage2/runs/2026-11-15-75cm/V0/75-M-H-10/seeds3042/eval): no R0 record of walk/nominal at any delay; at 2 and 8 ms only walk/v0.4 and walk/v0.4_mu0.54 exist under R2/42, R2/43 and R0 (my listing). So tier 5's walk/nominal at 8 and 2 ms has no stand-in row, and CJ5's third record (walk/nominal at 10 ms) has no R0 count. CJ8 forbids running any stand-in row.

**Fix.** Either drop walk/nominal from tier 5, or add the missing comparison records as new registered units of C2 in the new run directory: R2/42, R2/43 at 2 and 8 ms on walk/nominal, and R0@10ms (and R0@0ms) on walk/nominal. Cost (design estimate): 2 rows x 2 delays x 20 x 14.06 s x 0.1574 / 3600 = 0.05 core-hours, R0 about 0.015. Say in CJ4 that a missing comparison prints "no registered record". These are new conditions for old rows, not re-runs; say so against rule R2.

## 4. [major] Section 5 (expectation) and CJ5 (the reading of "cured"), CJ2 tier order

**Problem.** A known result is not stated: the stand-in falls at 8 ms as well. CJ5 reads 10 ms alone, and the 8 ms walks sit in tier 5, after three reported tiers, where the budget can cut them. A row that stands at 10 ms and falls at 8 ms would still be printed "the refit removes the 10 ms falls".

**Evidence.** Simulated, committed records: R2/42@8ms and R2/43@8ms, walk/v0.4 and walk/v0.4_mu0.54: 0 of 20 up in all four (80 of 80 fall); at 10 ms 76 of 80 fall; R0@8ms 19 and 20 of 20 up (my tabulation of the V0 records). One third of the fit's episodes (8 and 10 ms) are at delays where the start falls. LESSONS_LEARNED.md section 2.3 says a 10 ms verdict is read beside its neighbours.

**Fix.** State the 8 ms counts in section 5. Move the 8 ms walks to tier 2 (0.10 core-hours, design estimate). Make CJ5's sentence carry the 8 ms count of the same three records, and print "stays up at 10 ms but not at 8 ms" where that is so. Also state what the start's decisive cell is expected to show (about 0 to 5 of 100 up, from R2/43@10ms: 1, 0, 1, 0 to 1, 1 of 20).

## 5. [major] CJ4 (paired counts) and section 5, expectation 2 ("fewer falls than the start")

**Problem.** The pairing is not defined and there is no rule for reading "fewer falls". Two readers could pair R2L-V0/42 with R2/42 (same number) or with R2/43 (its start). A result such as 8 of 20 up would be worded after the fact.

**Evidence.** CF6: both fits start from seed43_gen40, the gains of R2/43 (I confirmed speedfit.start_record gives seed43_gen40, equal value for value to panels.r2_gains(…, 43)). CJ4 says only "under the new row only, under the stand-in only". CJ5 has two thresholds (20 of 20; 17 of 20) and nothing between. The stand-in's 10 ms counts are 0 to 3 of 20 up.

**Fix.** Write: each new row is paired by seed with R2/43@<ms>ms, its start; R2/42 is printed for reference and is not its pair. Put "both fits start from R2/43's gains" in every table head. Define "fewer falls": per record, an exact two-sided sign test at 0.05 on the 20 seed pairs (as H5 uses), reported, no verdict; or state that only counts are printed and no word is used. Confirm with the lead that one start for both fits is the reading of P37's "refit that controller (two fits)".

## 6. [major] CJ7 ("Not replicated"), CJ8 (no second seed block), section 8 point 10

**Problem.** The label reuses a registered word with another meaning, and the replication question is left to be decided after a result.

**Evidence.** STAGE2_75_PLAN.md H4 (lines 647-649): "not replicated" means the seeds5042 record fails. The runner prints "a control is not replicated" for V0 (bar.py:2836). H4's exclusion was written for control sets beside the candidate, not for a new controller whose pass would be news. Section 8.10: "If tier 1 shows a pass and the lead wants it repeated, that is a labelled amendment after a result." bar.evaluate refuses --passed-of on V0 (bar.py:2354-2356) but allows a plain seeds5042 run under a task.

**Fix.** Decide now. Either pre-register: every seeds3042 record of an R2L-V0 row at 10 ms that passes in reading 2 or 3 is run again on seeds5042 right after tier 1, inside C2 (0.018 core-hours per record, design estimate), printed beside and never pooled; or state that none is run and print "replication not run (pre-registered)". Never print the bare words "not replicated". Add: the 75 cm panels' registered R2 verdicts stand and are never merged with R2L-V0's cells.

## 7. [major] CF9 (what may run before generation 0) and CF7 (block estimate)

**Problem.** CF9 departs from the registered check rule without a label, lets the fit's own first episodes be seen before the fit, and CF7 does not say which rate it uses.

**Evidence.** bar.RULES["checks"] (bar.py:332-340): a build check is at most one episode per condition and R0 row on V0, "never an R2 row"; 75 H6: rates are measured on Stage 1's own conditions, "not on any reference of the three sets". CF9 (a) runs R2/43@10ms on V0 seed 3042; (b) runs generation 0's five episodes of seed 42 at the start's gains on V0. CF9 is tagged "[75 R4's pattern]". CF7 says "the committed rate"; section 8.8 leaves open whether COST.md is measured again or copied. No rule uses cost.json.

**Fix.** Drop (b): CF7 and C2 use rates.R2 = 0.1574 wall s per simulated s of the 15 Nov COST.md, copied by sha256 into the new directory (or a re-measurement on Stage 1's walk/brisk as on 15 Nov, decided now). Keep (a) but tag it "[differs: 75 amendment 5, R0 rows only]", list it in dev_episodes.json with that reason, and compare the evaluator's own judged episode (row.episode then bar.judge) with the committed episode record on every field but wall_s and trace. bar.judge does not return what write_trace needs (bar.py:1447-1453), so a trace sha256 can only come from bar.run_episode, which tests the bar and not the evaluator.

## 8. [major] Section 3.2, the driver (--task N --cap CORE_H)

**Problem.** The cap is a command-line argument. A stop rule could be changed at run time, and the driver could write into the registered tasks' ledger.

**Evidence.** Budget.load sets t["cap_core_h"] = self.cap on every call (bar.py:1806-1810), so a later call with a larger cap silently raises it. bar's own parser guards this with choices 3 and 4 (bar.py:3468); the driver removes that guard.

**Fix.** The driver reads task numbers and caps from a constant table committed with the rules, takes no --cap, refuses tasks 3 and 4, refuses --out equal to stage2/runs/2026-11-15-75cm, and refuses a cap that differs from the one already in budget.json. Test each refusal.

## 9. [major] CJ6 (trades) and section 7, first risk row

**Problem.** The named main risk (a "cure" by turning the feedback down, which loses push recovery) is not caught by the trade rule.

**Evidence.** CJ6 calls a trade only the loss of a 0 ms pass: the stop, the walk but for A1, walk/nominal. No nudge passes at 0 ms under the stand-in (19 to 20 of 20 up, 11 to 15 succeed; committed records), so a drop to R0's level (3 to 17 of 20 up at 0 ms) would not be called a trade.

**Fix.** Add a pre-registered push reading: per nudge condition at 0 ms and at 10 ms, episodes up under the new row against R2/43 (paired by seed) and against R0, with a fixed word, for example "loses push recovery" if the count up at 0 ms falls by 4 or more of 20 in any of the four μ 0.9 nudges. Reported, no verdict.

## 10. [major] Section 7 (code state drifts), CO, C2's budget

**Problem.** The dependence on items A and B is not a rule. If item B needs a floor that changes bar.py or setio.py, every new record has another bar source hash than the stand-in's, and CJ4 prints every pair as "other code state".

**Evidence.** Committed key.sources.bar = 45719d03… (R2/43@10ms/walk/v0.4.json; my read). The design prices the re-run of the stand-in at 2.4 to 3.4 core-hours (I get 2.37 at 0.1574 for two R2 rows at four delays) but C2's 4.5 core-hours cap does not hold it and CO does not say who decides.

**Fix.** Add to CO: item C's bar tiers run on the code state whose bar and panel source hashes equal the registered ones; the driver checks this before tier 1 and refuses otherwise. If a key file must change first, the stand-in's rows at 10, 0, 4 and 6 ms (and 8 ms) are tier 0 of C2 with a stated extra cap, both sets of records kept (S2-17).

## 11. [major] Section 8 (ten points for the lead) and the rule text

**Problem.** Choices that decide what is run are still open in the text to paste: task integers (C1, C2), the COST.md source, the start of the fits, tiers 5 and 6, the zero-gain candidate, fits in series or side by side. CO also lets the rules be committed after item A's results.

**Evidence.** Section 4 header: "the lead gives them integers"; section 8 points 1 to 10; CO steps 1 and 2 (A done, then these rules committed).

**Fix.** Close every point in the text before review of the final rules. Commit item C's rules before item A's first episode, or label them "committed after item A's results; no rule depends on them". Tier 6 (new rows at 14, 16, 20 ms) is outside the fit's mix and outside what P37 named; get the lead's yes or drop it.

## 12. [major] CF7 and CJ3 (one fit cut or not judged)

**Problem.** No rule says what is run and printed if one fit is "cut" (fewer than 12 generations) or fails. A verdict needs two fits.

**Evidence.** CF7: "A fit that reached fewer than 12 generations is not judged". CJ3 and bar.verdict (bar.py:2745) would then give "incomplete" for every cell; CJ5 has no case for it.

**Fix.** Add: if either fit is not judged, no bar tier runs for item C beyond the judged fit's tier 1, every cell reads "incomplete: one fit", and CJ5 reads "not answered". Also say whether an unused part of a fit's 2.4 core-hours passes to the selection (I read CS4's share as a fixed 0.6).

## 13. [minor] Section 3.4, objective test; CF5

**Problem.** The test "an evaluation with one more fall always scores worse" is false for tune.score.

**Evidence.** tune.py:687-702: S adds 1 + (1 − survived share) per fall. One fall at the start gives S = 2; two falls at the very end give S = 2. Then J differs only by P, and the one-fall case can have the larger P (up to 1). Example: J = 3.0 with one fall, about 2.4 with two.

**Fix.** Test what holds: every term in [0, 1], P ≤ 1, each fall adds at least 1 to S. Reword CF5's "a fall always outweighs every other term" to that.

## 14. [minor] CF4 and CS2 (episode seeds)

**Problem.** The seeds are the ones the start's gains were fitted and selected on in Stage 1, with the same derived start and noise draws. The start candidate has a small home advantage, and fit 43 differs from fit 42 in this.

**Evidence.** Stage 1 fitted R2/43 on tune.training_order(43)[g] and selected it on 2000 to 2019 (tune.py:106-107, 1173; tune.json settings). The bar derives start and noise seeds as tune.derived_seed(seed, 0, purpose) (bar.RULES["seeds"]). CF4 uses tune.training_order(fit seed)[5g + j]: for seed 43 its first 60 entries are Stage 1 fit 43's episodes. CS2's first block is 2000 to 2019.

**Fix.** Use seeds no committed record has used, for example an order from default_rng([seed, 2]) over 1000 to 1999 and validation seeds 6000 to 6099; or keep them and label the overlap.

## 15. [minor] CS4 (reported cells)

**Problem.** Five episodes per kind and delay is too few to read, the order of the 15 cells is not given, and the departure from S2-18 is not labelled.

**Evidence.** CS4: "validation evaluations 0 to 4 (25 episodes per cell)", "as far as the share reaches". S2-18: the selection "reads its validation episodes per delay". The round C draft labels the same cut "[differs: S2-18 … partly]" (PREREG_TRACKB_draft.md Q4).

**Fix.** Give the order (for example start, then selection 42, then 43; k = 0, 2, 4, 1, 3). Print every count as "n of 5" with no pass word. Add the [differs] tag.

## 16. [minor] CF3 (nudge direction per block)

**Problem.** The forward nudge is met twice as often as the others, and a fit judged at 12 or 18 generations never met the backward and right (or right) nudge.

**Evidence.** 30 generations are 5 blocks: B mod 4 = 0, 1, 2, 3, 0. CF7 allows judging at 12 generations.

**Fix.** State it in the label of a fit judged before generation 24, or rotate the direction by generation and show the coverage table in rotation.json.

## 17. [minor] CS1 (zero-gain candidate) and CF5 (failed evaluation scores as falls)

**Problem.** It is unverified that R2 with all 13 gains at zero runs at all. If it raises, the rule scores it as five falls, and the diagnostic would read as "falls in 100 of 100".

**Evidence.** control.GAIN_SPEC bounds allow 0 for all 13 (control.py:183-197), but no committed record ran that point. CF5: an evaluation that raises scores J = 11.

**Fix.** Print errors apart from falls in every selection cell. A candidate with any error reads "failed to run", not a count.

## 18. [minor] Section 2.2 point 4 and section 5 point 3 ("the step size never grew")

**Problem.** True for the Tapered body only.

**Evidence.** My read of history[*].sigma: 60-N-T-10 0.071 to 0.095; 60-N-H-10 0.076 to 0.107 (above sigma0 0.1); 60-N-T-10-K 0.039 to 0.095.

**Fix.** Write "stayed at or under 0.107".

## 19. [minor] CJ3, CJ7, CJ8 (no attribution) and CJ1 ("a free check")

**Problem.** Not attributing a judged controller's falls departs from 75 H5 read with S2-16, and is given as a tool limit, not as a rule. The check in CJ1 is not free.

**Evidence.** S2-16 applies section H to "the judged controller's two seeds in place of R2/42 and R2/43". bar.ATTRIBUTED_CONTROLLERS = ("R2",) (bar.py:1603), so failing 10 ms cells of the new pair will mostly print "not attributed" (bar.py:2763) beside the stand-in's attributed words. A row equal to its start still runs 340 episodes at 10 ms: 0.30 core-hours at 0.1574.

**Fix.** Tag the rule "[differs: 75 H5 with S2-16]". Count the check's cost in C2 or cut it to the three walk records.

## Checked and found right

Code claims I verified in the files: bar.py --task choices 3 and 4 (3468, 140); Budget takes any task with a cap (1800-1804); evaluate with a task passes _check_rule on V0 (2207); delays 0 to 20 ms in even steps (135-137, 852-856); rows() and --row-module (920-930, 3479, 3485-3486); workers rebuild rows from spec, class panels.TrackerRow (2141; panels.py 829-843, 853-857); register_rows and tracker_rows (panels.py 865-893); r2_gains (677-687); the four key files (bar.py 1677-1685); attribution for R2 only (1603, 2281); --passed-of refused on V0 and a plain seeds5042 run not refused (2354-2356); rate_of falls back to the largest rate (1889-1894); speedfit BUDGET 360, SEEDS (42,), delay drawn from the seed (93-94, 159-167, 282-285), servo_delay (325-361), start_record (597), the R2s hook (1210-1218); tune constants, hinge, score, training_order, candidate_generations, selection_key (104-109, 238-257, 687-702, 1173, 1384-1390); all 13 searched gains have lower bound 0 (control.py 183-197); no hash is taken over designlab/stage2 (globs only at tune.py 386, plant.py 271, panels.py 1512); regress check runs no episode. My commands: sha256 of the four key files equal the design's and the committed record's; `regress check` says current (256 of 256); speedfit.mix_files(): 19 files, 19 in Stage 1's run, 17 missing in V0; start_record gives seed43_gen40, passed, equal to R2/43's gains; panels.controller_of("R2L-V0/42") is "R2L-V0"; candidate_generations(30, 24, 18, 12) = thirds; 150 distinct fit seeds per fit; bar dry-run (plans only, rates given) accepts the new row names: 680 episodes, 0.593 core-hours for 16 gated conditions plus walk/nominal, two rows, one delay; planned seconds 21.08, 19.62, 20.05, 20.07, 20.38, 14.06. git check-ignore: records not ignored, traces, partial, .lock, logs ignored. Counts (simulated, committed V0 records): every figure of section 5's table; 76 of 80 walk falls at 10 ms split 29, 12, 35; 2 of 16 and 4 of 16 conditions pass at 0 ms; R0 counts; trunk start about 1.38 bound-ranges from zero gains. Stage 1b: 16 of 100 selection episodes at 10 ms, the walking turn at 10 ms in 2 of 30 generations, rates 0.1943 to 0.1973. Cost (design estimate, recomputed): one fit 34,340 simulated s = 1.50 core-hours at 0.1574 (COST.md rates.R2) to 2.13 at 0.2230; ledger overrun of the 10 V0 units at 0 to 6 ms 1.06 to 1.42, median 1.13 (budget.json); C1 91,104 s = 3.98 to 5.64; C2 65,449 s = 2.86 to 4.05; item 6.85 to 9.70 against caps 6.4 + 4.5; wall 2.3 to 3.2 h at 3 processes. Realised rates in the 15 Nov records at 3 processes were 0.178 to 0.198, so the middle columns are the likely ones. Rules that agree with the registered texts: S2-9 and S2-18 (fixed rotation, same episodes for every member, 10 ms decisive, table written first), H1 to H3 pass rule, B1 seeds and episodes, the unit-by-unit budget rule of H6, R2 no overwrite (new run directory), two fits (P37).

## Not checked

No episode was run, so nothing about the fit's behaviour is checked: whether tune.measure works on a bar turn, stop or nudge episode as CF5 needs; whether R2 runs with all gains at zero; whether a start-selected row reproduces R2/43's trace sha256; the real rate of a trace-less evaluation with 3 workers. The module r2lfit.py, its tests and the driver do not exist, so the build itself is unreviewed. I did not check barreport end to end on a run directory that holds only R2L-V0 rows; the design names no script for CJ4 to CJ6's tables and that work is in no file list or estimate. I did not read the round C design files beyond PREREG_TRACKB_draft.md section Q, nor the whole of STAGE2_75_REPORT.md; the toe-clearance figures (24.2, 18.3 to 18.9 mm against 20.07 mm) I checked against the report's text only. I did not check whether seeds 2000 to 2099 or 1000 to 1999 are used by other Stage 2 records. Items A and B were not reviewed; their effect on item C's code state is only flagged. Nothing here is measured on hardware; the 10 ms delay is assumed.
