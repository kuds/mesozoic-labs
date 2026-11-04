# Lane m4: M4 (K1-K4) at the PHASE J checkpoints and the pinned parent rule (report only)

Folder: /tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/measure/m4 (all paths below are relative to it unless absolute).
Nothing here changes the pinned reading (criteria.txt line 4, CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT CONFIRMATORY); M4 only feeds the parent rule.

## Provenance (two attempts)
- First attempt (2026-11-02 23:35Z to the container restart about 00:38Z): 12 judge records, PHASE s60 and s42 at all six J labels (panels/stance7_r10LTSBpc_s{60,42}/*/m4/judge_*.json, wall times in judge_status.txt); classes_pc.json (byte-equal to port/review_m4/out/parent/classes_pc.json, cmp rc 0, re-checked this attempt: logs/cmp_classes.out); parent/parent_noM4.{json,log}; s6ref/kscore_stance6_r10LTSB_s{60,42,80}.json (git show origin/main). Kept, not re-measured.
- This attempt (RESUME line in CMDLOG.txt at 00:40Z): the 6 PHASE s80 judges (run_judges_s80.sh, the first attempt's run_judges.sh restricted to s80; s80/4194304/m4 had been left empty by the killed job), the record check of all 18, the 3 kscore runs, the parent rule on the measured records, and this summary.

## Record checks (18 of 18 OK) [measured: records_check_all18.json, logs/check_records_all18.out]
Every record: hard_failures [], smoke false, mode full, no relaxation (task_binding "require", no allow flag true), model_sha256 = sha256 of the fetched zip, vecnorm_sha256 = the fetched pkl's, checkpoint num_timesteps = label (final 12,001,280), seed = run seed, pc_binding arm PHASE, git_commit_match true. Judge run: pc/judge_r10ltsb.py --arm LTSB --seed S --steps LABEL --clone port/m4/copies/copy_1dfe2327 --run-id RUN, as in port/m4/NOTES.md (literal commands in CMDLOG.txt).
kscore: every PHASE run's 6 cells VALID (missing [], invalid [], notes []) [measured: kscore/kscore_<RUN>.json].
Tool files unchanged: the 60 port/m4 entries of port_tools_SHA256SUMS_pre_measure.txt (excluding tmp_views and caches) all OK [measured: logs/sha_check_m4b.out].

## Per run and J label
Columns from the judge records via the pinned score_stance3.panel_counts (clean, LCB, K2a/K2b, roll rail) and the judge's noise blocks, K2c median/offset from kscore [measured: summary_data.json, kscore/kscore_<RUN>.json]. STANCE6 columns: origin/main docs/investigations/stance_2026_10/stance6_results/kscore/kscore_stance6_r10LTSB_s<S>.json, as recorded for STANCE6; OFF = STANCE6 bytes [measured: s6ref/]. "K4 roll rail" per label is the panel's roll-rail count; K4 itself scores only 6.3M and final. Judge seconds are the record's judge_seconds.

### stance7_r10LTSBpc_s60 (seed 60)  |  beside it: stance6_r10LTSB_s60 (as recorded for STANCE6; OFF = STANCE6 bytes)

| J label | v2 panel clean/40 | panel | LCB | joint noise falls/20 (seeds) | own-sigma falls/20 | K2a | K2b (seeds) | K2c median mm | K2c offset mm | K4 roll rail | judge s || STANCE6 clean/40 | STANCE6 K2a | STANCE6 K2b | STANCE6 K2c median | STANCE6 offset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.2M | 39 | PASS | 0.8868 | 1 [3061] | 0 [] | 1 | 1 [3072] | 50.389404 | -2.348374 | 0 | 294.7 || 40 | 0 | 0 [] | 47.913681 | -4.824097 |
| 5.2M | 40 | PASS | 0.9278 | 1 [3061] | 0 [] | 0 | 0 [] | 50.600236 | -2.137542 | 0 | 294.1 || 39 | 0 | 0 [] | 53.48909 | 0.751312 |
| 6.3M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 53.238611 | 0.500834 | 0 | 298.8 || 40 | 0 | 0 [] | 46.372337 | -6.365441 |
| 8.4M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 51.246048 | -1.49173 | 0 | 294.0 || 34 | 0 | 0 [] | 49.86458 | -2.873198 |
| 10.5M | 40 | PASS | 0.9278 | 1 [3061] | 0 [] | 0 | 0 [] | 51.38948 | -1.348298 | 0 | 293.3 || 38 | 0 | 0 [] | 51.813812 | -0.923966 |
| final | 40 | PASS | 0.9278 | 1 [3061] | 0 [] | 0 | 0 [] | 51.359925 | -1.377853 | 0 | 294.1 || 40 | 0 | 0 [] | 48.689683 | -4.048095 |

| verdict | PHASE (this lane) | STANCE6 (as recorded; OFF = STANCE6 bytes) |
|---|---|---|
| K1 | PASS pass_at {'6.3M': True, 'final': True} durable True below_37 [] below_33 [] | PASS pass_at {'6.3M': True, 'final': True} durable True below_37 ['8.4M'] below_33 [] |
| K1 failure modes | {"by_checkpoint": {}, "new_modes": false} | {"by_checkpoint": {"8.4M": {"rails": {"max_actuator_saturation_fraction": 6}, "terminations": {}}}, "new_modes": true} |
| K2 | PASS (K2a PASS, K2b PASS, K2c PASS window True range 2.849207 mm True) | PASS (K2a PASS, K2b PASS, K2c PASS window True range 7.116753 mm True) |
| K3 | PASS joint {'6.3M': 0, 'final': 1} own {'6.3M': 0, 'final': 0} | PASS joint {'6.3M': 1, 'final': 1} own {'6.3M': 0, 'final': 0} |
| K4 | PASS roll rail {'6.3M': 0, 'final': 0} | PASS roll rail {'6.3M': 0, 'final': 0} |
| complete / failed / HOLDS / FORE_AFT_HOLDS | True / [] / True / True | True / [] / True / True |
| cells missing / invalid / notes | [] / [] / [] | [] / [] / [] |

### stance7_r10LTSBpc_s42 (seed 42)  |  beside it: stance6_r10LTSB_s42 (as recorded for STANCE6; OFF = STANCE6 bytes)

| J label | v2 panel clean/40 | panel | LCB | joint noise falls/20 (seeds) | own-sigma falls/20 | K2a | K2b (seeds) | K2c median mm | K2c offset mm | K4 roll rail | judge s || STANCE6 clean/40 | STANCE6 K2a | STANCE6 K2b | STANCE6 K2c median | STANCE6 offset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.2M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 52.566153 | -0.171625 | 0 | 303.7 || 3 | 0 | 0 [] | 61.382802 | 8.645024 |
| 5.2M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 51.431635 | -1.306143 | 0 | 303.0 || 30 | 0 | 0 [] | 43.615857 | -9.12192 |
| 6.3M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 51.166842 | -1.570936 | 0 | 296.1 || 39 | 0 | 0 [] | 61.37324 | 8.635462 |
| 8.4M | 40 | PASS | 0.9278 | 1 [3061] | 0 [] | 0 | 0 [] | 50.454638 | -2.28314 | 0 | 280.8 || 4 | 0 | 0 [] | 63.147672 | 10.409894 |
| 10.5M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 51.138143 | -1.599635 | 0 | 277.2 || 7 | 0 | 0 [] | 54.105563 | 1.367785 |
| final | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 50.57451 | -2.163268 | 0 | 273.3 || 38 | 0 | 0 [] | 44.550599 | -8.187179 |

| verdict | PHASE (this lane) | STANCE6 (as recorded; OFF = STANCE6 bytes) |
|---|---|---|
| K1 | PASS pass_at {'6.3M': True, 'final': True} durable True below_37 [] below_33 [] | FAIL pass_at {'6.3M': True, 'final': False} durable False below_37 ['4.2M', '5.2M', '8.4M', '10.5M'] below_33 ['4.2M', '5.2M', '8.4M', '10.5M'] |
| K1 failure modes | {"by_checkpoint": {}, "new_modes": false} | {"by_checkpoint": {"4.2M": {"rails": {"max_actuator_saturation_fraction": 36, "max_settle_stance_width_change_m": 1}, "terminations": {}}, "5.2M": {"rails": {"max_actuator_saturation_fraction": 10, "max_settle_stance_width_change_m": 1}, "terminations": {}}, "8.4M": {"rails": {"max_actuator_saturation_fraction": 35, "max_settle_stance_width_change_m": 1}, "terminations": {}}, "10.5M": {"rails": {"max_actuator_saturation_fraction": 33}, "terminations": {}}, "final": {"rails": {"max_actuator_saturation_fraction": 2}, "terminations": {}}}, "new_modes": true} |
| K2 | PASS (K2a PASS, K2b PASS, K2c PASS window True range 2.111515 mm True) | PASS (K2a PASS, K2b PASS, K2c PASS window True range 19.531815 mm True) |
| K3 | PASS joint {'6.3M': 0, 'final': 0} own {'6.3M': 0, 'final': 0} | PASS joint {'6.3M': 1, 'final': 0} own {'6.3M': 0, 'final': 0} |
| K4 | PASS roll rail {'6.3M': 0, 'final': 0} | PASS roll rail {'6.3M': 0, 'final': 0} |
| complete / failed / HOLDS / FORE_AFT_HOLDS | True / [] / True / True | True / ['K1'] / False / False |
| cells missing / invalid / notes | [] / [] / [] | [] / [] / [] |

### stance7_r10LTSBpc_s80 (seed 80)  |  beside it: stance6_r10LTSB_s80 (as recorded for STANCE6; OFF = STANCE6 bytes)

| J label | v2 panel clean/40 | panel | LCB | joint noise falls/20 (seeds) | own-sigma falls/20 | K2a | K2b (seeds) | K2c median mm | K2c offset mm | K4 roll rail | judge s || STANCE6 clean/40 | STANCE6 K2a | STANCE6 K2b | STANCE6 K2c median | STANCE6 offset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4.2M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 50.168844 | -2.568934 | 0 | 365.6 || 39 | 0 | 0 [] | 45.64036 | -7.097418 |
| 5.2M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 49.872709 | -2.865069 | 0 | 371.1 || 37 | 0 | 0 [] | 41.239417 | -11.498361 |
| 6.3M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 49.816164 | -2.921613 | 0 | 352.0 || 5 | 0 | 0 [] | 37.029151 | -15.708626 |
| 8.4M | 40 | PASS | 0.9278 | 1 [3061] | 0 [] | 0 | 0 [] | 48.648748 | -4.089029 | 0 | 367.7 || 40 | 0 | 0 [] | 43.471954 | -9.265824 |
| 10.5M | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 47.093118 | -5.64466 | 0 | 372.9 || 40 | 0 | 0 [] | 44.788318 | -7.94946 |
| final | 40 | PASS | 0.9278 | 0 [] | 0 [] | 0 | 0 [] | 46.245101 | -6.492677 | 0 | 339.7 || 40 | 0 | 0 [] | 49.857776 | -2.880002 |

| verdict | PHASE (this lane) | STANCE6 (as recorded; OFF = STANCE6 bytes) |
|---|---|---|
| K1 | PASS pass_at {'6.3M': True, 'final': True} durable True below_37 [] below_33 [] | FAIL pass_at {'6.3M': False, 'final': True} durable False below_37 ['6.3M'] below_33 ['6.3M'] |
| K1 failure modes | {"by_checkpoint": {}, "new_modes": false} | {"by_checkpoint": {"6.3M": {"rails": {"max_actuator_saturation_fraction": 35}, "terminations": {}}}, "new_modes": true} |
| K2 | PASS (K2a PASS, K2b PASS, K2c PASS window True range 3.923743 mm True) | PASS (K2a PASS, K2b PASS, K2c PASS window True range 12.828625 mm True) |
| K3 | PASS joint {'6.3M': 0, 'final': 0} own {'6.3M': 0, 'final': 0} | FAIL joint {'6.3M': 5, 'final': 2} own {'6.3M': 0, 'final': 0} |
| K4 | PASS roll rail {'6.3M': 0, 'final': 0} | PASS roll rail {'6.3M': 0, 'final': 0} |
| complete / failed / HOLDS / FORE_AFT_HOLDS | True / [] / True / True | True / ['K1', 'K3'] / False / False |
| cells missing / invalid / notes | [] / [] / [] | [] / [] / [] |

## Notes on the PHASE panels
- The one unclean PHASE panel episode in 18 x 40 = 720: s60 at 4.2M, seed 3072, fell at length 221 (terminations fallen 1; reasons horizon, window displacement 0.21 m, sole tilt 36.6 deg, corner lift 0.088 m, yaw change 37.8 deg, CoP fore-aft 0.997); it is also the K2a and K2b count 1 at that label. Panel still PASS (39/40, LCB 0.8868) [measured: logs/s60_4M_unclean.out]. 4.2M is not a K1 slot.
- Joint-space noise falls: every PHASE fall is spawn seed 3061, 1 of 20 where present (s60 4.2M, 5.2M, 10.5M, final; s42 8.4M; s80 8.4M). Own-sigma noise falls 0 of 20 at all 18 [measured: records_check_all18.json].
- K1 failure modes: none for any PHASE run (by_checkpoint {}, new_modes false) [measured: kscore JSON].

## K1-K4 verdicts [measured: kscore/kscore_<RUN>.json]
| run | K1 | K2 | K3 | K4 | failed | STANCE6 same seed (as recorded) |
|---|---|---|---|---|---|---|
| stance7_r10LTSBpc_s60 | PASS | PASS | PASS | PASS | [] | K1 PASS, K2 PASS, K3 PASS, K4 PASS, failed [] |
| stance7_r10LTSBpc_s42 | PASS | PASS | PASS | PASS | [] | K1 FAIL (final not passed; below 33 at 4.2M, 5.2M, 8.4M, 10.5M; saturation rail), failed ['K1'] |
| stance7_r10LTSBpc_s80 | PASS | PASS | PASS | PASS | [] | K1 FAIL (6.3M 5/40; saturation rail), K3 FAIL (joint falls 5 at 6.3M, 2 at final), failed ['K1','K3'] |

## Parent rule (pc/parent_m4.py, line 4, classes = score_pc.json per_seed PHASE rows; all three gate PASS, outcome POSITIVE)
1. Reproduction without M4 records (round-dir handoff/round): named null, "not named: an input is absent", same candidate runs as score_pc.json reading.parent (only the round-dir path in the 'missing' text differs) [measured: parent/parent_noM4.json, parent/parent_noM4_compare.txt; run in the first attempt, comparison re-run this attempt].
2. On these records (round-dir this folder; each panels/RUN/final/m4/ holds only the valid full-mode judge_final.json): named [stance7_r10LTSBpc_s42, stance7_r10LTSBpc_s80], tie true. Basis: clean panel episodes at final 40 / 40 / 40 (s60 / s42 / s80), all tied; then noise falls (own + joint) 1 / 0 / 0, so s60 drops out and s42 and s80 tie and are both named [measured: parent/parent_M4.json].
3. Both named finals pass K1-K4: s42 and s80 each K1 (final 40/40, LCB 0.9278, full 40, hop_or_fall 0), K2, K3 (final joint 0, own 0), K4 (final roll rail 0) all PASS [measured: kscore/kscore_stance7_r10LTSBpc_s{42,80}.json].

## Seconds per judge [measured: record judge_seconds; wall in judge_status.txt]
s60: 294.7, 294.1, 298.8, 294.0, 293.3, 294.1 (first attempt). s42: 303.7, 303.0, 296.1, 280.8, 277.2, 273.3 (first attempt). s80: 365.6, 371.1, 352.0, 367.7, 372.9, 339.7 (this attempt; slower, inferred from another lane running in parallel on the 4 CPUs).
