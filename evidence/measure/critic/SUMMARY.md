# Critic lane: PC round critic readouts (REPORT ONLY)

Lane folder `L = /tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/measure/critic/`.
Written 2026-11-03 (UTC). Every command and its UTC start time are in `L/CMDLOG.txt`.
Labels: **[measured: file]** = read from a file this lane wrote or read; **[inferred: basis]** = reasoning.
Nothing here changes the pinned reading (`score_pc.txt` line 357: CONSISTENT WITH THE PHASE CRITIC REMOVING THE HABIT, NOT CONFIRMATORY).

## 0. What was run, with what
- Tool: the critic port, run from a sha256-verified copy at `L/tool/` (`L/tool_copy_sha256_check.txt` at 23:35 UTC and
  `L/logs/tool_copy_sha256_check_resume.txt` at 00:40 UTC: all 12 files OK against `port/critic/TOOL_SHA256SUMS.txt`;
  the port's tool files also unchanged against `port_tools_SHA256SUMS_pre_measure.txt`, `L/logs/port_tools_check_resume.txt` empty, rc 0).
  CRITIC_CODE=1dfe2327, OMP/MKL threads 1, nice 10, one simulation at a time (`L/scripts/seq.sh`).
- Loop readouts: `critic_run.py loop --run <RUN> --steps all --out L/out/loop_<RUN>` (fresh --out per run, never the builder's rows).
  The arm is read from `stage_config.json` and checked against the zip's policy class: PHASE rows have critic form
  `phase+raw+settled`, OFF rows `plain` [measured: `L/out/loop_*/loop.jsonl`].
- V - G: `vg_tool/vg_sigma.py --pair <final.zip> <final_vecnorm.pkl> --form phase|plain --seed 6271 --episodes 32
  --expect-steps 12001280 --out L/out/vg_<RUN>` (PS production settings), then `critic_run.py vg-summary --out L/out/vg_summary <6 json>`.
- Before the finals: each run's `stage1_final.zip` 'data' member read as JSON (never unpickled): num_timesteps 12001280
  in all six, matching pkl present [measured: `L/logs/final_check.txt`, ALL OK].
- Validity [measured: `L/logs/validate.txt`]: all six loop.jsonl have 12 rows in grid order + final, right run, code_key
  1dfe2327, right arm, zip sha256 recomputed = row's; one tool_sha256 set per run. All six vg_sigma.json: status
  measured, problems [], mode pair, num_timesteps 12001280, zip sha256 = the fetched zip's, 32 episodes all truncated at 1000, seed 6271.
- Equality checks (existing outputs, no re-run for this purpose) [measured]: OFF s80 all 12 loop rows = builder's
  `port/critic/out/v1_off_s80` 116/116 numeric leaves bitwise (`L/logs/cmp_prior_off80.txt`); OFF s80 V-G = builder's
  `v2_vg_off_s80_final` 84/84 bitwise, PHASE s42 V-G = builder's `v4_vg_pc_s42_final` 153/153 bitwise
  (`L/logs/cmp_vg_prior.txt`); PHASE s42 final, OFF s60 5,242,880, OFF s42 1,048,576 / 2,097,152 / 3,145,728 rows
  bitwise = builder / reviewer rows (`L/logs/cmp_prior_{pc42,off60,off42}.txt`, first attempt).
- OFF vs STANCE6 early-warning records (`rec/ew_series.csv`, records-phase-screen 4ff705f7, 5 dp): 36 pairs x
  {own14, c0_own, G_own, FP_own, c0_ref, G_ref, FP_ref} = 252 comparisons, max |d| 4.98e-6 (all within 5 dp rounding);
  the 10 early-bias slopes on record (`rec/eb_realized.json` sens_N) bitwise equal (max |d| 0.0) [measured: `L/logs/tables_full.txt` last line].
- PHASE self-checks [measured: `L/logs/phase_checks.txt`]: actor phase invariance 0.0 at all 36 PHASE pairs; max |value_net
  phase weights| used 0.125-0.150 (phase in force); on-policy n 16,384 per pair; clipped states <= 15 (PHASE), 0 (OFF).
- Resume: the first attempt (23:35-00:38 UTC) made the loop rows of PHASE s60/s42/s80, OFF s60/s42 (12 each) and OFF s80's
  first 8 (1,048,576-8,388,608; utc 00:32:35-00:38:31); this attempt (from 00:40:49 UTC) made OFF s80 9,437,184-final
  (utc 00:42:10-00:45:15, `critic_run` skipped the 8 present pairs), all six V-G and the vg-summary.

## 1. Slope, loop gain G, offset c0, margin b* per seed and pair
Column meanings. Slopes: dV/dq3 per degree on the on-policy states (one collect_rollouts per seed 9301 and 9302, 16,384
states). PHASE: "eff" = phase fed from k, "settled" = phi(1000), "raw" = phase columns zero (report only, an input the
critic never saw in training). OFF: the stock critic with its own feed. G, c0, b*_P = 1 - G - c0: OWN states (actor only).
"S6 ... (rec)" = STANCE6's values AS RECORDED in `ew_series.csv` (5 dp); "S6 b*" is 1 - G - c0 of the two recorded
values (derived); "S6 eb slope (rec)" = early-bias record where one exists. All PHASE/OFF columns [measured: `L/out/loop_<RUN>/loop.jsonl`]
via `L/scripts/tables.py` -> `L/logs/tables_full.txt`.
### seed 60: on-policy states (seeds 9301/9302) critic slope dV/dq3 per degree; loop G, c0, b*_P = 1 - G - c0 (OWN)
| steps | PHASE eff | PHASE settled | PHASE raw | PHASE G | PHASE c0 | PHASE b* | OFF slope | OFF G | OFF c0 | OFF b* | S6 G_own (rec) | S6 c0_own (rec) | S6 b* (1-G-c0 of rec) | S6 eb slope (rec) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 |  -0.00954 |  -0.00725 |  -0.02084 |  -1.69830 |  -0.05120 |  +2.74950 |  -0.00361 |  -1.38134 |  +0.01549 |  +2.36584 |  -1.38134 |  +0.01549 |  +2.36585 |        -- |
| 2,097,152 |  -0.00343 |  -0.00189 |  -0.01258 |  -2.59961 |  -0.04418 |  +3.64379 |  +0.00331 |  -1.74933 |  +0.01726 |  +2.73208 |  -1.74933 |  +0.01726 |  +2.73207 |        -- |
| 3,145,728 |  -0.01142 |  -0.01058 |  -0.01637 |  -3.37763 |  -0.02362 |  +4.40126 |  -0.00902 |  -1.94288 |  +0.04009 |  +2.90280 |  -1.94288 |  +0.04009 |  +2.90279 |        -- |
| 4,194,304 |  -0.01624 |  -0.01455 |  -0.02960 |  -3.34905 |  -0.00161 |  +4.35066 |  +0.01002 |  -1.57262 |  +0.17934 |  +2.39328 |  -1.57262 |  +0.17934 |  +2.39328 |        -- |
| 5,242,880 |  -0.01948 |  -0.01769 |  -0.03259 |  -3.49767 |  +0.00949 |  +4.48819 |  -0.00143 |  -0.51215 |  +0.18114 |  +1.33101 |  -0.51215 |  +0.18114 |  +1.33101 |  -0.00143 |
| 6,291,456 |  +0.01716 |  +0.01666 |  +0.02383 |  -4.41474 |  +0.02444 |  +5.39030 |  +0.03623 |  -0.07144 |  +0.28750 |  +0.78394 |  -0.07144 |  +0.28750 |  +0.78394 |  +0.03623 |
| 7,340,032 |  -0.01267 |  -0.01153 |  -0.01997 |  -4.59204 |  -0.01076 |  +5.60280 |  +0.02484 |  +0.31263 |  +0.48970 |  +0.19766 |  +0.31263 |  +0.48970 |  +0.19767 |  +0.02484 |
| 8,388,608 |  -0.01889 |  -0.01603 |  -0.04059 |  -3.97939 |  -0.04689 |  +5.02628 |  +0.03497 |  +0.36115 |  +0.61417 |  +0.02467 |  +0.36115 |  +0.61417 |  +0.02468 |  +0.03497 |
| 9,437,184 |  -0.02151 |  -0.01811 |  -0.05206 |  -3.82744 |  -0.00206 |  +4.82951 |  +0.01265 |  +0.31410 |  +0.64070 |  +0.04520 |  +0.31410 |  +0.64070 |  +0.04520 |        -- |
| 10,485,760 |  -0.00648 |  -0.00354 |  -0.03265 |  -3.47918 |  -0.03535 |  +4.51453 |  +0.02514 |  +0.36373 |  +0.51931 |  +0.11697 |  +0.36373 |  +0.51931 |  +0.11696 |        -- |
| 11,534,336 |  -0.00098 |  +0.00241 |  -0.03046 |  -3.78178 |  -0.06643 |  +4.84821 |  +0.01520 |  +0.43250 |  +0.38015 |  +0.18735 |  +0.43250 |  +0.38015 |  +0.18735 |        -- |
| 12,001,280 |  +0.01856 |  +0.02170 |  -0.00962 |  -3.74973 |  -0.06738 |  +4.81710 |  +0.01232 |  +0.46514 |  +0.32215 |  +0.21271 |  +0.46514 |  +0.32215 |  +0.21271 |        -- |

seed 60 OWN / REF slopes (PHASE effective | settled | raw ; OFF):
| steps | PH OWN eff | PH OWN set | PH OWN raw | OFF OWN | PH REF eff | PH REF raw | OFF REF | PH REF G | PH REF c0 | OFF REF G | OFF REF c0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 |  -0.00413 |  -0.00413 |  -0.01396 |  -0.00625 |  -0.01533 |  -0.02758 |  +0.00321 |  -1.53338 |  -0.13005 |  -1.32482 |  -0.01724 |
| 2,097,152 |  +0.00097 |  +0.00097 |  -0.00583 |  +0.01192 |  -0.00529 |  -0.01340 |  +0.01016 |  -2.37421 |  -0.09820 |  -1.66658 |  -0.02091 |
| 3,145,728 |  -0.00869 |  -0.00869 |  -0.01022 |  -0.00104 |  -0.01053 |  -0.01382 |  -0.02162 |  -3.12667 |  -0.05912 |  -1.86673 |  +0.08072 |
| 4,194,304 |  -0.01271 |  -0.01271 |  -0.02596 |  +0.02218 |  -0.00565 |  -0.01520 |  -0.03700 |  -3.09268 |  -0.04841 |  -1.49025 |  +0.13889 |
| 5,242,880 |  -0.01368 |  -0.01368 |  -0.02528 |  +0.01753 |  -0.00737 |  -0.01802 |  +0.01488 |  -3.31181 |  +0.01026 |  -0.49318 |  +0.07546 |
| 6,291,456 |  +0.01900 |  +0.01900 |  +0.03010 |  +0.04285 |  +0.02542 |  +0.03420 |  +0.02859 |  -4.25686 |  +0.02858 |  -0.09435 |  +0.05582 |
| 7,340,032 |  -0.00903 |  -0.00903 |  -0.01501 |  +0.02154 |  +0.00189 |  -0.00662 |  -0.00078 |  -4.51126 |  +0.05522 |  +0.26923 |  -0.07886 |
| 8,388,608 |  -0.01402 |  -0.01402 |  -0.03672 |  +0.03748 |  +0.00105 |  -0.01989 |  +0.00350 |  -3.97371 |  +0.03813 |  +0.32689 |  -0.11908 |
| 9,437,184 |  -0.01348 |  -0.01348 |  -0.04675 |  +0.01146 |  +0.00691 |  -0.01975 |  -0.00351 |  -3.77899 |  +0.06316 |  +0.27995 |  -0.14796 |
| 10,485,760 |  +0.00310 |  +0.00310 |  -0.02337 |  +0.02435 |  +0.01081 |  -0.01124 |  +0.00238 |  -3.45054 |  +0.02600 |  +0.32824 |  -0.22811 |
| 11,534,336 |  +0.00880 |  +0.00880 |  -0.02266 |  +0.01737 |  +0.01827 |  -0.00384 |  -0.00848 |  -3.77572 |  -0.00122 |  +0.39962 |  -0.18359 |
| 12,001,280 |  +0.02517 |  +0.02517 |  -0.00688 |  +0.01708 |  +0.03912 |  +0.01773 |  -0.00138 |  -3.78084 |  +0.00836 |  +0.42762 |  -0.17803 |

### seed 42: on-policy states (seeds 9301/9302) critic slope dV/dq3 per degree; loop G, c0, b*_P = 1 - G - c0 (OWN)
| steps | PHASE eff | PHASE settled | PHASE raw | PHASE G | PHASE c0 | PHASE b* | OFF slope | OFF G | OFF c0 | OFF b* | S6 G_own (rec) | S6 c0_own (rec) | S6 b* (1-G-c0 of rec) | S6 eb slope (rec) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 |  +0.01070 |  +0.00726 |  +0.03418 |  -0.43195 |  -0.10018 |  +1.53213 |  -0.00781 |  -0.72536 |  -0.05916 |  +1.78451 |  -0.72536 |  -0.05916 |  +1.78452 |  -0.00781 |
| 2,097,152 |  +0.00179 |  +0.00236 |  +0.00121 |  -1.14912 |  -0.06559 |  +2.21471 |  +0.02143 |  -0.63639 |  +0.23771 |  +1.39868 |  -0.63639 |  +0.23771 |  +1.39868 |  +0.02143 |
| 3,145,728 |  +0.01464 |  +0.01477 |  +0.01763 |  -1.91733 |  -0.08891 |  +3.00624 |  +0.05854 |  +0.34893 |  +0.41142 |  +0.23965 |  +0.34893 |  +0.41142 |  +0.23965 |  +0.05854 |
| 4,194,304 |  -0.00242 |  -0.00130 |  -0.00724 |  -2.29499 |  -0.03905 |  +3.33404 |  +0.06516 |  +0.52885 |  +0.51110 |  -0.03995 |  +0.52885 |  +0.51110 |  -0.03995 |        -- |
| 5,242,880 |  +0.00300 |  +0.00365 |  +0.00022 |  -2.37990 |  -0.02969 |  +3.40959 |  +0.05210 |  +0.52281 |  +0.41435 |  +0.06285 |  +0.52281 |  +0.41435 |  +0.06284 |        -- |
| 6,291,456 |  -0.01145 |  -0.00911 |  -0.03054 |  -2.03336 |  +0.00610 |  +3.02726 |  +0.00702 |  +0.50720 |  +0.14930 |  +0.34350 |  +0.50720 |  +0.14930 |  +0.34350 |        -- |
| 7,340,032 |  +0.01611 |  +0.02181 |  -0.03115 |  -1.84437 |  -0.05316 |  +2.89752 |  +0.02287 |  +0.63426 |  +0.35033 |  +0.01541 |  +0.63426 |  +0.35033 |  +0.01541 |        -- |
| 8,388,608 |  -0.00302 |  +0.00118 |  -0.03687 |  -1.68863 |  +0.05386 |  +2.63477 |  +0.01032 |  +0.73886 |  +0.30567 |  -0.04453 |  +0.73886 |  +0.30567 |  -0.04453 |        -- |
| 9,437,184 |  +0.01178 |  +0.01366 |  -0.00738 |  -1.65072 |  +0.03526 |  +2.61545 |  +0.03974 |  +0.83132 |  +0.16406 |  +0.00462 |  +0.83132 |  +0.16406 |  +0.00462 |        -- |
| 10,485,760 |  +0.00496 |  +0.00772 |  -0.02071 |  -1.36993 |  +0.04570 |  +2.32423 |  +0.06747 |  +0.69774 |  +0.35751 |  -0.05525 |  +0.69774 |  +0.35751 |  -0.05525 |        -- |
| 11,534,336 |  +0.00123 |  +0.00346 |  -0.02124 |  -1.66068 |  +0.06033 |  +2.60034 |  +0.03351 |  +0.55004 |  +0.41523 |  +0.03473 |  +0.55004 |  +0.41523 |  +0.03473 |        -- |
| 12,001,280 |  -0.00178 |  +0.00077 |  -0.02967 |  -1.84827 |  +0.02005 |  +2.82821 |  +0.04726 |  +0.48255 |  +0.50359 |  +0.01386 |  +0.48255 |  +0.50359 |  +0.01386 |        -- |

seed 42 OWN / REF slopes (PHASE effective | settled | raw ; OFF):
| steps | PH OWN eff | PH OWN set | PH OWN raw | OFF OWN | PH REF eff | PH REF raw | OFF REF | PH REF G | PH REF c0 | OFF REF G | OFF REF c0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 |  +0.00638 |  +0.00638 |  +0.03647 |  +0.00428 |  -0.01083 |  +0.00548 |  +0.00071 |  -0.38891 |  -0.16127 |  -0.70413 |  -0.09700 |
| 2,097,152 |  -0.00057 |  -0.00057 |  +0.00012 |  +0.03040 |  +0.00047 |  -0.00526 |  +0.05960 |  -1.09365 |  -0.07953 |  -0.61949 |  +0.12442 |
| 3,145,728 |  +0.01065 |  +0.01065 |  +0.01565 |  +0.06477 |  +0.01047 |  +0.00875 |  +0.04593 |  -1.87812 |  -0.03865 |  +0.32630 |  +0.21338 |
| 4,194,304 |  -0.00389 |  -0.00389 |  -0.00756 |  +0.07407 |  +0.00096 |  -0.00483 |  +0.02627 |  -2.19174 |  -0.00458 |  +0.51282 |  -0.07867 |
| 5,242,880 |  +0.00110 |  +0.00110 |  +0.00084 |  +0.05695 |  +0.01037 |  +0.01077 |  +0.01967 |  -2.28088 |  +0.00424 |  +0.53641 |  -0.13947 |
| 6,291,456 |  -0.01238 |  -0.01238 |  -0.03137 |  +0.00931 |  +0.00495 |  -0.00584 |  -0.01296 |  -1.94151 |  +0.06616 |  +0.47846 |  -0.16047 |
| 7,340,032 |  +0.02195 |  +0.02195 |  -0.03179 |  +0.02663 |  +0.04093 |  +0.00094 |  +0.01103 |  -1.83145 |  +0.07224 |  +0.61344 |  -0.13302 |
| 8,388,608 |  -0.00140 |  -0.00140 |  -0.03637 |  +0.00783 |  +0.02901 |  +0.00379 |  +0.00499 |  -1.60635 |  +0.09470 |  +0.71934 |  -0.18043 |
| 9,437,184 |  +0.01326 |  +0.01326 |  -0.00415 |  +0.04157 |  +0.03841 |  +0.02916 |  +0.02347 |  -1.57028 |  +0.09465 |  +0.78660 |  -0.24966 |
| 10,485,760 |  +0.00668 |  +0.00668 |  -0.01987 |  +0.07265 |  +0.03444 |  +0.01675 |  +0.04472 |  -1.30117 |  +0.07656 |  +0.67911 |  -0.12963 |
| 11,534,336 |  +0.00064 |  +0.00064 |  -0.02168 |  +0.03687 |  +0.02417 |  +0.00894 |  +0.03402 |  -1.58589 |  +0.09851 |  +0.55436 |  +0.06235 |
| 12,001,280 |  -0.00133 |  -0.00133 |  -0.03058 |  +0.05265 |  +0.01818 |  -0.00480 |  +0.03868 |  -1.78356 |  +0.08962 |  +0.48660 |  +0.11013 |

### seed 80: on-policy states (seeds 9301/9302) critic slope dV/dq3 per degree; loop G, c0, b*_P = 1 - G - c0 (OWN)
| steps | PHASE eff | PHASE settled | PHASE raw | PHASE G | PHASE c0 | PHASE b* | OFF slope | OFF G | OFF c0 | OFF b* | S6 G_own (rec) | S6 c0_own (rec) | S6 b* (1-G-c0 of rec) | S6 eb slope (rec) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 |  -0.00082 |  -0.00090 |  -0.00423 |  -0.95227 |  -0.02022 |  +1.97249 |  +0.00435 |  -1.37871 |  -0.03614 |  +2.41486 |  -1.37871 |  -0.03614 |  +2.41485 |  +0.00435 |
| 2,097,152 |  +0.00387 |  +0.00295 |  +0.00870 |  -1.54257 |  -0.03118 |  +2.57375 |  +0.02962 |  -1.97729 |  +0.08237 |  +2.89492 |  -1.97729 |  +0.08237 |  +2.89492 |  +0.02962 |
| 3,145,728 |  -0.00752 |  -0.00799 |  -0.00769 |  -2.21516 |  +0.00198 |  +3.21318 |  -0.00550 |  -1.16386 |  +0.17849 |  +1.98537 |  -1.16386 |  +0.17849 |  +1.98537 |  -0.00550 |
| 4,194,304 |  -0.00236 |  -0.00371 |  +0.00497 |  -2.01348 |  -0.04425 |  +3.05773 |  +0.01645 |  -1.12397 |  +0.03439 |  +2.08958 |  -1.12397 |  +0.03439 |  +2.08958 |        -- |
| 5,242,880 |  -0.00430 |  -0.00513 |  -0.00391 |  -1.71705 |  -0.03098 |  +2.74803 |  +0.03479 |  -0.57080 |  -0.03623 |  +1.60703 |  -0.57080 |  -0.03623 |  +1.60703 |        -- |
| 6,291,456 |  +0.00716 |  +0.00610 |  +0.00523 |  -1.92828 |  -0.00958 |  +2.93787 |  +0.01682 |  -0.20492 |  +0.22678 |  +0.97814 |  -0.20492 |  +0.22678 |  +0.97814 |        -- |
| 7,340,032 |  +0.00673 |  +0.00541 |  +0.00616 |  -1.73172 |  -0.00213 |  +2.73385 |  +0.01877 |  -0.13366 |  +0.15503 |  +0.97863 |  -0.13366 |  +0.15503 |  +0.97863 |        -- |
| 8,388,608 |  +0.00256 |  +0.00049 |  +0.00984 |  -1.75382 |  +0.05254 |  +2.70127 |  +0.04909 |  +0.06555 |  +0.12075 |  +0.81370 |  +0.06555 |  +0.12075 |  +0.81370 |        -- |
| 9,437,184 |  -0.01306 |  -0.01655 |  +0.00823 |  -1.62828 |  +0.05548 |  +2.57280 |  +0.02082 |  -0.16085 |  +0.28384 |  +0.87701 |  -0.16085 |  +0.28384 |  +0.87701 |        -- |
| 10,485,760 |  +0.00001 |  -0.00344 |  +0.02392 |  -1.79986 |  +0.03376 |  +2.76610 |  +0.03008 |  +0.02846 |  +0.46604 |  +0.50550 |  +0.02846 |  +0.46604 |  +0.50550 |        -- |
| 11,534,336 |  +0.01141 |  +0.00694 |  +0.04219 |  -1.72172 |  +0.04982 |  +2.67190 |  +0.03407 |  +0.19890 |  +0.49961 |  +0.30149 |  +0.19890 |  +0.49961 |  +0.30149 |        -- |
| 12,001,280 |  +0.00722 |  +0.00364 |  +0.02411 |  -1.71299 |  +0.07718 |  +2.63582 |  +0.04635 |  +0.27256 |  +0.48613 |  +0.24132 |  +0.27256 |  +0.48613 |  +0.24131 |        -- |

seed 80 OWN / REF slopes (PHASE effective | settled | raw ; OFF):
| steps | PH OWN eff | PH OWN set | PH OWN raw | OFF OWN | PH REF eff | PH REF raw | OFF REF | PH REF G | PH REF c0 | OFF REF G | OFF REF c0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 |  +0.00334 |  +0.00334 |  +0.01300 |  +0.00303 |  +0.00375 |  +0.01743 |  +0.00947 |  -0.89330 |  -0.06299 |  -1.35031 |  -0.02909 |
| 2,097,152 |  +0.00195 |  +0.00195 |  +0.00829 |  +0.03203 |  -0.00893 |  +0.00778 |  +0.05790 |  -1.44631 |  -0.06085 |  -1.90445 |  +0.05224 |
| 3,145,728 |  -0.00924 |  -0.00924 |  -0.00905 |  -0.00572 |  -0.01941 |  -0.00572 |  +0.00112 |  -2.02036 |  -0.07536 |  -1.17279 |  +0.07412 |
| 4,194,304 |  -0.00306 |  -0.00306 |  +0.00516 |  +0.02050 |  -0.01848 |  +0.00214 |  -0.00663 |  -1.86578 |  -0.10328 |  -1.10984 |  +0.03512 |
| 5,242,880 |  -0.00309 |  -0.00309 |  -0.00299 |  +0.03248 |  -0.02079 |  -0.00307 |  +0.00735 |  -1.54552 |  -0.11993 |  -0.58010 |  +0.03755 |
| 6,291,456 |  +0.00926 |  +0.00926 |  +0.00714 |  +0.00888 |  -0.01367 |  +0.00311 |  +0.01111 |  -1.73531 |  -0.09309 |  -0.17869 |  +0.00366 |
| 7,340,032 |  +0.00718 |  +0.00718 |  +0.00533 |  +0.01205 |  -0.01411 |  +0.00301 |  +0.03413 |  -1.54342 |  -0.08823 |  -0.11258 |  -0.01862 |
| 8,388,608 |  +0.00116 |  +0.00116 |  +0.00815 |  +0.05740 |  -0.01760 |  +0.00711 |  +0.06520 |  -1.51035 |  -0.09845 |  +0.07476 |  -0.03766 |
| 9,437,184 |  -0.01610 |  -0.01610 |  +0.00601 |  +0.01349 |  -0.04994 |  -0.01695 |  +0.06139 |  -1.40602 |  -0.07663 |  -0.14075 |  +0.00768 |
| 10,485,760 |  +0.00175 |  +0.00175 |  +0.02671 |  +0.02288 |  -0.03947 |  -0.00884 |  +0.05621 |  -1.57951 |  -0.06086 |  +0.03524 |  +0.04776 |
| 11,534,336 |  +0.01053 |  +0.01053 |  +0.04349 |  +0.02770 |  -0.03535 |  -0.00012 |  +0.03940 |  -1.47539 |  -0.08737 |  +0.21352 |  +0.00530 |
| 12,001,280 |  +0.00774 |  +0.00774 |  +0.02562 |  +0.04595 |  -0.03380 |  -0.00668 |  +0.04298 |  -1.45950 |  -0.08217 |  +0.27378 |  +0.00017 |


Per-arm ranges over the 12 pairs [measured: `L/logs/aggregates.txt`]:
- PHASE on-policy effective slope, median |.|: s60 0.01445, s42 0.00399, s80 0.00551 (ranges -0.02151..+0.01856, -0.01145..+0.01611, -0.01306..+0.01141).
- OFF on-policy slope, median |.|: s60 0.01248, s42 0.03662, s80 0.02522 (ranges -0.00902..+0.03623, -0.00781..+0.06747, -0.00550..+0.04909).
- PHASE loop gain G: negative at all 36 pairs (s60 -4.59204..-1.69830, s42 -2.37990..-0.43195, s80 -2.21516..-0.95227); b*_P min +2.74950 / +1.53213 / +1.97249.
- OFF loop gain G: positive at 6 / 10 / 4 of 12 pairs (s60 / s42 / s80); b*_P min +0.02467 / -0.05525 / +0.24132.
- At the final pair: PHASE eff slope +0.01856 / -0.00178 / +0.00722, OFF +0.01232 / +0.04726 / +0.04635 (s60 / s42 / s80);
  PHASE G -3.74973 / -1.84827 / -1.71299, OFF G +0.46514 / +0.48255 / +0.27256.

## 2. V - G by episode-step bucket at the final pair (12,001,280)
Mean over 32 episodes of each episode's bucket mean, critic units; SE in brackets [measured: `L/out/vg_<RUN>/vg_sigma.json`
via `L/logs/tables_full.txt`]. PHASE in the phase form (the training feed), OFF plain (stock critic).
| run | [0,50) | [50,150) | [150,200) | [200,250) | [250,1000] | status | loop-vs-batched max abs |
|---|---:|---:|---:|---:|---:|---|---:|
| PHASE s60 | -0.0116 (0.0085) | -0.0012 (0.0038) | -0.0144 (0.0017) | -0.0089 (0.0016) | -0.0044 (0.0013) | measured  | 1.91e-06 |
| PHASE s42 | -0.0028 (0.0049) | -0.0048 (0.0038) | -0.0225 (0.0025) | -0.0146 (0.0019) | -0.0029 (0.0009) | measured  | 1.43e-06 |
| PHASE s80 | +0.0095 (0.0062) | +0.0042 (0.0039) | -0.0104 (0.0019) | -0.0079 (0.0018) | +0.0009 (0.0011) | measured  | 1.43e-06 |
| OFF s60 | +0.0498 (0.0109) | +0.0644 (0.0182) | +0.0183 (0.0161) | -0.0709 (0.0099) | -0.0144 (0.0019) | measured  | 1.43e-06 |
| OFF s42 | +0.0089 (0.0099) | +0.0712 (0.0160) | +0.0068 (0.0114) | -0.1017 (0.0084) | -0.0036 (0.0014) | measured  | 1.43e-06 |
| OFF s80 | +0.0190 (0.0097) | +0.0854 (0.0154) | +0.0130 (0.0124) | -0.0944 (0.0083) | -0.0193 (0.0030) | measured  | 1.91e-06 |
| PHASE median of 3 | -0.0028 | -0.0012 | -0.0144 | -0.0089 | -0.0029 | | |
| OFF median of 3 | +0.0190 | +0.0712 | +0.0130 | -0.0944 | -0.0144 | | |

mean |V - G| per bucket:
| PHASE s60 | 0.0471 | 0.0192 | 0.0173 | 0.0118 | 0.0088 |
| PHASE s42 | 0.0330 | 0.0165 | 0.0239 | 0.0147 | 0.0081 |
| PHASE s80 | 0.0396 | 0.0195 | 0.0160 | 0.0097 | 0.0070 |
| OFF s60 | 0.0744 | 0.0999 | 0.0850 | 0.0730 | 0.0179 |
| OFF s42 | 0.0551 | 0.1025 | 0.0728 | 0.1018 | 0.0202 |
| OFF s80 | 0.0606 | 0.1064 | 0.0751 | 0.0950 | 0.0257 |
| PHASE median | 0.0396 | 0.0192 | 0.0173 | 0.0118 | 0.0081 |
| OFF median | 0.0606 | 0.1025 | 0.0751 | 0.0950 | 0.0202 |

PHASE raw (phase columns zero; report only) V - G:
| PHASE s60 raw | -0.0235 (0.0089) | -0.0828 (0.0054) | -0.2898 (0.0035) | -0.4490 (0.0036) | -0.4458 (0.0026) |
| PHASE s42 raw | -0.0158 (0.0048) | -0.0961 (0.0054) | -0.3196 (0.0052) | -0.4881 (0.0052) | -0.4847 (0.0030) |
| PHASE s80 raw | -0.0034 (0.0064) | -0.0747 (0.0052) | -0.2734 (0.0038) | -0.4369 (0.0030) | -0.4306 (0.0021) |

per-run: num_timesteps, episodes (n, truncated, terminated), sigma_r, tail_3 sigma, mean V / mean G per bucket
PHASE s60: 12001280 ep 32 trunc 32 term 0 sigma_r 40.9711 tail3 sigma 0.10290; V/G 5.033/5.045 5.168/5.169 5.372/5.386 5.540/5.549 5.553/5.558
PHASE s42: 12001280 ep 32 trunc 32 term 0 sigma_r 40.8356 tail3 sigma 0.09625; V/G 5.067/5.069 5.189/5.194 5.383/5.406 5.554/5.569 5.575/5.578
PHASE s80: 12001280 ep 32 trunc 32 term 0 sigma_r 40.8054 tail3 sigma 0.09958; V/G 5.076/5.066 5.195/5.190 5.390/5.401 5.554/5.562 5.572/5.571
OFF s60: 12001280 ep 32 trunc 32 term 0 sigma_r 39.6529 tail3 sigma 0.11379; V/G 5.186/5.136 5.301/5.237 5.463/5.444 5.537/5.608 5.607/5.622
OFF s42: 12001280 ep 32 trunc 32 term 0 sigma_r 38.9865 tail3 sigma 0.09499; V/G 5.213/5.205 5.373/5.301 5.515/5.508 5.567/5.669 5.659/5.663
OFF s80: 12001280 ep 32 trunc 32 term 0 sigma_r 39.6185 tail3 sigma 0.12314; V/G 5.168/5.149 5.334/5.249 5.460/5.447 5.510/5.604 5.579/5.598

vg-summary [measured: `L/out/vg_summary/vg_summary.json`; E1/E2 strings from `vg_sigma.expectations` on it, `L/logs/aggregates.txt`]:
         [0,50): OFF median +0.0190 (n 3) |.| 0.0606; PHASE median -0.0028 (n 3) |.| 0.0396; OFF-PHASE +0.0218 / |.| +0.0211; paired s42 +0.0117/+0.0220, s60 +0.0614/+0.0273, s80 +0.0096/+0.0211
       [50,150): OFF median +0.0712 (n 3) |.| 0.1025; PHASE median -0.0012 (n 3) |.| 0.0192; OFF-PHASE +0.0723 / |.| +0.0834; paired s42 +0.0759/+0.0860, s60 +0.0656/+0.0807, s80 +0.0812/+0.0870
      [150,200): OFF median +0.0130 (n 3) |.| 0.0751; PHASE median -0.0144 (n 3) |.| 0.0173; OFF-PHASE +0.0274 / |.| +0.0578; paired s42 +0.0293/+0.0489, s60 +0.0327/+0.0677, s80 +0.0234/+0.0591
      [200,250): OFF median -0.0944 (n 3) |.| 0.0950; PHASE median -0.0089 (n 3) |.| 0.0118; OFF-PHASE -0.0855 / |.| +0.0831; paired s42 -0.0871/+0.0871, s60 -0.0620/+0.0612, s80 -0.0865/+0.0853
     [250,1000]: OFF median -0.0144 (n 3) |.| 0.0202; PHASE median -0.0029 (n 3) |.| 0.0081; OFF-PHASE -0.0115 / |.| +0.0121; paired s42 -0.0007/+0.0121, s60 -0.0099/+0.0091, s80 -0.0202/+0.0186
    expectations (vg_sigma.expectations, descriptive): {"E1 [150,200)": "not consistent", "E1 [200,250)": "consistent", "E2 control sign pattern": "consistent"}

Label issue (known): for a PC critic the V-G field `loop_values_vs_raw_max_abs` compares the loop's phase-fed forward
values with the batched PHASE-FED values (port change V2), not with the raw (phase-zero) values its name suggests. The
values 1.43e-6 to 1.91e-6 above are therefore "loop vs batched phase-fed" for PHASE and "loop vs batched stock" for OFF.
The "PHASE raw" rows are the phase-zero critic (report only, out-of-distribution input).

## 3. What it says about the hidden-clock explanation (all interpretive sentences [inferred])
- [inferred: OFF V-G table, E2 "consistent"] The phase-blind (OFF) critic still carries the clock-shaped error at 12M in all
  three seeds: it overvalues [50,150) (+0.064..+0.085, 3.5-5.5 SE) and undervalues [200,250) (-0.071..-0.102, 7-12 SE),
  the pattern the why-probe and PS described at 2.10M. Its [150,200) sign is positive but within 0.6-1.1 SE of 0 in each run.
- [inferred: PHASE V-G table] With the phase fed, the critic's error across the step-200 boundary is 4-8 times smaller in
  mean |V - G| ([50,150) 0.019 vs 0.103, [150,200) 0.017 vs 0.075, [200,250) 0.012 vs 0.095, arm medians) and the
  OFF sign pattern is gone; this is what the hidden-clock explanation predicts if the clock is what the critic lacked.
- [inferred: same] The phase critic is not error-free: it sits slightly but clearly below G in [150,200) and [200,250)
  in all three runs (-0.008..-0.023, 4-9 SE). E1 for [150,200) reads "not consistent" as worded (|PHASE median| 0.0144 >
  |OFF median| 0.0130) because OFF's signed median there is a near-zero average of large errors; on |V - G| the PHASE
  arm is 3-5 times lower in that bucket in every seed (paired OFF-minus-PHASE |.| +0.049..+0.068).
- [inferred: PHASE raw rows] Zeroing the phase columns drops PHASE V by 0.27-0.49 after step 150: the trained critic uses
  the phase input heavily for its late-episode values. This is report only (an input never seen in training).
- [inferred: section 1] The PHASE critic's on-policy tail_3 slope stays near zero at every pair in s42 and s80 (median |.|
  0.004-0.006 vs OFF 0.025-0.037); in s60 it is not smaller than OFF's (0.0145 vs 0.0125), and its sign wanders
  (-0.022..+0.019). So "the phase removes the critic's tail slope" holds for 2 of 3 seeds on this measure, not 3.
- [inferred: section 1] The actor's tail loop never turns positive in the PHASE runs (G < 0 at all 36 pairs, b* >= 1.53),
  while OFF reaches G > 0 (the self-reinforcing tail loop) at 20 of 36 pairs, margin near or below 0 in s42 and s60.
  This fits the explanation's chain (clock error -> critic rewards tail -> positive actor loop -> rail) being cut at its
  first link, but it is also what any trajectory that did not form the habit would show.
- [inferred: design, NEXT_STEPS/ALL_EXPERIMENTS_SUMMARY Link 4] Limits: 3 seeds; same-seed PHASE/OFF runs diverge from step 0,
  so the differences are between whole trajectories, not a controlled phase-only change; no sham-input arm separates the
  phase information from "any extra critic input". These readouts are consistent with the hidden-clock explanation and
  do not confirm it. The pinned reading is unchanged.

## 4. Files
- Loop rows: `L/out/loop_stance7_r10LTSB{pc,off}_s{60,42,80}/loop.jsonl` (+ loop.txt; OFF s80's loop.txt was written at the end of the resumed run). Logs: `L/logs/loop_*.log`,
  `L/logs/loop_stance7_r10LTSBoff_s80.attempt1.log` (first attempt's log, copied before the resume overwrote it).
- V - G: `L/out/vg_<RUN>/{vg_sigma.json,vg_sigma.txt,run_info.json}`; summary `L/out/vg_summary/vg_summary.json`, log `L/logs/vg_summary.log`.
- Tables / checks: `L/logs/tables_full.txt`, `L/logs/aggregates.txt`, `L/logs/validate.txt`, `L/logs/phase_checks.txt`,
  `L/logs/final_check.txt`, `L/logs/cmp_*.txt`. Records: `L/rec/ew_series.csv`, `L/rec/eb_realized.json`, `L/rec/ref.txt`.
