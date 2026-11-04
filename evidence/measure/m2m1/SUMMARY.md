# m2m1 lane: M2 posture at J and M1 tail commands at non-J labels (PHASE runs), report only

Lane folder: /tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/measure/m2m1 (L below).
Nothing here changes the pinned reading (criteria.txt line 4). Labels: [measured: file] or [inferred: basis].

## Provenance and validation

- Resume: the lane folder did not exist at resume (00:57 UTC 2026-11-03), so the first attempt left no m2m1 outputs. Every output here is from this attempt [measured: ls; L/CMDLOG.txt line 1].
- Tool: port/m2/tool/measure_r10round.py (PC pin 75cb4a9b...) on the sealed copy port/m2/copies/copy_1dfe2327 (tree 9417b765..., before and after every run) with --approved handoff/round/APPROVED. The 20 port/m2 tool and check files match the pre-measure sha256 snapshot (sha256sum -c: all OK) [measured: L/logs/m2_tool_sums_check.txt].
- Statue: a new panel at L/panels/statue/m2 has 40/40 clean and mean reward 4470.368636249949. compare_m2.py against port/m2/panels/V2_1dfe2327/statue/m2 says "EQUAL (but the listed expected fields)". The only differences are utc, argv path, seconds, panel_json_sha256, utc_end and seconds. series.npz matches in 200 of 200 arrays and panel.json in 4968 of 4970 leaves [measured: L/logs/statue_vs_V2_statue.out].
- Equality cross-check: this lane's PHASE s42 final M2 panel matches the port's V3 smoke panel. The verdict is EQUAL and 0 differences are unexpected [measured: L/logs/s42final_vs_V3smoke.out].
- M1 validation: M1 was re-run on PHASE s60 final (L/validation/m1_stance7_r10LTSBpc_s60_final/m1) and compared with F2's record handoff/round/panels/stance7_r10LTSBpc_s60/final/m1.
  - All 3975 equal leaves of m1_summary.json match, including cmd_a, every per-episode number and checkpoint sha256. csv_sha256.txt (the 20 per-episode csv sha256s), flags.csv and checkpoint.sha256 also match. Only paths, times, stack.python and trace_json_sha256 differ.
  - episode_trace.json has 5068 leaves. Only 2 differ, the checkpoint and vecnorm path strings. No numeric leaf differs, so trace_json_sha256 differs only because of paths.
  - Verdict: EQUAL in every measured field. The other M1s were therefore run [measured: L/validation/m1_s60_final_vs_F2.out, L/validation/m1_s60_final_trace_vs_F2.out].
- Every record carries the correct filing:
  - run_id and seed 60/42/80
  - the label, with checkpoint num_timesteps equal to the label (final = 12001280)
  - run_records git_commit 1dfe2327
  - the same copy tree digest before and after

  [measured: the m*_summary.json files under L/panels]
- M2 setup: 40 deterministic episodes per panel, seeds 3042-3081, scored over window steps 200..L-1. M1 setup: 20 episodes, seeds 3042-3061 (the F2 invocation). The OFF columns come from the STANCE6 records on origin/main (stance6_results/panels/stance6_r10LTSB_s*/<label>/m2/m2_summary.json), copied by git show into L/ref/. They are labelled OFF/S6 "as recorded".

## Headline [measured: L/head_figures.json, L/tables.md, L/ref/*]

- PHASE M2, all 18 panels:
  - Head-tip height: 0.9021 to 0.9406 m. That is 0.0021 to 0.0406 m above the reward's full-credit bar of 0.90 m and 0.018 to 0.0565 m below the statue's 0.9586 m.
  - Mean tail3 command: -0.008 to 0.027.
  - Full-horizon/clean: 719/720 and 719/720. The one failure is s60 4194304, seed 3072 (episode 30), which fell backward at step 221.
- OFF/S6 M2 at the same seed and labels, as recorded:
  - Head-tip height: 0.901 to 1.3457 m.
  - Mean tail3 command: -0.028 to 0.98.
  - Full horizon 720/720 but only 553/720 clean. The not-clean episodes are full-horizon episodes that fail gate rails.
- PHASE neck pitch is 13.5 to 16.7 deg against the statue's 12.27 deg, and head pitch is 12.9 to 17.8 deg against 12.64 deg. Both are nose-down a little more than the statue. OFF/S6 neck pitch ranges from -17.4 to 21.5 deg, and is mostly below the statue's.
- PHASE M1 at non-J labels: every tail cmd_a has |value| <= 0.0682. The largest is s42 1048576, tail_3_pitch -0.0682. Every tail saturation share (>=0.99 and >=0.9) is 0.0000, and there are no tail flags or fall flags (20/20 full horizon). F2's J records for the same runs look the same: |cmd_a| <= 0.0658 and saturation 0.

## Interpretation [inferred]

- At every J label and every non-J grid label of all three seeds, the PHASE policies hold the tail near zero command with no saturation. Their head height sits just above the 0.90 m bar, below the statue's head, with neck and head pitched slightly nose-down of the statue.
- The OFF runs carry tail3 commands up to about 0.98, and at most labels raise the head well above both the statue and the bar, by lifting the neck (negative neck pitch). This posture is not in any PHASE panel.
- This is consistent with the pinned reading (the phase critic removes the habit). It adds posture detail and does not change that reading.
- The PHASE head-tip height settles near the bar late in training (final: 0.902-0.904 m for all three seeds). The reward's clearance term is flat above 0.90 m, so no head-height incentive pushes it higher. Across training the head drifts down toward the bar, from 0.91-0.94 m at 4M steps (basis: head_clearance quality 0.9975-1.0 at every panel; reward_functions smoothstep).
- The single s60 4194304 backward fall is one episode in 40 at an early J label. It is reported as a fail.
- PHASE s60's evaluation falls were not examined in this lane. The fall above is the only fall that M2 or M1 saw for s60.

Statue (this lane's panel): head_tip 0.9586415027890884 m, neck 12.265124474616762 deg, head 12.642758347904218 deg, n_clean 40/40, mean_reward 4470.368636249949

### M2 head figures, PHASE (this lane) vs OFF (= STANCE6 record, same seed and label)

| seed | label | arm | full/clean/n | fall_dirs | mean_reward | head_tip m | -0.90 | -statue | neck deg | -statue | head deg | -statue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s60 | 4194304 | PHASE | 39/39/40 | {"backward": 1, "none": 39} | 4391.5 | 0.9352 | 0.0352 | -0.0234 | 13.55 | 1.28 | 14.74 | 2.09 |
| s60 | 4194304 | OFF/S6 | 40/40/40 | {"none": 40} | 4447.6 | 1.0614 | 0.1614 | 0.1028 | 5.39 | -6.88 | 10.33 | -2.32 |
| s60 | 5242880 | PHASE | 40/40/40 | {"none": 40} | 4484.4 | 0.9406 | 0.0406 | -0.0180 | 13.48 | 1.21 | 14.19 | 1.55 |
| s60 | 5242880 | OFF/S6 | 40/39/40 | {"none": 40} | 4443.0 | 1.0210 | 0.1210 | 0.0624 | 8.78 | -3.49 | 9.26 | -3.38 |
| s60 | 6291456 | PHASE | 40/40/40 | {"none": 40} | 4486.6 | 0.9110 | 0.0110 | -0.0476 | 15.07 | 2.80 | 15.10 | 2.46 |
| s60 | 6291456 | OFF/S6 | 40/40/40 | {"none": 40} | 4435.1 | 1.0578 | 0.1578 | 0.0991 | 6.06 | -6.21 | 9.80 | -2.85 |
| s60 | 8388608 | PHASE | 40/40/40 | {"none": 40} | 4490.9 | 0.9285 | 0.0285 | -0.0301 | 14.39 | 2.13 | 14.05 | 1.41 |
| s60 | 8388608 | OFF/S6 | 40/34/40 | {"none": 40} | 4376.2 | 1.0870 | 0.1870 | 0.1284 | 5.14 | -7.12 | 6.72 | -5.93 |
| s60 | 10485760 | PHASE | 40/40/40 | {"none": 40} | 4484.9 | 0.9155 | 0.0155 | -0.0432 | 15.05 | 2.78 | 14.86 | 2.22 |
| s60 | 10485760 | OFF/S6 | 40/38/40 | {"none": 40} | 4383.6 | 1.0742 | 0.1742 | 0.1155 | 5.95 | -6.32 | 6.75 | -5.89 |
| s60 | final | PHASE | 40/40/40 | {"none": 40} | 4487.6 | 0.9023 | 0.0023 | -0.0563 | 15.94 | 3.67 | 14.74 | 2.10 |
| s60 | final | OFF/S6 | 40/40/40 | {"none": 40} | 4396.0 | 1.1153 | 0.2153 | 0.1567 | 1.54 | -10.72 | 8.70 | -3.94 |
| s42 | 4194304 | PHASE | 40/40/40 | {"none": 40} | 4484.2 | 0.9132 | 0.0132 | -0.0454 | 16.05 | 3.78 | 12.93 | 0.28 |
| s42 | 4194304 | OFF/S6 | 40/3/40 | {"none": 40} | 4349.1 | 0.9239 | 0.0239 | -0.0348 | 14.81 | 2.54 | 10.70 | -1.94 |
| s42 | 5242880 | PHASE | 40/40/40 | {"none": 40} | 4481.6 | 0.9022 | 0.0022 | -0.0564 | 16.65 | 4.39 | 13.45 | 0.80 |
| s42 | 5242880 | OFF/S6 | 40/30/40 | {"none": 40} | 4330.5 | 1.0790 | 0.1790 | 0.1204 | 6.01 | -6.25 | 7.66 | -4.99 |
| s42 | 6291456 | PHASE | 40/40/40 | {"none": 40} | 4481.8 | 0.9076 | 0.0076 | -0.0511 | 16.72 | 4.46 | 12.89 | 0.24 |
| s42 | 6291456 | OFF/S6 | 40/39/40 | {"none": 40} | 4412.4 | 0.9010 | 0.0010 | -0.0577 | 17.16 | 4.90 | 11.16 | -1.48 |
| s42 | 8388608 | PHASE | 40/40/40 | {"none": 40} | 4484.7 | 0.9080 | 0.0080 | -0.0507 | 16.56 | 4.29 | 13.23 | 0.59 |
| s42 | 8388608 | OFF/S6 | 40/4/40 | {"none": 40} | 4338.3 | 0.9294 | 0.0294 | -0.0293 | 21.47 | 9.20 | 0.32 | -12.32 |
| s42 | 10485760 | PHASE | 40/40/40 | {"none": 40} | 4486.3 | 0.9038 | 0.0038 | -0.0549 | 16.54 | 4.28 | 13.94 | 1.30 |
| s42 | 10485760 | OFF/S6 | 40/7/40 | {"none": 40} | 4351.4 | 1.0236 | 0.1236 | 0.0650 | 10.04 | -2.23 | 5.42 | -7.23 |
| s42 | final | PHASE | 40/40/40 | {"none": 40} | 4487.4 | 0.9035 | 0.0035 | -0.0552 | 16.21 | 3.95 | 14.65 | 2.01 |
| s42 | final | OFF/S6 | 40/38/40 | {"none": 40} | 4353.1 | 1.1552 | 0.2552 | 0.1966 | -2.55 | -14.82 | 9.15 | -3.50 |
| s80 | 4194304 | PHASE | 40/40/40 | {"none": 40} | 4488.4 | 0.9386 | 0.0386 | -0.0201 | 13.92 | 1.65 | 14.01 | 1.37 |
| s80 | 4194304 | OFF/S6 | 40/39/40 | {"none": 40} | 4416.3 | 1.2683 | 0.3683 | 0.3097 | -6.51 | -18.78 | 3.03 | -9.61 |
| s80 | 5242880 | PHASE | 40/40/40 | {"none": 40} | 4486.1 | 0.9356 | 0.0356 | -0.0230 | 13.87 | 1.61 | 14.30 | 1.65 |
| s80 | 5242880 | OFF/S6 | 40/37/40 | {"none": 40} | 4418.0 | 1.3079 | 0.4079 | 0.3493 | -12.56 | -24.83 | 6.73 | -5.91 |
| s80 | 6291456 | PHASE | 40/40/40 | {"none": 40} | 4489.1 | 0.9249 | 0.0249 | -0.0337 | 14.47 | 2.20 | 14.61 | 1.96 |
| s80 | 6291456 | OFF/S6 | 40/5/40 | {"none": 40} | 4377.4 | 1.3457 | 0.4457 | 0.3871 | -17.38 | -29.64 | 10.23 | -2.41 |
| s80 | 8388608 | PHASE | 40/40/40 | {"none": 40} | 4487.7 | 0.9196 | 0.0196 | -0.0391 | 14.78 | 2.51 | 15.01 | 2.37 |
| s80 | 8388608 | OFF/S6 | 40/40/40 | {"none": 40} | 4426.4 | 1.0139 | 0.1139 | 0.0553 | 11.46 | -0.80 | 9.60 | -3.04 |
| s80 | 10485760 | PHASE | 40/40/40 | {"none": 40} | 4480.7 | 0.9024 | 0.0024 | -0.0562 | 15.56 | 3.30 | 16.25 | 3.61 |
| s80 | 10485760 | OFF/S6 | 40/40/40 | {"none": 40} | 4380.6 | 1.2597 | 0.3597 | 0.3010 | -0.90 | -13.16 | -3.56 | -16.20 |
| s80 | final | PHASE | 40/40/40 | {"none": 40} | 4479.3 | 0.9021 | 0.0021 | -0.0565 | 14.74 | 2.47 | 17.75 | 5.11 |
| s80 | final | OFF/S6 | 40/40/40 | {"none": 40} | 4386.9 | 1.1663 | 0.2663 | 0.2076 | 6.48 | -5.78 | -3.34 | -15.98 |

### M2 other posture means (full-horizon), PHASE vs OFF/S6

| seed | label | arm | tail3_cmd_mean | tail1_cmd_mean | tail2_cmd_mean | q1_deg | q2_deg | q3_deg | chord_base_deg | chord_dist_deg | tip_z_m | com_fa_mm | com_lat_mm | pitch_deg | roll_deg | pelvis_z_m | others_max_sat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | statue | statue | 0.000 | 0.000 | 0.000 | 0.009 | 0.008 | 0.013 | 0.097 | 3.243 | 0.932 | -27.254 | -0.002 | 0.923 | 0.002 | 0.926 | 0.000 |
| s60 | 4194304 | PHASE | 0.002 | 0.003 | 0.010 | 0.034 | 0.072 | 0.025 | -0.152 | 3.014 | 0.925 | -29.490 | -0.207 | 0.593 | 0.004 | 0.926 | 0.000 |
| s60 | 4194304 | OFF/S6 | 0.073 | 0.019 | 0.089 | 0.142 | 0.544 | 0.393 | 0.655 | 4.090 | 0.944 | -33.014 | -0.183 | 0.721 | -0.005 | 0.926 | 0.000 |
| s60 | 5242880 | PHASE | 0.007 | -0.002 | 0.008 | 0.000 | 0.056 | 0.048 | -0.215 | 2.958 | 0.923 | -29.053 | -0.290 | 0.563 | 0.003 | 0.926 | 0.000 |
| s60 | 5242880 | OFF/S6 | 0.130 | 0.022 | 0.075 | 0.157 | 0.461 | 0.694 | 1.216 | 4.766 | 0.959 | -26.940 | -0.060 | 1.164 | -0.005 | 0.927 | 0.000 |
| s60 | 6291456 | PHASE | 0.005 | 0.004 | 0.000 | 0.038 | 0.011 | 0.041 | 0.007 | 3.168 | 0.929 | -26.712 | 0.007 | 0.787 | 0.003 | 0.926 | 0.000 |
| s60 | 6291456 | OFF/S6 | 0.276 | 0.038 | 0.145 | 0.273 | 0.884 | 1.443 | 1.627 | 5.604 | 0.967 | -33.699 | 0.253 | 0.715 | -0.002 | 0.926 | 0.000 |
| s60 | 8388608 | PHASE | -0.003 | 0.003 | -0.011 | 0.034 | -0.054 | -0.001 | -0.168 | 2.959 | 0.925 | -28.519 | -0.128 | 0.689 | -0.000 | 0.926 | 0.000 |
| s60 | 8388608 | OFF/S6 | 0.887 | 0.062 | 0.221 | 0.458 | 1.372 | 4.590 | 4.260 | 9.746 | 1.030 | -30.405 | -0.053 | 1.035 | 0.004 | 0.926 | 0.000 |
| s60 | 10485760 | PHASE | -0.002 | 0.022 | 0.012 | 0.160 | 0.081 | 0.004 | 0.057 | 3.217 | 0.930 | -28.512 | -0.140 | 0.681 | 0.003 | 0.926 | 0.000 |
| s60 | 10485760 | OFF/S6 | 0.764 | 0.022 | 0.203 | 0.187 | 1.258 | 3.962 | 3.689 | 8.870 | 1.017 | -28.618 | -0.345 | 1.172 | 0.002 | 0.926 | 0.000 |
| s60 | final | PHASE | -0.008 | 0.007 | -0.015 | 0.058 | -0.079 | -0.027 | -0.001 | 3.109 | 0.929 | -28.428 | -0.176 | 0.866 | 0.000 | 0.926 | 0.000 |
| s60 | final | OFF/S6 | 0.593 | -0.022 | 0.145 | -0.115 | 0.897 | 3.079 | 2.382 | 7.091 | 0.986 | -31.640 | -0.239 | 0.938 | -0.003 | 0.926 | 0.000 |
| s42 | 4194304 | PHASE | -0.006 | -0.014 | -0.026 | -0.086 | -0.145 | -0.020 | -0.138 | 2.960 | 0.926 | -27.898 | 0.053 | 0.920 | 0.002 | 0.926 | 0.000 |
| s42 | 4194304 | OFF/S6 | 0.980 | -0.167 | 0.024 | -1.065 | 0.189 | 5.067 | 2.915 | 8.349 | 1.001 | -19.271 | 0.056 | 1.869 | -0.007 | 0.926 | 0.000 |
| s42 | 5242880 | PHASE | -0.003 | -0.021 | -0.019 | -0.127 | -0.104 | -0.002 | -0.126 | 2.989 | 0.926 | -28.560 | -0.113 | 0.932 | 0.003 | 0.926 | 0.000 |
| s42 | 5242880 | OFF/S6 | 0.900 | -0.169 | 0.115 | -1.075 | 0.721 | 4.672 | 1.887 | 7.265 | 0.975 | -36.996 | -0.678 | 0.657 | 0.041 | 0.928 | 0.000 |
| s42 | 6291456 | PHASE | 0.005 | -0.020 | -0.020 | -0.119 | -0.107 | 0.041 | -0.192 | 2.942 | 0.925 | -28.956 | 0.155 | 0.836 | -0.002 | 0.926 | 0.000 |
| s42 | 6291456 | OFF/S6 | 0.358 | -0.116 | 0.025 | -0.751 | 0.169 | 1.883 | 1.119 | 5.133 | 0.958 | -19.124 | -0.152 | 1.541 | -0.002 | 0.927 | 0.000 |
| s42 | 8388608 | PHASE | 0.013 | -0.023 | -0.025 | -0.133 | -0.140 | 0.082 | -0.265 | 2.880 | 0.923 | -29.640 | -0.398 | 0.781 | 0.007 | 0.926 | 0.000 |
| s42 | 8388608 | OFF/S6 | 0.913 | -0.192 | 0.064 | -1.228 | 0.423 | 4.662 | 2.466 | 7.772 | 0.990 | -17.319 | -0.132 | 1.624 | -0.002 | 0.926 | 0.000 |
| s42 | 10485760 | PHASE | 0.018 | -0.011 | -0.031 | -0.056 | -0.173 | 0.108 | -0.238 | 2.911 | 0.923 | -28.958 | -0.041 | 0.742 | 0.003 | 0.926 | 0.000 |
| s42 | 10485760 | OFF/S6 | 0.958 | -0.052 | -0.010 | -0.318 | -0.012 | 4.962 | 3.329 | 8.673 | 1.011 | -25.323 | 0.016 | 1.750 | -0.007 | 0.926 | 0.000 |
| s42 | final | PHASE | 0.010 | -0.018 | -0.031 | -0.097 | -0.175 | 0.064 | -0.349 | 2.779 | 0.920 | -29.266 | -0.051 | 0.697 | 0.001 | 0.926 | 0.000 |
| s42 | final | OFF/S6 | 0.900 | 0.116 | 0.009 | 0.787 | 0.098 | 4.666 | 3.773 | 9.010 | 1.019 | -34.906 | -0.717 | 1.168 | 0.007 | 0.927 | 0.000 |
| s80 | 4194304 | PHASE | -0.004 | -0.004 | 0.011 | -0.005 | 0.074 | -0.007 | -0.314 | 2.838 | 0.920 | -30.329 | -0.249 | 0.487 | 0.006 | 0.926 | 0.000 |
| s80 | 4194304 | OFF/S6 | 0.012 | -0.045 | 0.059 | -0.272 | 0.365 | 0.072 | -0.397 | 2.855 | 0.918 | -34.402 | -0.569 | 0.400 | 0.018 | 0.926 | 0.000 |
| s80 | 5242880 | PHASE | -0.003 | -0.000 | 0.015 | 0.016 | 0.100 | -0.003 | -0.202 | 2.958 | 0.923 | -30.704 | -0.409 | 0.555 | 0.008 | 0.926 | 0.000 |
| s80 | 5242880 | OFF/S6 | -0.028 | -0.033 | 0.033 | -0.192 | 0.206 | -0.132 | -0.677 | 2.449 | 0.911 | -37.100 | 0.172 | 0.277 | 0.006 | 0.926 | 0.044 |
| s80 | 6291456 | PHASE | -0.002 | 0.004 | 0.015 | 0.042 | 0.101 | 0.005 | -0.091 | 3.073 | 0.926 | -30.693 | -0.073 | 0.635 | 0.001 | 0.926 | 0.000 |
| s80 | 6291456 | OFF/S6 | 0.175 | 0.019 | 0.073 | 0.165 | 0.446 | 0.900 | -0.273 | 3.366 | 0.917 | -42.010 | -0.313 | -0.437 | -0.004 | 0.925 | 0.548 |
| s80 | 8388608 | PHASE | 0.019 | 0.018 | 0.021 | 0.132 | 0.131 | 0.113 | 0.028 | 3.247 | 0.929 | -31.445 | -0.017 | 0.580 | 0.000 | 0.926 | 0.000 |
| s80 | 8388608 | OFF/S6 | 0.114 | -0.022 | -0.077 | -0.102 | -0.439 | 0.602 | -1.100 | 2.209 | 0.899 | -36.245 | -0.166 | -0.142 | -0.002 | 0.925 | 0.000 |
| s80 | 10485760 | PHASE | 0.010 | 0.004 | 0.035 | 0.040 | 0.216 | 0.064 | -0.058 | 3.158 | 0.926 | -33.203 | -0.190 | 0.548 | 0.004 | 0.926 | 0.000 |
| s80 | 10485760 | OFF/S6 | 0.416 | 0.146 | -0.151 | 0.981 | -0.884 | 2.147 | 0.907 | 4.804 | 0.948 | -36.035 | 0.068 | 0.270 | -0.010 | 0.925 | 0.000 |
| s80 | final | PHASE | 0.027 | 0.035 | 0.061 | 0.242 | 0.374 | 0.150 | 0.263 | 3.553 | 0.934 | -33.841 | -0.046 | 0.495 | 0.002 | 0.926 | 0.000 |
| s80 | final | OFF/S6 | 0.577 | 0.234 | -0.149 | 1.556 | -0.869 | 2.973 | 2.116 | 6.385 | 0.977 | -30.236 | 0.003 | 0.432 | -0.005 | 0.926 | 0.000 |

### M1 tail commands (cmd_a, window mean over full-horizon episodes) and saturation

| seed | label | source | full/n | tail_1_pitch | tail_1_yaw | tail_2_pitch | tail_3_pitch | tail max share>=0.99 (ep mean) | tail max share>=0.9 | tail_flags | fall_flags | max|cmd| any |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s60 | 1048576 | lane | 20/20 | -0.0297 | 0.0042 | 0.0072 | -0.0251 | 0.0000 | 0.0000 | 0 | 0 | 0.465 |
| s60 | 2097152 | lane | 20/20 | -0.0256 | 0.0033 | 0.0058 | -0.0133 | 0.0000 | 0.0000 | 0 | 0 | 0.705 |
| s60 | 3145728 | lane | 20/20 | -0.0091 | 0.0004 | 0.0255 | -0.0040 | 0.0000 | 0.0000 | 0 | 0 | 0.753 |
| s60 | 4194304 | F2 | 20/20 | 0.0031 | 0.0022 | 0.0113 | 0.0022 | 0.0000 | 0.0000 | 0 | 0 | 0.921 |
| s60 | 5242880 | F2 | 20/20 | -0.0011 | 0.0002 | 0.0067 | 0.0058 | 0.0000 | 0.0000 | 0 | 0 | 0.852 |
| s60 | 6291456 | F2 | 20/20 | 0.0044 | 0.0011 | 0.0023 | 0.0055 | 0.0000 | 0.0000 | 0 | 0 | 0.895 |
| s60 | 7340032 | lane | 20/20 | 0.0024 | 0.0024 | -0.0088 | 0.0006 | 0.0000 | 0.0000 | 0 | 0 | 0.923 |
| s60 | 8388608 | F2 | 20/20 | 0.0040 | -0.0002 | -0.0111 | -0.0042 | 0.0000 | 0.0000 | 0 | 0 | 0.860 |
| s60 | 9437184 | lane | 20/20 | 0.0075 | -0.0000 | 0.0003 | 0.0025 | 0.0000 | 0.0000 | 0 | 0 | 0.857 |
| s60 | 10485760 | F2 | 20/20 | 0.0220 | 0.0005 | 0.0097 | -0.0028 | 0.0000 | 0.0000 | 0 | 0 | 0.770 |
| s60 | 11534336 | lane | 20/20 | 0.0217 | -0.0016 | -0.0021 | -0.0084 | 0.0000 | 0.0000 | 0 | 0 | 0.752 |
| s60 | final | F2 | 20/20 | 0.0086 | -0.0003 | -0.0152 | -0.0092 | 0.0000 | 0.0000 | 0 | 0 | 0.682 |
| s42 | 1048576 | lane | 20/20 | 0.0150 | 0.0008 | 0.0459 | -0.0682 | 0.0000 | 0.0000 | 0 | 0 | 0.392 |
| s42 | 2097152 | lane | 20/20 | -0.0051 | 0.0047 | -0.0163 | -0.0271 | 0.0000 | 0.0000 | 0 | 0 | 0.456 |
| s42 | 3145728 | lane | 20/20 | -0.0151 | 0.0019 | -0.0468 | -0.0205 | 0.0000 | 0.0000 | 0 | 0 | 0.661 |
| s42 | 4194304 | F2 | 20/20 | -0.0150 | -0.0011 | -0.0321 | -0.0042 | 0.0000 | 0.0000 | 0 | 0 | 0.572 |
| s42 | 5242880 | F2 | 20/20 | -0.0157 | 0.0002 | -0.0253 | 0.0007 | 0.0000 | 0.0000 | 0 | 0 | 0.583 |
| s42 | 6291456 | F2 | 20/20 | -0.0207 | -0.0016 | -0.0279 | 0.0074 | 0.0000 | 0.0000 | 0 | 0 | 0.616 |
| s42 | 7340032 | lane | 20/20 | -0.0197 | 0.0013 | -0.0300 | -0.0060 | 0.0000 | 0.0000 | 0 | 0 | 0.625 |
| s42 | 8388608 | F2 | 20/20 | -0.0209 | -0.0034 | -0.0319 | 0.0157 | 0.0000 | 0.0000 | 0 | 0 | 0.642 |
| s42 | 9437184 | lane | 20/20 | -0.0167 | 0.0006 | -0.0305 | 0.0164 | 0.0000 | 0.0000 | 0 | 0 | 0.601 |
| s42 | 10485760 | F2 | 20/20 | -0.0044 | -0.0003 | -0.0337 | 0.0228 | 0.0000 | 0.0000 | 0 | 0 | 0.587 |
| s42 | 11534336 | lane | 20/20 | -0.0053 | -0.0015 | -0.0373 | 0.0253 | 0.0000 | 0.0000 | 0 | 0 | 0.572 |
| s42 | final | F2 | 20/20 | -0.0171 | -0.0018 | -0.0369 | 0.0147 | 0.0000 | 0.0000 | 0 | 0 | 0.617 |
| s80 | 1048576 | lane | 20/20 | -0.0058 | -0.0021 | -0.0209 | -0.0117 | 0.0000 | 0.0000 | 0 | 0 | 0.450 |
| s80 | 2097152 | lane | 20/20 | 0.0252 | -0.0018 | -0.0017 | -0.0067 | 0.0000 | 0.0000 | 0 | 0 | 0.680 |
| s80 | 3145728 | lane | 20/20 | 0.0157 | -0.0000 | 0.0206 | 0.0049 | 0.0000 | 0.0000 | 0 | 0 | 0.745 |
| s80 | 4194304 | F2 | 20/20 | 0.0060 | -0.0001 | 0.0104 | -0.0010 | 0.0000 | 0.0000 | 0 | 0 | 0.744 |
| s80 | 5242880 | F2 | 20/20 | 0.0093 | 0.0000 | 0.0149 | -0.0005 | 0.0000 | 0.0000 | 0 | 0 | 0.696 |
| s80 | 6291456 | F2 | 20/20 | 0.0105 | -0.0005 | 0.0130 | -0.0018 | 0.0000 | 0.0000 | 0 | 0 | 0.775 |
| s80 | 7340032 | lane | 20/20 | 0.0067 | 0.0010 | 0.0097 | 0.0030 | 0.0000 | 0.0000 | 0 | 0 | 0.706 |
| s80 | 8388608 | F2 | 20/20 | 0.0214 | -0.0007 | 0.0190 | 0.0167 | 0.0000 | 0.0000 | 0 | 0 | 0.708 |
| s80 | 9437184 | lane | 20/20 | 0.0363 | 0.0013 | 0.0251 | 0.0199 | 0.0000 | 0.0000 | 0 | 0 | 0.676 |
| s80 | 10485760 | F2 | 20/20 | 0.0095 | -0.0006 | 0.0354 | 0.0099 | 0.0000 | 0.0000 | 0 | 0 | 0.678 |
| s80 | 11534336 | lane | 20/20 | 0.0272 | -0.0022 | 0.0608 | 0.0178 | 0.0000 | 0.0000 | 0 | 0 | 0.662 |
| s80 | final | F2 | 20/20 | 0.0415 | -0.0004 | 0.0658 | 0.0279 | 0.0000 | 0.0000 | 0 | 0 | 0.649 |
