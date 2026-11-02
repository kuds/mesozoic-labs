# Criteria audit: when is the T. rex stance "done"?

**Bottom line**
1. Every written per-run criterion that was **measured** at the three PHASE finals is met in 3 of 3: tail CLEAR, no habit elsewhere, the six-heading gate certificate at 40/40 seeds clean at every heading, and every stance-quality bar at 0/40 failures. That is a screen measurement, labelled "not certification".
2. The plan's "done" is certification on main: 2 passing seeds (`certification_seeds = 2`) on main's committed task before any T. rex walk (GQ-17). There are none. These runs trained another task, and "screen runs certify nothing".
3. Not measured: the M4 judge K1-K4 (at 6,291,456 and every J checkpoint, not only the final), the 7042-7081 fresh-seed re-judge, and the parent rule ("not named").
4. The cross-seed test of record failed formally (NOT SEPARATED, p = 20/20). GQ-3 (a) also holds walker sessions on today's walk reward, whatever the stance shows.

Labels: [M: source] measured; [I: basis] inferred. Short names as in ST/final/NEXT_STEPS.md. M3R = ST/round/panels/stance7_r10LTSBpc_<seed>/final/m3/stance_heading_report.txt. OUT = this folder.

## (a) The screen gate certificate the round ran (M3)

| Criterion (source) | s60 | s42 | s80 | Evidence |
|---|---|---|---|---|
| GATE CLASS PASS: `stance_quality_heading/v1` on stage1_final, six headings, seeds 3042-3081, bound on seeds clean at every heading ≥ 0.80 (CRIT 197-201; ROUND l.126-128, 280-283; MAIN stance.toml 531, 685, 689) | MET | MET | MET | 40/40, LCB95 0.9278, v2 PASS at each heading, "GATE: PASS" [M: M3R 10, 14-20, 72] |
| TAIL CLASS CLEAR, HABIT ELSEWHERE FREE, RUN OUTCOME POSITIVE (CRIT 197-209) | MET | MET | MET | [M: SCORE 321, 325, 329] |
| Reading line 4: r 0, p 0, g 3, hs 0 (CRIT 264-272) | MET (arm level) | | | [M: SCORE 357] |

- **Main's gate.** The gate digest sha256:d26c34e6… equals MAIN's stance `gate_sha256` [M: M3R 8; MAIN configs/digest_snapshot.generated.txt 383]. RB's `[curriculum]` bars equal MAIN's; only the cadence keys and the collapse reference differ [M: OUT/gate_block_rb_vs_main.diff].
- **Still a screen.** The label reads "final checkpoint, measured on a screen task; not certification". The gate reads the final checkpoint only, so a rail between evaluations or after 12M steps is not seen [M: CRIT 265-266, 282, 515].

## (b) M4 judge K1-K4 (not run at the PHASE finals)

Bars [M: S1/foot/stance5/replication/criteria.txt 53-57, cited by ROUND l.129-133]:
- **K1:** v2 PASS at 6,291,456 and at the final; across J at most one checkpoint below 37/40, and none below 33/40.
- **K2:** at every J, K2a and K2b each at most 3/40, and K2c (the panel median within [−60, +30] mm of the statue's; its range over J at most 30 mm).
- **K3:** at 6,291,456 and the final, joint-space noise falls at most 2/20 and own-sigma falls at most 1/20.
- **K4:** at the same two checkpoints, at most 3/40 episodes over the tilt or corner-lift rail.

In this round K1-K4 are report only, "if the maintainer asks" (CRIT 331). M4 at the finals was owed F2 work once line 4 read, to feed the parent rule (CRIT 269-272). It was not run [M: RUNLOG 967; the panels hold only c8, m1 and m3].

| | All three | Partial evidence (none of it is the record) |
|---|---|---|
| K1 | NOT MEASURED | the final's 0° cell reads v2 PASS 40/40 [M: M3R 14-20]; nothing at 6,291,456 or the other J checkpoints |
| K2 | NOT MEASURED | final, 0° cell: sole CoP fore-aft worst 0.1586 / 0.3337 / 0.217 against 0.8 [M: M3R 55]; nothing for K2b, K2c or J |
| K3 | NOT MEASURED | in-run panel, report only ("the offline judge's K3 stays the K3 of record"): at 6,291,456, joint 0/20 and own 0/20 in all three; at 12,001,280, joint 1 / 0 / 0 and own 0 / 0 / 0 of 20 [M: OUT/k3_gv.out] |
| K4 | NOT MEASURED | final, 0° cell: tilt and corner lift 0/40 [M: M3R 50-51]; nothing at 6,291,456 |

- **Final-only evidence is not enough.** STANCE6 s80 failed K1 at 6.29M steps and was 40/40 at its final [M as recorded: NEXT_STEPS 252-254].
- **Naming a parent is not passing K1-K4.** The pinned parent rule never reads a K1-K4 pass [M as recorded: NEXT_STEPS 201-204]. Today it prints "not named: an input is absent" [M: SCORE 361].

## (c) The plan's certification on MAIN

| Criterion (source) | All three PHASE finals | Evidence |
|---|---|---|
| Certificate = the official judge's verdict on a committed task. "Screen runs certify nothing": the judge refuses their task fingerprint, and the notebook certificate is "report only" (BRP 1320 D-D37, 1334 D-D51; GQP 1093) | NOT APPLICABLE YET | task 1e5964b6… vs MAIN's 533fff20…; PPO hyperparameters c64211e8… vs MAIN's 9d1598a6… [M: SCORE 74-75, 120-121, 166-167; digest_snapshot 382, 384]. The notebook's own gate_verdict reads passed = true, but on robust_best_model.zip [M: OUT/k3_gv.out] |
| `certification_seeds = 2`; below it a deliverable is "provisional". A replicate needs equal task, plant, gate and hyperparameter digests (MAIN stance.toml 954; BRP 614-619, 1128, 1135) | NOT MET: 0 of 2 | stages.toml 34: "no r10 seed yet (bar 2)" [M] |
| GQ-17 / §12: "a stance retrain (two seeds at today's `certification_seeds`) before any trex walk" (GQP 199, 700, 912-913) | NOT MET | formally "open"; D-D36 was a declared exception to it [M: GQP 27, 1090] |
| D-D27: at least two seeds, "each judged on 3042–3081 and re-judged on 7042–7081", the re-judge coming "before a trex stance is handed to recovery or locomotion" (BRP 1311; GQP 772; KNOWN_ISSUES 141) | 3042-3081: MET (screen); 7042-7081: NOT MEASURED | M3 used seeds 3042-3081 only [M: OUT/seeds_check.out] |
| GQ-12's 0.80 bar (37/40) | MET (screen) | LCB95 0.9278 [M: M3R 10] |

- **Counting screen runs.** Whether screen runs count toward `certification_seeds` once their configuration is committed byte for byte "is not ruled" [M: BRP 1320].
- **Walking side.** GQ-3 (a), "no walker session on today's rewards" (GQP 185), is still open; walking pilot 1 ran only as a declared exception (D-D36). MAIN's trex locomotion stage still uses `reward_and_length/v1` and has no gait-phase key [M: MAIN locomotion.toml 88; git grep]. The gait-r1 walk "still waits for a certified stance" [M: GQP 781, 1093]. NOT MET, and not a property of the stance.

## (d) GQP quality and realism bars for stance

GQP judges stance realism through the floor-truth gate `stance_quality/v2` (D-D23, D-D24, D-D27, D-D28) inside the certificate in (a). In each final's 0° cell every bar has 0/40 failures; worst values are given as s60 / s42 / s80 [M: OUT/m3_bars.out, from M3R 37-55]:
- **Chatter and hops: MET.** Touchdown rate ≤ 0.25 (0 / 0 / 0); settle touchdowns ≤ 2 (1 / 1 / 1); 0 settle airborne substeps; hop-or-fall rail ≤ 1 (0).
- **Foot placement and stance width: MET.** Settle width change ≤ 0.08 m (0.0611 / 0.05185 / 0.04482); foot load share ≥ 0.4 (0.4893 / 0.4931 / 0.4881); support ≥ 0.98 (1 / 1 / 1).
- **Centre of pressure and pads: MET.** Sole CoP fore-aft ≤ 0.8 (0.1586 / 0.3337 / 0.217); tilt ≤ 2° (1.115 / 1.065 / 1.247); corner lift ≤ 0.004 m (0.001824 / 0.001752 / 0.002095).
- **Slide, yaw and saturation: MET.** Displacement ≤ 0.1 m (0.01075 / 0.009342 / 0.01114); yaw change ≤ 25° (16.62 / 11.46 / 13.34); saturation ≤ 0.1 (0 / 0 / 0).
- **Head height: no written bar.** Head posture stays "report-only … no gate change" [M: RUNLOG 607 (2)]. The PHASE head tips are at 0.902-0.913 m against the statue's 0.959 m; this is a post-hoc look only, and the ruling is the maintainer's [M as recorded: NEXT_STEPS 211-215, 547].
- **K2b/K2c (whole-foot CoP, drift): NOT MEASURED.** They exist only in M4.

## Where "meets our criteria across multiple seeds" stands

**True today (3 of 3 seeds, screen level, final checkpoint):**
- The gate certificate passes: six headings, 40/40 seeds, LCB95 0.9278, on main's gate digest.
- Every GQP stance-quality bar has 0/40 failures (chatter, width, CoP, pads, slide, yaw, saturation, hop-or-fall).
- Tail CLEAR, no habit elsewhere, RUN OUTCOME POSITIVE; the reading is line 4.
- The 3042-3081 half of D-D27's judging is met, as is GQ-12's 0.80 bar.

**Not true, or not yet known:**
- Certification: 0 of 2 `certification_seeds` on main's task and code, so GQ-17 is not met. For the PHASE runs it is NOT APPLICABLE YET: their task and hyperparameter digests differ from main's.
- M4 K1-K4 at 6,291,456, at each J checkpoint and at the final: NOT MEASURED.
- The 7042-7081 re-judge: NOT MEASURED.
- A parent: "not named".
- The test of record: NOT SEPARATED (p = 20/20). The flagged deviation reads LOWER (p = 1/20) [M: DEV 393]. Three seeds measure no rate (CRIT 492).
- Walking readiness on the plan's terms: GQ-3 (a) and GQ-17 are unmet. Moving on now needs a declared exception, as D-D36 was [I: GQP 27, 906-913].

Scripts (read only over existing files): OUT/m3_bars.py, OUT/k3_gv.py and OUT/seeds_check.py, with their .out files; and OUT/gate_block_rb_vs_main.diff.
