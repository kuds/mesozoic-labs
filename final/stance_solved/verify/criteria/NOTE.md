# Verification of the "criteria audit" note: is the T. rex stance done?

**Bottom line**
1. The audit's numbers hold. Every M3, SCORE, CRIT, GQP, BRP and MAIN figure I opened matches its source. It has one misattributed limit and a few loose citations. Its central split is right: the screen measurement passed in 3 of 3 runs, and certification on main stands at 0 of 2.
2. It does not say how low the measured bar is. The zero-action statue passes the same six-heading certificate, 40/40 at every heading. For a heading-free policy the six headings repeat one panel. "Meets our criteria" therefore means "no hack, no habit, steady", not "certified" and not "better than standing still". The PHASE finals out-earn the statue by 8.9-17.2 on 4470.4.
3. It misses the maintainer's own walking trigger, D-D45: "a new pilot is designed once r10 has a stance parent". No parent is named only because M4 was not run in F2. That is CPU work, not more training.
4. It treats `certification_seeds` as a walking gate. On main it is a publication label that "bites only at publication". What blocks walking are GQP's §12 note on GQ-17 and GQ-3 (a). Both are open plan items, and the plan has excepted them once before (D-D36) for a weaker parent.
5. It lists the one test's NOT SEPARATED as a stance criterion. It is an attribution test, and it failed on blindness (the OFF arm entered as missing), not on the numbers.

Labels: [M: source] measured or read; [I: basis] inferred; [R: source] measured as recorded there. Short names as in ST/final/NEXT_STEPS.md. M3R = ST/round/panels/stance7_r10LTSBpc_<seed>/final/m3/stance_heading_report.txt. ROUND = S2/r10round/final_lean/criteria_r10round_final.txt (CRIT l.41). V = this folder.

## Checked and confirmed

- **M3 at the three finals.** 40/40 seeds clean at every heading, LCB95 0.9278, v2 PASS at all six headings, "GATE: PASS". Every worst value the audit quotes (width 0.0611 / 0.05185 / 0.04482; CoP 0.1586 / 0.3337 / 0.217; tilt, corner lift, displacement, yaw, load share) matches [M: M3R lines 10, 14-20, 37-59, 72, all three seeds].
- **Classes and reading.** TAIL CLEAR, GATE PASS, POSITIVE and HABIT ELSEWHERE FREE in all three [M: SCORE 321, 325, 329]. Line 4 holds with r 0, p 0, g 3, hs 0 [M: SCORE 357, 360]. The parent rule prints "not named: an input is absent" [M: SCORE 361].
- **Digests.** The gate is sha256:d26c34e6…, equal to MAIN's stance `gate_sha256` [M: M3R 8; MAIN digest_snapshot 383]. The task 1e5964b6… and PPO hyperparameters c64211e8… differ from MAIN's 533fff20… and 9d1598a6… [M: SCORE 74-75, 120-121, 166-167; digest_snapshot 382, 384].
- **Curriculum block.** BR's `[curriculum]` equals MAIN's except eval_freq, eval_episodes, checkpoint_freq, max_checkpoints and collapse_peak_floor_reference [M: V/curr_diff.out, my own TOML parse].
- **K1-K4 bars** [M: S1/foot/stance5/replication/criteria.txt 53-56]. K1-K4 are report only, "if the maintainer asks" [M: CRIT 331]. M4 at the finals was F2 work once line 4 read [M: CRIT 269-272], and it was not run [M: RUNLOG 967; every panel folder holds only c8 and m1, plus m3 at final].
- **In-run K3 panel.** At 6,291,456: 0/20 joint and 0/20 own in all three. At 12,001,280: joint 1 / 0 / 0, own 0 / 0 / 0 [M: V/k3_check.out, my own recount].
- **History and rules.** STANCE6 s80 failed K1 at 6.3M (5/40), 40/40 at its final [M: REVIEW6 400-401]; the parent rule ignores K1-K4 results [R: NEXT_STEPS 201-204].
- **Plan quotes,** all verbatim: BRP 1311, 1320, 1334; GQP 27, 185, 700, 772, 912-913, 1090; MAIN stance.toml 954, stages.toml 34, locomotion.toml 88 [M].
- **DEV and limits.** DEV LOWER, p = 1/20 [M: DEV 393]; "no rate" from three runs [M: CRIT 492]; head posture "no gate change" [M: RUNLOG 607 (2)].

## Corrections

- **Misattributed limit.** The audit writes "the gate reads the final checkpoint only, so a rail between evaluations or after 12M steps is not seen". CRIT 515 gives these as two separate limits: "RAIL reads the ledger and the end; a rail after 12M or between evaluations is not seen; the gate reads the final only" [M].
- **Loose citation.** CRIT 197-201 defines the classes, not the 0.80 bound. The bound is in MAIN stance.toml 689 and ROUND l.126-128 [M].
- **"Cross-seed test of record."** It is THE ONE TEST, PHASE against OFF. It read NOT SEPARATED because all three OFF runs were ruled not blind (first opening at 17:44 UTC) and entered as missing [M: SCORE 362; DEV 399]. The numbers separate completely: PHASE TD 0.0079 / 0.0173 / 0.0122 against OFF 0.388 / 0.635 / 0.154 [M: DEV 393]. The OFF runs are bitwise copies of STANCE6 [M: RUNLOG 970-971]. The test asks whether the phase critic caused the change. It is not a bar the stance must meet, so it should not sit in a "not met" list of stance criteria [I].
- **"Nothing at 6,291,456 or the other J checkpoints" (K1).** This understates the record. M1's deterministic replays (seeds 3042-3061) at all six labels, 4.19M to final, show 20/20 full-horizon episodes, 0 fall flags and 0 episodes with any actuator over the bar, in all three runs [M: V/m1_fh.out, read from m1_summary.json]. That is not K1 (not a v2 panel, and only 20 seeds). It does argue against a STANCE6-s80-like collapse at J, but weakly: STANCE6 s80's 6.3M failure might still have shown full horizons [I].

## Missing

1. **The statue passes the measured bar.** On the round's task the zero-action statue scores "40/40" on the panel, and its "six-heading statue certificate [is] 40/40 at every heading" [R: ROUND l.506-507, 555-556]. On main's task it is "clean on 40/40 at every heading, grouped bound 0.928" [M: MAIN stance.toml 679-681]. The PHASE LCB 0.9278 is the same figure. Policy reward is 4487.6 / 4487.4 / 4479.3 against the statue's 4470.4 [M: M3R 15-20, 61], a gain of +17.2 / +17.0 / +8.9, or 0.2-0.4% [I: my arithmetic]. The gate rules out hacks, hops and habits. It does not show a stance better than a do-nothing pose.
2. **The six headings are one panel.** The observation is "heading-free since policy interface r14" [M: MAIN stance.toml 527]. A heading-free stance "faces the same task at each" heading [M: stance.toml 670-671]. The per-heading rewards are identical in every run [M: M3R 15-20]. "Six headings × 40 seeds" is 40 seeds of evidence per run, and M3R line 11 itself says the pooled 240 is "NOT the statistic".
3. **D-D45 is the walking trigger.** "A new pilot is designed once r10 has a stance parent" [M: BRP 1328; GQP 62-63; RUNLOG 607 (3)]. It is not met only because M4 is missing, which is an F2 omission [R: NEXT_STEPS 194-200]. Running M4 names a parent whatever K1-K4 read [R: NEXT_STEPS 203-204].
4. **`certification_seeds` is a publication bar.** "Enforcement bites only at publication; training and checkpoint advancement are unchanged" [M: BRP 617-619]. MAIN's own comment reads "Publication bar, not a gate threshold" [M: stance.toml 954]. The two-seed-before-walking rule comes only from GQP §12, and the plan aims for three seeds ("plan three") [M: KNOWN_ISSUES 141; GQP 194].
5. **The precedent for an exception.** D-D36's pilot parent was "40/40 clean on the `stance_quality/v2` panel at one heading and has no gate verdict" [M: GQP 914-916]. The PHASE finals carry more evidence than that.
6. **Quality signals outside the written criteria.** PHASE s60 ended 9 of 910 training-evaluation episodes early, the last at 11,665,408 steps; the other five runs ended 0-1 [M: W/short_eps.out; SCORE 324]. PHASE s80's tail_2 command rose to +0.055 [M: SCORE 332]. Neither is a criterion, so neither fails, but both cut against calling the model "quality" untested.
7. **Practical blockers for walking.** The phase-critic code is not on main, and as built it would refuse a warm start from a PHASE checkpoint into walking [R as inferred: NEXT_STEPS 122-124].

## Answer to the maintainer's framing

- **True:** across three seeds the PHASE finals meet every written screen criterion that was measured (tail, habit, the gate certificate, all v2 bars), at the final checkpoint.
- **Not true or not known:** "certified" (0 of 2 on main's task and code; these runs are another task); K1-K4 (unmeasured, including mid-run); the 7042-7081 re-judge; a named parent. And "quality" relative to doing nothing: the measured bar is one the statue also clears.
- **Moving on.** On the maintainer's own D-D45 terms, the next step is CPU-only: run M4, name a parent, then design a pilot. Walking before certification needs a declared exception to GQ-3 (a) and GQ-17, as D-D36 was [I: GQP 27, 906-913].

Scripts (read only over existing outputs): V/curr_diff.py, V/k3_check.py, V/m1_fh.py, V/statue_check.py, with their .out files. The TOML and doc copies used for the diff were removed after use.
