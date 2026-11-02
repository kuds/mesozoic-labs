# VERIFY: the "criteria audit" note (ST/final/stance_solved/criteria/NOTE.md)

The verdict is **mostly CONFIRMED**. No number is wrong. One limit is misattributed, and some key context is missing. The full evidence is in NOTE.md.

**Confirmed against source.**
- M3R lines 8-72 for all three seeds, including every worst value.
- SCORE 74-75, 120-121, 166-167, 321-361 and DEV 393.
- CRIT 197-209, 264-272, 331, 492 and 515.
- stance5 replication criteria 53-56.
- GQP 27, 185, 700, 772, 781, 912-913, 1090 and 1093.
- BRP 614-619, 1128, 1135, 1311, 1320 and 1334.
- MAIN stance.toml 531, 685, 689 and 954; stages.toml 34; locomotion.toml 88; digest_snapshot 382-384; KNOWN_ISSUES 141.
- RUNLOG 607 and 967; NEXT_STEPS 201-204, 211-215 and 252-254; REVIEW6 400-401.
- My own recounts: the BR-against-MAIN curriculum diff (curr_diff.out) and the in-run K3 falls (k3_check.out).

**Wrong (minor).** The note says "the gate reads the final checkpoint only, so a rail between evaluations … is not seen". CRIT 515 attributes the unseen rail to RAIL's ledger reading, not to the gate.

**Overclaim or mislabel.** The note calls THE ONE TEST "the cross-seed test of record" and lists it as an unmet stance criterion. It is an attribution test (PHASE against OFF). Its pinned NOT SEPARATED comes from the OFF arm being ruled not blind and entered as missing (SCORE 362; DEV 399). The TDs themselves separate completely (DEV 393).

**Underclaim.**
- The note says there is "nothing at 6,291,456 or the other J checkpoints". M1 at all six labels shows 20/20 full horizon, 0 fall flags and 0 over-bar episodes in all three runs (m1_fh.out). This is partial evidence only; it is not K1.
- The note leaves out PHASE s60's 9/910 early training-evaluation endings (W/short_eps.out).

**Missing.**
1. The zero-action statue passes the same six-heading certificate, 40/40 at every heading (ROUND l.507, 555-556; MAIN stance.toml 679-681). The PHASE rewards beat it by only 8.9-17.2 of 4470.4.
2. For a heading-free policy the six headings repeat one panel (stance.toml 527, 670-671; identical per-heading rewards in M3R 15-20).
3. D-D45: "a new pilot is designed once r10 has a stance parent" (BRP 1328; GQP 62-63; RUNLOG 607 (3)). This trigger fails only because M4 is missing.
4. `certification_seeds` "bites only at publication; training and checkpoint advancement are unchanged" (BRP 617-619).
5. D-D36's precedent: walking pilot 1's parent had less evidence than the PHASE finals (GQP 914-916).
6. The plan aims for three seeds (KNOWN_ISSUES 141 "plan three"; GQP 194).

**Files in this folder.** NOTE.md, VERIFY.md, and four read-only scripts with their outputs (curr_diff, k3_check, m1_fh, statue_check). The TOML and doc copies used were removed after use.
