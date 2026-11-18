# Crawl development runs (18 Nov 2026): ledger

Written after the independent check asked for it, from the session's printed outputs (the per-run logs were not
saved). **Development runs, not judged results.** Seeds 100-103 only. "Base" is PR #669's foot (passive pitch ankle,
sole rigid in roll); "variant" adds a passive roll hinge at each ankle. Unless stated, the model is T20 closed and the
seed is 100. Diagnostic re-runs of an existing setting are counted, as the pre-registration's budget counts runs.

| # | Model | What changed from the previous row | Outcome |
|---|---|---|---|
| 1 | base, first build (PR #669 gains) | first crawl: planned margin 30 mm, sideways ≤ 70 mm, shift 0.5 s, swing 0.5 s, sin² lift | ran; the report crashed (a NumPy boolean in JSON), no result kept |
| 2 | same | report fixed | C1-C3 pass; C4, C5 fail; margin 11.4 mm, pin 29.5 mm, travel 1.113 m, 0.26 % saturation |
| 3 | same | per-step diagnosis | swing foot unloaded only 43-71 % of its window; a support foot at 0 % on every swing; measured margin fell to 11-15 mm on forefoot steps (planned 30) |
| 4 | same | planned margin 45 mm, sideways ≤ 90 mm, fast-lift swing (25 % lifting, 25 % lowering) | C4, C5 fail; unloaded 73-84 %; margin 13.5 mm; pin 36.7 mm; tilt 6.1° |
| 5 | same | time series of the loads | the robot balanced on a diagonal pair of feet; body drifted back toward the centreline during swings |
| 6 | same | 4 mm contact preload | no change (margin 14.6, pin 32.7 mm) |
| 7 | base, corrected models (T. rex gains) | rebuilt models | margin 29.3 mm, pin 25.3 mm, travel 1.268 m; C6 fails (hind pitch at its limit 1.8 % of samples) |
| 8 | same | time series of a hind swing | body rolled to -2.65° about the diagonal; the opposite forefoot rose 13 mm off the floor |
| 9 | same | geometry at t = 5.8 s | centre of mass 34.1 mm inside the triangle; total ground reaction 33.7 mm off the diagonal, carried by the two diagonal soles; third foot in the air |
| 10 | same | planned margin 75 mm, sideways ≤ 120 mm | fell over (tilt 107°): shifts too large and too fast |
| 11 | same | planned margin 60 mm, sideways ≤ 100 mm, shift 1.0 s | C1, C3, C6 pass; C2 fails (body ended 105 mm sideways); C4 fails (third foot 1.5-5.5 %); pin 30.3 mm; tilt 0.8° |
| 12 | same | pin motion per stance | each foot ~28 mm out and back per stance, net ~6 mm: rocking, not sliding |
| 13 | same | trace of the left fore sole | sole rolled ±15° onto its edges as the body swayed |
| 14 | same | planner prefers fore-aft shifts (sideways cost × 3, fore-aft ≤ 80 mm), planned margin 50 mm, final return to centre | C1, C2, C3, C6 pass; C4, C5 fail; pin 19.2 mm |
| 15 | same | per-step loads | third foot 1-3 % on several steps |
| 16 | same | sideways ≤ 30 mm, fore-aft ≤ 120 mm, margin 35 mm | C2, C4, C5 fail; travel 0.894 m |
| 17 | variant (new) | row-14 settings | C4 fails (forefoot steps: third foot 3.0 %); C5 passes, 8.8 mm |
| 18 | variant | time series of a forefoot swing | third foot 9-15 % through the swing, a dip to 3.8 % about 0.13 s after lift-off |
| 19 | variant | swing 0.7 s, 30 % lifting | C4 still fails (2.7-2.8 %) |
| 20 | variant | planned margin 65 mm, sideways ≤ 100 mm | C4 passes (third foot ≥ 8.5 %); C5 fails, 11.0 mm (first stance) |
| 21 | variant | pin motion per stance | the 11.0 mm in the first cycle only; steady 4.7-9.3 mm |
| 22 | variant | two setup swings before the cycles | C5 9.9 mm passes; C4 fails (setup step unloaded 78 %) |
| 23 | variant | per-step loads incl. setup | as row 22 |
| 24 | variant | lifting 20 % of the swing | C4 passes; C5 fails, 10.3 mm; 0.02 % saturation |
| 25-26 | variant | sideways cost 0.5 | C5 fails, 12.0 mm (hind feet slide); travel 1.124 m |
| 27 | variant | row-24 settings, pin detail | hind feet ~9.2 mm per stance |
| 28-29 | variant | stride 0.12 m | all six pass; pin 9.4 mm |
| 30-31 | variant | contact preload 2 mm | all six pass; pin 9.1 mm |
| 32 | variant | trace of the left hind ankle | passive pitch reached +25.5°, past its ±22° stop, early in stance; the sole tipped and crept ~5 mm |
| 33-34 | variant | sideways cost 1.0, fore-aft ≤ 50 mm | C5 fails, 12.8 mm; travel 0.861 m |
| 35-39 | variant | rows 30-31 settings on seeds 101-103 (T20) and 100-101 (T15) | 4 of 5 pass; T20 seed 101 fails C5 at 10.5 mm |
| 40-43 | variant and base | foot landing rule changed (see the report's correction on its wording) | variant T20 seeds 100, 101 pass (7.9, 8.8 mm); base T20 and T15 seed 100 fail C4, C5 (25.0, 24.9 mm) |
| 44-49 | variant | remaining development seeds | T20 102, 103 and T15 100-103 all pass (6.5-9.2 mm) |

The amendment says 48 runs; this ledger counts 49 simulations (rows 25-26, 28-31 and 33-34 each include a diagnostic
re-run of the same setting). Both are under the pre-registered budget of 60.
