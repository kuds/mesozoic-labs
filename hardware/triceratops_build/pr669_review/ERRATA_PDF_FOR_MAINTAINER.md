# Draft: corrections for the maintainer of Triceratops-design-package.pdf

For the owner to send to whoever builds the PDF (PR #669, head 84a13e1e). The repository holds only the built PDF, so
these cannot be made here; until a new PDF exists, the README's errata section carries them. Drafted 18 Nov 2026 and
revised the same day after an independent check. Page numbers are the PDF's own. Items 1 to 5 are the corrections the
consolidation plan asked for; items 6 to 10 were found in this review. The parts list's proposals cited below are not
adopted: mass, compute, bus voltage and chassis are open decisions for the owner.

| # | Page | Where | Now says | Change to |
|---|---|---|---|---|
| 1 | 6 | Plan view, compute box label | "Orin / MCU" | "Compute / MCU" (p.2 specifies a Pi 5 + AI HAT+; compute is an open decision, the parts list recommends a Jetson Orin Nano Super + Teensy 4.1) |
| 2a | 2 | "75 cm mass budget" | "battery 0.70" | "battery 0.755" (Gens ace G-Tech 6S 5000 mAh, repository catalog, maker page read 7 Nov 2026), and take the 0.055 kg from the 1.84 kg reserve |
| 2b | 6 | Side and plan views, "COMPUTE" / "BATTERY" and "Orin / MCU" / "6S pack" boxes | to scale but no dimension given; the battery box measures about 200 mm long, sits above the frame rails and runs from about 16 mm ahead of the body centre to about 185 mm behind it | Draw the boxes as the OpenSCAD and MuJoCo files place them (on the tray between the rails, 330-395 mm off the floor; the battery bay 120-280 mm behind the centre, which sets the 55 % front share), but with the battery bay at least 165 mm long (the 6S 5000 mAh pack is 165 × 45 × 47 mm); note that the 12S option's two 6S 3300 mAh packs are 138 × 42 × 43 mm each |
| 3 | 2, 3 | "Four limb modules"; "Size the fore elbow ..." | "a guarded 1:1 belt ... Provide a tensioner" | Add: "belt not sized; at a 20 mm pulley radius the trot design torque pulls 1.3 kN (15 kg) to 1.8 kN (20 kg); no belt maker's table has been read. The parts list (section 2.7) proposes a four-bar pushrod on 35-36 mm cranks, or an 8 mm-pitch high-modulus belt sized from its maker's table" |
| 4 | 2, 3 | Mass budget; fore elbow sizing | 15 kg brief peak only | Add a continuous case and a 20 kg case: held paw pose 12.2 N·m (15 kg) and 16.2 N·m (20 kg) at the elbow, continuous; 13.5 and 18.0 N·m at the motor through the 1:1 belt, 113 % and 150 % of 12 N·m, the X6-50-L's thermal-equilibrium torque (used as a stand-in: the X6-60 has no published table). 20 kg peaks: 32.1 N·m walk, 35.8 N·m trot (parts list, 2.3 and 2.6) |
| 5a | 7-9 | T. rex research appendix | Status as of 18 Nov 2026 before the pause | Stamp each page "Superseded: the T. rex study was paused and preserved on 18 Nov 2026 (T. rex decision log, P47-P51); the T. rex is the flagship for end 2027 / early 2028" |
| 5b | 5, 7 | "The branch reached the Colab smoke-test handoff"; "recorded delivery of the next training handoff" | as written | "The day-one Colab smoke test was skipped when the study paused (P50); the launch commit and data branch are kept for a restart" |
| 6 | 1, 4, 7 | "Stretch 0.40 m/s" | stretch target | Remove, or mark "not credible with X4 pitch drives: needs 75-77 rpm, 77-79 % of the X4's about 97 rpm no-load speed on 6S at 21 V (above the 70 % rule), and 91-93 % of the 83 rpm cap on 12S, with no margin under load (the X4's rated speed above 24 V, reseller text quoting MyActuator; to confirm)" |
| 7 | 2, 6, 8 | "6S battery"; "on the actual 6S bus" (p.2); "6S pack" (p.6); "removable 6S pack" (p.8) | 6S | Add: "the X6 is a 48 V drive; at 6S (21 V) its no-load speed is about 77 rpm. The 1:1 swing (48.1 rpm) fits at 62 % (59 % at the 22.2 V nominal), but 6S rules out the 1.3:1 linkage and the larger front drive the heat case needs. Bus voltage is an open decision; the parts list recommends 12S" |
| 8 | 2 | "Teensy 4.1 on two CAN buses, six drives each" | as written | Add: "loads 67-81 % at 500 Hz, with a worst-case exchange of about 2.2 ms against the 2 ms cycle; no room for 1 kHz. The parts list proposes four buses, one per leg" |
| 9 | 3 | "plan a lateral body shift toward the supporting triangle" | no number | Add: "at least 18 mm sideways (17.8 mm at the nominal footprint, support taken at the ankle pins), plus a margin, before a forefoot lifts" |
| 10 | 1 | "Aim for 55 % front / 45 % rear" | as written | Add: "the MuJoCo model reaches 55 % with its 1.84 kg reserve block 118 mm ahead of centre; without it, about 53.5 %" |

Rechecked in this review and left as they are ([arith_check_out.txt](arith_check_out.txt)): the leg-load arithmetic at
15 kg (14.5 N·m at the elbow, 21.7 N·m with the 1.5× allowance, 24.1 N·m at the motor; 19.8 N·m rear), the 53 N·m
forefoot-lift moment, the 25-36 min run-time range and the 15.00 kg mass sum. All 23 links to repository files in the
PDF (15 distinct files) resolve at their pinned commits ([pdf_links_check.txt](pdf_links_check.txt)); its 15
pull-request links were not followed. The swing speeds (69.0 / 48.1 rpm) were reproduced by the 18 Nov survey
(`hardware/triceratops_kickoff/reviews/survey_pr669_18nov.md`), not by this review.
