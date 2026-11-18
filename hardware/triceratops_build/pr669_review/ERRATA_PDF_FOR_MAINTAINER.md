# Draft: corrections for the maintainer of Triceratops-design-package.pdf

For the owner to send to whoever builds the PDF (PR #669, head 84a13e1e). The repository holds only the built PDF, so
these cannot be made here; until a new PDF exists, the README's errata section carries them. Drafted 18 Nov 2026.
Page numbers are the PDF's own. Items 1 to 5 are the corrections the consolidation plan asked for; items 6 to 10 were
found in this review.

| # | Page | Where | Now says | Change to |
|---|---|---|---|---|
| 1 | 6 | Plan view, compute bay label | "Orin / MCU" | "Compute / MCU" (p.2 specifies a Pi 5 + AI HAT+; compute is an open decision, the parts list recommends a Jetson Orin Nano Super + Teensy 4.1) |
| 2a | 2 | "75 cm mass budget" | "battery 0.70" | "battery 0.755" (Gens ace G-Tech 6S 5000 mAh, repository catalog, maker page read 7 Nov 2026), and take the 0.055 kg from the 1.84 kg reserve |
| 2b | 6 | Side and plan views, battery bay | drawn 160 mm long | at least 165 mm long (the pack is 165 × 45 × 47 mm); note that the 12S option's two 6S 3300 mAh packs are 138 × 42 × 43 mm each |
| 3 | 2, 3 | "Four limb modules"; "Size the fore elbow ..." | "a guarded 1:1 belt ... Provide a tensioner" | Add: "belt not sized; at a 20 mm pulley radius the design torque pulls 1.3 kN (15 kg) to 1.8 kN (20 kg). See the parts list, section 2.7: a four-bar pushrod on 35-36 mm cranks, or an 8 mm-pitch high-modulus belt sized from its maker's table" |
| 4 | 2, 3 | Mass budget; fore elbow sizing | 15 kg brief peak only | Add a continuous case and a 20 kg case: held paw pose 12.2 N·m (15 kg) and 16.2 N·m (20 kg) at the elbow, continuous; X6 at 1:1 at 113 % / 150 % of its thermal-equilibrium torque. 20 kg peaks: 32.1 N·m walk, 35.8 N·m trot (parts list, 2.3 and 2.6) |
| 5a | 7-9 | T. rex research appendix | Status as of 18 Nov 2026 before the pause | Stamp each page "Superseded: the T. rex study was paused and preserved on 18 Nov 2026 (T. rex decision log, P47-P51); the T. rex is the flagship for end 2027 / early 2028" |
| 5b | 5, 7 | "The branch reached the Colab smoke-test handoff"; "recorded delivery of the next training handoff" | as written | "The day-one Colab smoke test was skipped when the study paused (P50); the launch commit and data branch are kept for a restart" |
| 6 | 1, 4, 7 | "Stretch 0.40 m/s" | stretch target | Remove, or mark "not credible with X4 pitch drives: needs 75-77 rpm against the X4's 83 rpm rated-speed cap above 24 V (reseller text quoting MyActuator; to confirm)" |
| 7 | 2, 3 | "6S battery"; "on the actual 6S bus" | 6S | Add: "the X6 is a 48 V drive; at 6S (21 V) its no-load speed is about 77 rpm. Bus voltage is an open decision; the parts list recommends 12S" |
| 8 | 2 | "Teensy 4.1 on two CAN buses, six drives each" | as written | Add: "loads 67-81 % at 500 Hz; cannot run 1 kHz. The parts list uses four buses, one per leg" |
| 9 | 3 | "plan a lateral body shift toward the supporting triangle" | no number | Add: "at least 18 mm sideways (17.8 mm at the nominal footprint), plus a margin, before a forefoot lifts" |
| 10 | 1 | "Aim for 55 % front / 45 % rear" | as written | Add: "the MuJoCo model reaches 55 % with its 1.84 kg reserve block 118 mm ahead of centre; without it, about 53.5 %" |

Rechecked in this review and left as they are: the leg-load arithmetic at 15 kg (14.5 N·m at the elbow, 24.2 N·m at
the motor; 19.8 N·m rear), the 53 N·m forefoot-lift moment, the 25-36 min run-time range, the 15.00 kg mass sum, and
every repository link in the PDF (21 file links, all resolve at their pinned commits;
[pdf_links_check.txt](pdf_links_check.txt)). The swing speeds (69.0 / 48.1 rpm) were reproduced by the 18 Nov survey
(`hardware/triceratops_kickoff/reviews/survey_pr669_18nov.md`), not by this review.
