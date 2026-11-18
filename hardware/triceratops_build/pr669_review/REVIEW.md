# Review of PR #669, the Triceratops design study

18 Nov 2026, Triceratops build session. PR #669 head 84a13e1e (2 commits on 0748ac12), 8 new files under
`hardware/triceratops_design_study/`. Read-only: nothing was pushed to the PR's branch and nothing was posted on the PR.

## Verdict

**Sound as a concept study; merge it once the errata section is in.** Its geometry, masses and load arithmetic agree
across the PDF, the OpenSCAD envelope and the MuJoCo model, and its own check passes under mujoco 3.10.0. It is not a
build specification: the parts list (`hardware/triceratops_kickoff/PARTS_PREMIUM.md`) replaces its drives,
transmission and power choices.

- Errata text for its README: [ERRATA_README_SECTION.md](ERRATA_README_SECTION.md) (ten items).
- List for the PDF's maintainer: [ERRATA_PDF_FOR_MAINTAINER.md](ERRATA_PDF_FOR_MAINTAINER.md).
- Merge: `git merge-tree` of the PR head against today's main (4b86f5c, after PR #670) is clean. Use a merge commit.

## What was run (simulation and file checks; nothing measured on hardware)

| Check | Result | Record |
|---|---|---|
| `models/check_models.py`, mujoco 3.10.0, Python 3.13 | **Passes**, exit 0. Robot: 14 actuators, 15.00 kg, settled foot loads 40.8 / 40.5 / 33.0 / 32.8 N. Anatomical: 26 actuators, 5,936 kg, 16,492 / 16,492 / 12,645 / 12,645 N | [check_models_out.txt](check_models_out.txt) |
| Stance, torque and balance check (this review's script) | Front share 55.2 % after 3 s at zero command. Settled torques: front elbow 5.5 N·m, hind knee 4.3 N·m, roll ≤ 0.4 N·m, pitch ≤ 0.15 N·m. Sideways shift needed before a forefoot lifts: 18.1 mm (right fore) and 19.1 mm (left fore); a hind foot can lift with 15.9-16.8 mm of margin and no shift | [stance_balance_check.py](stance_balance_check.py), [stance_balance_out.txt](stance_balance_out.txt) |
| PDF links | 21 links to repository files, all resolve at their pinned commits (one needs the #663/#664 branch fetched; the consolidation plan keeps that branch). 17 links to pull requests not checked | [pdf_links_check.txt](pdf_links_check.txt) |
| README links (pinned to 42e6f507) | Both resolve | - |
| Merge with main | Clean against 4b86f5c | - |

The settled elbow torque, 5.5 N·m at 15 kg, sits below the parts list's 6.1 N·m standing figure (parts list, section
2.3): the parts list puts the foot exactly under the shoulder and ignores the ankle spring, so it is the conservative
one. The 17.8 mm shift on paper (survey, section 2) becomes 18.1-19.1 mm in the model because the settled feet sit 2 mm
wider and further out and the model's centre of mass is 0.5 mm left of the centreline.

## Findings

Items 1-5 were asked for by the consolidation plan (section 2) and are confirmed here against the files; 6-10 are
additions.

1. **p.6 "Orin / MCU" label** contradicts p.2 (Pi 5 + AI HAT+) and the model (`pi5_ai_hat_enclosure`, 0.39 kg).
2. **Battery.** Model geom `6s_5ah_pack` is 160 × 90 × 65 mm and 0.70 kg; the OpenSCAD `pack_bay_mm = [160,90,65]`.
   The repository catalog's 6S 5000 mAh G-Tech pack (`designlab/data/parts.json`, `lipo_6s_5000`, maker page read
   7 Nov 2026) is 0.755 kg and 165 × 45 × 47 mm. The bay is 5 mm short.
3. **The 1:1 belt is not sized**: 1.34 kN (15 kg) and 1.79 kN (20 kg) at a 20 mm radius at the trot design torque
   (parts list, Appendix A). The model maps the drive to the joint rigidly, so belt stretch and play are absent.
4. **No heat or 20 kg sizing.** The X6 at 1:1 runs 113 % (15 kg) and 150 % (20 kg) of the X6-50-L's 12 N·m
   equilibrium torque in a held paw pose (parts list, section 1 and Appendix A).
5. **T. rex pages 7-9, and p.5's "reached the Colab smoke-test handoff"**, predate the pause (T. rex decision log
   P47-P51 in `hardware/trex_form_study/DECISION_LOG.md`).
6. **0.40 m/s stretch** (pp.1, 4, 7): not credible with X4 pitch drives (parts list F1).
7. **6S starves the 48 V X6** (p.2 and the README's "intended 6S supply").
8. **Two CAN buses of six drives** cannot carry a 1 kHz loop (67-81 % load at 500 Hz).
9. **The body shift before a forefoot lift** has no number in the PDF; the check above gives at least 18 mm.
10. **The 55 % front load leans on the 1.84 kg reserve block** placed 118 mm forward; without it about 53.5 %
    (computed: centre of mass moves from x = +34.3 mm to +22.6 mm over the 650 mm stance).

## What the review did not do

- No walking, turning or push test: the PR has none, and free-base walking has never been simulated for this robot.
- The OpenSCAD file was not rendered, the turning circle was not recomputed, and `anatomical.xml` was read only
  through its check.
- The swing speeds (69.0 / 48.1 rpm) were not recomputed here; the 18 Nov survey reproduced them.
- No vendor page was opened; prices and specifications are the parts list's, with its sources and dates.
