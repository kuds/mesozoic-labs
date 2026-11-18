# Review of PR #669, the Triceratops design study

18 Nov 2026, Triceratops build session; revised the same day after an independent check (last section). PR #669 head
84a13e1e (2 commits on 0748ac12), 8 new files under `hardware/triceratops_design_study/`. Read-only: nothing was pushed
to the PR's branch and nothing was posted on the PR.

## Verdict

**Sound as a concept study; merge it once the errata section is in.** Its main dimensions (stance 650 mm, track
350 mm, joints at 400 mm, links 235 / 230 mm, 110 × 80 mm pads), its 15.00 kg mass budget and its load arithmetic agree
across the PDF, the OpenSCAD envelope and the MuJoCo model, and its own check passes under mujoco 3.10.0. The PDF's
p.6 packaging boxes are schematic and do not match the tray layout of the other two files (finding 2). It is not a
build specification: the parts list (`hardware/triceratops_kickoff/PARTS_PREMIUM.md`) proposes replacing its drives,
transmission and power choices, which are open decisions for the owner.

- Errata text for its README: [ERRATA_README_SECTION.md](ERRATA_README_SECTION.md) (eleven items).
- List for the PDF's maintainer: [ERRATA_PDF_FOR_MAINTAINER.md](ERRATA_PDF_FOR_MAINTAINER.md).
- Merge: `git merge-tree` of the PR head against today's main (4b86f5c, after PR #670) is clean. Use a merge commit.

## What was run (simulation and file checks; nothing measured on hardware)

| Check | Result | Record |
|---|---|---|
| `models/check_models.py`, mujoco 3.10.0, Python 3.13 | **Passes**, exit 0. Robot: 14 actuators, 15.00 kg, settled foot loads 40.8 / 40.5 / 33.0 / 32.8 N. Anatomical: 26 actuators, 5,936 kg, 16,492 / 16,492 / 12,645 / 12,645 N | [check_models_out.txt](check_models_out.txt) |
| Mutation: battery mass 0.7 → 5.7 kg | `check_models.py` still passes, exit 0: 20.00 kg, 46.3 % front | [mutation_check_out.txt](mutation_check_out.txt) |
| Stance, torque and balance (this review's script) | Front share 55.2 % after 3 s at zero command. Settled torques: front elbow 5.5 N·m, hind knee 4.3 N·m, roll ≤ 0.42 N·m, pitch ≤ 0.15 N·m | [stance_balance_check.py](stance_balance_check.py), [stance_balance_out.txt](stance_balance_out.txt) |
| Support margin when one foot lifts | Support at the ankle pins: a forefoot lift needs an 18.0-19.1 mm sideways shift; a hind foot lifts with 15.9-16.8 mm of margin. Counting each sole's 80 mm width at its pin (soles are rigid in roll): the centre of mass is already 18.4-19.3 mm inside for a forefoot lift | [stance_balance_out.txt](stance_balance_out.txt); `../sim/support_margin.py`, `../sim/results/support_margin_2026-11-18/out.txt` |
| Arithmetic of the PDF | 14.5 / 21.7 / 24.1 N·m elbow, 19.8 N·m rear, 52.7 N·m lift moment, 25.4-35.5 min, 15.00 kg: all as the PDF states | [arith_check_out.txt](arith_check_out.txt) |
| PDF links | 38 links: 23 to repository files (15 distinct files), all resolve at their pinned commits (one needs the #663/#664 branch fetched; the consolidation plan keeps that branch); 15 to pull requests (14 distinct), not followed | [pdf_links_check.txt](pdf_links_check.txt) |
| README links (pinned to 42e6f507) | Both resolve | - |
| Merge with main | Clean against 4b86f5c | - |

**The settled elbow torque** (5.5 N·m at 15 kg) is below the parts list's 6.1 N·m standing figure. The independent
check split it about the elbow axis: the vertical load alone gives 6.3 N·m (the settled foot is 2.3 mm ahead of the
shoulder), the 2.5 N fore-aft ground force takes off 0.5 N·m and the lower leg's own weight 0.3 N·m; the ankle spring
adds almost nothing. The parts list ignores the last two, so its figure is the conservative one.

**The 17.8 mm shift on paper becomes 18.0-19.1 mm in the model** mainly because the model's centre of mass sits 1.3 mm
further forward (55.2 % front, not 55.0 %) and 0.5 mm left of the centreline; the settled feet add about 0.1 mm.

## Findings

Items 1 to 5 were asked for by the consolidation plan (section 2) and are confirmed here against the files; 6 to 11
are additions.

1. **p.6 "Orin / MCU" label** contradicts p.2 (Pi 5 + AI HAT+) and the model (`pi5_ai_hat_enclosure`, 0.39 kg).
2. **Battery and bay.** Model geom `6s_5ah_pack` is 160 × 90 × 65 mm and 0.70 kg; the OpenSCAD `pack_bay_mm =
   [160,90,65]`. The repository catalog's 6S 5000 mAh G-Tech pack (`designlab/data/parts.json`, `lipo_6s_5000`, maker
   page read 7 Nov 2026) is 0.755 kg and 165 × 45 × 47 mm: the bay is 5 mm short. The PDF's p.6 gives no bay dimension;
   its compute and battery boxes are schematic, drawn above the frame rails, the battery box about 200 mm long at the
   drawing's scale (checked on a render of the page and by the independent check's vector measurement, 198 mm).
3. **The 1:1 belt is not sized**: 1.34 kN (15 kg) and 1.79 kN (20 kg) at a 20 mm radius at the trot design torque
   (parts list, Appendix A). No belt maker's table was read (here or in the parts list); the survey's one unverified
   retailer figure is 285 N working tension for a 12 mm HTD 5M belt. The model maps the drive to the joint rigidly, so
   belt stretch and play are absent.
4. **No heat or 20 kg sizing.** In a held paw pose the X6 at 1:1 needs 13.5 N·m (15 kg) and 18.0 N·m (20 kg) at the
   motor: 113 % and 150 % of the X6-50-L's 12 N·m equilibrium torque, used as a stand-in (parts list, section 1 and
   Appendix A).
5. **T. rex pages 7-9, and p.5's "reached the Colab smoke-test handoff"**, predate the pause (T. rex decision log
   P47-P51 in `hardware/trex_form_study/DECISION_LOG.md`).
6. **0.40 m/s stretch** (pp.1, 4, 7): not credible with X4 pitch drives on either bus (79 % of no-load on 6S; 91-93 % of
   the 83 rpm cap on 12S; parts list F1).
7. **6S and the X6** (pp.2, 6, 8 and the README): the PR's 1:1 swing fits at 59-62 % of the X6's 6S no-load speed,
   but 6S rules out the 1.3:1 linkage and the larger front drive that finding 4 calls for.
8. **Two CAN buses of six drives** load 67-81 % at the PDF's 500 Hz, with a 2.2 ms worst-case exchange against a 2 ms
   cycle; there is no room for 1 kHz.
9. **The body shift before a forefoot lift** has no number in the PDF. At the ankle pins it is at least 18 mm; with the
   soles' width counted the centre of mass is already inside, but only by loading the supporting soles' edges. The
   crawl (task 4) must say which basis its margin uses.
10. **The 55 % front load leans on the 1.84 kg reserve block** placed 118 mm forward; without it 53.5 % (the centre of
    mass moves from x = +34.3 mm to +22.6 mm over the 650 mm stance; the independent check re-simulated it at 53.46 %).
11. **`check_models.py` is weak as a regression check**: it does not assert the 15 kg total or the front share, its
    load tolerance is ±10 %, and its bare `assert` statements vanish under `python -O`. A 20 kg, 46.3 %-front mutant
    passed.

## What the review did not do

- No walking, turning or push test: the PR has none, and free-base walking has never been simulated for this robot.
- The OpenSCAD file was not rendered, the turning circle was not recomputed, and `anatomical.xml` was read only
  through its check.
- The swing speeds (69.0 / 48.1 rpm) were not recomputed here; the 18 Nov survey reproduced them.
- No vendor page was opened; prices and specifications are the parts list's, with its sources and dates.

## Independent check (18 Nov 2026)

A separate reviewer, given the PR's files, the parts list, the kickoff documents and this folder's records (not this
review's conclusions), re-ran both scripts, wrote its own balance and force-balance scripts, measured p.6's vectors and
mutated the PR's model. Verdict: **sound after fixes**. Its fourteen findings and what was done:

| # | Finding | Done |
|---|---|---|
| 1 | "Bay drawn 160 mm" was wrong: the PDF gives no dimension; p.6's boxes are schematic and above the rails | Fixed (erratum 2, PDF row 2b, finding 2, verdict) after checking a render of p.6 |
| 2 | Link counts wrong (23 file links, 15 PR links) | Fixed |
| 3 | X6-80 saving is 0.52 kg per front leg, 1.0 kg for the pair | Fixed in the outreach draft |
| 4 | Erratum 8 corrected a 1 kHz claim the PDF never makes | Restated around the 500 Hz load and 2.2 ms exchange |
| 5 | "6S starves the X6" overstated for the PR's own 1:1 design | Restated (erratum 7) |
| 6 | "Replaces" implied the parts list was adopted; P1 marked 75 cm adopted | "Proposes"; P1 corrected |
| 7 | Belt capacity claimed without a source | Qualified (erratum 3) |
| 8 | `check_models.py` weaknesses missed | Added as erratum 11 after re-running the 5.7 kg mutant here (46.3 % front; the reviewer's figure was 45.9 %) |
| 9 | Balance figures: 18.0 not 18.1 mm; cause of the rise; the sole-width basis | Fixed after this session's own three-basis script confirmed 18.4-19.3 mm inside on the sole-width basis |
| 10 | Elbow-torque explanation wrong; roll 0.42 not 0.4 | Fixed, citing the reviewer's decomposition |
| 11 | PDF rows 4, 6, 7: stand-in wording, the 6S speed argument, page numbers | Fixed |
| 12 | "Rechecked" with no saved record | `arith_check.py` and its output added |
| 13 | Errata cited a folder not on main | Branch and commit named |
| 14 | Vendor drafts stated open choices as plans; P48 credited with the merge | Reworded |
