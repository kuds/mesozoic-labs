# Draft: errata section for PR #669's README

For the owner to paste at the end of `hardware/triceratops_design_study/README.md` on PR #669's branch before
merging it with a merge commit. Drafted 18 Nov 2026 in the Triceratops build session and revised the same day after an
independent check (see [REVIEW.md](REVIEW.md), "Independent check"); the session does not push to the PR's branch.
Everything between the two rules below is the proposed text. Items 1 to 5 are the corrections the consolidation plan
asked for; items 6 to 11 were found in this review. Delete any you do not want.

---

## Errata (added before merging)

This folder is a **concept study**, not a build specification. The checked parts list
[`hardware/triceratops_kickoff/PARTS_PREMIUM.md`](../triceratops_kickoff/PARTS_PREMIUM.md) proposes replacements for its
drive, transmission and power choices; those are open decisions for the owner. The design package PDF has no source
in this repository, so its corrections are listed here and sent to its maintainer. Nothing below has been built or
measured; figures are estimates from the parts list unless marked as a simulation result.

1. **PDF p.6, plan view: the "Orin / MCU" label is wrong for this design.** The package's own text (p.2) and the
   MuJoCo model (`pi5_ai_hat_enclosure`) use a Raspberry Pi 5 + AI HAT+ with a Teensy 4.1. Read the label as "Compute /
   MCU". The compute choice is still open: the parts list recommends a Jetson Orin Nano Super with a Teensy 4.1.
2. **Battery mass and bay length.** The 6S 5000 mAh pack in the repository catalog (Gens ace G-Tech) weighs **0.755
   kg**, not the 0.70 kg budgeted (PDF p.2), and is 165 × 45 × 47 mm. The bay is 160 mm long in
   `Triceratops-envelope.scad` (`pack_bay_mm`) and in `models/robot.xml` (geom `6s_5ah_pack`), so it must grow to **at
   least 165 mm**. The PDF's p.6 drawing gives no bay dimension, but it is to scale, and its battery box differs from those files
   in size, height and fore-aft position: about 200 mm long, above the frame rails, from about 16 mm ahead of the body
   centre to about 185 mm behind it, where the files put the bay 120-280 mm behind the centre on the tray (the position
   that sets the 55 % front share). (The parts list's 12S option uses two
   6S 3300 mAh packs instead: 1.012 kg together, each 138 × 42 × 43 mm.)
3. **The 1:1 flexion belt is not sized (PDF pp.2-3; models README).** At the 20 mm pulley radius the trot design
   torque pulls **1.3 kN at 15 kg and 1.8 kN at 20 kg**. No belt maker's table has been read; one unverified retailer
   listing gives a 285 N working tension for a 12 mm HTD 5M belt, so the belt looks badly undersized. Its stretch and
   play are not in the model either (the model maps the drive to the joint rigidly). The parts list (sections 1 and
   2.7) proposes a rigid four-bar pushrod on 35-36 mm cranks, sized for 1.16-1.82 kN at a 40° transmission angle, or an
   8 mm-pitch high-modulus belt sized from its maker's table.
4. **Heat and the 20 kg case are not sized.** The study checks a brief 15 kg peak only. With the X6 at 1:1 on the
   elbow, a held one-forefoot ("paw") pose needs 13.5 N·m at the motor at 15 kg and 18.0 N·m at 20 kg: 113 % and 150 %
   of 12 N·m, the thermal-equilibrium torque in the X6-50-L's maker table (the nearest published one; the X6-60 has
   none). At 20 kg the front elbow needs 32.1 N·m walking peak, 35.8 N·m trotting peak and 16.2 N·m held (parts list,
   sections 1, 2.3 and 2.6).
5. **The T. rex pages (PDF pp.7-9) are superseded.** On 18 Nov 2026 the T. rex study was paused and preserved (T. rex
   decision log, rows P47-P51): training paused (P47), the study kept findable through a pull request into main (P48;
   merged as PR #670), the Colab smoke test skipped (P50), and the T. rex kept as the flagship for end 2027 / early 2028
   (P51). These pages, and p.5's line that the branch "reached the Colab smoke-test handoff", record the state before
   the pause. The open-pull-request list on p.9 is the 18 Nov count, before the owner's triage.
6. **The 0.40 m/s stretch walk (PDF pp.1, 4 and 7) is not credible with X4 pitch drives.** It needs 75-77 rpm at the
   pitch joint. On the PR's 6S bus that is 77-79 % of the X4's no-load speed at 21 V (about 97 rpm), above the 70 % rule;
   on a 12S bus the X4 is capped at its 83 rpm rated speed above its 24 V rating (reseller text quoting MyActuator; to
   be confirmed with the maker), which leaves no margin under load. The museum walk is **0.30 m/s (0.7 mph)**.
7. **6S and the X6 (PDF p.2; README "on the intended 6S supply").** The X6 is a 48 V drive; at 6S (21 V minimum) its
   no-load speed falls to about 77 rpm. The PR's own 1:1 swing (48.1 rpm) still fits at 62 % of that (59 % at the 22.2 V nominal), but 6S rules
   out the 1.3:1 linkage and the larger front drive that the heat fix in item 4 needs. The bus voltage is an open
   decision; the parts list recommends a 48 V class (12S) bus with a regen clamp and precharge.
8. **Two CAN buses with six drives each are tight at 500 Hz (PDF p.2).** They load 67-81 % at the 500 Hz the PDF asks
   for (p.5), and the worst-case exchange takes about 2.2 ms against the 2 ms cycle (T. rex study,
   `stage2/jetson75/LATENCY_AND_BUS.md`), so there is no room for a 1 kHz loop. The parts list proposes four buses, one
   per leg.
9. **The sideways body shift before a forefoot lifts needs a number (PDF p.3).** With the centre of mass on the
   centreline 358 mm ahead of the hind feet and the support taken at the ankle pins, the body must shift sideways
   **at least 18 mm** (17.8 mm on the nominal footprint; 18.0-19.1 mm in this model after 3 s of stance), plus a margin,
   before either forefoot lifts. Because each sole is rigid in roll, its 80 mm width also supports: on that basis the
   centre of mass is already 18-19 mm inside the three-foot area, but only by loading the edges of the supporting soles.
   A hind foot lifts with about 16 mm of margin at the pins. (Simulation, `robot.xml`, mujoco 3.10.0, 18 Nov 2026.)
10. **The model's 55 % front load includes the 1.84 kg reserve block** (`design_mass_reserve`, 118 mm ahead of the
    chassis centre). Without it the parts as modelled put about 53.5 % on the front feet.
11. **`models/check_models.py` checks less than its output suggests.** It prints the total mass and the four foot loads
    but asserts neither the 15 kg total nor the 55 % front share: it only requires each foot to carry more than 2 % of
    the weight and the loads to sum to within 10 % of it. A copy with the battery set to 5.7 kg (20.00 kg, 46.3 % front; `hardware/triceratops_build/pr669_review/mutation_check_out.txt` on that branch)
    passed with exit 0. Its checks are bare `assert` statements, which `python -O` skips. Worth tightening if the script
    is kept as a regression check.

**Checks run on this folder (18 Nov 2026).** `models/check_models.py` passes with mujoco 3.10.0 (exit 0): robot 15.00
kg, settled foot loads 40.8 / 40.5 / 33.0 / 32.8 N (55.2 % front); anatomical adult 5,936 kg, foot loads 16,492 /
16,492 / 12,645 / 12,645 N. The run, the stance and balance checks, the arithmetic recheck and the PDF link check are
kept in `hardware/triceratops_build/pr669_review/` on the branch `claude/pensive-pasteur-o3bqtv` (the records from
commit 33665ec on). No continuous integration runs these models (the CI path filter leaves out `hardware/**`).

---
