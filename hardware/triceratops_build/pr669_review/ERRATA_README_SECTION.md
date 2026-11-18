# Draft: errata section for PR #669's README

For the owner to paste at the end of `hardware/triceratops_design_study/README.md` on PR #669's branch before
merging it with a merge commit. Drafted 18 Nov 2026 in the Triceratops build session; the session does not push to
the PR's branch. Everything between the two rules below is the proposed text. Items 1 to 5 are the corrections the
consolidation plan asked for; items 6 to 10 were found in this review ([REVIEW.md](REVIEW.md)). Delete any you do not
want.

---

## Errata (added before merging)

This folder is a **concept study**, not a build specification. The checked parts list
[`hardware/triceratops_kickoff/PARTS_PREMIUM.md`](../triceratops_kickoff/PARTS_PREMIUM.md) replaces its drive,
transmission and power choices. The design package PDF has no source in this repository, so its corrections are
listed here and sent to its maintainer. Nothing below has been built or measured; figures are estimates from the parts
list unless marked as a simulation result.

1. **PDF p.6, plan view: the "Orin / MCU" label is wrong for this design.** The package's own text (p.2) and the
   MuJoCo model (`pi5_ai_hat_enclosure`) use a Raspberry Pi 5 + AI HAT+ with a Teensy 4.1. Read the label as "Compute /
   MCU". The compute choice is still open: the parts list recommends a Jetson Orin Nano Super with a Teensy 4.1.
2. **Battery mass and bay length (PDF p.2 and p.6; `models/robot.xml` geom `6s_5ah_pack`; `Triceratops-envelope.scad`
   `pack_bay_mm`).** The 6S 5000 mAh pack in the repository catalog (Gens ace G-Tech) weighs **0.755 kg**, not the
   0.70 kg budgeted, and is 165 × 45 × 47 mm. The bay is drawn and modelled 160 mm long, so it must grow to **at least
   165 mm**. (The parts list's 12S option uses two 6S 3300 mAh packs instead: 1.012 kg together, each 138 × 42 × 43 mm.)
3. **The 1:1 flexion belt is not sized (PDF pp.2-3; models README).** The 12 mm belt on 20 mm pulleys would carry
   **1.3 kN at 15 kg and 1.8 kN at 20 kg** at the trot design torque, far above what a 12 mm belt of this size carries,
   and its stretch and play are not in the model (the model maps the drive to the joint rigidly). The parts list
   (sections 1 and 2.7) replaces it with a rigid four-bar pushrod on 35-36 mm cranks, sized for 1.16-1.82 kN at a 40°
   transmission angle, or an 8 mm-pitch high-modulus belt sized from its maker's table.
4. **Heat and the 20 kg case are not sized.** The study checks a brief 15 kg peak only. With the X6 at 1:1 on the
   elbow, a held one-forefoot ("paw") pose needs 113 % of the drive's thermal-equilibrium torque at 15 kg and 150 % at
   20 kg (against the X6-50-L's maker table, 12 N·m, the nearest published one; the X6-60 has none). At 20 kg the front elbow needs 32.1 N·m walking peak, 35.8 N·m trotting peak and 16.2 N·m held (parts list,
   sections 1, 2.3 and 2.6).
5. **The T. rex pages (PDF pp.7-9) are superseded.** On 18 Nov 2026 the T. rex study was paused and preserved (T. rex
   decision log, rows P47-P51): training paused (P47), the study merged into main (P48), the Colab smoke test skipped
   (P50), and the T. rex kept as the flagship for end 2027 / early 2028 (P51). These pages, and p.5's line that the
   branch "reached the Colab smoke-test handoff", record the state before the pause. The open-pull-request list on p.9
   is the 18 Nov count, before the owner's triage.
6. **The 0.40 m/s stretch walk (PDF pp.1, 4 and 7) is not credible with X4 pitch drives.** It needs 75-77 rpm at the
   pitch joint. Above its 24 V rating the X4 is capped at its 83 rpm rated speed (reseller text quoting MyActuator; to
   be confirmed with the maker), which leaves no margin under load, and on 6S its no-load speed at 21 V is about
   97 rpm. The museum walk is **0.30 m/s (0.7 mph)**.
7. **The 6S bus starves the X6 (PDF p.2; README "on the intended 6S supply").** The X6 is a 48 V drive; at 6S (21 V
   minimum) its no-load speed falls to about 77 rpm. The bus voltage is an open decision; the parts list recommends a
   48 V class (12S) bus with a regen clamp and precharge.
8. **Two CAN buses with six drives each cannot run a 1 kHz loop (PDF p.2).** They load 67-81 % at 500 Hz. The parts
   list uses four buses, one per leg.
9. **The sideways body shift before a forefoot lifts needs a number (PDF p.3).** With the centre of mass on the
   centreline 358 mm ahead of the hind feet, the body must shift sideways **at least 18 mm** (17.8 mm on the nominal
   footprint; 18.1-19.1 mm in this model after 3 s of stance), plus a margin, before either forefoot lifts. A hind foot
   can lift with about 16 mm of margin without a shift (simulation, `robot.xml`, mujoco 3.10.0, 18 Nov 2026).
10. **The model's 55 % front load includes the 1.84 kg reserve block** (`design_mass_reserve`, 118 mm ahead of the
    chassis centre). Without it the parts as modelled put about 53.5 % on the front feet.

**Checks run on this folder (18 Nov 2026).** `models/check_models.py` passes with mujoco 3.10.0 (exit 0): robot 15.00
kg, settled foot loads 40.8 / 40.5 / 33.0 / 32.8 N (55.2 % front); anatomical adult 5,936 kg, foot loads 16,492 /
16,492 / 12,645 / 12,645 N. The run, the stance and balance check, and the PDF link check are kept in
`hardware/triceratops_build/pr669_review/`. No continuous integration runs these models (the CI path filter leaves out
`hardware/**`).

---
