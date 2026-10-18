# T. rex foot, ankle and posture audit (main 2e734e29, physics r9)

2026-10-17. Measurement only; nothing in the model or repository was changed. Evidence paths are relative to `scratchpad/foot/audit_2026_10_17/`. Model numbers come from `geometry/scripts/trex_geom_audit.py` (→ `geometry/geometry_audit.json`), `verify/scripts/vx_static.py` and `vx_roll.py` (→ `verify/out/*.json`), `posture/tools/posture_roll.py` and `posture_measure.py` (→ `posture/out/posture_table.txt`), and `literature/scripts/lit_model_frame.py` and `lit_convert.py` (→ `literature/out/`).

Terms: *home* is the authored keyframe. *Settled statue* is a 1000-step zero-action settle with no reset noise, averaged over steps 200–999. *s60* and *s52* are the r9 arm L checkpoints at 6,291,456 steps, run with options off on seeds 3042–3051.

## Short answer

- **Matches:** footprint-to-hip-height ratio, digit lengths, load-centre position, COM ahead of the hip, knee range, and a knee near the ground-force line.
- **Differs:**
  - The COM is 4–18 cm too high relative to the hip, in every pose.
  - The thigh is light, the forelimbs heavy, and the shank and metatarsus proportions juvenile.
  - The digit spread is 3–5× too narrow and in reversed order.
  - The rear contact is a midline pad, not a lateral digit IV pad.
  - The ankle extends 11° past the only published T. rex model range.
- **Nothing measured in the anatomy pushes the quiet stance forward.** In the statue and both 6M policies the COM projects 2.4–6.8 cm ahead of MTP III, inside the proximal-phalanx zone where tyrannosaur tracks are deepest.
- **Two features make the stance read, or let it go, further forward:**
  - The gate reads the pad's own CoP. That sits 0.10–0.25 pad half-lengths ahead of the COM, because 16–28% of the load is on the digits right at the MTP joints.
  - The rigid pad front, the edge the policy follows, sits 4.6–6.0 cm beyond the published load zone.
- **The gate is unsigned,** so a track-like stance over the MTP joints (−0.86) would fail it.
- **Not measured:** the drift state itself.

## Audit table

| Quantity | Published value (source) | Model value (evidence) | Reading |
|---|---|---|---|
| Metatarsus ÷ femur | Adults 0.52–0.60, juvenile 0.62 ([Persons & Currie 2016](https://doi.org/10.1038/srep19828) **Table 2**, fetched `literature/dl/persons_currie2016/PMC4728391.xml`). Sue 0.54–0.55: my estimate from Brochu 2003 Fig. 100, digitised (`literature/out/brochu_pes_summary.json`) | 0.613 (`verify/out/static.json`) | Juvenile; above every adult |
| Tibia ÷ femur | Adults 0.90–0.93, juvenile 1.00 (same table) | 1.000 (`static.json`) | Juvenile |
| Hip height ÷ femur | MOR 555 1.95–2.46 ([Boeye et al. 2026](https://doi.org/10.1098/rsos.252139)) | 2.50; leg ÷ femur 2.61 (`literature/out/model_frame.json`) | High, because the shank is long |
| Digit III ÷ metatarsus | 0.87 for MOR 555 ([Hutchinson et al. 2005](https://doi.org/10.1666/0094-8373%282005%29031%5B0676%3AAOHMMA%5D2.0.CO%3B2) App. 2); 0.94–0.98 for Sue (figure estimate) | 0.88 without the capsule cap, 1.01 with it (`model_frame.json`) | Not over-long |
| Digit spread | II–III 22–38°, III–IV 29–42°, total 51–80°; III–IV usually the wider pair ([McCrea et al. 2014](https://doi.org/10.1371/journal.pone.0103613) Table 1) | 8.7° / 7.6° / 16.3° (`static.json`) | 3.1–4.9× narrower, order reversed. Affects roll, not fore-aft |
| Rear contact | Digit II–III MTP pads typically held clear (Thulborn 1990). The digit IV MTP pad was "thought to have commonly been impressed" (Farlow et al. 2000). Both via [Lallensack, Farlow & Falkingham 2022](https://researchonline.ljmu.ac.uk/15899/1/Lallensack%20Farlow%20Falkingham%202021%20Palaeontology%20elongate%20tracks.pdf) | Midline box pad from −0.010 to +0.130 m about MTP III, ±0.045 m laterally. All three MTP joints at one fore-aft station (`static.json`) | Rear edge agrees within about 1 cm fore-aft (inference). Wrong side: the pad covers the II–III region that tracks show held clear |
| Load centre | Tracks deepest "around the proximal pedal phalanges" (Boeye 2026; locomotion tracks, and depth is not pressure). Ostrich running: digit III 77–84%, claw 1.5–2.4% ([Zhang et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5530993/)) | Pad centre 0.060 m ahead of MTP III (27.5% of digit III). The III-1 zone spans 0 to 0.070 m (Sue) or 0.084 m (MOR 555) (`literature/out/load_centre_frame.json`; inference) | Pad centre lies inside the zone. Plausible |
| Digit share of foot load | No standing data exist for any theropod or ratite | 13.5% at the home instant, which is not an equilibrium (`static.json`). Settled statue 24.1%, range 21.7–25.8% (`settle_statue_noise0.json`). Statue with spawn noise 16 ± 5%; s60 17%; s52 28% (`posture/out/posture_table.txt`). Load sits 2–21% of digit length from the MTP; claws carry none | Proximal, so bird- and track-like. The report's "22–23% at home" traced to a stale `base_env.py` comment and is withdrawn |
| COM ahead of hip | 30–45% of femur, Sue 44–61% ([Hutchinson et al. 2011](https://doi.org/10.1371/journal.pone.0026037), verbatim). [Bates et al. 2009](https://doi.org/10.1371/journal.pone.0004532) Table 8, whole body: 0.468 m (MOR 555), 0.572 m (BHI 3033) | Home 41.9%. Statue 45.4%, s60 43.5%, s52 45.6% (`static.json`, `posture_table.txt`) | Top edge of the band once settled |
| COM below hip | 22–50% of femur for four adults; 0.077–0.176 m on the model femur (Hutchinson 2011) | Home 2.1% *above* the hip. Statue 11.2%, s60 6.6%, s52 7.7% below | **Mismatch in every pose**: 0.04–0.18 m higher than published (my arithmetic) |
| Segment mass shares | Adult minimal models, Tables 4–5 (Hutchinson 2011): head 4.1–11.1%, neck 3.2–7.3%, tail 9.9–20.7%, forelimb 0.1–0.4%, thigh 12.6–17.4%, shank 4.3–5.8%, foot 1.3–2.9%. Whether these are per limb or for both limbs is **unresolved**: Tables 9–10 say per limb, but the segment masses add up to body mass only if each limb segment is counted once (`verify/out/h2011_mass_sum.txt`) | Head 10.1%, neck 5.8%, trunk 40.8%, tail 18.2%, forelimbs 1.05%. Per side: thigh 5.8%, shank 4.1%, foot 2.1%. Both hindlimbs 23.9% (`static.json`) | Head, neck, tail and foot inside. Forelimbs heavy. **Thigh light under either reading** (2.2–3.0× per limb; 0.67–0.93× read as a pair). Shank just below the per-limb range. Total hindlimb share depends on the convention |
| Footprint ÷ hip height | 0.216–0.270 (Bellatoripes, 62 cm track at 2.30–2.87 m hip height; McCrea 2014, my arithmetic) | 0.260; footprint 0.228 m (`static.json`) | Inside |
| Knee range | 90–190° included (Hutchinson 2005 Table 1; a model-allowed range). Fig. 8 knee 135° (verified, `literature/dl/hutchinson2005/h2005.txt`) | 85–185°, home 135.0° (`static.json`) | Within 5°. The MJCF's citation is verified |
| Ankle range | T. rex model 90–180° included (Hutchinson 2005 Table 1). Ostrich stance ≥164° walking / ≥160° running, briefly 140° in fast running ([Schaller et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705294/); included-angle reading is my inference). Guineafowl ROM *width* 162–176° ([Kambic et al. 2017](https://doi.org/10.1111/joa.12680)) | 91.2–191.2° (100° wide) (`static.json`) | Flexion limit matches. **Allows 11.2° of hyperextension** that the published model does not |
| MTP range | 45–180° included (Hutchinson 2005 Table 1) | 74.3–149.3°, passive k 40, c 15, no actuator (`static.json`) | Inside the published model range. It is a model allowance, not a measured limit |
| Hip range | 65° flexion to 45° extension about a columnar pose (Hutchinson 2005 Table 1) | Femur from 87.5° ahead of vertical to 42.5° behind (`static.json`) | Flexes 22.5° further, assuming a level pelvis |
| Stiffness ÷ mgh, about the ankle pivot | Human intrinsic stiffness 91 ± 23% ([Loram & Lakie 2002](https://doi.org/10.1113/jphysiol.2002.025049)) and 64 ± 8% ([Casadio et al. 2005](https://doi.org/10.1016/j.gaitpost.2004.05.005)); long-range 13 ± 2% ([Loram et al. 2007](https://doi.org/10.1113/jphysiol.2007.140046)) | mgh about the ankle: 552 (home), 513 (settled) N·m/rad. Digit springs 240 = 0.43–0.47. Ankle servos 1800 = 3.3–3.5× (`static.json`, `settle_statue_noise0.json`) | Rough index only. The report's 0.32 and 2.4× used floor height |
| Tail pitch authority | Tail as an active balancer ([Libby et al. 2012](https://doi.org/10.1038/nature10710)); a passive tail had negligible effect in *Coelophysis* ([Bishop et al. 2021](https://doi.org/10.1126/sciadv.abi7348)) | tail_1 rests on its −12° stop. A full-up command cannot lift it: servo 41.9 + spring 16.8 N·m against gravity 70.0–70.6 N·m. Its full range moves the COM about 0.07 pad half-lengths, about 5 mm (`geometry/geometry_audit.json` `tail_1_holding_budget_at_ventral_stop`, `balance_authority_*`) | No static fore-aft authority |
| Neck and head authority | Hip / counter-rotation strategy ([Horak & Nashner 1986](https://doi.org/10.1152/jn.1986.55.6.1369); [Hof 2007](https://doi.org/10.1016/j.jbiomech.2005.12.016)) | Statue droops 12°. A ramped full-up command moves the COM −0.40 pad half-lengths, full-down −0.28. A full-up step command topples the statue backward at step 224 (`geometry_audit.json`). s60 holds the neck at −5 to −6.5° (`posture_s60_6M.json`) | Strongest channel, but it only moves the COM rearward |

## Posture: statue, learned stances and the published band

| Steps 200–999 | Home | Statue (no noise / spawn noise) | s60 | s52 | Published |
|---|---|---|---|---|---|
| Femur from vertical | 22.5° | 23.7° / 23.7° | 22.4° | 27.6° | "Largely upright, with a subvertical femoral orientation" ([Bishop et al. 2018](https://doi.org/10.7717/peerj.5777)). "A narrow region of neither very columnar nor very flexed poses", at mid-stance of locomotion ([Gatesy et al. 2009](https://doi.org/10.1671/039.029.0213), abstract). No numeric value reached |
| Knee, included | 135.0° | 135.4° | 136.1° | 134.5° | "Required some flexion in the limbs to stand still" ([Hutchinson et al. 2007](https://doi.org/10.1016/j.jtbi.2007.01.023)). McCrea 2014's flexed hip-height case uses 110° |
| Ankle, included | 135.7° | 137.4° | 136.8° | 140.1° | McCrea 2014 flexed case 140° (my inference: an included angle). Ostrich ≥160° |
| Metatarsus from vertical / MTP angle | 21.8° / 111.8° | 21.7° / 111.7° | 21.7° / 111.6° | 22.0° / 111.8° | No value |
| Pelvis pitch (+ nose down) | 0.0° | +1.53° | +0.12° | −2.41° | No value |
| COM ahead of / below hip (% femur) | 41.9 / −2.1 | 45.4 / 11.2 | 43.5 / 6.6 | 45.6 / 7.7 | 30–45 / 22–50 |
| Knee ahead of vertical through CoP | −6 mm (to pad centre) | −18 mm | −19 mm | +3 mm | "Near the line of the vertical ground reaction force" (Bishop 2018, as an initial-pose choice; no tolerance given) |

Sources: `posture/out/posture_table.txt` and `verify/out/settle_statue_noise0.json`. The verifier's 3-seed re-roll of s60 matched to within 0.003 (`verify/out/posture_s60_compare.txt`).

**All three stances fit the qualitative band.** Each is subvertical and slightly flexed, with the knee within 2 cm of the force line and every joint within 5° of home. The rigid flat pad fixes the metatarsus slope and the MTP angle, so posture can change only above the ankle.

**None leans forward.** Settling moves the COM forward of the hip (41.9% → 45.4%), but the hip moves back over the foot: it ends 0.153–0.160 m behind the force line, against 0.140 m at home. s52 stands back, and s60 creeps backward within each episode, from +0.46 to +0.02 (`posture/out/posture_timecourse.json`).

The drift itself (s60 at 5M) was not rolled.

## The CoP statistic in the anatomical frame

| Point | Ahead of MTP III | Signed pad statistic | Fraction of footprint |
|---|---|---|---|
| Pad rear edge | −0.010 m | −1.00 | 0.00 |
| MTP II–IV joints: flamingo CoP ([Chang & Ting 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5454233)), track rear edge | 0 | −0.86 | 0.044 |
| s52 COM / whole-foot CoP | 0.024 m | −0.51 | 0.15 |
| Pad centre (27.5% of digit III) | 0.060 m | 0 | 0.307 |
| Statue COM / whole-foot CoP | 0.064 m | +0.05 | 0.32 |
| s60 COM / whole-foot CoP | 0.068 m | +0.11 | 0.34 |
| Front of the III-1 load zone (Sue / MOR 555) | 0.070 / 0.084 m | +0.14 / +0.34 | 0.35 / 0.41 |
| Human quiet stance, 45.7% of foot length ([Fujiwara et al. 2003](https://doi.org/10.2466/pms.2003.96.2.381)) | 0.094 m | +0.49 | 0.457 |
| **Gate bar 0.80** | 0.116 m | +0.80 | 0.553 |
| Drifted stance; rigid pad front edge | 0.130 m | +1.00 | 0.614 |
| Claw zone (figure estimate) | — | +1.45 to +2.26 | — |

Sources: `static.json` `frame_conversions`, `load_centre_frame.json` and `posture_table.txt`.

- **Every 6M stance sits inside the published load zone**, between the MTP joints and the front of III-1.
- **The gate reads the pad CoP, not the COM.** It takes the mean |pad CoP| ÷ 0.07 m (`stance_metrics._sole_metrics`). The digit load sits at the MTP, behind the pad CoP, so the pad CoP reads ahead of the COM projection:
  - noise-free statue +0.31 against +0.06
  - statue with spawn noise +0.19 against +0.05
  - s60 +0.27 against +0.11
  - s52 −0.41 against −0.51
- **The 0.80 bar sits 3–5 cm ahead of the III-1 zone**, near the end of phalanx III-2 on Sue's proportions (my arithmetic; inference). The rigid front edge is further forward still.
- **A rear stance fails the bar.** Over the MTP joints it reads 0.86 unsigned; s52 seed 3050 failed this way, at the heel edge (−1.00).
- A signed, whole-foot statistic would fix both problems. That is a gate change and needs the maintainer's approval.

## Candidate physics changes

Each item below is a **physics change — needs the maintainer's approval**.

1. **Lower the COM into the published band:** add thigh mass, cut forelimb mass, lower the trunk mass.
   - *Effect:* closer to the evidence; lower mgh raises the servo and spring ratios. The effect on drift is unknown (inference).
   - *Risk:* invalidates every checkpoint and the mgh constants (the XML's 704.4). The COM must stay 30–45% of femur ahead of the hip.
2. **Cap ankle extension at 180°**, as in Hutchinson 2005 Table 1.
   - *Effect:* none on quiet stance (136–140°); removes 11° of unpublished hyperextension.
   - *Risk:* low; may alter the walking pilot.
3. **Pull the rigid tipping edge back** to about 0.07–0.08 m ahead of MTP III (the III-1 zone), and make anything forward of it compliant or digit-borne.
   - *Effect:* if the policy follows the edge, as r10 showed, its stance would move back toward the track evidence (inference).
   - *Risk:* a smaller static margin; drift may move onto the digit hinges; retraining. Speculative.
4. **Make the rear contact a lateral digit IV MTP pad** at the same fore-aft station.
   - *Effect:* fidelity and a different lateral base; no fore-aft change expected.
   - *Risk:* roll dynamics change; retraining.
5. **Widen the digit spread** toward II–III 22–38° and III–IV 29–42°.
   - *Effect:* roll support only.
   - *Risk:* new contact layout; retraining.
6. **Let tail_1 hold level** (it needs at least 12 N·m more).
   - *Effect:* static fore-aft authority stays ≤5 mm, and lifting the tail raises the COM, which works against change 1.
   - *Risk:* low, as is the benefit.
7. **Adopt adult limb proportions.**
   - *Effect:* fidelity only (inference).
   - *Risk:* a large rebuild; low priority.

Not supported by this audit: changing the neck or head, which already give rearward authority, or extending rigid support forward.

## Corrections to the literature report

- **Model figures:** digit share is 24% settled, 13.5% at the keyframe instant (not "22–23% at home"). "+0.10" is the COM projection, not the gate value. 42%, "at hip height" and mgh ≈ 740 are keyframe-only. Normalise stiffness about the ankle pivot. The human analogue is +0.49.
- **Readings:** the shank is outside the published range; the per-limb convention is unresolved; Hutchinson 2005 Table 1 gives T. rex model joint ranges; Kambic's 162–176° is a ROM width.
- **Citations:** Persons & Currie **Table 2**. Bishop 2018's MTP reserve was a design allowance scaled from the chicken model. Gatesy 2009: "a narrow region of neither very columnar nor very flexed poses" (locomotion). Hutchinson 2004's hip limit is for running. The 86 cm track is Lockley & Hunt 1994 (via the notes). The Lallensack points are Thulborn 1990 and Farlow et al. 2000. The flamingo knee angle is dropped.

## Not measured or not obtained

- **Sources not reached:**
  - Brochu 2003's measurement tables (paywalled and restricted). The digitised figures, at about ±1–2 cm, cannot place the digit IV MTP joint.
  - Farlow et al. 2013, Lockley & Hunt 1994, and the full texts of Gatesy 2009 and Hutchinson 2004.
- **Not found:** numeric T. rex standing joint angles; standing pressure maps for any theropod or ratite.
- **Not measured:** drift-state posture and CoP; posture under action noise or other headings; transient balance authority. The verifier re-rolled 3 of 10 posture seeds.
