# Independent cross-check of the round-two picks

Three judges, each with one lens (behavior risk, hardware engineering, the owner's stated priorities), chose three bodies blind: they could read every record and screen but not the scorecard (`round2/SHORTLIST.md`) or this review. An adversarial comparator then checked each disagreement against the data, trying to refute both sides. Agents' conclusions, checked against the records; not measurements. Workflow script and raw results: session scratch, summarized here.

## Picks

| Who | Picks |
|---|---|
| Scorecard (`designlab/shortlist.py`, version 1) | `60-N-H-10`, `75-M-T-10-bighead`, `60-N-T-10` |
| Judge: behavior-risk | `75-M-T-10`, `75-M-H-10`, `60-M-T-10` |
| Judge: hardware | `60-M-T-10`, `60-M-H-10`, `75-M-T-10` |
| Judge: owner-priorities | `60-M-T-10`, `60-N-T-10`, `75-M-T-10` |
| **Comparator's recommendation** | `60-M-T-10`, `75-M-T-10`, `60-M-H-10` |

Every judge's `75-M-T-10` is the what-if record (`round2/whatif_fr05`, decision log F15), not the body as sized, and every judge adds the stronger ROBOTIS XM540-W270 neck servo, a combination that still has to be sized and screened.

## Comparator's summary

The scorecard proposed 60-N-H-10, 75-M-T-10-bighead and 60-N-T-10. All three judges instead picked 60-M-T-10 and 75-M-T-10, the latter rebuilt with the larger hip-pitch and thigh-twist motors from the "what-if" study (F15). For their third body they split three ways: 60-M-H-10, 75-M-H-10 (also with the what-if motors) or 60-N-T-10. No body was chosen by both the scorecard and at least two judges.

The scorecard's 75 cm pick, the larger-skull body, wins only because its heavier head selects the stronger XM540-W270 neck servo. The scorecard counts the base body's weaker neck four times (natural walking, turning, endurance, head and tail). The W270 costs the same, and with it the base 75-M-T-10 scores about 0.62, above the larger-skull body's 0.61. As built, the larger-skull body also cannot do the animal-like Froude 0.05 walk (hip pitch at 1.08 of its limit).

Its two 60 cm picks are both narrow-stance bodies. Their lead over the base body is 0.017 points or less, and it rests entirely on a lower friction need in turns. The scorecard never scores their 1.2–1.6 mm clearance in turns, though its own comment says it does. Two smaller flaws: the stress test is never used to rank, and it overstates the speed-limited cases.

Recommended set: 60-M-T-10, the what-if 75-M-T-10 fitted with the W270 neck (it needs one re-size, because that combination has not been screened), and 60-M-H-10. Comparing the two Tapered bases gives a clean 60 vs 75 cm comparison. Comparing the two 60 cm bases gives a clean Tapered vs Hybrid comparison, at the size where drive headroom (0.77–0.79 of limit) will not hide the difference between the bodies. If you would rather test stance width than body family, swap 60-N-T-10 in for 60-M-H-10.

## Disagreements, checked against the data

* `60-N-H-10` (scorecard only; scorecard's choice not defensible): Top total, but only by 0.0006 over 60-N-T-10 and 0.017 over 60-M-T-10 (0.714 / 0.713 / 0.697). The whole lead over the base body comes from one turning sub-score: friction need 0.53 against 0.65 gives clip(0.8-mu, 0.30) = 0.90 against 0.51. That is +0.024 weighted, partly offset by packaging (0.17 vs 0.48). Its turning gaps are 1.50 / 1.48 / 1.16 mm (base: 4.0 / 4.0 / 4.0) and it has 2 + 2 tight pairs in the walks (review_index clearance counts). The scorecard never scores either, even though the packaging comment says it does. The data supports the low friction need, the lower hip-roll hold (cont 0.608 vs 0.685) and Fr 0.05 at 0.959. No judge picked it, and the hardware judge ranks it 7th. A lead this small, on a body whose unscored risk is the tightest packaging at 60 cm, does not justify making it the only 60 cm base-family representative.
* `75-M-T-10-bighead` (scorecard only; scorecard's choice not defensible): It is the best 75 cm body only because the larger head selects the XM540-W270 neck. The scorecard counts the base body's W150 neck four times. On 75-M-T-10 the neck (0.966) is the sizing worst ratio used in natural walking. It is also the turning worst (review_index turning.worst_targets = neck cont, in_place_0.50, 0.9695), the session hottest drive used in endurance (0.9658) and the head-and-tail ratio. Bighead's gain is endurance 0.78 vs 0.42 and head and tail 0.82 vs 0.45. The W270 costs the same in the ledger (expression_actuators $504.89 on both records). Rescored with the scorecard's own code and a W270 neck, 75-M-T-10 as sized comes to about 0.62, above bighead's 0.606 (scratchpad comparator/neck_cf.py). The body itself is worse: tightest joint margin 0.1505 rad at the knee against 0.176 rad on the base. Its hip pitch is 1.079 in the Fr 0.05 walk, so as sized it cannot do the animal-like walk. Verifying it would mostly measure the 24 V RobStride 03 speed limit.
* `60-N-T-10` (scorecard only; scorecard's choice defensible): The owner-priorities judge also picked it. Its claims check out: Fr 0.05 knee speed 0.938 (second best after 60-M-H-10-compact at 0.911) and walking-turn friction 0.541 against 0.646. Hip-roll cont is 0.607 vs 0.685 (-11 %), and reference sway is 2 x 0.0333 / 0.4047 = 0.164 H against 0.193 H. Runtime is 47.7 min. The risks are recorded: turning gaps 1.56 / 1.55 / 1.23 mm, 4 tight pairs in the nominal walk, and a tightest joint margin of 0.135 rad. As a stance-width probe it is a legitimate pick. Alongside 60-N-H-10, though, both 60 cm slots go to the tightest-packaged bodies, and the size comparison with the 75 cm pick changes stance and skull at the same time.
* `60-M-T-10` (judges only; scorecard's choice not defensible): All three judges picked it, and every claim checked against the record. Thigh twist is 0.771 (3.52 of 4.56 N·m), knee speed 0.771, ankle 0.763 and neck 0.761. The stressed sizing value is 0.998. Fr 0.05 knee speed is 0.960. It has 4.0 mm in all three turns, 0 tight pairs and the best hip-roll margin of the 20 (0.196 rad). The scorecard ranks it 4th at 0.697, 0.017 behind the leader, because its friction need is 0.646 against 0.53. Leaving out the cleanest 60 cm base body for near-tied narrow twins, on unscored clearance, is not defensible. It is also the natural anchor for both the size and the family comparisons.
* `75-M-T-10` (judges only; scorecard's choice not defensible): All three judges mean the F15 what-if record (round2/whatif_fr05/candidates/75-M-T-10.json), not the scored as-sized body, which the scorecard was right not to pick on Fr 0.05 (hip pitch 1.080). The what-if passes every gate (shortlist.gates all True). Its Fr 0.05 worst is the neck at 0.976, with knee speed 0.949. Thigh twist falls to 0.777, mass to 14.56 kg, runtime to 36.6 min, and cost rises to $18.2k / $25.2k. The record confirms all of this. The scorecard never sees this record, and if it did it would still score it 0.542, because the W150 neck again sets the Fr 0.05, sizing, turning and session terms. Every judge also adds the W270 neck; that combination is unscreened and needs a re-size. Hip roll stays at 0.956 in the what-if.
* `60-M-H-10` (judges only; scorecard's choice defensible): Hardware judge only. Its numbers are correct: thigh twist 0.781 (stressed 1.011), knee speed 0.786, Fr 0.05 0.983, 3.39 mm in the 0.5 rad/s walking turn and a hip-roll margin of 0.148. The scorecard ranks it 5th (0.681) and already had a Hybrid in 60-N-H-10, so omitting it is defensible under its own rule. It is still the right Hybrid for a clean family comparison, because it differs from 60-M-T-10 only in family and both sit at 0.77-0.79 drive ratios.
* `75-M-H-10` (judges only; scorecard's choice defensible): Behavior judge only, with the what-if drives. The record supports the claims: Fr 0.05 knee 0.974, hip roll 0.956, neck 0.961, turning gaps 4.0 / 3.9 / 3.52 mm and hip-roll margin 0.170. As sized it ranks last (0.551) and the what-if scores 0.554, so the scorecard is defensible here. A 75 cm family comparison puts two slots on the class with 0.96 drives and $25.3k high-side cost. The wider what-if hips also partly hide the Hybrid pelvis's hip-roll penalty, which a 60 cm comparison would show.

## Errors found

* Scorecard logic: the neck drive enters four weighted categories. designlab/shortlist.py _worst_margin covers neck and tail, so on 75-M-T-10 the W150 neck (0.966) is the natural-walking sizing worst ratio. It is also the turning worst (review_index turning.worst_targets {group: neck, metric: cont, case: in_place_0.50} = 0.9695), the endurance session worst (envelope session_worst neck 0.9658) and the head-and-tail ratio. On the what-if record it is also the Fr 0.05 worst (0.976). This single part choice decides the 75 cm pick, and the W270 has the same ledger cost (expression_actuators 504.89 on both 75-M-T-10 and 75-M-T-10-bighead).
* Scorecard logic: the stress test is reported but never used. build() sorts by total only, while docstring step 4 says 'a candidate whose lead depends on optimistic numbers drops'. It also multiplies speed ratios by 1.10/0.85: the Fr 0.05 worst is knee speed at 60 cm and hip-pitch speed at 75 cm, and mass does not change speed demand. 60-M-T-10's stressed 1.24 should be about 0.96/0.85 = 1.13 on that term, and 75 cm 1.40 should be about 1.27. ROUND2_REVIEW.md §6 ('every body exceeds a limit, 1.21-1.43') inherits the inflated range.
* Scorecard logic: the packaging comment says 'tightest joint-range margin and closest near-contact while walking', but the code uses only joint_range_margins. Turning min gaps (60-N-H-10 1.50/1.48/1.16 mm, 60-N-T-10 1.56/1.55/1.23 mm, 75-N-T-10 0.77 mm in place) and tight-pair counts (60-N-T-10: 4 nominal; 60-N-H-10: 2 + 2) are scored nowhere. That is where the narrow picks carry their risk.
* Scorecard logic: the nominal CoP sub-score clip(cop+10, 30) is 1.0 on all 20 bodies (lowest min_cop_margin_single is 29.2 mm on 60-M-H-10-compact), so one third of the top-weighted natural-walking category is a constant that dilutes the drive-headroom terms.
* Scorecard logic: the Fr 0.05 animal-like walk is not gated. A body whose drives cannot produce it (75-M-T-10-bighead, hip pitch 1.079) just gets 0 on one sub-score and can still be picked as the 75 cm representative.
* Scorecard input: the what-if records (round2/whatif_fr05, F15), which all three judges treat as the 75 cm body to verify, are outside candidate_ids(). The scorecard scores them only by hand (DECISION_LOG F15: 0.54-0.55), and the neck double-counting means it cannot show their Fr 0.05 improvement (the what-if 75-M-T-10 scores 0.542, below the as-sized 0.555).
* Behavior-risk judge: '75-M-T-00 ... The what-if drive change was run only on 75-M-T-10' is wrong. round2/whatif_fr05/candidates/75-M-H-10.json exists (DECISION_LOG F15), and the judge's own second pick relies on it.
* Owner-priorities judge: '75-N-T-10 ... friction (0.37) are the study's best' is slightly wrong. 75-N-H-10 has a lower turning friction need (mu_needed_max 0.366 against 0.374).
* All three judges: the W270 neck they add to the what-if 75-M-T-10 is unscreened on that body. The record keeps the W150 (neck cont 0.966, Fr 0.05 0.976), so the combination needs a re-size before verification. The hardware judge's 0.665 estimate (1.178 / 1.7702 N·m) is correct arithmetic but is not a screen result.

## Each judge's view on size and family

### behavior-risk

None of the 20 fails a hard requirement on the screens as run (clearance, target turns, torso at home, runtime, carpet gate), so every choice below is on margin.

Size: keep both, but they are not equally ready.

60 cm is the robust class. On its base, narrow and level bodies the governing drive sits at 0.77-0.79 of its planning limit, the combined stress ends at 0.99-1.02, and the Froude 0.05 walk passes at 0.94-0.99. The hottest drive over a session is the neck at 0.76. These bodies weigh 10.8-11.0 kg, run 46-48 min and cost $14.8-14.9k nominal ($20.6-20.7k high).

75 cm is under-driven as sized:
- AK45-36 thigh twist 0.96-0.98 and hip roll 0.92-0.93 on the M bodies
- W150 neck 0.96-0.97
- RobStride 03 at 1.07-1.08 in the Froude 0.05 walk
- 1.24-1.27 under stress

But 75 cm has the better carpet clearance (10.1-11.2 vs 5.3-6.1 mm over high pile), turning friction (0.37-0.43 vs 0.53-0.65), battery service (dorsal and ventral vs ventral only) and camera height (0.726 vs 0.572 m). Its shortfalls are part choices: F15's X6-60 hip pitch and X4-36-L thigh twist pass every gate for +$0.8k, 0.34 kg less and 1.6 min less runtime, and the W270 neck costs nothing in the ledger. So do not drop 75 cm, but do not verify it on its round-two drive set either. Otherwise verification will measure a 24 V speed limit and an undersized neck rather than the body.

Runtime does not separate the bodies: walking power could rise 1.6-2.4x before 20 min is lost. High-side cost does ($20.5-20.8k vs $24.4-24.8k, $25.2k re-driven).

Family: Tapered beats Hybrid on every hardware measure at both sizes, by small margins:
- hip-pitch hold 3.20 vs 3.52 N·m at 60 cm, 8.40 vs 8.91 N·m at 75 cm
- outward hip roll +0.308 vs +0.260 rad at 60 cm, +0.262 vs +0.215 rad at 75 cm
- turning roll margins 0.121-0.143 vs 0.072-0.095 rad at 60 cm, 0.075-0.104 vs 0.028-0.057 rad at 75 cm
- 0.11-0.13 kg lighter and $0.1k cheaper

None of this disqualifies Hybrid. Its roll limit is the pelvis lateral cover meeting the thigh shell, which hip spacing fixes: F15's 11.5 mm wider hips give the 75 cm Hybrid +0.262 rad. So keep both families, but compare them at 60 cm, where drive margins of 0.77-0.79 will not confound the body difference, and leave the 75 cm Hybrid out of this round.

Variants:
- Compact: drop. Hip-pitch hold +24-40 %, the knee drive over its limit in the Froude 0.05 walk on three of four, a 0.958 knee on the fourth, and ankle drives moved at 75 cm.
- Narrow: trades 10-11 % less hip-roll load and lower 60 cm turning friction (0.53-0.54) for 0.8-1.6 mm turning gaps and packaging-search changes. Acceptable at 60 cm only.
- Level: the same bodies as the 10° bases (P15), with the camera 0.065-0.081 m lower. On hardware they are interchangeable with their 10° twins; I rank them behind 75-M-T-10 only because they would duplicate bodies already in the set.
- Bighead: neutral at 60 cm Tapered (neck 0.86), negative elsewhere. Its one useful lesson is that the 75 cm bighead's W270 neck is the fix for the base neck.

My calculations are in /tmp/claude-0/-home-user-mesozoic-labs/c17e65d8-8171-548a-aef1-2a557e9379dd/scratchpad/r2/xc/ (stress tables: analyze.py; packaging tables: clear.py; camera check: cam.py).

Key risks:

* No body fails a hard requirement, so every exclusion and ranking here is on margin, and margins are what a controller and real hardware use up. All 20 pass the round-two clearance gates (TABLES §6), the target turns with every drive inside its planning limits (§8; worst 0.984), the torso-at-home check (§7), runtime (37.7-47.7 min, §1) and the carpet gate (§9). Every screen is a prescribed motion. The Froude 0.05 walk (stride 0.80 H, lift 0.11 H) was never run through the clearance screen, which covers only home, nominal and brisk walks. The 60 cm knee is usable only to 0.81 rad and already uses 0.58 rad in the sizing walks.
* 24 V caps the speed available for natural walking. The CubeMars AK45-36 on hip roll and thigh twist accepts only 15-28 V (E9), which holds every body on the 6S bus; drives rated above 24 V therefore run slowed (the RobStride 03 is 48 V-nominal, F2, and gets 6.13 rad/s here). The Froude 0.05 walk needs 0.94-0.98 of the X4-36-L knee speed at 60 cm and 1.07-1.08 of the RobStride 03 at 75 cm; the F15 re-size moves the 75 cm limit to the knee at 0.95-0.97. The records evaluate speed at the 24.0 V bus, but A11 puts the pack at 21.0 V at the end of usable charge. If speed falls with voltage as E9 states, that costs another 12.5 %, enough to put every body over in the Froude 0.05 walk late in a session. Moving to 12S would require replacing the AK45-36s.
* Assumption A1 (sustained torque = 0.57 x rated) is least supported exactly where the 75 cm margins are thinnest. CubeMars publishes no thermal definition, yet the AK45-36 governs the 75 cm thigh twist (one-leg hold 4.39 of 4.56 N·m, 0.96) and hip roll (0.93), and the 60 cm thigh twist (0.77). If the true fraction is 0.50 (the low end of MyActuator's 50-80 % equilibrium range), 75 cm goes to 1.10 and 1.05 and 60 cm to 0.88. The 75 cm figures come from the one-leg hold rule: the thigh-twist walking RMS is 3.59 N·m (0.79) and the session ratio 0.69. A bench test of the AK45-36 at one-leg-hold duty therefore decides whether 75 cm needs bigger hip drives.
* Stress case (drives 15 % weaker, robot 10 % heavier; torque x1.294, speed /0.85). All ten 75 cm bodies end at 1.24-1.27, with two to four drive groups over. The six 60 cm base, narrow and level bodies end at 0.99-1.02, with at most the AK45-36 thigh twist over, by up to 2 %. The 60 cm bighead bodies reach 1.11-1.12 and the compact bodies 1.16-1.24 (neck or knee). On speed, the 75 cm RobStride 03 reaches 1.00 in the brisk walk, and every body goes over 1.0 in the Froude 0.05 walk. Re-driven per F15 and given a W270 neck, 75-M-T-10 still has the AK45-36 hip roll at 0.956 (1.24), the STS3215 tail peak at 0.895 (1.16) and the knee speed at 0.949 (1.12).
* Head and tail servos at 75 cm. The XM540-W150 neck runs at 0.96-0.97 of its sustained torque over a whole session, almost all of it holding the head against gravity (hold 1.165 of 1.178 N·m RMS). The STS3215 tail peak is at 0.77 (0.87 in the Froude 0.05 walk). The sizing walks move the head ±0.05 rad and the tail ±0.12 rad. At the spec's upper end (±0.15 / ±0.2 rad), my linear extrapolation of the motion part of the torque puts the W150 at about 1.05 RMS and 1.09 peak, and the tail at about 1.16 peak. The same estimate keeps the 60 cm STS3250 neck at or below 0.85 and the STS3215 tail at or below 0.70. The W270 brings the 75 cm neck back to about 0.67.
* 60 cm on carpet and in turns. In the 0.35 rad/s walking turn, single support needs friction 0.63-0.65 on the M bodies, above the carpet band's 0.54 lower edge (F11, still open). The cause is the stance pad's yaw moment against a 35 mm torsional radius (44 mm at 75 cm). Toe clearance over high pile in the nominal walk is 5.3-6.1 mm, and 0.3-1.2 mm if the pile fully compresses (A16); a toe-down of only 4.1-4.6 deg reaches the pile top. At 75 cm the same figures are 10.1-11.2 mm, 5.2-6.2 mm and 5.8-6.0 deg. Servo-held, the 60 cm bodies stand only on low pile (softest pile that holds: 5.6-6.7 mm).
* Hip packaging margins are small, and the clearance screen measures mid-surface proxies, so the real outer skins sit 0.8-1.5 mm closer than reported. On every Tapered M body the closest approach is the fixed 4.0 mm centreline hip-bracket gap. The Hybrid pelvis lateral cover limits outward hip roll to +0.260 rad (60 cm) and +0.215 rad (75 cm), against +0.308 and +0.262 rad for Tapered; that leaves only 0.028-0.057 rad of hip-roll margin in the 75 cm Hybrid turns. The narrow stances run at 0.8-1.6 mm in turns. A learned controller's turns will deviate from the prescribed ones.
* Thigh-twist joint range at 75 cm. Turning in place at 0.5 rad/s uses 0.78 of the ±0.30 rad range, leaving 0.066 rad (0.045 rad in the 0.5 rad/s walking turn). At 60 cm it uses 0.67 and leaves 0.098-0.099 rad. This is a hard joint limit, so a controller turning in place has little slack on any 75 cm body.
* Service access and cost. On every 60 cm body the battery comes out only through a ventral hatch, with 2.35-2.36 mm path clearance; the dorsal path is blocked by the tail motor (-4.35 mm). Swaps therefore need the 11 kg robot lifted or on a stand, while 75 cm clears both dorsally and ventrally (2.54 mm). Latches, connectors and hand access are not modelled. The high-side cost is over the $20k ceiling everywhere: $20.5-20.8k at 60 cm, $24.4-24.8k at 75 cm and $25.2-25.3k for the F15 re-size. These lines are scaled from PR #662 (A8), not quoted.
* Person-following with the 60 cm head camera. The camera sits at 0.572 m (0.507 m on the level bodies) against 0.726 m at 75 cm. At the ±0.35 rad neck limits the screen gives torso visibility of 0.08 nose-down and 0.54-0.62 nose-up at 60 cm, against 0.23 and 0.69-0.77 at 75 cm. My rotation-only check, which reproduces the screen's 0.077 at -0.35 rad on 60-M-T-10, gives about -0.11 rad of nose-down pitch at 60 cm (-0.08 rad level) before a person at 1.5 m drops below 75 % torso visibility. At 75 cm the figure is about -0.2 rad (slightly optimistic there). The spec allows head motion up to ±0.15 rad. Tilting the 60 cm camera further up is cheap and should be set before verification.

### hardware

Keep both sizes in verification. 75 cm is the size more likely to reach natural walking on carpet, but only with the Fr 0.05 (what-if) drive set; do not verify the 75 cm bodies as sized, because their hip pitch makes the animal-like walk impossible (1.07–1.08).

The 60 cm handicaps are geometric. The hip hardware spacing is the same at both sizes (18.8 vs 18.7 cm), so the 60 cm stance is wider relative to height (feet 0.42 H vs 0.33 H) and stands on smaller pads (effective torsional radius 35 vs 43.7 mm). That makes its friction need with the yaw moment about 1.5 times the 75 cm value at every speed (Fr 0.05: 1.26 vs 0.84). Carpet piles do not scale with the robot, so its clearance over high pile is 6.1 mm against 11.2 mm. The 75 cm handicaps are drive selections, which the what-if shows can be fixed.

Do not drop 60 cm yet. It has the most drive headroom (≤0.77), the longest runtime (46 min) and the lowest cost ($14.8k). Verifying 60-M-T-10 answers directly whether a controller can manage the yaw moment on the stance pad. If it cannot, the 60 cm class needs larger pads or narrower hip hardware rather than more controller work. The narrow stance cuts the friction need by about 15 % but needs a few millimetres more clearance than the search's 0.8–1.6 mm.

Tapered is at least as good as Hybrid on every behavior margin as sized. Outward hip roll usable: 0.308 vs 0.260 rad at 60 cm, 0.262 vs 0.215 at 75 cm. CoM ahead of the hip: 0.087 vs 0.095 H at 60 cm, 0.127 vs 0.134 H at 75 cm. Knee contact with the carpet lift at 60 cm: −0.71 vs −1.87 mm. Hybrid's only edge is a 2–3 % lower friction need. In the what-if build the two families converge (hip roll 0.262 rad on both; knee 1.00 vs 0.95 rad; Fr 0.05 0.976 vs 0.974), so compare the families at 75 cm. If one family has to go, drop the 60 cm Hybrid.

Among the variants, drop compact: three of four fail Fr 0.05 (knee 1.08–1.11), and the Hybrid compact's knee runs at 0.96. Drop the 75 cm bighead for its knee range. The level posture is behaviorally the same body within ±2 % and does not need its own verification slot.

Key risks:

* No body fails a hard requirement. All 20 pass clearance in both walks and the target turns (closest approach 0.77–4.0 mm). The worst drive ratio in the sizing walks and target turns is 0.98. A person's torso is visible at home on every body. Runtime is 37.7–47.7 min. Every body passes the carpet gate. The disqualifiers above are behavior risks, not hard-requirement failures.
* Stance-foot spin is the friction limit that binds. The carpet gate checks only the sideways force ratio (0.21–0.28). The turning screen's single-support requirement also counts the yaw moment the stance pad carries, and it sets the need. 60 cm: 0.47–0.57 nominal, 0.73–0.88 brisk, 0.53–0.65 in the walking turn, and in my SHA-matched rebuild 1.05–1.27 at Fr 0.05. 75 cm: 0.33–0.38, 0.51–0.59, 0.37–0.43 and 0.72–0.88. Available friction: carpet lower edge 0.54, robot foot on carpet 0.82, robot foot on hard floor 0.68 at the low end, compiled model 0.9. The prescribed walk holds the trunk heading fixed, so the controller must counter-yaw through the stance thigh twist. That drive is at 0.96–0.98 on the 75 cm bodies as sized and 0.78 with the what-if drives.
* As sized, every 75 cm body fails the Fr 0.05 walk on hip-pitch speed (1.07–1.08) or knee (1.08–1.11 on the compact bodies). Only the what-if drive set clears it (0.97–0.98), and that set leaves the hip roll at 0.956 and the neck at 0.961–0.966.
* Carpet swing height conflicts with knee range. None of the bodies meets the 0.06 H target with the reference lift. With the lift that target needs (0.176–0.193 H), the thigh fairing touches the calf fairing in the nominal walk on 17 of the 20 bodies as sized (−0.07 to −5.8 mm) and at Fr 0.05 on 18. Only the two 75 cm compact bodies (ankle drives at 0.42 of the shank) and both what-if bodies (knee usable 0.95–1.00 rad, against 0.71–0.81) stay clear.
* Fr 0.05 combined with the carpet lift exceeds a drive on every body: knee speed 1.15–1.17 on the what-if bodies and 1.22–1.36 at 60 cm, hip pitch 1.24 on the 75 cm bodies as sized. On carpet the controller must trade walking speed against toe clearance.
* Thin margins for head and tail motion and for following at 75 cm. The XM540-W150 neck runs at 0.96–0.97 of its sustained torque over a session, while following needs the head held near level. The tail servo's peak reaches 0.77 in sizing and 0.87–0.91 at Fr 0.05 with the tail at ±0.12 rad, the bottom of the spec's ±0.1–0.2 rad band (my linear estimate exceeds the limit at 0.2 rad). At 60 cm the neck is at 0.76 and the tail peak at 0.45.
* No body can stop on one foot at the spec's 1.5 m/s²: the limit is 1.19–1.27 m/s² on all 20. Stops need both feet down or a step.
* Compliant stance: servo-held with no balance control, only 75-M-T-10, 75-N-T-10, 75-M-T-00, 75-M-H-00 and both what-if bodies stand on medium pile, and none stands on high pile. Active ankle balance is required, and at 60 cm the ankle drives already run at 0.74–0.77 of sustained (0.50–0.58 at 75 cm).
* The narrow stances clear the target turns by 0.77–1.6 mm. The 75 cm gaps are toe against toe; the 60 cm gaps are pelvis plate against thigh shell, measured on mid-surface proxies whose outer skin sits closer still. Any deviation by a controller makes contact. The 0.10 rad joint-margin gate covers only the straight walks: in the walking turn the 75 cm Hybrid bodies as sized keep 0.054–0.057 rad of hip roll.
* The one-step push bound is about 0.24 √(gH) on every body, so the 75 cm bodies tolerate larger absolute bumps: 7.6–8.2 N·s against 4.9–6.4 N·s at 60 cm, under ideal balance control. My checks are in the scratchpad: /tmp/claude-0/-home-user-mesozoic-labs/c17e65d8-8171-548a-aef1-2a557e9379dd/scratchpad/r2/xc/behavior_risk/fr05_all.json and /tmp/claude-0/-home-user-mesozoic-labs/c17e65d8-8171-548a-aef1-2a557e9379dd/scratchpad/r2/xc/behavior_risk/carpet_lift_all.json (scripts fr05_check.py and carpet_lift_check.py in the same folder).

### owner-priorities

Size: keep both. They fail in different places, so verification needs one of each.

60 cm wins on natural walking, head and tail motion, runtime and budget:
- Leg-drive walking RMS is at most 0.62 of sustained.
- Target turns stay at or below 0.78 on the base and narrow bodies.
- Froude 0.05 fits within the as-sized drives on 8 of 10 bodies.
- Neck 0.76, tail at most 0.47.
- Runtime 42-48 min; cost $14.8-15.0k nominal.

75 cm wins on carpet, turning on carpet and head-camera following:
- Toe margin over high pile 10.1-11.2 mm vs 5.3-6.1 mm.
- Four 75 cm bodies stand servo-held on medium pile; no 60 cm body does.
- Softest rigid-holding pile 7.7-9.4 mm vs 5.6-6.7 mm.
- Turning friction need 0.37-0.43 vs 0.53-0.65.
- Nose-down window 0.175 vs 0.10 rad; 0.68-0.72 vs 0.56 of a person visible at 1 m.

But as sized, no 75 cm body can do the Froude 0.05 walk (hip pitch 1.07-1.08; compact knee 1.08-1.11). The what-if fixes that for +$0.8k, 1.5-1.6 min less runtime, 0.34 kg less mass and 10 mm more width. The 75 cm neck should also get the W270.

Family: Tapered is equal to or better than Hybrid on almost every behaviour number at both sizes, but only by small margins, and both use identical drive sets. Hybrid's only advantages are 0.4-0.6 points less camera self-occlusion, 0.01-0.02 lower turning friction and a nearer floor view (2.27 vs 2.39 m, 2.40 vs 2.59 m). Against that it carries +0.11-0.13 kg, +10 % / +6 % hip-pitch hold, 0.05 rad less hip-roll range margin, and at 75 cm as sized no medium-pile stance. I would not drop Hybrid from the program: the family is a looks decision for the owner. But I would not spend one of three verification slots on it, because an equal-budget comparison cannot resolve differences this small. Stance width and size are the real alternatives. If the owner picks Hybrid on looks, swap in 60-M-H-10 and the what-if 75-M-H-10 (it also stands on medium pile, softest pile 9.06 mm).

Variants:
- Narrow stance is the only variant with a behaviour upside, but at 75 cm it conflicts with the Froude 0.05 drive set.
- Compact is the worst on every priority (hip-pitch hold +24-40 %, Froude 0.05 fails on 3 of 4).
- Larger skull costs 60 cm neck headroom (0.85-0.86 vs 0.76) but brings the W270 neck at 75 cm.
- Level posture keeps loads within 1 % but narrows the head-camera window, and cuts 60-M-H-00's ToF foot-path coverage to 56 %.

The ranks for 75-M-T-10 and 75-M-H-10 assume the whatif_fr05 drives. As sized, they would sit with 75-M-T-00.

Key risks:

* Carpet is the 60 cm class's weak point, and both 60 cm picks share it. No 60 cm body stands servo-held on medium pile. A rigid stance holds only on piles compressing less than 5.6-6.7 mm at 98 kPa, against the spec's 8-15 mm carpet pile. Toe margin over high pile is 5.3-6.1 mm. The 15 mm carpet swing target needs a 0.19 H lift (reference 0.10-0.11 H), on a knee that already sets the Froude 0.05 walk on speed (0.96 / 0.94). The ankle one-leg hold is 0.76-0.77 of sustained (0.49-0.51 at 75 cm), so active ankle balance on pile has less room.
* Turning on carpet at 60 cm: the 0.35 rad/s walking turn needs single-support friction 0.65 (60-M-T-10) and 0.54 (60-N-T-10), against the carpet band's 0.54 lower edge. The need comes from the stance pad's yaw moment (2.43 N·m on a 35 mm torsional radius), not from sliding (F_t/F_n 0.21-0.24). At 75 cm the need is 0.37-0.46.
* Head-camera following: the sensor screen tests only home and ±0.35 rad of neck pitch. I ran sensors.py's own person check at 0.025 rad steps (script /tmp/claude-0/-home-user-mesozoic-labs/c17e65d8-8171-548a-aef1-2a557e9379dd/scratchpad/r2/xc/neckvis.py; it reproduces the screen at 0 and ±0.35 rad). On the 60 cm 10-deg bodies the torso at 1.5-3 m stays at least 75 % visible only from -0.10 to +0.20 rad straight ahead, and from -0.05 (Tapered) or -0.025 rad (Hybrid) to +0.20 rad at ±30 deg. At 75 cm the nose-down limits are -0.175 and -0.125 rad. So the spec's ±0.05-0.15 rad head motion fits at 60 cm only if biased nose-up. The level postures lose more: at 60 cm they have no nose-down margin at ±30 deg.
* Head and tail headroom at 75 cm: the XM540-W150 neck runs at 0.96-0.97 of sustained in the walks, the target turns and a following session, with only the ±0.05 rad head motion the reference prescribes (spec allows up to ±0.15). The W270 chosen on the 75 cm bighead and compact bodies runs at 0.76-0.80. The tail servo is at 0.77 of peak as sized and 0.90 in the what-if, with ±0.12 rad of tail yaw (spec up to ±0.2).
* The what-if drive set has thin margins of its own: hip-roll hold rises from 0.93 to 0.96 because the hips are 11.5 mm wider, and the Froude 0.05 walk sits at knee speed 0.95 and neck 0.98. Every screen is a necessary-condition check on prescribed motion, using A1's unverified 0.57 x rated sustained torque. That matters most for drives at 0.96-0.98.
* Narrow-stance packaging (60-N-T-10): 1.2-1.6 mm pelvis-plate to thigh-shell clearance in the turns and 2.0 mm walking straight, measured to the shell's mid-surface. The real skin gap is about half a wall less (walls 1.6-1.8 mm at 60 cm per A4), which is also close to outsourced-part tolerances.
* Reference sway exceeds the spec on every body: the CoM sways 0.13-0.20 H peak-to-peak (2 x sway_amp, nominal walk) against 0.12 H. The controller must cut it by about 27 % on 60-N-T-10, 26 % on the what-if 75-M-T-10 and 38 % on 60-M-T-10.
* Budget: high-side cost exceeds the owner's $20k ceiling on every body. Both 60 cm picks are $20.6k (3 % over, $14.8k nominal). The what-if 75-M-T-10 is $25.2k (26 % over, $18.2k nominal). Under P8 this is a note, not a gate.
* Stopping: one foot holds only 1.19-1.27 m/s² of deceleration on every body, against the spec's 1.5 m/s², so stops at the spec limit need a step or both feet down.

