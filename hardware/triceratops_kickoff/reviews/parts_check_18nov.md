Verdict: sound after fixes. The sizing method and arithmetic hold up. My own script, written from scratch (/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/parts_check/check_sizing.py), reproduces every peak and heat figure within 8 %, and the list is always the conservative side, mainly because it applies the fore-aft force at floor height instead of at the ankle pin. At 20 kg, 55 % front, my figures are: walk 31.3 N·m (list 32.1), trot 34.9 (35.8), pitch walk 17.9 (18.6), roll 25.4 (25.4), paw hold 16.2 (16.2).

Every BOM line total and category sum is right as written (check_bom.py), and the mass budget closes (14.61 / 18.83 kg listed, 15.00 / 20.00 kg total).

The faults are one speed claim, interface mismatches, missing parts and stale prices. The corrected document is at /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/parts_check/PARTS_PREMIUM_corrected.md, built by apply_fixes.py and fix_bom.py. The author's PARTS_PREMIUM.md is left unchanged.

F1. X4 speed on 12S
- The list's 12S argument assumed a 24 V X4 at 42 V reaches about 194 rpm no-load. Resellers quoting MyActuator (search, 18 Nov 2026) say a drive run above its rated voltage must not exceed its rated speed, which is 83 rpm for the X4-36.
- Pitch then uses 0.70–0.72 of that cap at 0.30 m/s (0.83 with PR #669's own swing profile), 0.91–0.93 at 0.40 m/s, and 0.52–0.55 in the 0.60 m/s trot. T15 roll uses 0.36.
- The 0.40 m/s walk is not credible with X4 pitch drives on either 6S or 12S. 12S is still needed, but for the 48 V X6 and X8, not the X4.
- The same caveat applies to the T. rex if it moves to 12S.

F2. Pushrod force
- The list's pulls assume a 90° transmission angle, but the list itself allows angles down to 40°.
- At 40°, on the list's own torques: T20 1.59 kN design and 1.82 kN in the pad-edge abuse case; T15 1.16 and 1.33 kN. My slightly lower torques give 1.55 / 1.79 and 1.13 / 1.30 kN.
- Use steel rod ends rated at least 2× these figures, not polymer ones, and size the crank pins and output bearings the same way.

F3. Wireless stop
- The $1,900 Hoosier Crane listing for the Cattron Safe-D-Stop (PL d confirmed) does not show that a DC machine-side receiver is included. Cattron's unit for that job is the XBMCU (9–36 V DC, Cat 3 / PL d inputs).
- Added a receiver line at $0 / $800 / $1,500. This is an estimate; get a quote. Its mass is not published and comes out of the mass reserve.
- The whole chain is not PL d as built: the custom watchdog board is not a safety-rated part. Reworded.

F4. E-stop buttons
- The XB4BS8445 has 1 NO + 1 NC contacts, which cannot feed the Pilz relay's two input channels. Changed to the XB4BS8444 (2 NC). Its price is assumed equal and was not checked.

F5. Foot load cells
- The FX29K0 is an I2C part (2.7–5.5 V). Four of them on one I2C bus down the legs, beside 48 V motor cables, will not work. Added a differential I2C link per leg (PCA9615 class, $80 estimate), or use a small CAN node at each foot.
- Distributor lead time is 21–23 weeks. The 2.5× overload figure was found only for the 25 lbf sibling, and the I2C address is unconfirmed.

F6. Eyelid servos
- The XL330-M288-T runs on 3.7–6 V (5 V recommended) over TTL, so it cannot share the 12 V RS-485 servo bus. Added a 5 V rail ($20 estimate) and routed the servos to the U2D2's TTL port.
- Price is $27.49 (robotis.us, sold out), not $24.

F7. ROBOTIS prices
- XM540-W270-R and XM540-W150-R: $494.39, not $482.89 (the list assumed the -R versions cost the same as -T).
- XM430-W350-R: $333.39, not $310.39.
- robotis.us shows the W270-R and the W350-R sold out, the W270-R with a long lead time.

F8. Jetson prices
- NVIDIA raised the Orin Nano Super kit from $249 to $399 in July 2026 (cnx-software, 22 Jul 2026), and kits went out of stock. The low case is now $399.
- The Orin NX high case is now $1,364.29 (RobotShop reComputer J4012; the listing is undated and its currency unconfirmed).

F9. Small price errors
- Pololu D42V110F12: $59.95, not $69.95 (currently rationed).
- PCAN-USB FD: $379 at DigiKey plus $15 shipping, not $368.

F10. BMS
- The Orion Jr 2 at $440 is confirmed, but its current sensor is a separate option that Orion strongly recommends. Added it at $100 (estimate).
- Its charge and discharge outputs are logic signals, not contactor drivers. Route discharge-enable into the watchdog so a BMS trip runs the same sit-then-cut sequence.

F11. Missing for a museum build
- Cooling (fans, ducts, filters): $90.
- Tools: $350.
- Padded road case for travel: $800.
- All three are estimates.

F12. Totals and wording
- T20: $22.9k / $37.3k / $66.4k landed (was $21.9k / $34.5k / $60.3k), or $22.3k / $31.8k / $51.2k measured the T. rex way.
- T15: $20.0k / $32.8k / $58.5k landed, or $19.4k / $28.3k / $46.3k the T. rex way.
- On the landed basis the T20 nominal is above the brief's $17k–$32k range. Measured the T. rex way it sits at the top.
- With the listed savings (bench gear charged to the T. rex, team-painted shells, receiver confirmed as included, road case deferred, US-landed drives) T20 comes to about $29.3k landed.
- The single-leg bench set is $7.1k of parts, $8.9k landed. The list said $6.3k / $7.6k, and its $7.6k did not follow its own landing rules.
- The 73 % "duty" case is really a reseller's price, which includes the reseller's margin. Relabelled.
- "Quasi-direct-drive" is wrong for 19.6:1 and 36:1 planetary drives. Reworded.

F13. T. rex timing
- Buying T. rex hardware only after the Q3 2027 demo leaves one or two quarters before the end-2027 / early-2028 milestone, and these drives are already backordered.
- If that milestone is a walking T. rex, resume its simulation in Q1 2027 and order its drives in Q2 2027. This is the lead's call.

Checks that do not change the design:
- Swing torque at the pitch joint is about 6–7 N·m.
- Standing up from a belly crouch needs at most 17.7 N·m at the front elbow at 20 kg, below the walk peak.
- The hind-flexion heat figure of 0.94 is conservative: a hind foot in the paw pose carries about 0.39 of the weight, which gives about 0.81.

Confirmed by web search, 18 Nov 2026 (result summaries only):
- RobStride 03: $225 on the Seeed datasheet (updated 21 Oct 2026).
- MyActuator resellers: X6-60 $741.84 at RobotShop, $490 backordered elsewhere; X4 $400; X8-150 $645 backordered.
- Gens ace G-Tech 3300 6S: $107.19, 506 g, 138 × 42 × 43 mm.
- Mean Well RSP-1000-48: $267.90; trims up to about 52.8 V.
- iCharger X12: $179.99, 11–53 V input.
- OAK-D Pro W: $529.
- MTi-3 dev kit: $417.06 at Arrow.
- Pilz PNOZ s5 C: €358 on Pilz's own shop. The $611.99 Zoro listing did not come up again.
- Altran AREVS150-BAN and Littelfuse BF1 100 A / 58 V: both exist, with 16–23-week lead times on the fuse.
- Holybro PM08-CAN: 2–14S.

Not done:
- myactuator.us and openelab.com are outside this session's allowed network, so I could not open them; they have to be allowed in the session's network settings to check live. I opened no vendor page.
- The X4 speed cap rests on reseller text, not the maker's datasheet.
- The estimated (EST) lines, including the CNC frame, shells and paint, were not quoted.
- No simulation run.