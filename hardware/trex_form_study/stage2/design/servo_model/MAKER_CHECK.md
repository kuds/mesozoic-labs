<!-- Working paper, 16 Nov 2026: the independent checker's report on the five maker questions. No maker page could be opened from the session; the 'maker / web' column holds web-search extracts. One unconfirmed e-mail address is left out. Nothing here is a rule. -->

MAKER CHECK OF THE FIVE MESSAGES (Appendix A, SERVO_MODEL_DESIGN.md lines 792-836)

VERDICT
Send after the listed corrections, and with one hold.
- Messages 1, 3 and 5: can be sent as they stand. Every figure is what the record holds.
- Message 4: correct first. One sentence is contradicted by the study's own record, and the message does not say which store or when.
- Message 2: hold until someone has read the 0x400 section of the maker's protocol document, then apply one wording correction. That section may state what the message asks.
- No question was found to be already answered in public. This test is incomplete: no maker page could be opened from this session.

Limit on part 2: the session's network allows only package registries. Direct reads of www.myactuator.com, myactuator.us, cdn.robotshop.com and manuals.plus were refused (proxy 403 / host not resolved). Everything in the "maker / web" column is a web-search extract, not a page read. To do a real read, www.myactuator.com and myactuator.us must be allowed in the session's network settings.

FIGURES CHECKED
(RES = actuators_2026-11-07.json, LAT = LATENCY_AND_BUS.md. "Dist." = a distributor's copy of the maker's table, not the maker's page.)

| Msg | Figure as quoted | Record (line) | Maker / web, by search extract | Verdict |
|---|---|---|---|---|
| 1 | RMD-X4-P36-36-E-L (X4-36-L, V4.1) | same (RES:57) | page title "V4.1-X4-36-L Details", https://www.myactuator.com/x4-36-l-details | agrees |
| 1 | inertia 0.000082 kg·m² | "inertia 0.000082 kg·m²" (RES:85) | found nowhere outside the record | agrees with record; maker unchecked |
| 1 | RMD-X6-P20-60-E (X6-60, V4) | same (RES:125) | page title "V4-X6-60 Details", https://www.myactuator.com/x6-60details | agrees |
| 1 | 0.66 kg·cm² | "0.66 kg·cm²", no label (RES:154) | Dist.: "Inertia 0.66 Kg.cm²", https://openelab.io/products/myactuator-rmd-x6-p20-actuator | agrees; label is "Inertia", side not stated |
| 1 | ratio 36 | 36 (RES:70) | Dist.: 36:1, https://openelab.io/products/myactuator-rmd-x4-p36-36-e-l | agrees |
| 1, 3 | ratio 19.612 | 19.612 (RES:138) | Dist.: 19.612, https://www.switch-science.com/products/10516 | agrees |
| 2 | Motion Mode Control Command, 0x400 + ID | 'Motion Mode Control Command_CAN (0x400 + ID)' (LAT:441) | same name in copies of protocol V3.9, V4.01, V4.3, https://files.seeedstudio.com/products/Myactuator/CAN_BUS_Motor_Motion_Protocol_V4.3.pdf | agrees; section content not read |
| 2 | sibling RMD-X4-P36-24 | same (LAT:442) | manual titled "X4-24 (RMD-X4-P36-24) Motor User Manual V1.2 (240412)" | agrees |
| 2 | KP 0 to 500 | 12 bits, 0-4095 for 0.0-500.0 (LAT:442) | "Parameter KP value range: 0~4095 corresponds to 0.0f~500.0f", https://manuals.plus/myactuator/rmd-x4-p36-24-rmd-series-motor-manual | agrees |
| 2 | KD 0 to 5 | 9 bits, 0-511 for 0.0-5.0 (LAT:442) | "Parameter KD value range: 0~511 corresponds to 0.0f~5.0f", same page | agrees |
| 2 | "with no units" | none given (LAT:442) | none beside KP and KD, but the manual prints the control law (see below) | literally right, incomplete |
| 3 | rated 20 N·m at 153 rpm | 20 (RES:131), 153 (RES:134), "at 153 rpm, 48 V" (RES:132) | Dist.: 20 N·m, 153 rpm | agrees |
| 3 | no-load 176 rpm | 176 (RES:135) | Dist.: "No Load Speed 176 RPM" | agrees |
| 3 | "both at 48 V" | 48 V (RES:136); the no-load line carries no voltage | Dist.: input voltage 48 V | agrees for rated; inferred for no-load |
| 3 | supply 20 to 55 V | "20-55 V" (RES:137) | one retailer page: "Operating voltage range 20-55 V" | agrees |
| 3 | peak 60 N·m | 60 (RES:133) | Dist.: 60 N·m | agrees |
| 3 | back-EMF 16 Vdc/krpm | "back-EMF 16 Vdc/krpm" (RES:154) | Dist.: 16 Vdc/krpm (openelab, RobotShop) | agrees |
| 3 | "about 55 V" | arithmetic | 16 × 176 × 19.612 / 1,000 = 55.23 V | right |
| 4 | listing "RMD-X6-P20-60" | same, read 7 Nov 2026 (RES:146) | listed on https://myactuator.us/ | agrees |
| 4 | listing: 0.78 kg | 0.78 kg (RES:151, 154) | not in any extract | agrees with record; unchecked today |
| 4 | listing: peak 50 N·m | 50 N·m (RES:151, 154) | "Rated Torque 20 N.m, Peak Torque 50 N.m", https://myactuator.us/ | agrees |
| 4 | listing: 10 arcmin | 10 arcmin (RES:151, 154) | not in any extract | agrees with record; unchecked today |
| 4 | X6-60: 0.82 kg, 60 N·m | 820 g (RES:127), 60 (RES:133), both in RES:154 | Dist.: 0.82 kg, 60 N·m | agrees |
| 4 | X6-60: at most 15 arcmin | 15 (RES:143), "<=15", 2026-10-29 manual (RES:154) | Dist. copies of older V4 documents: 10 arcmin | agrees with record; older maker documents said 10 |
| 4 | "figures look like those of the RMD-X6-P20-50-E-L (X6-50-L, V4.1)" | X6-50-L: 780 g (RES:161), peak 45 N·m (RES:167), stall 50 N·m (RES:186), backlash 15 (RES:177) | not reached | DIFFERENCE: only the mass matches |
| 5 | X4-36-L: 17 and 18 bit | "dual encoder 17-bit/18-bit" (RES:76) | Dist.: 17-bit input, 18-bit output | agrees |
| 5 | X6-60: 17 and 17 bit | "dual encoder 17/17-bit" (RES:144) | Dist.: "ABS-17BIT (Input)/17BIT (Output)", https://robotopian.com/products/myactuator-rmd-x6-60-motor | agrees with record; one conflicting extract (below) |
| 5 | backlash up to 15 arcmin, both | 15 (RES:75, RES:143) | Dist.: 15 for the -E-L | agrees |

All line citations under the messages point to lines that hold what the note says.

Message 3 arithmetic: 176 × 19.612 = 3,451.7 rpm at the motor; × 16 / 1,000 = 55.23 V. "About 55 V" is right and is above 48 V. The same sum on the record's values gives 23.98 V for the V4 X4-36 (its supply is 24 V), 17.2 V for the X4-36-L and 40.4 V for the X6-50-L. Only the X6-60 exceeds its supply, so the question is fair.

WORDING
All five are polite, self-contained and free of any project, person or organisation name. Model names are spelt the same in each subject line and body. Each asks for stated units, except message 5 part 2, which properly asks the maker to name the unit.

CORRECTIONS (exact replacement text)

Message 4, required. Replace the body with:
> Hello. On 7 November 2026 the listing "RMD-X6-P20-60" on your US store (myactuator.us/products/rmd-x-series/rmd-x6-p20-60) gave a mass of 0.78 kg, a peak torque of 50 N·m and a backlash of 10 arcmin. The manual and the data sheet of the RMD-X6-P20-60-E (X6-60, V4) give 0.82 kg, 60 N·m and at most 15 arcmin. The listing's mass is that of the RMD-X6-P20-50-E-L (X6-50-L, V4.1), 0.78 kg, and 50 N·m is that part's stall torque. Which part is shipped today when this listing is ordered? Please answer with the full model name and version, and with its mass in kg, its peak torque in N·m and its backlash in arcmin. Thank you.

Reasons:
- The record gives the X6-50-L a peak of 45 N·m and a backlash of 15 arcmin, so "the listing's figures look like" it holds for the mass only. RES:154's own phrase "which match the V4.1 X6-50-L" is contradicted by RES:167 and RES:177.
- "The store listing" does not say which store or when. The search extract of the store's home page today shows USD 599.00 for this listing against the record's USD 428.00, so the page may have changed since 7 Nov.

Message 2, required. Replace the last question with:
> The user manual of another model of the series, the RMD-X4-P36-24 (X4-24), gives KP 0 to 500 and KD 0 to 5 for its force position hybrid control, with no units printed: do those ranges, and the same units, apply to the Motion Mode Control Command of these two parts?

Reason: the sibling's frame is not the 0x400 command (LAT:442 itself calls it a "force position hybrid control" frame, with KD in 9 bits).

Message 1, optional. Replace the first two sentences with:
> Hello. Your data for the RMD-X4-P36-36-E-L (X4-36-L, V4.1) list an inertia of 0.000082 kg·m² (0.82 kg·cm²). Your data for the RMD-X6-P20-60-E (X6-60, V4) list an inertia of 0.66 kg·cm² (0.000066 kg·m²).

Message 3, optional. After "would answer this." add:
> If your dynamometer test report covers 24 V, please send it.

Message 5, optional. Replace "(17 and 18 bit on the RMD-X4-P36-36-E-L, 17 and 17 bit on the RMD-X6-P20-60-E)" with:
> (listed as 17 bit on the input side and 18 bit on the output side for the RMD-X4-P36-36-E-L, and 17 and 17 bit for the RMD-X6-P20-60-E)

WHAT THE MAKER'S PUBLIC MATERIAL ALREADY ANSWERS
- Question 1: nothing found. Distributor copies of the maker's table label the figure only "Inertia (Kg.cm²)", with no side.
- Question 2: partly, for the sibling only. Copies of the X4-24 manual print "motor_current = (KP*(pos_desired - pos_current) + KD*(spd_desired - spd_current) + Torque) / KT" and "Expected output end position value range: 0~65536 corresponds to -12.5rad~12.5rad" (https://manuals.plus/myactuator/rmd-x4-p36-24-rmd-series-motor-manual). That implies N·m/rad and N·m·s/rad at the output for that model. Nothing was found for the two drives.
- Question 2, the hold: the maker's downloads page lists "CAN BUS Motor Motion Protocol V4.4" (https://www.myactuator.com/downloads-xseries); the study read V4.01 and V4.3 through extracts. I could not open any version. A lead from memory, not a checked fact: the 0x400 section may itself list kp 0 to 500 and kd 0 to 5 with position in rad and torque in N·m. If it does, message 2 should cite that section, not the sibling manual.
- Question 3: no 24 V curve found. The downloads page says "This data is tested by dynamometer. Please send email to [address] to get test report".
- Question 4: not answered. The store's home page still shows "Peak Torque 50 N.m".
- Question 5: nothing found on which encoder closes the loop.

SUPPORT CHANNEL (search extracts; confirm on the live pages)
- Technical support page: https://www.myactuator.com/service-page/technical-support. It lists a phone number and an e-mail address. Two extracts show it as a free-mail address, [a free-mail address shown in two search extracts; not confirmed on the page], which is unusual and must be confirmed on the page before use.
- Downloads page, https://www.myactuator.com/downloads-xseries: names tangxc@myactuator.com for test-report requests.
- Contact page with regional sales addresses: https://www.myactuator.com/contact.
- https://www.myactuator.com/globaldistributor says buyers in the United States, South Korea and Japan go through the authorised agent for after-sales service. Message 4 is about the US store, so it belongs with that store: "Contact Us" in the footer of https://myactuator.us/; the form or address was not in any extract.
Nobody was contacted and no form was filled in.

WHAT I COULD NOT CHECK
- Any maker page, manual, data sheet or protocol PDF by direct read.
- The 0.000082 kg·m² inertia of the X4-36-L, anywhere outside the record.
- The store listing's 0.78 kg and 10 arcmin as they stand today.
- The X6-60's output encoder: one extract of the maker's older "X Series-V4 product manual-250211" shows "ABS-17BIT (Input) /18BIT (Output)" in a scrambled table whose model column could not be read. The record and a distributor say 17/17.
- The content of the 0x400 section, and the reply units that bear on message 5 part 2.

No repository file was changed and no simulation was run. Working notes: /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/maker_check/notes.txt
