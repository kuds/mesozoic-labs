# Five questions for the maker of the leg drives (16 Nov 2026, 16:37 UTC)

Working paper for the owner, who sends them (`DECISION_LOG.md` P41). Each message stands alone and names no project, person or organisation.

**How far they were checked.** An independent checker compared every figure and model name in the five messages with the study's own research record (`designlab/data/research/actuators_2026-11-07.json`, read from the maker's pages on 7 Nov 2026) and with web-search extracts. All agree after the two corrections below. **No maker page was opened**: this session could not read the maker's sites. So nothing here is confirmed on the maker's live pages, and no question was shown to be already answered there. The checker's report is `MAKER_CHECK.md`.

**Where to send.** The maker's site has a technical support page (www.myactuator.com/service-page/technical-support) and a contact page (www.myactuator.com/contact); message 4 is about the US store (myactuator.us), whose own contact link is in its footer. The addresses were not read from the live pages, so take them from the pages themselves.

**After sending.** Gate 1 is judged after the answers are in the servo model, or two weeks after the messages went out without an answer, whichever comes first (P41). Tell the lead the date they were sent.

## 1. The rotor inertia: before or after the gearbox

*Checked against the study's record of 7 Nov 2026 only. The inertia of the first part was found nowhere outside that record.*

```
Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: is the listed inertia before or after the gearbox?

Hello. Your data for the RMD-X4-P36-36-E-L (X4-36-L, V4.1) list an inertia of 0.000082 kg·m² (0.82 kg·cm²). Your data for the RMD-X6-P20-60-E (X6-60, V4) list an inertia of 0.66 kg·cm² (0.000066 kg·m²). For each part: is this the inertia of the rotor alone, before the gearbox, or the inertia seen at the output shaft, after the gearbox (ratio 36 and 19.612)? Please also give the inertia seen at the output shaft of each part, in kg·m². Thank you.
```

## 2. The gain ranges of the motion mode, and their units

***Hold.** Checked against the study's record only. The maker's protocol document may already state these ranges and units in its section on this command; nobody has read that section. Read it first, or send as it stands and accept that the answer may be "see the document".*

```
Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: range and units of KP and KD in the Motion Mode Control Command

Hello. I plan to drive the RMD-X4-P36-36-E-L and the RMD-X6-P20-60-E with the Motion Mode Control Command over CAN (0x400 + ID). For each part: (1) Does it accept this command? (2) What are the smallest and the largest KP and KD it accepts? (3) In which units do KP and KD act? I need them as N·m/rad and N·m·s/rad at the output shaft. If they are defined on the motor side or as a current, please give the conversion. The user manual of another model of the series, the RMD-X4-P36-24 (X4-24), gives KP 0 to 500 and KD 0 to 5 for its force position hybrid control, with no units printed: do those ranges, and the same units, apply to the Motion Mode Control Command of these two parts? Thank you.
```

## 3. The hip-pitch drive at 24 V

*Checked against the study's record only. The arithmetic (16 × 176 × 19.612 / 1,000 = 55.2 V) was redone.*

```
Subject: RMD-X6-P20-60-E: torque against speed at 24 V, and the back-EMF constant

Hello. The RMD-X6-P20-60-E (X6-60, V4) is listed with a rated torque of 20 N·m at 153 rpm and a no-load speed of 176 rpm, both at 48 V, and with a supply range of 20 to 55 V. I plan to run it from a 24 V supply. (1) At 24 V, what is the no-load speed in rpm, and up to which speed in rpm can it give its peak torque of 60 N·m? A torque-speed curve at 24 V, or a few pairs of N·m and rpm, would answer this. If your dynamometer test report covers 24 V, please send it. (2) The listed back-EMF constant is 16 Vdc/krpm. At 176 rpm at the output and a gear ratio of 19.612 that gives about 55 V, which is more than the 48 V supply. Is the constant per 1,000 rpm of the motor or of the output shaft, and how is its voltage defined? Please give it in V per 1,000 rpm of the motor, with the definition. Thank you.
```

## 4. Which hip-pitch part ships

*Corrected: the first draft said the listing's figures look like the other part's, which holds for the mass only. Checked against the study's record only; the listing may have changed since 7 Nov.*

```
Subject: Store listing "RMD-X6-P20-60": is it the X6-60 (V4) or the X6-50-L (V4.1)?

Hello. On 7 November 2026 the listing "RMD-X6-P20-60" on your US store (myactuator.us/products/rmd-x-series/rmd-x6-p20-60) gave a mass of 0.78 kg, a peak torque of 50 N·m and a backlash of 10 arcmin. The manual and the data sheet of the RMD-X6-P20-60-E (X6-60, V4) give 0.82 kg, 60 N·m and at most 15 arcmin. The listing's mass is that of the RMD-X6-P20-50-E-L (X6-50-L, V4.1), 0.78 kg, and 50 N·m is that part's stall torque. Which part is shipped today when this listing is ordered? Please answer with the full model name and version, and with its mass in kg, its peak torque in N·m and its backlash in arcmin. Thank you.
```

## 5. Which encoder closes the position loop

*Checked against the study's record only.*

```
Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: which encoder closes the position loop, and which one is reported?

Hello. Both parts are listed with two encoders (17 and 18 bit on the RMD-X4-P36-36-E-L, 17 and 17 bit on the RMD-X6-P20-60-E) and with a backlash of up to 15 arcmin. (1) In position control and in the Motion Mode Control Command, does the position loop close on the encoder on the motor side of the gearbox, or on the one on the output side? (2) Which encoder's angle does the drive report in its replies: motor side or output side? In which unit, and with how many counts per turn of the output shaft? (3) Can the user choose between the two? Thank you.
```
