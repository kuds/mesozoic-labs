# Draft messages: questions for WIM Robotics and MyActuator

18 Nov 2026. **Drafts for the owner to send; this session contacts no vendor.** Sources: `PLEM_FIT.md` section 5.4
(the full PLEM questions), `CONSOLIDATION_PLAN.md` section 6 (the abridged list, with the museum-quadruped reframe),
and `PARTS_PREMIUM.md` section 8 (the bench questions only the maker can answer).

**Why now.** A PLEM bench pilot runs only if WIM's and MyActuator's answers clear three blockers (PLEM_FIT.md,
recommendation C): PLEM does not support brakeless drives, it runs only on the Jetson AGX Orin, and one bus carries at
most 8 axes. The pilot's go or no-go is the owner's, by end of February 2027 (decision D7), and it stays off the
critical path. The MyActuator questions also feed the foreleg bench plan and the parts list whatever happens to PLEM.

**How to send.**
- WIM: privately, to the contact who pitched PLEM. PLEM's public repository has a "Drive support request" issue
  template, but an issue there would be public.
- MyActuator: the maker's technical support page (www.myactuator.com/service-page/technical-support) or contact page
  (www.myactuator.com/contact), as the T. rex's `MAKER_QUESTIONS.md` suggests; take the addresses from the live pages
  (not read in this session). Each MyActuator message stands alone and names no project, person or organisation, as
  the T. rex's maker questions did.
- **The T. rex's five maker questions** (`hardware/trex_form_study/stage2/design/servo_model/MAKER_QUESTIONS.md`)
  apply to the same drives. If they have not been sent, send them in the same batch; its message 4 (which X6 ships
  under the "RMD-X6-P20-60" listing) is the Triceratops bench's question 1 too, so it is not repeated below.
- Tell the session the date sent, so the answers can be chased after two weeks and logged.

---

## To WIM Robotics (one message)

```
Subject: PLEM on a slow museum quadruped: seven questions before a bench pilot

Hello,

Thank you for the PLEM and W-RC pitch. Our plans have changed since then: the near-term robot is now a
four-legged Triceratops for live museum demos in Q3 2027 (75 cm to the frill, 15 to 20 kg, a slow 0.30 m/s walk,
three MyActuator drives per leg, twelve in all). The two-legged T. rex stays on our roadmap as a flagship for end
2027 / early 2028. We would like to know whether a PLEM bench pilot makes sense for the quadruped. We read the open
repository at v0.11.0; these are the questions it left open for us.

1. Would you reframe the case around a museum quadruped in 2027, with the T. rex as the 2028 flagship case?

2. Brakeless drives. Have you run MyActuator RMD-X V4 / V4.1 "-E" drives, or any other drive without a holding
   brake, under PLEM? Our leg drives have no brakes, so a stop that disables the drives drops the robot about
   400 mm. Is support planned for a stop that holds position under power, or sits the robot down damped, before
   torque is removed?

3. Axis count. Will one EtherCAT bus carry more than 8 axes? Has a two-bus setup run on one AGX Orin, and how
   would a controller coordinate the two buses (a shared clock or cycle)?

4. Compute. When will PLEM support the Jetson Orin NX and Orin Nano Super, and through which network port? Is the
   Orin Nano devkit's built-in Realtek port usable with IgH's native r8169 driver?

5. Delay and user control. Is there a measured end-to-end delay from a user's controller (a Python stream or a
   ros2_control controller) to the drive's target? Can a user's controller (a learned policy or a PD law) run
   inside the real-time loop? Are the feed-forward objects 0x60B1 and 0x60B2 planned?

6. Safety. For a brakeless legged robot, what do you recommend if Linux crashes? Is W-RC a product we can buy,
   and can it drive a "sit, then cut power" sequence rather than a plain power cut? Do you support STO or FSoE
   drives, or an external safety controller?

7. Commercial terms. What does PLEM Pro add for legged robots (torque control, gravity compensation, whole-body
   control), and how is it licensed and priced? What would your 3-day bring-up cost on a one-foreleg bench (three
   drives), or would you do it as a joint case study? What release cadence and support do you commit to for the
   open layer, including issues raised from platforms outside the supported one?

We would run any pilot on two drives first, then on one foreleg, beside our CAN control path, and decide by the
end of February 2027.

Best regards,
```

---

## To MyActuator (four messages, each standing alone)

### M1. EtherCAT on the "-E" drives (PLEM_FIT.md 5.4, MyActuator question 1)

```
Subject: RMD-X4-P36-36-E-L, RMD-X6-P20-60-E, RMD-X8-P20-150-E: EtherCAT cycle time, sync modes and units

Hello. I am considering running the RMD-X4-P36-36-E-L, RMD-X6-P20-60-E and RMD-X8-P20-150-E over EtherCAT.
(1) What is the shortest EtherCAT cycle each part supports, and which sync modes does it support: Free Run,
SM-synchronous, DC SYNC0?
(2) In CiA 402, in which units are the torque and position objects? Does 0x6064 (position actual value) count the
motor-side encoder or the output-side encoder, and is that position absolute across power cycles?
(3) Please send the ESI (XML) file and the EtherCAT manual for these parts.
Thank you.
```

### M2. Fault reactions and the choice of bus (PLEM_FIT.md 5.4, MyActuator question 2)

```
Subject: RMD-X "-E" drives: fault reactions over EtherCAT, and choosing CAN or EtherCAT

Hello. For the RMD-X4-P36-36-E-L, RMD-X6-P20-60-E and RMD-X8-P20-150-E:
(1) Which reactions do objects 0x605A (quick stop), 0x605C (disable operation), 0x605E (fault reaction) and 0x6007
(abort connection) support?
(2) When the EtherCAT sync-manager watchdog expires, what does the drive do: hold position under power, brake the
motor by shorting its windings, or remove torque?
(3) Is CAN or EtherCAT chosen by the wiring or by a setting? Can a drive configured for EtherCAT still accept CAN
commands as a fallback?
Thank you.
```

### M3. The X4-36 above its rated voltage (PARTS_PREMIUM.md section 8, item 14)

```
Subject: RMD-X4-P36-36-E-L (V4.1) on a 42-50 V supply: speed and torque limits

Hello. The RMD-X4-P36-36-E-L is rated at 24 V with a 20-55 V supply range. Reseller listings say that above its
rated voltage the drive must not exceed its rated speed. I am considering running it from a 12-cell lithium
battery, 42-50 V.
(1) Does that rule apply to the V4.1 RMD-X4-P36-36-E-L, and is its rated speed 83 rpm at the output?
(2) At 42 V and at 50 V, what are its continuous and peak torque, and its highest allowed output speed?
(3) Does the drive limit the speed itself above 24 V, or must the controller do it?
(4) Is there a temperature-rise table for this part at 48 V?
Thank you.
```

### M4. The X6-80, the X6-60's heat data, and the US store's stock and terms (PARTS_PREMIUM.md section 8, items 3 and 13)

*Revised 18 Nov 2026 after the owner's information on duties: items in stock at the Fremont warehouse have cleared
customs; factory-dispatched units pay about 41.5 % in tariffs plus brokerage. Questions (3) and (4) now ask about
dispatch, the X6 variant and the educational discount. The owner has a contact address for MyActuator USA; it is not
recorded here.*

```
Subject: RMD-X6-P30-80-E price and heat data; RMD-X6-P20-60-E heat data; US store prices and lead times

Hello.
(1) RMD-X6-P30-80-E (V4.1): what is its price, and is it in stock? Is 80 N·m its peak or its stall torque? Please
send its temperature-rise table: how much torque does it hold continuously at 0 rpm and at 30 rpm without the
motor passing 100 °C?
(2) RMD-X6-P20-60-E (V4): is there a temperature-rise table for it, like the one published for the
RMD-X6-P20-50-E-L?
(3) I am considering an order of about 11 RMD-X4-P36-36-E-L and 3 RMD-X6 units from myactuator.us. Would they all
ship from your Fremont, California stock, with no duties due on delivery? If any would ship from the factory, can
you quote delivered-duty-paid terms?
(4) The store listing "RMD-X6-P20-60" shows 0.78 kg and 50 N·m. Is the unit shipped the RMD-X6-P20-60-E (V4) or
the RMD-X6-P20-50-E-L (V4.1)?
(5) The store advertises 20 % off for educational and research institutions. What are the conditions?
Thank you.
```

---

## What each answer unblocks

| Question | Unblocks |
|---|---|
| WIM 2, 3, 4 | The three PLEM blockers; decision D7 (pilot go / no-go, by end Feb 2027) |
| WIM 5, M1, M2 | Whether EtherCAT beats the four-bus CAN path on delay; the bench plan's PLEM rows |
| WIM 6, M2 | The stop chain: whether a drive fault or a lost master ends in a hold or damped sit, not a drop |
| WIM 7 | Pilot cost (not in any total yet) |
| M3 | Whether 0.30 m/s is the X4's walking limit on 12S (E1); also the T. rex's X4 swing speeds if it moves to 12S |
| M4 (1) | Whether an X6-80 can replace the front X8: 0.52 kg lighter per front leg, about 1.0 kg for the pair (parts list, 2.5) |
| M4 (2) | The hind-knee and roll heat check, now judged against the X6-50-L's table as a stand-in |
| M4 (3) | Whether the build lands near $20k: no duty from Fremont stock, about +41.5 % from the factory |
| M4 (4) | Which X6 the model and bench should use (bench question B2) |
| M4 (5) | A possible -$0.85k on the drives |
