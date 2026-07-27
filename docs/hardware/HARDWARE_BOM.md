# Hardware BOM & Build-Feasibility — Compsognathus & Velociraptor

> **Status:** Cost/parts estimate and buildability check. Companion to
> [`SIM_TO_REAL_PLAN.md`](SIM_TO_REAL_PLAN.md). Prices are **web-sourced (2026,
> USD)** where a vendor link is given, otherwise a flagged knowledge estimate;
> real totals can swing **±15–25%**. Covers **hardware only** — the sim-to-real
> software (state estimator, perception, domain-randomization retraining) is the
> true long pole and is scoped separately in the sim-to-real plan. No robot has
> been built; `hardware_prototype` is still `planned`
> ([`configs/species_manifest.toml`](../../configs/species_manifest.toml)).

## Bottom line

| Robot | Buildable? | Recommended build | Realistic cost (recommended) | Primary constraint |
|---|---|---|---:|---|
| **Compsognathus** (~2.5 kg biped) | **Yes**, with constraints | ~7-DOF walking bring-up, hobby smart servos | **~$950** (range $480–$2,540) | Frame must print light (~0.8–1.2 kg); no CAD/MJCF exists yet |
| **Velociraptor** (13.5 kg biped) | **Walking: yes.** Sprinting: **no** (not from COTS parts) | ~10-DOF walking platform, QDD BLDC actuators | **~$8,000–8,700** (range $5,600–$12,340) | Sim hip torque (225 N·m) exceeds any single COTS actuator (~120 N·m max); mass runs over the 13.5 kg target |

**One-time tooling** (shared, not per-robot): an enclosed FDM 3D printer,
~$400 (Bambu A1) to ~$725 (P1S, needed for the raptor's PA-CF load-bearing
parts).

The Compsognathus is the genuinely easy build; the Velociraptor is feasible as a
**walking/trotting** machine but **not** as the aggressive sprinter its sim
policy implies. Both hardware totals **exclude** the sim-to-real software.

---

## 1. Compsognathus (~2.5 kg bipedal)

Roadmap's chosen physical target (`ROADMAP.md:332-336,508-513`). **Does not exist
in sim yet** — CAD and a matching MJCF must be authored first. Torques for a
2–3 kg biped (1–2 N·m static, 3–6 N·m dynamic peak) sit squarely in the
off-the-shelf hobby smart-servo band, so there is **no torque wall** here.

### 1.1 DOF plan (v1 = walking, 7 actuated DOF)

| Joint | Qty | Req. peak torque | Actuator | Actuator peak | Margin |
|---|---:|---|---|---:|---|
| hip_pitch | 2 | 2–3 N·m (walk) / 5–6 (sim peak) | Feetech STS3250 | 4.9 N·m | 1.6–2.4× walk; thin at sim peak |
| knee | 2 | 2–3 / 4–5 | Feetech STS3250 | 4.9 N·m | ~1.0–1.2× at peak |
| hip_roll | 2 | 1–2 / 3 | Feetech STS3215 | 3.0 N·m | **~1.0× — weakest joint** |
| ankle | 2 | 2–4 (if actuated) | **passive compliant foot** (v1) | — | deferred to save mass/cost |
| tail_pitch | 1 | 0.5–1 / 1.5 | Feetech STS3215 | 3.0 N·m | 2–6× (ample) |

> **Correction from verification:** the STS3215 "3 N·m @ 12 V" spec is optimistic
> — the widely-sold STS3215 is a 7.4 V (2S) servo; at safe voltage it delivers
> closer to ~2 N·m, which pushes hip_roll *under* its 3 N·m lateral peak. And
> hobby-servo "stall" torque is a momentary figure — usable **continuous** torque
> is ~30–50% of stall. Net: the trained policy must be **torque-limited to ~4 N·m**
> and sustained loads kept low, and hip_roll is the joint to watch (upgrade to
> STS3250 or XM430 if lateral push-off is marginal).

### 1.2 Bill of materials — recommended (~$954)

| Item | Cat. | Qty | Unit $ | Line $ |
|---|---|---:|---:|---:|
| Feetech STS3250 (12 V, 50 kg·cm) bus servo | actuators | 4 | 55.00 | 220.00 |
| Feetech STS3215 (30 kg·cm) bus servo | actuators | 3 | 15.99 | 47.97 |
| Raspberry Pi 5 (8 GB) + active cooler | electronics | 1 | 185.95 | 185.95 |
| Waveshare Feetech bus servo driver board | electronics | 1 | 9.99 | 9.99 |
| ESP32-S3 sensor/real-time co-processor | electronics | 1 | 15.00 | 15.00 |
| E-stop + PDB (5 V/12 V BEC) + wiring kit | electronics | 1 | 50.99 | 50.99 |
| Adafruit BNO085 9-DOF IMU | sensors | 1 | 24.95 | 24.95 |
| Roller-lever microswitch foot contacts | sensors | 2 | 2.95 | 5.90 |
| 4S 3000 mAh LiPo + UBEC + alarm + fusing | power | — | — | 77.39 |
| PETG filament (~2.5 kg → ~1 kg finished frame) | structure | 1 | 55.00 | 55.00 |
| Bearings, thrust washers, fasteners, inserts, shafts | structure | — | — | 111.60 |
| Assembly consumables | misc | 1 | 25.00 | 25.00 |
| **Subtotal** | | | | **829.74** |
| **+15% contingency → total** | | | | **954** |

**By category:** actuators $268 · electronics $262 · sensors $31 · power $77 ·
structure $167 · misc $25.

### 1.3 Cost tiers

| Tier | What changes | Total |
|---|---|---:|
| **Budget** | All 7 joints on STS3215 (~$16), ESP32-only compute (no Pi), BNO055, 3S pack, PLA frame | **~$480** |
| **Mid (recommended)** | The BOM above: STS3250 hips/knees, Pi 5 + ESP32, BNO085, 4S pack, PETG frame | **~$954** |
| **Premium/research** | Dynamixel drivetrain (XM430/XC430/XL430 + U2D2), load-cell feet, PA-CF frame | **~$2,540** |

### 1.4 Feasibility — **buildable with constraints**

- **Torque:** ✅ every actuated joint has positive headroom; no torque wall.
  hip_roll (~1.0×) and the STS3215 voltage caveat are the only soft spots.
- **Mass budget:** ✅ closes — actuators (~0.5 kg) + battery (~0.33 kg) = ~0.8 kg,
  realistic total ~2.0–2.2 kg vs 2.5 kg target. **Make-or-break:** the printed
  frame must be held to ~0.8–1.2 kg (a solid 2 kg print alone would eat the whole
  budget).
- **Power:** ✅ non-issue. ~60 W walking avg, ~100–150 W peak; a 4S 3000 mAh pack
  gives ~30–45 min with huge C-rating headroom.
- **Control:** ✅ non-issue. A ~200 K-param MLP at 50–100 Hz runs easily on a Pi 5
  (even an ESP32); 7 servos on one 1 Mbps bus fit the loop.
- **Upstream blocker:** no CAD, no MJCF. The sim model must be **co-authored to
  match this BOM** before a transferable policy can be trained.

**Build effort:** ~80–140 h over 6–10 weeks solo (print, assemble, firmware,
tuning). **Skills:** FDM printing, CAD, mechanical assembly, LiPo/electronics,
embedded (Pi + ESP32 + Feetech SDK), RL sim-to-real, IMU state estimation.

---

## 2. Velociraptor (13.5 kg bipedal)

Exists in sim (`environments/velociraptor/assets/raptor.xml`). Prune the 22 sim
actuators (toes, sickle claws, forelimb stubs, tail-yaw dropped/made compliant)
to an **essential 10 DOF**: per leg hip_pitch + hip_roll + knee + ankle (8) + 2
actuated tail-pitch for balance.

### 2.1 The torque-to-weight crux

The sim hip actuator caps at **±225 N·m** (`forcerange`, gear=1). Two readings:

- **First-principles walk:** static per-leg load ~66–132 N at a 0.2–0.3 m moment
  arm → ~13–40 N·m static, ×2–4 dynamic → **~40–80 N·m peak** at hip/knee.
  **COTS QDD actuators meet this.**

> ⚠️ **The sim's static numbers currently understate the requirement** (found
> 2026-07-27, see [../reviews/VELOCIRAPTOR_PLANT_REVIEW.md](../reviews/VELOCIRAPTOR_PLANT_REVIEW.md)
> §3.2). No raptor leg joint sets `springref`, so passive joint springs supply
> **145 N·m at the home stance** — load carried by an element with no BOM entry.
> Deleting those springs drops zero-action survival from 95% to 0%, so they are
> load-bearing, and a real robot's actuators would have to supply what they
> supply. The direction is therefore known — **the true static requirement is
> above what the sim's actuator forces show** — while the magnitude needs the
> spring/gain retune tracked in [../KNOWN_ISSUES.md](../KNOWN_ISSUES.md).
> Re-derive this section after that lands. The T-Rex, brachiosaurus and
> dibothrosuchus reference their springs to the stance and are unaffected.
- **Sim-cap / sprint:** the actuator-saturation study showed a 0.8×kp (120 N·m)
  cap clipped 34–40% of gait torque, so real gait torque sits **above 120 N·m**
  in the sprint/impact regime, bounded by the 225 N·m cap. **No single COTS
  actuator reaches this** — the strongest units available are AK80-64 (120 N·m
  peak) and RMD-X10-S2 (100 N·m peak), and both are so heavily geared they
  cannot deliver that torque *and* stride speed.

**Conclusion:** a **walking** raptor is buildable; a **sprint-capable** raptor
needs custom dual-motor or harmonic-drive hips, outside off-the-shelf scope.

### 2.2 DOF plan (v1 = walking, 10 actuated DOF)

| Joint | Qty | Req. peak torque | Actuator | Actuator peak | Margin |
|---|---:|---|---|---:|---|
| hip_pitch | 2 | 40–80 (walk) / 120–225 (sim) | MyActuator RMD-X10-S2 (35:1) | 100 N·m | 1.25–2.5× walk; **misses sim cap** |
| hip_roll | 2 | 20–40 / 80 | MyActuator RMD-X6-60 (~19.6:1) | 60 N·m | 1.5–3× walk |
| knee | 2 | 40–80 / 145 | MyActuator RMD-X6-60 | 60 N·m | **~0.75× at top of walk band** |
| ankle | 2 | 30–60 / 150 | MyActuator RMD-X6-60 | 60 N·m | **~1.0× — zero margin** |
| tail_pitch | 2 | 10–20 | CubeMars AK70-10 (10:1) | 24.8 N·m | 1.2–2.5× |

> All actuators kept on one **CAN 2.0** bus. Knee and ankle saturate at hard
> push-off (walking-only). The "RMD-X6-**60**" label is peak torque (N·m), not
> gear ratio (~19.6:1).

### 2.3 Bill of materials — recommended walking build (~$8,665)

| Item | Cat. | Qty | Unit $ | Line $ |
|---|---|---:|---:|---:|
| MyActuator RMD-X10-S2 (100 N·m, hips) | actuators | 2 | 720* | 1,440 |
| MyActuator RMD-X6-60 (60 N·m, roll/knee/ankle) | actuators | 6 | 490 | 2,940 |
| CubeMars AK70-10 (24.8 N·m, tail) | actuators | 2 | 499* | 998 |
| Raspberry Pi 5 + cooler, Teensy 4.1 (CAN-FD master) | electronics | — | — | 220 |
| CANable Pro isolated USB-CAN, PDB, ESP32 | electronics | — | — | 108 |
| E-stop + 150 A contactor + wiring | electronics | — | — | 64 |
| BNO085 IMU | sensors | 1 | 24.95 | 24.95 |
| AS5048A joint-side absolute encoders | sensors | 8 | 19.40 | 155.20 |
| FSR 406 + microswitch foot sensors | sensors | 4 | — | 28.42 |
| 6S 8000 mAh LiPo + BEC/regulator + fusing | power | — | — | 197.34 |
| PETG + PA-CF frame filament | structure | 1 | ~300–500 | ~300–500 |
| Carbon-fiber tubes/plate, bearings, thrust, fasteners | structure | — | — | ~545 |
| **Subtotal** | | | | **~7,220** |
| **+20% contingency → total** | | | | **~8,665** |

**By category:** actuators **$5,378** (the dominant cost) · electronics $392 ·
sensors $209 · power $197 · structure ~$1,045.

`*` RMD-X10-S2 ($720) is a flagged knowledge estimate; AK70-10 street price is
often ~$350–450 (booked at $499). Actuators drive the total — verify these two
first.

### 2.4 Cost tiers

| Tier | What changes | Total |
|---|---|---:|
| **Budget (gentle-walk)** | 10× Unitree GO-M8010-6 (23.7 N·m) — hips/knees **under-torqued**, stands & walks gently only | **~$5,615** |
| **Mid (recommended walking)** | The BOM above (RMD hips/knees/ankles, AK70 tail) | **~$8,665** |
| **Premium/research** | AK80-64 hips (120 N·m, COTS ceiling), Jetson + VectorNav VN-100 tactical IMU, load-cell feet, 1 hr battery | **~$12,340** |

Even the premium tier **cannot sprint** — it documents the gap rather than
closing it.

### 2.5 Feasibility — **buildable as a walker, with constraints**

- **Torque:** ⚠️ walking is met (1–2.5× margin at the hip); knee (~0.75×) and
  ankle (~1.0×) saturate at hard push-off; **sprint torque is unmet** by any
  single COTS unit.
- **Mass budget:** ❌ **over budget.** Actuators (~8.3–9.6 kg) + battery
  (~1.1 kg) ≈ **9.4–10.7 kg = ~70–79% of the 13.5 kg target**, leaving too little
  for a 2 m frame + electronics. Realistic build ~**14–16 kg** — and a heavier
  robot needs *more* torque, worsening the hip problem. (The BOM's "16.5 kg
  filament" line is inconsistent with a ~4.6 kg frame estimate; treat structure
  mass as the softest number and design the frame aggressively light.)
- **Power:** ✅ feasible. >1 kW peak leg-slam is ~45 A at 22.2 V; a 6S 8000 mAh
  35C pack sources ~280 A. ~30 min walking runtime. Heat in the actuators, not
  the pack, is the limit.
- **Control:** ✅ compute is trivial; the real risk is the **observation gap** —
  world-frame base velocity and the prey beacon are privileged sim state needing
  a base-state estimator + perception stack (neither exists).

**Build effort:** ~200–320 h over 2–4 months part-time (CAD, ~16 kg of printing,
assembly, CAN/FOC bring-up, PD tuning, sim-to-real). **Skills:** robot CAD,
engineering-filament FDM, QDD/BLDC FOC + CAN, real-time embedded firmware, 6S
power electronics, legged control + state estimation, RL sim-to-real.

---

## 3. Cross-cutting caveats

1. **Pricing is approximate (2026 US).** Raspberry Pi 5 booked at ~$175
   (tariff-elevated, ~2× normal); RMD-X10-S2 at $720 and AK70-10 at $499 are
   high-side. Totals can move ±15–25%.
2. **Structure mass is estimated, not measured** — and it is the make-or-break
   variable for both. Compso closes *only* with a ~0.8–1.2 kg frame; the raptor
   is already mass-over-budget and the frame estimate is internally uncertain.
3. **The raptor's 225 N·m sim caps are `forcerange` ceilings, not steady-gait
   demand.** Walking needs ~40–80 N·m (COTS meets it). The honest reading is
   "walking feasible, sprinting unmet," not "impossible."
4. **Compsognathus has no CAD/MJCF** — its sim model must be authored to match
   this BOM before a policy can be trained and transferred.
5. **Both totals are HARDWARE ONLY.** The base-state estimator, perception/
   prey-beacon stack, domain randomization, and policy retraining for the
   as-built mass are excluded and are the real critical path (see
   [`SIM_TO_REAL_PLAN.md`](SIM_TO_REAL_PLAN.md) §3.1, §5).
6. **Geared hobby actuators deliver peak torque only momentarily** and wear/
   overheat under repeated foot-impact loading. Keep sustained torque well below
   peak; budget for periodic gear/servo replacement.

---

## 4. Recommendation

**Build the Compsognathus first**, as a ~$950 walking bring-up. It is the
lowest-cost, lowest-risk path to a working legged robot and builds the
servo-bus, control-loop, IMU-estimator, and sim-to-real skills the raptor needs
— but the first task is authoring its **CAD + MJCF**, since neither exists.

**Treat the Velociraptor as a follow-on WALKING platform** at the ~$8–8.7 k
recommended tier, explicitly scoped as walking/trotting. **Do not** attempt a
sprint-capable raptor unless a dedicated research budget (~$12 k+ hardware plus
custom dual-motor/harmonic-drive hips) is approved — the sprint torque-to-weight
demand cannot be met with off-the-shelf actuators.

For **both** robots, budget separately for the sim-to-real software — it is the
long pole, not the parts.
