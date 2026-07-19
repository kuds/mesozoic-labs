# Running Raptor Plans + a Sub-$10k Agile-Runner Species

> **Status:** Cost/parts + feasibility study for *dynamic-running* dino robots.
> Companion to [`HARDWARE_BOM.md`](HARDWARE_BOM.md) (which covers *walking*
> builds) and [`SIM_TO_REAL_PLAN.md`](SIM_TO_REAL_PLAN.md). Prices are
> web-sourced 2026 USD, **parts only** (no engineering labor). Figures reflect
> adversarial verification — two of three builds were independently checked and
> came back "minor-issues"; corrections are folded in below.

## Bottom line

Three plans, one clear recommendation:

| Build | Mass | Cost (parts) | Runs & agile? | Custom actuators? |
|---|---|---:|---|---|
| **Full raptor** (1:1, 13.5 kg biped) | 13.5 kg | **$26k–45k** (~$36k) + labor | Plausible but **frontier research** | **Yes, required** |
| **Scaled raptor** (~0.72×, biped) | ~6.5–7 kg | **$5.6k–8k** (~$6.9k) | Runs, but fights biped balance | No (all-COTS) |
| **Compsognathus** (~2.5–3 kg biped) | ~3 kg | **~$4k–6k** (can approach $5k) | Runs on paper; **control-limited** | No (all-COTS, comfortable headroom) |
| **Scutellosaurus quadruped** ⭐ *recommended* | ~7–9 kg | **~$6.9k**, or **$3.5k–4.5k** budget | **Yes — genuinely agile** | No (all-COTS) |

**The core principle:** peak joint torque scales with roughly the **4th power of
size** (mass ∝ L³, torque ∝ mass·L ∝ L⁴), and **quadrupeds are the only proven
cheap path to agile running**. Every sub-$10k robot that actually runs — Stanford
Doggo (~$3k), ODRI Solo-8 (~$4.3k), MIT Mini-Cheetah — is a quadruped. The
cheapest credible *bipeds* (Berkeley Humanoid Lite, <$5k) only **walk**. So
"agile + runs + no custom actuators + under $10k" points hard at a **small
quadruped**, not a full-size biped.

---

## 1. Full raptor that runs (13.5 kg) — a research program, ~$36k

At full scale the sim's ~150–225 N·m hip/knee run torques exceed **every single
COTS actuator that can also move at running speed**. The units that reach 200+ N·m
(AK80-64, harmonic drives) get there through high gear reduction and are far too
slow and non-backdrivable to run.

**Actuator strategy — custom is unavoidable.** The four propulsion joints (2 hip
pitch, 2 knee) need **custom dual-motor QDD modules** (two large BLDC summed
through a shared 8–10:1 planetary, ~$2,900 each built) to deliver ~220 N·m *at
running speed* while staying backdrivable. Abduction/ankle use COTS CubeMars
AKE90-8 (170 N·m); the tail uses AK80-9.

| Cost block | ~USD |
|---|---:|
| Actuators (4 custom modules + 4 AKE90-8 + 2 AK80-9) | ~$14,500 |
| Frame (CF/machined Al), power (12S, 2–3 kW), compute, tactical IMU, sensing | ~$11,200 |
| Spares + contingency (dynamic bring-up destroys parts) | ~$10,300 |
| **Parts total** | **~$36,000** ($26k–45k) |
| Engineering / NRE (control, custom-actuator dev) | **+$15k–40k** |

> **Verification flag:** even the custom hip module (220 N·m peak) is *under* its
> own worst-case requirement (225 N·m), and the knee has only ~1.2× margin. So the
> full raptor's actuation is marginal at the top of its band — this is a
> multi-quarter R&D effort, **not remotely a sub-$10k target**.

---

## 2. Scaled raptor that runs (~5.5 kg biped) — ~$6.9k, all-COTS

Shrink the raptor to 0.72× linear scale (5.5 kg target) and the torque law does
the work: hip run peak drops to **~30–55 N·m**, squarely inside COTS QDD. No
custom actuators.

**9-DOF, all off-the-shelf integrated QDD** (each ships with driver + encoder):

| Joint | Qty | Run torque | Actuator | Peak |
|---|---:|---|---|---:|
| Hip pitch | 2 | 30–55 N·m | MyActuator RMD-X6-60 | 60 N·m |
| Knee | 2 | 25–45 N·m | MyActuator RMD-X4-36 *(geared, marginal)* | 34 N·m |
| Hip roll | 2 | 15–25 N·m | CubeMars AK70-10 (true 10:1 QDD) | 24.8 N·m |
| Ankle | 2 | 10–20 N·m | SteadyWin GIM8108-8 | 22 N·m |
| Tail yaw | 1 | 5–9 N·m | CubeMars AK60-6 | 9 N·m |

**Cost:** actuators ~$3,180 + frame/power/compute/sensing/spares ~$2,380 + 25%
contingency → **~$6,900** (range $5.6k–8k).

> **Honest caveats:** (1) the mass-optimal knee (RMD-X4-36, 36:1) is a geared
> servo, not true QDD — marginal backdrivability for agile impacts; (2) actuator
> mass pushes realistic all-up weight to ~6.5–7 kg, not 5.5 kg; (3) **biped
> dynamic balance, not torque or cost, is the hard part** — cheap bipedal running
> has essentially no prior art. It *will* run, but the control problem will
> dominate the schedule. (This build was not independently verified.)

---

## 3. ⭐ Recommended: a Scutellosaurus quadruped — ~$6.9k, or under $5k

If the goal is a robot that **genuinely runs and is agile** on catalog parts,
**go quadruped** — it sidesteps the biped-hip torque wall entirely (load splits
across four legs → per-joint run peaks fall to **~15–20 N·m**) and inherits a
solved dynamic-gait control problem.

**Why this species.** **Scutellosaurus lawleri** (~3 kg basal armored
ornithischian) was **genuinely facultatively quadrupedal** — so building it on
four legs is biologically honest, not a fudge — and its heavy armored tail maps
naturally onto a **rear battery/compute counterweight**. It sits dead-center in
the 2–8 kg sweet spot and maps cleanly onto the Stanford Doggo / ODRI Solo-8 /
Mini-Cheetah lineage. *(Alternative: a **juvenile Psittacosaurus**, which was
ontogenetically quadrupedal at this scale — equally defensible.)*

**12-DOF (3 per leg), all-COTS integrated QDD:**

| Joint | Qty | Run torque | Actuator (primary) | Peak |
|---|---:|---|---|---:|
| Hip abduction | 4 | ~8–12 N·m | CubeMars AK70-10 | 24.8 N·m |
| Hip flex/extend | 4 | ~15–20 N·m | CubeMars AK70-10 | 24.8 N·m |
| Knee | 4 | ~15–20 N·m | CubeMars AK70-10 | 24.8 N·m |

| Cost block | Primary (AK70-10) | Budget (GIM8108-8) |
|---|---:|---:|
| 12× actuators | $4,787 | $2,220 |
| Frame, power (6S), Jetson Orin Nano, IMU, feet, wiring | ~$890 | ~$890 |
| Spares & breakage reserve | $1,200 | ~$600 |
| **Total** | **~$6,900** | **~$3,500–4,500** |

- **Under $10k route:** 12× CubeMars AK70-10 (24.8 N·m peak, true 10:1
  backdrivable QDD) → **~$6,900**. Comfortable trot/turn margin.
- **Under $5k route:** swap to 12× SteadyWin GIM8108-8 Mini-Cheetah-style modules
  (22 N·m peak, integrated open-source CAN driver, ~$185 each) → **~$3.5k–4.5k**.

> **Verification caveats:** realistic all-up mass is **~9 kg** (12 motors alone
> are ~6.3 kg), i.e. genuine Mini-Cheetah class / an upper-mass Scutellosaurus. At
> 9 kg the AK70-10 top-end margin tightens, and the sub-$5k GIM8108-8 variant is
> **fine for reliable trotting/turning at ~2.5–3 m/s but marginal for the full
> bound/backflip/self-right envelope** — scope the cheap build to trotting. Also:
> COTS QDD margins are quoted against *peak* torque; sustained running is
> thermally limited by the much lower continuous rating, so expect
> ~10–20 min of active running per battery.

---

## 4. Bipedal runner options — is it a motor-availability limit?

If you specifically want a **bipedal** dino that is nimble and can run, the good
news is that **it is not primarily a motor-availability problem** — at least not
at small scale. It's a *scale* problem, and above all a *control* problem.

**Why scale decides it.** A biped puts the whole body's weight on **one** leg
during single-support and flight phases, so its per-joint torque is roughly **2×
a quadruped's** of the same mass. Combined with the L⁴ torque law, that sets a
mass ceiling below which cheap COTS QDD actuators comfortably cover a running
gait:

| Biped mass | Hip run peak | COTS QDD situation | Real limiter |
|---|---|---|---|
| **~2.5–3 kg** (Compsognathus) | ~10–18 N·m | Several **true-QDD** units with headroom — AK70-10 (24.8), AK80-9 (22), GIM8108-8 (22) | **Control**, not motors |
| **~5–7 kg** (Segisaurus, scaled raptor) | ~30–55 N·m | In-band, but the backdrivable units get marginal → you drift to **geared** servos (RMD-X6-60 @20:1, RMD-X4-36 @36:1) | Edge: motors OK, backdrivability compromised |
| **~13.5 kg** (full raptor) | 150–225 N·m | **No single COTS unit** delivers this at running speed | **Motors → custom required** |

So: for a **≤~3 kg biped, motors are plentiful and cheap** and give real
headroom. The actuator wall only appears at raptor scale. **The binding
constraint for any cheap biped runner is the control problem** — cheap bipedal
*dynamic running* has essentially no prior art: the cheapest credible DIY biped
(Berkeley Humanoid Lite, <$5k) only **walks**, and bipeds that genuinely run
(Cassie, Digit) are ~30 kg, $100k+ research platforms. A quadruped of the same
mass runs on the *same motors* with a solved gait stack.

**Best bipedal species (nimble + run + to scale + budget):**

1. **Compsognathus (~2.5–3.5 kg)** — the standout. Obligate biped, high cursorial
   agility (elongate hindlimbs, tibia:femur ~1.3, long counterbalancing tail),
   dead-center in the 2–8 kg band, and already the roadmap's physical target.
   Comfortable true-QDD motor headroom. **~$4k–6k** as a running build (~8–9 DOF:
   6× small QDD hips/knees + lighter ankle/tail units + 6S power + compute +
   spares); a lean build **can approach ~$5k**. Under $10k easily.
2. **Segisaurus (~4–7 kg)** — robust cursorial coelophysoid; a sturdier biped if
   you want more mass. COTS-feasible but at the marginal edge (same profile as the
   scaled raptor), so backdrivability is compromised.
3. **Sinosauropteryx (up-scaled to ~2–3 kg)** — the **longest tail relative to
   body** of any theropod, i.e. a superb active counterweight for a biped runner;
   small enough for comfortable motor headroom.
4. *Iconic-but-oversized runners* — **Hypsilophodon** and **Lesothosaurus** are
   textbook "built for speed" cursors, but ~20 kg; only viable if down-scaled to
   ~3–5 kg (which puts them back in the marginal-motor band).

**The honest bottom line for bipeds:** a small bipedal runner is *affordable and
motor-feasible* (~$5k for a Compsognathus), so you are **not** hitting an
actuator wall at that scale. What you're up against is that **cheap bipedal
running is an unsolved controls problem** — the parts budget buys a machine that
*can* run, with real risk the gait proves hard to achieve. A quadruped of the
same budget remains the surer bet for *actually-achieved* agile running.

---

## 5. Reference points (why these numbers are credible)

| Robot | Type | Mass | Runs? | Cost |
|---|---|---|---|---:|
| Stanford Doggo | QDD quad, open-source | <5 kg | Yes (jumps/backflips) | ~$3,000 |
| ODRI Solo-8 | QDD quad, open-source | ~2.5 kg | Yes | ~$4,300 |
| MIT Mini-Cheetah | QDD quad | 9 kg | Yes (2.5 m/s) | ~$3.6k motors / ~$10k build |
| Unitree Go2 | QDD quad, commercial | ~15 kg | Yes, turnkey | $1,600+ |
| Berkeley Humanoid Lite | biped, 3D-printed | ~16 kg | **No — walks only** | <$5,000 |
| SpotMicro / hobby-servo quads | servo quad | ~2 kg | **No — slow walk** | $300–600 |

The pattern is unambiguous: **cheap + agile + runs ⇒ QDD quadruped.**

---

## 6. Recommendation & caveats

**Build the Scutellosaurus quadruped.** It's the single best path to an agile,
genuinely-running dino robot on catalog parts with zero custom actuators — under
$10k with AK70-10 (~$6.9k), or under $5k via the GIM8108-8 swap (~$3.5–4.5k,
scoped to trotting). If a **biped is required**, the best small-biped pick is a
**Compsognathus (~2.5–3 kg, ~$4–6k)** — small enough that COTS true-QDD motors
give comfortable headroom (§4) — with the scaled raptor (~$6.9k) as the larger
"stay-a-theropod" option; either way you accept the unsolved cheap-biped-running
control problem. Treat the full 13.5 kg raptor as a funded research program, not
a build.

**Caveats:**
- **All figures are parts-only.** Dynamic running is a substantial controls
  effort (MPC+WBC or an RL policy) — the BOM buys the body, the software makes it
  run. Budget engineering months.
- **Spares are load-bearing, not padding** — dynamic bring-up cracks frames,
  shears encoder magnets, and fries drivers.
- **Continuous ≠ peak:** margins are quoted against transient peak torque;
  sustained running is thermally limited (~10–20 min/pack).
- **Morphology is a curatorial call:** the quadruped adds a new species rather
  than reusing the roster's raptor. It's the right engineering choice for cheap
  agile running, but it's your decision to make.
