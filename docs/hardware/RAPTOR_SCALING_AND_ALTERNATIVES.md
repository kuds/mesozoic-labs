# Running Raptor Plans + a Sub-$5k Quadruped Starter

> **Status:** Cost/parts + feasibility study for *dynamic-running* dino robots.
> Companion to [`HARDWARE_BOM.md`](HARDWARE_BOM.md) (which covers *walking*
> builds) and [`SIM_TO_REAL_PLAN.md`](SIM_TO_REAL_PLAN.md). Prices are USD,
> **parts only** (no engineering labor), and were last checked **2026-07-24**.
> The sub-$5k recommendation is sourced to an orderable, open platform; its
> cosmetic dinosaur parts and reserves are planning allowances rather than
> vendor quotes.

## The one-line answer

> **For a first physical Mesozoic robot under $5k, build a juvenile
> _Psittacosaurus_ around the open Pupper V3 quadruped.** Deliver reliable
> walking/trotting first; treat 1 m/s as a measured research target, not a
> purchase-time promise.

The *parts* for a bipedal runner exist under $10k (~$5–7k of COTS motors, frame,
and compute). What doesn't exist is a proven, replicable "build-it-and-it-runs"
recipe for a cheap biped. An exact open quadruped platform provides a much
shorter path because its CAD, electronics, controller, and assembly sequence
have already been exercised by other builders. The practical options are:

- **Owner-built, open, and under $5k** → juvenile _Psittacosaurus_ on Pupper V3,
  about $4.2–4.8k including a light shell, spares, safety gear, freight/tax, and
  contingency (§3).
- **Advertised 2–3 m/s immediately, under $5k** → a prebuilt Unitree Go2 Air or
  Pro with a passive shell. This is faster to demonstrate, but it is not the
  recommended owner-built research path because the official comparison marks
  Air and Pro as unavailable for secondary development (§3).
- **Two legs, cheap, reliable → a *walker*, not a runner** → Compsognathus, ~$1k
  walking; a ~$5k QDD version *can attempt* running, but the gait is a research
  bet, not a deliverable (§4).
- **Full-size bipedal running (13.5 kg raptor) → a $20k+ research program** (§1).

Two dials explain the whole table below: **money buys _size_** (torque scales
~size⁴, so above ~10 kg you need custom actuators), and **legs buy _reliable
running_** (quadrupeds have a solved cheap gait stack; cheap bipedal running is
unsolved at any budget today).

## Bottom line

Three plans, one clear recommendation:

| Build | Mass | Cost (parts) | Runs & agile? | Custom actuators? |
|---|---|---:|---|---|
| **Full raptor** (1:1, 13.5 kg biped) | 13.5 kg | **$26k–45k** (~$36k) + labor | Plausible but **frontier research** | **Yes, required** |
| **Scaled raptor** (~0.72×, biped) | ~6.5–7 kg | **$5.6k–8k** (~$6.9k) | Running is **unproven**; biped balance dominates | No (all-COTS) |
| **Compsognathus** (~2.5–3 kg biped) | ~3 kg | **~$4k–6k** (can approach $5k) | Runs on paper; **control-limited** | No (all-COTS, comfortable headroom) |
| **Juvenile Psittacosaurus / Pupper V3** ⭐ *recommended first build* | 3 kg stock; **≤3.5 kg** dressed target | **~$4.2k–4.8k program cap** | Proven dynamic-quadruped base; dressed speed must be measured | No custom motor; printed parts + orderable PCBs |

**The core principle:** peak joint torque scales with roughly the **4th power of
size** (mass ∝ L³, torque ∝ mass·L ∝ L⁴). More importantly for a solo first
build, an exact replicated quadruped preserves a demonstrated mechanical and
control stack. The [Stanford Pupper paper](https://arxiv.org/abs/2110.00736)
reports a 12-DOF, off-the-shelf, torque-controlled platform independently
reproduced across institutions. A new dinosaur chassis using similar motors
does **not** automatically inherit that result.

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
> has essentially no prior art. This is a running-capable hardware estimate, not
> evidence that the assembled robot will run. (This build was not independently
> verified.)

---

## 3. ⭐ Recommended first build: juvenile Psittacosaurus on Pupper V3

### Why this species

A juvenile **_Psittacosaurus lujiatunensis_** is the cleanest biological match
for a small quadruped. A histological and limb-allometry study inferred a
quadrupedal-to-bipedal shift during growth, while regarding the adult as mainly
bipedal ([Zhao et al., 2013](https://www.nature.com/articles/ncomms3079)).
Call the result **juvenile-inspired**, not a scale reconstruction: the robot's
geometry is constrained by a proven quadruped chassis.

The previous recommendation, _Scutellosaurus_, was too confident. A detailed
2021 anatomical reassessment identified it as the only definitive bipedal
thyreophoran ([Breeden et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8292774/)).
A later simulation found that rare quadrupedal behavior may have been possible,
but predicted it would be unusual
([Anderson et al., 2023](https://doi.org/10.1002/ar.25189)). That makes it a poor
default for an explicitly four-legged robot.

### Preserve the proven robot; add the animal around it

Use **Pupper V3 without changing its load-bearing leg geometry, transmissions,
electronics, or baseline controller**. Its official documentation specifies:

- 12 actuated joints (three per leg), 3 kg stock mass, and a 25 × 20 × 22 cm
  crouched envelope;
- twelve orderable SteadyWin GIM4305 actuators with 10:1 planetary gearing,
  about 3.5 N·m peak / 1.0 N·m continuous torque, and about 30 rad/s maximum
  speed;
- Raspberry Pi 5 (8 GB), a BNO086 IMU, IMX296 fisheye camera, microphone,
  actuator angle/velocity/effort telemetry, and battery-voltage sensing; and
- open CAD, build instructions, software, and simulation/RL course material.

Sources: [Pupper V3 technical specifications](https://pupper-v3-documentation.readthedocs.io/en/latest/learn_more/tech_specs.html),
[sourcing guide and official BOM](https://pupper-v3-documentation.readthedocs.io/en/latest/guide/sourcing_parts.html),
and the [replication paper](https://arxiv.org/abs/2110.00736).

The V1 dinosaur conversion should be deliberately passive:

- lightweight removable head and torso skins;
- a hollow, compliant tail attached to the body, **not** to a leg or motor
  housing;
- no active jaw, neck, tail, or facial mechanisms until locomotion passes with
  ballast; and
- **≤0.5 kg total morphology allowance**, with the center of mass kept inside
  the stock support polygon. Validate that allowance in 100 g ballast steps
  before printing finished parts.

This keeps the stock robot near its demonstrated operating point. It also makes
the shell sacrificial during falls and preserves access to the back-mounted
E-stop. Pupper's safety guide explicitly warns about motor heat and extended
exertion, so thermal/current logging is a release gate, not an afterthought
([Pupper V3 safety](https://pupper-v3-documentation.readthedocs.io/en/latest/using_pupper/safety.html)).

### Budget cap

The live Pupper V3 BOM currently totals **$2,158.91** (minimum-price column
$1,785.91), including twelve actuators, Pi, IMU/control PCB, two batteries,
controller, camera options, structure, and ordinary build tools. A current
supplier also lists a preconfigured full-parts bundle at $2,350, but it is
backordered and still excludes printed parts, batteries, controller, and setup
items; self-sourcing from the official BOM is therefore the budget route
([official BOM](https://docs.google.com/spreadsheets/d/1e6Hyhc8V6_9mfPMaPCI1v3W4uGBJp4r0xIzH3rUtAeI/edit?usp=sharing),
[supplier listing](https://aifitlab.com/products/pupper-v3-stanford-open-source-robotics-dog?variant=43740448555144)).
A second endorsed supplier lists a more complete kit at $3,400, which leaves
little reserve under $5k
([Present Perfection kit](https://www.present-perfection.com/product-page/pupper-v3-full-kit)).
Access to a 3D printer and availability of the self-sourced actuator/PCB path
are therefore procurement gates.

| Cost block | Planning allowance | Basis |
|---|---:|---|
| Complete self-sourced Pupper V3 | **$2,159** | Live official BOM total |
| Two spare actuators + printed leg/hardware reserve | **$450** | BOM actuator bundle is $1,320/12; balance is local-fabrication reserve |
| Passive juvenile-Psittacosaurus head, shell, and tail | **$400** | Project allowance; prototype in foam/cardboard before final print |
| Tether/stand, floor protection, battery/fire-safe handling | **$350** | Project safety/test allowance |
| Freight and sales-tax allowance | **$550** | Location-dependent reserve |
| Unallocated breakage/price contingency | **$800** | Held until the stock robot passes acceptance |
| **Program cap** | **≈$4,700** | Parts only; owner labor and major shop tools excluded |

This budget uses **catalog integrated motors**, not a custom motor design. It
does require 3D-printed parts and two orderable PCBs, so it is a DIY build rather
than a zero-fabrication assembly.

### Capability gates — do not promise 2–3 m/s up front

Pupper V3's official material demonstrates agile locomotion but does **not**
publish a verified physical top speed. Stanford's current course asks students
to train a stable policy up to 0.75 m/s **in simulation**, then explicitly
compare its behavior after physical deployment
([CS123 Lab 5](https://cs123-stanford.readthedocs.io/en/latest/schedule/labs/spring-25/lab-5.html)).
The correct V1 acceptance sequence is:

1. Build the stock Pupper and reproduce its baseline walk/trot with no dinosaur
   parts.
2. Repeat with 0.1, 0.3, and 0.5 kg ballast at the planned shell locations;
   reject any placement that causes current, temperature, or fall-rate
   regression.
3. Install the passive shell and qualify turning, stopping, disturbance
   recovery, camera/IMU logging, and E-stop access.
4. Measure speed over a repeatable 5 m course, progressing through 0.25, 0.5,
   and **0.75 m/s**. Attempt **1.0 m/s** only while thermal and fall-rate gates
   pass.

Reliable walking/trotting at 0.5–0.75 m/s is the target deliverable.
**1 m/s is a stretch experiment; 2–3 m/s is not a credible Pupper-based
commitment without a demonstrated controller and dressed-robot test data.**

If advertised speed matters more than openness, the official Unitree comparison
lists the 15 kg Go2 Air at **$1,600 / 2.5 m/s** and Pro at **$2,800 / 3.5 m/s**
before tax and freight. It also marks both as unavailable for secondary
development. Either could wear a passive juvenile ceratopsian shell under a
$5k parts cap, but that is a **prebuilt, vendor-gait robot**, not the recommended
owner-built research platform
([Unitree Go2 specifications](https://www.unitree.com/go2/)).

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

## 5. Reference points for the sub-$5k decision

| Robot | Build model | Mass | Speed evidence | Price evidence |
|---|---|---:|---|---:|
| [Pupper V3](https://pupper-v3-documentation.readthedocs.io/en/latest/) | Open CAD/software; owner-built | 3 kg | Course target **0.75 m/s in simulation**; physical top speed unpublished | About $2,000; live BOM **$2,158.91** |
| [Stanford Pupper (paper)](https://arxiv.org/abs/2110.00736) | Open, independently reproduced | 2.1 kg | Dynamic omnidirectional reference gait + 5 m sprint benchmark | Historical build **under $2,000** |
| [Unitree Go2 Air](https://www.unitree.com/go2/) | Prebuilt; no official secondary development | ~15 kg | Vendor-rated **2.5 m/s** | **$1,600** before tax/freight |
| [Unitree Go2 Pro](https://www.unitree.com/go2/) | Prebuilt; no official secondary development | ~15 kg | Vendor-rated **3.5 m/s** | **$2,800** before tax/freight |

The decision is not simply "quadruped versus biped." It is **replicate a proven
open platform versus design a new robot**. For a solo first build, replication
is the lower-risk use of the first $5k.

---

## 6. Recommendation & caveats

**Build a juvenile-Psittacosaurus Pupper V3.** It is the best first use of a
sub-$5k budget because the exact chassis is open, documented, small enough to
work on safely, already includes the essential camera/IMU/proprioceptive sensor
set, and uses orderable integrated motors. Keep V1 to a passive removable
morphology and preserve the stock robot beneath it.

If **2.5 m/s on day one** is non-negotiable, use a Go2 Air as a prebuilt base and
accept the loss of official secondary-development access. If a **biped is
required**, the best small-biped pick remains a **Compsognathus (~2.5–3 kg,
~$4–6k)**, with the important caveat that affordable dynamic biped control is a
research bet. Treat the full 13.5 kg raptor as a funded research program, not a
first build.

**Caveats:**
- **All figures are parts-only.** Dynamic running is a substantial controls
  effort (MPC+WBC or an RL policy) — the BOM buys the body, the software makes it
  run. Budget engineering months.
- **Spares are load-bearing, not padding** — dynamic bring-up cracks frames,
  shears encoder magnets, and fries drivers.
- **Continuous ≠ peak:** Pupper's documented actuator figures are about
  3.5 N·m peak versus 1.0 N·m continuous with air cooling. Do not infer sustained
  speed from peak torque.
- **Morphology is a curatorial call:** the quadruped adds a new species rather
  than reusing the roster's raptor. It's the right engineering choice for cheap
  first-build agility, but it should remain visibly labeled
  "juvenile-inspired."
