# Running Raptor Plans + $5k/$10k Quadruped Paths

> **Status:** Cost/parts + feasibility study for *dynamic-running* Mesozoic
> animal robots.
> Companion to [`HARDWARE_BOM.md`](HARDWARE_BOM.md) (which covers *walking*
> builds) and [`SIM_TO_REAL_PLAN.md`](SIM_TO_REAL_PLAN.md). Prices are USD,
> **parts only** (no engineering labor), and were last checked **2026-07-24**.
> The sub-$5k recommendation is sourced to an orderable, open platform. The
> $10k option is a clean-sheet owner-built chassis using catalog actuators.
> Cosmetic animal parts, fabrication, freight, and reserves are planning
> allowances rather than vendor quotes.

## The one-line answer

> **Under $5k, build a juvenile _Psittacosaurus_ around the open Pupper V3.
> Near $10k, a custom _Dibothrosuchus elaphros_-inspired quadruped using
> catalog actuators can credibly target 2 m/s and treat 3 m/s as a
> short-sprint stretch goal.**

The *parts* for a bipedal runner exist under $10k (~$5–7k of COTS motors, frame,
and compute). What doesn't exist is a proven, replicable "build-it-and-it-runs"
recipe for a cheap biped. An exact open quadruped platform provides a much
shorter path because its CAD, electronics, controller, and assembly sequence
have already been exercised by other builders. A $10k clean-sheet quadruped can
buy enough actuator, power, and compute headroom for 2–3 m/s, but controls and
low-inertia mechanical design remain research work. The practical options are:

- **Owner-built, open, and under $5k** → juvenile _Psittacosaurus_ on Pupper V3,
  about $4.2–4.8k including a light shell, spares, safety gear, freight/tax, and
  contingency (§3).
- **Owner-built, custom chassis, and near $10k** → an approximately 10–12 kg
  _Dibothrosuchus_-inspired quadruped using twelve catalog CubeMars actuators.
  Use **2 m/s as acceptance, 2.5 m/s as a success target, and 3 m/s as a
  stretch**, not a purchase-time promise (§4).
- **Advertised 2–3 m/s immediately, under $5k** → a prebuilt Unitree Go2 Air or
  Pro with a passive shell. This is faster to demonstrate, but it is not the
  recommended owner-built research path because the official comparison marks
  Air and Pro as unavailable for secondary development (§3).
- **Two legs, cheap, reliable → a *walker*, not a runner** → Compsognathus, ~$1k
  walking; a ~$5k QDD version *can attempt* running, but the gait is a research
  bet, not a deliverable (§5).
- **Full-size bipedal running (13.5 kg raptor) → a $26k+ parts program** (§1).

Two dials explain the whole table below: **scale and support pattern set the
torque requirement** (torque scales roughly with size⁴, and a biped loads fewer
legs), while **replication buys a shorter controls path**. A roughly 10 kg
quadruped can use catalog actuators; a full-scale 13.5 kg raptor biped cannot.

## Bottom line

Five plans, with the recommendation determined by budget and speed:

| Build | Mass | Cost (parts) | Runs & agile? | Custom actuators? |
|---|---|---:|---|---|
| **Full raptor** (1:1, 13.5 kg biped) | 13.5 kg | **$26k–45k** (~$36k) + labor | Plausible but **frontier research** | **Yes, required** |
| **Scaled raptor** (~0.72×, biped) | ~6.5–7 kg | **$5.6k–8k** (~$6.9k) | Running is **unproven**; biped balance dominates | No (all-COTS) |
| **Compsognathus** (~2.5–3 kg biped) | ~3 kg | **~$4k–6k** (can approach $5k) | Runs on paper; **control-limited** | No (all-COTS, comfortable headroom) |
| **Juvenile Psittacosaurus / Pupper V3** ⭐ *recommended first build* | 3 kg stock; **≤3.5 kg** dressed target | **~$4.2k–4.8k program cap** | Proven dynamic-quadruped base; dressed speed must be measured | No custom motor; printed parts + orderable PCBs |
| **Dibothrosuchus-inspired custom quadruped** ⭐ *speed-focused DIY build* | **~10–12 kg** | **~$9k–11k** (~$10k target) | **2 m/s credible; 2.5 m/s target; 3 m/s short-sprint stretch** | No custom motor; custom chassis/remote-knee transmissions |

**The core principle:** peak joint torque scales with roughly the **4th power of
size** (mass ∝ L³, torque ∝ mass·L ∝ L⁴). More importantly for a solo first
build, an exact replicated quadruped preserves a demonstrated mechanical and
control stack. The [Stanford Pupper paper](https://arxiv.org/abs/2110.00736)
reports a 12-DOF, off-the-shelf, torque-controlled platform independently
reproduced across institutions. A new Mesozoic-animal chassis using similar
motors does **not** automatically inherit that result. The $10k plan therefore
buys a capable mechanical envelope, not guaranteed running performance.

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

## 4. $10k custom quadruped — a credible 2–3 m/s research platform

Increasing the program cap from $5k to approximately $10k changes the design
space. It does **not** make Pupper V3 twice as fast; it permits a new, larger,
48 V chassis around higher-power catalog actuators. The result can have a
credible 2–3 m/s mechanical envelope while remaining owner-built and avoiding
custom motor or gearbox development.

This is still a clean-sheet robot. The budget buys capable hardware, not the
years of controls refinement behind a commercial quadruped.

### Mechanical and electrical envelope

Use a Mini-Cheetah-class layout:

- **12 DOF:** hip abduction/adduction, hip pitch, and knee pitch on each leg;
- **10–12 kg all-up**, with a roughly 45–50 cm structural body, 25–30 cm
  effective legs, and a 1.1–1.3 m lightweight animal exterior including tail;
- **12S LiPo power** (44.4 V nominal / 50.4 V fully charged), a main fuse and
  contactor, precharge path, accessible physical E-stop, DC-DC rails, and a
  battery sized for short test sessions rather than long endurance;
- ab/adduction and hip-pitch actuators mounted at the body, with each knee
  actuator co-located at the shoulder or pelvis and driving the anatomical knee
  through a toothed belt, chain, or rigid linkage; and
- three independent 1 Mbit/s CAN buses, four actuators per bus, a deterministic
  motor/safety controller targeting **400–500 Hz** command and telemetry, and a
  separate Jetson-class computer for perception and the high-level policy.
  Measure worst-case bus utilization and reduce telemetry before increasing the
  loop rate; this design does not assume that twelve motors can exchange full
  command/feedback frames at 1 kHz on Classic CAN.

The actuator envelope must be treated as a chassis constraint, not hidden by an
idealized simulation. The AK80-9 is 98 mm in diameter and 38.5 mm thick, and
this study conservatively uses the detailed-table mass of 490 g; the same
official page's summary says 480 g. Twelve units therefore contribute
**5.88 kg**, roughly 49–59% of a 10–12 kg robot. The motor diameter is about
33–39% of a 25–30 cm effective leg, so an exposed AK80-9 at the anatomical knee
would not produce a credible slender limb
([CubeMars AK80-9 V3](https://www.cubemars.com/product/ak80-9-v3-0-robotic-actuator.html)).

Make the exterior slender, not the load-bearing chassis. **No AK80-9 belongs at
the anatomical knee.** Keep the motors inside locally fuller shoulder and
pelvic volumes, leave the abdomen and distal limbs slender, and use carbon or
thin-wall aluminum for the moving links. Model the actual motor cylinders,
transmission ratio and mass, reflected inertia, backlash, compliance, and joint
limits. The primary simulation and controller plant must use the eventual
_Dibothrosuchus_ hardware geometry, mass distribution, joint axes, and
remote-knee transmission—not an idealized _Terrestrisuchus_ skeleton. Run
_Terrestrisuchus_ only as a separate, secondary morphology-stress case. MIT's
Mini Cheetah architecture provides relevant precedent for
co-locating hip and knee actuators and transmitting knee torque through the
upper leg, but its thesis also documents a roughly 30 Hz belt resonance; the
remote-knee assembly therefore needs its own one-leg impact, bandwidth, and
thermal fixture before the chassis ratio is frozen
([Katz, 2018](https://hdl.handle.net/1721.1/118671)).

The proximal-knee rule is non-negotiable for speed. Low distal inertia lets the
feet reverse direction quickly and reduces impact energy during a fall. During
CAD and simulation, sweep **180, 210, and 240 mm** internal shoulder/pelvic
spacing at the same exterior length. Select the narrowest closed-box chassis
that passes lateral-push recovery, 2 m/s running, 2–2.5 m/s braking and turning,
and short 3 m/s tests; then repeat with head and tail ballast at their real
radii. The representative 12S pack's nominal dimensions are
**278.1 × 48.7 × 37.9 mm** (vendor tolerances ±5 × ±2 × ±2 mm), so the same CAD
sweep must reserve a rigid, ventilated battery bay, connector and lead-bend
clearance, and a tool-free retention/egress path rather than treating the pack
as a point mass
([Gens Ace](https://www.genstattu.com/gens-ace-sport-g-tech-3500mah-12s-80c-44-4v-lipo-battery-pack-with-xt90-plug/)).

CubeMars specifies 48 V rated and 18–52 V allowable input for the large driver
class. A fully charged 12S pack is inside that published window but leaves only
1.6 V for regenerative bus rise. Confirm the exact purchased driver's limit,
monitor bus voltage, start with a **4.10 V/cell (49.2 V) charge ceiling**, and
validate the charge ceiling plus any braking clamp on a one-leg fixture before
dynamic tests
([CubeMars AK-series product manual](https://img.cubemars.com/products/cubemars-product-parameter/AK-Series-Module-Product-Manual-v3.0.0-Download.pdf)).

### Catalog actuator plan

The speed-first configuration is **12× CubeMars AK80-9 V3**. The current
official listing gives 22 N·m peak torque, 9 N·m rated torque, 390 rpm rated
output speed (about 40.8 rad/s), 570 rpm no-load speed (about 59.7 rad/s), an
integrated driver/encoder, and a listed price of $479.90. This study uses the
page's conservative 490 g detailed-table value rather than its conflicting
480 g summary value. Twelve therefore cost approximately **$5,759 before
freight and tax**
([AK80-9 V3](https://www.cubemars.com/product/ak80-9-v3-0-robotic-actuator.html)).

A cost/weight-optimized alternative uses **8× AK80-9** at the hip-pitch and
knee joints plus **4× AK60-6 V3** at the lateral hips. At the current listed
prices that actuator set is approximately **$4,759** and saves about 440 g, but
the AK60-6 provides only 9 N·m peak / 3 N·m rated torque. Use it only if
simulation shows adequate margin for cornering, disturbance recovery, and
self-righting
([AK60-6 V3](https://www.cubemars.com/product/ak60-6-v3-0-kv80-robotic-actuator.html)).
The official page lists $229.90 and says the package includes a driver, but its
option selector is ambiguous; confirm the integrated-driver configuration in
the cart before treating $4,759 as a quote.
For an explicitly agile build, twelve identical AK80-9 units are the safer
starting point and simplify spares.

The much smaller **AK45-10 is rejected for a primary leg axis in this design**.
At Ø53 × 43 mm and 260 g it is 46% narrower and 47% lighter than an AK80-9, but
its 7 N·m peak / 2.5 N·m rated torque is respectively 68% / 72% lower. It is
also a 24 V actuator, so it cannot share the proposed 12S motor bus without a
separate conversion stage. It could be evaluated for a later neck or tail axis,
but using it at the hips or knees would solve the visual packaging problem by
discarding the torque margin that makes the 2–3 m/s program credible
([CubeMars AK45-10](https://www.cubemars.com/product/AK45-10-robotic-actuatuor.html);
[small-driver 18–28 V working range](https://www.cubemars.com/images/file/20250522/1747899365958473.pdf)).

The closest published reference actuator is the MIT Mini Cheetah module:

| Parameter | CubeMars AK80-9 V3 | MIT Mini Cheetah module |
|---|---:|---:|
| Envelope | Ø98 × 38.5 mm | Ø96 × 40 mm |
| Actuator mass | 490 g, conservative | 480 g |
| Peak / maximum torque | 22 N·m peak | 17 N·m maximum |
| Rated / continuous torque | 9 N·m rated | 6.9 N·m continuous |
| Output speed | 40.8 rad/s rated; 59.7 rad/s no-load | 40 rad/s maximum at 24 V |
| Twelve-actuator mass | 5.88 kg | 5.76 kg |

The physical envelope, mass, and headline torque-speed class are unusually
close. Even at the proposed 12 kg upper mass limit, the simple peak
torque/body-mass proxy is about 1.83 N·m/kg versus 1.89 N·m/kg for the 9 kg Mini
Cheetah; the rated/continuous proxy is about 0.75 versus 0.77 N·m/kg. These are
only coarse comparisons—joint geometry and duty cycle still control the actual
foot-force envelope. The Mini Cheetah reached a highest stable treadmill speed
of **3.7 m/s** with mature model-predictive and whole-body control. The same
experiment briefly observed 4 m/s before loss of balance
([actuator thesis](https://hdl.handle.net/1721.1/118671);
[high-speed locomotion paper](https://arxiv.org/abs/1909.06586);
[hardware context](https://arxiv.org/abs/2110.02799)).

This is an existence proof for the actuator and mass class, **not actuator
equivalence or evidence that a new chassis will immediately match it**.
CubeMars's "rated" torque is not the same test protocol as MIT's "continuous"
figure, and the catalog page does not provide directly comparable dynamic-duty,
output-inertia, or current-loop validation. This robot is also heavier and adds
a shell and remote transmissions. Dynamometer-test one AK80-9 and one complete
remote-knee leg before freezing the structure.

### $10k planning allocation

| Cost block | Planning allowance | Basis |
|---|---:|---|
| 12× CubeMars AK80-9 V3 actuators | **$5,759** | Current official unit listing; driver/encoder included |
| DIY aluminum/carbon frame, limb links, and feet | **$600** | Assumes owner fabrication and access to a printer/basic shop |
| Four remote-knee transmissions, output encoders, and one-leg fixture | **$500** | Provisional allowance; selection must pass the fixture gate below |
| 12S battery, charger/bench supply, contactor, fuse, precharge, monitoring, and DC-DC conversion | **$900** | Sourced representative stack below; short test-session endurance |
| Jetson-class compute, real-time controller, and isolated CAN | **$450** | Separate high-level and deterministic safety/control layers |
| Depth/RGB camera, independent body IMU, and battery monitors | **$350** | Includes a ~$269 OAK-D Lite-class camera |
| E-stop, connectors, harnesses, and test wiring | **$250** | Safety and serviceability allowance |
| Lightweight head, torso skin, and passive tail | **$250** | Foam/thin composite; no active morphology |
| Structural and ordinary breakage reserve | **$450** | Does not guarantee a complete spare actuator |
| Freight, tax, price movement, and unallocated contingency | **$491** | Location- and vendor-dependent |
| **Program target** | **$10,000** | Expect roughly **$9k–11k** in practice |

The $500 transmission allowance covers four nominally 1:1 remote-knee drives:
five reinforced belts (one spare), eight matching pulleys, shafts/hubs,
bearings, tensioners/idlers, guards, and four output-side magnetic encoder
boards plus magnets and wiring. Gates lists 5M PowerGrip GT3 belts in
9, 15, and 25 mm widths, but **5M is only a starting family, not a frozen
selection**. Use the manufacturer's drive-design procedure to select pitch,
width, tooth count, center distance, wrap, and installation tension against
measured actuator torque and shock load
([Gates PowerGrip GT3](https://www.gates.com/us/en/power-transmission/synchronous-belts/rubber-synchronous-belts.p.9356-000000-000000.html);
[drive-design manual](https://www.gates.com/content/dam/documents-library/catalogs/powergrip-gt3-drive-design-manual-en.pdf)).
The motor encoder alone cannot reveal output-side compliance or lost motion, so
V1 includes four 14-bit-resolution AS5048A SPI adapter boards, currently about
$18 each, at the anatomical knees
([Infineon AS5048A datasheet](https://www.infineon.com/assets/row/public/documents/24/49/infineon-as5048a-as5048b-datasheet-en.pdf);
[Mouser adapter board](https://www.mouser.com/en/ProductDetail/Infineon-Technologies/AS5048A-TS_EK_AB)).
Resolution is not accuracy: validate nonlinearity, latency, EMI, magnet
alignment, and transmission deflection on the fixture, then protect and
strain-relieve the evaluation boards before dynamic tests.

This remains a planning allowance, not a purchase-ready transmission BOM. Buy
one belt, two pulleys, one encoder, and the bearings for the single-leg fixture
first. Freeze and duplicate the drive only after it survives commanded torque,
impact, thermal, tooth-jump, backlash, bandwidth, and EMI tests. If that
validated design cannot be sourced and built inside $500—or requires
professional pulley machining—the honest result is a program above $10k, not a
quiet cut to the safety or actuator budget.

The priced power items below total about **$825** against the $900 power
allowance, while the E-stop consumes roughly $50 of the separate $250
safety/wiring allowance. Precharge, the 58 V fuse holder, charge adapter,
power enclosure, motor/signal harnesses, and test leads consume the remaining
allowances. The robot-side design should impose a **100 A motor-bus burst
limit** and a lower continuous limit established by harness, connector,
contactor, fuse, power-monitor, and thermal tests; the battery manufacturer's
C-rating is not a safe system-current specification. At 44.4 V nominal, that
initial software limit caps input near 4.4 kW.

The listed Holybro module provides **one main motor-bus voltage/current
channel, not per-CAN-bus current**. Select its XT90 variant: the vendor rates
the module for 200 A continuous / 400 A for one second and supplies 8 AWG leads,
while this robot remains fused at 100 A. Its DroneCAN telemetry must still be
calibrated against bench instruments and demonstrate correct current sign and
range under regenerative load. Do not assume that it can simply share a
saturated motor bus: validate identifiers, termination, update rate, and
worst-case utilization on the fixture, or add a dedicated isolated CAN
interface within the compute/wiring allowance. Log that channel with
pack-surface temperature and each actuator's current and temperature telemetry.
During fixture and speed gates, spot-check every power connector with temporary
thermocouples or a calibrated IR instrument; do not raise the limit merely
because the pack advertises a higher discharge rating.

| Power function | Catalog part | Published evidence |
|---|---|---|
| Robot battery | Gens Ace G-Tech 12S 3.5 Ah LiPo, XT90 | **$189.99**, 44.4 V, 1.054 kg, nominal 278.1 × 48.7 × 37.9 mm, 10 AWG leads; vendor claims 80C ([Gens Ace](https://www.genstattu.com/gens-ace-sport-g-tech-3500mah-12s-80c-44-4v-lipo-battery-pack-with-xt90-plug/)) |
| Balance charger | Junsi iCharger X12 | **$179.99**, supports up to 12S LiPo and adjustable termination; requires an external 11–53 V DC supply ([MPI Hobby](https://mpihobby.com/products/icharger-x12-1100w-30a-12s-balance-battery-charger)) |
| Off-robot charger supply | Mean Well RSP-1000-48 | **$267.90**, 48 V / 21 A, universal AC input ([DigiKey](https://www.digikey.com/en/products/detail/mean-well-usa-inc/RSP-1000-48/7706283)) |
| Main contactor | Altran AREVS150-BAN | **$54.45**, normally-open, bidirectional, 200 A continuous, 12 V coil, and auxiliary contact; bidirectionality matters because braking returns energy toward the pack ([DigiKey](https://www.digikey.com/en/product-highlight/a/altran-magnetics/arevs150-series-high-voltage-dc-contactors)) |
| Main fuse | Littelfuse BF1 142.5631.6102 | **$4.70**, bolt-down 100 A / 58 VDC slow-blow fuse with 1 kA interrupt rating; use a matching **58 V-rated** MIDI holder ([DigiKey](https://www.digikey.com/en/products/detail/littelfuse-inc/142-5631-6102/2515912); [holder family](https://www.digikey.com/en/product-highlight/l/littelfuse-commercial-vehicle-products/midi-498-series-bolt-down-fuseholder)) |
| Logic rail | Mean Well RSD-60L-12 | **$42.60**, enclosed 18–72 V input to 12 V / 5 A DC-DC ([DigiKey](https://www.digikey.com/en/products/detail/mean-well-usa-inc/RSD-60L-12/7706266)) |
| Motor-bus voltage/current telemetry | Holybro PM08-CAN Power Module, 14S/200 A, XT90 | **$85**, DroneCAN, 7–60.9 V, 200 A continuous / 400 A for one second, 8 AWG leads, ±0.1 V voltage and ±5% current accuracy; validate regenerative-current sign and transient response ([Holybro](https://holybro.com/collections/dronecan-power-module/products/dronecan-pm08-power-module-14s-200a)) |
| Physical E-stop | Schneider Harmony XB4BS8445 | Latching, turn-to-release 40 mm mushroom with one normally-closed and one normally-open contact; street price varies around **$45–55** ([Schneider Electric](https://www.se.com/us/en/product/XB4BS8445/harmony-emergency-stop-latching-turn-release-red-40-mm-1-nc-and-1-no/)) |

This table is a procurement baseline, **not a wiring diagram**. Before mounting
all twelve actuators, validate connector temperature, fuse behavior, contactor
dropout, precharge, undervoltage cutoff, regenerative voltage, and the physical
E-stop on a one-leg fixture. Put the E-stop's normally-closed contact in the
12 V contactor-coil circuit so opening the motor bus does not depend on
software; feed the logic rail from a separately fused auxiliary branch so the
controller can command zero torque and preserve logs. Charge LiPo packs in a
fire-resistant location under supervision and store them at the charger
manufacturer's storage voltage. Do not assume this RC pack contains a
conventional on-pack BMS. Use balance charging, install the independent BS12
cell alarm, log total bus voltage/current and pack-surface temperature, and
record the BS12's per-cell minima after each run. The BS12 is an operator alarm,
not a controller-readable automatic cutoff. Until the controller has validated
shutdown logic for total voltage and temperature, testing remains supervised
and short-duration. Use appropriately rated bench instrumentation to capture
regenerative bus transients on the one-leg fixture; the logging channel is not
a substitute for that test. Use a keyed, verified adapter between the pack's
two G-Tech balance connectors, the charger, and the BS12; never hand-probe or
repin energized cell taps.

Representative orderable control/sensing parts within their allowances are:

| Function | Catalog part | Published evidence |
|---|---|---|
| High-level perception/policy compute | NVIDIA Jetson Orin Nano Super developer kit | **$249**, up to 67 TOPS, 7–25 W ([NVIDIA](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)) |
| Deterministic motor/safety controller | Teensy 4.1 plus isolated CAN transceivers | 600 MHz Cortex-M7, three CAN controllers (one CAN-FD); external transceivers are required ([PJRC](https://www.pjrc.com/store/teensy41.html)) |
| Forward RGB + stereo depth | Luxonis OAK-D Lite | **$269**, onboard stereo/RGB processing and BMI270 IMU ([Luxonis](https://shop.luxonis.com/products/oak-d-lite-1)) |
| Independent prototype body IMU | Adafruit BNO085 breakout | **$24.95**, fused 9-DOF orientation; prototype only until measured update rate, timestamps, latency, and vibration tolerance pass the state-estimation gate ([Adafruit](https://www.adafruit.com/product/4754)) |
| Independent per-cell warning | Chargery BS12 V2.3 | **$24.50**, in-stock 2S–12S monitor with configurable audible/visual low-cell alarm and stored minimum for each cell; not a controller cutoff ([Chargery official store](https://www.chargerystore.com/index.php?product_id=52&route=product%2Fproduct)) |
| Pack-surface temperature | Adafruit high-temperature waterproof DS18B20 | **$14.95**, 1-Wire digital output, −55 to 125°C range, and ±0.5°C specified accuracy from −10 to 85°C; bond and strain-relieve it against the pack surface ([Adafruit](https://www.adafruit.com/product/642)) |

The sensing allocation supports motor position/velocity, four independent
output-side knee angles, actuator current and temperature, one main motor-bus
voltage/current measurement, pack-surface temperature, an independent per-cell
alarm, an independent body IMU subject to the validation gate above, and
forward RGB/stereo depth. The four explicitly priced sensing/perception items
total about **$333** against the $350 allowance. It does not yet claim a
deterministic low-latency IMU or controller-readable per-cell telemetry. Initial
foot contact can be estimated from joint kinematics and actuator current.
Research-grade six-axis force/torque sensors, lidar, multiple depth cameras,
and an active head/tail do **not** fit this budget. This is a complete set for
supervised locomotion and basic autonomy, not a contact-metrology package.

Outsourcing the entire structure to a CNC shop can add $1k–3k. Treat access to
a 3D printer, drill press, saw, and simple fixturing as a budget assumption.

### Performance gates

These are robot program gates, not guaranteed performance or an estimate of the
extinct animal's speed:

- **2.0 m/s:** repeatable primary acceptance on level, high-traction ground,
  including a controlled stop and no protection fault or uncontrolled fall;
- **2.5 m/s:** tuned success target after state-estimation, gait, current, and
  thermal tuning;
- **3.0 m/s:** several-second straight-line sprint stretch on a prepared
  surface; and
- **above 3 m/s:** unsupported commitment for a first clean-sheet build.

Qualify the bare chassis at 2 m/s before fitting animal parts. Add morphology as
ballast in measured increments, and do not add an actuated neck, jaw, or tail
until the dressed robot repeats the 2 m/s test without a thermal, current, fall,
or emergency-stop regression. Reaching the upper end will likely require MPC +
whole-body control or a simulation-trained policy with accurate state
estimation and domain randomization.

### Shape and movement realism

A convincing exterior silhouette and plausible erect terrestrial-quadruped
movement are achievable; X-ray-level anatomical fidelity is not. Use passive
compliant ankles and segmented toes to soften contact without adding four more
motors. Keep the head hollow and the distal tail passive for the first speed
qualification. An active neck or tail base belongs in a later morphology phase
after the dressed robot repeats the 2 m/s acceptance test.

Neither anatomical source below provides a species-specific maximum-speed
estimate, and exact _Dibothrosuchus_ movement is unknowable from skeletal
material alone. For the proposed 25–30 cm effective leg, 2 m/s corresponds to a
Froude number of approximately 1.4–1.6 and 3 m/s to approximately 3.1–3.7,
using `Fr = v²/(gL)`. That supports calling the commanded behavior dynamically
"running"; it does **not** recover the animal's biological top speed
([Alexander & Jayes, 1983](https://zslpublications.onlinelibrary.wiley.com/doi/10.1111/j.1469-7998.1983.tb04266.x)).

### Which species fits this chassis?

Small adult quadrupedal dinosaurs are scarce. A strict dinosaur identity
therefore usually requires a juvenile reconstruction; allowing other Mesozoic
animals produces a more natural full-size match.

| Species | Representation on this chassis | Engineering fit and scientific caveat |
|---|---|---|
| **_Dibothrosuchus elaphros_** ⭐ | _Dibothrosuchus_-inspired exterior | **Primary physical reference.** This Early Jurassic non-crocodyliform crocodylomorph is known from substantial skeletal material, including a nearly complete skull and mandible plus partial postcranium. Its estimated total length was 1.3 m, its long, slender limbs were interpreted as adapted to quadrupedal terrestrial gait, and later work discusses digitigrade forelimb evidence. That larger documented scale is a more honest engineering reference for concealing Ø98 mm proximal actuator clusters, although this packaging conclusion is an engineering inference and the shell is not an exact replica. It is Mesozoic, but not Dinosauria ([Wu & Chatterjee, 1993](https://www.tandfonline.com/doi/abs/10.1080/02724634.1993.10011488); [Ruebenstahl et al., 2022](https://anatomypubs.onlinelibrary.wiley.com/doi/10.1002/ar.24949)). |
| **_Terrestrisuchus gracilis_** | Secondary morphology-stress case | **Aggressive stress test, not the physical or primary simulation baseline.** Its highly gracile, digitigrade, terrestrial quadrupedal anatomy is useful for testing how a narrower dressed morphology changes inertia and packaging. It must not replace the actual _Dibothrosuchus_ hardware geometry in controller training. All sampled specimens were skeletally immature and no known specimen exceeded 1 m, so a larger adult reconstruction would be speculative ([Spiekman et al., 2024](https://onlinelibrary.wiley.com/doi/full/10.1002/spp2.1577)). |
| **Juvenile _Psittacosaurus_** | Early juvenile-inspired | **Best dinosaur/speed compromise.** Early juveniles were quadrupedal and shifted toward bipedality during growth, so it must not be presented as an adult quadruped ([Landi et al., 2021](https://onlinelibrary.wiley.com/doi/full/10.1111/pala.12529)). |
| **Juvenile _Protoceratops_** | Cat- to small-dog-sized juvenile | **Clearest familiar four-legged dinosaur silhouette—an engineering/curatorial judgment, not a speed inference.** Its stockier form, shorter-looking legs, and large head/frill work against maximum agility. Keep the frill hollow and non-structural. Adults were about 2 m / 180 kg, while young juveniles could be cat-sized ([AMNH](https://www.amnh.org/exhibitions/dinosaurs-ancient-fossils/display-or-defense/my-what-a-big-skull-you-have)). |
| **Juvenile _Mussaurus patagonicus_** | Approximately one-year-old | **Excellent mass match but uncertain posture.** Known juveniles were estimated at 8.3–10.9 kg, but their quadrupedal-versus-bipedal stance was ambiguous; neonates were more clearly quadrupedal ([Pol et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8531321/)). |
| **_Repenomamus giganticus_** | Approximately adult-scale | **Good body-size class if Mesozoic mammals are acceptable.** It was badger/jackal-sized and avoids a juvenile label, but its mammalian, lower-slung silhouette is less naturally compatible with the selected cursorial chassis ([Hu et al., 2005](https://www.nature.com/articles/nature03102)). |

**Recommendation:** build the first physical chassis and its primary simulation
as _Dibothrosuchus_-inspired, using the same geometry, mass properties, and
transmission model in both. Use _Terrestrisuchus_ only as a secondary gracile
morphology-stress case. If the platform must be a dinosaur, use an explicitly
labeled juvenile _Psittacosaurus_; if unambiguous quadrupedal dinosaur anatomy
matters more than 3 m/s, use juvenile _Protoceratops_. Design the load-bearing
chassis independently of the skin so those lightweight exteriors can be
swapped. Keep the **entire removable morphology at or below 0.6 kg** for
initial tests. Gate it on both net center of mass and rotational inertia: place
ballast at the actual head/tail radii, then reject the design if pitch/yaw
tracking or impact loads regress. Do not “balance” a heavy head by adding a
heavy tail.

---

## 5. Bipedal runner options — is it a motor-availability limit?

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

## 6. Reference points for the platform decisions

| Robot | Build model | Mass | Speed evidence | Price evidence |
|---|---|---:|---|---:|
| [Pupper V3](https://pupper-v3-documentation.readthedocs.io/en/latest/) | Open CAD/software; owner-built | 3 kg | Course target **0.75 m/s in simulation**; physical top speed unpublished | About $2,000; live BOM **$2,158.91** |
| [Stanford Pupper (paper)](https://arxiv.org/abs/2110.00736) | Open, independently reproduced | 2.1 kg | Dynamic omnidirectional reference gait + 5 m sprint benchmark | Historical build **under $2,000** |
| [Unitree Go2 Air](https://www.unitree.com/go2/) | Prebuilt; no official secondary development | ~15 kg | Vendor-rated **2.5 m/s** | **$1,600** before tax/freight |
| [Unitree Go2 Pro](https://www.unitree.com/go2/) | Prebuilt; no official secondary development | ~15 kg | Vendor-rated **3.5 m/s** | **$2,800** before tax/freight |
| [MIT Mini Cheetah](https://arxiv.org/abs/1909.06586) | Research platform; custom chassis/control | ~9 kg | Demonstrated **3.7 m/s** with mature control | Not a retail kit; performance reference only |

The decision is not simply "quadruped versus biped." It is **replicate a proven
open platform versus design a new robot**. For a solo first build, replication
is the lower-risk use of the first $5k.

---

## 7. Recommendation & caveats

**Under $5k, build a juvenile-Psittacosaurus Pupper V3.** It is the best first
use of a sub-$5k budget because the exact chassis is open, documented, small
enough to work on safely, already includes the essential
camera/IMU/proprioceptive sensor set, and uses orderable integrated motors. Keep
V1 to a passive removable morphology and preserve the stock robot beneath it.

**Near $10k, build a _Dibothrosuchus elaphros_-inspired 12-DOF custom
quadruped around catalog AK80-9 actuators.** This is the first budget tier in
this study where an owner-built chassis has a credible 2–3 m/s mechanical
envelope. Make 2 m/s the repeatable acceptance target, 2.5 m/s the tuned goal,
and 3 m/s a short straight-line stretch. No custom motor or planetary gearbox
is required, but custom links, remote-knee transmissions, frame, harness,
safety system, and locomotion software are.

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
- **$10k is a planning target, not a fixed vendor quote.** Freight, tax, CNC
  outsourcing, or a complete spare-actuator set can push the custom quadruped
  toward $11k–13k.
- **Spares are load-bearing, not padding** — dynamic bring-up cracks frames,
  shears encoder magnets, and fries drivers.
- **Continuous ≠ peak:** Pupper's documented actuator figures are about
  3.5 N·m peak versus 1.0 N·m continuous with air cooling. Do not infer sustained
  speed from peak torque.
- **Morphology is a curatorial call:** each quadruped adds a new species rather
  than reusing the roster's raptor. Label the $5k _Psittacosaurus_
  "juvenile-inspired"; label the $10k _Dibothrosuchus_ as an Early Jurassic
  crocodylomorph, not a dinosaur, and describe the shell as "inspired" rather
  than an exact life reconstruction.
