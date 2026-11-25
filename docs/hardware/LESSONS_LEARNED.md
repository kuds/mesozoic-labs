# Lessons learned

What the project tried on the way to a physical robot, what happened and what carries over. Nothing has been built or
bought. Results are in simulation (MuJoCo models) or desk estimates; prices are dated list prices or cost-model
estimates, not quotes. Sizes and size terms (crown height, frill height, the Hybrid and Tapered bodies):
[SIZES.md](SIZES.md). Costs and cost terms (landed, turnkey): [PARTS_AND_COSTS.md](PARTS_AND_COSTS.md). Status and
records: [README.md](README.md).

## In short

- **Measure motor delay first.** Motor delay is the time from a command to the motor acting on it. The 75 cm T. rex
  walked reliably in simulation only with 6 ms or less; the design estimate is close to that and unmeasured.
- **Start with four legs.** A biped must balance at every moment; a slow quadruped with three feet down need not. The
  T. rex's controllers mostly fell under delay; the Triceratops trot passed at 10 ms.
- **The small tail servo ran over its limit.** A tail swinging against the body roughly halved its load.
- **Feet and grip limit a small robot before torque does.** In simulation a stiff flat sole tipped onto its edges; a
  rounded sole was the only one of four that passed every rule.
- **Size buys little:** speed grows slowly with size, and drives, computers and safety parts barely shrink
  ([SIZES.md](SIZES.md)).
- **Keep the walking loop on a microcontroller,** so that a crash of the main computer still ends in a controlled stop.
- **Get quotes for the frame, coverings and assembly first:** they carry most of the cost uncertainty.
- **Treat results without records as unverified.** The best walking numbers of PR #668 had none, and independent checks
  caught wrong prices and missing parts in the project's own lists.

## What was tried

### July to October 2026: feasibility notes and first concepts

- **19 Jul 2026, feasibility notes.** Parts estimates priced a walking Compsognathus and a walking Velociraptor. There
  was no proven recipe for a two-legged runner under $10k: small bipeds are limited by control, not motors. A
  [sim-to-real plan](SIM_TO_REAL_PLAN.md) named six gaps between simulation and hardware.
- **24 Jul 2026, quadrupeds first.** The recommended first build became a juvenile Psittacosaurus shell on the
  open-source Pupper V3 quadruped, with a larger Dibothrosuchus-inspired quadruped as a near-$10k tier.
- **6–7 Sep 2026, Compsognathus robot, revision B.** The robot model stood in simulation at up to 85.9 % of its servos'
  torque cap. No walking has been shown.
- **24–28 Oct 2026, first T. rex concepts ([#635][pr635] and others).** Unmerged pull requests sketched a 27 cm mini, a
  medium body, a 60 cm skeleton and a one-metre concept; two stood and crouched in simulation, none was shown walking.

Detail: [walking parts][bom], [running robots][rsa], [Compsognathus](../reviews/COMPSOGNATHUS_ROBOT_DESIGN.md).

### November 2026: the T. rex form study

- **7 Nov 2026, posture study and round one.** In a posture study no 60 cm body could shift its weight onto one leg,
  because the sideways hip load was too high (see the balance lessons below). Round one compared four heights from 60 to
  120 cm and moved both hip-roll motors to the centreline, which cut that load.
- **7–8 Nov 2026, round two.** Twenty 60 and 75 cm bodies were compared. At an animal-like pace every 75 cm body
  overloaded its drives until four targeted drive upgrades fixed it. Picked: two narrow 60 cm bodies and a 75 cm Hybrid.
- **8–10 Nov 2026, closed-loop trials and speed refit.** A balance controller tuned without motor delay walked all three
  bodies slowly and turned them, but none walked faster, took a push while walking or walked on carpet. The 75 cm body
  walked even without balance feedback; no 60 cm body walked at 10 ms of delay, and learned controllers added nothing.
- **10–14 Nov 2026, character over speed.** The aim became a powered head and tail, not speed. A lifelike walk (hip bob,
  toe-off, body roll, a counter-swinging tail) and an expressive head were modelled; the 75 cm body went first.
- **14–16 Nov 2026, 75 cm Jetson build proposal (PR #668).** It added a Jetson Orin Nano Super with a real-time
  control layer on an isolated core driving the CAN drives over USB-CAN. Its walking table had no records and half its
  result lines were not run as claimed: unverified, closed as superseded.
- **15 Nov 2026, 75 cm agility trials.** With the same balance controller the 75 cm body stayed up in 20 of 20 runs at
  0.24–0.40 m/s with 6 ms of delay or less, fell in every 0.40 m/s run at 8 ms, and kept at most 3 of 20 up at the
  assumed 10 ms. The stop passed; pushes and the quick 90° turn failed, and the tail servo exceeded its planning load.
- **16 Nov 2026, staged buy gate.** Each purchase was to wait for a pass mark set in advance: first a trained policy in
  simulation, then a bench, a legs-only robot and the full build (costs in [PARTS_AND_COSTS.md](PARTS_AND_COSTS.md)).
  Nothing was bought.
- **17 Nov 2026, first training trial.** A short training run on the 75 cm walk gave no gain over its baseline
  controller.
- **18 Nov 2026, the pause.** A biped must balance at every moment and needs many rounds of hardware testing, for which
  the project lacked the equipment and time; in simulation its controllers mostly fell under delay and training gave no
  gain. A slow quadruped with three feet down stays balanced, so the work moved to a Triceratops and the T. rex paused.

### November 2026: the Triceratops

- **17–18 Nov 2026, design study ([#669][pr669]).** A 75 cm, 15 kg quadruped concept stood in simulation; walking, heat
  and bus delay were not checked.
- **18 Nov 2026, 75 cm parts lists.** Planned 20 kg and 15 kg builds were priced well above the $20k target. In a slow
  simulated crawl their stiff soles tipped.
- **18 Nov 2026, the 60 cm option.** The planned first build: 60 cm at the frill and about 12 kg, with the 75 cm
  T. rex's drive types (ten X4, two X6); landed estimate $20.1k–$23.0k against the $20k target.
- **19 Nov 2026, walking trials.** A model-based trot passed every rule on two floors under two contact settings at
  10 ms of delay: no falls in 40 flat-floor trials, 0.247–0.254 m/s at 0.25 m/s commanded, worst joint at 0.86 of its
  continuous rating. In a wider sweep it kept walking with up to 60 ms of delay. Ramps were its weak side: 27 % slower
  at 5°, and at 8° the hind knees reached 1.09 of their rating.
- **21–22 Nov 2026, review and feet.** A review found the parts, CAD and safety design not ready to order; of four feet
  compared in simulation, only a rounded capsule sole passed every rule.

Detail: [walking trials][walk60], [delay sweep][sweep60].

## Lessons that carry over

### Actuators and motor delay

- **Measure motor delay before trusting a simulated walk.** The design estimate for a microcontroller on two CAN buses
  maps to 2–4 ms of simulated delay (2–6 ms read cautiously), every term of it unmeasured; a Jetson driving one USB-CAN
  channel was estimated at up to 12 ms.
- **A controller fitted without delay did not tolerate it,** and adding delay afterwards still fell at 10 ms. Training
  through random delay with a realistic servo model, as Microduck does
  ([REFERENCE_MICRODUCK_2026_10.md](REFERENCE_MICRODUCK_2026_10.md)), was planned; a short trial gave no gain.
- **Size on continuous torque at the real voltage.** Hobby servos sustain about 30–50 % of stall torque. Reseller text
  caps the X4 at 83 rpm above 24 V; the 60 cm T. rex knee hits rated speed at the animal pace at 21 V (computed).
- **Upgrade only the drives that bind, and share them.** Targeted upgrades beat blanket headroom at 75 cm
  ([SIZES.md](SIZES.md)), and one drive family (MyActuator X4 and X6) is planned for both robots.
- **Sit the robot down before cutting power.** The geared leg drives have no brakes, so a robot whose drives are
  switched off collapses; every stop must first command a damped sit. The planned foreleg bench checks this.

### Balance and body plan

- **Leg geometry sets the hip loads, not the covering.** Every 60 cm shape needed 9.5–9.9 N·m of hip roll on one leg
  against a 6 N·m sustained allowance; narrower feet cannot remove it, while hip-roll motors on the centreline cut it to
  2.9 N·m (model).
- **Use the battery as a counterweight.** In the tail base it cut the 60 cm hip-pitch load from 8.0 to 4.5 N·m; without
  it the knee ran past its speed limit (model).
- **Bigger is steadier, but it hits harder.** The 75 cm T. rex walked without balance feedback where the 60 cm bodies
  fell, but a fall carries 63 J against 36 J.

### Heads and tails

- **The small tail servo ran over its planning load,** an assumed sustained limit that was never bench-tested. The 75 cm
  T. rex's hobby tail servo was 5 % over at 0.35 m/s and 19–27 % over at 0.50 m/s; about half of the modelled load came
  from joint friction and damping values with no source.
- **Swing the tail against the body.** A counter-phased tail cut that load at 0.40 m/s to 0.40–0.48 of the limit,
  against 1.02–1.11, and is part of the lifelike look.
- **The tail sets the turning room.** In a planned quick 90° turn the 1.76 m T. rex's tail tip sweeps a 2.22 m circle;
  the 75 cm Triceratops needs a 1.62–1.92 m circle.
- **An expressive head costs little mass but heats the neck.** A neck turn and a jaw add 0.150 kg, 2.2–2.7 min less
  runtime (model) and about $250 (75 cm cost estimate, 14 Nov 2026); the 60 cm neck hold rose from 0.75 to 0.84 of its
  limit.

### Power, compute and runtime

- **Keep the walking loop on a microcontroller.** With the loop on Linux a crash cuts torque and a biped falls. With it
  on the microcontroller the stop stays controlled, and the main computer does not change the delay.
- **Measure computer power before sizing the battery.** Round one booked the Jetson at 31 W, about twice published
  measurements; in round two the Pi 5 set at 9.5 W was the main reason runtimes rose from 30–34 to 38–46 min (model).
  Runtimes by computer: [SIZES.md](SIZES.md).
- **Plan power and heat with the computer.** The Jetson needs a regulator (its carrier takes 9–20 V; the 6S pack gives
  21.0–25.2 V), and its heat in a closed chest was not addressed. Drive makers' heat ratings are for open air.
- **Battery plus DC input, not a trailing tether:** 5 m of floor cable drags 1.2–3.6 N against a 7–12 N tip threshold.
- **Match voltages and leave bus room.** 48 V drives starve on a 6S (24 V) pack, and 12S (48 V) needs a regen clamp and
  precharge. Twelve drives on two classic CAN buses load them 67–81 % at 500 Hz; add a third bus or one per leg.

### Feet, floors and grip

- **Grip, not drive torque, limits the small robot.** The foot's twist on the floor (stance-foot spin) sets the friction
  a walk needs: in the model the narrow 60 cm T. rex needed up to 0.75 for a brisk walk and the 75 cm body with bigger
  feet up to 0.56, against an estimated 0.42–0.8 for soft polyurethane treads on laminate and vinyl (unmeasured).
- **Carpet is unknown.** No carpet model is sourced, and every T. rex walking start on the thin-carpet model fell. Plan
  demonstrations on hard venue floors.
- **A stiff flat sole tips; a rounded one rolls.** The Triceratops's roll-rigid sole tipped onto its edges in a slow
  crawl; only a capsule sole passed every rule (simulation).
- **Simulate only sensors on the parts list:** the Triceratops controllers used foot-load signals it had dropped.

### Costing

- **Get quotes for the frame, coverings and assembly first.** In the 75 cm T. rex estimate (14 Nov 2026) these lots,
  scaled from a 60 cm estimate, make up most of the gap between the low and high totals
  ([PARTS_AND_COSTS.md](PARTS_AND_COSTS.md)).
- **Price where and when you will buy.** On 18 Nov 2026 RobotShop asked about twice the drive maker's own US-store
  price, and duty ran from 0 % (US stock) to about 41.5 % (from China), to confirm per order. Check what a list price
  includes: the CubeMars drive price used in July left out the driver board.
- **Buy in stages with pass marks set in advance,** check every list independently, and watch lead times: foot load
  cells had a 21–23 week lead time on 18 Nov 2026.

## Open questions

Nothing has been measured, so the first bench holds most of the answers:

- each drive type's real delay on its bus (every verdict so far assumes 10 ms, also the planned foreleg bench's pass
  mark), and its heat inside a closed shell;
- the tail servo's continuous limit, and the joints' friction and damping;
- whether the X4 is capped at 83 rpm above 24 V, and which X6 version ships at $428;
- whether a 52.5 V clamp keeps the 48 V bus under 55 V in a hard stop, and tread grip on the target floors;
- the leg drives' rotor inertia (before or after the gearbox), the motion mode's gain limits, the hip-pitch drive's
  speed at 24 V, which hip-pitch part ships and which encoder closes the loop.

Still open in simulation and design: whether a policy trained through delay walks the T. rex at 10 ms (full training
never ran); whether quick turns need more thigh-twist range; the 60 cm Triceratops hind knees (1.09 of rating on an 8°
ramp); the choice of computer.

## If the T. rex resumes

- **Proposed timing (kickoff plan, planning estimates).** Flagship milestone end 2027 or early 2028. For a built robot:
  simulation restarts by 1 Mar 2027; delay test on the Triceratops foreleg bench, Feb–Apr 2027; leg drives ordered in
  Q2 2027 once that bench confirms them; legs-only prototype Q3–Q4 2027; walking under a catch Q4 2027 to Q1 2028.
- **First,** measure the delay and re-run the verdicts at it. Then, before buying T. rex parts, a policy trained through
  random delay and a realistic servo model must walk at 0.30 m/s on hard floors at every delay of 0–10 ms.
- **Kept:** the 75 cm Hybrid body and its reference walks. [README.md](README.md) lists the branches and pull requests
  that hold the records.

[bom]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/HARDWARE_BOM.md
[rsa]: https://github.com/kuds/mesozoic-labs/blob/87bafc864fefb16b040bd7a997fea8273bfc6649/docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md
[walk60]: https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/walk/results/walk60_2026-11-19/RESULTS.md
[sweep60]: https://github.com/kuds/mesozoic-labs/blob/7bfeef5334d0d7484e5428f1b77d2d1847d35c8c/hardware/triceratops_build/walk/results/sweep_walk60/summary_walk60.md
[pr635]: https://github.com/kuds/mesozoic-labs/pull/635
[pr669]: https://github.com/kuds/mesozoic-labs/pull/669
