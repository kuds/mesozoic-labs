# T-Rex size decision: miniature, 50 cm and 80 cm

**Recommendation: make 50 cm the next design target, beginning with a loaded
motor bench and packaging study.** It offers enough physical presence for an
expressive character while remaining a manageable solo prototype. Preserve the
miniature as the inexpensive learning platform. Treat 80 cm as a later flagship
when a venue, collaborator or audience need justifies its greater development
cost and logistics.

The companion [robot variants and build plan](ROBOT_VARIANTS_AND_BUILD_PLAN.md)
proposes repository structure and a phased mini-to-50 cm schedule, with separate
identities, calibrations and evidence for each robot design.

This is a preliminary engineering and project-cost study dated **2026-10-24**.
It includes current manufacturer actuator specifications, explicit mass/cost
allowances and reproducible arithmetic. **Neither larger robot has CAD fit,
dynamic simulation or hardware walking validation.** The miniature's existing
diagnostic gait also fails longer trials. Size does not solve that control problem.

## Decision tiers

These rankings are judgments for the current goal: a distinctive, cute T-Rex
that can eventually walk and perform expressive gestures. They are not scores
for demonstrated performance or predicted commercial returns.

| Tier / role | Height | Mass basis | First-project USD | Main benefit | Main consideration |
| --- | ---: | --- | ---: | --- | --- |
| **Best low-cost learning platform** | 25.8 cm | 0.825 kg current tail model; upgraded hardware mass unresolved | **2,100–3,100** provisional | Portable, inexpensive parts and accessible maker replication | Tight packaging and leg thermal margin; needs a better gait |
| **Best next investment** | 50 cm | 5.91 kg budget, including 0.50 kg reserve | **5,600–8,000** conditional | Clearer gestures, more service space, useful floor-level presence | Cheapest actuator choice has limited load margin and must pass bench tests |
| **50 cm fallback with stronger legs** | 50 cm | 6.73 kg with the same reserve | **9,200–11,700** | More catalog torque margin while retaining the medium-size format | Higher price and mass; still needs full mechanical/control validation |
| **Later live-demo flagship** | 80 cm | 9.85–11.41 kg budgets, including 1 kg reserve | **14,000–21,000** rounded | Most visible at events and in full-room interactions | Near-one-metre length, substantial fall fixture, transport and maintenance burden |

These are **one-off development-project allowances**, not finished robot prices
or production-unit costs. They include basic test equipment, spare/rework
allowances, modest cloud experimentation and contingency. They exclude labor,
tax, freight, import duties, new workshop machines, a new host computer,
outsourced engineering and event operations. They do not assume supplier
discounts or equipment already owned. The more expensive 50 cm fallback should
be part of the funding decision: USD 8,000 is not an assured ceiling for a walker.

The current all-XL330 miniature is not used as a supposedly qualified cheap
walker. Its new cost allowance provisionally budgets six XC330 upgrades and nine
XL330s, totaling USD 867.75 for 15 axes. This pays for an investigation; those
substitutions are **not** qualified by the old torque-limited simulation. The
added motor mass, camera, brackets and changed motor behavior need reconciliation.
This is a more explicit planning range than the older USD 2,000 allocation.

## What changes mechanically

The 50 cm candidate uses twelve AK45-10 V3.0 leg actuators and six character
axes: head yaw/nod, jaw, two independent up/down arms and tail yaw. Its leg
motors list 2.5 Nm rated torque. The limited similarity screen estimates about
2.02 Nm knee RMS, leaving little allowance for unfavorable packaging, sustained
crouch or extra mass. The first action is to measure low-speed heating and
improve the posture/load path, not order twelve motors. A stronger alternative
uses 4 Nm-rated RMD-X4-P12.5-10 joints. See the
[50 cm feasibility study](studies/trex_50cm/README.md) for sourced specifications,
the reconciled mass ledger, electrical architecture and stopping rules.

At 80 cm, two costed leg approaches use twelve 10.5 Nm-rated RMD-X4-P36-36-E
motors at 24 V or twelve 9 Nm-rated AK80-9 V3.0 motors at 48 V. The compact
high-reduction choice saves mass and cost, but its friction, reflected inertia
and impact response need comparison with the lower-reduction option. Neither
is selected for purchase. See the
[80 cm feasibility study](studies/trex_80cm/README.md).

Both larger concepts have six axes per leg, compared with the mini's five.
Ankle roll allows a richer stance/contact strategy, but does not establish
stable locomotion by itself. New dynamics, actuator identification, foot
contact sensing and controller development are required. Expect the visual
language of a small character biped—head tilts, short steps and expressive
pauses—but do not promise the MicroDuck gait from proportions alone.

Increasing the miniature by only 10% gives 28.4 cm. With equal density and
geometrically similar mass distribution, gravity torque scales with length to
the fourth power: **1.1⁴ = 1.464**, about 46% more. Real motors/electronics do
not scale with the shells, so that is a sensitivity case rather than a forecast.
The modest visual gain is not a reason to retain the same motors without
rechecking mass and loads. A 50 cm or 80 cm robot requires a new actuator class,
not a scale slider applied to the current configuration.

Disney's reported Olaf is **88.7 cm and 14.9 kg**, with 25 degrees of freedom
and a specialized controller. The 80 cm proposal is a comparable presence
category, not the same geometry, engineering maturity or budget. Its example
supports investigating expressive bipeds; it does not establish that our
T-Rex can match its performance. [Disney research paper](https://arxiv.org/html/2512.16705v2).

## Comparable cost structure

All amounts below are USD. Core costs include actuators, structure, cosmetic
parts, electronics, power distribution, battery and charger. Development
extras include basic bench equipment, spares/rework and cloud experiments.
Contingency applies once to the combined subtotal.

| Architecture | Actuators, included in core | Core parts and fabrication | Development extras | Contingency | Total first project |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mini, provisional upgrades | 867.75 | 1,373–1,838 | 300–650 | 25% | 2,091–3,110 |
| 50 cm, AK45 candidate | 2,601.54 | 3,862–5,102 | 600–1,300 | 25% | 5,577–8,002 |
| 50 cm, stronger RMD legs | 5,518.74 | 6,779–8,019 | 600–1,300 | 25% | 9,223–11,648 |
| 80 cm, compact 24 V | 6,801.84 | 9,202–11,702 | 1,200–2,600 | 30% | 13,522–18,592 |
| 80 cm, lower-ratio 48 V | 8,372.64 | 10,773–13,273 | 1,200–2,600 | 30% | 15,564–20,634 |

Actuator rows use listed manufacturer prices; other amounts are planning
allowances. For the AK80, conflicting official listings are handled by budgeting
the higher USD 579.90 figure until the exact driver-equipped package is quoted.
Mechanical brackets, shafts and bearings must be costed from the final layout.

The 50 cm Dynamixel comparison is approximately USD 10,051–12,476, but its
literal reference speed/load screen is unfavorable at 12 V. It is retained in
the detailed study for comparison rather than recommended as a drop-in solution.
The displayed low/high totals describe different spending assumptions, not
statistical confidence intervals or guaranteed maximum budgets.

Cash is only one part of the cost. Allow roughly **300–600 focused hours for
50 cm** and **600–1,200 for 80 cm**, with large uncertainty and additional
supplier delays. At 15–20 hours/week, these span approximately 4–9 months and
7–19 months. They are planning judgments, not delivery commitments. Paid
engineering can dominate the parts bill; price that separately using actual
rates and agreed scope. Building all three is not necessary: preserve reusable
software, fixtures and measured knowledge without assuming frames or learned
policies transfer unchanged.

## Appearance, sensors and serviceability

At every scale, keep an oversized but hollow head, compact snout, short arms,
broad feet and a tapered lightweight tail. Use a recognizable brow/eye shape,
restrained colors and a consistent motion vocabulary for a distinct identity.
Increasing the shell size should not mean making its walls proportionally thick.
Use the internal frame for loads and removable skins for appearance.

Expose selected hip/link mechanisms and accessible fasteners. Removable belly
and back panels should reveal the battery, electronics and cable connections.
Cover pinch points and protect wiring; leave service access and ventilation
around loaded actuators. At 80 cm especially, a removable tail and separated
transport supports are more valuable than an uninterrupted cosmetic shell.

The camera remains a **snout-mounted design intent**, with decorative eyes.
The existing simulation camera does not constitute a fabricated sensor mount.
Neither larger study assumes eye-mounted cameras, stereo vision or extra facial
motors. A single camera, pelvis IMU and joint/contact feedback are a sensible
starting set; validate actual camera field of view and near-foot visibility.

Simple arms and gentle tail wagging fit the preliminary torque allowances at
both larger sizes. The 50 cm tail scenario is 80 g and the 80 cm scenario is
180 g; these are strict lightweight design assumptions. Tail-root bearings
carry bending loads. The larger tail uses a stronger servo, and neither is
counted on for balance. CAD must still establish clearance and cable travel.

## Marketing and other practical considerations

The following are strategic judgments, not measured audience or revenue forecasts.

| Consideration | Miniature | 50 cm | 80 cm |
| --- | --- | --- | --- |
| Video and online storytelling | Strong close-up subject; easy to film on a small set | Clear gestures and human-scale context without a large stage | Strong scale reveal; needs wider framing and more filming space |
| Live demonstrations | Best with a screen or close viewing | Useful for small-group education and demos | Most visible across a room; stronger photo-op presence |
| Open-source adoption | Lowest barrier to replication, shipping and workshops | More expensive, but a useful research/maker reference | Likely fewer replicators; stronger institutional or sponsored-project fit |
| Transport and setup | Small protective case; low handling burden | Removable tail, carry case, floor mat and fall-test space | Large padded case/supports, vehicle planning, more setup time and likely a helper |
| Maintenance | Cheap spares, fiddly access | Best compromise between access and manageable parts | Expensive spares and heavier handling; larger cosmetic repairs |
| Main opportunity cost | Time spent overcoming tiny packaging limits | A substantial solo controls/mechanical project | Less iteration per dollar and longer delays before useful public evidence |

For online reach, a reliable mini or 50 cm robot with repeatable personality can
be more effective than an unreliable larger one. A recognisable reaction—look,
head tilt, small arm lift and gentle wag—plus clear sound and good framing does
not require 80 cm. Size alone does not predict attention, sponsorship or sales.

For outreach, demonstrate a repeatable three-to-five-minute routine, explain
what the visible mechanisms do, and provide a quick reset. Publish uncut walking
evidence, measured endurance and accessible build information. Those establish
credibility beyond attractive renders. Consider the 80 cm version when a museum,
event or collaborator specifically benefits from its physical presence and can
help with transport, staffing or funding.

Event equipment is separate from the engineering budget. Provisional additional
allowances for a transport case, demonstration mat, spare session battery and
simple signage are **USD 100–300 mini, 300–1,000 at 50 cm, and 1,000–3,000 at
80 cm**. These are not quotes; reconcile overlaps with owned equipment before
adding them. Travel, paid operators, venue requirements and insurance need
event-specific costing. Battery transport arrangements also need checking for
the chosen pack and journey. Do not price a traveling show as just a parts BOM.

## Next investment and evidence gates

1. **50 cm packaging and one/two-motor bench:** provisional USD 700–1,200.
   Test the AK45 around the relevant stance and motion loads, with temperature,
   current and timing measurements. Confirm voltage/regeneration behavior and
   the actual mounting envelope. Spend this before buying a full motor set.
2. **Choose the actuator budget:** retain the cheaper route only if geometry,
   mass and measured thermal duty support it. Otherwise evaluate the stronger
   50 cm option or revise the mechanical target; do not silently exceed ratings.
3. **One six-axis leg and supported pelvis:** approximately USD 1,500–2,500
   cumulative for the cheaper 50 cm route. Build the measured model, then test
   two-leg standing and stepping before adding cosmetic mass and gestures.
4. **Revisit 80 cm after repeatable locomotion and a use case:** an initial
   comparative motor bench is about USD 1,500–3,000, with a one-leg stage around
   USD 4,000–6,000 cumulative. These stages are portions of project spending,
   not automatic extra charges on top of the totals.

For a future larger model, CI should check mass/inertia ledgers, support/contact,
clearance and actuator envelopes, then seeded longer walking/turning trials
with head, arms and tail in different poses. Thermal behavior needs measured
parameters and physical endurance tests. **A passing miniature balance test
cannot qualify either larger robot.** This study adds repeatable sizing
arithmetic, not a substitute for those future tests.

## Reproduce and inspect

From the repository root:

```sh
python3 docs/hardware/studies/trex_50cm/calculate.py
python3 docs/hardware/studies/trex_50cm/calculate.py \
  --input docs/hardware/studies/trex_80cm/inputs.json \
  --output docs/hardware/studies/trex_80cm/results.json
python3 docs/hardware/studies/compare_costs.py
```

The [cost comparison](studies/cost_comparison.json) records the mini assumptions
and derives larger-tier totals from the sourced
[50 cm inputs](studies/trex_50cm/inputs.json) and
[80 cm inputs](studies/trex_80cm/inputs.json). Input, script and baseline report
hashes preserve the calculation basis. Rerunning arithmetic does not validate
walking, fabrication fit, motor life or commercial viability.
