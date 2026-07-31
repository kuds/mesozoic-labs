# Foot touch sensors versus `mj_contactForce`

Discharges the sensor-verification prerequisite (`STAGE1_SPLIT_PLAN.md` §7.2), which gates the
hopping claim in §1.5, the ground-reaction-force diagnostic in §6.2, and the
`foot_load_balance` repair in §7.1.

The question §7.2 asks is whether `unsupported_duty = 0.209` on a plant that holds 0.93 m
pelvis height and never falls is real airtime or a sensor under-reporting. A MuJoCo touch
sensor sums only contacts on geoms belonging to its site's **own body**, which has already
produced two silent under-reads here — the T-Rex repair (`aa87445`) and the raptor repair
(`aa3395c`) — so the concern is well founded. It turns out to be well founded for two species
that nobody had checked, and unfounded for the one it was raised about.

**Method.** Settle each species 200 zero-action steps from its stage-1 config at
`reset_noise_scale = 0`, then compare the summed foot touch sensors against the vertical
component of `mj_contactForce` over every floor contact, and both against body weight.
Reproduce with `python environments/shared/scripts/foot_sensor_report.py`.

**Scope.** §1–§5 are the **static** check, on a settled plant. §6 is the dynamic half §7.2 also
asks for — the cross-check against kinematic flight phases — done by sweeping driver policies
rather than by replaying the trained checkpoint. Read §6 for what that substitution does and
does not buy.

## Results  `[measured]`

| species | touch sensors | `mj_contactForce` | animal weight | sensor / contact | contact / weight |
|---|---|---|---|---|---|
| trex | 842.2 N | 842.2 N | 840.9 N | **1.000** | 1.002 |
| dibothrosuchus | 84.9 N | 84.9 N | 84.9 N | **1.000** | 1.000 |
| velociraptor | 73.3 N | 132.4 N | 132.4 N | **0.553** | 1.000 |
| brachiosaurus | **0.0 N** | 1699.2 N | 1719.7 N | **0.000** | 0.988 |

`contact / weight` is 1.00 on all four species, so the contact accounting itself is sound and
the plants really are standing in equilibrium. Every discrepancy below is a *sensor* defect,
not a physics one.

## 1. The T-Rex sensors are accurate — which strengthens the hopping reading

T-Rex touch sensors agree with `mj_contactForce` to three decimals, and total floor reaction
equals body weight to 0.2%. The post-`aa87445` pad-plus-digits summation is correct.

This does not prove the plant hops, but it removes the alternative §7.2 raised: the
`unsupported_duty = 0.209` reading cannot be explained by sensors failing to see contact,
because statically they see all of it. §6 then closes the dynamic half, and together they
exhaust the sensor-artifact explanation.

## 2. §6.2 reads the GRF invariant backwards  `[correction]`

§6.2 proposes alarming when `mean(total_contact_force) / (m·g)` leaves `[0.95, 1.05]`, and says
"this would already be firing: the statue's static total is 841 N while the policy's logged
mean is 1460 N" — implying 841 N is the anomaly.

It is not. **841 N is exactly right.** The T-Rex animal masses 85.72 kg, so its weight is
840.9 N, and the measured static total of 842.2 N gives a ratio of **1.002**.

The 1483 N figure that makes 841 N look low is `mj_getTotalmass`, which includes the **65.45 kg
prey body** — 43% of the 151.17 kg model total. Any GRF diagnostic must divide by the mass of
the *animal's* kinematic subtree, or it reports a false 0.57 on a plant standing in perfect
equilibrium. `foot_sensor_report.py` does this by summing only the subtrees containing
actuated joints.

So the diagnostic would fire — but on the **policy's** 1460 N, at 1.74× body weight, not on the
statue. For strictly periodic motion the time-averaged GRF must equal body weight, so a
sustained 1.74× indicates either a non-periodic average or one conditioned on contact steps.
Both readings point at intermittent ground contact rather than a steady stance, consistent with
§1.5 — but 1460 N is `[artifact-derived]`, was not reproduced here, and should be re-measured
with the corrected denominator before being relied on.

## 3. Velociraptor under-reports foot load by 45%  `[measured, new]`

Not previously filed. Per foot, the floor reaction divides as:

| body | force | seen by sensor |
|---|---|---|
| `toe_d3` (primary digit) | 36.64 N | yes |
| `metatarsus` | 17.54 N | **no** |
| `toe_d4` (lateral digit) | 12.03 N | **no** |
| total | 66.21 N | 36.64 N = **55.3%** |

The `r_foot` / `l_foot` touch sites sit on `r_toe_d3` / `l_toe_d3`, and a touch sensor counts
only contacts on its own body's geoms. `aa3395c` fixed the *site size* — the original r=0.02
sphere at the capsule midpoint missed both end contacts and read 0 during stance — but not the
*body scope*, so the secondary digit and the metatarsus remain invisible. Same defect class as
`aa87445`, one repair short.

This is the species whose **complete** stage-1 gate a zero-action policy clears
(`STAGE1_SPLIT_PLAN.md` §1.4), so its instrumentation deserves more confidence than it
currently earns.

## 4. OQ-6 confirmed: brachiosaurus is blind to its own feet  `[measured]`

All four brachiosaurus foot touch sensors read **exactly 0.0 N** while `mj_contactForce`
reports 1699.2 N over four floor contacts — 98.8% of the animal's 1719.7 N weight. The contacts
are real and correctly carrying the animal; the sensors do not see them at all.

## 5. Both under-reads are latent in reward and active in observation

PR #471 placed the brachiosaurus defect on the reward path. That is not where it bites today:

* `configs/brachiosaurus/stage1_balance.toml` sets none of `foot_contact_gate`,
  `foot_contact_weight`, `bilateral_support_weight` or `foot_load_balance_weight`.
* `configs/velociraptor/stage1_balance.toml` likewise sets none, and `gait_symmetry_weight = 0.0`
  disables the only other consumer.

So no stage-1 reward term reads these sensors on either species. The **observation** does:

* `BrachioEnv._get_obs` feeds all four sensors into the policy input at indices **75–78** of 83.
  Verified directly — after 200 settled zero-action steps, exactly four observation channels
  are zero, at exactly those indices.
* `RaptorEnv._get_obs` feeds both sensors in, so the raptor policy sees 55% of true per-foot
  load.

Brachiosaurus therefore trains with four permanently dead input dimensions and no foot-contact
information whatsoever. That is worth weighing against §9's brachiosaurus risk row, which notes
zero action never reaches the horizon (0 of 40, `n_standing = 0`) and asks whether the plant is
corrupt: a policy that cannot perceive ground contact is a plausible partial explanation that
does not require plant corruption — though it does not establish one either.

Both defects also matter directly for the split. The §7.1 `foot_load_balance` repair and the
§6.3 support-state transition matrix both key off these sensors, and a stance or recovery gate
that reads contact state cannot be trusted on a species whose sensors report 0% or 55% of the
truth.

**Not fixed here.** Both repairs are MJCF changes of the same shape as `aa87445` — give each
contacting geom its own touch site and sensor, append them so existing sensor indices keep
their positions, and sum per foot on both the Gymnasium and MJX paths. Each moves that species'
physics and policy fingerprints and so belongs in its own change with its own revision bump.
Both invalidate that species' existing checkpoints, since the observation values change.

## 6. The dynamic half: do the duty metrics track kinematic ground truth?  `[measured]`

§1–§5 verify the sensors at one operating point — quiet bilateral stance, steady load. The
`unsupported_duty = 0.209` reading comes from a different regime entirely: touchdown transients,
rapid load transfer, and whatever flight phases exist. A sensor exact at 842 N steady can still
mis-time the *transitions* that decide how a step is classified.

**Method.** Sweep driver policies that produce airborne fractions from 3% to 67%, and compare
three signals per timestep — the summed touch sensors, `mj_contactForce` over foot-geom floor
contacts, and `mj_geomDistance` from every foot geom to the floor. The third is computed from
geometry alone and never touches the sensor path, so it is the arbiter: if the steps labelled
`unsupported` are the steps where the feet are genuinely clear of the ground, the metric
measures what it claims. Reproduce with
`python environments/shared/scripts/stance_duty_validation.py`.

| regime | airborne (kinematic) | `unsupported_duty` | misclassified |
|---|---|---|---|
| drop test (+0.25 m spawn) | 8.17% | 8.17% | **0.000%** |
| settled stance | 3.37% | 3.51% | 0.137% |
| low-amplitude jitter | 16.07% | 16.21% | 0.216% |
| forced hop | 67.25% | 66.74% | 0.515% |
| random thrash | 64.51% | 63.45% | 2.613% |

**The metric is sound.** Across a twenty-fold range of airtime it tracks kinematic truth to
within 0.52% of steps, degrading to 2.6% only under uniform-random actuation, which is not a
regime any policy occupies. The two error directions also very nearly cancel — in the jitter
band, 9 false-airborne against 2 false-supported out of 5095 steps, a net overstatement of
0.14%.

The **low-amplitude jitter** row is the one that matters: at 16.07% true airtime against 16.21%
reported it straddles the 0.209 figure under dispute, and it is the regime whose contact
transitions most resemble a balancing policy's. Mean sensor-versus-force error there is 4.4 N
against a 13,079 N peak.

**What this settles.** `unsupported_duty = 0.209` can be taken at face value: the instrument
producing it is accurate in exactly that band, so the trained policy really is off the ground
about 21% of the time. Combined with §1 — the sensors see all of the contact statically — and
with the classifier's 0.1 N threshold against ~421 N per foot in quiet stance, which is far too
low for partial unloading to manufacture a false `unsupported`, the sensor-artifact explanation
for §1.5 is exhausted. **§1.5's hopping reading should move from `[inferred]` to `[measured
instrument, inferred behaviour]`.**

**What this does not settle.** This validates the *instrument* across swept regimes; it does not
replay the trained policy. The 0.209 figure comes from that run's logged diagnostics, and I have
not independently reproduced the policy's behaviour — the checkpoint is a 4 MB artifact on Drive
and could not be pulled into this environment. The remaining claim in §1.5 is causal ("the
reward *caused* the hop"), which no amount of sensor verification can establish; that needs the
counterfactual run with the §7.1 repair in place.

**A caution for anyone writing similar diagnostics.** Two bugs surfaced while building this, both
worth avoiding:

* Comparing the foot sensors against *every* floor contact counts tail and torso strikes as
  sensor disagreement. Restrict the force reference to foot-geom contacts.
* Selecting foot geoms by name with a `foot|toe|metatarsus|pad` filter **misses
  `r_plantar_geom` and `l_plantar_geom`**, which carry the majority of the load — 190–200 kN·steps
  against 4.6–6.0 kN·steps for the metatarsus across a 15-episode sweep. Enumerate from actual
  contacts instead. This is the same mistake class as the sensor-scope defects above: a
  plausible-looking name filter silently omitting the primary load path.
