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

**Scope.** This is the **static** check. §7.2 also asks for the cross-check *during a policy
rollout*, against kinematic flight phases; that still needs the checkpoint and is not done
here. What follows constrains the sensor hypothesis without settling the behavioural one.

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
because statically they see all of it. §1.5's claim should move from "pending sensor
verification" to "sensor hypothesis excluded statically" — the outstanding work is the rollout
comparison against kinematic flight phases, not sensor fidelity.

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
