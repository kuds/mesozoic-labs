# Robot walking validation across species

**Status: proposed engineering workflow, 2026-10-25.** The unsupported stance
regressions exist; the general feasibility solver and locomotion acceptance
panel below do not. This extends the [gait-quality plan](../GAIT_QUALITY_PLAN_2026_09.md)
and [sim-to-real plan](SIM_TO_REAL_PLAN.md), without adopting an undecided gate,
changing a registered plant, or certifying either standalone T-Rex example.
The accompanying [design review](../reviews/ROBOT_DESIGN_REVIEW_2026_10.md)
records the checks actually run.

## Recommendation

Build a **contact- and actuator-constrained feasibility screen**, followed by a
separate controller replay gate. Do not score walkability from a head/body/leg
ratio, a center-of-mass dot inside two feet, or forward speed alone. A robot can
stand comfortably on two feet yet lack the reach, sideways weight transfer,
ankle authority or swing clearance needed for one step.

The next useful milestone for both T-Rex sizes is a repeatable, slow sequence:
two feet loaded → transfer load → lift one foot → place it → transfer back.
Test both sides. For a quadruped the same machinery uses its declared contact
schedule; it must not inherit a biped's left/right alternation requirement.
Keep head, arms and tail quiet first, then repeat with their intended motions.

## 1. Inputs: qualify the model before the controller

Use actual joint locations, axes, travel, transmissions, motor packages, foot
surfaces, mass centers and inertia tensors. Separate a decorative mesh from
the geometry allowed to carry a contact force. Record tolerances, cable travel
and bearings; a convex visual shell is not fabrication clearance evidence.

Each robot configuration needs explicit metadata for its root, feet and support
geoms, permitted contacts, limb chains, actuator mapping, controller rate and
intended gait class. Reuse registered models' `FOOT_SUPPORT` and `ModelId`.
Keep the mini/medium examples outside the registered manifest until their
interfaces and hardware assumptions are qualified. The proposed variant key
in [ROBOT_VARIANTS_AND_BUILD_PLAN.md](ROBOT_VARIANTS_AND_BUILD_PLAN.md) is not
an implemented runtime feature.

Masses, motor limits and contact properties need evidence levels: **measured**,
**vendor specification**, or **planning assumption**. Unknown motor curves
produce an *incomplete hardware assessment*, not an invented continuous rating.
For the current examples, torque caps are simulation assumptions; neither their
servo gains nor their assigned shell inertias have been identified on hardware.

## 2. Calculate whether the limbs can perform the motion

For an ideal two-link leg, the distance from hip to ankle must satisfy

$$
|L_1-L_2| \le r \le L_1+L_2.
$$

This is an inexpensive rejection screen, not a solution for a real digitigrade
leg. Joint limits, motor offsets, foot orientation and collisions reduce that
workspace. Solve inverse kinematics on the actual model for the intended
stance, lateral shift and swing path. Require a connected path with joint-limit
reserve, positive sole clearance, legal foot orientation and collision margin.
Check the path between sampled poses as well as the poses themselves.

Report achievable forward/lateral step length, clearance and body-height range;
also examine the translational foot Jacobian near the target poses. A near-zero
singular value identifies a direction with little motion authority. Do not mix
angular and linear Jacobian rows without an explicit normalization. Constraint
IK with joint and minimum-distance bounds is a standard implementation option;
Drake documents these capabilities [1]. MuJoCo kinematics/Jacobians plus a
constrained optimizer can provide the initial screen without a new simulator.

For the mini, specifically test lateral transfer with its five leg axes and
pitch-only ankle. For the medium, test the extra ankle-roll axis rather than
assuming that adding it solves transfer. Neither inference follows from height.

## 3. Calculate whether the motors and feet can support it

At a stationary pose solve for actuator torques and contact forces together:

$$
g(q) = B(q)\tau + J_c(q)^T f + \tau_{\rm passive}(q,0),
\qquad f_{n,i}\ge0,\quad \|f_{t,i}\|\le\mu_i f_{n,i}.
$$

Here `g(q)` is the generalized gravity load, `B` maps actuator effort through
the actual transmission, and `J_c` maps admissible contact wrenches. Include
floating-base equilibrium, the real support patch/CoP limits and any physical
passive mechanism. Apply the relevant continuous torque bounds; avoid treating
stall torque as a continuous holding allowance. At a fixed pose this can be a
linear or conic feasibility problem, depending on the friction approximation.

A positive geometric support-polygon margin alone ignores actuator limits.
The actuation-aware feasible-region literature addresses precisely this gap
[2]. Report both geometric margin and minimum torque/friction reserve. Solve
double support, each intended reduced-support phase and the full weight-transfer
path. Failure at sampled poses rejects those poses; failure to find a solution
with a local optimizer does not prove that no possible gait exists.

An illustrative moment screen is `tau ≈ m g d`. With the medium's assigned
7.957 kg mass, a single support axis carrying the full weight with a 50 mm
horizontal lever arm needs about **3.90 Nm** before acceleration. That is near
its 4 Nm model cap. The mini at 0.800 kg and 20 mm gives **0.157 Nm**, above its
0.10 Nm *stance sensitivity* cap. These are simplified lever examples, not
predictions of every joint's torque; use the full equilibrium above to allocate
loads. They explain why a two-foot standing pass cannot establish single support.

## 4. Test a dynamic gait, including impacts and recovery

Use the full equations along a proposed trajectory:

$$
M(q)\dot v+h(q,v)=B(q)\tau+J_c(q)^T f+\tau_{\rm passive}(q,v),
\qquad \dot q=N(q)v.
$$

Contact phases must agree with foot motion and unilateral/friction constraints.
Include joint speeds, time-dependent torque-speed limits, contact impact bounds,
collision clearance and motor duty. Optimize a periodic step or finite sequence,
then replay it with feedback and perturbations. A feasible nominal trajectory
shows one model-based possibility, not a robust physical controller. Whole-body
and contact-sequence trajectory optimization provide established routes [3,4].

For quick flat-floor reasoning, the constant-height inverted-pendulum model uses

$$
p_{\rm ZMP}=c_{xy}-\frac{h}{g}\ddot c_{xy},\qquad
\xi=c_{xy}+\dot c_{xy}\sqrt{h/g}.
$$

`xi` is the capture point in that approximation. Compare it with reachable
future support at the planned landing time. Requiring it to remain inside the
current foot at every instant would incorrectly reject ordinary stepping.
These formulas assume nearly constant CoM height and negligible change in
whole-body angular momentum; a moving head/tail, crouch or impact can violate
those assumptions [5,6]. Use full centroidal momentum and contact wrench
feasibility when they do, rather than adding an arbitrary margin to a dot plot.

For a repeatable gait, an optional later analysis is the step-to-step return map
`x_next = P(x)`. At a periodic solution, eigenvalues of the linearized map inside
the unit circle indicate local orbital stability, after removing translation
and other symmetry directions [7]. This is stronger than watching one nominal
cycle, but remains local and model-dependent; it does not replace disturbance
trials or hardware testing.

## 5. Measure walking from physics, not animation or reward

Extend `environments/shared/gait/`, which already has morphology, substep contact
recording, stance metrics, force clipping, CoP and balance telemetry. Registered
models should share that measurement layer and the existing held-out evaluation
machinery. The small `shared.robot_validation` helper protects standalone stance
experiments; it is not a second production gait recorder or certification gate.

| Measurement | What it prevents or answers |
|---|---|
| Full-horizon survival, height/attitude envelope, unexpected contacts | Falling, leaning on the tail, or accepting a truncated favorable segment |
| Per-foot ground load at physics substeps, touchdown liftoff timing | Phantom touch support, stacked feet, contact chatter |
| Loaded support sequence, swing clearance, cadence, duty and flight fractions | Distinguishing standing, dragging, hopping, walking and running |
| Body displacement plus support-foot slip relative to terrain | Calling sliding or treadmill motion forward walking |
| Velocity/heading tracking, starts, stops and turns | A single straight trajectory that cannot be controlled |
| Joint travel, speed, transmitted torque, RMS duty and actual force clipping | A gait that depends on impossible actuator behavior |
| Foot impacts, self contact, model warnings and timestep sensitivity | Excessive strikes, interpenetration or a numerical artifact |
| Electrical current, pack sag and motor temperature on hardware | Confusing joint mechanical power with battery/thermal feasibility |

Touch sensors alone are insufficient; use named foot-to-terrain contact pairs
and resolved forces. MuJoCo's normal is the first contact-frame axis and its
contacts are compliant, so frame conversion and transient interpretation matter
[8]. Settled load should reconcile with body weight for stance, not at every
instant of dynamic walking.

Normalize cross-size comparisons using an explicitly reported mechanical length
`L` (for example nominal hip-to-ground height, never decorative overall height):
step length/L, clearance/L, speed/sqrt(g L), cadence*sqrt(L/g), and torque/(m g L).
These proposed comparison coordinates are not pass thresholds. Tail-heavy and
head-heavy species can have different inertias even when all these ratios match.

For energy, report positive mechanical work per `m g distance` and electrical
energy per `m g distance` separately. Guard zero-distance cases. A holding motor
can dissipate substantial electrical heat at zero joint mechanical power;
`sum(tau * joint_speed)` is not a battery-runtime estimate.

## 6. Proposed acceptance and CI layers

These are development milestones, not manufacturer specifications or already
adopted production criteria. Calibrate thresholds on the intended gait, real
measurement uncertainty and negative controls before freezing a gate version.

| Layer | Proposed job | Evidence required |
|---|---|---|
| Every relevant PR | Deterministic assembly, source regeneration, balance, weak-motor and hidden-support controls | Current mini: 30 s at 0.10 Nm; medium: 10 s in CI, longer recorded replay; bounds in executable tests |
| Next mechanics milestone | Constrained IK and static load transfer, both sides/contact schedules | Feasible continuous transfer, lift/place clearance and effort margin; no gait certificate |
| Controller development | Fixed nominal 30–60 s replay plus starts/stops | Actual walking metrics, no resets/support; all failures retained |
| Before promotion | Frozen held-out seed/scenario panel, longer endurance and disturbance sweeps | Per-episode pass criteria and confidence bound, independent of training selection |
| Before physical release | Bench actuator identification, instrumented tethered work, then unsupported trials | Measured torque-speed/thermal behavior, supply/latency, falls and repeatability |

Suggested *initial experiment ranges*: mass ±10%, plausible battery/CoM offsets,
friction 0.4–0.8, command delay 0–40 ms, sensor noise, reduced battery voltage,
small slopes and normalized horizontal impulses `J/(m sqrt(g L))`. These are
screening ranges to replace with measured bounds, not claims that a model passed
them. Vary axes independently before combined cases. Use held-out values for
acceptance; never select a controller on its acceptance panel.

Negative controls must include a welded robot, gravity assistance, deliberately
weak motors, unreachable steps, a shifted heavy head, a frozen standing policy,
sliding feet, a two-foot hopper and short-lived motion. A walker gate should
reject the inappropriate controls even if their reward or speed is high. A
quadruped trot/run gate may correctly admit flight; a slow biped walk gate need
not. Declare that choice in the task, not in species-specific measurement code.

Bind every report to model/assets, mass ledger, actuator specification, controller,
measurement version, thresholds, seeds and runtime versions. Include the exact
command and failed trials. Preserve historical reports when a geometry or gain
change alters the plant; do not relabel them as current evidence. For registered
models use the existing plant/interface/task revision and certificate lineage
rules. These standalone engineering reports are not production certificates.

## 7. Implementation order

1. Keep the new unsupported stance guards and long mini hold in CI. This already
   caught creep hidden by the earlier six-second test.
2. Add a reusable geometry/actuation feasibility module with explicit foot and
   motor metadata; pilot it on mini and medium, then a registered quadruped.
   Output feasible workspace, limiting joints and numerical residuals, including
   *inconclusive* when the solver or hardware inputs are insufficient.
3. Implement the missing locomotion metrics on the existing substep trace, and
   calibrate the proposed gait gate with the negative controls above. Follow the
   existing gait plan's decision process before changing production adoption.
4. Develop one slow load-transfer/step controller before optimizing speed or
   adding expressive tail motion. Freeze separate quiet and expressive panels.
5. Identify real motor, contact and timing parameters, rerun the same panels,
   then assess hardware. More compute does not resolve an infeasible limb/load
   combination; change geometry, mass distribution or actuators when the screen
   identifies the limiting constraint.

## Primary references

1. [Drake inverse kinematics API](https://drake.mit.edu/pydrake/pydrake.multibody.inverse_kinematics.html): joint, pose and distance constraints.
2. [Orsolino et al., Feasible Region: an Actuation-Aware Extension of the Support Region (2020)](https://arxiv.org/abs/1903.07999).
3. [Winkler et al., Fast Trajectory Optimization for Legged Robots using Vertex-based ZMP Constraints (2017)](https://arxiv.org/abs/1705.10313).
4. [Dafarra et al., Dynamic Complementarity Conditions and Whole-Body Trajectory Optimization for Humanoid Robot Locomotion (2022)](https://arxiv.org/abs/2207.03198).
5. [Tedrake, Highly-articulated Legged Robots](https://underactuated.mit.edu/humanoids.html): contact-constrained dynamics and reduced balance models.
6. [Pratt et al., Capture Point: A Step toward Humanoid Push Recovery (2006)](https://www.ambarish.com/paper/Pratt_Goswami_Humanoids2006.pdf): capture regions and angular momentum.
7. [Tedrake, Simple Models of Walking and Running](https://underactuated.mit.edu/simple_legs.html): hybrid gait dynamics and return maps.
8. [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html): contact frames, forces and soft constraints. Executable checks here use the repository-pinned MuJoCo 3.10.0.
