# Which signals carry the CoP the gate reads? (scripted plant probe, no checkpoint, no training)

Script: `probe_cop_signals.py`; output: `probe_cop_signals.out`, `probe_cop_signals.json` (same folder). Run on the read-only checkout (`environments.__file__` printed at the top of the output), MuJoCo 3.10.0, stance config through `config.build_env(ModelId("tyrannosaurus_rex", "anatomical"), "stance")`.

Design. 8 held postures x 4 reset seeds (3042-3045). An action offset (ankles, one ankle, one knee, both hip pitches, or none) is ramped in over steps 200-300 and held to step 600, with N(0, 0.03) jitter on every actuator at every step. Floor truth comes from `SubstepContactRecorder` (`EpisodeTrace.foot_cop`, m ahead of the ankle axis; `sole_cop[..., 0]`, pad half-lengths). Rows are steps 300 onward on which the limb carries at least 42 N: 18,262 foot-steps. Every `sym_ankle_-0.05` episode nosedived between steps 451 and 541; the rest ran to 600. Fits are ordinary least squares with an intercept; "LOPO" = leave one posture out (train on 7 postures, test on the 8th).

| Predictor | Is it available to the policy today? | Whole-foot CoP (sd 0.0454 m): R^2 / rms | LOPO rms median / max (m) | Pad CoP (sd 0.621): R^2 / rms | LOPO rms median / max |
|---|---|---|---|---|---|
| A. ankle joint torque / limb floor force (`actuator_force + qfrc_passive` at the ankle dof) | No (needs a new observation) | 0.870 / 0.0164 | 0.0159 / 0.0213 | 0.860 / 0.232 | 0.231 / 0.273 |
| B. ankle servo error x (kp + spring) / F (needs the applied `ctrl`) | No (needs a new observation) | 0.842 / 0.0180 | 0.0177 / 0.0233 | 0.833 / 0.254 | 0.248 / 0.321 |
| C. the foot's 4 existing touch sensors as shares (pad, d2, d3, d4; sensordata 10/11 + 24-29) | No: they are summed into one channel per foot | **0.908 / 0.0138** | **0.0129 / 0.0189** | 0.867 / 0.227 | 0.221 / 0.325 |
| D. observed today: that foot's 3 toe angles + 4 leg joints + projected gravity, quadratic (65 terms) | Yes | 0.729 / 0.0237 | 0.0278 / 10.6 (extrapolation blows up) | 0.732 / 0.322 | 0.381 / 123 |

Per-posture means (`probe_cop_signals.json`): the statue's whole-foot CoP sits 0.144 m ahead of the ankle (pad CoP 0.15-0.16); held `sym_ankle_-0.05` sits at 0.210 m (pad CoP 0.94-0.95, past the 0.80 bar); a single `r_ankle_-0.06` offset drives the OTHER foot to pad CoP 0.81-0.82 and its own to 0.33-0.34. That is the closed-chain coupling the investigation describes (`TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md:196-203`).

Reading:
- The CoP is not observable from the 63 dims in a way that generalises across postures. D fits in sample, but its held-out error explodes, consistent with `run_a/observability.txt:14-24`, where a linear body model fails out of posture (rms up to 2.5).
- Two cheap physical signals carry it. A force sense at the ankle (A or B, the robot's joint torque / the animal's tendon organ) gives about 1.6 cm rms on a 14 cm pad. The split of the foot's load between pad and digits (C, sensors that already exist) gives about 1.3 cm, with one linear map that holds across postures.
- Caveats: scripted postures with small jitter, not trained-policy rollouts; one plant (r9); fits are linear in-sample maps. A trained policy may need more or less than this. This shows that the information is available in these signals. It does not show that observing them fixes the stance.
