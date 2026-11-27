# Triceratops robot port to the main pipeline (design record, 2026-11-27)

Status: design, to be built in this PR. Base: `main` at 2c1039f4; every `file:line` is at 2c1039f4 unless it names
the build. Main has no `hardware/` folder (removed in c42b6a93), so build files are cited by commit, never by a branch
path: build commit 8198001c (the build tip) or f6deec9f (the commit that last wrote the source model). Short build
paths (`rl/trike_env.py`, `rl/train.py`, `rl/README.md`) are under `hardware/triceratops_build/` at 8198001c.

Two kinds of statement appear here. **Design** is what the implementers build. **Measured on the prototype** is
simulation evidence from scratch prototypes of each part, run at 2c1039f4 and since deleted; none of it is on any
branch. A pass is claimed only where a tool's output showed it.

## 1. Summary

- The 60 cm Triceratops robot becomes main-pipeline model `triceratops_horridus/robot`: the eighth model, seventh
  species; 12 leg servos, 63 observations, 12 actions, 1 ms physics, 50 Hz control (D32, D36).
- It trains on main's stance stage (gate: reward 900 over at least 950 of 1000 steps) then locomotion (reward 2000,
  550 steps, 0.15 m/s on a 4 m course, 0.3 m arrival radius). The gates are provisional (D40).
- Three shared features come first, opt-in and off for every existing model: a servo delay (0-20 ms drawn per
  training episode, 10 ms elsewhere, D33); physics randomisation (friction 0.6-1.0, per-body mass, servo kp x0.9-1.1,
  force limit x0.85-1.0, and whole-robot mass x0.9-1.1 plus a 0-1 kg payload in training only, D38); sensor noise.
- One rule decides who sees what: the trainer marks its rollout envs; every other env is an evaluation env (D37).
- Arriving at the target pays the standing reward of the remaining steps (D39).
- Nothing moves for any existing model: no digest, golden, plant manifest entry, task hash or checkpoint. No T. rex
  file changes. Five commits, each green on its own.

## 2. Owner decisions

Rows of the build's decision log (build commit 8198001c, `hardware/triceratops_build/DECISION_LOG.md`), 27 Nov 2026.

| Row | Decision (as recorded) | How this design meets it |
|---|---|---|
| D32 | The 60 cm robot becomes a main-pipeline model (Triceratops horridus, robot embodiment) on main's own stance and walking stages, with the servo delay, and mass, friction, servo-gain and force-limit randomisation and sensor noise as opt-in shared features, off by default for every existing model; steering commands and heat costs stay in the build stack | §8 adds the model on stance and locomotion only; §5-§7 add the features with `None`/`False` defaults; no steering mode or heat term of the build is ported (§11). The robot's 11 behaviour recipes are main's own command tasks, shipped because tests and CI require them and inert unless run (§8.7, owner point 6) |
| D33 | Delay opt-in for all models, off by default; on for the 60 cm robot with 0-20 ms drawn per episode in training and a fixed 10 ms in evaluation | §5: `actuation_delay_range_s = [0.0, 0.020]`, `actuation_delay_eval_s = 0.010` in every robot stage and recipe; drawn in trainer rollout envs only (§4.3) |
| D36 | One draft PR into main carrying only the pipeline port, on the new branch `claude/trike60-main-pipeline`: the model, its configs, the shared delay, randomisation and noise, and docs; the build tree stays on its branch, linked from main's hardware notes by commit | §3: five commits on that branch; no build file is copied or imported; the asset is derived by a script that names its source by commit (§8.1) |
| D37 | Evaluate as the build: per-body randomisation and sensor noise on in evaluation, the delay fixed at 10 ms, whole-robot mass and payload in training only; one shared rule (the trainer marks its rollout envs; every other env is an evaluation env) | §4.3: the `training_rollouts` flag and its table; per-body draws in every env built from the stage config; `observation_noise_eval = true` for the robot |
| D38 | Whole-robot mass factor (x0.9-1.1) and payload (0-1 kg): two more opt-in shared keys, off by default; on for the 60 cm robot in training rollouts; evaluation leaves them out | §6: `randomize_total_mass_scale` and `randomize_payload_kg` in the robot's stages and recipes, applied only while `training_rollouts` is true |
| D39 | Arrival bonus: on arrival the robot is paid the standing reward its remaining steps would have earned, so arriving is never worse than stalling; a change to the robot model only | §8.5: `reward_arrival = (max_episode_steps - steps taken) x 1.8` in locomotion, its own info key. The 1.8 counts alive, posture, height and heading, the four terms on at every step (§8.5). The guarantee covers the four always-on terms; the forward term is outside it (§8.5, §8.10) |
| D40 | Starting gates: stance reward 900 over at least 950 of 1000 steps; walking reward 2000, 550 steps, 0.15 m/s; course 4 m, 0.3 m arrival radius; provisional, revised with the owner after the first main-pipeline run | §8.6 uses them; the stage TOML comments, the species README and this record say "provisional (D40)"; a robot test pins them |

## 3. Landing order

One commit each, in this order; each passes every gate of §9 on its own. Later commits are review fixes only.

| Commit | Contents and files (`environments/shared/` unless a path is given) | What moves |
|---|---|---|
| 1 | This record and its row in `docs/README.md` ("Plans & designs") | nothing |
| 2 | Hook, stream helper, role flag (§4) and actuation delay (§5): new `episode_start.py`, `actuation_delay.py`; `base_env.py`, `train_base.py`, `train_behaviors.py`, `task_fingerprint.py`, `behavior_checkpoint.py`, `symmetry.py`, docstrings of `harnesses/digest_snapshot.py:493` and `scripts/warm_start_noise_scan.py`; `docs/PLANT_CONTRACT.md`, `docs/ROADMAP.md`, `CONTRIBUTING.md`, `CHANGELOG.md`; tests (the SB3 ones in `tests/test_train_base.py` and `tests/test_behavior_checkpoint.py`, §5) | nothing |
| 3 | Domain randomisation (§6): new `episode_physics.py`; `base_env.py`, `behavior_env.py`, `task_fingerprint.py`, `behavior_checkpoint.py`; `configs/species_manifest.toml` and the catalog; docs, with a status note in `docs/hardware/SIM_TO_REAL_PLAN.md`; tests. `plant_contract/validation.py` is not touched | the `domain_randomization` status in `website/src/data/species.generated.json` only |
| 4 | Observation noise (§7): new `observation_noise.py`; `base_env.py`, the `observation_segments()` refactor of `plant_contract/policy_layer.py`, `policy_probe.py`, `task_fingerprint.py`, `behavior_checkpoint.py`; docs, adding noise to the `SIM_TO_REAL_PLAN.md` note; tests | nothing |
| 5 | The model, its configs, tests, docs and website (§8); every generated file regenerated on the merged tree | only the robot's own lines |

**Must not move, in any commit:** the existing entries of both plant manifests (`configs/plant_manifest.generated.json`,
`environments/shared/data/plant_manifest.generated.json`); every existing line of
`configs/digest_snapshot.generated.txt` (725 at 2c1039f4) and of the `--exact` capture (40,568 at 2c1039f4,
measured on the prototype); `tests/fixtures/phase_c_reset_golden.json` and its `rng_state`; the T. rex mirror map;
every existing stage's `task_sha256` and `stage_config.json` view; every existing checkpoint's plant identity. No file
under `environments/tyrannosaurus_rex/**` or `configs/tyrannosaurus_rex/**` changes.

**Why this order.** The model forwards the features' kwargs to `BaseDinoEnv.__init__`, which accepts none of them at
2c1039f4 (`base_env.py:217-237`). Measured on the prototype: the model's patch alone fails at construction with
`TypeError: ... unexpected keyword argument 'actuation_delay_range_s'` (M1). So the model lands last, and its
generated files come only from the tree that has all three features.

## 4. Shared pieces

### 4.1 The episode-start hook (commit 2)

New module `environments/shared/episode_start.py`, not fingerprinted:

```python
class EpisodeStartHook(gym.Env):
    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)  # gymnasium seeds np_random
        self._begin_episode(seed)
        return result

    def _begin_episode(self, seed: int | None) -> None:
        """Where each opt-in feature begins its episode. Nothing by default."""
```

`class BaseDinoEnv(gym.Env, ABC)` (`base_env.py:70`) becomes `class BaseDinoEnv(EpisodeStartHook, ABC)`. Its own
`super().reset(seed=seed)` (`base_env.py:1603`) then lands in the hook, before the keyframe is loaded (`:1612`).
`BaseDinoEnv.reset`'s source does not change; the policy layer hashes it as `home_reset` by qualname and tokens
(`policy_layer.py:392-393`, `digests.py:154-166`), and nothing hashes class bases, so no digest moves. No species
overrides `reset`; the behaviour mixin's `reset` (`behavior_env.py:424`) selects the pool model (`:460`) before
`super().reset` (`:476`), so the hook acts on the live pool model. `BaseDinoEnv._begin_episode(seed)` is the one place
each feature begins its episode, in this order, each behind one `is not None` check on a class attribute that
defaults to `None`: `self._episode_physics.begin_episode(self, seed)` (§6), `self._begin_actuation_delay_episode(seed)`
(§5), `self._begin_observation_noise_episode(seed)` (§7). With every feature off it calls nothing, reads no generator
and writes nothing.

### 4.2 One stream policy (commit 2)

Each feature has its own PCG64 stream; none ever draws from `np_random`, so the reset draws (joint, qvel, height,
target, push schedule, command: `base_env.py:1617-1674`) are bit-identical with any feature on or off.

```python
EPISODE_PHYSICS_STREAM_KEY = 0xD0A1        # §6
ACTUATION_DELAY_STREAM_KEY = 0xDE1A        # §5
OBSERVATION_NOISE_STREAM_KEY = 0x0B5E4015  # §7

def feature_stream(env, seed, key) -> np.random.Generator:
    if seed is not None:
        entropy = int(seed)
    else:
        entropy = int(getattr(env, "np_random_seed", -1))   # gymnasium >= 1.0
        if entropy < 0:                                     # np_random assigned directly
            bit_generator = env.np_random.bit_generator
            if not isinstance(bit_generator, np.random.PCG64):
                raise ValueError("... needs a seeded reset or a PCG64 np_random; got <type>")
            state = bit_generator.state["state"]            # read, never drawn
            entropy = (int(state["state"]) << 128) | int(state["inc"])
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy, spawn_key=(key,))))
```

A feature calls it when `seed is not None` or its stream is unset: rebuilt on every seeded reset and on the first,
continued across unseeded resets (SB3's auto-reset), as `np_random` is. A spawn key never coincides with gymnasium's
`SeedSequence(seed)`. The keys are named constants in `episode_start.py`, tested distinct. `pyproject.toml:26` allows
`gymnasium>=0.29.0`, which lacks `np_random_seed`; the fallback covers it (R6, R7).

### 4.3 The role flag (commit 2)

`BaseDinoEnv.training_rollouts = False` in `__init__`: every env is an evaluation env unless the trainer marks it.
`set_training_rollouts(enabled)` sets it; every feature reads it in `_begin_episode`, so a change takes effect at the
next reset, for all three alike. It is set only through `mark_training_rollouts(env, enabled)` in `episode_start.py`,
which looks up `set_training_rollouts` on `env.unwrapped` and does nothing for an env without it, such as the plain
`gym.Env` double of `test_behavior_checkpoint.py:71-93, 1243-1248` (delay P1); a vectorised env (`num_envs` or
`env_method`), whose `unwrapped` lacks the method, raises `TypeError` instead of passing silently. The only call sites
(an AST test pins them):

- `train_base.make_env(..., plant_identity=None, *, training: bool = False)` (`train_base.py:366-393`): in `_init`,
  after `validate_environment_plant`, before `Monitor` and the first `reset` (`:383-390`), since the features draw at
  reset: `if training: mark_training_rollouts(env, True)`. `create_vec_env(..., critic_phase_gamma=None, training:
  bool = False)` (`:396-460`, keyword-only) forwards it in `env_fns` (`:432`).
- `_train_stage_body` passes `training=True` only on `train_env = create_vec_env(...)` (`:1609-1620`). The `eval_env`
  call (`:1624-1634`) is unchanged, keeping the pin `eval_env.args[4] == 'eval_env_seed(seed)'`
  (`test_sb3_notebook_pins.py:2692`).
- `train_behaviors.main`: `mark_training_rollouts(env, True)` right after `create_behavior_env` (`:306`);
  `mark_training_rollouts(env, False)` right before `evaluate_behavior` (`:472`), which reuses the training env.

**Who sees what** (robot values in brackets):

| Env kind | Built at (2c1039f4) | Role | Delay | Per-body randomisation | Whole-robot mass, payload | Noise |
|---|---|---|---|---|---|---|
| Stage training rollouts | `train_base.py:1609-1620` via `create_vec_env(training=True)`, `make_env` | training | k per episode, uniform over whole physics steps in [lo, hi] [0-20] | on | on | on |
| Behaviour training | `train_behaviors.py:306`, then marked | training | drawn | on | on | on |
| Behaviour scoring | the same env, unmarked before `train_behaviors.py:472` | evaluation | `actuation_delay_eval_s` [10 ms] | on | off | if `observation_noise_eval` [on] |
| Checkpoint selection, curriculum gate | EvalCallback env, `train_base.py:1624-1634` | evaluation | eval value | on | off | if eval flag |
| CLI evaluation, stage videos | `evaluation.py:625-636`, `:317`, `:331` | evaluation | eval value | on | off | if eval flag |
| Stage reports | `reporting/stage_artifacts.py:638, 1662`, its dt probes `:1719, 1945` and its stance and trace reports through `env_class`/`env_kwargs` `:2179-2261`; `stance_report.py:251-252, 1260-1270, 1419, 2158-2186`; `episode_trace.py:476-481` | evaluation | eval value | on | off | if eval flag |
| `config.build_env` users | `config.py:431-459`: recovery gate and calibration, zero-action baseline, `widen_checkpoint`, report scripts, `scripts/warm_start_noise_scan.py:686` | evaluation | eval value | on | off | if eval flag |
| Other direct builders from a stage config or recipe | behaviour certification and replay panels (`behavior_certification.py:30-33`); `scripts/calibrate_recovery.py:128`; `scripts/stance_quality_baseline.py:118`; `harnesses/recovery_offdist_panel.py:95, 196` | evaluation | eval value | on | off | if eval flag |
| Digest harness reward captures | `harnesses/digest_snapshot.py:497-499` (stages), `:447-452` (behaviour recipes) | evaluation | eval value | on | off | if eval flag |
| Policy-probe bank | `train_base.py:1771-1773` with `bank_env_kwargs` (`policy_probe.py:163-178`) | evaluation | eval value | on | off | forced off |
| Constructor defaults | plant manifest `manifest.py:83`; harness policy lines `digest_snapshot.py:312`; mirror derivation `symmetry.py:1271`; plant probe on synthetic state (`policy_layer.py:175-210`); `gym.make`; `env_smoke` | none | off | off | off | off |

An eval value of `None` makes evaluation envs draw from the range. The bank forces noise off as it forces pushes off:
`bank_env_kwargs` sets `observation_noise_eval = False` only where the stage declares it (noise P4, option b).
`compsognathus_longipes/scripts/probe_ppo_updates.py:118` builds an unmarked training env; no model there opts in.
`warm_start_noise_scan` is a deliberate evaluation-env diagnostic (fixed 10 ms), as its docstring says (delay P11).
This fails safe: no evaluation builder can be forgotten, a forgotten trainer path trains at 10 ms, and checkpoint
selection and the gate judge one task distribution (each EvalCallback round draws new plants and noise, §6).

## 5. Actuation delay (commit 2)

**Keys** (kwargs appended after `command_config`, `base_env.py:237`): `actuation_delay_range_s: tuple[float, float] |
None = None`, seconds, the training distribution and the switch (`None` is off); `actuation_delay_eval_s: float |
None = None`, seconds, the fixed delay of every env that is not a trainer rollout env (`None`: draw from the range
there too; it need not lie in the range). TOML arrays become tuples (`config.py:395-400`); `--override` works the same
way. A species that does not list the keys raises `TypeError` if a TOML sets them. Robot: `[0.0, 0.020]` s, 21 values
of 0-20 whole 1 ms steps; `0.010` s, 10 steps. The source is D33; the build's steer1 presets carry the same 0-20 ms
(build commit 8198001c, `rl/README.md:461`) and its evaluations a fixed 10 (`rl/train.py:503`); the build's default
0-10 ms (`rl/trike_env.py:95`) is not the source (P9, M6).

**Module `environments/shared/actuation_delay.py`** (not fingerprinted): `ACTUATION_DELAY_ENV_KEYS`;
`ACTUATION_DELAY_IMPLEMENTATION = "ctrl_fifo/v1+per-physics-step+zero-order-hold+reset-ctrl-prime+episode-stream"`,
bumped on any change of meaning like `SCHEDULE_IMPLEMENTATION` (`task_fingerprint.py:77`);
`delay_steps(seconds, timestep, *, name)`, which refuses a bool, a non-number, a non-finite or negative value, and any
`|round(s / dt) x dt - s| > 1e-9` s, naming the timestep and the nearest whole value (0.010 s is 10 steps at 1 ms and
5 at 2 ms; 0.003 s at 2 ms is refused); `validate_actuation_delay(range_s, eval_s, timestep, *, episode_steps=None)`,
which refuses an eval value without a range, a string or non-pair range, `lo > hi` and, given the episode's
`max_episode_steps x frame_skip` physics steps (the env passes them), a range end or eval value of at least that many
steps, named in seconds and steps, since no target would reach the plant (milliseconds typed into a seconds key);
`primed_fifo(hold, steps)`, `steps` float64 copies with `maxlen = steps + 1`.

**State** (after the filter state, `base_env.py:258`): `_actuation_delay` (validated config or `None`),
`_actuation_delay_rng`, `_actuation_delay_k`, `_actuation_delay_fifo`; read-only property `actuation_delay_steps`, valid
from reset on (P3). Off path: two `is not None` checks per step (before the ctrl write and before the info keys) and one
per substep; nothing is copied. **Episode start:** rebuild the stream (§4.2) when seeded or unset; k is the eval value
in an evaluation env when it is not `None`, `lo` when `lo == hi`, else `int(rng.integers(lo, hi + 1))`, so a draw
happens only where k comes from the range; the FIFO is cleared.

**The FIFO.** `step()` keeps filter, scale, write (`base_env.py:1281-1282` stay). At the episode's first step
(`_step_count == 0`) the FIFO is primed from `data.ctrl` as reset left it, before that write. In the substep loop,
immediately before `mujoco.mj_step` (`:1310`): append this step's target, then `data.ctrl[:] = fifo[0]`. A target
set at control step n reaches `data.ctrl` exactly k physics steps later; substeps 0 to k-1 of an episode hold the
prime. k = 0 is bit-identical to off; k may exceed `frame_skip`. The whole nu-wide ctrl is delayed: on the robot, the
12 leg drives, as the build delays them.

**Priming.** The prime is `data.ctrl` as reset left it, which equals `key_ctrl` of `_reset_keyframe_id`
(`mj_resetDataKeyframe`, `base_env.py:1612`) because no shared code writes `data.ctrl` between reset and the first
step. Under every home-keyframe mapping that is `_scale_action(0)`, so zero action is delay-invariant. This is a
precondition: a robot test asserts `_scale_action(np.zeros(12))` equals `model.key_ctrl[env._reset_keyframe_id]` bit
for bit and that the mapping is in `_HOME_KEYFRAME_ACTION_MAPPINGS` (P10).

**Info** (after `info["distance_traveled"]`, `base_env.py:1373`), only while on: `actuation_delay_steps` (int) and
`actuation_delay_s = round(k x timestep, 12)`, since `9 x 0.001` is `0.009000000000000001` in float64 (P4). Tests
compare steps. The reset info stays `{"step": 0}` (`:1677`, in the hashed reset).

**Fingerprint.** After the command section (`task_fingerprint.py:277-278`): when
`payload["env"].get("actuation_delay_range_s") is not None`, add
`payload["actuation_delay"] = {"implementation": ACTUATION_DELAY_IMPLEMENTATION}`. `"actuation_delay"` joins
`_differing_sections` (`:307-309`) and the tuple at `behavior_checkpoint.py:420` (P6); `None == None` for every
existing model. A changed value is refused naming `env`; switching the delay on or off, or a tag bump, also names
`actuation_delay` (P7). Steps are not recorded: seconds and `physics_sha256` (which pins the timestep) fix them.
Existing species do not list the keys, and the fingerprint reads the species signature (`:168-219`).

**Mirroring.** `symmetry._mirror_env_attributes` (`symmetry.py:1178-1235`) copies `_actuation_delay_fifo` through the
ctrl map (`em.model.ctrl.apply`, as at `:1160`) and `_actuation_delay_k`, not `training_rollouts`, which is the env's
role, not episode state (P5). `mirror_env_state` refuses two envs whose delay configurations differ: a delay-off source
would leave a delayed destination with no k, and a delay-off destination would drop the in-flight targets.

**Interactions.** Rewards are still priced on the filtered command (`base_env.py:1359`); torque terms see the delayed
plant. T. rex's `_filter_action` override is unaffected. Pushes write only `xfrc_applied` (`:1264-1279`). Readers of
`data.ctrl` after a step see the applied ctrl. A behaviour pool swap happens in reset and the FIFO re-primes at the
next first step. The FIFO never touches `env.model`, so plant validation holds at any time. The docstring at
`digest_snapshot.py:493` becomes "as `train_base.make_env` builds evaluation envs (`training=False`)" (P8).

**Why a FIFO, not MuJoCo's actuator delay** (3.10.0, measured on the prototype): the same arithmetic (max |dqpos| 0.0
over 150 steps), but `delay=` needs an MJCF edit; its history is zeros after a keyframe reset; a per-episode draw
mutates `actuator_delay`, a physics-digest field (`physics_layer.py:175`); an over-long delay passes silently.

**Files:** §3. **Tests** (new `test_episode_start.py`, `test_actuation_delay.py` in `environments/shared/tests/`):

- Hook and streams: every manifest model's MRO has `EpisodeStartHook` right after `BaseDinoEnv`; `env.reset` resolves
  to `BaseDinoEnv.reset` or `SpeciesBehaviorMixin.reset`; the entropy rules of §4.2 (MT19937 raises, PCG64 is read,
  not drawn); distinct keys; `np_random` after reset equals a features-off env's.
- Role: `mark_training_rollouts` on a plain `gym.Env` is a no-op; an AST scan finds it only in `make_env` and
  `train_behaviors.main`, `training=True` only on `train_env`. Both new files import no SB3 and run in CI's `test`
  matrix. The two checks that need SB3 (`make_env` calls `_ensure_sb3`, `train_base.py:375-377`, which exits without
  it, `policy_loading.py:177-207`) go in files the test-sb3 step already runs (`python-ci.yml:487-524`), so
  `test_ci_tool_pins.py:364-376` holds: `create_vec_env(training=True)` marks each env and the default none, in
  `test_train_base.py` behind `pytest.importorskip("stable_baselines3")` as its other SB3 tests are; and
  `train_behaviors.main` marks its env True before training and False before `evaluate_behavior`, in
  `test_behavior_checkpoint.py` (module-level `importorskip`), on a `PilotEnv` with a recording
  `set_training_rollouts`.
- Off by default: `_actuation_delay is None` exactly when the stage omits the keys (so commit 5 needs no edit); then no
  delay info key, and marking the env leaves a seeded zero-action plus random-action rollout byte-identical. The
  models listing a key are exactly an opted-in test table (empty now, the robot in commit 5), each with both keys at
  `None` and the carve-out; on every other model a TOML setting one raises `TypeError`.
- Exact delay for k = 0, 1, 3, 10, 25 on a test-only Compsognathus-robot subclass (2 ms, `frame_skip` 10; commit 5
  adds the robot): per-substep ctrl is the prime for s < k, then `targets[(s - k) // frame_skip]`; bit-equal to an
  independent deque FIFO on a copied `MjData` over 40 steps; k = 0 bit-identical to off.
- Evaluation envs hold the eval value (20 seeds); training envs draw k in [lo, hi], at least 15 distinct values over 60
  seeds, readable right after reset, independent of the actions; `None` eval draws; a degenerate range never draws,
  and an evaluation env with an eval value (5 steps, or 0) holds it over the range's one value (3) while a training
  env holds 3, which pins the order of the episode-start rule. The FIFO holds the filtered, scaled target. A pushed
  250-step zero-action rollout is bit-identical with the delay.
- No plant digest moves; `validate_environment_plant` passes mid-episode. Nine invalid configs refused at
  construction, two of them as long as an episode.
- Fingerprint: no section on any existing stage; a changed eval value names `env`; switching off names `env` and
  `actuation_delay`; an existing species opting in with the `None` carve-out (monkeypatched registry) keeps every
  committed Compsognathus-robot digest. `test_symmetry.py`: a delayed Velociraptor subclass mirrored mid-episode gives
  max obs error 0.0 and keeps its role. `test_behavior_checkpoint.py`: a changed `actuation_delay` section is refused.

**Measured on the prototype.** Off path bit-identical on all 7 models over 30 rollouts (obs, reward, info, qpos,
qvel, ctrl, generator state); exact k-step delay at 1 ms / 20 and 2 ms / 10; `--exact` 40,568 lines, empty diff;
`digest_snapshot --check` current (725 lines); `plant_contract --check` and `symmetry --check` current. Dropping
in-flight targets in a mirror diverges (max obs error 366 on a T. rex test subclass, 128 on a Velociraptor one).
Overhead within run-to-run noise. The prototype drew k from `np_random` at the first step; commit 2 draws it at reset
from its own stream (§4.2), as designed.

**Built in commit 2, where it differs from the text above** (changed after the build and its review, not after any
training result):

- `BaseDinoEnv._configure_actuation_delay(range_s, eval_s)` validates and installs the delay; `__init__` calls it, and a
  test subclass of a species that does not forward the keys calls it again after `super().__init__()`.
- `set_training_rollouts` refuses a non-bool (`TypeError`); stepping a delayed env before its first reset raises
  `RuntimeError`, since k is chosen at reset; the FIFO is primed again if it is missing mid-episode.
- All three stream keys of §4.2 are defined now, with a `FEATURE_STREAM_KEYS` tuple; in commit 2 only the delay's is
  used (commit 3 uses the physics key).
- The training-draw test uses the range 0-40 ms on the 2 ms Compsognathus-robot subclass (21 values), since 0-20 ms is
  only 11 values at 2 ms.
- The Velociraptor mirror test uses a delay of 6-12 physics steps (`frame_skip` 5): with k below `frame_skip` the FIFO
  holds only the last target, so dropping it would not show. Dropping the FIFO there gives max obs errors of 35.9 and
  73.8 (seeds 3047 and 3059), not the prototype's 128, which came from another configuration.
- Test names avoid the retired species identifiers (`test_retired_species_identifiers.py`).

## 6. Domain randomisation (commit 3)

**Keys** (six kwargs, each `None` or `(low, high)`: finite, `low <= high`, `low > 0` for every key but the payload
(friction included), payload `low >= 0`):

| Key | Unit | What an episode writes | Robot |
|---|---|---|---|
| `randomize_friction` | absolute sliding coefficient | one draw to `geom_friction[:, 0]` of every static floor geom (`_static_floor_geoms`, `base_env.py:1490`) and of the geoms `_randomized_friction_geoms()` returns | [0.6, 1.0] |
| `randomize_body_mass_scale` | factor | per animal body (free-root subtree): mass and inertia by one factor | [0.9, 1.1] |
| `randomize_actuator_kp_scale` | factor | per position servo (FIXED gain, AFFINE bias, `biasprm[1] == -gainprm[0]`): `gainprm[0]` and `biasprm[1]` together; kv kept | [0.9, 1.1] |
| `randomize_actuator_force_limit_scale` | factor | per force-limited actuator: both ends of `forcerange` | [0.85, 1.0] |
| `randomize_total_mass_scale` | factor | one factor on every animal body's mass and inertia, after the per-body factors; training rollouts only | [0.9, 1.1] (D38) |
| `randomize_payload_kg` | kg | added to the payload body (the root if it has mass, else the heaviest welded body, `perturbation.py:201-205`), inertia x (m + p) / m; training rollouts only | [0.0, 1.0] (D38) |

The per-body values are the build's (build commit 8198001c, `rl/trike_env.py:100-103`); the whole-robot values are
D38's (the steer1 presets, `rl/README.md:461-462`). Friction is absolute, not the multiplicative `friction_range` of
`docs/ROADMAP.md:248-250`: it reads as the build's numbers and is model-independent. Refused at construction:
friction with explicit contact pairs (`npair > 0`); kp with no position servo (Allosaurus); force limits with none;
a payload on a zero-mass body (R8). The robot's payload body is the chassis (5.9986 kg).

**Friction hook** (R5). `BaseDinoEnv._randomized_friction_geoms()` defaults to every colliding animal geom; the robot
returns its four `<foot>_sole` geoms, as the build writes the floor, the soles and its terrain boxes only
(`rl/trike_env.py:797-799`). Self-contacts keep their authored friction. On a behaviour heightfield pool the
heightfield is a static floor geom and takes the draw.

**Draws.** `begin_episode(env, seed)`: restore every model this env changed; rebuild the stream (§4.2,
`EPISODE_PHYSICS_STREAM_KEY`) when seeded or unset; draw `u = rng.random(3 + nbody + 2 x nu)` whatever is enabled
(49 on the robot): `u[0]` friction, then one per body, one per actuator for kp, one per actuator for force, `u[-2]`
total, `u[-1]` payload, each `low + (high - low) x u`; write the enabled quantities from the nominal copies in the
order friction, per-body, total, payload, kp, force. Total factor and payload apply only while `env.training_rollouts`
is true (elsewhere 1 and 0 kg), so a seed's per-body factors are the same in training and evaluation. Enabling one key
never moves another's draw. Per-seed draws make the fixed-seed panels paired: the recovery evaluation
(`recovery_evaluation.py:195, 268`), the stance report and the publication evaluation after
`eval_env.seed(PUBLICATION_SEED_START)` (`train_base.py:2519`) give every checkpoint and null controller the same
plant in episode k. The EvalCallback env is seeded once in `make_env` (`train_base.py:390`) and then reset unseeded,
so each round sees new plants, as it already sees new poses (R15).

**mj_setConst on scratch data.** If a mass key is on, `mj_setConst(model, record.scratch)` runs on the record's own
`MjData`, never the live one: on live data it moves qpos to `qpos_spring` (measured on all 7 manifest models and the
60 cm model). Then `stat.center`, `stat.extent` and `stat.meansize` go back to nominal; setConst recomputes them from
geometry unlike the compiler, and nothing randomised moves geometry.

**Exact restore.** Each model's record keeps exact copies of `RANDOMIZED_FIELDS` (geom_friction, body_mass,
body_inertia, actuator_gainprm, actuator_biasprm, actuator_forcerange), `SETCONST_FIELDS` (body_subtreemass,
body_invweight0, dof_invweight0, dof_M0, tendon_invweight0, tendon_length0, actuator_acc0, actuator_length0,
cam_pos0, cam_poscom0, cam_mat0, light_pos0, light_poscom0, light_dir0) and `stat` (meaninertia, meanmass, meansize,
extent, center), plus a scratch `MjData`. Restore is `np.copyto` from these, never another setConst (not a no-op on
four unmodified models, measured). Ranges are validated at construction; a model's record is taken the first time an
episode starts on it, which covers the constructor model and behaviour pool models alike and keeps a species edit made
after `super().__init__()` (R16). `nominal_physics()` captures the live values, restores nominal, yields, and writes
them back, bit-exact both ways; it captures on entry and nothing else is kept (R10). `episode_physics_factors` exposes
the episode's draws or `None`. No manifest method is added (as noise P10 drops one).

**Behaviour-env pools.** `SpeciesBehaviorMixin._select_contact_model` (`behavior_env.py:248`) first calls
`self._restore_episode_physics()`, before the swap and `_cache_ids()` (`:261`). Species `_cache_ids` cache weights from
masses (Compsognathus, Allosaurus, T. rex read-only; Velociraptor caches its weight once in `__init__`,
`base_env.py:439`, before any draw); without the restore the Compsognathus contact threshold followed the previous
episode's mass, up to x1.15 (measured).

**Pushes, settle, readers.** Push forces stay in newtons from the nominal plant (`base_env.py:299-329`), as the stage
fingerprint derives them; the settled pose depends on geometry only; no info key is added. The robot caches its
nominal weight in `_cache_ids` and never reads live masses in `_get_obs` or `_get_reward_info`. Readers of the live
model after a reset (`stance_report.py:993`, `balance_telemetry.py:131`, `body_telemetry.py:466`,
`floor_contact/morphology.py:498`) are checked on a robot env; a layout built there moves into `nominal_physics()`.

**Validation.** The physics digest covers every randomised field (`physics_layer.py:95-96, 145, 167-168, 179`), so the
identity is the nominal plant. `plant_contract/validation.py` stays unchanged and strict (R4). Every live site
validates before the first reset (`train_base.py:382-390`, `evaluation.py:316-326, 625-634`,
`stance_report.py:251-258`, `episode_trace.py:482-484`, `warm_start_noise_scan.py:703`, `config.py:457-458`,
`behavior_env.py:143`); after a randomised reset both validators refuse, which is fail-closed. A diagnostic that needs
the nominal plant later uses `with env.nominal_physics():`. `docs/PLANT_CONTRACT.md` states the rules: randomised
values live only between a reset and the next, are never fingerprinted, and no managed field may be written after
construction while randomisation is on.

**Fingerprint.** When any `randomize_*` key in the env section is not `None`:
`payload["episode_physics"] = {"implementation": "mesozoic.episode-physics-randomization/v1"}`; it joins
`_differing_sections` and `behavior_checkpoint.py:420` (R3).

**Catalog and docs.** `configs/species_manifest.toml:44-46`: `domain_randomization` status `"experimental"` (allowed,
`species_catalog.py:103`); `python -m environments.shared.species_catalog` regenerates the site JSON (R11).
`docs/PLANT_CONTRACT.md` gains "Per-episode physics randomisation". `docs/ROADMAP.md:240-266` marks friction, mass,
kp and force-limit randomisation done, notes absolute friction, and notes that the robot randomises in both stages, as
the build does, against the "Stage 2+ configs only" advice (`:264-265`, R12). `docs/hardware/SIM_TO_REAL_PLAN.md`
says the plumbing is absent (`:51-54`) and lists domain randomisation as planned (`:81`); it gains a dated status
note at its top, as its 2026-11-23 one, naming the delay and randomisation as opt-in shared features and the
capability as `experimental` (commit 4 adds noise to it). `docs/RECOMMENDATIONS.md:231-232` ("Not started") stays:
it is a dated snapshot, not maintained (`docs/README.md:48`).

**Files:** §3. **Tests** (new `test_episode_physics.py`):

- Off: `_episode_physics` and `episode_physics_factors` are `None` exactly when the stage omits every key;
  `BaseDinoEnv` has all six with default `None`; the classes listing a key are exactly an opted-in test table (empty
  now, the robot in commit 5), each defaulting it to `None` with an `OPT_IN_TASK_KWARGS` carve-out (R2). An off reset
  never reaches the randomiser (its `begin_episode` patched to raise; the reset record equals the golden's).
- With a randomiser injected by test subclasses on every manifest model: reset qpos, qvel, mocap, push schedule and
  `np_random` state equal the off env's; a managed field differs; on the Compsognathus-robot recovery stage the push
  force and `perturbation_manifest()` are equal. After a restore, every ndarray of `env.model` and the five stats
  equal a randomisation-off env of the same kwargs exactly (R13).
- Each quantity writes only its fields (friction on the floor and the hook's geoms, columns 1-2 kept; inertia ratio
  equals mass ratio; world and mocap masses kept; `gainprm[0] == -biasprm[1]`, `biasprm[2]` kept; one force factor;
  total mass = nominal x factor + payload with inertia ratio (m + p) / m in training, nominal in evaluation for the same
  seed; stats kept). setConst on scratch: with every mass key on and reset noise > 0, reset qpos equals the off env's.
- Draws: repeatable per seed, new on unseeded resets, equal across two envs, friction independent of other keys,
  `3 + nbody + 2 x nu` per episode, `np_random` equal on and off. Both validators refuse after a randomised reset and
  pass inside `nominal_physics()`. Pools restore before `_cache_ids` (Compsognathus robot, `mixed_terrain`, 8 resets).
- Refusals: friction (0.0, 1.0), (1.0, 0.5), (nan, 1), a 3-tuple; payload (-0.1, 0); kp and force keys on
  Allosaurus; a contact `<pair>` model built with MjSpec; a zero-mass payload body. Fingerprint: no section while off,
  the tag while live, off keys carved out.

**Measured on the prototype.** Off path: `plant_contract --check` and `digest_snapshot --check` current (725 lines),
`--exact` byte-identical (40,568 lines); 648 focused tests pass, and the full shared suite (7,305) fails only the 12
sandbox failures base also fails. Restore bit-identical to a fresh compile on all 7 models. On a thin 60 cm test env
over 2000 resets: per-body factors x0.928-1.072 of the 12.7506 kg animal (build: x0.930-1.073); with the whole-robot
factor and payload x0.865-1.211, mean 1.039 (build: x0.864-1.230, mean 1.038; `rl/README.md:400-401`). Reset 233 us
off, 429 us on; step time unchanged. The prototype wrote friction to 33 geoms; this design writes 5 on the robot.

**Built in commit 3, where it differs from the text above** (changed after the build and its review, not after any
training result):

- `BaseDinoEnv._configure_episode_physics(**ranges)` validates and installs the randomisation, restoring any randomised
  model first; `__init__` calls it after `_cache_ids` and the push derivation, and the test subclasses
  (`environments/shared/tests/episode_physics_helpers.py`) of species that do not forward the keys call it again after
  `super().__init__()`, as commit 2's delay subclasses call `_configure_actuation_delay`.
- `episode_physics_factors` holds the applied values by quantity, for the enabled keys only: `friction`,
  `body_mass_scale` (one per animal body), `actuator_kp_scale` (per position servo), `actuator_force_limit_scale` (per
  force-limited actuator), `total_mass_scale` (1.0 in an evaluation env) and `payload_kg` (0.0 there); `None` while
  off, before the first reset and after an episode whose values could not be written (a refusal when its model's
  record is taken, or a raise mid-write). A reset refused before it begins (inside `nominal_physics()`, below) leaves
  them, and everything else, as they were.
- More refusals than the list above, each a `ValueError` naming the key, at construction and again when a model's
  record is taken (its first episode; so a species edit made after `super().__init__()` also decides the payload
  body, the servo and force sets and the friction geoms, R16): friction with no static floor geom; a
  `_randomized_friction_geoms()` that returns no geom, anything but a 1-D array of integer ids (a boolean mask would
  index as geoms 0 and 1), an id outside the model, or a geom that is not a colliding animal geom; a behaviour pool
  model of another `nbody` or `nu` (at its first episode, since the draw layout is the constructor model's);
  `animal_bodies` refuses a model with no free joint. The construction check leaves the env's geom caches as it found
  them, so a collision-mask edit made after `super().__init__()` still reaches the settle.
- `nominal_physics()` blocks nest and write the episode's values back when they raise; a reset inside one raises
  `RuntimeError`, since the block's exit would write the previous episode's values over the new one, and so does
  reconfiguring the randomisation (`_configure_episode_physics`), whose new records would take the episode's values as
  nominal. Both are refused before anything moves: the hook's `reset` gains a first call,
  `self._check_episode_may_begin()`, before `super().reset` (so gymnasium reseeds nothing; nothing by default,
  `BaseDinoEnv`'s one `is not None` check while randomisation is off), and the behaviour mixin's `reset` calls it first
  too, before it moves its episode counters or selects a pool model. No committed digest hashes either source (§4.1:
  `episode_start.py` is not fingerprinted, and the policy layer's `home_reset` is the canonical env's
  `BaseDinoEnv.reset`, unchanged), so no digest moves (gates 2 and 3); a refused reset changes no attribute of the env
  (tested on the Compsognathus robot and its `mixed_terrain` behaviour env, where the refused reset would have swapped
  the plane for the heightfield). A raise while an episode's values are written restores the nominal plant before it
  propagates.
- `symmetry.mirror_env_state` refuses an env that holds a randomised plant (source, destination or in place): it
  mirrors the state, not the plant, and per-body and per-servo draws are not mirror-symmetric (pair-tied draws, §11).
- The scratch `MjData` is defensive: `begin_episode` runs before the keyframe load, which overwrites the live data, so
  setConst on the live data inside `reset` would not move the reset pose (it does anywhere else). The test checks
  which data setConst sees.
- The payload rule spreads the payload like the payload body's own mass, so it assumes a body heavy against the
  payload: a 1 kg payload on a 0.5 kg body triples that body's inertia, and only a massless body is refused. The
  robot's chassis (6 kg, up to 1 kg, x1.17) satisfies it; the module docstring and `docs/PLANT_CONTRACT.md` say so.
- setConst runs only when an episode wrote a mass, not whenever a mass key is on: in an evaluation env with only the
  two training-only keys on, nothing is written and the model stays exactly nominal, so the validators pass there (a
  test pins it). The values are the same either way.
- `TRAINING_ONLY_ENV_KEYS` names the two training-only keys, `GEOMETRIC_STAT_FIELDS` the three statistics put back
  after setConst, and `colliding_animal_geoms` is `_root_subtree_geoms()`'s definition read without its cache.
- Files beyond §3's commit-3 list: `episode_start.py` (the pre-reset check above), `symmetry.py` (the mirror refusal)
  and `tests/test_actuation_delay.py`, whose signature test now places the two delay keys right after
  `command_config` rather than last, since the six randomisation keys follow them.
- Tests: the payload range is 0 to a tenth of each model's animal mass; the pool test runs 16 resets (seed 0, then 15
  unseeded), which return to each pool model while the other's record is the newer; the draw layout is pinned index by
  index on every model, in training and evaluation.
- Handed to commit 5, besides the four readers named under "Pushes, settle, readers": `Morphology.from_env` reads
  `body_subtreemass` for `body_weight_n` (`floor_contact/morphology.py:498`), and `actuator_force_limits`
  (`gait/balance_telemetry.py:128`) the live force ranges, so the layouts built from them after a reset are
  per-episode: `reporting/stance_report.py:294`, `evaluation.py:482`, `eval_diagnostics.py:723, 954` and
  `rollout_ledger._layout_of` (`rollout_ledger.py:311`, built after the first reset and cached). On the robot env
  they are built inside `nominal_physics()` or before the first reset, or documented as per-episode.

## 7. Observation noise (commit 4)

**Keys:** `observation_noise_std: Mapping[str, float] | None = None` (std per segment, in its own units; kept as a
plain `dict`, since `SubprocVecEnv` pickles attributes) and `observation_noise_eval: bool = False`. The table is an
inline TOML table; `--override env.observation_noise_std.<segment>=x` works into a declared one (`cli.py:85-110`). A
child stage that `extends` another replaces a nested table whole (`config.py:246-248`), so it must restate or inherit
the whole table (P12). The robot's locomotion is written whole; a robot test asserts its table equals stance's.

**Where.** In `BaseDinoEnv.step` on the final observation, after the command recompute (`base_env.py:1384-1385`),
before render and return (`:1388-1391`), when active. Nothing is added in the hashed `_get_obs` or `reset`
(`policy_layer.py:378-406`). **The reset observation stays noise-free**: one per episode, 0.1 % of a 1000-step
episode; the build noises it (build commit 8198001c, `rl/trike_env.py:903`). Noising it in the hashed reset would move
every home-keyframe model's policy digest; a `Monitor` subclass in `make_env` and `evaluate` could do it later, as
transparent to SB3's `get_attr` as today's `Monitor` (P8).

**Active** when `_observation_noise is not None` and (`training_rollouts` or `observation_noise_eval`), latched in
`_begin_episode`. The spec's three roles, `set_observation_noise_role` and its `evaluation.py` edits are dropped.

**Module `environments/shared/observation_noise.py`** (numpy only at top level):
`OBSERVATION_NOISE_IMPLEMENTATION = "obs_noise/v1+gaussian-per-segment+post-step+episode-stream+unit-segment-renormalize"`;
`observation_layout(env)`, names from `policy_layer.observation_segments(...)`, widths from `_segment_width` except
`foot_contact`, whose width is the number of foot groups (Brachiosaurus: 8 touch sensors, 4 channels);
`ObservationNoise.apply(obs, rng)` adds `std x rng.standard_normal(width)` in float64 over the full width in
observation order (so one segment on or off never changes another's realisation), renormalises each
`*_orientation_quaternion`, `*_projected_gravity`, `*_direction` or `*_direction_body` segment with std above 0
(`v / max(|v|, 1e-9)`), and returns float32; `resolve_observation_noise` returns `None` while off and refuses a
non-bool eval, eval without a table, a non-mapping, widths not summing to the observation width, an unknown segment
(listing valid names), a bool, non-numeric, non-finite or negative std, `command > 0`, and a table with no positive
std. No `observation_noise_manifest()` (P10).

**Stream.** `_begin_observation_noise_episode` rebuilds `_observation_noise_rng` (§4.2,
`OBSERVATION_NOISE_STREAM_KEY`) when seeded or unset, replacing the prototype's `_step_count == 1` detection (P7).

**Segment table** (moves no digest). The inline table and label logic of `_policy_interface_payload`
(`policy_layer.py:344-373`) move into a public `observation_segments(model, env, observation_schema)`, which the
payload calls; the heading-free width check stays in the payload (`:350-359`). Commit 4 keeps today's rule; commit 5
extends it (§8.4).

**Robot table** (schema `quadrupedal-target-heading-free/v1`, root `torso`; build values at build commit 8198001c,
`rl/trike_env.py:113-119`):

| Segment | Width | Std | Source |
|---|---|---|---|
| `joint_position` | 20 | 0.005 rad | `noise_joint_pos` 0.005 rad, legs and ankles (`:114`); here the neck and tail too |
| `joint_velocity` | 20 | 0.05 rad/s | `noise_joint_vel` 0.05 rad/s (`:115`) |
| `torso_projected_gravity` | 3 | 0.02, renormalised | 0.01 per quaternion component, renormalised (`:117`), tilts about 2 x 0.01 rad per axis: 1.436 deg mean error, against 1.443 deg at 0.02 and 0.719 deg at 0.01 (P2) |
| `torso_angular_velocity` | 3 | 0.02 rad/s | `noise_gyro` 0.02 rad/s (`:116`) |
| `torso_heading_linear_velocity` | 3 | 0.02 m/s | `noise_lin_vel` 0.02 m/s on the privileged velocity (`:118`); isotropic, so the frame does not matter |
| `torso_linear_acceleration` | 3 | 0.2 m/s^2 | not observed by the build; g x 0.02 = 0.196 m/s^2, so it carries no more tilt than the gravity channel (P3) |
| `foot_contact` | 4 | 2.5 N | `noise_foot_load` 0.02 of weight (`:119`): 0.02 x 12.7506 kg x 9.81 m/s^2 = 2.50 N, a literal |
| `target_direction_body`, `target_distance`, `command` | 3, 1, 3 | 0 | the command is the task's input |

`observation_noise_eval = true` (D37). Heading drift (`rl/trike_env.py:120-123`) is moot: the heading-free
observation carries no heading (P9). TOML comments cite the values by build commit (P13).

**VecNormalize.** Noise is added inside the env, before `VecNormalize`, so `obs_rms` learns the noisy distribution
(measured on the prototype: joint-velocity `obs_rms.var` rose by 0.00259 against std^2 = 0.0025). Never after
normalisation. The command slice is never noised (`load_vecnorm_stats(reseed_command_slice=True)` unaffected);
critic-phase columns are appended after normalisation.

**Fingerprint.** When `observation_noise_std` is not `None`, `payload["observation_noise"] = {"implementation":
OBSERVATION_NOISE_IMPLEMENTATION}`; it joins `_differing_sections` and `behavior_checkpoint.py:420`. **Docs.**
`docs/PLANT_CONTRACT.md`: noise is task-level, in `step()`, outside the hashed methods; probe, validation and the reset
observation are clean. `docs/ROADMAP.md:268-277`: noise done, per-segment keys in place of `obs_noise_scale`. The
`SIM_TO_REAL_PLAN.md` status note (§6) gains sensor noise. `CONTRIBUTING.md:343-344` names the three modules as the
shared robot code's first members (commit 2 starts it; P13).

**Files:** §3. **Tests** (new `test_observation_noise.py`):

- Off: `_observation_noise is None` exactly when the stage omits the table; while inactive, `step()` returns the very
  object `_get_obs` returned, on every model (P5). Marking an env as training is inert while off (byte-identical).
- Layout: widths sum to the observation width; names equal `observation_segments(...)` and the payload's
  `observation_segments`; perturbing each source on the probe state changes only its segments; every segment is fed.
- Std on brachiosaurus (legacy) and allosaurus (heading-free) with a test table: reset until at least 2000 noisy steps,
  tolerance `5 / sqrt(2 n)` (P6); zero elsewhere; unit segments of norm 1 within 1e-6.
- Reproducible per seed; never draws `np_random`; qpos, reward, info equal the clean env's; unseeded episodes differ;
  the reset observation is clean. Active only for training rollouts or with the eval flag; `bank_env_kwargs` sets
  `observation_noise_eval = False` only where declared. Refusals; validation and the plant probe clean; section only
  while on.

**Measured on the prototype.** Off path bit-identical on brachiosaurus, the Compsognathus robot and allosaurus;
`--exact` identical (40,568 lines); `plant_contract --check` current with the refactor. Measured std matched the
setting (brachiosaurus, 1600 steps: 0.00500, 0.04995, 0.02027, 0.01984, 2.526 N); the build's own noise: 0.00499,
0.04994, 0.02032, 0.01999, load 0.02026 of weight. Cost 5.4-6.3 us per call.

## 8. The triceratops_horridus/robot model (commit 5)

### 8.1 Asset derivation and provenance

- **Source** (never imported): build commit f6deec9f9be1566e2c5de8e0932bc4d9d99ea2fd,
  `hardware/triceratops_build/sim/generated/trike60_C_closed_capsule_fore128.xml` (the D25 128 mm foreleg), blob
  04d3078ceb545d72561e1f519e464e94276b1653, sha256 372ac07b14338c26ce2fa8dbd0a34952597080fafcc82149464b10b25a059eb1:
  1 ms step, implicitfast, 16 position actuators (12 legs at kp 598.5; neck and tail at kp 8/6/5/8, kv
  0.3/0.2/0.15/0.2), four spring ankles (2.032 N m/rad), keyframes `neutral` and `home`, no mocap, 12.750593 kg.
- **Script** `environments/triceratops_horridus/scripts/derive_robot_model.py`, ElementTree edits (an MjSpec round
  trip lost the zero-distance hind-sole contacts); it refuses another sha256. (1) Model name
  `triceratops_horridus_robot`. (2) Leg prefixes become `fl_`, `fr_`, `rl_`, `rr_` (the morphology labels a foot by the
  first `_` token, `floor_contact/morphology.py:394-401`), and 8 unsided part pairs get side names. (3) Each held head
  and tail servo folds into its joint spring about target 0 (stiffness += kp, springref = k0 r0 / (k0 + kp), damping
  += kv); the 4 actuators and their force sensors go (nu 16 to 12, nsensor 61 to 57): neck_pitch 8.4 N m/rad,
  springref -1.471904762 deg, damping 0.34; neck_yaw 6/0/0.24; tail_yaw 5/0/0.19; tail_pitch 8/0/0.24. (4) Leg
  ctrlranges become the build's reach: roll +-0.15 rad, pitch and flexion +-0.40 rad (`rl/trike_env.py:84-86`).
  (5) One mocap body `target` at (2, 0, 0.3) m with an invisible site (`policy_layer.py:213-221` needs one). (6)
  `home` is the only keyframe, settled from qpos0 at ctrl 0 for 3000 x 1 ms, root x = y = 0, yaw removed, 9
  decimals: chassis z 0.319100188 m, clearance -0.215 mm.
- **Provenance**: the asset's first comment and `environments/triceratops_horridus/data/robot_model_provenance.json`
  (`mesozoic.derived-model-provenance/v1`) name the script, the source commit, path, blob and sha256, MuJoCo 3.10.0 and
  the renames; no branch path. Regenerate: `git show f6deec9f9be1566e2c5de8e0932bc4d9d99ea2fd:hardware/triceratops_build/sim/generated/trike60_C_closed_capsule_fore128.xml > /tmp/fore128.xml`,
  then `python -m environments.triceratops_horridus.scripts.derive_robot_model --source /tmp/fore128.xml`; `--check`
  compares byte for byte on MuJoCo 3.10.0.
- **Not mirror-symmetric** (IMU at y = +0.020 m, a 138 g parts reserve at y = -0.055 m, brow-horn meshes differing by
  about 3e-5 m): `derive_model_mirror` refuses it; no mirror map, as for the Compsognathus robot. The prototype's
  `symmetry.py` `OBSERVATION_LAYOUTS` entry is dropped, the module docstring no longer calls home mirror-symmetrised,
  and the renames' comment no longer says they let the mirror map derive (M7).
- **Spring fold, measured on the prototype:** the build's model-based trot (3 seeds x 20 s, randomisation, 10 ms)
  peaks neck_pitch at 1.13-1.26 N m against a 1.33 N m limit; held servos sit at their limit in 0.014-0.057 % of
  servo-steps. Filtered random leg targets at 0.2: roots within 1 mm for 10 s; at 0.4-0.6 they part after 0.7-2.4 s,
  against a 4-9 s chaos baseline.

### 8.2 Timestep

The 1 ms step stays: the build trains and judges at 1 ms with kp 598.5 implicitfast servos, and D33's values are whole
1 ms steps. Eight shared readers take the control step as `env_kwargs.get("timestep", 0.002) x frame_skip`:
`result_bundle/evidence.py:690, 861`, `reporting/gates.py:731-732`, `episode_trace.py:468`,
`stance_report.py:1236, 2151`, `eval_diagnostics.py:2794, 2828-2829`. So the species takes `timestep: float = 0.001`,
raises `ValueError` unless it equals `model.opt.timestep`, and every robot stage and recipe declares `frame_skip = 20`
and `timestep = 0.001`, pinned by a robot test (M8). `frame_skip` stays the default 20, which the policy payload
records (`policy_layer.py:418-420`).

### 8.3 Species base and robot env

- `TriceratopsHorridusEnv(BaseDinoEnv)` (`environments/triceratops_horridus/base_env.py`) holds every hashed method,
  so a later anatomical model moves no robot qualname. Attributes: `HEIGHT_LABEL = "Torso height"`,
  `BEHAVIOR_SECTION_LABEL = "Forward walking"`, `SUCCESS_KEYS = ("course_complete",)`, SB3 only,
  `action_mapping = "home-keyframe-residual/v1"`, camera tracking `chassis`; `FEET = ("fr", "fl", "rr", "rl")`.
- Defaults: `frame_skip=20`, `timestep=0.001`, `max_episode_steps=1000` (20 s); forward weight 0.0, `forward_vel_max`
  0.25 m/s; `alive_bonus=1.0`, `energy_penalty_weight=0.01`, `fall_penalty=-50.0`; `posture_weight=1.0`,
  `height_weight=0.5`, `height_tolerance=0.03` m; drift, heading, lateral, spin, smoothness, gait weights 0.0;
  `course_length=4.0` m, `course_radius=0.3` m; `healthy_z_range=(0.19, 0.6)` m, `max_tilt_angle=0.611` rad (35 deg),
  `reset_noise_scale=0.02`; the `perturbation_*` and command kwargs; the ten feature kwargs, each off. The class
  declares carve-outs with off value `None` (R2): `("actuation_delay_range_s", None, ACTUATION_DELAY_ENV_KEYS)`,
  `(name, None, (name,))` per `randomize_*` key, and `("observation_noise_std", None, OBSERVATION_NOISE_ENV_KEYS)`.
  Every robot stage sets every key, so they remove nothing from any committed task.
- `_cache_ids`: `_cache_home_keyframe`; chassis, floor and target ids; IMU offsets by sensor name (orientation 0,
  gyro 4, accelerometer 7; the base defaults differ); one touch sensor `<foot>_normal_load` per foot;
  `target_standing_z` 0.3191 m; nominal `body_mass` 12.7506 kg; `_contact_threshold` = 0.02 x weight = 2.50 N;
  `_body_ground_geoms` = the 28 colliding subtree geoms other than the soles. Gait contact threshold 2.0 N.
- `TriceratopsHorridusRobotEnv` (`robot/env.py`): `embodiment = "robot"`; `action_filter_cutoff_hz =
  (exp(2 pi x 4 x 0.02) - 1) / (2 pi x 0.02) = 5.19724 Hz`, giving the shared filter (`action_filter.py:56`) the
  build's 4 Hz alpha 0.395077; `FOOT_SUPPORT = {"support": ("{s}_sole",), "sole": None, "foot_site": "{s}_load_cell"}`
  (capsules have no flat pad, `morphology.py:610-621`); `REWARD_PROBE = (0.2, "body", "chassis")`;
  `scripts/view_model.py`. The shared filter seeds from the first action (`base_env.py:1233-1249`); the build from 0.

### 8.4 Observation schema and action mapping

New shared name `quadrupedal-target-heading-free/v1` (no world yaw, which a robot without a magnetometer cannot know).
`heading_observation.py` gains constants only: `QUADRUPEDAL_HEADING_FREE_OBSERVATION_SCHEMA` and
`HEADING_FREE_OBSERVATION_SCHEMAS`; `HEADING_FREE_FUNCTIONS` keeps its token hashes. `observation_segments()` uses
root `torso` for both quadrupedal schemas and the heading-free table for membership in
`HEADING_FREE_OBSERVATION_SCHEMAS`; existing schema strings take their old branch (noise P1). The payload's own
`heading_free` flag (`policy_layer.py:349`, an equality test against the bipedal schema today) becomes the same
membership test, so the robot's payload runs the width and command-last check (`:350-359`) and hashes
`HEADING_FREE_FUNCTIONS` (`:384-388`), the frame helpers its `_get_obs` calls; the existing schema strings take the
same branch as today, so no digest moves. `_get_obs`, 63 float32:
`qpos[7:]` (20: neck_pitch, neck_yaw, tail_yaw, tail_pitch, then fl, fr, rl, rr x roll, pitch, flexion, ankle),
`qvel[6:]` (20), projected gravity (3), gyro (3), heading-frame velocity (3, privileged), accelerometer (3), foot loads
(4, N), target direction in the chassis frame (3), distance (1), command (3). No previous action (§11). A
`docs/PLANT_CONTRACT.md` paragraph names the schema. Actions use `home-keyframe-residual/v1` unchanged: home ctrl is 0
and ctrlrange is the reach, so `_scale_action(a)` is exactly `a x reach`, the build's `neutral + a x scale`.

### 8.5 Rewards, termination and the arrival bonus (D39)

Shared terms in one left-associated chain, with Brachiosaurus' info keys: alive; forward, backward, drift
(`_progress_terms`); heading, lateral (`_heading_terms`); posture; `reward_target_centered_height`; spin; energy;
smoothness; quadruped gait symmetry; then `reward_arrival` (below). Info adds the per-foot MIN forces
(`r_/l_/rr_/rl_foot_contact`), `course_complete`, `reward_total` and `derive_stance_info`. Termination: non-finite
state; `_root_termination` (fallen, too high, excessive tilt); a non-sole floor contact (`body_contact`); and, when
`forward_vel_weight > 0`, the chassis within `course_radius` of the target in xy with at least 2 feet above 2.50 N at
the step's boundary sample (`course_complete`, success). Not the substep MIN: after a noisy reset all four MIN forces
can read 0 in the first steps (measured).

**Arrival bonus (D39).** The standing reward per step is the most the four always-on terms can pay a robot standing
still, read from the reward code (`reward_functions.py`): the sum of `alive_bonus`, 0 (posture is
`-w (tilt / max_tilt)^2`, at most 0), `height_weight` (height is `w exp(-(err / tol)^2)`) and `max(heading_weight, 0)`
(heading is `w cos`, `reward_functions.py:655-667`, on at every step). Locomotion: 1.0 + 0 + 0.5 + 0.3 = **1.8 per
step**.

**Why heading counts.** Heading is the fourth term on at every step in locomotion. With alive, posture and height
alone (1.5 per step), a robot stalled facing its course would earn 0.3 per remaining step more than the arriver, which
fails D39.

**What the guarantee covers.** In those four terms, arriving is never worth less than stalling. Every other term but
forward is a penalty, at most 0. The forward term (weight 1.5 x `clip(v / vmax, -1, 1)`, `reward_functions.py:31-45`)
is outside the bound: a statue's velocity jitter earns a little (the zero-action statue measured 1799.5-1800.3 per
1000 steps in locomotion, so up to +0.3 over 1.8 x 1000), and a robot that stops with walking momentum earns forward
reward while it slows. Stalling can therefore out-earn arriving by at most the forward reward the stalled robot earns.
The test (§8.10) checks the four-term bound, not the whole return.

In `_get_reward_info` the robot calls `TriceratopsHorridusEnv._is_terminated(self)` (a pure read of the post-step
state and the substep latch, as the behaviour mixin calls it, `behavior_env.py:335`). When the reason is
`course_complete` it adds `reward_arrival = (max_episode_steps - _step_count) x standing_reward_per_step` as the last
term of the chain and records it in info (0.0 otherwise). `step()` then gets the same decision; success means no fall
penalty (`base_env.py:1362-1366`). Stance never arrives (`forward_vel_weight = 0`); behaviour envs set it to 0
(`behavior_env.py:113-139`), so they never pay it. Estimate, not measured: a 0.25 m/s walker arriving at step 800
earns about 3.3 x 800 + 1.8 x 200 = 3000, against about 2940 for a 0.19 m/s walker that never arrives.

### 8.6 Stages and gates (provisional, D40)

`stages.toml` (`mesozoic.stage-manifest/v2`): `stance` (legacy 1, recipe `stance`, deliverable), then `locomotion`
(legacy 2, recipe `walk`, `warm_start_from = "stance"`, deliverable); no recovery stage. Both TOMLs written whole.

| | Stance | Locomotion |
|---|---|---|
| Task `[env]` | the defaults; drift 0.5, smoothness 0.02, spin 0.02 | forward 1.5, vmax 0.25 m/s, backward 0.5, heading 0.3, lateral 0.5, smoothness 0.02, spin 0.02, course 4.0 m, radius 0.3 m |
| Gate, `reward_and_length/v1` (D40, provisional) | reward 900 (0.6 x the statue's 1500), length 950 of 1000 | reward 2000 (statue 1800), length 550, forward 0.15 m/s |
| Other | collapse floor 1120, 3 consecutive, 3 M steps | collapse floor 1500, 3 consecutive, 10 M steps |

Feature block, in both stages and every recipe, each line's comment citing its decision: `actuation_delay_range_s =
[0.0, 0.02]`, `actuation_delay_eval_s = 0.01` (D33); `randomize_friction = [0.6, 1.0]`, `randomize_body_mass_scale =
[0.9, 1.1]`, `randomize_actuator_kp_scale = [0.9, 1.1]`, `randomize_actuator_force_limit_scale = [0.85, 1.0]` (D32);
`randomize_total_mass_scale = [0.9, 1.1]`, `randomize_payload_kg = [0.0, 1.0]` (D38, training only);
`observation_noise_eval = true` (D37) and the §7 table (D32). PPO (`rl/train.py:108-121`, `:731-737`): lr 3e-4 to 3e-5
linear, `n_steps` 1024, batch 2048, 5 epochs, gamma 0.99, GAE 0.95, clip 0.2, `ent_coef` 0, `max_grad_norm` 1.0,
[256, 256] for the policy and the value net, `log_std_init` -1.6; the build's `vf_coef` 0.5 is SB3's default. The
build's `activation_fn=torch.nn.ELU` (`:736`) is not carried over: a TOML `policy_kwargs` cannot name a class (main
handles one only in `critic_phase_ppo.py`), so the robot's PPO uses SB3's default, Tanh. The stage TOMLs' `[ppo]`
comment says so, and §11 lists it. SAC: lr 3e-4, batch 256, buffer 300,000, starts 10,000, `train_freq` 4, 4 gradient
steps. The statue passes the stance gate (1500 at every reset noise 0.01-0.2, measured): a warm start, as for
Brachiosaurus.

### 8.7 Behaviour recipes

All 11 ship (`test_behavior_recipes.py:49-61` and `python-ci.yml:191-210` require them), rescaled from the Compsognathus
robot's: cruise 0.25 m/s, speed range [0.1, 0.25] m/s, yaw rate up to 0.4 rad/s, switches every 4.0 s; terrain 7.5 m,
301 x 301 samples, features 0.006 m high (1.9 % of 0.319 m), radius 0.20-0.35 m. `[env]`: `frame_skip = 20`,
`timestep = 0.001`, `max_episode_steps = 1250` (25 s), `course_distance = 5.0`, and locomotion's feature block. Every
recipe is one of main's own command tasks (direction, speed and yaw-rate commands from `[commands]`), not the build's
steering mode (D32). They ship only because `test_behavior_recipes.py` and CI require the full set for every SB3 model,
and none runs unless someone runs it: the demo is flat ground (D28), so the 9 terrain recipes have no use yet, and
`follow_direction` and `follow_direction_speed` train heading and speed commands, the kind of steering D32 keeps in the
build stack. The README says so; owner point 6 asks whether shipping them fits D32. Behaviour training draws the delay
and whole-robot factors because `train_behaviors` marks its env (§4.3, M10).

### 8.8 Manifest and generated files

- `configs/species_manifest.toml`: `[species.triceratops_horridus]` (display order 7) and its `robot` embodiment
  (tagline "Twelve-Servo 60 cm Quadruped Prototype", entry point
  `environments.triceratops_horridus.robot.env:TriceratopsHorridusRobotEnv`, the asset, SB3, notebook `sb3_training`);
  success metric `course_complete`; deliverable metrics (stance reward rail, noting the statue clears it; walk
  forward-velocity gate). `configs/plant_versions.toml`: note 24 (source commit and sha256, D25, the edits, the new
  schema) and `[plants.triceratops_horridus.robot]` at revisions 1/1/1.
- Regenerate in order on MuJoCo 3.10.0, after the formatting step of §9 (M2): `plant_contract --write`,
  `species_catalog`, `harnesses.digest_snapshot --block-optional-backends --write` (each `python -m
  environments.shared.<module>`); then every gate of §9.
- Measured on the prototype, before D38 and D39 changed the robot's task and reward: each plant manifest +42 lines;
  digest snapshot 725 to 815 lines, all 90 added lines the robot's (17 plant, 29 policy, 14 stage, 22 behaviour, 8
  reward); `--exact` 40,568 to 43,984 lines, no other line changed.

### 8.9 Shared tests to generalise

Each table gains `TRI = ModelId("triceratops_horridus", "robot")` or a generalisation:

| File (`environments/shared/tests/`) | Change |
|---|---|
| `test_mjcf_assets.py` (`EXPECTED_MODELS`, :27) | (27, 26, 12, 12.750593) |
| `test_behavior_env.py` | `MODEL_IDS`; `SPEEDS[TRI] = 0.25` |
| `test_plant_contract_digest_snapshot.py` | `_SUCCESS[TRI] = {"locomotion": "course_complete"}`; `_POSE_REASONS[TRI]`; new `_ROLL_STANDS = {TRI}` pinned the other way at :273 (the harness's fixed kick does not topple it at 0.2, 0.5 or 1.0; 9 of 40 random held kicks do) |
| `test_plant_contract_layers.py` | `SB3_INTERFACE_QUALNAMES[TRI]`: `TriceratopsHorridusEnv._get_obs`, `_scale_action`, `_cache_ids` |
| `test_plant_contract_manifest.py` | 63 / 12 / (63, 12) |
| `test_gait_morphology.py` | weight 125.08 N; feet (fr, fl, rr, rl); root `chassis`; sole `None`; a `<foot>_roll_yoke` branch in `test_legs_are_whole_limbs` |
| `test_gait_pad_telemetry.py`, `test_gait_recorder.py` | whole foot (False, 0), no pad or rail feet; statue COP ceiling `None` |
| `test_species_names.py` | "Triceratops horridus", "Triceratops horridus (Robot)" |
| `test_species_catalog.py` | `MODEL_IDS` (:64); dims (63, 12, 27, 26, 12, 12.750593); schema; revisions 1/1/1; a TRI branch in the early-gates test |
| `test_species_integration.py` | TRI in `MODEL_ENVS` (:37-47), `SPECIES_DIMS` (63, 12), `SPECIES_REWARD_KEYS` (the 12 terms, `reward_arrival`, `reward_total`) and `test_gym_make`; excluded next to Allosaurus at :396; not in `EXEMPT` (M3) |
| `test_stage_manifest.py` | `WALK_ONLY_MODELS = frozenset({ALLOSAURUS_FRAGILIS, TRI})` at :562, :683, :770-777, :815 |
| `test_curriculum_manager.py:480` | `len(model_ids()) == 8` |
| `test_symmetry.py` (:107) | `NOT_MIRRORED = {ROBOT: ..., TRI: ...}`, with reasons |
| `test_behavior_recipes.py:191` | `frame_skip` from the recipe or the class default, not `10 if compsognathus else 5` |
| `test_model_identity_contract.py` | passes once §8.11's site and docs files are tracked |
| §5-§7 feature tests | the opted-in table gains TRI; the exact-delay test gains the robot |

### 8.10 Robot tests

New `environments/triceratops_horridus/tests/` (`test_robot_env.py`, `test_robot_features.py`), added to
`pyproject.toml` testpaths and the CI species suite (`python-ci.yml:345-352`); `test_ci_tool_pins.py` checks both.
Neither file imports SB3 or calls `make_env`, so both run in the species matrix and stay out of the test-sb3 step; the
`make_env` and `train_behaviors` marking is tested once in the shared SB3 files (§5).

- Asset: source commit, path and sha256 in the header; dt 0.001 s; (nq, nv, nu, nkey, nmocap) = (27, 26, 12, 1, 1);
  key 0 `home`, reproduced by re-running `settle` within 1e-6; the 12 drives in order with ctrlrange exactly +-reach;
  the four spring folds; the provenance record matches the asset and script.
- Zero action with the features off: 1000 steps, truncated, height drift under 2 mm, every foot above 15 % of weight,
  foot sum = `model.body_subtreemass[chassis] x g` read after reset, within 2 % (M4). Under every feature, stance and
  locomotion, training and evaluation, 8 seeds: stands, with no weight check (M4, R14). Measured on the prototype
  without D38's draws: 12 of 12 seeds stood the full 1000 steps in both stages and both modes; with them, not yet.
- Interface: obs 63, act 12, dt 0.02 s; `_scale_action(a) == a x reach`; filter alpha `1 - exp(-2 pi x 4 x 0.02)`
  within 1e-12; schema, touch order, morphology labels; `_scale_action(0)` equals the reset key's ctrl (P10).
- Course: stance never completes; locomotion does; a chassis lifted 0.1 m never does. A non-sole floor contact ends
  the episode; `timestep=0.002` raises; every stage and recipe declares the model's timestep and `frame_skip = 20`
  (M8); the gates equal D40's values.
- Arrival (D39): `reward_arrival == (1000 - n) x 1.8` exactly, inside `reward_total`, no fall penalty. On a twin from
  the same seed and the same actions to step n, with the target just out of reach, then zero action to truncation,
  every step's alive + posture + height + heading is at most 1.8, so their sum over the remaining steps is at most the
  bonus. The test compares that four-term sum only, not the twin's whole return, which also holds forward reward
  (§8.5).
- Features: stance, locomotion and every recipe resolved by `read_recipe` carry the one block, and the noise tables are
  equal (P12); constructor defaults are off; a `config.build_env` env reports `actuation_delay_steps == 10` at every
  step while the same env marked with `mark_training_rollouts(env, True)` draws 0-20 over 60 seeds; a
  `create_behavior_env` env draws 0-20 when marked and holds 10 when not (M10); for one seed the whole-robot factor and
  payload apply in training only and the per-body factors match (D37, D38); friction reaches the floor and the four
  soles only (R5); the noise table equals §7; projected-gravity tilt error 1.44 +- 0.1 deg (P2); std within
  `5 / sqrt(2 n)` (P6); both validators pass before the first reset and inside `nominal_physics()` after one (R4).

### 8.11 Docs and website

- `environments/triceratops_horridus/README.md`: Models, Run the model, Training (stance then walk, the features, flat
  ground D28, gates provisional D40), Task and rewards (D39), Files, Further reading (the build by commit; the adult
  study in `examples/triceratops_horridus` stays outside the manifest). `notebooks/sb3_training.ipynb`: the species
  list gains `triceratops_horridus`; two markdown lines name the robot.
- Website (paths under `website/`): `docs/models/triceratops_horridus/_category_.json` (position 7), `robot.mdx`;
  `docs/intro.md` (eight training models across seven species; line 23's eight animals stay); `docs/api/overview.md`;
  `getting-started/installation.md:58`, `training/recipes.md:107, 466` ("walk for Allosaurus and the Triceratops
  robot"); `src/components/AnimalPage/interpretation.ts` robot entry (`tests/animal_interpretation.test.mjs:82-103`);
  one animal page: `src/data/animals.ts` folds the model into the hand-written `triceratops` card,
  `plugins/animal-pages.cjs` skips its auto route, and `src/data/triceratops.ts` and `src/pages/animals/triceratops.tsx`
  link to the model page. `npm test` in `website/` where `node_modules` exists.
- `docs/SPECIES_NAMING.md` row; root `README.md` prose under the generated table; the module docstring of
  `environments/shared/scripts/warm_start_noise_scan.py` (`:13`, `:89`: "seven models" becomes eight); `docs/ROADMAP.md`
  Phase 6 (`:515-528`, M9); `CHANGELOG.md`. `docs/hardware/README.md` (`:5-10`) says only an adopted design becomes a
  species' robot model (the maintainer's decision of 2026-11-26); D32 and D36 make the 60 cm design that model in
  simulation, so the page says so, citing them, and its `60 cm Triceratops build` row (`:71`) adds the model, derived
  from build commit f6deec9f (§11).

## 9. Verification gates

Every code commit (2-5) passes all of these locally; the outputs are kept and their summary lines quoted in the PR
description. Install with the canonical MuJoCo and tensorboard (the `train` extra): `python -m pip install -e
".[train,test]" "mujoco==3.10.0" "ruff==0.16.9" "mypy==2.3.1" -c constraints.txt`. Before the gates, run `ruff check
--fix` and `ruff format` on the commit's Python files (R9, P11, M2).

1. Lint and types (`python-ci.yml:101, 104, 107, 118, 430`):

   ```bash
   ruff check environments/ notebooks/colab_install.py
   ruff format --check environments/ notebooks/colab_install.py
   mypy environments/ notebooks/colab_install.py --ignore-missing-imports
   mypy environments/ --ignore-missing-imports          # with SB3 and torch installed
   python environments/shared/tests/notebook_cells.py
   ```

2. Generated files (`python-ci.yml:141-142, 151, 162`; `docs/PLANT_CONTRACT.md:98`):

   ```bash
   python -m environments.shared.plant_contract --check
   git show 2c1039f4:configs/plant_manifest.generated.json > /tmp/base-plant-manifest.json
   python -m environments.shared.plant_contract --check --baseline /tmp/base-plant-manifest.json
   python -m environments.shared.species_catalog --check
   python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check
   python -m environments.shared.symmetry --check
   git diff --exit-code 2c1039f4 -- configs/digest_snapshot.generated.txt configs/plant_manifest.generated.json \
       environments/shared/data/plant_manifest.generated.json   # commits 2-4; commit 5 only adds robot lines
   ```

3. Bit-exact capture against main (CONTRIBUTING.md step 4; `-m` from each checkout's root, as the harness enforces):

   ```bash
   git worktree add --detach /tmp/base 2c1039f4
   (cd /tmp/base && python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --exact) > /tmp/exact_base.txt
   python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --exact > /tmp/exact_head.txt
   diff /tmp/exact_base.txt /tmp/exact_head.txt                                   # commits 2-4: empty
   diff /tmp/exact_base.txt <(grep -v triceratops_horridus /tmp/exact_head.txt)   # commit 5: empty
   ```

4. The full suite over `pyproject.toml` testpaths (commit 5 adds the robot's directory), compared by node id with
   main's run the same way in the base worktree:

   ```bash
   PYTHONPATH=$PWD python -m pytest -rfE -p no:cacheprovider 2>&1 | tee /tmp/pytest_head.log
   (cd /tmp/base && PYTHONPATH=/tmp/base python -m pytest -rfE -p no:cacheprovider) 2>&1 | tee /tmp/pytest_base.log
   comm -13 <(grep -oE '^(FAILED|ERROR) [^ ]+' /tmp/pytest_base.log | sort -u) \
            <(grep -oE '^(FAILED|ERROR) [^ ]+' /tmp/pytest_head.log | sort -u)    # must print nothing
   ```

   A head failure that main also has is listed by node id in the PR description. Remove the base worktree after.

   **CI placement.** The new test files import no SB3 and run in the `test` matrix: `test_episode_start.py`,
   `test_actuation_delay.py`, `test_episode_physics.py` and `test_observation_noise.py` in the shared suite, the
   robot's two in the species suite. The SB3 checks (`create_vec_env(training=True)`, `train_behaviors.main`'s marking)
   go in `test_train_base.py` and `test_behavior_checkpoint.py`, which the test-sb3 step already runs, so that step is
   unchanged and `test_ci_tool_pins.py:364-376` holds (§5).

   **Measured on the prototypes**, shared suite only, without tensorboard, failures by node id:
   - Randomisation, full suite (7,305 tests): 12 failures, all also on base: the 9 tensorboard failures
     (`test_compsognathus_longipes_training.py` x4, `test_curriculum_cli_end_to_end.py` x4, `test_train_wiring.py` x1)
     and 3 subprocess imports of `environments` without an install
     (`test_plant_contract_digest_snapshot.py::test_refusals_come_before_the_run` x3).
   - Delay, full suite with `PYTHONPATH` set (the skeptic's run): the 9 tensorboard failures, plus
     `test_behavior_checkpoint.py::test_real_ppo_cli_resume_preserves_recipe_and_releases_stage_warmup`, which passes
     on base. `mark_training_rollouts` fixes it by design (P1).
   - Merged model prototype, full suite: those 10, plus
     `test_species_catalog.py::test_every_species_model_page_is_tracked_by_git`, an artefact of the new files being
     untracked (it passes after `git add -N`).
   - Noise: focused tests only (615 passed, 1 failed; the fix was the noise role helper this design drops).

   No full-suite pass exists yet for the reconciled design.

## 10. Skeptic findings and their resolution

| ID | Finding | Resolution | Section |
|---|---|---|---|
| Delay P1 | `train_behaviors` called `set_training_rollouts` on a plain test env | `mark_training_rollouts` on `env.unwrapped`, no-op without the method | §4.3 |
| Delay P2 | Only 1243 of 7305 shared tests run; no test gate | Full suite over testpaths, compared by node id | §9 |
| Delay P3 | `actuation_delay_steps` stale between reset and the first step | k chosen at reset; property valid from reset | §5 |
| Delay P4 | `k x timestep` is not the decimal seconds | Info carries the steps and the seconds rounded to 12 decimals | §5 |
| Delay P5 | Mirroring copied the role | FIFO and k only | §5 |
| Delay P6 | `behavior_checkpoint` lacks the `actuation_delay` section | Added, with `episode_physics` and `observation_noise` | §5 |
| Delay P7 | "Refused naming env and actuation_delay" overstated | Reworded; switch-off case tested | §5 |
| Delay P8 | Harness docstring says stages are built as `make_env` builds them | Names evaluation envs | §5 |
| Delay P9 | Citations off; 0-20 ms is not the build default | Corrected; D33 and the steer1 presets cited | §5 |
| Delay P10 | Zero-action invariance needs prime == `_scale_action(0)` | Precondition stated; robot test | §5, §8.10 |
| Delay P11 | `warm_start_noise_scan` sees 10 ms, not the training draw | Listed as an evaluation diagnostic; docstring note | §4.3 |
| Rand. R1 | Delay and randomisation differ on RNG and on evaluation-only draws | One stream policy; one role flag read by the whole-robot draws | §4.2, §4.3, §6 |
| Rand. R2 | "No species lists a key" fails once the robot does | Opted-in table with carve-outs | §6, §8.3 |
| Rand. R3 | `episode_physics` missing from `_differing_sections` | Added, and to `behavior_checkpoint` | §6 |
| Rand. R4 | The `nominal_physics()` wrap in validation is unused and weakens it | `validation.py` unchanged and strict | §6 |
| Rand. R5 | Friction on every animal geom changes self-contacts | `_randomized_friction_geoms()`; the robot's 4 soles | §6 |
| Rand. R6 | `np_random_seed` needs gymnasium >= 1.0 | Seed argument first, then `getattr` with fallback | §4.2 |
| Rand. R7 | The state fallback assumes PCG64 | PCG64 read; others raise `ValueError` | §4.2 |
| Rand. R8 | A massless payload body divides by zero | Refused at construction | §6 |
| Rand. R9 | `ruff format` fails on the module | `ruff format` before the gates | §9 |
| Rand. R10 | `_applied` captured and never read | Removed; `nominal_physics()` captures on entry | §6 |
| Rand. R11 | Capability still "planned" on the site | `experimental`, catalog regenerated | §6 |
| Rand. R12 | Randomising stance departs from ROADMAP's "Stage 2+" | Both stages, as the build; noted; owner point | §6, §11 |
| Rand. R13 | Restore test against a fresh compile fails for post-edited models | Compare with an off env of the same kwargs | §6 |
| Rand. R14 | Masses included the mocap target; standing measured on a thin env | Animal subtree masses; standing re-tested on the robot env | §6, §8.10 |
| Rand. R15 | "Same plant in episode k" overstated | Fixed-seed panels only; EvalCallback rounds reset unseeded and see new plants | §6 |
| Rand. R16 | Later writes to managed fields silently undone | Rule in PLANT_CONTRACT.md; record taken at a model's first episode | §6 |
| Noise P1 | Robot table used names its schema lacks | Heading-free names; `observation_segments()` dispatch | §7, §8.4 |
| Noise P2 | Gravity 0.01 halves the build's tilt noise | 0.02; tilt test 1.44 +- 0.1 deg | §7, §8.10 |
| Noise P3 | A perfect accelerometer leaks tilt | 0.2 m/s^2 | §7 |
| Noise P4 | Three features, three rules for who sees them | Option b: training rollouts or the eval flag; bank forces off | §4.3, §7 |
| Noise P5 | Off-by-default test fails for the robot | Off exactly when the stage omits the key | §7 |
| Noise P6 | 3 % tolerance too tight for short episodes | At least 2000 noisy samples, `5 / sqrt(2 n)` | §7 |
| Noise P7 | `_step_count == 1` detection is a heuristic | Stream built in the hook | §4.2, §7 |
| Noise P8 | The wrapper option's cost overstated | Restated as a `Monitor` subclass; a follow-up | §7, §11 |
| Noise P9 | Heading drift framed for a quaternion | Moot: no heading channel | §7 |
| Noise P10 | `observation_noise_manifest()` has no caller | Dropped | §7 |
| Noise P11 | `ruff format` fails on the module | Formatted before the gates | §9 |
| Noise P12 | `extends` replaces a nested table whole | Documented; equal-tables robot test | §7, §8.10 |
| Noise P13 | Provenance by branch path; stale CONTRIBUTING line | Cite by build commit; CONTRIBUTING names the modules | §7 |
| Model M1 | The model cannot land alone | Model last, regenerated on the merged tree | §3 |
| Model M2 | ruff errors and formatting; mypy not run | `ruff check --fix`, `ruff format` before regenerating; mypy in the gates | §8.8, §9 |
| Model M3 | Exempting the robot from `test_species_integration` drops coverage | Added to its tables; not exempt | §8.9 |
| Model M4 | Standing test depends on seed 0's mass draw | Live subtree weight; features-on test without a weight check | §8.10 |
| Model M5 | The reward discourages arriving | D39 arrival bonus | §8.5 |
| Model M6 | Delay and mass values attributed to the build's defaults | D33 and D38 cited; whole-robot draws on, training only | §5, §6, §8.6 |
| Model M7 | Dead `symmetry.py` entry; contradictory docstrings | Entry dropped; docstrings fixed | §8.1 |
| Model M8 | A TOML missing `timestep` is not caught | Robot test on every stage and recipe | §8.2, §8.10 |
| Model M9 | ROADMAP Phase 6 and the capability status go stale | Both updated | §6, §8.11 |
| Model M10 | Behaviour envs held 10 ms without the trainer's mark | `train_behaviors` marks via the helper; robot test | §4.3, §8.10 |

## 11. Not in this PR, and points for the owner

**Follow-ups:**

- A training-mode capture in the digest snapshot; a per-episode delay column in `train_episodes.csv`
  (`rollout_ledger.py:390`).
- A previous-action (filter-state) observation segment, as the build has: a policy-interface revision needing fresh
  checkpoints. Until then in-flight targets are hidden state.
- The eight readers of §8.2 reading the model's own timestep; noise on the reset observation; per-joint noise.
- Existing species opting in, with the carve-out at off value `None` (`save_stage_config` converts tuples first,
  `config.py:1137-1152`); measured on the prototype for the Compsognathus robot: every committed digest kept while off.
- Pair-tied draws for a mirror-augmented model; 16 actuators if the head and tail dynamics matter; steering and heat
  costs (D32); a smaller reset qvel noise; the arrival bonus under `VecNormalize`'s reward clip of 50
  (`constants.py:22`), checked in the first run.
- A supported way to name the PPO activation in a stage TOML, so the robot can train with the build's ELU (§8.6); until
  then it trains with SB3's default Tanh, a declared deviation from the build.

**Points for the owner to confirm:**

1. `docs/hardware/README.md` says only an adopted design becomes a species' robot model (2026-11-26). This PR makes
   the 60 cm design `triceratops_horridus/robot` in simulation (D32, D36) and updates the page. Confirm the intent, and
   that it does not by itself adopt the design for building.
2. The gates and course values are provisional (D40), revised after the first main-pipeline run.
3. The arrival bonus counts the heading term (1.8 per step, not 1.5; §8.5), so in the four always-on
   terms arriving is never worth less than stalling facing the course. The forward term is outside that bound: a
   statue's velocity jitter measured up to +0.3 per 1000 steps, and a robot that stops with walking momentum earns
   forward reward while it slows (§8.5). Whether a walker that passes more than 0.3 m to the side out-earns one that
   arrives depends on how the forward reference turns as it passes the target; estimated from the weights at up to
   about 3300 against 3000, not measured. Commit 5 measures it on the robot env; D39 does not cover that case.
4. The robot randomises in stance too, as the build does (R12); friction is absolute, not ROADMAP's scale.
5. The reset observation is noise-free, unlike the build (one observation per episode).
6. Taken as the model design recommended: head and tail folded into springs (nu 12); the new schema name; all 11
   recipes on a flat-ground robot (D28); a stance the statue passes, kept as a warm start; one animal page on the site.
   On the recipes, a D32 scope question: all 11 are main's own command tasks, and `follow_direction` and
   `follow_direction_speed` train heading, speed and yaw-rate commands, while D32 keeps steering in the build stack.
   They ship only because `test_behavior_recipes.py` and CI require the full set, and they are inert unless run
   (§8.7). Confirm that shipping them fits D32, or say whether the robot should be exempted from the full-set rule.
