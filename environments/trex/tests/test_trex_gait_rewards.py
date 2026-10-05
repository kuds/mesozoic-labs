"""The gait reward kit wired into the T. rex env (GAIT_QUALITY_PLAN_2026_09 §5.2-§5.4).

At its legacy values the kit is inert: no hook, no info key, the same reward
and the same task digests.  Set, its terms read the real floor contacts
without touching the dynamics.  The locomotion stage sets it (task revision
``gait-r1``, decision D-D23); every other stage keeps its pre-kit digest, so
the certified r13 stance (run 20260914_123816, seed 42) stays reusable as the
trunk of a gait-r1 locomotion run.  The terms themselves are pinned on
scripted contact sequences in ``environments/shared/tests/test_gait_rewards.py``.
"""

from __future__ import annotations

import inspect
from typing import Any

import mujoco
import numpy as np
import pytest

from environments.shared.gait.morphology import GaitMorphology
from environments.shared.gait.recorder import SubstepContactRecorder
from environments.shared.gait_rewards import GAIT_REWARD_KIT_LEGACY
from environments.trex.envs.trex_env import TRexEnv

#: The kit knobs of the trex locomotion task revision gait-r1 (configs/trex/locomotion.toml, decision
#: D-D23): floor support, gait phase 0.5, flight 1.0, slip 0.2.
GAIT_R1: dict[str, Any] = {
    "support_source": "floor",
    "gait_phase_weight": 0.5,
    "flight_penalty_weight": 1.0,
    "foot_slip_penalty_weight": 0.2,
}
#: The rest of the revision: the speed cap lowered with its slope kept, and the 20 s horizon.
GAIT_R1_TASK: dict[str, Any] = {
    **GAIT_R1,
    "forward_vel_weight": 1.0,
    "forward_vel_max": 1.25,
    "max_episode_steps": 2000,
}
#: Every knob set, none of them ending an episode.
ALL_TERMS: dict[str, Any] = {
    **GAIT_R1,
    "foot_collision_penalty_weight": 0.5,
    "leg_contact_penalty_weight": 1.0,
}
KIT_INFO_KEYS = {
    "gait_support_force_r",
    "gait_support_force_l",
    "gait_feet_down",
    "gait_flight",
    "reward_flight",
    "gait_foot_slip",
    "reward_foot_slip",
    "gait_foot_collision",
    "reward_foot_collision",
    "gait_leg_contact",
    "reward_leg_contact",
    "gait_phase_steps",
    "gait_phase_quality",
    "reward_gait_phase",
}
KIT_REWARDS = ("reward_flight", "reward_foot_slip", "reward_foot_collision", "reward_leg_contact", "reward_gait_phase")


def _rollout(env: TRexEnv, steps: int, seed: int = 3, amplitude: float = 0.25) -> list[tuple]:
    """A seeded random-action rollout: per step (obs, qpos, reward, terminated, info)."""
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        action = rng.uniform(-amplitude, amplitude, env.model.nu).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        out.append((obs.copy(), env.data.qpos.copy(), reward, terminated, info))
        if terminated or truncated:
            break
    return out


@pytest.fixture
def make_env():
    envs: list[TRexEnv] = []

    def make(**kwargs) -> TRexEnv:
        env = TRexEnv(**kwargs)
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


# ---------------------------------------------------------------------------
# legacy values: inert
# ---------------------------------------------------------------------------


def test_constructor_defaults_are_the_legacy_values() -> None:
    parameters = inspect.signature(TRexEnv).parameters
    for name, legacy in GAIT_REWARD_KIT_LEGACY.items():
        default = parameters[name].default
        assert default == legacy and type(default) is type(legacy), name


def test_legacy_kit_is_inert(make_env) -> None:
    env = make_env()
    explicit = make_env(**GAIT_REWARD_KIT_LEGACY)
    for candidate in (env, explicit):
        assert not candidate._gait_reward_kit.active
        assert candidate._substep_reward_hook is None
    legacy, spelled = _rollout(env, 60), _rollout(explicit, 60)
    assert len(legacy) == len(spelled) == 60
    for (obs, qpos, reward, _, info), (obs2, qpos2, reward2, _, info2) in zip(legacy, spelled, strict=True):
        assert np.array_equal(obs, obs2) and np.array_equal(qpos, qpos2)
        assert reward == reward2  # bit for bit
        assert list(info) == list(info2)  # same keys, same order
        assert not KIT_INFO_KEYS & set(info)


def test_kit_is_read_only_and_adds_only_its_terms(make_env) -> None:
    # Every non-terminating knob set: the same trajectory as the legacy env, and the
    # reward differs by exactly the kit's terms (T. rex's bilateral weights are 0).
    legacy = _rollout(make_env(), 200)
    kit = _rollout(make_env(**ALL_TERMS), 200)
    assert len(legacy) == len(kit) > 50
    for (obs, qpos, reward, terminated, info), (obs2, qpos2, reward2, terminated2, info2) in zip(
        legacy, kit, strict=True
    ):
        assert np.array_equal(obs, obs2) and np.array_equal(qpos, qpos2)
        assert terminated == terminated2
        assert KIT_INFO_KEYS <= set(info2)
        assert [key for key in info2 if key not in KIT_INFO_KEYS] == list(info)
        assert reward2 == pytest.approx(reward + sum(info2[key] for key in KIT_REWARDS), abs=1e-9)
        assert info2["r_foot_contact"] == info["r_foot_contact"]  # the touch keys stay touch


# ---------------------------------------------------------------------------
# set: what the terms read
# ---------------------------------------------------------------------------


def test_floor_support_is_the_recorders_floor_force(make_env) -> None:
    # The kit's per-foot floor force is the gait recorder's (the checker's floor truth), MIN over the step.
    env = make_env(support_source="floor")
    env.reset(seed=5)
    morphology = GaitMorphology.from_env(env, "trex")
    rng = np.random.default_rng(5)
    with SubstepContactRecorder(env, morphology) as recorder:
        for _ in range(80):
            _, _, terminated, _, info = env.step(rng.uniform(-0.3, 0.3, env.model.nu).astype(np.float32))
            rows = recorder.trace()["floor_force_n"][-env.frame_skip :]
            assert info["gait_support_force_r"] == pytest.approx(float(rows[:, 0].min()), rel=1e-12, abs=1e-12)
            assert info["gait_support_force_l"] == pytest.approx(float(rows[:, 1].min()), rel=1e-12, abs=1e-12)
            if terminated:
                break


def test_floor_source_feeds_the_bilateral_terms(make_env) -> None:
    # Bogus touch readings reach the bilateral terms under "touch" and not under "floor".
    for source in ("touch", "floor"):
        env = make_env(support_source=source, bilateral_support_weight=2.0, foot_contact_saturation_force=100.0)
        env.reset(seed=0)
        for _ in range(10):  # settle onto both feet
            env.step(np.zeros(env.model.nu, dtype=np.float32))
        env._foot_contact_forces = lambda: (0.0, 1000.0)
        env._invalidate_substep_aggregates()
        _, info = env._get_reward_info(np.zeros(env.model.nu, dtype=np.float32))
        assert info["r_foot_contact"] == 0.0  # the touch info key reads touch either way
        if source == "touch":
            assert info["bilateral_support_quality"] == 0.0
        else:
            assert info["gait_support_force_r"] > 100.0  # the real floor force under the standing foot
            assert info["bilateral_support_quality"] == 1.0


def test_statue_earns_no_gait_phase_and_never_flies(make_env) -> None:
    env = make_env(**GAIT_R1)
    env.reset(seed=0)
    zeros = np.zeros(env.model.nu, dtype=np.float32)
    steps = 0.0
    for step in range(300):
        _, _, terminated, _, info = env.step(zeros)
        assert not terminated
        assert info["reward_gait_phase"] == 0.0
        assert info["gait_flight"] == 0.0
        # Both feet carry load once the reset has settled: the second foot lands in the first step,
        # a touchdown that only starts its stride clock.
        assert info["gait_feet_down"] == 2.0 or step < 2
        steps += info["gait_phase_steps"]
    assert steps <= 2


def test_a_control_locked_hop_earns_almost_no_gait_phase_and_pays_flight(make_env) -> None:
    # The 20 Hz hop of test_trex_env's substep-aggregation witness, filter off: both feet unload
    # together, and the few landings that drift apart have 10-30 ms swings (A under 0.1).
    class UnfilteredTRexEnv(TRexEnv):
        action_filter_cutoff_hz = 0.0

    env = UnfilteredTRexEnv(reset_noise_scale=0.0, nosedive_termination_threshold=0.35, **GAIT_R1)
    try:
        env.reset(seed=0)
        zeros = np.zeros(env.model.nu, dtype=np.float32)
        for _ in range(200):
            env.step(zeros)
        steps = flights = phase = 0.0
        control_steps = 0
        for t in range(400):
            action = np.full(env.model.nu, 0.3 * np.sin(2 * np.pi * t / 5.0), dtype=np.float32)
            _, _, terminated, _, info = env.step(action)
            steps += info["gait_phase_steps"]
            flights += info["gait_flight"]
            phase += info["reward_gait_phase"]
            control_steps += 1
            if terminated:
                break
        assert steps >= 20 and flights >= 3
        # Under 1% of a walk, which earns the weight on every control step.
        assert 0.0 <= phase < 0.01 * GAIT_R1["gait_phase_weight"] * control_steps
    finally:
        env.close()


def test_leg_and_foot_contacts_are_read_from_the_floor(make_env) -> None:
    env = make_env(leg_contact_penalty_weight=1.0, foot_collision_penalty_weight=0.5)
    kit = env._gait_reward_kit
    env.reset(seed=0)
    assert kit._scan(False, True) == (None, False, False)
    # Lowered 0.2 m, the shanks reach the floor.
    env.data.qpos[2] -= 0.2
    mujoco.mj_forward(env.model, env.data)
    env._invalidate_substep_aggregates()
    assert kit._scan(False, True) == (None, False, True)
    _, info = env._get_reward_info(np.zeros(env.model.nu, dtype=np.float32))
    assert info["gait_leg_contact"] == 1.0 and info["reward_leg_contact"] == -1.0
    # Hip rolls 0.1 rad inward bring the feet together.
    env.reset(seed=0)
    model = env.model
    for name, sign in (("r_hip_roll", 1.0), ("l_hip_roll", -1.0)):
        env.data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]] += sign * 0.1
    mujoco.mj_forward(model, env.data)
    env._invalidate_substep_aggregates()
    assert kit._scan(False, True)[1] is True
    _, info = env._get_reward_info(np.zeros(env.model.nu, dtype=np.float32))
    assert info["gait_foot_collision"] == 1.0 and info["reward_foot_collision"] == -0.5


def test_leg_contact_terminates_only_when_asked(make_env) -> None:
    for terminate in (False, True):
        env = make_env(leg_contact_penalty_weight=1.0, terminate_on_leg_contact=terminate)
        env.reset(seed=0)
        env._gait_reward_kit._flags = lambda: (False, True)
        _, reward, terminated, _, info = env.step(np.zeros(env.model.nu, dtype=np.float32))
        assert info["reward_leg_contact"] == -1.0
        assert terminated is terminate
        assert info.get("termination_reason") == ("leg_contact" if terminate else None)
    # Off by default: the legacy env never asks.
    assert make_env()._gait_reward_kit.leg_contact_terminates() is False


def test_reset_clears_the_kit(make_env) -> None:
    env = make_env(**GAIT_R1)
    kit = env._gait_reward_kit
    _rollout(env, 50)
    assert kit._tracker is not None and kit._tracker._feet is not None and kit._step_tag > 0
    env.reset(seed=1)
    assert kit._tracker._feet is None and kit._tracker._samples == 0 and kit._step_tag == -1
    assert kit._min_floor is None and kit._phase_reward == 0.0


def test_identical_rollouts_give_identical_terms(make_env) -> None:
    first = _rollout(make_env(**ALL_TERMS), 150, seed=11)
    second = _rollout(make_env(**ALL_TERMS), 150, seed=11)
    for (*_, reward, _, info), (*_, reward2, _, info2) in zip(first, second, strict=True):
        assert reward == reward2
        assert {key: info[key] for key in KIT_INFO_KEYS} == {key: info2[key] for key in KIT_INFO_KEYS}


def test_kit_refuses_what_it_cannot_measure(make_env) -> None:
    with pytest.raises(ValueError, match="flight_min_feet"):
        make_env(flight_min_feet=3)
    with pytest.raises(ValueError, match="support_source"):
        make_env(support_source="touchdown")


# ---------------------------------------------------------------------------
# the certified r13 stance stays an ancestor
# ---------------------------------------------------------------------------

#: The plant identity the r13 runs recorded (20260914_123816 at 35dd44c; its seed-44 sibling
#: 20260920_010912's stance was reused under it by 20260925_033501 at 9d729e8).
R13_PLANT = {
    "action_dim": 15,
    "model_path": "environments/trex/assets/trex.xml",
    "nq": 28,
    "nu": 15,
    "nv": 27,
    "observation_dim": 64,
    "physics_revision": 7,
    "physics_sha256": "sha256:72c662858639193e24a503b98e06d84737150ae6888a50a3620c2c99804cb97b",
    "policy_interface_revision": 13,
    "policy_interface_sha256": "sha256:0b43de19860b3b6e8ea7d010d8b7069b265b14040d57ece7fffab49a9cb51252",
    "schema": "mesozoic.plant-identity/v1",
    "source_closure_sha256": "sha256:e3919447fc76fbcd6396284afa028ad2831797461d03bbe6c9b52a5fc1c50753",
    "species": "trex",
    "visual_revision": 4,
    "visual_sha256": "sha256:7042dcd597fbd0ce25489164a7f132d51e1c2adbc050729c9866d7bde42e35a4",
}
#: The task digests the r13 verdicts recorded (docs/NEXT_STEPS.md §2), and the recovery and
#: behavior stages' pre-kit digests.  The locomotion digest is the task of the certified hops
#: (20260914_123816, 20260925_033501), which gait-r1 replaces.
PRE_KIT_TASKS = {
    "stance": "sha256:82528a2ecfefd57172e06ace0a5d90a4b1b08ebe9c9f45ff6405cb78fdf60140",
    "recovery": "sha256:2c6f4a47154ec9f6683adc4509b4de260fb733cf78b6f586e04ff18178739c57",
    "locomotion": "sha256:3138319203a4674382baa98916176aea760d5ab9db3411f672583a489c4f99c2",
    "behavior": "sha256:5acd008d5be18b59e58c2a639c482604d4c43827d7f3debe073942c5fc9a5688",
}


def _stage_tasks(overrides: dict | None = None) -> dict[str, dict]:
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.stage_manifest import load_stage_manifest
    from environments.shared.task_fingerprint import derive_stage_task_fingerprint

    identity = current_plant_identity("trex").to_dict()
    tasks = {}
    for entry in load_stage_manifest("trex").stages:
        env_kwargs = {**load_stage_config("trex", entry.reference).get("env_kwargs", {}), **(overrides or {})}
        tasks[entry.id] = derive_stage_task_fingerprint(
            species="trex",
            stage=entry.reference,
            backend="stable-baselines3",
            env_kwargs=env_kwargs,
            plant_identity=identity,
        )
    return tasks


#: The trex locomotion task under gait-r1 (configs/digest_snapshot.generated.txt).
GAIT_R1_LOCOMOTION_TASK = "sha256:bb2ed29166edd4dfab7656a7a7a8b4e71926e925b5c614eb6fff57fb247df52f"


def test_every_trex_stage_but_locomotion_keeps_its_pre_kit_task_digest() -> None:
    # The stance keeps the digest its 20260914_123816 verdict recorded, so that run stays the trunk
    # of a gait-r1 locomotion run; locomotion is the revision, and only it records kit knobs.
    tasks = _stage_tasks()
    assert {stage: task["task_sha256"] for stage, task in tasks.items()} == {
        **PRE_KIT_TASKS,
        "locomotion": GAIT_R1_LOCOMOTION_TASK,
    }
    for stage, task in tasks.items():
        kit_keys = set(GAIT_REWARD_KIT_LEGACY) & set(task["env"])
        assert kit_keys == (set(GAIT_R1) if stage == "locomotion" else set()), stage


def test_the_locomotion_toml_is_the_gait_r1_revision() -> None:
    from environments.shared.config import load_stage_config

    config = load_stage_config("trex", "locomotion")
    env, curriculum = config["env_kwargs"], config["curriculum_kwargs"]
    assert {key: env[key] for key in GAIT_R1_TASK} == GAIT_R1_TASK
    # Inert knobs stay unset: no alive conditioning, no leg or foot-on-foot terms, the default step length.
    for key in (
        "support_conditioned_alive_fraction",
        "gait_phase_step_over_leg",
        "flight_min_feet",
        "foot_collision_penalty_weight",
        "leg_contact_penalty_weight",
        "terminate_on_leg_contact",
    ):
        assert key not in env, key
    # The gate is not enforced yet; its length rail and the statue reference follow the 2000-step horizon.
    assert curriculum["gate_kind"] == "reward_and_length/v1"
    assert curriculum["min_avg_episode_length"] == 1500
    assert curriculum["collapse_peak_floor_reference"] == pytest.approx(2186.8)
    assert curriculum["collapse_peak_floor_fraction"] == 0.45


def test_gait_r1_knobs_are_a_task_revision() -> None:
    # Set on any trex stage, the kit knobs move its task digest and are recorded as set.
    tasks = _stage_tasks(GAIT_R1)
    for stage, task in tasks.items():
        assert task["task_sha256"] != PRE_KIT_TASKS[stage]
        assert {key: task["env"][key] for key in GAIT_R1} == GAIT_R1
        # The unset knobs stay carved out: their legacy values are pinned.
        assert "flight_min_feet" not in task["env"] and "gait_phase_step_over_leg" not in task["env"]


def test_the_r13_stance_checkpoint_stays_loadable_under_gait_r1_knobs(make_env) -> None:
    from environments.shared.plant_contract import (
        PlantIdentity,
        current_plant_identity,
        validate_environment_plant,
        validate_recorded_identity,
    )

    current = current_plant_identity("trex")
    # Ancestor rule 6: the recorded identity validates against the current plant, no legacy allowance ...
    validate_recorded_identity(R13_PLANT, current, artifact="20260914_123816 stance")
    assert current.compatibility_errors(PlantIdentity.from_mapping(R13_PLANT)) == []
    # ... and the env a gait-r1 run builds, every kit knob set, is that plant (the policy-interface
    # probe and fingerprints of _get_obs, _scale_action and reset included).
    validate_environment_plant(make_env(**ALL_TERMS, terminate_on_leg_contact=True), current, artifact="gait-r1")


def test_the_kit_reads_the_terrain_the_behavior_env_swaps_in() -> None:
    # A trex terrain behavior with the gait-r1 knobs: the floor force follows the env's model swap
    # between the plane and the heightfield, so a statue is supported on both.
    from environments.shared.behavior_env import get_behavior_env_class
    from environments.shared.direction_commands import DirectionCommandConfig
    from environments.shared.terrain import TerrainConfig
    from environments.shared.terrain_sampling import TERRAIN_FAMILIES, TerrainSamplerConfig

    env = get_behavior_env_class("trex")(
        commands=DirectionCommandConfig(cruise_speed=1.05, speed_scale=1.5),
        terrain=TerrainConfig(extent=8.0, nrow=81, ncol=81, apron_radius=4.0, template="bumps"),
        terrain_sampler=TerrainSamplerConfig(**{**dict.fromkeys(TERRAIN_FAMILIES, 0), "flat": 1, "bumps": 1}),
        run_seed=0,
        **GAIT_R1,
    )
    try:
        surfaces = set()
        for episode in range(4):
            env.reset(seed=episode)
            for _ in range(60):
                _, _, terminated, _, info = env.step(np.zeros(env.model.nu))
                assert not terminated
            surfaces.add(int(env.model.geom_type[env._static_floor_geoms()][0]))
            assert info["gait_feet_down"] == 2.0
            assert min(info["gait_support_force_r"], info["gait_support_force_l"]) > 0.25 * 840.0
            assert info["reward_gait_phase"] == 0.0
        assert surfaces == {int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD)}
    finally:
        env.close()
