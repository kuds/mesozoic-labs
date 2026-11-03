"""The warm-start drop of the critic's episode phase (critic_phase_ppo.drop_critic_phase, train_base's wiring): a
phase checkpoint warm-started across a node's edge into a stage without ``[ppo] critic_phase`` loses the critic's
three phase inputs and becomes stock PPO; the actor is untouched.  Every other mismatch is still refused.  Also the
names the phase-critic runs' archives depend on, and an archive written by the screen branch's code (1dfe2327).
Needs stable-baselines3 (the SB3 job)."""

from __future__ import annotations

import copy
import json
import random
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.buffers import RolloutBuffer  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from environments.shared import critic_phase as CP  # noqa: E402
from environments.shared import critic_phase_ppo as CPP  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402

REPOSITORY = Path(__file__).resolve().parents[3]
TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
FIXTURE = Path(__file__).parent / "fixtures" / "critic_phase"
SETTLE, HORIZON, GAMMA = 5, 12, 0.98
SPEC = CP.CriticPhaseSpec(settle_steps=SETTLE, gamma=GAMMA)
LAYER = "mlp_extractor.value_net.0.weight"
ACTOR = ("mlp_extractor.policy_net.", "action_net.", "log_std")


class PhaseToyEnv(gym.Env):
    """The T. rex interface's shapes (63 observations, 15 actions) with a BaseDinoEnv-style step counter."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (63,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (15,), np.float32)

    def __init__(self, settle: int = SETTLE):
        self.stance_width_settle_steps = settle
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self.np_random.normal(0.3, 1.5, 63).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = self.np_random.normal(0.3, 1.5, 63).astype(np.float32)
        gate = 1.0 if self._step_count > self.stance_width_settle_steps else 0.0
        reward = float(gate - 0.1 * np.sum(np.square(action)) + obs[3] * action[4])
        return obs, reward, False, self._step_count >= HORIZON, {"step": self._step_count}


def _venv(seed: int, *, phase: bool = False, gamma: float = 0.995) -> VecNormalize:
    envs = DummyVecEnv([lambda: Monitor(PhaseToyEnv()) for _ in range(2)])
    venv = CPP.CriticPhaseVecNormalize(envs, critic_phase_gamma=gamma, gamma=gamma) if phase else VecNormalize(envs)
    venv.seed(seed)
    return venv


_PARENT_KWARGS: dict[str, Any] = {
    "n_steps": 32,
    "batch_size": 32,
    "n_epochs": 2,
    "learning_rate": 3e-4,
    "gamma": GAMMA,
    "target_kl": None,
    "device": "cpu",
}
#: The child: another discount, as locomotion's 0.995 after the stance's 0.98.
_CHILD_KWARGS: dict[str, Any] = {**_PARENT_KWARGS, "gamma": 0.995, "learning_rate": 5e-5}


@pytest.fixture(scope="module")
def phase_zip(tmp_path_factory) -> Path:
    """A trained phase checkpoint (Adam moments exist), saved as train_base saves one."""
    venv = _venv(3, phase=True, gamma=GAMMA)
    model = PPO(
        env=venv,
        seed=3,
        **CPP.new_model_kwargs(venv.critic_phase, {"net_arch": [32, 32], "log_std_init": -1.5}),
        **_PARENT_KWARGS,
    )
    setattr(model, CP.CHECKPOINT_ATTRIBUTE, venv.critic_phase.record())
    model.learn(total_timesteps=64)
    path: Path = tmp_path_factory.mktemp("phase") / "parent.zip"
    model.save(path)
    return path


def _load(path: Path, seed: int = 21) -> PPO:
    return PPO.load(str(path), env=_venv(seed), seed=seed, **_CHILD_KWARGS)


def _state(policy) -> dict[str, Any]:
    return {key: value.detach().clone() for key, value in policy.state_dict().items()}


def _adam(model) -> dict:
    state = model.policy.optimizer.state_dict()["state"]
    return {
        key: {name: torch.as_tensor(value).clone() for name, value in entry.items()} for key, entry in state.items()
    }


def _probe(rows: int = 16) -> Any:
    return torch.as_tensor(np.random.default_rng(5).normal(size=(rows, 63)), dtype=torch.float32)


def _cut_archive(source: Path, target: Path) -> Path:
    """An independent stock archive of *source* with the phase columns cut out (never unpickles the phase classes)."""
    loaded: Any = load_from_zip_file(
        str(source),
        device="cpu",
        custom_objects={"policy_class": ActorCriticPolicy, "rollout_buffer_class": RolloutBuffer},
    )
    data, params, variables = loaded
    data["policy_kwargs"] = {k: v for k, v in data["policy_kwargs"].items() if k not in CP.POLICY_KWARGS}
    data["rollout_buffer_kwargs"] = {}
    data.pop(CP.CHECKPOINT_ATTRIBUTE)
    data["_last_obs"] = data["_last_original_obs"] = None
    index = list(params["policy"]).index(LAYER)
    params["policy"][LAYER] = params["policy"][LAYER][:, :63].clone()
    for key in ("exp_avg", "exp_avg_sq"):
        params["policy.optimizer"]["state"][index][key] = params["policy.optimizer"]["state"][index][key][
            :, :63
        ].clone()
    save_to_zip_file(str(target), data=data, params=params, pytorch_variables=variables)
    return target


def _create(path, *, mode="initialize_next_stage", crosses_edge=True, spec=None, venv=None, seed=21, **kwargs):
    from environments.shared.policy_loading import _ensure_sb3
    from environments.shared.train_base import _create_or_load_model

    return _create_or_load_model(
        _ensure_sb3(),
        "ppo",
        {**_CHILD_KWARGS, "seed": seed, "policy_kwargs": {"net_arch": [8]}},
        venv if venv is not None else _venv(seed),
        None if path is None else str(path),
        task_load_mode=mode,
        critic_phase=spec,
        crosses_edge=crosses_edge,
        **kwargs,
    )


# ---------------------------------------------------------------------------------------------------- the drop itself
def test_the_actor_is_bitwise_unchanged(phase_zip):  # D1
    model = _load(phase_zip)
    assert type(model.policy) is CPP.CriticPhaseActorCriticPolicy
    before = _state(model.policy)
    probe = _probe()
    actions = torch.as_tensor(np.random.default_rng(6).uniform(-1, 1, (16, 15)), dtype=torch.float32)
    with torch.no_grad():
        distribution = model.policy.get_distribution(probe)
        mode, log_prob, entropy = distribution.mode(), distribution.log_prob(actions), distribution.entropy()
    CPP.drop_critic_phase(model)
    after = _state(model.policy)
    for name in before:
        if name.startswith(ACTOR):
            assert torch.equal(before[name], after[name]), name
    with torch.no_grad():
        distribution = model.policy.get_distribution(probe)
        assert torch.equal(distribution.mode(), mode)
        assert torch.equal(distribution.log_prob(actions), log_prob)
        assert torch.equal(distribution.entropy(), entropy)
    np.testing.assert_array_equal(model.predict(probe.numpy(), deterministic=True)[0], mode.numpy().clip(-1, 1))


def test_the_critic_keeps_every_weight_but_the_phase_columns(phase_zip):  # D2
    model = _load(phase_zip)
    before = _state(model.policy)
    probe = _probe()
    with torch.no_grad():
        zero_phase = model.policy.predict_values(torch.cat([probe, torch.zeros(16, 3)], dim=1))
    CPP.drop_critic_phase(model)
    after = _state(model.policy)
    assert before.keys() == after.keys()
    assert tuple(before[LAYER].shape) == (32, 66) and tuple(after[LAYER].shape) == (32, 63)
    assert torch.equal(after[LAYER], before[LAYER][:, :63])
    for name in before:
        if name != LAYER:
            assert torch.equal(before[name], after[name]), name
    with torch.no_grad():
        # The phase-free critic is the phase critic at phase (0, 0, 0): equal up to the shorter matmul's rounding.
        torch.testing.assert_close(model.policy.predict_values(probe), zero_phase, rtol=0, atol=1e-6)


def test_the_adam_moments_lose_the_same_columns_and_nothing_else(phase_zip):  # D3
    model = _load(phase_zip)
    before = _adam(model)
    index = [name for name, _ in model.policy.named_parameters()].index(LAYER)
    CPP.drop_critic_phase(model)
    after = _adam(model)
    assert before.keys() == after.keys()
    for key in before:
        assert before[key].keys() == after[key].keys()
        for name in before[key]:
            expected = before[key][name][:, :63] if key == index and name in CPP.DROP_MOMENTS else before[key][name]
            assert torch.equal(after[key][name], expected), (key, name)
    held = {id(p) for group in model.policy.optimizer.param_groups for p in group["params"]}
    assert held == {id(p) for p in model.policy.parameters()}


def test_a_drop_warm_start_trains_bitwise_as_a_stock_warm_start_from_a_cut_archive(phase_zip, tmp_path):  # D4
    cut = _cut_archive(phase_zip, tmp_path / "cut.zip")
    results = []
    # Each loads (re-seeding the global generators with the stage's seed) right before it learns, as a run does.
    for path in (phase_zip, cut):
        model = _create(path)
        assert type(model.policy) is ActorCriticPolicy and type(model.rollout_buffer) is RolloutBuffer
        model.learn(total_timesteps=64, reset_num_timesteps=True)
        buffer = model.rollout_buffer
        results.append(
            (
                _state(model.policy),
                _adam(model),
                {n: np.array(getattr(buffer, n)) for n in ("observations", "actions", "rewards", "values", "returns")},
            )
        )
    (a, a_adam, a_buffer), (b, b_adam, b_buffer) = results
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert all(torch.equal(a_adam[k][n], b_adam[k][n]) for k in a_adam for n in a_adam[k])
    assert all(np.array_equal(a_buffer[n], b_buffer[n]) for n in a_buffer)


def test_the_drop_draws_no_random_number(phase_zip):  # D5
    model = _load(phase_zip)
    torch_state, numpy_state, python_state = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
    CPP.drop_critic_phase(model)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]
    assert random.getstate() == python_state


def test_the_child_is_stock_ppo_and_its_archive_loads_without_the_phase_module(phase_zip, tmp_path):  # D6
    model = _load(phase_zip)
    report = CPP.drop_critic_phase(model)
    assert type(model.policy) is ActorCriticPolicy and model.policy_class is ActorCriticPolicy
    assert type(model.rollout_buffer) is RolloutBuffer and model.rollout_buffer.obs_shape == (63,)
    assert model.rollout_buffer_class is RolloutBuffer and model.rollout_buffer_kwargs == {}
    assert not set(model.policy_kwargs) & set(CP.POLICY_KWARGS) and model.policy_kwargs["net_arch"] == [32, 32]
    assert CP.CHECKPOINT_ATTRIBUTE not in vars(model) and model._last_obs is None
    assert report["schema"] == CPP.DROP_SCHEMA and report["source"] == SPEC.record()
    assert report["columns"] == [63, 64, 65] and report["optimizer_moments"] == ["exp_avg", "exp_avg_sq"]
    assert len(report["column_l2"]) == 3 and all(value > 0 for value in report["column_l2"])
    json.dumps(report)  # it goes into the checkpoint's JSON lineage
    model.learn(total_timesteps=64)
    model.save(tmp_path / "child.zip")
    assert b"critic_phase" not in zipfile.ZipFile(tmp_path / "child.zip").read("data")
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPOSITORY)!r})\n"
        "sys.modules['environments.shared.critic_phase_ppo'] = None\n"
        "from environments.shared.policy_loading import load_sb3_model\n"
        f"model = load_sb3_model({str(tmp_path / 'child.zip')!r}, algorithm='ppo', device='cpu')\n"
        "print(type(model.policy).__name__, model.policy.mlp_extractor.value_net[0].in_features)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, timeout=600, check=False
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.split() == ["ActorCriticPolicy", "63"]


# ---------------------------------------------------------------------------------------------------- what is refused
def test_a_load_that_crosses_no_edge_or_mismatches_is_still_refused(phase_zip, tmp_path):  # D8
    other = CP.CriticPhaseSpec(SETTLE + 1, GAMMA)
    cut = _cut_archive(phase_zip, tmp_path / "stock.zip")
    # A resume into a stage without the option, and initialize_next_stage without an edge (a root re-initialised
    # from itself, or a direct caller's default).
    with pytest.raises(CP.CriticPhaseError, match="declares no"):
        _create(phase_zip, mode="resume_same_stage", crosses_edge=False)
    with pytest.raises(CP.CriticPhaseError, match="declares no"):
        _create(phase_zip, crosses_edge=False)
    # crosses_edge on a resume is not an edge.
    with pytest.raises(CP.CriticPhaseError, match="declares no"):
        _create(phase_zip, mode="resume_same_stage", crosses_edge=True)
    for mode in ("resume_same_stage", "initialize_next_stage"):
        # Another settle step, then another discount: refused, the edge or not.
        for spec, gamma in ((other, GAMMA), (CP.CriticPhaseSpec(SETTLE, 0.99), 0.99)):
            venv = CPP.CriticPhaseVecNormalize(
                DummyVecEnv([lambda s=spec: Monitor(PhaseToyEnv(settle=s.settle_steps))]),
                critic_phase_gamma=gamma,
            )
            with pytest.raises(CP.CriticPhaseError, match="reads"):
                _create(phase_zip, mode=mode, spec=spec, venv=venv)
        # A stock parent into a phase stage: no zero-append at a warm start.
        with pytest.raises(CP.CriticPhaseError, match="reads no phase"):
            _create(cut, mode=mode, spec=SPEC, venv=_venv(3, phase=True, gamma=GAMMA))
    # The same spec across the edge loads as the phase critic (the phase continues).
    same = _create(phase_zip, spec=SPEC, venv=_venv(3, phase=True, gamma=GAMMA))
    assert type(same.policy) is CPP.CriticPhaseActorCriticPolicy


def _edited(source: Path, target: Path, **changes: Any) -> Path:
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    data = json.loads(members["data"])
    data.update(changes)
    members["data"] = json.dumps(data).encode()
    with zipfile.ZipFile(target, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return target


def test_a_malformed_or_inconsistent_archive_is_refused(phase_zip, tmp_path):  # D8
    from environments.shared.train_base import _critic_phase_drop_record

    stock = _cut_archive(phase_zip, tmp_path / "stock.zip")
    malformed = _edited(phase_zip, tmp_path / "malformed.zip", **{CP.CHECKPOINT_ATTRIBUTE: {"schema": "x"}})
    hand_edited = _edited(stock, tmp_path / "hand.zip", **{CP.CHECKPOINT_ATTRIBUTE: SPEC.record()})
    with pytest.raises(CP.CriticPhaseError, match="not a mesozoic.critic-phase/v1 record"):
        _create(malformed)
    with pytest.raises(CP.CriticPhaseError, match="not a mesozoic.critic-phase/v1 record"):
        _critic_phase_drop_record(
            {},
            stage=2,
            algorithm="ppo",
            task_load_mode="initialize_next_stage",
            parent_id="stance",
            load_path=str(malformed),
        )
    # A record on a stock policy: the archive says one thing and its classes another.
    with pytest.raises(CP.CriticPhaseError, match="cannot drop the critic phase"):
        _create(hand_edited)
    # A moment Adam does not keep (amsgrad's), shaped like the wide weight.
    model = _load(phase_zip)
    wide = model.policy.mlp_extractor.value_net[0].weight
    model.policy.optimizer.state[wide]["max_exp_avg_sq"] = torch.zeros_like(wide)
    with pytest.raises(CP.CriticPhaseError, match="only Adam's"):
        CPP.drop_critic_phase(model)
    assert type(model.policy) is CPP.CriticPhaseActorCriticPolicy  # refused before anything changed


@pytest.mark.parametrize(
    "case, match",
    [
        ({"task_load_mode": "resume_same_stage"}, "crosses none"),
        ({"parent_id": None}, "crosses none"),
        ({"algorithm": "sac"}, "a PPO option"),
    ],
    ids=["resume", "root", "sac"],
)
def test_the_pre_flight_refuses_what_the_drop_cannot_serve(case, match, phase_zip):  # D8
    from environments.shared.train_base import _critic_phase_drop_record

    kwargs: dict[str, Any] = {
        "stage": 2,
        "algorithm": "ppo",
        "task_load_mode": "initialize_next_stage",
        "parent_id": "stance",
        "load_path": str(phase_zip),
    }
    assert _critic_phase_drop_record({}, **kwargs) == SPEC.record()
    with pytest.raises(CP.CriticPhaseError, match=match):
        _critic_phase_drop_record({}, **{**kwargs, **case})
    # A stage that declares the phase is checked at load time instead; a stock archive or none at all reads None.
    assert _critic_phase_drop_record({"ppo_kwargs": {"critic_phase": True}}, **kwargs) is None
    assert _critic_phase_drop_record({}, **{**kwargs, "load_path": None}) is None
    assert _critic_phase_drop_record({}, **{**kwargs, "load_path": str(phase_zip.parent / "absent.zip")}) is None


def test_a_refused_load_leaves_no_stage_directory(phase_zip, tmp_path):  # D8
    from environments.shared import train_base
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config

    with pytest.raises(CP.CriticPhaseError, match="crosses none"):
        train_base.train(
            get_species_config(TYRANNOSAURUS_REX),
            load_all_stages(TYRANNOSAURUS_REX),
            1,
            64,
            n_envs=1,
            load_path=str(phase_zip),
            task_load_mode="resume_same_stage",
            output_dir=str(tmp_path / "stage"),
            use_tensorboard=False,
            report_metrics=False,
            save_on_interrupt=False,
        )
    assert not (tmp_path / "stage").exists()


def test_the_drop_comes_before_the_command_column_neutralisation(phase_zip):  # D10
    from environments.shared.policy_loading import PolicyLoadError, neutralize_command_columns

    undropped = _load(phase_zip)
    with pytest.raises(PolicyLoadError, match="Unsupported policy input layer: mlp_extractor.value_net.0"):
        neutralize_command_columns(undropped, observation_dim=63)
    dropped = _load(phase_zip)
    CPP.drop_critic_phase(dropped)
    assert neutralize_command_columns(dropped, observation_dim=63)


@pytest.mark.parametrize("mode", ["resume_same_stage", "initialize_next_stage", "other"])
@pytest.mark.parametrize("parent_id", [None, "stance"])
@pytest.mark.parametrize("load_path", [None, "", "parent.zip"])
def test_the_edge_rule_is_the_stage_entry_shaping_s(mode, parent_id, load_path):  # B1c
    from environments.shared.train_base import _load_crosses_edge, _stage_entry_shaping_callbacks

    config = {"curriculum_kwargs": {}, "env_kwargs": {"forward_vel_weight": 0.0}}
    shaping = _stage_entry_shaping_callbacks(config, task_load_mode=mode, parent_id=parent_id, load_path=load_path)
    assert _load_crosses_edge(mode, parent_id, load_path) is bool(shaping)


# ---------------------------------------------------------------------------------------------------- the trainer
_TRAIN: dict[str, Any] = {
    "n_envs": 1,
    "seed": 42,
    "eval_freq": 10**9,
    "verbose": 0,
    "use_tensorboard": False,
    "report_metrics": False,
    "save_on_interrupt": False,
}


def _trex_configs() -> dict:
    from environments.shared.config import load_all_stages

    configs = copy.deepcopy(load_all_stages(TYRANNOSAURUS_REX))
    for stage in (1, 2):
        configs[stage]["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=1)
    configs[1]["ppo_kwargs"]["critic_phase"] = True
    return configs


@pytest.fixture(scope="module")
def trex_phase_parent(tmp_path_factory) -> Path:
    """train_base.train on the real T. rex stance with the phase: one 64-step update, its final checkpoint."""
    from environments.shared import train_base
    from environments.shared.species_registry import get_species_config

    out = tmp_path_factory.mktemp("trex") / "stance"
    train_base.train(
        get_species_config(TYRANNOSAURUS_REX), _trex_configs(), 1, 64, save_freq=10**9, output_dir=str(out), **_TRAIN
    )
    parent: Path = out / "models" / "stage1_final.zip"
    return parent


def test_the_trainer_drops_records_and_resumes_the_child_stock(trex_phase_parent, tmp_path, monkeypatch, capsys):
    """train_base.train on the real T. rex stages: a stance with the phase, then locomotion warm-started from it
    (the node's edge; locomotion declares no phase), then a same-stage resume of locomotion's periodic checkpoint
    (D7, D9)."""
    from environments.shared import train_base
    from environments.shared.species_registry import get_species_config

    species = get_species_config(TYRANNOSAURUS_REX)
    configs = _trex_configs()
    common = _TRAIN
    parent = trex_phase_parent
    parent_policy = load_from_zip_file(str(parent), device="cpu", load_data=False)[1]["policy"]
    record = json.loads(zipfile.ZipFile(parent).read("data"))[CP.CHECKPOINT_ATTRIBUTE]

    loaded: list[Any] = []
    original = train_base._create_or_load_model

    def spy(*args, **kwargs):
        model = original(*args, **kwargs)
        loaded.append((kwargs.get("crosses_edge"), _state(model.policy), type(model.policy)))
        return model

    monkeypatch.setattr(train_base, "_create_or_load_model", spy)
    capsys.readouterr()
    child = train_base.train(
        species,
        configs,
        2,
        64,
        save_freq=64,
        output_dir=str(tmp_path / "locomotion"),
        load_path=str(parent),
        task_load_mode="initialize_next_stage",
        **common,
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Warm start:")]
    assert len(lines) == 1 and lines[0].startswith("Warm start: dropped the critic's 3 episode-phase inputs")
    crosses_edge, state, policy_class = loaded[-1]
    assert crosses_edge is True and policy_class is ActorCriticPolicy
    # The actor right after the load is the parent's, bitwise (the T. rex locomotion task records no live command,
    # so no command column is neutralised here; the order against that step is the test above); the critic keeps
    # every weight but the phase columns.
    actor = [name for name in state if name.startswith(ACTOR)]
    assert actor and all(torch.equal(state[name], parent_policy[name]) for name in actor)
    assert torch.equal(state[LAYER], parent_policy[LAYER][:, :63]) and tuple(parent_policy[LAYER].shape) == (512, 66)
    assert (
        type(child) is PPO and type(child.policy) is ActorCriticPolicy and type(child.rollout_buffer) is RolloutBuffer
    )

    run = json.loads((tmp_path / "locomotion" / "stage_config.json").read_text())["run"]
    assert run["critic_phase_dropped"] == record and run["load_mode"] == "initialize_next_stage"
    with zipfile.ZipFile(tmp_path / "locomotion" / "models" / "stage2_final.zip") as archive:
        data = json.loads(archive.read("data"))
    assert CP.CHECKPOINT_ATTRIBUTE not in data and "critic_phase_ppo" not in json.dumps(data)
    lineage = data["mesozoic_task_lineage"]
    assert lineage["dropped_critic_phase"]["source"] == record
    assert len(lineage["dropped_critic_phase"]["column_l2"]) == 3 and lineage["mode"] == "initialize_next_stage"

    # The notebook's RESUME cell: the child's own checkpoint is stock, so nothing is dropped again.
    capsys.readouterr()
    resumed = train_base.train(
        species,
        configs,
        2,
        64,
        save_freq=10**9,
        output_dir=str(tmp_path / "locomotion"),
        load_path=str(tmp_path / "locomotion" / "models" / "stage2_64_steps.zip"),
        task_load_mode="resume_same_stage",
        **common,
    )
    assert not [line for line in capsys.readouterr().out.splitlines() if line.startswith("Warm start:")]
    assert loaded[-1][0] is False and type(resumed.policy) is ActorCriticPolicy and resumed.num_timesteps == 128
    run = json.loads((tmp_path / "locomotion" / "stage_config.json").read_text())["run"]
    assert run["critic_phase_dropped"] == record and run["load_mode"] == "initialize_next_stage"
    assert run["resume_load_path"].endswith("stage2_64_steps.zip")


def test_the_drop_and_the_log_std_reset_apply_at_the_same_warm_start(trex_phase_parent, tmp_path, monkeypatch, capsys):
    """[curriculum] warm_start_log_std on the child as well (D11): the drop in the load, then the reset; both are
    printed and recorded, and every actor tensor but log_std is the parent's, bitwise."""
    from environments.shared import train_base
    from environments.shared.curriculum.gate_schema import WARM_START_LOG_STD_KEY
    from environments.shared.species_registry import get_species_config

    configs = _trex_configs()
    configs[2]["curriculum_kwargs"][WARM_START_LOG_STD_KEY] = -1.5
    parent_policy = load_from_zip_file(str(trex_phase_parent), device="cpu", load_data=False)[1]["policy"]
    after_reset: dict[str, Any] = {}
    original = train_base._apply_warm_start_log_std

    def spy(model, value):
        original(model, value)
        after_reset.update(_state(model.policy))

    monkeypatch.setattr(train_base, "_apply_warm_start_log_std", spy)
    capsys.readouterr()
    train_base.train(
        get_species_config(TYRANNOSAURUS_REX),
        configs,
        2,
        64,
        save_freq=10**9,
        output_dir=str(tmp_path / "locomotion"),
        load_path=str(trex_phase_parent),
        task_load_mode="initialize_next_stage",
        **_TRAIN,
    )
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("Warm start:")]
    assert len(lines) == 2
    assert lines[0].startswith("Warm start: dropped the critic's 3 episode-phase inputs")
    assert lines[1].startswith("Warm start: policy log_std set to -1.5 (std 0.223) on all 15 actuators")
    assert torch.equal(after_reset["log_std"], torch.full((15,), -1.5))
    for name, value in after_reset.items():
        if name.startswith(ACTOR) and name != "log_std":
            assert torch.equal(value, parent_policy[name]), name
    run = json.loads((tmp_path / "locomotion" / "stage_config.json").read_text())["run"]
    assert run[WARM_START_LOG_STD_KEY] == -1.5 and run["critic_phase_dropped"]["settle_steps"] == 200


# ---------------------------------------------------------------------------------------------------- compatibility
def test_the_names_the_phase_critic_archives_depend_on_are_unchanged():  # L1
    # As the phase-critic stance runs recorded them (written by the screen branch's code, 1dfe2327).
    assert CPP.__name__ == "environments.shared.critic_phase_ppo"
    for cls, name in (
        (CPP.CriticPhaseActorCriticPolicy, "CriticPhaseActorCriticPolicy"),
        (CPP.CriticPhaseRolloutBuffer, "CriticPhaseRolloutBuffer"),
        (CPP.CriticPhaseMlpExtractor, "CriticPhaseMlpExtractor"),
        (CPP.CriticPhaseVecNormalize, "CriticPhaseVecNormalize"),
    ):
        assert (cls.__module__, cls.__qualname__) == ("environments.shared.critic_phase_ppo", name)
    assert CPP.CriticPhaseActorCriticPolicy.critic_phase_width == 3
    assert CPP.new_model_kwargs(SPEC, {})["rollout_buffer_kwargs"] == {"critic_phase_width": 3}
    assert set(CPP.new_model_kwargs(SPEC, {})["policy_kwargs"]) == {"critic_phase_settle_steps", "critic_phase_gamma"}


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    record: dict[str, Any] = json.loads((FIXTURE / "manifest.json").read_text())
    return record


def test_an_archive_the_screen_branch_wrote_loads_on_the_deploy_path(manifest):  # L2
    from environments.shared.policy_loading import inspect_sb3_archive, load_sb3_checkpoint

    archive = FIXTURE / manifest["archive"]
    assert manifest["source_commit"].startswith("1dfe2327")
    assert not inspect_sb3_archive(archive).bytecode_members  # the classes are pickled by reference
    data = json.loads(zipfile.ZipFile(archive).read("data"))
    assert data[CP.CHECKPOINT_ATTRIBUTE] == manifest["record"] == SPEC.record()
    assert data["policy_class"]["__module__"] == "environments.shared.critic_phase_ppo"
    model, normalizer, _ = load_sb3_checkpoint(str(archive), str(FIXTURE / manifest["sidecar"]), PhaseToyEnv)
    assert type(model.policy) is CPP.CriticPhaseActorCriticPolicy and model.policy.critic_phase == SPEC
    assert type(normalizer) is VecNormalize and normalizer.observation_space.shape == (63,)
    observations = np.asarray(manifest["observations"], dtype=np.float32)
    actions = model.predict(observations, deterministic=True)[0]
    np.testing.assert_allclose(actions, np.asarray(manifest["actions"]), rtol=0, atol=1e-6)


def test_an_archive_the_screen_branch_wrote_drops_across_an_edge_and_resumes_under_its_phase(manifest):  # L2
    archive = FIXTURE / manifest["archive"]
    dropped = _create(archive)
    assert type(dropped.policy) is ActorCriticPolicy and CP.CHECKPOINT_ATTRIBUTE not in vars(dropped)
    observations = np.asarray(manifest["observations"], dtype=np.float32)
    np.testing.assert_allclose(
        dropped.predict(observations, deterministic=True)[0], np.asarray(manifest["actions"]), rtol=0, atol=1e-6
    )
    resumed = _create(
        archive, mode="resume_same_stage", crosses_edge=False, spec=SPEC, venv=_venv(3, phase=True, gamma=GAMMA)
    )
    assert (
        type(resumed.policy) is CPP.CriticPhaseActorCriticPolicy and resumed.num_timesteps == manifest["num_timesteps"]
    )
    settled = np.concatenate([observations, SPEC.matrix([SETTLE] * len(observations)).astype(np.float32)], axis=1)
    with torch.no_grad():
        values = resumed.policy.predict_values(torch.as_tensor(settled)).numpy().ravel()
    np.testing.assert_allclose(values, np.asarray(manifest["settled_values"]), rtol=0, atol=1e-5)
