"""The critic's episode phase for SB3 PPO (``[ppo] critic_phase = true``): the four pieces, see
:mod:`environments.shared.critic_phase` for what the phase is and why.

1. :class:`CriticPhaseVecNormalize`, the TRAINING env's ``VecNormalize``: stock normalisation of the env's
   observation, then the phase of each env's own step counter appended as three raw columns, on ``reset()``, on
   ``step()`` and on the ``terminal_observation`` of every finished episode (the counter of its last step, so a
   time-limit truncation bootstraps on ``phase(max_episode_steps)``; a finished env's next observation is its reset
   observation, ``k = 0``).  Its ``observation_space`` stays the env's: the policy interface.  ``save()`` writes a
   stock ``VecNormalize`` pickle (the env's observation width, the env's statistics), so a checkpoint's sidecar is
   the sidecar every loader, evaluation and judge already reads.
2. :class:`CriticPhaseRolloutBuffer`: PPO's rollout buffer, 3 columns wider, so ``train()`` (and MirrorKLPPO's
   mirrored critic) see the phase of every stored row.
3. :class:`CriticPhaseActorCriticPolicy`: SB3's ``MlpPolicy`` whose actor's first layer reads the observation's
   columns alone (the same shapes, and on a 63-wide input the same tensor, as the stock actor) and whose critic's
   first layer reads the observation and the phase.  Every actor path -- ``forward`` in the rollout,
   ``evaluate_actions`` and ``get_distribution`` in ``train()``, MirrorKLPPO's mirrored KL term, ``predict()`` in
   evaluation, the probe, the K3 panel and the judge -- goes through ``mlp_extractor.forward_actor``, which slices
   the observation's columns off before the first layer: the actor is exactly independent of the phase, and gets no
   gradient through it.  ``predict()`` takes the env's own observation (the deploy interface) as well as the
   training one; a value asked of the critic without the phase is refused, never guessed.  A fresh model is a
   stock ``MlpPolicy`` of the same seed, bitwise, plus three zero columns in the critic's first layer: the critic
   starts out reading no phase and learns to from its first update (the phase screen's zero-column start).
4. MirrorKLPPO maps the phase columns to themselves with sign +1 (``mirror_kl_ppo``): a function of the step
   counter is its own mirror image.  The observation statistics stay the env's width, symmetrised as before.

Checkpoints: ``model.save`` records the policy class and its two keyword arguments (``policy_kwargs``), the buffer
class, and the JSON attribute :data:`~environments.shared.critic_phase.CHECKPOINT_ATTRIBUTE`; ``train_base``
refuses any load whose phase (or its absence) differs from the stage's, both ways (:func:`critic_phase_mismatch`).
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn

from .critic_phase import (
    CHECKPOINT_ATTRIBUTE,
    PHASE_WIDTH,
    SETTLE_ATTRIBUTE,
    CriticPhaseError,
    CriticPhaseSpec,
)

#: The attributes :class:`CriticPhaseVecNormalize` adds to a stock ``VecNormalize`` (left out of its sidecar).
_OWN_ATTRIBUTES = ("critic_phase", "critic_observation_space")


def critic_input_space(observation_space: spaces.Space, width: int = PHASE_WIDTH) -> spaces.Box:
    """The critic's input: the flat Box observation followed by *width* phase columns."""
    if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
        raise CriticPhaseError(f"the critic phase needs a flat Box observation, got {observation_space}")
    return spaces.Box(-np.inf, np.inf, (int(observation_space.shape[0]) + int(width),), np.float32)


# ---------------------------------------------------------------------------------------------------- 1. the env
class CriticPhaseVecNormalize(VecNormalize):
    """The training ``VecNormalize`` that appends the critic's phase (module docstring, 1).

    ``critic_phase_gamma`` is the stage's PPO discount; the settle step is read from every env instance
    (``stance_width_settle_steps``) and must agree.  Every other keyword is ``VecNormalize``'s.
    """

    def __init__(self, venv: Any, *, critic_phase_gamma: float, **kwargs: Any) -> None:
        super().__init__(venv, **kwargs)
        try:
            settles = venv.get_attr(SETTLE_ATTRIBUTE)
        except AttributeError as exc:
            raise CriticPhaseError(
                f"the critic phase reads the env's {SETTLE_ATTRIBUTE} (the reward gates' settle step); "
                f"{type(venv).__name__}'s envs have none"
            ) from exc
        if len({repr(value) for value in settles}) != 1:
            raise CriticPhaseError(f"the envs disagree on {SETTLE_ATTRIBUTE}: {settles!r}")
        self.critic_phase = CriticPhaseSpec(settle_steps=settles[0], gamma=critic_phase_gamma)
        self.critic_observation_space = critic_input_space(self.observation_space)

    def _with_phase(self, observations: np.ndarray, counters: np.ndarray) -> np.ndarray:
        phase = self.critic_phase.matrix(counters).astype(observations.dtype)
        return np.concatenate([observations, phase], axis=-1)

    def reset(self) -> np.ndarray:  # type: ignore[override]
        observations = super().reset()
        assert isinstance(observations, np.ndarray)
        # Every env was just reset: its _step_count is 0.
        return self._with_phase(observations, np.zeros(self.num_envs, dtype=np.int64))

    def step_wait(self) -> Any:
        observations, rewards, dones, infos = super().step_wait()
        assert isinstance(observations, np.ndarray)
        counters = np.zeros(self.num_envs, dtype=np.int64)
        for index, info in enumerate(infos):
            step = info.get("step")
            if step is None:
                raise CriticPhaseError(
                    f"env {index} returned no info['step'] (BaseDinoEnv.step's _step_count): no phase to give the critic"
                )
            if dones[index]:
                # The env was reset by the VecEnv: the returned observation is its reset observation (k = 0); the
                # finished episode's last observation, which a time-limit truncation bootstraps on, has k = step.
                terminal = info.get("terminal_observation")
                if terminal is not None:
                    info["terminal_observation"] = self._with_phase(
                        np.asarray(terminal)[None], np.array([step], dtype=np.int64)
                    )[0]
            else:
                counters[index] = int(step)
        return self._with_phase(observations, counters), rewards, dones, infos

    def save(self, save_path: Any) -> None:
        """Write a stock ``VecNormalize`` pickle: the env's observation width and statistics, no phase."""
        stock = VecNormalize.__new__(VecNormalize)
        stock.__dict__.update({key: value for key, value in self.__dict__.items() if key not in _OWN_ATTRIBUTES})
        with open(save_path, "wb") as handle:
            pickle.dump(stock, handle)


# ---------------------------------------------------------------------------------------------------- 2. the buffer
class CriticPhaseRolloutBuffer(RolloutBuffer):
    """PPO's rollout buffer for the training env's observations with the phase (``critic_phase_width`` wider)."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Any = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        critic_phase_width: int = PHASE_WIDTH,
    ) -> None:
        if int(critic_phase_width) != PHASE_WIDTH:
            raise CriticPhaseError(f"the phase is {PHASE_WIDTH} columns wide, not {critic_phase_width}")
        self.critic_phase_width = int(critic_phase_width)
        super().__init__(
            buffer_size,
            critic_input_space(observation_space, critic_phase_width),
            action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )


# ---------------------------------------------------------------------------------------------------- 3. the policy
class CriticPhaseMlpExtractor(MlpExtractor):
    """``MlpExtractor`` whose actor reads the first ``actor_input_dim`` features and whose critic, once
    :meth:`widen_critic` has run, reads ``critic_phase_width`` more.

    Built as SB3 builds a stock one (the same layers, the same random draws); the policy widens the critic's first
    layer after SB3's own initialisation (:meth:`CriticPhaseActorCriticPolicy._build`).
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Any,
        activation_fn: type[nn.Module],
        device: Any = "auto",
        *,
        critic_phase_width: int = PHASE_WIDTH,
    ) -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)
        first = self.value_net[0] if len(self.value_net) else None
        if not isinstance(first, nn.Linear) or first.in_features != feature_dim:
            raise CriticPhaseError("the critic phase needs a critic with at least one hidden layer (net_arch)")
        self.actor_input_dim = int(feature_dim)
        self.critic_phase_width = int(critic_phase_width)
        self.critic_input_dim = int(feature_dim)  # until widen_critic()

    def widen_critic(self) -> None:
        """Give the critic's first layer the phase columns: its observation columns are the layer as it is, its
        phase columns exactly zero (the critic starts out reading no phase and learns to), with no random draw --
        so a fresh model's every other parameter is the stock model's of the same seed (the widen tool's rule for
        a new input: zero columns)."""
        first = self.value_net[0]
        if first.in_features != self.actor_input_dim:
            raise CriticPhaseError(f"the critic's first layer already reads {first.in_features} inputs")
        wide = th.nn.utils.skip_init(
            nn.Linear,
            self.actor_input_dim + self.critic_phase_width,
            first.out_features,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        with th.no_grad():
            wide.weight[:, : self.actor_input_dim] = first.weight
            wide.weight[:, self.actor_input_dim :] = 0.0
            wide.bias.copy_(first.bias)
        self.value_net[0] = wide
        self.critic_input_dim = self.actor_input_dim + self.critic_phase_width

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # The observation's columns alone, as one contiguous tensor: a 63-wide input passes as the very tensor it is,
        # a 66-wide one as the copy of its first 63 columns -- the same tensor in both cases, so the actor computes
        # the same numbers whatever the phase (and whether one is attached), and nothing of the phase reaches it.
        latent: th.Tensor = self.policy_net(features[..., : self.actor_input_dim].contiguous())
        return latent

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        if features.shape[-1] != self.critic_input_dim:
            raise CriticPhaseError(
                f"the critic reads {self.critic_input_dim} inputs (the {self.actor_input_dim}-value observation and "
                f"the {self.critic_input_dim - self.actor_input_dim}-value phase), got {features.shape[-1]}: append "
                "the phase of each observation's step counter (critic_phase.CriticPhaseSpec.matrix)"
            )
        latent: th.Tensor = self.value_net(features)
        return latent


class CriticPhaseActorCriticPolicy(ActorCriticPolicy):
    """SB3's ``MlpPolicy`` with the critic's phase input (module docstring, 3).

    ``observation_space`` is the env's observation (the actor's input); the critic's input is
    :func:`critic_input_space` of it.  ``critic_phase_settle_steps`` / ``critic_phase_gamma`` are the
    :class:`~environments.shared.critic_phase.CriticPhaseSpec` the training env appends, kept here so a loaded
    checkpoint says what its critic reads.
    """

    #: Read by MirrorKLPPO (``getattr``, 0 on a stock policy): the phase columns map to themselves.
    critic_phase_width = PHASE_WIDTH
    mlp_extractor: CriticPhaseMlpExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Any,
        *args: Any,
        critic_phase_settle_steps: int,
        critic_phase_gamma: float,
        **kwargs: Any,
    ) -> None:
        spec = CriticPhaseSpec(settle_steps=critic_phase_settle_steps, gamma=critic_phase_gamma)
        critic_input_space(observation_space)
        if kwargs.get("features_extractor_class", FlattenExtractor) is not FlattenExtractor:
            raise CriticPhaseError("the critic phase needs SB3's FlattenExtractor (feature i is observation i)")
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        self.critic_phase = spec

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CriticPhaseMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            critic_phase_width=PHASE_WIDTH,
        )

    def _build(self, lr_schedule: Any) -> None:
        # SB3's build and orthogonal initialisation, exactly a stock MlpPolicy's (the same draws), then the critic's
        # phase columns (zero) and the optimizer over the widened parameters (Adam draws nothing).
        super()._build(lr_schedule)
        self.mlp_extractor.widen_critic()
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(self.critic_phase.policy_kwargs())
        return data

    def obs_to_tensor(self, observation: Any) -> tuple[Any, bool]:
        """SB3's, plus the training observation with the phase (the time-limit bootstrap's terminal observation)."""
        if not isinstance(observation, dict):
            array = np.array(observation)
            width = self.mlp_extractor.critic_input_dim
            if array.ndim in (1, 2) and array.shape[-1] == width:
                return obs_as_tensor(array.reshape((-1, width)), self.device), array.ndim == 2
        return super().obs_to_tensor(observation)


# ---------------------------------------------------------------------------------------------------- checks
def model_critic_phase(model: Any) -> CriticPhaseSpec | None:
    """The phase a loaded or built model's critic reads (``None``: a stock critic)."""
    return getattr(getattr(model, "policy", None), "critic_phase", None)


def critic_phase_mismatch(model: Any, wanted: CriticPhaseSpec | None) -> str | None:
    """Why *model* cannot train under a stage whose phase is *wanted* (``None``: the stage has none), else ``None``."""
    loaded = model_critic_phase(model)
    recorded = getattr(model, CHECKPOINT_ATTRIBUTE, None)
    if wanted is None:
        if loaded is not None or recorded is not None:
            return (
                f"the checkpoint's critic reads the episode phase ({recorded!r}) and the stage declares no "
                "[ppo] critic_phase"
            )
        return None
    if loaded is None:
        return "the stage declares [ppo] critic_phase = true and the checkpoint's critic reads no phase"
    if not isinstance(model.policy, CriticPhaseActorCriticPolicy):
        return f"the checkpoint's policy is {type(model.policy).__name__}, not CriticPhaseActorCriticPolicy"
    if loaded != wanted:
        return f"the checkpoint's critic reads {loaded.record()!r}, the stage's {wanted.record()!r}"
    if recorded is not None and dict(recorded) != wanted.record():
        return f"the checkpoint records {recorded!r}, the stage {wanted.record()!r}"
    buffer = getattr(model, "rollout_buffer", None)
    if not isinstance(buffer, CriticPhaseRolloutBuffer):
        return f"the model's rollout buffer is {type(buffer).__name__}, not CriticPhaseRolloutBuffer"
    width = int(model.observation_space.shape[0]) + PHASE_WIDTH
    if tuple(buffer.obs_shape) != (width,) or model.policy.mlp_extractor.critic_input_dim != width:
        return f"the buffer ({buffer.obs_shape}) or the critic does not read {width} inputs"
    return None


def new_model_kwargs(spec: CriticPhaseSpec, policy_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """The PPO constructor arguments that build a model whose critic reads *spec*'s phase."""
    return {
        "policy": CriticPhaseActorCriticPolicy,
        "policy_kwargs": {**dict(policy_kwargs or {}), **spec.policy_kwargs()},
        "rollout_buffer_class": CriticPhaseRolloutBuffer,
        "rollout_buffer_kwargs": {"critic_phase_width": PHASE_WIDTH},
    }
