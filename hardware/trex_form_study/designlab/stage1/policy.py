"""Stage 1's privileged-critic actor-critic for SB3 PPO (``STAGE1_PLAN.md`` section 2; the pattern of the repository's
``environments/shared/critic_phase_ppo.CriticPhaseActorCriticPolicy``).

The env's observation (``env.obs_layout``) is the actor's deployable inputs (``env.ACTOR_DIM`` columns) followed by the
privileged inputs only the critic reads.  ``PrivilegedCriticPolicy`` is SB3's ``MlpPolicy`` (separate actor and critic
networks, ``net_arch``) whose actor's first layer reads the actor's columns alone and whose critic's first layer reads
every column.  Every actor path -- ``forward`` in the rollout, ``evaluate_actions`` and ``get_distribution`` in
``train()``, ``predict()`` -- goes through ``mlp_extractor.forward_actor``, which slices the actor's columns off before
the first layer, as one contiguous tensor: the actor is exactly independent of the privileged inputs (whatever their
values, and whether they are attached) and gets no gradient through them.  ``predict()`` also takes the actor's columns
alone (the deploy interface): the privileged part is then filled with zeros, which the actor never reads; a value asked
of the critic that way would be meaningless, and the deploy path asks none.

A model saves its ``actor_dim`` with its other constructor arguments, so ``PPO.load`` rebuilds the same policy.
``DESIGN2`` holds design 2's network and PPO settings (plan section 2: "SB3 PPO uses Design 2's settings").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor
from stable_baselines3.common.utils import get_device, obs_as_tensor
from torch import nn

#: Design 2's settings (the design study, "PPO settings"): MlpPolicy [256, 256]; 4 envs x 2,048 steps; batch 256;
#: 5 epochs; gamma 0.99; lambda 0.95; clip 0.2; learning rate 1e-4 -> 3e-5 (linear); target_kl 0.03; the stack's
#: VecNormalize (clip_obs 10, clip_reward 50).
DESIGN2 = dict(net_arch=(256, 256), n_envs=4, n_steps=2048, batch_size=256, n_epochs=5, gamma=0.99, gae_lambda=0.95,
               clip_range=0.2, lr_start=1e-4, lr_end=3e-5, target_kl=0.03, clip_obs=10.0, clip_reward=50.0)


class PrivilegedError(ValueError):
    """The policy cannot be built or asked as requested."""


class PrivilegedMlpExtractor(MlpExtractor):
    """``MlpExtractor`` whose actor reads the first ``actor_dim`` features and whose critic reads all of them.

    Built as SB3 builds a stock one, then the actor's first layer is made ``actor_dim`` wide (before SB3's own
    initialisation, which therefore initialises it as any other layer)."""

    def __init__(self, feature_dim: int, net_arch: Any, activation_fn: type[nn.Module], device: Any = "auto", *,
                 actor_dim: int) -> None:
        super().__init__(feature_dim, net_arch, activation_fn, device)
        actor_dim = int(actor_dim)
        if not 0 < actor_dim <= feature_dim:
            raise PrivilegedError(f"actor_dim {actor_dim} outside 1..{feature_dim}")
        first = self.policy_net[0] if len(self.policy_net) else None
        if not isinstance(first, nn.Linear) or first.in_features != feature_dim:
            raise PrivilegedError("the privileged critic needs an actor with at least one hidden layer (net_arch)")
        if not len(self.value_net) or not isinstance(self.value_net[0], nn.Linear):
            raise PrivilegedError("the privileged critic needs a critic with at least one hidden layer (net_arch)")
        self.policy_net[0] = nn.Linear(actor_dim, first.out_features).to(get_device(device))
        self.actor_dim = actor_dim
        self.critic_dim = int(feature_dim)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # the actor's columns alone, as one contiguous tensor: the same numbers whatever the privileged columns hold
        latent: th.Tensor = self.policy_net(features[..., : self.actor_dim].contiguous())
        return latent

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        if features.shape[-1] != self.critic_dim:
            raise PrivilegedError(f"the critic reads {self.critic_dim} inputs (the {self.actor_dim} actor inputs and the "
                                  f"privileged ones), got {features.shape[-1]}")
        latent: th.Tensor = self.value_net(features)
        return latent


class PrivilegedCriticPolicy(ActorCriticPolicy):
    """SB3's ``MlpPolicy`` with a privileged critic (module docstring).  ``actor_dim``: the actor's columns, the first
    of the flat Box observation."""

    mlp_extractor: PrivilegedMlpExtractor

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space, lr_schedule: Any, *args: Any,
                 actor_dim: int, **kwargs: Any) -> None:
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise PrivilegedError(f"the privileged critic needs a flat Box observation, got {observation_space}")
        if kwargs.get("features_extractor_class", FlattenExtractor) is not FlattenExtractor:
            raise PrivilegedError("the privileged critic needs SB3's FlattenExtractor (feature i is observation i)")
        self.actor_dim = int(actor_dim)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PrivilegedMlpExtractor(self.features_dim, net_arch=self.net_arch,
                                                    activation_fn=self.activation_fn, device=self.device,
                                                    actor_dim=self.actor_dim)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data: dict[str, Any] = super()._get_constructor_parameters()
        data["actor_dim"] = self.actor_dim
        return data

    def obs_to_tensor(self, observation: Any) -> tuple[Any, bool]:
        """SB3's, plus the actor's columns alone (the deploy interface): the privileged columns as zeros."""
        if not isinstance(observation, dict):
            array = np.asarray(observation, dtype=np.float32)
            full = int(self.observation_space.shape[0])
            if self.actor_dim != full and array.ndim in (1, 2) and array.shape[-1] == self.actor_dim:
                pad = np.zeros(array.shape[:-1] + (full - self.actor_dim,), np.float32)
                return obs_as_tensor(np.concatenate([array, pad], axis=-1).reshape((-1, full)), self.device), \
                    array.ndim == 2
        stock: tuple[Any, bool] = super().obs_to_tensor(observation)
        return stock


def linear_schedule(start: float, end: float):
    """SB3 schedule: ``start`` at the beginning (progress 1) to ``end`` at the end (progress 0)."""
    start, end = float(start), float(end)

    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * float(progress_remaining)

    return schedule


def model_kwargs(actor_dim: int, *, net_arch=DESIGN2["net_arch"], log_std_init: float = -1.0) -> dict[str, Any]:
    """The PPO constructor arguments of design 2 with the privileged critic (``n_envs`` is the vec env's)."""
    d = DESIGN2
    return dict(policy=PrivilegedCriticPolicy,
                policy_kwargs=dict(actor_dim=int(actor_dim), net_arch=dict(pi=list(net_arch), vf=list(net_arch)),
                                   log_std_init=float(log_std_init)),
                n_steps=d["n_steps"], batch_size=d["batch_size"], n_epochs=d["n_epochs"], gamma=d["gamma"],
                gae_lambda=d["gae_lambda"], clip_range=d["clip_range"],
                learning_rate=linear_schedule(d["lr_start"], d["lr_end"]), target_kl=d["target_kl"])


__all__ = ["DESIGN2", "PrivilegedCriticPolicy", "PrivilegedError", "PrivilegedMlpExtractor", "linear_schedule",
           "model_kwargs"]
