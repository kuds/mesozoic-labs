"""Mirror-symmetric PPO: stock Stable-Baselines3 PPO plus a mirror loss (``[ppo] symmetry = "mirror_kl"``).

A left-right symmetric animal on a level floor should act left-right symmetrically: in the mirror image of a
state its policy should take the mirror image of the action, ``pi(. | M_o s) = M_a pi(. | s)``, with ``M_o`` and
``M_a`` the observation and action maps of :mod:`environments.shared.symmetry`.  Stock PPO has no reason to find
that policy: the T. rex stance runs of 2026-09 and 2026-10 settled into lopsided postures (a lean onto one pad, a
left-right ankle offset) that a symmetric policy cannot express.  :class:`MirrorKLPPO` trains toward the symmetric
policy with four additions, all confined to training; a checkpoint is still a stock ``MlpPolicy`` zip and a stock
``VecNormalize`` pickle, which evaluation, reports and the next stage load with stock PPO.

1. Symmetric observation statistics (:class:`SymmetricRunningMeanStd`): the training env's ``VecNormalize``
   statistics are those of each batch together with its mirror image, symmetrised bitwise after every update, so
   ``normalize(M_o s) == M_o normalize(s)`` exactly and the mirrored observation of a stored (normalised) one is
   ``o[:, perm] * sign``.  They pickle as the stock ``RunningMeanStd``.
2. A tied exploration noise: a gradient hook averages each mirror pair's ``log_std`` gradient, so the pairs stay
   bitwise equal through the gradient clip and Adam (:func:`register_log_std_tie`).
3. The mirror loss, the KL divergence ``KL(pi(. | M_o s) || M_a pi(. | s))`` with its second argument and the
   noise scale held fixed, per mini-batch::

       o_m    = o[:, obs_perm] * obs_sign
       sigma  = exp(log_std).detach().clamp_min(sigma_floor)
       target = (mu(o)[:, act_perm] * act_sign).detach()
       L_K    = mean over rows of sum_i (mu(o_m)_i - target_i)^2 / (2 sigma_i^2)
       loss   = stock PPO loss + kappa * L_K

4. A mirrored critic: the value loss is ``0.5 (mse(R, V(o)) + mse(R, V(o_m)))``.

Everything else is stock: the rollout, GAE, the clipped surrogate on the original rows, the entropy term, the
``target_kl`` early stop, the gradient clip and the Adam step.  :meth:`MirrorKLPPO.train` is SB3 2.9.0's
``PPO.train`` with four added lines, each marked ``# [mirror]`` (:func:`train_source_diff`); the copied source is
pinned by its digest (:data:`SB3_PPO_TRAIN_SHA256`), and another SB3 refuses to enable the class.  With
``kappa = 0`` and the mirrored critic off it trains bit for bit as stock PPO with the same statistics and tie.

Why not mirror augmentation.  The usual method (Yu, Turk and Liu 2018; Mittal et al. 2024) adds the mirrored
transitions ``(M_o s, M_a a)`` to the PPO batch with the original advantages and old log-probabilities.  Its
restoring force is the clipped ratio of the mirrored action under the current policy, and that ratio measures the
asymmetry in units of the exploration noise sigma.  At the stance stage's small noise (knee and ankle sigma about
0.03 at 3M steps, about 0.02 projected at 6M) an asymmetry of a few hundredths of an action unit already pushes
most mirrored rows out of the clip band, where their gradient vanishes (on the stored 3M stance rollouts 86% of the
mirrored rows clip at an asymmetry of 0.03, and the restoring force is about zero beyond 0.1, or beyond 0.06 at
the projected 6M noise), while the unclipped rows with negative advantage are unbounded, so the gradient norm
turns heavy-tailed (its largest pre-clip value an order of magnitude or more above stock PPO's).  The KL form has neither problem: its gradient is linear
in the asymmetry and never collapses, its balance against the PPO gradient depends only on the asymmetry over
sigma (held fixed, so the loss cannot be lowered by widening the noise, and floored at ``sigma_floor``, so it
stays bounded as the noise shrinks), and the PPO block is unchanged.

The defaults (kappa 0.3, sigma floor 0.01, mirrored critic on) are the T. rex stance screen's of 2026-10 (arm L,
with the lean term and the settled leg centre of ``TyrannosaurusRexEnv``): three seeds to 6M steps trained
mirror-symmetric policies (every leg group's median left-right action difference under 0.1 degree) that stood
without falls and passed ``stance_quality/v2`` on two of three seeds, but not the screen's pad centre-of-pressure
and noise-robustness criteria; so no stage enables it.  :func:`enable_mirror_kl` turns a constructed or loaded
stock ``PPO`` into a :class:`MirrorKLPPO` in place (``train_base`` does, when a stage's ``[ppo]`` table carries the
symmetry keys), and records the settings in the checkpoint attribute ``mesozoic_symmetry``.
"""

from __future__ import annotations

import difflib
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import stable_baselines3
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from .symmetry import CHECKPOINT_ATTRIBUTE, MirrorMap, SignedPermutation, SymmetryConfig, SymmetryError

#: The SB3 release whose ``PPO.train`` :meth:`MirrorKLPPO.train` copies, and the digest of that method's source.
SB3_VERSION = "2.9.0"
SB3_PPO_TRAIN_SHA256 = "7c8d09940123242f0ac401c5ec2cc858b8bf3339f571c222382f048ffd8306ae"
#: Every line :meth:`MirrorKLPPO.train` adds to the stock source carries this marker.
MIRROR_MARKER = "# [mirror]"


def stock_ppo_train_sha256() -> str:
    return hashlib.sha256(inspect.getsource(PPO.train).encode("utf-8")).hexdigest()


def check_sb3_pin() -> None:
    """Refuse an SB3 whose ``PPO.train`` is not the one :meth:`MirrorKLPPO.train` copies."""
    if stable_baselines3.__version__ != SB3_VERSION or stock_ppo_train_sha256() != SB3_PPO_TRAIN_SHA256:
        raise SymmetryError(
            f"MirrorKLPPO copies stable-baselines3 {SB3_VERSION}'s PPO.train (source sha256 {SB3_PPO_TRAIN_SHA256}); "
            f"this is {stable_baselines3.__version__} (sha256 {stock_ppo_train_sha256()})"
        )


# ---------------------------------------------------------------------------------------------------- statistics
class SymmetricRunningMeanStd(RunningMeanStd):
    """``RunningMeanStd`` over each batch and its mirror image, kept bitwise symmetric; pickles as the stock class."""

    mean: np.ndarray
    var: np.ndarray
    count: float

    def __init__(self, obs_map: SignedPermutation, epsilon: float = 1e-4, shape: tuple[int, ...] = ()) -> None:
        super().__init__(epsilon=epsilon, shape=shape)
        if tuple(self.mean.shape) != (obs_map.dim,):
            raise SymmetryError(
                f"observation statistics of shape {self.mean.shape} against a map of width {obs_map.dim}"
            )
        self._perm = obs_map.perm
        self._sign = obs_map.sign

    @classmethod
    def from_stock(cls, rms: Any, obs_map: SignedPermutation) -> SymmetricRunningMeanStd:
        """A copy of *rms*; refused unless its statistics are already bitwise symmetric (a fresh or a symmetric run)."""
        out = cls(obs_map, shape=tuple(np.asarray(rms.mean).shape))
        out.mean = np.array(rms.mean, dtype=np.float64, copy=True)
        out.var = np.array(rms.var, dtype=np.float64, copy=True)
        out.count = float(rms.count)
        if out.residual() != (0.0, 0.0):
            raise SymmetryError(
                f"the observation statistics are not mirror-symmetric (residual {out.residual()}): a checkpoint "
                "trained without symmetry cannot continue with it"
            )
        return out

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        mirrored_mean = batch_mean[..., self._perm] * self._sign
        # The exact moments of [x; M_o x]: the mean of the two means, and the mean of the two variances plus the
        # variance of the two means.
        union_mean = (batch_mean + mirrored_mean) / 2
        union_var = (batch_var + batch_var[..., self._perm]) / 2 + ((batch_mean - mirrored_mean) / 2) ** 2
        self.update_from_moments(union_mean, union_var, 2 * arr.shape[0])
        self.mean = (self.mean + self.mean[..., self._perm] * self._sign) / 2
        self.var = (self.var + self.var[..., self._perm]) / 2

    def residual(self) -> tuple[float, float]:
        """``(max |mean - M_o mean|, max |var - P var|)``, both exactly 0 when bitwise symmetric."""
        return (
            float(np.max(np.abs(self.mean - self.mean[..., self._perm] * self._sign))),
            float(np.max(np.abs(self.var - self.var[..., self._perm]))),
        )

    def __reduce__(self) -> Any:
        # Pickle (and deep-copy) as the stock class: the VecNormalize sidecar and the eval env's copy stay stock.
        state = {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}
        return (RunningMeanStd, (1e-4, tuple(self.mean.shape)), state)


def install_symmetric_obs_rms(vec_normalize: Any, mirror_map: MirrorMap) -> None:
    """Make a ``VecNormalize``'s observation statistics a :class:`SymmetricRunningMeanStd` (a no-op if already)."""
    rms = vec_normalize.obs_rms
    if isinstance(rms, SymmetricRunningMeanStd):
        if rms.residual() != (0.0, 0.0):
            raise SymmetryError(f"the symmetric observation statistics lost their symmetry: {rms.residual()}")
        return
    if not isinstance(rms, RunningMeanStd):
        raise SymmetryError(f"mirror-symmetric training needs a Box observation, got statistics {type(rms).__name__}")
    vec_normalize.obs_rms = SymmetricRunningMeanStd.from_stock(rms, mirror_map.obs)


# ---------------------------------------------------------------------------------------------------- log_std tie
def log_std_pair_gap(policy: Any, mirror_map: MirrorMap) -> float:
    """The largest difference between a ``log_std`` entry and its mirror partner's (0 when tied)."""
    log_std = policy.log_std.detach().cpu().numpy().astype(np.float64)
    return float(np.max(np.abs(log_std - log_std[mirror_map.action.perm])))


def register_log_std_tie(policy: Any, mirror_map: MirrorMap) -> Any:
    """Register ``g <- (g + g[pair_perm]) / 2`` on ``policy.log_std``; refused unless the pairs are already equal."""
    if log_std_pair_gap(policy, mirror_map) != 0.0:
        raise SymmetryError(
            "the policy's log_std mirror pairs are not equal: a checkpoint trained without symmetry cannot continue "
            "with it"
        )
    perm = th.as_tensor(mirror_map.action.perm, device=policy.log_std.device)

    def tie(grad: th.Tensor) -> th.Tensor:
        return (grad + grad[perm]) / 2

    return policy.log_std.register_hook(tie)


# ---------------------------------------------------------------------------------------------------- the loss
def mirror_kl_loss(
    mu_mirrored: th.Tensor,
    mu: th.Tensor,
    log_std: th.Tensor,
    act_perm: th.Tensor,
    act_sign: th.Tensor,
    sigma_floor: float,
) -> tuple[th.Tensor, th.Tensor]:
    """``(L_K, asymmetry)``: ``L_K = mean_rows sum_i (mu_mirrored - M_a mu)_i^2 / (2 sigma_i^2)``, the target and
    ``sigma = exp(log_std).clamp_min(sigma_floor)`` detached (so ``dL_K / dlog_std == 0``); ``asymmetry`` is the
    detached row mean of ``max_i |mu_mirrored - M_a mu|_i / 2``, in action units."""
    sigma = log_std.exp().detach().clamp_min(sigma_floor)
    target = (mu[:, act_perm] * act_sign).detach()
    deviation = mu_mirrored - target
    loss = (deviation**2 / (2 * sigma**2)).sum(-1).mean()
    asymmetry = deviation.detach().abs().amax(-1).mean() / 2
    return loss, asymmetry


@dataclass
class _MirrorState:
    """What a :class:`MirrorKLPPO` holds beyond a stock PPO (never saved)."""

    config: SymmetryConfig
    mirror_map: MirrorMap
    tie_handle: Any = None
    tensors: dict[str, th.Tensor] = field(default_factory=dict)


class MirrorKLPPO(PPO):
    """Stock PPO plus the mirror loss and the mirrored critic (see the module docstring).

    Made only by :func:`enable_mirror_kl` from a constructed or loaded stock PPO.
    """

    _mirror: _MirrorState | None = None

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_mirror"]

    def _mirror_state(self) -> _MirrorState:
        if self._mirror is None:
            raise SymmetryError("a MirrorKLPPO is made by enable_mirror_kl")
        return self._mirror

    def _mirror_install(self) -> None:
        """(Re)install the symmetric statistics on the training env and the log_std tie (each learn())."""
        state = self._mirror_state()
        vec_normalize = self.get_vec_normalize_env()
        if vec_normalize is not None and vec_normalize.norm_obs:
            install_symmetric_obs_rms(vec_normalize, state.mirror_map)
        if state.tie_handle is not None:
            state.tie_handle.remove()
            state.tie_handle = None
        state.tie_handle = register_log_std_tie(self.policy, state.mirror_map)

    def learn(self, *args: Any, **kwargs: Any) -> Any:
        self._mirror_install()
        return super().learn(*args, **kwargs)

    # ------------------------------------------------------------------ the four steps train() adds
    def _mirror_begin_update(self) -> dict[str, Any]:
        state = self._mirror_state()
        if not state.tensors or state.tensors["obs_perm"].device != self.device:
            state.tensors = {
                "obs_perm": th.as_tensor(state.mirror_map.obs.perm, device=self.device),
                "obs_sign": th.as_tensor(state.mirror_map.obs.sign, dtype=th.float32, device=self.device),
                "act_perm": th.as_tensor(state.mirror_map.action.perm, device=self.device),
                "act_sign": th.as_tensor(state.mirror_map.action.sign, dtype=th.float32, device=self.device),
            }
        return {"kl": [], "asymmetry": [], "value": [], "observations": None}

    def _mirror_value_loss(self, update: dict[str, Any], rollout_data: Any, value_loss: th.Tensor) -> th.Tensor:
        state = self._mirror_state()
        tensors = state.tensors
        update["observations"] = rollout_data.observations[:, tensors["obs_perm"]] * tensors["obs_sign"]
        if not state.config.mirror_critic:
            return value_loss
        mirrored_value_loss = F.mse_loss(
            rollout_data.returns, self.policy.predict_values(update["observations"]).flatten()
        )
        update["value"].append(mirrored_value_loss.item())
        return 0.5 * (value_loss + mirrored_value_loss)

    def _mirror_loss(self, update: dict[str, Any], rollout_data: Any, loss: th.Tensor) -> th.Tensor:
        state = self._mirror_state()
        tensors = state.tensors
        with th.no_grad():
            mu = self.policy.get_distribution(rollout_data.observations).mode()  # the Gaussian's mean
        weighted = state.config.kappa > 0.0
        with th.set_grad_enabled(weighted):
            mu_mirrored = self.policy.get_distribution(update["observations"]).mode()
            kl, asymmetry = mirror_kl_loss(
                mu_mirrored,
                mu,
                cast(th.Tensor, self.policy.log_std),
                tensors["act_perm"],
                tensors["act_sign"],
                state.config.sigma_floor,
            )
        update["kl"].append(kl.item())
        update["asymmetry"].append(asymmetry.item())
        return loss + state.config.kappa * kl if weighted else loss

    def _mirror_end_update(self, update: dict[str, Any]) -> None:
        self.logger.record("train/mirror_kl_loss", float(np.mean(update["kl"])) if update["kl"] else float("nan"))
        self.logger.record(
            "train/mirror_asymmetry", float(np.mean(update["asymmetry"])) if update["asymmetry"] else float("nan")
        )
        if update["value"]:
            self.logger.record("train/mirror_value_loss", float(np.mean(update["value"])))

    # ------------------------------------------------------------------ SB3 2.9.0 PPO.train, four lines added
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]
        mirror = self._mirror_begin_update()  # [mirror] the maps as tensors; this update's telemetry

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                value_loss = self._mirror_value_loss(mirror, rollout_data, value_loss)  # [mirror] o_m; mirrored critic

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss = self._mirror_loss(mirror, rollout_data, loss)  # [mirror] + kappa L_K

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self._mirror_end_update(mirror)  # [mirror] train/mirror_* telemetry


def train_source_diff() -> dict[str, list[str]]:
    """The lines :meth:`MirrorKLPPO.train` adds to, and removes from, the stock ``PPO.train`` source (dedented)."""

    def lines(function: Any) -> list[str]:
        source = inspect.getsource(function).splitlines()
        indent = len(source[0]) - len(source[0].lstrip())
        return [line[indent:] for line in source]

    added, removed = [], []
    for line in difflib.ndiff(lines(PPO.train), lines(MirrorKLPPO.train)):
        if line.startswith("+ "):
            added.append(line[2:])
        elif line.startswith("- "):
            removed.append(line[2:])
    return {"added": added, "removed": removed}


def enable_mirror_kl(model: Any, config: SymmetryConfig, mirror_map: MirrorMap) -> MirrorKLPPO:
    """Turn a constructed or loaded stock ``PPO`` into a :class:`MirrorKLPPO` in place, and return it.

    Construction and loading stay SB3's own (a fresh model has stock PPO's initial parameters).  The training env's
    observation statistics must already be mirror-symmetric (fresh, or from a symmetric run) and the ``log_std``
    pairs equal: both are installed and checked here and again at every ``learn()``.  The model gains the
    JSON-only attribute ``mesozoic_symmetry`` (:meth:`SymmetryConfig.checkpoint_record`).
    """
    check_sb3_pin()
    if type(model) not in (PPO, MirrorKLPPO):
        raise SymmetryError(f"mirror-symmetric training needs a stock PPO model, got {type(model).__name__}")
    if not isinstance(model.observation_space, spaces.Box) or not isinstance(model.action_space, spaces.Box):
        raise SymmetryError("mirror-symmetric training needs Box observation and action spaces")
    if type(model.policy.action_dist) is not DiagGaussianDistribution:
        raise SymmetryError(
            "mirror-symmetric training needs the state-independent Gaussian policy (no gSDE, no squashing), got "
            f"{type(model.policy.action_dist).__name__}"
        )
    if (int(model.observation_space.shape[0]), int(model.action_space.shape[0])) != (
        mirror_map.obs.dim,
        mirror_map.action.dim,
    ):
        raise SymmetryError(
            f"the model's spaces {model.observation_space.shape} / {model.action_space.shape} are not the mirror "
            f"map's widths {mirror_map.obs.dim} / {mirror_map.action.dim}"
        )
    model.__class__ = MirrorKLPPO
    mirrored = cast(MirrorKLPPO, model)
    if mirrored._mirror is not None and mirrored._mirror.tie_handle is not None:
        mirrored._mirror.tie_handle.remove()
    mirrored._mirror = _MirrorState(config=config, mirror_map=mirror_map)
    mirrored._mirror_install()
    setattr(mirrored, CHECKPOINT_ATTRIBUTE, config.checkpoint_record(mirror_map))
    return mirrored
