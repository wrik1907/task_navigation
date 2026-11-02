"""PPO variants with auxiliary next-latent prediction losses."""

from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from .feature_extractors.koopman_head import (
    KoopmanLinearHead,
    MLPNextLatentHead,
)


class AuxiliaryPPO(PPO):
    """PPO with an auxiliary next-latent prediction objective."""

    auxiliary_head_class: type[th.nn.Module] | None = None

    def __init__(
        self,
        *args: Any,
        aux_loss_coef: float = 0.01,
        **kwargs: Any,
    ) -> None:
        if aux_loss_coef < 0:
            raise ValueError("aux_loss_coef must be non-negative")

        # Must be set before PPO initializes the model.
        self.aux_loss_coef = float(aux_loss_coef)

        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        """Create the auxiliary head and register its parameters."""
        super()._setup_model()

        if self.auxiliary_head_class is None:
            raise TypeError(
                "Set auxiliary_head_class in the PPO subclass"
            )

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError(
                "Next-latent prediction expects a continuous Box action space"
            )

        latent_dim = int(self.policy.features_extractor.features_dim)
        action_dim = int(np.prod(self.action_space.shape))

        self.policy.auxiliary_head = self.auxiliary_head_class(
            latent_dim=latent_dim,
            action_dim=action_dim,
        ).to(self.device)

        # The PPO optimizer was created before the auxiliary head.
        self.policy.optimizer.add_param_group(
            {"params": self.policy.auxiliary_head.parameters()}
        )

    def _compute_auxiliary_loss(self, batch_size: int) -> th.Tensor:
        """Predict z(t+1) from z(t) and action(t).

        This method is called during PPO training, after
        rollout_buffer.get() has flattened observations and actions.

        SB3 flattens rollout samples in environment-major order:
            flat_index = env_index * n_steps + time_index

        episode_starts remains in [n_steps, n_envs] layout. Transitions
        crossing episode boundaries and the last step of each environment
        are excluded.
        """
        buffer = self.rollout_buffer

        n_steps = int(buffer.buffer_size)
        n_envs = int(buffer.n_envs)
        total_samples = n_steps * n_envs

        if n_steps < 2 or n_envs < 1 or batch_size <= 0:
            return th.zeros((), device=self.device)

        starts = np.asarray(buffer.episode_starts, dtype=bool)
        actions = np.asarray(buffer.actions)
        observations = buffer.observations

        if starts.shape != (n_steps, n_envs):
            raise RuntimeError(
                "Unexpected episode_starts shape: "
                f"{starts.shape}; expected {(n_steps, n_envs)}"
            )

        if not isinstance(observations, dict) or not observations:
            raise RuntimeError(
                "Expected Dict observations, got "
                f"{type(observations).__name__}"
            )

        # This implementation expects SB3's flattened rollout arrays.
        if actions.shape[0] != total_samples:
            raise RuntimeError(
                "Actions are not in the expected flattened layout: "
                f"shape={actions.shape}, expected first dimension "
                f"{total_samples}. Ensure this method is called after "
                "rollout_buffer.get() starts iteration."
            )

        for key, value in observations.items():
            if np.asarray(value).shape[0] != total_samples:
                raise RuntimeError(
                    f"Observation '{key}' is not flattened as expected: "
                    f"shape={np.asarray(value).shape}; "
                    f"expected first dimension {total_samples}"
                )

        # Indices are organized by environment, then by timestep.
        # Drop each environment's last step because its next observation
        # is not stored as the next row in that environment's segment.
        current_indices = np.arange(total_samples).reshape(
            n_envs, n_steps
        )[:, :-1]

        # starts[t + 1, env] marks the beginning of a new episode.
        # Transpose to align its dimensions with current_indices.
        valid_mask = ~starts[1:, :].T
        valid_indices = current_indices[valid_mask]

        if valid_indices.size == 0:
            return th.zeros((), device=self.device)

        sample_count = min(batch_size, int(valid_indices.size))

        selected = np.random.choice(
            valid_indices,
            size=sample_count,
            replace=False,
        )

        # Because the final timestep of each environment was excluded,
        # selected + 1 always remains inside that environment's segment.
        obs_t_np = {
            key: np.asarray(value)[selected]
            for key, value in observations.items()
        }

        obs_next_np = {
            key: np.asarray(value)[selected + 1]
            for key, value in observations.items()
        }

        obs_t = {
            key: th.as_tensor(value, device=self.device)
            for key, value in obs_t_np.items()
        }

        obs_next = {
            key: th.as_tensor(value, device=self.device)
            for key, value in obs_next_np.items()
        }

        action_np = actions[selected].reshape(sample_count, -1)

        action_t = th.as_tensor(
            action_np,
            dtype=th.float32,
            device=self.device,
        )

        # Current latent: gradients flow through the shared extractor.
        z_t = self.policy.extract_features(obs_t)

        # Target latent: stop-gradient, as required by the auxiliary loss.
        with th.no_grad():
            z_next = self.policy.extract_features(obs_next)

        z_pred = self.policy.auxiliary_head(z_t, action_t)

        return F.mse_loss(z_pred, z_next.detach())

    def train(self) -> None:
        """Optimize PPO and the auxiliary objective in one optimizer step.

        To reduce peak GPU memory, backpropagate the auxiliary objective
        before constructing the PPO forward graph. Do not clear gradients
        between the two backward passes: their gradients must accumulate.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(
            self._current_progress_remaining
        )

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        auxiliary_losses = []
        total_losses = []
        approx_kl_divs = []

        continue_training = True

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Clear once at the start of this minibatch.
                # Gradients from the auxiliary and PPO objectives will
                # accumulate before a single optimizer step.
                self.policy.optimizer.zero_grad(set_to_none=True)

                # Use a small auxiliary batch to reduce activation memory.
                # The loss is still sampled from valid transitions across
                # all available environments.
                auxiliary_loss = self._compute_auxiliary_loss(
                    batch_size=min(8, len(rollout_data.actions))
                )

                auxiliary_losses.append(
                    auxiliary_loss.detach().item()
                )

                auxiliary_objective = (
                    self.aux_loss_coef * auxiliary_loss
                )

                # Backpropagate now so the auxiliary activation graph can
                # be released before the PPO forward pass.
                if auxiliary_objective.requires_grad:
                    auxiliary_objective.backward()

                actions = rollout_data.actions

                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = (
                    self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                    )
                )

                values = values.flatten()
                advantages = rollout_data.advantages

                if (
                    self.normalize_advantage
                    and len(advantages) > 1
                ):
                    advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)

                ratio = th.exp(
                    log_prob - rollout_data.old_log_prob
                )

                policy_loss_1 = advantages * ratio

                policy_loss_2 = advantages * th.clamp(
                    ratio,
                    1 - clip_range,
                    1 + clip_range,
                )

                policy_loss = -th.min(
                    policy_loss_1,
                    policy_loss_2,
                ).mean()

                pg_losses.append(policy_loss.detach().item())

                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()

                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = (
                        rollout_data.old_values
                        + th.clamp(
                            values - rollout_data.old_values,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )

                value_loss = F.mse_loss(
                    rollout_data.returns,
                    values_pred,
                )

                value_losses.append(value_loss.detach().item())

                if entropy is None:
                    entropy_loss = th.mean(log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(
                    entropy_loss.detach().item()
                )

                # The auxiliary gradient has already been accumulated.
                # This loss contributes the standard PPO gradients.
                ppo_loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                total_loss_value = (
                    ppo_loss.detach().item()
                    + auxiliary_objective.detach().item()
                )

                total_losses.append(total_loss_value)

                with th.no_grad():
                    log_ratio = (
                        log_prob - rollout_data.old_log_prob
                    )

                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).item()

                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False

                    if self.verbose >= 1:
                        print(
                            f"Early stopping at epoch {epoch} "
                            "due to reaching target KL."
                        )

                    # No optimizer step for this minibatch.
                    # Any accumulated gradients are cleared at the next
                    # minibatch start.
                    break

                # Accumulate PPO gradients on top of the auxiliary
                # gradients; do NOT zero gradients again here.
                ppo_loss.backward()

                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )

                # One optimizer step updates both the shared extractor
                # and auxiliary head using the combined gradients.
                self.policy.optimizer.step()

            self._n_updates += 1

            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record(
            "train/entropy_loss",
            float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        )

        self.logger.record(
            "train/policy_gradient_loss",
            float(np.mean(pg_losses)) if pg_losses else 0.0,
        )

        self.logger.record(
            "train/value_loss",
            float(np.mean(value_losses)) if value_losses else 0.0,
        )

        self.logger.record(
            "train/auxiliary_loss",
            float(np.mean(auxiliary_losses))
            if auxiliary_losses
            else 0.0,
        )

        self.logger.record(
            "train/aux_loss_coef",
            self.aux_loss_coef,
        )

        self.logger.record(
            "train/approx_kl",
            float(np.mean(approx_kl_divs))
            if approx_kl_divs
            else 0.0,
        )

        self.logger.record(
            "train/clip_fraction",
            float(np.mean(clip_fractions))
            if clip_fractions
            else 0.0,
        )

        self.logger.record(
            "train/loss",
            float(np.mean(total_losses))
            if total_losses
            else 0.0,
        )

        self.logger.record(
            "train/explained_variance",
            explained_var,
        )

        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )

        self.logger.record(
            "train/clip_range",
            clip_range,
        )

        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std",
                th.exp(self.policy.log_std).mean().item(),
            )


class KoopmanPPO(AuxiliaryPPO):
    """PPO with the linear Koopman next-latent prediction head."""

    auxiliary_head_class = KoopmanLinearHead


class MLPAuxiliaryPPO(AuxiliaryPPO):
    """PPO with a two-layer MLP next-latent prediction control."""

    auxiliary_head_class = MLPNextLatentHead