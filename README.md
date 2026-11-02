# Koopman Linear-Latent Auxiliary Loss for ROSnav

A minimal source snapshot of the Koopman linear-latent auxiliary PPO
implementation developed for the Arena navigation task.

## Method

The implementation adds an auxiliary linear latent prediction objective
to PPO:

    z_hat(t+1) = A z(t) + B a(t)

The auxiliary objective is trained alongside the PPO objective.

## Repository structure

- `rosnav_rl/`: Modified and added ROSnav RL source files.
- `experiments/koopman_experiment.yaml`: Experiment configuration.

## Usage

This is a source-code extension, not a standalone ROSnav package.
Overlay the files under `rosnav_rl/` onto the corresponding paths in
the base ROSnav RL project, then use the experiment configuration.

Base project: https://github.com/Arena-Rosnav/rosnav-rl

## Training checkpoints and configurations

The repository includes the final saved checkpoints and corresponding
training configurations for three experiments:

| Experiment | Checkpoint | Configuration |
|---|---|---|
| Baseline PPO | `models/baseline_ppo/last_model.zip` | `configs/baseline_ppo/training_config.yaml` |
| Koopman PPO (lambda=0.01) | `models/koopman_l01/last_model.zip` | `configs/koopman_l01/training_config.yaml` |
| MLP Auxiliary (lambda=0.01) | `models/mlp_aux_l01/last_model.zip` | `configs/mlp_aux_l01/training_config.yaml` |

These are the saved `last_model.zip` checkpoints from the respective
training runs, not intermediate checkpoints.
