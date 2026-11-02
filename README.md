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
