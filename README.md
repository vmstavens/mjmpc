# mjmpc
A collection of sampling based Model Predictive Control algorithms.

If you use this repository as part of your research please cite the following publication::
```
@inproceedings{
bhardwaj2021blending,
title={Blending {\{}MPC{\}} {\&} Value Function Approximation for Efficient Reinforcement Learning},
author={Mohak Bhardwaj and Sanjiban Choudhury and Byron Boots},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=RqCC_00Bg7V}
}
```

## Installation

Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required.
From the repository root:

```sh
uv sync --locked
uv run python examples/example_random_policy.py --steps 200
uv run mjmpc --config examples/configs/pendulum.yml --controller mppi --save_dir ./experiments
```

`pyproject.toml` defines the package and `uv.lock` pins the complete environment.
MuJoCo is installed as a wheel; no license key, separate simulator download, Conda,
`mujoco_py`, or `mjrl` installation is needed. Native MuJoCo 3.x, Gymnasium 1.x,
NumPy 2.x, and current PyTorch are used throughout the maintained implementation.

For SAC training tools and the Stable Baselines3 example:

```sh
uv sync --locked --extra training
uv run --extra training python -m mjmpc.control.softqmpc.scripts.train_sac_stable --steps 10000
```

## Examples and environments

Run examples from the repository root. `example_mpc.py` supports `--config`,
`--controller`, `--save_dir`, `--dyn_randomize_config`, and `--dump_vids`.
It saves metrics and `trajectories.pkl` beneath the selected output directory.
`example_mpc_cl.py` runs closed-loop linear Gaussian REINFORCE;
`TorchModelVecEnv` provides neural policy rollouts; `job_script.py` supports
parameter sweeps. Multiprocessing uses `spawn`, so custom entry points must use
an `if __name__ == "__main__":` guard.

Bundled tasks are `SimplePendulum-v0`, `Swimmer-v0`, `HalfCheetah-v0`,
`reacher_7dof-v0`, and `continual_reacher-v0`; their task rewards and dynamics
are retained. `LQREnv` is available for custom linear systems. Native Gymnasium
MuJoCo environments such as `HalfCheetah-v5` also work with `GymEnvWrapper`.
The older locomotion IDs are retained for existing configurations and may produce
Gymnasium version warnings.

The hand, Sawyer, Panda, point-mass, and lowercase cartpole/acrobot configurations
refer to external environments that were never bundled here. They remain as
experiment references; running them requires a separately registered **Gymnasium**
implementation with equivalent task dynamics and state access. The old `mj_envs`
package is not imported or installed automatically. The missing `continual_maze-v0`
implementation is no longer registered. These tasks have not been substituted with
different benchmark environments.

Rendering is selected when constructing an environment (`render_mode="human"`
or `"rgb_array"`). `--dump_vids` uses imageio/FFmpeg. On a headless Linux machine,
MuJoCo rendering can use `MUJOCO_GL=egl` if an EGL driver is available, or
`MUJOCO_GL=osmesa` with OSMesa installed. Ordinary simulation and tests need no display.

## Migration notes

- Environments follow [Gymnasium's API](https://gymnasium.farama.org/introduction/migration_guide/):
  `reset(seed=...)` returns `(observation, info)` and `step` returns
  `(observation, reward, terminated, truncated, info)`.
- `GymEnvWrapper` and the controller vector interfaces retain their MPC contract:
  observation-only reset and `(observation, reward, done, info)` step. Both end flags
  are retained in `info`; `done` is their union. This is the single adaptation boundary.
- Snapshots now contain `env_state`, `rng_state`, and `elapsed_steps`. MuJoCo
  snapshots use its [native integration state API](https://mujoco.readthedocs.io/en/latest/programming/simulation.html),
  including actuator and solver state. Saved state dictionaries from the old backend
  need conversion; old pickled trajectories are not automatically portable.
- Candidate rollouts restore the starting state, including after an exception.
  Ended episodes are padded with zero rewards and terminal observations. Structured
  observations are flattened with Gymnasium's space utilities in rollout arrays.
- Dynamics randomization retains body mass/inertia, joint damping/friction loss,
  and geometry size/friction. Joint parameters address all of the joint's DoFs.
  The obsolete `sensor_noise` model field is not supported by this native backend.
- Value-function fitting uses `torch.linalg`. Video export uses imageio instead of
  scikit-video. Unused deprecated SAC copies, a commented-out closed-loop prototype,
  and the incomplete, unexported SAC-MPC prototype have been removed. The standalone
  SAC, SoftQ, and neural random-shooting implementations remain.
- iLQR was an unfinished placeholder in the original repository and remains so;
  this migration does not implement a new controller algorithm.

## Development

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv build
```

Tests cover environment contracts, snapshot/replay behavior, time limits, dynamics
randomization, serial/subprocess rollouts, sampling controllers, and value fitting.
Historical `*_test.py` files are standalone research diagnostics, not pytest suites.
CI runs on Python 3.11–3.13 and builds the distributable, including MuJoCo XML assets.

## Controllers
Following parameters are common for all controllers
| Parameter         |                                                                       |
|-------------------|-----------------------------------------------------------------------|
| ``horizon``       | rollout horizon                                                       |
| ``num_particles`` | number of particles to rollout                                        |
| ``n_iters``       | number of iterations of optimization per timestep                     |
| ``gamma``         | discount factor                                                       |
| ``filter_coeffs`` | coefficients for autoregressive filtering (generate correlated noise) |
| ``base_action``   | action to append at the end after shifting distribution for next step |


Additionally, each controller has it's own specific parameters

### Gaussian Controllers
These controllers use a Gaussian control distribution and have the following common parameters

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``init_cov``  | initial covariance of Gaussian                        |
| ``step_size`` | step size for updating distribution at every timestep |


#### Random Shooting
Samples particles from a Gaussian with fixed covariance and selects next mean to be the rollout with minimum cost. Has no additional parameters.

#### Model Predictive Path Integral Control (MPPI)
Based on [Williams et al.](https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf), it samples particles from a Gaussian with fixed covariance and updates the mean using a softmax of rollouts. Has the followig additional parameters:

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``lam``       | temperature for softmax                               |
| ``alpha``     | flag to enable control costs in rollouts (0: enable, 1: disable) |


#### Cross Entropy Method (CEM)
Samples particles from a Gaussian control distribution and updates the mean and covariance using sample estimates from a set of elite samples based on cost.

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``cov_type``       | 'diag' means covariance is forced to be diagonal, and 'full' allows actions to be correlated to each other.               |
| ``elite_frac``     | fraction of total samples considered elite  |
| ``beta``           | ``beta * I`` is added to covariance to grow it at each timestep |

#### Gaussian DMD-MPC
From [Wagener et al.](https://arxiv.org/pdf/1902.08967.pdf). We use the exponentiated utility function and allow it to adapt the covariance as well. Has the same parameters as MPPI with the addition of ``cov_type`` and ``beta``.

## Non-Gaussian Controllers

#### Particle Filter MPC

Uses a non-parametric distribution represented by particles and updates it using a particle filtering approach, where particles weighted using an exponential of running cost with temperature ``lam``

| Parameter     |                                                       |
|---------------|-------------------------------------------------------|
| ``cov_shift``       |  noise added to particles when shifting to next step    |
| ``cov_resample``    | noise for resampling particles  |



## Using your own environments

Controllers accept `rollout_fn` and `set_sim_state_fn` callbacks and are independent
of MuJoCo. `GymEnvWrapper` accepts Gymnasium environments exposing `get_env_state`,
`set_env_state`, and `get_obs` (or `_get_obs`). Native MuJoCo `model`/`data` state
is handled automatically. Keep arbitrary custom task state in the environment's
snapshot methods; see `mjmpc/envs/basic/reacher_env.py` for a target and timed-event example.

## Control Parameters

(TO BE UPDATED!!)

These parameters are currently manually tuned.

| Env Name          | Episode Length | Horizon | Num Particles | Lambda | Covariance | Step Size | Gamma | Num Iters |
|-------------------|----------------|---------|---------------|--------|------------|-----------|-------|-----------|
| SimplePendulum-v0 | 200            | 32      | 24            | 0.01   | 3.5        | 0.55      | 1.0   | 1         |
| Swimmer-v0        | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
| HalfCheetah-v0    | 500            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
| trajopt_reacher-v0| 200            | 32      | 36            | 0.01   | 3.0        | 0.55      | 1.0   | 1         |
