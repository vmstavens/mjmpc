# mjmpc

Model predictive control with native MuJoCo and Gymnasium. Includes MPPI, CEM,
random shooting, DMD-MPC, particle-filter MPC, and closed-loop REINFORCE.
MuJoCo physics and the MPPI examples run on the CPU; the viewer requires OpenGL.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
the repository root. Python and dependencies are managed by uv.

```bash
uv sync --locked
```

For the commands below, use a Bash-compatible terminal. Limit numerical-library
threads to avoid oversubscribing laptop CPUs:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Run the examples

### MuJoCo robot arm with MPC and a viewer

```bash
MUJOCO_GL=glfw uv run python examples/example_mpc.py \
  --config examples/configs/reacher_7dof-v0.yml \
  --controller mppi --save_dir ./experiments
```

Requires a graphical desktop. The window shows the arm executing MPPI actions;
press Enter in the terminal after the run to replay it. The config uses eight
workers. For a smaller memory footprint, set `num_cpu: 2` and
`particles_per_cpu: 16` in its `mppi` section (still 32 candidate trajectories).

### Pendulum MPC without a display

```bash
uv run python examples/example_mpc.py \
  --config examples/configs/pendulum.yml \
  --controller mppi --save_dir ./experiments
```

This pendulum uses analytic dynamics, not MuJoCo. `uv run mjmpc` is also available
as an alias for `uv run python examples/example_mpc.py`.

### Closed-loop pendulum MPC

```bash
uv run python examples/example_mpc_cl.py \
  --config examples/configs/pendulum_cl.yml \
  --controller reinforce --save_dir ./experiments
```

### Random policy in MuJoCo

```bash
MUJOCO_GL=glfw uv run python examples/example_random_policy.py \
  --env reacher_7dof-v0 --steps 200 --render
```

Omit `--render` to run without a window.

### Benchmark the reacher with MPPI

```bash
MUJOCO_GL=glfw uv run python examples/job_script.py \
  --config examples/configs/reacher_7dof-v0.yml \
  --controllers mppi --save_dir ./experiments
```

The job script reads the sweep settings from the YAML config. Set `render: false`
in that config for unattended runs.

### Replay a saved reacher trajectory

After running the robot-arm MPC example above:

```bash
MUJOCO_GL=glfw uv run python examples/visualize_trajectories.py \
  --env_name reacher_7dof-v0 --repeat 3 \
  --file "$(find experiments/reacher_7dof-v0 -name trajectories.pkl -print -quit)"
```

MPC runs save metrics and `trajectories.pkl` under `experiments/<env>/<time>/<controller>/`.
To save MP4 videos instead of opening the viewer, set `render: false` in the config
and add `--dump_vids` to the MPC command. Headless MuJoCo recording requires an EGL
or OSMesa driver and `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`, respectively.

## Optional SAC example

SAC is reinforcement learning, separate from the MPC examples above.

```bash
uv run --extra training python -m mjmpc.control.softqmpc.scripts.train_sac_stable \
  --env SimplePendulum-v0 --steps 10000 --model sac_pendulum

uv run --extra training python -m mjmpc.control.softqmpc.scripts.train_sac_stable \
  --env SimplePendulum-v0 --model sac_pendulum --evaluate
```

## Notes

- Hand, Panda, external Sawyer, point-mass, and lowercase cartpole/acrobot configs
  need separately installed Gymnasium task implementations. The bundled
  `reacher_7dof-v0` used above needs no external task package.
- Old `mujoco_py` trajectory snapshots need conversion. iLQR remains an unfinished
  prototype; use the controllers listed above.

## Development

```bash
uv run pytest
uv run ruff check .
uv build
```

## Citation

Mohak Bhardwaj, Sanjiban Choudhury, and Byron Boots.
[Blending MPC & Value Function Approximation for Efficient Reinforcement Learning](https://openreview.net/forum?id=RqCC_00Bg7V), ICLR 2021.
