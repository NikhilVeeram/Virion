# End-to-End Runbook

This guide walks through running Virion from a fresh machine to a simulated mission, including improved AI training loops inspired by community GAN practices (e.g., the MSPaint-to-terrain GAN experiment).

## 1) System prerequisites

- **OS:** Linux or macOS with Python 3.10+.
- **GPU:** CUDA-capable GPU recommended for GAN/RL training; CPU works for smoke tests.
- **Core tools:** Git, Python venv, CMake, and ROS 2 Humble with Gazebo (for full simulation).
- **Python deps:** Installed via `pip` (see step 3). Optional: install `networkx` for path-planning extras.

## 2) Clone and initialize the workspace

```bash
# From a working directory with enough disk space
git clone https://github.com/<your-org>/Virion.git
cd Virion
python -m venv .venv
source .venv/bin/activate
```

> Tip: If using Conda, replace the `venv` lines with `conda create -n virion python=3.10` and `conda activate virion`.

## 3) Install Python dependencies

```bash
pip install --upgrade pip
pip install -e .  # installs virion, numpy, torch, opencv-python
pip install networkx  # optional but recommended for planners
```

Verify the install by compiling the sources:

```bash
python -m compileall src
```

## 4) Seed data for AI modules

1. **Synthetic terrain-to-heightmap GAN data:**
   - Follow the idea from [MSPaint-to-terrain GAN](https://www.reddit.com/r/MachineLearning/comments/7dwj1q/p_fun_project_mspaint_to_terrain_map_with_gan/) to draw simple semantic masks (water, land, cliffs) and pair them with target heightmaps.
   - Export paired `mask.png` / `height.png` images into `data/gan/`.
2. **Perception + SLAM rehearsal:** Use Gazebo’s underwater world plus ROS bag playback to generate image/IMU/sonar tuples. Save into `data/slam/`.
3. **Policy rollouts:** Use the simulator (step 6) to record trajectories; store observations and rewards in `data/rl/`.

## 5) Run the simulator backbone

1. **Launch ROS 2 + Gazebo:**
   ```bash
   # sourced from your ROS 2 Humble installation
   ros2 launch gazebo_ros empty_world.launch.py
   ```
2. **Start the Virion bridge:** Hook the bridge into ROS topics using your preferred launch file (see `docs/simulation.md` for topic conventions). The bridge only needs to publish control commands and subscribe to sensor streams.
3. **Spawn sensors/actuators:** Ensure camera, IMU, depth, and sonar plugins are active and publishing to the topics listed in `docs/ai_modules.md`.

## 6) Exercise control + estimation loops

Run a dry PID + state-estimator check before adding AI:

```bash
python - <<'PY'
from virion.control.pid import PID
from virion.navigation.state_estimator import StateEstimator

pid = PID(kp=0.6, ki=0.1, kd=0.05)
estimator = StateEstimator()

state = estimator.update(position=0.0, velocity=0.0)
cmd = pid.update(setpoint=1.0, measurement=state.position, dt=0.02)
print({"state": state, "cmd": cmd})
PY
```

You should see stable, bounded outputs confirming the control loop is wired correctly.

## 7) Train the curiosity GAN (improved stability)

The GAN now includes spectral normalization, a minibatch standard-deviation layer, and feature-matching loss to tame mode collapse:

```bash
python - <<'PY'
import torch
from virion.ai.curiosity.gan import CuriosityGAN, CuriosityGANConfig

config = CuriosityGANConfig(base_channels=32, lambda_gp=10.0)
gan = CuriosityGAN(config)

real = torch.randn(4, 3, 64, 64)
noise = torch.randn(4, config.latent_dim, 1, 1)
losses = gan.compute_losses(real, noise)
print({k: float(v) for k, v in losses.items()})
PY
```

Use paired `mask/height` images (step 4) to build a `DataLoader` and iterate `compute_losses` inside a training loop. Gradient-penalty + feature-matching keep the discriminator smooth while encouraging perceptual fidelity.

## 8) Add PPO with intrinsic rewards

Wire the GAN’s novelty scores into the PPO agent:

```bash
import torch
from virion.ai.curiosity.agent import CuriosityAgent, CuriosityPolicy, PPOConfig
from virion.ai.curiosity.gan import CuriosityGAN

obs_dim, action_dim = 32, 4
policy = CuriosityPolicy(PPOConfig(obs_dim=obs_dim, action_dim=action_dim))
gan = CuriosityGAN()
agent = CuriosityAgent(policy, gan)

obs = torch.randn(8, obs_dim)
action, log_prob, value = agent.act(obs)
print(action.shape, log_prob.shape, value.shape)
```

During simulation rollouts, compute intrinsic rewards via `agent.compute_intrinsic_reward(observation_tensor)` and mix them with environment rewards before PPO updates.

## 9) Validate detection + planning

- Export a handful of simulated RGB frames to fine-tune the YOLO-like detector in `virion.ai.detection.yolo` (quantize to INT8 for laptop deployment).
- Use `networkx`-backed planners in `virion.ai.planning.planner` with the SLAM pose graph to confirm waypoint execution.

## 10) Full mission rehearsal

1. Start ROS 2 + Gazebo (step 5).
2. Launch the autonomy stack: control loop → SLAM → curiosity GAN/agent → detector → planner.
3. Monitor topics (`/map`, `/odom`, `/virion/curiosity_reward`, `/virion/detections`) and log them to a rosbag for later replay.
4. Evaluate success criteria: stable PID, closed SLAM loops, rising intrinsic reward for novel regions, safe planner outputs.

Following these steps gives you a repeatable, start-to-finish workflow that mirrors the deployment path: deterministic control first, then progressively enabling AI modules with stabilized GAN training.
