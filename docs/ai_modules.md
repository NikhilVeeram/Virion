# AI, SLAM, and Autonomy Modules

This document breaks down each AI-oriented requirement so every component can be
implemented and tested independently.

## SLAM stack (underwater focus)

1. **Front-end sensors**
   - FPGA delivers optical-flow vectors + pre-processed grayscale frames.
   - IMU + depth measurements streamed via micro-ROS.
   - Optional sonar intensity scans processed at 10 Hz.
2. **Feature extraction**
   - FAST corners + BRIEF descriptors (Vitis HLS core) ensure resilience to turbidity.
   - Keyframe selection triggered when average parallax >10° or travel >0.5 m.
3. **State estimation**
   - Multi-sensor EKF predicts pose using IMU bias-corrected data and updates with
     camera reprojection + sonar range.
   - Outlier rejection via Mahalanobis gating.
4. **Back-end optimization**
   - GTSAM factor graph with nodes for pose, velocity, IMU bias, and sonar offsets.
   - Loop closures discovered through bag-of-words (DBoW2) on FPGA descriptors.
5. **Outputs**
   - `/map`, `/odom`, `/tf`, and `/pointcloud` topics for downstream planners.
   - `slam_status` diagnostic to inform the autonomy manager.

## Curiosity engine (GAN + RL)

1. **GAN architecture**
   - Generator: ResNet18 encoder → ConvLSTM bottleneck → transposed-conv decoder.
   - Discriminator: Lightweight PatchGAN to evaluate realism.
   - Losses: L1 reconstruction + adversarial + perceptual.
2. **Reward shaping**
   - Curiosity reward = λ₁ * |observation − prediction| + λ₂ * PID residual magnitude.
   - Penalize redundant revisits using intrinsic count-based bonus.
3. **RL policy**
   - PPO with 4-layer MLP actor-critic conditioned on SLAM state, PID commands, and
     GAN novelty signal.
   - Action space: target attitude offsets (±15°) and thrust deltas; clipped to stay
     within PID actuator limits.
4. **Training regimen**
   - Train in Gazebo underwater world for 5e6 steps.
   - Use domain randomization (turbidity, current strength, lighting) to enhance
     robustness.
   - Export TorchScript policy for deployment on-device.

## Object detection / path planning (terrestrial mode)

1. **Detector**
   - YOLOv5n backbone/head, input 640×640, trained on mixed real/synthetic terrain
     dataset with classes {obstacle, ramp, target, hazard}.
   - Apply quantization-aware training, export to ONNX, then convert to TensorRT INT8.
2. **Terrain classifier**
   - MobileNetV3-small model to label terrain patches (sand, mud, rock, vegetation).
   - Output used to adjust friction coefficients inside the planner.
3. **Planner**
   - D* Lite for dynamic navigation. Frontier-based exploration toggled when no
     mission waypoints remain.
   - MPC module solves 2 s horizon (10 steps) with thrust/attitude constraints to
     smooth commands.

## Energy-aware autonomy

1. Battery model uses EKF with inputs: current draw (from ESC telemetry), voltage,
   and temperature.
2. Mission manager monitors SoC and triggers `return_to_base` when <15%.
3. Docking logic uses fiducial markers detected by YOLO + SLAM pose graph to align
   with charger.

## Interfaces and topics

| Module | Publish | Subscribe |
| --- | --- | --- |
| PID Controller | `/virion/thruster_cmd`, `/virion/state_estimate` | `/virion/setpoint` |
| SLAM Front-end | `/virion/keypoints`, `/virion/bag` | `/virion/image_proc`, `/virion/imu_raw` |
| SLAM Back-end | `/map`, `/odom` | `/virion/keypoints`, `/virion/bag` |
| Curiosity Engine | `/virion/curiosity_reward`, `/virion/exploration_cmd` | `/virion/image_proc`, `/map`, `/virion/pid_residual` |
| Object Detection | `/virion/detections` | `/virion/image_rgb` |
| Planner | `/virion/setpoint` | `/map`, `/odom`, `/virion/detections`, `/virion/curiosity_reward`, `/virion/battery_state` |
| Autonomy Manager | `/mission/status` | `/virion/battery_state`, `/mission/objectives` |

This breakdown ensures every PRD requirement (PID stability, FPGA preprocessing, SLAM,
curiosity-driven RL, object detection, A*/D* planning, energy safety) is accounted for.
