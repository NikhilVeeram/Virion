# Data Collection and Dataset Design

This plan ensures Virion gathers the multimodal data required for supervised,
self-supervised, and reinforcement learning workflows.

## Capture goals

| Modality | Purpose | Notes |
| --- | --- | --- |
| RGB + IR video (underwater) | Train SLAM, GAN, YOLO; study turbidity | Dual-camera rig with synchronized shutters. |
| IMU (9-DoF) | PID tuning, EKF fusion | 500 Hz logging with temperature compensation. |
| Depth / pressure | Navigation, return-to-base | Calibrate at multiple salinity levels. |
| Sonar / acoustic | SLAM robustness | Capture at 5–10 Hz for cave/low-visibility areas. |
| Motor thrust + ESC telemetry | MPC + energy models | Record commanded vs actual RPM. |
| Battery telemetry | Energy-aware autonomy | Voltage, current, temperature, SoC ground truth via coulomb counter. |
| Environmental metadata | Domain randomization | GPS (for surface truth), water turbidity, lighting notes. |

## Logging architecture

1. Use ROS 2 bagging with synchronized timestamps from the companion computer.
2. Microcontroller publishes raw sensor topics; FPGA streams processed video frames.
3. Store raw + processed data simultaneously to allow supervised and self-supervised
   experiments (e.g., GAN learns on raw, SLAM uses processed).
4. Compress video using H.265 intra-frame to maintain detail while saving storage.
5. Create metadata manifests (JSON) describing environment, mission objectives,
   anomalies, and labels.

## Dataset packaging

* **Train/val/test splits**: stratify by environment (pool, lake, ocean, terrestrial).
* **Annotation pipelines**:
  - Use CVAT/label-studio for object detection labels.
  - Generate SLAM ground truth with motion-capture or fiducial grids.
  - Curiosity rewards derived from GAN residuals stored alongside logs.
* **File formats**:
  - Images: `.mkv` or `.mp4` plus extracted PNG frames for labeling.
  - Sensor streams: ROS bag + CSV exports.
  - Maps: `.ply` and `.pcd` point clouds.

## Synthetic data generation

1. Use Gazebo + Blender to render underwater scenes with varying turbidity, lighting,
   and marine life.
2. Leverage NVIDIA Isaac Sim or Unreal Engine for terrestrial navigation scenes.
3. Generate corresponding sensor data (IMU, depth) via physics simulation to augment
   RL training and supervised learning when real data is scarce.

## Data management

* Store datasets on external SSDs with redundancy (RAID1 or cloud backup).
* Provide data loaders in PyTorch + ROS bag converters for rapid experimentation.
* Enforce standardized topic names (see `docs/ai_modules.md`) for compatibility across
  all training scripts.

By following this plan, the Virion team can gather high-fidelity datasets that satisfy
all downstream learning tasks and accelerate simulation-to-reality transfer.
