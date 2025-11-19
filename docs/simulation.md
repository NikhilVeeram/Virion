# Simulation and Validation Plan

The following workflow lets you exercise **every Virion module** without physical
hardware. Each step references free/open-source tooling and runs comfortably on an
M2 Max MacBook Pro with 32 GB unified memory.

## 1. Base environment

1. Install ROS 2 Humble (Apple Silicon via Docker or Ubuntu 22.04 VM).
2. Install Gazebo Garden and `ros-gz-bridge`.
3. Install `micro-ros-agent`, `colcon`, and `ros2_control`.
4. Install Xilinx/Vivado (for bitstream synthesis) or SymbiFlow, plus **Verilator** for
   cycle-accurate FPGA simulation.
5. Install Python tooling: `conda create -n virion python=3.10 pytorch torchvision torchaudio -c pytorch -c apple` plus `onnx`, `tensorrt`, `gtsam`, `open3d`, `pytorch-lightning`.

## 2. Control/PID loop simulation

1. Launch the Gazebo world `ros2 launch virion_sim world.launch.py` (URDF includes
   buoyancy, thrusters, IMU, depth sensor, and camera). The URDF references the PRD
   flowchart and exposes PID tunables via ROS 2 parameters.
2. Run the micro-ROS agent: `micro-ros-agent udp4 --port 8888`.
3. Flash the ESP32-S3 with the simulated firmware (Zephyr RTOS + micro-ROS). If you do
   not have hardware, compile with `CONFIG_ESP_CONSOLE_USB_CDC` and run inside the
   `espressif/idf` Docker container using QEMU `idf.py qemu`. The firmware publishes
   `/virion/imu_raw`, `/virion/depth`, `/virion/thruster_cmd` topics and subscribes to
   `/virion/setpoint`.
4. Use ROS 2 rqt_plot to verify the PID loop latency (1 kHz update). Gazebo provides
   ground-truth pose for comparison.

## 3. FPGA workload simulation

1. Model the Arty A7 peripherals in Verilator using the provided HDL testbenches
   (optical flow, frame conditioning, SLAM front-end, sensor fusion helper, YOLO preproc).
2. Feed recorded camera/IMU data (see `docs/data_collection.md`) through the testbench
   using `cocotb` scripts. Verify deterministic latency and throughput >60 FPS.
3. Export synthesized IP as AXI-stream blocks and integrate inside a **LiteX** SoC to
   communicate with the microcontroller emulator.
4. Use GTKWave to inspect signal timing; ensure <5 ms cumulative latency.

## 4. SLAM + navigation simulation

1. In ROS 2, launch the navigation stack: `ros2 launch virion_slam slam_nav.launch.py`.
2. Nodes launched:
   - `slam_frontend` (subscribes to FPGA-processed frames or recorded bag files).
   - `gtsam_backend` (factor graph optimizer).
   - `kalman_fusion` (fuses IMU/depth/sonar into EKF state estimate).
   - `mpc_controller`, `dstar_planner`, `frontier_detector`.
3. Play back rosbag logs to test underwater and terrestrial modes. Validate loop closure
   by examining `/tf` and `/map`.
4. For unit testing, run `colcon test --packages-select virion_slam`.

## 5. Curiosity engine simulation

1. Use the PyTorch Lightning project in `ai/curiosity` (GAN + PPO). It consumes
   simulated observations from Gazebo using the ROS 2 Python bridge (`rclpy`).
2. Train the GAN to predict next-frame images; compute novelty reward as L1 residual.
3. Train the PPO agent with the reward signal while constraining actions to the PID
   outputs via imitation loss.
4. Evaluate by letting the agent explore a synthetic underwater cave world and logging
   coverage statistics.

## 6. Object detection + path planning simulation

1. Fine-tune YOLOv5n on the curated dataset (see data plan). Export to ONNX, then run
   TensorRT INT8 calibration using synthetic scenes.
2. Deploy the engine inside a ROS 2 node that subscribes to the Gazebo RGB camera.
3. Combine detections with D* Lite planner to avoid obstacles in terrestrial mode.

## 7. Energy + return-to-base logic

1. Simulate battery discharge curves using the `virion_energy` ROS 2 node (publishes
   `/virion/battery_state`).
2. Trigger SoC <15% events to confirm the autonomy manager transitions to return mode,
   overrides curiosity rewards, and commands docking maneuvers.

## 8. Continuous integration hooks

* `colcon test`: unit tests for ROS 2 nodes.
* `pytest ai/`: AI model regression tests (curiosity GAN, PPO policy, YOLO pipeline).
* `verilator --cc hdl/*.v --exe tests/*.cpp`: FPGA simulation regression.

Following this pipeline ensures the entire Virion stack remains simulable, testable,
and integration-ready until hardware arrives.
