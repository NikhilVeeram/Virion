# Virion Architecture

This document enumerates the full software + hardware stack required to realize the
Virion autonomous exploration platform while remaining compatible with the most common
student hardware and a 32 GB VRAM workstation.

## Hardware selection

| Subsystem | Recommendation | Rationale |
| --- | --- | --- |
| Microcontroller | **ESP32-S3 (dual-core Xtensa, 240 MHz, 512 KB SRAM, 2.4 GHz Wi-Fi)** | Low cost (<$15), rich timer/PWM peripherals, dual cores allow separation of sensor IO and PID loop. |
| FPGA | **Digilent Arty A7-35T (Xilinx Artix-7)** | Widely used in labs (~$219), abundant BRAM/DSPs for optical flow and SLAM front-end, well supported by open toolchains (Vivado / SymbiFlow / Verilator). |
| Companion Computer | **M2 Max MacBook Pro (32 GB unified memory)** | Runs ROS 2, Gazebo, PyTorch, and TensorRT for real-time inference while staying within the provided compute constraints. |
| Motor drivers | Off-the-shelf 30–40 A bidirectional ESCs driven via PWM from ESP32-S3. |
| Sensors | 9-DoF IMU (BNO085), barometric depth sensor (MS5837), HD+IR global-shutter camera (e.g., Arducam OV9782 + IR filter), optional sonar (Blue Robotics Ping). |

## Control domain responsibilities

1. **Sensor acquisition (ESP32-S3 core 0):** Poll IMU over SPI at 500 Hz, depth sensor at 100 Hz, thruster RPM feedback via capture timers.
2. **Position PID loop (ESP32-S3 core 1):**
   - Run at 1 kHz with watchdog ensuring <10 ms latency.
   - PID tuned per axis (roll/pitch/yaw/heave). Gains stored in flash and exposed via UART for updates.
   - PWM outputs generated with MCPWM peripheral, 400 Hz center-aligned waveform to ESCs.
3. **Failsafe logic:** If communication to autonomy computer is lost for >2 s, hold last stable attitude and slowly surface.

## FPGA domain responsibilities

| Module | Function |
| --- | --- |
| Optical flow accelerator | Lucas–Kanade kernel on incoming camera stream (up to 640×480@90 FPS). Output packed into shared memory via AXI. |
| Frame conditioning | Downsampling, RGB→YUV, CLAHE, and temporal denoising for low-light underwater scenes. |
| SLAM front-end | FAST corner detection + brief descriptor extraction feeding into ROS 2 SLAM back-end running on the companion computer. |
| Sensor fusion helper | Fixed-point preprocessing for IMU + depth data (bias removal, Kalman prediction) with deterministic latency. |
| Object detection pre-processor | Letterboxing and quantization for YOLOv5n/tiny. |

The Arty A7 communicates with the ESP32-S3 through SPI (control) and high-speed
LVDS pairs (camera data). Processed data is streamed over USB or Wi-Fi to ROS 2 nodes.

## AI + autonomy domain responsibilities

* **SLAM:** Multi-sensor EKF front-end fused with FPGA optical flow. Back-end uses GTSAM
  factor graphs and runs on ROS 2 (Humble). Keyframes stored in SQLite-based map cache.
* **Curiosity Engine:**
  - GAN generator predicts next observation (ResNet18 encoder + lightweight decoder).
  - Discriminator residual used as novelty reward.
  - PPO-based reinforcement agent chooses exploratory thrust commands while respecting PID corrections.
* **Object detection:** YOLOv5n (6.1 M parameters) quantized to INT8 via TensorRT for on-land navigation.
* **Path planning:** Hybrid approach mixing D* Lite for dynamic re-planning and frontier
  search for exploration. MPC (Model Predictive Control) optionally optimizes velocity
  commands after SLAM localization.
* **Energy-aware autonomy:** Battery State-of-Charge estimated via Extended Kalman
  Filter. If SoC <15%, the autonomy manager triggers return-to-base behavior, overrides
  curiosity rewards, and reuses previously mapped safe corridors.

## Data flow summary

1. Power system energizes ESP32-S3 and FPGA; companion computer connects over USB-C.
2. Sensors push raw data to ESP32-S3 and FPGA simultaneously.
3. PID loop stabilizes thrusters and forwards normalized motion estimates to ROS 2 via micro-ROS.
4. FPGA-preprocessed camera frames feed SLAM front-end, YOLO accelerator, and GAN inputs.
5. ROS 2 nodes run SLAM back-end, curiosity policy, and path planners, then send velocity
   or pose targets back to the microcontroller.
6. Autonomy manager supervises mission objectives, battery safety, and return-to-base logic.

This architecture satisfies every module enumerated in the PRD and provides clear
ownership boundaries for firmware engineers, FPGA developers, and autonomy researchers.
