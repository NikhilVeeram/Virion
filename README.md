# Virion

Virion is an end-to-end autonomy stack for an amphibious exploration robot. The goal of
this repository is to provide all software blueprints – control loops, FPGA workloads,
AI autonomy modules, and data infrastructure – so the platform can be simulated today
and later transferred to affordable student hardware.

## Repository structure

| Path | Description |
| --- | --- |
| `docs/architecture.md` | Hardware choices, subsystem responsibilities, and data flow. |
| `docs/simulation.md` | Step-by-step workflow for simulating every subsystem with open tools. |
| `docs/ai_modules.md` | SLAM, curiosity engine, detection, and planning specifications. |
| `docs/data_collection.md` | Dataset requirements, capture plans, and logging schemas. |

The documentation is intentionally actionable: each section calls out concrete tools,
configurations, and validation steps so the entire stack is “implementation ready” and
can be rehearsed in software before hardware arrives.

## Quick start

1. **Read `docs/architecture.md`.** This explains how the ESP32-S3, Digilent Arty A7
   FPGA, and companion computer collaborate, plus the strict timing requirements for
   the PID + PWM loop.
2. **Follow `docs/simulation.md`.** Use ROS 2 Humble, Gazebo, micro-ROS, Verilator,
   and PyTorch-based AI notebooks to exercise each module in isolation and together.
3. **Implement AI + autonomy per `docs/ai_modules.md`.** Models are sized to fit
   within a 32 GB VRAM budget on an Apple Silicon M2 Max MacBook Pro.
4. **Use `docs/data_collection.md`** to plan field logging, synthetic data creation,
   and dataset packaging for downstream training.

These resources collectively ensure the Virion software stack is complete, testable,
and simulable today, yet grounded in realistic educational hardware constraints.

## Source layout

The `src/virion` package now mirrors the documentation and includes executable
scaffolds for every major subsystem:

- `ai/curiosity`: GAN-based novelty model plus PPO-style agent wrappers.
- `ai/slam`: visual-inertial SLAM pipeline built on ORB features.
- `ai/detection`: Tiny YOLO-like detector sized for the laptop companion.
- `ai/planning`: grid planner compatible with A*/D* workflows.
- `control`: PID loop utilities matching the ESP32 real-time budget.
- `sensors`, `simulation`, and `data`: drivers, ROS 2 bridge helpers, and
  dataset logging shims so sensor data can stream into training code.

Each module is lightweight enough to run on commodity hardware today yet exposes
clean interfaces for swapping in firmware-specific implementations later.
