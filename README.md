# UAV Mission Control

This ROS package contains the mission controller, visual detection interface, visual servo control, trajectory optimization helpers, and launch/config files for an autonomous UAV inspection task.

## Main Contents

- `src/`: C++ mission control, planning, detection processing, filtering, and visual servo logic.
- `include/`: Public headers for the UAV mission control modules.
- `scripts/`: Python visual detection and analysis utilities.
- `config/`: Mission, detector, visual servo, landing, and planner parameters.
- `launch/`: ROS launch files for mission startup.

## Notes

Large datasets, YOLO training outputs, model weights, bag files, and generated build artifacts are intentionally excluded from Git. Keep those files in external storage or Git LFS if they need to be shared.
