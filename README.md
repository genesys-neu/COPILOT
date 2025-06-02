# # COPILOT: Cooperative Perception using Lidar for Handoffs between Road Side Units

This repository contains the Python codes that enable vehicles to proactively select the most suitable mmWave RSUs (Road Side Units) for high-bandwidth, low-latency communication. It leverages cooperative perception between vehicles by sharing intermediate Lidar features instead of raw point clouds and fusing them intelligently using an attention-based mechanism.

## Key Features

-  Proactive handoff decision making using learned environmental features and link conditions  
-  Lightweight feature sharing instead of raw Lidar point clouds for efficient inter-vehicle messaging  
-  Attention-based spatial fusion to weigh contributions from neighboring vehicles more effectively than distance-based methods  
-  Real-world testbed evaluation using an autonomous vehicle and Talon AD7200 60GHz routers  
-  Public dataset with Lidar scans, vehicle positions, RSU connectivity, and ground-truth labels  

## 🧾 Citation

If you use **COPILOT** in your research, please cite:

```bibtex
@inproceedings{copilot2025,
  title     = {COPILOT: Cooperative Perception for Intelligent Proactive Handoffs in mmWave Vehicular Networks},
  author    = {Your Name and Collaborators},
  booktitle = {Proceedings of the ACM/IEEE Conference on Connected Vehicle Systems},
  year      = {2025}
}
