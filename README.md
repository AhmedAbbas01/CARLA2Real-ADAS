# Carla2Real-ADAS: A Simulation-to-Real Transfer Framework for Vision-Based ADAS

![Carla2Real-ADAS](https://img.shields.io/badge/Status-Active-brightgreen)
![CARLA](https://img.shields.io/badge/Simulator-CARLA__UE5-blue)

**Authors:** Ahmed M. Abbas, Mohamed Adel  
**Affiliation:** Information Technology and Computer Science School, Nile University, Giza, Egypt

## Introduction

Advanced Driver Assistance Systems (ADAS) heavily rely on data-driven perception models (e.g., object detection). However, collecting and annotating real-world data is expensive, time-consuming, and potentially dangerous in edge-case scenarios. 

**Carla2Real-ADAS** proposes an end-to-end Simulation-to-Real (Sim2Real) pipeline designed specifically for the development and validation of camera-based ADAS using the CARLA simulator. 

This framework integrates:
1. **Synthetic Data Generation:** Generates annotated datasets with object bounding boxes and relative distance estimations (in meters) within CARLA.
2. **Photorealistic Enhancement:** Utilizes [CARLA2Real](https://github.com/stefanos50/CARLA2Real) to enhance CARLA outputs, bridging the Sim2Real appearance gap.
3. **Perception Model Training:** Trains state-of-the-art object detection models (YOLOv8, RT-DETR, Faster R-CNN) on the enhanced synthetic data.
4. **Closed-Loop ADAS Module:** Re-integrates generalized models into the CARLA simulation to perform real-time perception. Detected objects and distances are fed to an ADAS decision module to control the ego vehicle dynamically (e.g., steering, emergency braking).

## Main Contributions

- A framework based on the CARLA simulator for generating photorealistic and annotated datasets to be used for object detection and distance estimation purposes.
- A closed-loop validation framework combining perception outputs with ADAS functionalities to validate decision-making safely.
- An evaluation of a simulation-to-real workflow demonstrating cost-reduction in development while preserving robust ADAS performance.

## Repository Structure

- `carla_unreal_engine_5/`: Contains Python scripts for dataset generation, G-buffer extraction, preprocessing, and model inference inside CARLA (built from source in Unreal Engine 5).
- `Dataset/`: Directory structure housing extracted data such as `BoundingBoxes`, `EPE` buffers, raw `Frames`, `Semantic` segmentations, and final `VisulOutput`.
- `AWS_Private_EC2.sh`: A helper bash script used to spin up an AWS EC2 instance with AWS SSM port forwarding for running the simulator remotely.

## Getting Started

Please refer to the internal README at [`carla_unreal_engine_5/README.md`](carla_unreal_engine_5/README.md) for detailed instructions on launching the simulator, data generation, preprocessing, running the photorealistic inference, and visualizing the outputs.

## License

See the `LICENSE` file for further information.
