# OmniClone

**OmniClone** is a whole-body humanoid teleoperation system designed for affordable, reproducible, and versatile robot control across diverse operator body shapes and task scenarios.

[[Paper](https://omniclone.github.io/resources/OmniClone.pdf)] [[Project Page](https://omniclone.github.io/)]

![Teaser](teaser.png)

## Overview

OmniClone takes a systematic perspective to develop an affordable, robust, and versatile whole-body teleoperation system. The system provides:

- **Bias mitigation** — identification and correction of distortions arising from retargeting and hardware fluctuations
- **Transformer-based whole-body control** — robust, affordable humanoid control with integrated optimizations
- **Multi-operator support** — stable operation across operator heights ranging from 147 cm to 194 cm
- **High-fidelity dexterous manipulation** — long-horizon task execution across diverse domestic environments
- **Autonomy policy training** — expert trajectory generation enabling downstream robot learning
- **Zero-shot motion execution** — diverse motion generation without task-specific training


## Operation Modes

| Mode | Input | Output | Main scripts |
| --- | --- | --- | --- |
| Standard teleoperation | Nokov or Pico XR | Whole-body G1 control | `lowcmd_publisher.py`, `nokov_to_robot.py` or `pico_to_robot.py`, `server.py` |
| Full teleoperation + recording | Pico XR + optional head / hands | Whole-body control, ego-view, episode recording | `lowcmd_publisher.py`, `pico_to_robot_whand.py`, `server_head.py` |

## Table of Contents

- [Release Status](#release-status)
- [Requirements](#requirements)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Optional Onboard Setup](#optional-onboard-setup)
- [Network Setup](#network-setup)
- [Run OmniClone](#run-omniclone)
- [Documentation and Related Projects](#documentation-and-related-projects)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Release Status

| Date | Release |
| --- | --- |
| TBD | Full teleoperation codebase and checkpoint |
| TBD | Full teleoperation codebase on-board version |
| TBD | Pre-collected demonstration datasets |
| TBD | Training code |

## Requirements

### Software Platform (Host PC)

| Component | Requirement |
| --- | --- |
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | 12.4 with cuDNN 9.1 |

### Core Hardware

| Device | Purpose | Notes |
| --- | --- | --- |
| Host PC | Runs motion retargeting, policy inference, and control stack | NVIDIA GPU recommended |
| Unitree G1 | Target humanoid robot | Controlled through Unitree SDK 2 (DDS) and ROS 2 |
| Pico Ultra XR headset | XR body tracking for teleoperation | Used by `pico_to_robot.py` / `pico_to_robot_whand.py` via GMR's `XrClient` |
| Nokov / OptiTrack system | Optical motion capture alternative to Pico XR | Used by `nokov_to_robot.py` |

### Optional Peripherals

| Device | Purpose | Notes |
| --- | --- | --- |
| Intel RealSense D455 | Ego-view RGB camera mounted on the robot head | Used for image streaming and data collection |
| Inspire dexterous hands | Hand end-effectors on the robot | Controlled through DDS topics |
| Head servo stack | 2-DoF robot head control | Required only for the full head-enabled setup |

## Repository Layout

| Path | Purpose |
| --- | --- |
| `GMR/` | General Motion Retargeting package and assets |
| `Sim2Everything/` | Deployment utilities for sim-to-real / sim-to-sim control |
| `assets/` | Robot assets, scenes, and hand retargeting configs |
| `models/` | Policy checkpoints |
| `sdks/` | Bundled third-party SDK wheels such as Nokov |
| `vla_data_collection/` | Ego-view streaming and episode collection utilities |
| `processing/` | Recording post-processing scripts |
| `docker/` | Release artifacts such as `omniclone-release.tar` |

## Environment Setup

> These steps assume Ubuntu, an NVIDIA GPU, and CUDA 12.4 are already available.
>
> Run all commands from the repository root unless noted otherwise.

### 1. Create the Conda Environment

```bash
conda create -n omniclone python=3.10 -y
conda activate omniclone

# Fix potential Mujoco / OpenGL rendering issues
conda install -c conda-forge libstdcxx-ng -y
```

### 2. Install Local Packages

#### GMR

```bash
cd GMR
pip install -e .
cd ..
```

> For dataset and body-model preparation details, see [GMR/README.md](./GMR/README.md).

#### Sim2Everything

```bash
cd Sim2Everything
pip install -e .
cd ..
```

> If `Sim2Everything/` is not already present in your workspace, clone it from <https://github.com/Yutang-Lin/Sim2Everything> first.

### 3. Install Common Python Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install opencv-python pillow scipy numpy matplotlib tqdm
pip install unitree-sdk2py dex-retargeting
pip install cyclonedds pyserial pyngrok pyyaml rich pyqt5 pyqtgraph
```

### 4. Install Feature-Specific Components

#### Nokov MoCap SDK

The Nokov wheel is bundled in `sdks/`:

```bash
pip install sdks/nokovpy-3.0.1-py3-none-any.whl
```

#### Pinocchio (Optional)

```bash
conda install pinocchio -c conda-forge
```

#### Pico XR SDK

Headset client:

- Install the Pico client from the [XRoboToolkit Unity client release page](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/).

PC service:

```bash
sudo dpkg -i XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb
```

Start `xrobotoolkit-pc-service` before launching teleoperation.

Python bindings:

```bash
git clone https://github.com/YanjieZe/XRoboToolkit-PC-Service-Pybind.git
cd XRoboToolkit-PC-Service-Pybind

mkdir -p tmp
cd tmp
git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git
cd XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK
bash build.sh
cd ../../../..

mkdir -p lib include
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h include/
cp -r tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann include/nlohmann/
cp tmp/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so lib/

conda install -c conda-forge pybind11
pip uninstall -y xrobotoolkit_sdk
python setup.py install
```

Runtime packages:

```bash
pip install xrobotoolkit-sdk xrobotoolkit-teleop
```

## Optional Onboard Setup

This section only applies to the full robot configuration with a 2-DoF head, ego-view camera, and Inspire hands.

| Component | External reference | Purpose |
| --- | --- | --- |
| RealSense D455 | [librealsense](https://github.com/realsenseai/librealsense) | Camera runtime |
| Inspire hand service | [inspire_hand_ws](https://github.com/le-ma-3308/inspire_hand_ws) | Hand driver / control service |
| Head servo service | [g1_comp_servo_service](https://github.com/le-ma-3308/g1_comp_servo_service) | 2-DoF head motion control |
| Onboard image sender | [XRoboToolkit-Orin-Video-Sender](https://github.com/le-ma-3308/XRoboToolkit-Orin-Video-Sender) | Ego-view streaming to Pico and data collection|

If those services are maintained outside this repository, make sure they are already running before you launch `server_head.py`.

## Network Setup

All machines should be on the same local network.

Quick checklist:

- Keep the host PC, robot, and headset / MoCap server on the same subnet.
- Run `server.py` or `server_head.py` on one clearly designated host.
- Point `--client_ip` to that host, or use `127.0.0.1` if everything runs on the same machine.

Disable the firewall on the Ubuntu host if needed:

```bash
sudo ufw disable
```

### Network Topology

```text
┌─────────────────────────────────────────────────────────────────────┐
│                           Local Network                             │
│                                                                     │
│  ┌──────────────┐    Wi-Fi / UDP    ┌─────────────────────────────┐ │
│  │ Pico XR      │ ───────────────── │ Host PC                     │ │
│  │ Headset      │   body tracking   │ motion retargeting          │ │
│  └──────────────┘                   │ policy inference            │ │
│                                     │ low-level control           │ │
│  ┌──────────────┐    SDK / LAN      │                             │ │
│  │ Nokov MoCap  │ ───────────────── │ `pico_to_robot.py`          │ │
│  │ Server       │                   │ `nokov_to_robot.py`         │ │
│  └──────────────┘                   │ `server.py`                 | │
│                                     │ `server_head.py`            │ │
│                                     │ `lowcmd_publisher.py`       │ │
│                                     └────────────┬────────────────┘ │
│                                                  │ DDS / ROS 2      │
│                                     ┌────────────▼────────────────┐ │
│                                     │ Unitree G1 Robot            │ │
│                                     │ optional D455 / hands / head│ │
│                                     └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```


## Run OmniClone

> All commands below assume the repository root as the working directory.

### Safety Checklist

Before enabling policy execution:

- Make sure the robot is physically supported and has free space around it.
- Source the required ROS 2 environment before running `lowcmd_publisher.py`.
- Verify that the correct policy checkpoint exists in `models/`.
- Confirm that your motion-input source is streaming normally.

> **Safety-critical:** After starting `server.py` and before pressing `L1` to begin policy inference, press `R2` once to normalize heading alignment. Skipping this step may cause a sudden turn at startup.

### A. Standard Teleoperation (No Head / No Hands)

This is the default real-time setup. Open the following processes in separate terminals on the host PC.

#### Host terminal 1: low-level motor command publisher

```bash
python lowcmd_publisher.py
```

#### Host terminals 2-3: choose one motion-input backend

Nokov MoCap:

```bash
python nokov_to_robot.py \
    --robot unitree_g1 \
    --mocap_ip 192.168.123.101 \
    --client_ip 127.0.0.1 \
    --client_port 7002
```

```bash
# policy server
python server.py \
    --server_ip 0.0.0.0 \
    --server_port 7002
```

Pico XR:

```bash
python pico_to_robot.py \
    --robot unitree_g1 \
    --client_ip 127.0.0.1 \
    --client_port 7002
```

```bash
# policy server
python server.py \
    --server_ip 0.0.0.0 \
    --server_port 7002
```

`server.py` will move the G1 toward its default pose before policy execution begins.

> **Safety-critical:** After starting `server.py` and before pressing `L1` to begin policy inference, press `R2` once to normalize heading alignment. Skipping this step may cause a sudden turn at startup.

### B. Full Teleoperation + Episode Recording (Pico Only)

Use this mode when you need:

- Pico-based whole-body teleoperation
- optional hand control
- ego-view image collection
- episode recording for downstream VLA workflows

Recommended launch order:

#### Onboard terminal 1: Realsense image stream
```bash
cd XRoboToolkit-Orin-Video-Sender
./open_image_sender.sh
```

#### Onboard terminal 2: Inspire hand control server
```bash
cd inspire_hand_ws/inspire_hand_sdk/example
python Headless_driver_double.py
```

#### Host terminal 1: low-level motor publisher

```bash
python lowcmd_publisher.py
```

#### Host terminal 2: Pico retargeting bridge

```bash
python pico_to_robot_whand.py \
    --robot unitree_g1 \
    --use_hand \
    --client_ip 127.0.0.1 \
    --client_port 7002
```

#### Host terminal 3: head-enabled server

```bash
python server_head.py \
    --server_ip 0.0.0.0 \
    --server_port 7002
```

### Gamepad Controls

| Button | Action |
| --- | --- |
| `L1` | Start policy execution |
| `L2` | Emergency stop |
| `R1` | Reset observation history |
| `R2` | Normalize motion / heading |
| `A` | Start episode recording (`server_head.py`) |
| `B` | Save the current episode (`server_head.py`) |

`server_head.py` will move the G1 toward its default pose before policy execution begins.
> **Safety-critical:** After starting `server_head.py` and before pressing `L1` to begin policy inference, press `R2` once to normalize heading alignment. Skipping this step may cause a sudden turn at startup.

### Pico Controller Hand Controls

> These bindings refer to the Pico controller, not the gamepad controls listed above.

| Button | Behavior |
| --- | --- |
| `X` | Press and hold to close the left hand; release to reopen it |
| `A` | Press and hold to close the right hand; release to reopen it |

## Documentation and Related Projects

Internal documentation:

- [GMR/README.md](./GMR/README.md) for retargeting assets and dataset preparation
- [Sim2Everything/README.md](./Sim2Everything/README.md) for deployment utility details

Related projects:

- [Sim2Everything](https://github.com/Yutang-Lin/Sim2Everything): Sim-to-real transfer utilities for humanoid robots
- [GMR](https://github.com/YanjieZe/GMR): Real-time human-to-humanoid motion retargeting on CPU
- [Gen2Humanoid](https://github.com/RavenLeeANU/Gen2Humanoid): Text-to-motion pipeline for humanoid robots
- [Clone](https://humanoid-clone.github.io/): A related humanoid teleoperation system
- [COLA](https://yushi-du.github.io/COLA/): Proprioception-only learning for human-humanoid collaborative carrying
- [LessMimic](https://lessmimic.github.io/): Long-horizon humanoid-scene interaction with unified distance fields

## License

This project is released under the [MIT License](LICENSE).

## Citation

```bibtex
@article{omniclone2026,
  title   = {OmniClone: Engineering a Robust, All-Rounder Whole-Body Humanoid Teleoperation System},
  author  = {Yixuan Li and Le Ma and Yutang Lin and Yushi Du and Mengya Liu and Kaizhe Hu and Jieming Cui and Yixin Zhu and Wei Liang and Baoxiong Jia and Siyuan Huang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Acknowledgements

We extend our sincere gratitude to Peiyang Li, Zimeng Yuan, Zhen Chen, Chengcheng Zhang, Yang Zhang, Nian Liu, Zhidan Liu, Zihui Liu and Mulin Sui, for their invaluable help in filming this demo.

This work is supported in part by the National Key Research and Development Program of China (2025YFE0218200), the National Natural Science Foundation of China (62172043 to W.L., 62376009 to Y.Z.), the PKU-BingJi Joint Laboratory for Artificial Intelligence, the Wuhan Major Scientific and Technological Special Program (2025060902020304), the Hubei Embodied Intelligence Foundation Model Research and Development Program, and the National Comprehensive Experimental Base for Governance of Intelligent Society, Wuhan East Lake High-Tech Development Zone.
