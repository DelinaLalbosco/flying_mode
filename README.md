# S10 Robot Simulation — Flying Mode, Navigation Mode, LiDAR & Camera

This repository contains simulation and control code for the S10 robot, including **flying mode**, **navigation mode**, and **LiDAR/camera simulation and information** using MuJoCo.

## Repository Structure

### 1. Flying Mode ✈️

The flying-mode implementation is mainly contained in:

* [`m7.py`](src/S10_sdk_deploy/interface/robot/simulation/m7.py)

`m7.py` contains the simulation/control implementation used for the flying mode.

---

### 2. Navigation Mode 🧭

The navigation-mode implementation uses the following files:

* [`s11.xml`](src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/s11.xml)
* [`m15.py`](src/S10_sdk_deploy/interface/robot/simulation/m15.py)

#### MuJoCo model

`s11.xml` contains the MuJoCo model/configuration used for the navigation-mode simulation.

#### Navigation simulation

`m15.py` contains the corresponding Python simulation/control code.

---

### 3. LiDAR and Camera 📡📷

The LiDAR and camera simulation/information is associated with:

* [`m11.py`](src/S10_sdk_deploy/interface/robot/simulation/m11.py)
* [`m12.py`](src/S10_sdk_deploy/interface/robot/simulation/m12.py)
* [`s12.xml`](src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/s12.xml)

#### `m11.py`

Contains LiDAR/camera-related simulation functionality.

#### `m12.py`

Contains additional sensor/simulation functionality.

#### `s12.xml`

Contains the corresponding MuJoCo model configuration for the sensor setup.

---

## Quick Overview

| Function             | Main files                    |
| -------------------- | ----------------------------- |
| ✈️ Flying mode       | `m7.py`                       |
| 🧭 Navigation mode   | `m15.py`, `s11.xml`           |
| 📡 LiDAR / 📷 Camera | `m11.py`, `m12.py`, `s12.xml` |

## Requirements

The project uses Python and MuJoCo. Depending on the simulation code being used, additional dependencies may be required.

A typical environment can be created with:

```bash
python3 -m venv mujoco_env
source mujoco_env/bin/activate
```

Install the required Python packages according to the project's dependency configuration.

## Running the Simulation

The exact command depends on which mode is being tested.

For example:

```bash
python3 src/S10_sdk_deploy/interface/robot/simulation/m7.py
```

for the flying-mode simulation, or:

```bash
python3 src/S10_sdk_deploy/interface/robot/simulation/m15.py
```

for the navigation-mode simulation.

Make sure the required MuJoCo model files are available before running the corresponding Python scripts.

## MuJoCo Models

The main MuJoCo models described in this repository include:

```text
src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/
├── s11.xml    # Navigation mode
└── s12.xml    # LiDAR / camera configuration
```

## Notes

The repository is under active development. File names and simulation configurations may change as the S10 robot simulation is developed further.

## Author

**DelinaLalbosco**

GitHub: [DelinaLalbosco](https://github.com/DelinaLalbosco)
