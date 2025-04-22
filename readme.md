# TorchIK

A PyTorch-based Inverse Kinematics (IK) solver that offers:
- **Fast performance** using Gauss-Newton optimization
- **Parallel computation** for batch IK solving
- **Visualization capabilities** for solutions

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the example script to see TorchIK in action with a Franka robot:

```bash
python ik_example.py
```

## Using with Your Robot

To use TorchIK with your own robot:

1. Specify the path to your URDF file when creating the `RobotModel`
2. For visualization:
   - Ensure mesh and link names are correctly aligned
   - If needed, check/modify the `load_meshes()` and `theta2mesh()` functions

## Features

- PyTorch-based implementation for GPU acceleration
- Batch processing for parallel IK solutions
- Built-in visualization with trimesh
- End-to-end differentiable forward and inverse kinematics

