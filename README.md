# PhysX vs. Custom SDF Discrepancy Analysis

## Overview

The aim of this project is to show the discrepancy coming out of the SDF generation process.

## Setup and Building

Open in the dev-container to get started.

### Build the C++ SDF Generator

Inside the running Docker container's shell, compile the C++ code:

```bash
# Navigate to the workspace (should be the default)
cd /workspace

# Create a build directory
mkdir build
cd build

# Run CMake to configure the project
cmake ..

# Compile the project
make
```

This will create an executable file at `/workspace/build/bin/sdf_example`.

### Generate the PhysX SDF

Run the compiled C++ executable to generate the SDF from the input STL file:

```bash
# Still inside the /workspace/build directory
./bin/sdf_example
```

This program will:
- Read `/workspace/data/watertightshape.stl`.
- Use PhysX to compute the SDF.
- Save the dimensions, bounds, and SDF distance values (in cm) to `/workspace/data/watertightshape.bin`.

## Running the Python Analysis and Visualization

After generating the PhysX SDF (`watertightshape.bin`), you can run the Python script to perform the comparison.

### Run the Python Script

Make sure you are in the `/workspace` directory inside the Docker container:

```bash
cd /workspace

# Run the Python script (assuming it's named visualize_sdf.py)
python visualize_sdf.py
```

This script will:
- Load the PhysX SDF from `data/watertightshape.bin`.
- Load the custom SDF from `data/gmr_internal_sdf.bin`.
- Load the reference point cloud from `data/pointcloud.ply`.
- Load query points from `data/collision_log.json`.
- For each query point:
  - Query the distance using the PhysX SDF interpolation.
  - Query the distance using the custom SDF interpolation.
  - Find the nearest point in the reference point cloud and calculate the Euclidean distance.
- Print a table comparing these three distance values (in cm) for each query point.
- Launch an interactive Open3D visualization window showing:
  - The reference point cloud (light gray points).
  - The query points as spheres:
    - Blue Sphere: The PhysX SDF distance is closer to the point cloud distance.
    - Red Sphere: The custom SDF distance is closer to the point cloud distance.
  - Green lines connecting each query point to its nearest neighbor in the reference point cloud.

## Interpretation

By observing the console output table and the colors of the spheres in the 3D visualization, you can analyze the discrepancy between the two SDF methods. Red spheres indicate query points where the custom SDF provides a distance value closer to the geometric ground truth (represented by the point cloud), potentially highlighting regions where the custom SDF is more accurate, especially near the object's surface.

## File Structure

```
.
├── Dockerfile                # Defines the build environment
├── CMakeLists.txt            # Build configuration for C++ code
├── sdf_discrepancy.cpp       # C++ source for PhysX SDF generation
├── visualize_sdf.py          # Python script for loading, querying, comparing, and visualizing SDFs (ASSUMED NAME)
└── data/
    ├── watertightshape.stl     # Input 3D model for PhysX
    ├── watertightshape.bin     # Output: PhysX-generated SDF (created by C++ code)
    ├── gmr_internal_sdf.bin    # Input: Pre-computed custom SDF
    ├── mergedPointcloud.ply    # Input: Reference point cloud (ground truth)
    ├── collision_log.json      # Input: Query points for analysis
```