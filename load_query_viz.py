import os
# Set XDG_RUNTIME_DIR if not already set
if 'XDG_RUNTIME_DIR' not in os.environ:
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-dir'
    # Create directory if it doesn't exist
    if not os.path.exists('/tmp/runtime-dir'):
        os.makedirs('/tmp/runtime-dir', mode=0o700)
import numpy as np
import json
import struct
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree  # Import KDTree from scipy

class Vec3:
    """Simple 3D vector class to emulate PxVec3"""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def magnitude_squared(self):
        return self.x**2 + self.y**2 + self.z**2

    def maximum(self, other):
        return Vec3(
            max(self.x, other.x),
            max(self.y, other.y),
            max(self.z, other.z)
        )

    def minimum(self, other):
        return Vec3(
            min(self.x, other.x),
            min(self.y, other.y),
            min(self.z, other.z)
        )

    def __sub__(self, other):
        return Vec3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def __add__(self, other):
        return Vec3(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )

    def __mul__(self, scalar):
        return Vec3(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )

    def to_array(self):
        return np.array([self.x, self.y, self.z])


class SDFDebugInfo:
    """Class to hold debug information"""

    def __init__(self):
        self.originalPos = []
        self.clampedPos = []
        self.diffVector = []
        self.diffMagnitude = 0.0
        self.exceededTolerance = False
        self.sdfDistance = 0.0
        self.finalDistance = 0.0
        self.gridI = 0
        self.gridJ = 0
        self.gridK = 0
        self.fractionalX = 0.0
        self.fractionalY = 0.0
        self.fractionalZ = 0.0


class SdfInfoFloat:
    """Class to hold SDF information"""

    def __init__(self):
        self.numX = 0
        self.numY = 0
        self.numZ = 0
        self.boundsLower = Vec3()
        self.boundsUpper = Vec3()
        self.resolution = 0.01
        self.arraySize = 0
        self.scaleFactor = 0.0
        self.sdfArray = []
        self.sdfArrayInt16 = []


class Interpolation:
    """Static methods for interpolation"""
    @staticmethod
    def PxSDFIdx(i, j, k, dimX, dimY):
        return i + j * dimX + k * dimX * dimY

    @staticmethod
    def PxTriLerp(v000, v100, v010, v110, v001, v101, v011, v111, fx, fy, fz):
        # Bilinear interpolation on the bottom face (z=0)
        xy00 = v000 * (1.0 - fx) + v100 * fx
        xy10 = v010 * (1.0 - fx) + v110 * fx

        # Bilinear interpolation on the top face (z=1)
        xy01 = v001 * (1.0 - fx) + v101 * fx
        xy11 = v011 * (1.0 - fx) + v111 * fx

        # Linear interpolation between the two faces
        yz0 = xy00 * (1.0 - fy) + xy10 * fy
        yz1 = xy01 * (1.0 - fy) + xy11 * fy

        # Final linear interpolation in z direction
        return yz0 * (1.0 - fz) + yz1 * fz


class SDFManager:
    """Class to manage SDF loading and querying"""

    def __init__(self):
        self.standardSdf = None
        self.customSdf = None
        self.sdf_spacing = 0.0

    def load_float_sdf_array(self, filename):
        """Load a standard SDF file format"""
        sdf = SdfInfoFloat()

        try:
            with open(filename, 'rb') as file:
                # Read dimensions
                sdf.numX = struct.unpack('I', file.read(4))[0]
                sdf.numY = struct.unpack('I', file.read(4))[0]
                sdf.numZ = struct.unpack('I', file.read(4))[0]

                # Read bounds
                boundsLower = struct.unpack('fff', file.read(12))
                sdf.boundsLower = Vec3(
                    boundsLower[0], boundsLower[1], boundsLower[2])

                boundsUpper = struct.unpack('fff', file.read(12))
                sdf.boundsUpper = Vec3(
                    boundsUpper[0], boundsUpper[1], boundsUpper[2])

                # Calculate number of elements
                numElements = sdf.numX * sdf.numY * sdf.numZ

                # Read SDF data
                sdf.sdfArray = []
                for _ in range(numElements):
                    sdf.sdfArray.append(struct.unpack('f', file.read(4))[0])

                print(
                    f"Loaded SDF with dimensions {sdf.numX} x {sdf.numY} x {sdf.numZ}")
                return sdf

        except Exception as e:
            print(f"Error loading SDF file: {e}")
            return None

    def load_custom_float_sdf_array(self, filename):
        """Load a custom SDF file format with header"""
        sdf = SdfInfoFloat()

        try:
            with open(filename, 'rb') as file:
                # Read magic number
                magic = struct.unpack('I', file.read(4))[0]
                if magic != 0x53444600:  # "SDF\0" in hex
                    raise ValueError(
                        f"Invalid SDF file format (wrong magic number): {filename}")

                # Read version
                version = struct.unpack('I', file.read(4))[0]
                if version != 1:
                    print(
                        f"Warning: Reading version {version}, but code was written for version 1")

                # Read dimensions
                sdf.numX = struct.unpack('I', file.read(4))[0]
                sdf.numY = struct.unpack('I', file.read(4))[0]
                sdf.numZ = struct.unpack('I', file.read(4))[0]

                # Read bounds
                boundsLower = struct.unpack('fff', file.read(12))
                sdf.boundsLower = Vec3(
                    boundsLower[0], boundsLower[1], boundsLower[2])

                boundsUpper = struct.unpack('fff', file.read(12))
                sdf.boundsUpper = Vec3(
                    boundsUpper[0], boundsUpper[1], boundsUpper[2])

                # Read resolution
                sdf.resolution = struct.unpack('f', file.read(4))[0]

                # Read array size
                sdf.arraySize = struct.unpack('I', file.read(4))[0]

                # Read scale factor
                sdf.scaleFactor = struct.unpack('f', file.read(4))[0]

                # Validate dimensions
                expected_elements = sdf.numX * sdf.numY * sdf.numZ
                if expected_elements != sdf.arraySize:
                    print(f"Warning: Array size ({sdf.arraySize}) doesn't match dimensions "
                          f"({sdf.numX} x {sdf.numY} x {sdf.numZ} = {expected_elements})")

                # Check format type and read data
                if sdf.scaleFactor == 0.0:
                    # New format: Read directly as float32 values
                    sdf.sdfArray = []
                    for _ in range(sdf.arraySize):
                        value = struct.unpack('f', file.read(4))[0]
                        # Multiply by 100 to convert to centimeters
                        sdf.sdfArray.append(value * 100.0)

                    # For backward compatibility, also populate int16 array
                    sdf.sdfArrayInt16 = [int(val) for val in sdf.sdfArray]
                else:
                    # Old format: Read as int16 and convert to float
                    sdf.sdfArrayInt16 = []
                    for _ in range(sdf.arraySize):
                        sdf.sdfArrayInt16.append(
                            struct.unpack('h', file.read(2))[0])

                    # Convert int16 values to float using scale factor
                    sdf.sdfArray = [
                        float(val) / 100 for val in sdf.sdfArrayInt16]

                # Set spacing value from resolution
                self.sdf_spacing = sdf.resolution

                print(
                    f"Successfully loaded custom SDF with dimensions {sdf.numX} x {sdf.numY} x {sdf.numZ}, resolution: {sdf.resolution}")
                return sdf

        except Exception as e:
            print(f"Error loading custom SDF file: {e}")
            return None

    def query_sdf(self, sdf, query_point, tolerance=float('inf')):
        """Query the SDF at a specific point"""
        if sdf is None:
            raise RuntimeError(
                "The SDF is not available. Ensure that the SDF data is properly loaded.")

        # Validate SDF dimensions
        if sdf.numX <= 0 or sdf.numY <= 0 or sdf.numZ <= 0:
            raise ValueError(
                "Invalid SDF dimensions. Dimensions must be positive values.")

        # Use the resolution from the SDF file or the stored spacing
        sdf_dx = getattr(sdf, 'resolution', self.sdf_spacing)
        if sdf_dx <= 0:
            raise ValueError(
                "Invalid SDF resolution value. Resolution must be positive.")

        # Create debug info object
        debug_info = SDFDebugInfo()

        # Convert query point to Vec3
        if isinstance(query_point, (list, tuple, np.ndarray)):
            point = Vec3(query_point[0], query_point[1], query_point[2])
        else:
            point = query_point

        # Perform the SDF sampling
        return self.sdf_sample_impl(
            sdf.sdfArray, point, sdf.boundsLower, sdf.boundsUpper,
            sdf_dx, 1.0 / sdf_dx, sdf.numX, sdf.numY, sdf.numZ,
            tolerance, debug_info
        ), debug_info

    def sdf_sample_impl(self, sdf_array, local_pos, sdf_box_lower, sdf_box_higher,
                        sdf_dx, inv_sdf_dx, dim_x, dim_y, dim_z, tolerance, debug_info):
        """Implementation of SDF sampling with trilinear interpolation"""
        # Store original input position
        clamped_grid_pt = local_pos.maximum(
            sdf_box_lower).minimum(sdf_box_higher)
        diff = local_pos - clamped_grid_pt

        # For debug collection
        if debug_info:
            debug_info.originalPos = [local_pos.x, local_pos.y, local_pos.z]
            debug_info.clampedPos = [clamped_grid_pt.x,
                                     clamped_grid_pt.y, clamped_grid_pt.z]
            debug_info.diffVector = [diff.x, diff.y, diff.z]
            debug_info.diffMagnitude = diff.magnitude()

        # Check tolerance
        if diff.magnitude_squared() > tolerance * tolerance:
            if debug_info:
                debug_info.exceededTolerance = True
                debug_info.sdfDistance = float('inf')
                debug_info.finalDistance = float('inf')
            return float('inf')

        # Original SDF sampling logic
        f = (clamped_grid_pt - sdf_box_lower) * inv_sdf_dx

        i = int(f.x)
        j = int(f.y)
        k = int(f.z)

        f.x -= i
        f.y -= j
        f.z -= k

        if i >= (dim_x - 1):
            i = dim_x - 2
            clamped_grid_pt.x -= f.x * sdf_dx
            f.x = 1.0
        if j >= (dim_y - 1):
            j = dim_y - 2
            clamped_grid_pt.y -= f.y * sdf_dx
            f.y = 1.0
        if k >= (dim_z - 1):
            k = dim_z - 2
            clamped_grid_pt.z -= f.z * sdf_dx
            f.z = 1.0

        # Store grid indices and fractional values for debug
        if debug_info:
            debug_info.gridI = i
            debug_info.gridJ = j
            debug_info.gridK = k
            debug_info.fractionalX = f.x
            debug_info.fractionalY = f.y
            debug_info.fractionalZ = f.z

        # Sample the 8 surrounding points
        s000 = sdf_array[Interpolation.PxSDFIdx(i, j, k, dim_x, dim_y)]
        s100 = sdf_array[Interpolation.PxSDFIdx(i + 1, j, k, dim_x, dim_y)]
        s010 = sdf_array[Interpolation.PxSDFIdx(i, j + 1, k, dim_x, dim_y)]
        s110 = sdf_array[Interpolation.PxSDFIdx(i + 1, j + 1, k, dim_x, dim_y)]
        s001 = sdf_array[Interpolation.PxSDFIdx(i, j, k + 1, dim_x, dim_y)]
        s101 = sdf_array[Interpolation.PxSDFIdx(i + 1, j, k + 1, dim_x, dim_y)]
        s011 = sdf_array[Interpolation.PxSDFIdx(i, j + 1, k + 1, dim_x, dim_y)]
        s111 = sdf_array[Interpolation.PxSDFIdx(
            i + 1, j + 1, k + 1, dim_x, dim_y)]

        # Perform trilinear interpolation
        sdf_dist = Interpolation.PxTriLerp(
            s000, s100, s010, s110, s001, s101, s011, s111, f.x, f.y, f.z
        )

        # Apply penalty for points outside the SDF
        final_dist = sdf_dist
        if diff.magnitude_squared() > 0.0:
            final_dist += diff.magnitude() * 100

        # Store final results for debug
        if debug_info:
            debug_info.sdfDistance = sdf_dist
            debug_info.finalDistance = final_dist
            debug_info.exceededTolerance = False

        return final_dist


class PointCloudAnalyzer:
    """Class to analyze point clouds and compute distances"""

    def __init__(self, point_cloud_data):
        """Initialize with point cloud data (numpy array of 3D points)"""
        self.points = point_cloud_data
        # Build KD-tree for efficient nearest neighbor queries
        self.kdtree = KDTree(self.points)
        print(f"KD-tree built for {len(self.points)} points")

    def find_nearest_distance(self, query_point, in_cm=True):
        """Find the distance to the nearest point in the point cloud

        Args:
            query_point: The point to query (Vec3 or array-like)
            in_cm: If True, return distance in centimeters, otherwise in meters

        Returns:
            tuple: (distance, nearest_point)
        """
        if isinstance(query_point, Vec3):
            query_array = np.array(
                [query_point.x, query_point.y, query_point.z])
        else:
            query_array = np.array(query_point)

        # Query the KD-tree to find the nearest neighbor
        distance, index = self.kdtree.query(query_array)

        # Convert to cm if requested
        if in_cm:
            distance = distance * 100.0

        # Return the distance and the nearest point
        return distance, self.points[index]

def main():
    # Create SDF manager
    sdf_manager = SDFManager()

    # Load SDFs
    standard_sdf_path = 'data/watertightshape.bin'  # Update with your actual path
    custom_sdf_path = 'data/gmr_internal_sdf.bin'      # Update with your actual path

    standard_sdf = sdf_manager.load_float_sdf_array(standard_sdf_path)
    custom_sdf = sdf_manager.load_custom_float_sdf_array(custom_sdf_path)

    # Load pointcloud
    pcd = o3d.io.read_point_cloud("data/pointcloud.ply")
    points = np.asarray(pcd.points)

    # Create point cloud analyzer with KD-tree
    point_analyzer = PointCloudAnalyzer(points)

    # Load collision points from JSON
    with open('data/collision_log.json', 'r') as f:
        collision_data = json.load(f)

    # Extract collision points from the JSON structure
    query_points = []
    for collision in collision_data:
        if 'collision_point' in collision:
            point = collision['collision_point']
            query_points.append([point['x'], point['y'], point['z']])

    # Query points in both SDFs and the point cloud, then visualize
    standard_sdf_values = []
    custom_sdf_values = []
    nearest_point_distances = []
    nearest_points = []

    print(f"\n{'Point':^30} | {'Standard SDF (cm)':^15} | {'Custom SDF (cm)':^15} | {'Point Cloud (cm)':^15}")
    print('-' * 85)

    for point in query_points:
        vec_point = Vec3(point[0], point[1], point[2])

        # Query standard SDF
        std_value, std_debug = sdf_manager.query_sdf(standard_sdf, vec_point)
        standard_sdf_values.append(std_value)

        # Query custom SDF
        custom_value, custom_debug = sdf_manager.query_sdf(custom_sdf, vec_point)
        custom_sdf_values.append(custom_value)

        # Find nearest point in the point cloud (in cm)
        nearest_dist, nearest_point = point_analyzer.find_nearest_distance(point, in_cm=True)
        nearest_point_distances.append(nearest_dist)
        nearest_points.append(nearest_point)

        # Print comparison of all three distances (all in cm)
        point_str = f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
        print(f"{point_str:^30} | {std_value:^15.6f} | {custom_value:^15.6f} | {nearest_dist:^15.6f}")

    # Try interactive visualization first, fall back to saving an image
    try:
        # Create a visualization window
        vis = o3d.visualization.Visualizer()
        success = vis.create_window(window_name="Distance Comparison", width=1024, height=768)
        
        if not success:
            raise RuntimeError("Failed to create visualization window")
            
        # Add point cloud
        if len(points) > 0:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
            vis.add_geometry(point_cloud)

        # Add collision points and lines
        for i, (point, std_val, cust_val, pc_dist) in enumerate(zip(
                query_points, standard_sdf_values, custom_sdf_values, nearest_point_distances)):
            # Create sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            sphere.translate(point)
            
            # Color based on which distance is closer to point cloud distance
            std_diff = abs(std_val - pc_dist)
            cust_diff = abs(cust_val - pc_dist)
            color = [0.2, 0.2, 0.8] if std_diff < cust_diff else [0.8, 0.2, 0.2]  # Blue or Red
            sphere.paint_uniform_color(color)
            vis.add_geometry(sphere)
            
            # Add line to nearest point
            nearest_pt = nearest_points[i]
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([point, nearest_pt])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.paint_uniform_color([0.0, 0.8, 0.0])  # Green
            vis.add_geometry(line)
            
        # Try running interactive visualization
        print("\nVisualization key:")
        print("  - Blue sphere: Standard SDF distance is closer to point cloud")
        print("  - Red sphere: Custom SDF distance is closer to point cloud")
        print("  - Green line: Connection to nearest point on point cloud")
        vis.run()
        vis.destroy_window()
            
    except Exception as e:
        print(f"Visualization error: {e}")
        print("Falling back to matplotlib visualization")
        
        # Simple matplotlib fallback
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot a subset of points (max 5000)
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            display_points = points[indices]
        else:
            display_points = points
            
        ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2],
                  c='gray', s=1, alpha=0.5)
        
        # Plot query points
        for i, point in enumerate(query_points):
            std_diff = abs(standard_sdf_values[i] - nearest_point_distances[i])
            cust_diff = abs(custom_sdf_values[i] - nearest_point_distances[i])
            color = 'blue' if std_diff < cust_diff else 'red'
            ax.scatter(point[0], point[1], point[2], c=color, s=50)
            
        plt.savefig('distance_comparison_mpl.png')
        print("Visualization saved to distance_comparison_mpl.png")


if __name__ == "__main__":
    main()
