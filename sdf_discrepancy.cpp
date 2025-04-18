#include <chrono>  // Required for sleep_for
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>  // Required for numeric_limits
#include <memory>
#include <string>
#include <thread>  // Required for sleep_for
#include <vector>

// PhysX includes
#include <cooking/PxCooking.h>
#include <cooking/PxSDFDesc.h>

#include "PxPhysicsAPI.h"

// PCL includes
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>  // For fromPCLPointCloud2
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>  // For loadPolygonFilePLY (if needed elsewhere)
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>  // For pcl::PolygonMesh
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// Eigen includes
#include <Eigen/Dense>

using namespace physx;

class MeshReader {
 public:
  static bool loadMesh(const std::string& filePath, Eigen::MatrixXd& V,
                       Eigen::MatrixXi& F) {
    // Check if the file exists
    std::ifstream file(filePath);
    if (!file.good()) {
      std::cerr << "Error: File not found: " << filePath << std::endl;
      return false;
    }
    file.close();

    try {
      // Load the STL file into PCL mesh
      pcl::PolygonMesh mesh;
      if (pcl::io::loadPolygonFileSTL(filePath, mesh) == -1) {
        std::cerr << "Error: Failed to load STL file: " << filePath
                  << std::endl;
        return false;
      }

      // Extract the point cloud
      pcl::PointCloud<pcl::PointXYZ> cloud;
      pcl::fromPCLPointCloud2(mesh.cloud, cloud);

      // Create vertices matrix
      V.resize(cloud.size(), 3);
      for (size_t i = 0; i < cloud.size(); ++i) {
        V(i, 0) = cloud.points[i].x;
        V(i, 1) = cloud.points[i].y;
        V(i, 2) = cloud.points[i].z;
      }

      // Create faces matrix
      F.resize(mesh.polygons.size(), 3);
      for (size_t i = 0; i < mesh.polygons.size(); ++i) {
        if (mesh.polygons[i].vertices.size() != 3) {
          std::cerr << "Warning: Non-triangular face found at index " << i
                    << std::endl;
          continue;
        }
        F(i, 0) = mesh.polygons[i].vertices[0];
        F(i, 1) = mesh.polygons[i].vertices[1];
        F(i, 2) = mesh.polygons[i].vertices[2];
      }

      return true;
    } catch (const std::exception& e) {
      std::cerr << "Error: Exception while loading STL file: " << e.what()
                << std::endl;
      return false;
    } catch (...) {
      std::cerr << "Error: Unknown exception while loading STL file."
                << std::endl;
      return false;
    }
  }
};

// Note: These might have internal equivalents in PhysX, but are needed
// for the provided PxSDFSampleImpl function signature.
namespace Interpolation {
inline PxU32 PxSDFIdx(PxU32 x, PxU32 y, PxU32 z, PxU32 dimX, PxU32 dimY) {
  return x + y * dimX + z * dimX * dimY;
}

inline PxReal PxTriLerp(PxReal v000, PxReal v100, PxReal v010, PxReal v110,
                        PxReal v001, PxReal v101, PxReal v011, PxReal v111,
                        PxReal x, PxReal y, PxReal z) {
  PxReal oneMinusX = 1.0f - x;
  PxReal oneMinusY = 1.0f - y;
  PxReal oneMinusZ = 1.0f - z;

  PxReal xy00 = oneMinusX * oneMinusY;
  PxReal xy10 = x * oneMinusY;
  PxReal xy01 = oneMinusX * y;
  PxReal xy11 = x * y;

  PxReal c0 = v000 * xy00 + v100 * xy10 + v010 * xy01 + v110 * xy11;
  PxReal c1 = v001 * xy00 + v101 * xy10 + v011 * xy01 + v111 * xy11;

  return c0 * oneMinusZ + c1 * z;
}
}  // namespace Interpolation

// SDF Data structure
struct SdfInfoFloat {
  PxU32 numX, numY, numZ;
  PxVec3 boundsLower, boundsUpper;
  std::vector<float> sdfArray;
};

// PhysX Foundation and related globals
PxDefaultAllocator gAllocator;
PxDefaultErrorCallback gErrorCallback;
PxFoundation* gFoundation = nullptr;
PxPhysics* gPhysics = nullptr;

// Main SDF class (Error handling simplified)
class SDFExample {
 private:
  std::shared_ptr<SdfInfoFloat> m_sdf;
  float m_sdf_spacing_ = 0.01f;
  PxU32 m_subgridsize_ = 0;
  PxU32 m_numThreadsForSdfConstruction_ = 20;
  // PxU32 m_bitsPerSubgridPixel_ = 5; // This is not used directly now,
  // setupSDFDesc uses enum

 public:
  SDFExample() {
    // Initialize PhysX (Simplified error handling: exit on failure)
    gFoundation =
        PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);
    if (!gFoundation) {
      std::cerr << "Fatal Error: Failed to create PxFoundation. Exiting."
                << std::endl;
      exit(1);
    }

    PxTolerancesScale scale;
    gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, scale);
    if (!gPhysics) {
      std::cerr << "Fatal Error: Failed to create PxPhysics. Exiting."
                << std::endl;
      if (gFoundation) gFoundation->release();
      exit(1);
    }
  }

  ~SDFExample() {
    if (gPhysics) gPhysics->release();
    if (gFoundation) gFoundation->release();
  }

  std::vector<PxVec3> createVertices(const Eigen::MatrixXd& V) {
    std::vector<PxVec3> vertices;
    vertices.reserve(V.rows());
    for (int i = 0; i < V.rows(); i++) {
      vertices.emplace_back(PxVec3(V(i, 0), V(i, 1), V(i, 2)));
    }
    return vertices;
  }

  std::vector<PxU32> createIndices(const Eigen::MatrixXi& F) {
    std::vector<PxU32> indices;
    indices.reserve(F.rows() * 3);
    for (int i = 0; i < F.rows(); i++) {
      indices.push_back(F(i, 0));
      indices.push_back(F(i, 1));
      indices.push_back(F(i, 2));
    }
    return indices;
  }

  PxSDFDesc setupSDFDesc() {
    PxSDFDesc sdfDesc;
    sdfDesc.spacing = m_sdf_spacing_;
    sdfDesc.subgridSize = m_subgridsize_;
    sdfDesc.numThreadsForSdfConstruction = m_numThreadsForSdfConstruction_;
    sdfDesc.bitsPerSubgridPixel = PxSdfBitsPerSubgridPixel::e8_BIT_PER_PIXEL;
    return sdfDesc;
  }

  PxTriangleMeshDesc createTriangleMeshDesc(const std::vector<PxVec3>& vertices,
                                            const std::vector<PxU32>& indices,
                                            PxSDFDesc& sdfDesc) {
    PxTriangleMeshDesc meshDesc;
    meshDesc.points.count = vertices.size();
    meshDesc.points.stride = sizeof(PxVec3);
    meshDesc.points.data = vertices.data();
    meshDesc.triangles.count = indices.size() / 3;
    meshDesc.triangles.stride = 3 * sizeof(PxU32);
    meshDesc.triangles.data = indices.data();
    meshDesc.sdfDesc = &sdfDesc;  // Link the SDF descriptor
    return meshDesc;
  }

  PxTriangleMesh* generateSDF(const std::string& stl_file_path) {
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;

    // Check if the STL file path is empty
    if (stl_file_path.empty()) {
      throw std::runtime_error("fail");
    }

    // Load the mesh
    MeshReader::loadMesh(stl_file_path, V, F);
    std::vector<PxVec3> vertices;
    std::vector<PxU32> indices;

    // Create vertices and indices from the loaded mesh
    vertices = createVertices(V);
    indices = createIndices(F);

    // Check if vertices or indices are empty
    if (vertices.empty() || indices.empty()) {
      throw std::runtime_error("fail");
    }

    PxSDFDesc sdfDesc;
    PxTriangleMeshDesc meshDesc;

    try {
      // Set up the SDF descriptor and triangle mesh descriptor
      sdfDesc = setupSDFDesc();
      meshDesc = createTriangleMeshDesc(vertices, indices, sdfDesc);

      // Validate mesh descriptor
      if (!meshDesc.isValid()) {
        throw std::runtime_error("fail");
      }
    } catch (const std::exception& e) {
      throw std::runtime_error("fail");
    }

    PxDefaultMemoryOutputStream writeBuffer;
    PxTriangleMeshCookingResult::Enum result;
    PxTolerancesScale scale;
    PxCookingParams params(scale);

    try {
      // Attempt to cook the triangle mesh
      bool status = PxCookTriangleMesh(params, meshDesc, writeBuffer, &result);

      // Check if the cooking process was successful
      if (!status || result != PxTriangleMeshCookingResult::eSUCCESS) {
        throw std::runtime_error("fail");
      }

      // Create triangle mesh from the cooked data
      PxDefaultMemoryInputData readBuffer(writeBuffer.getData(),
                                          writeBuffer.getSize());
      PxTriangleMesh* triangleMesh = gPhysics->createTriangleMesh(readBuffer);

      // Ensure the created triangle mesh is valid
      if (!triangleMesh) {
        throw std::runtime_error("fail");
      }

      return triangleMesh;
    } catch (const std::exception& e) {
      throw std::runtime_error("fail");
    }
  }

  bool extractAndSaveFloatSDF(const PxTriangleMesh* mesh,
                              const std::string& filename) {
    if (!mesh) {
      std::cerr << "Error: Input mesh is null." << std::endl;
      return false;
    }

    // Check if SDF data exists using the direct data pointer
    const PxReal* sdfDataPtr = mesh->getSDF();
    if (!sdfDataPtr) {
      std::cerr << "Error: SDF data not available in the provided mesh "
                   "(getSDF() returned null)."
                << std::endl;
      return false;
    }

    // Get dimensions directly using getSDFDimensions
    PxU32 numX, numY, numZ;
    mesh->getSDFDimensions(numX, numY, numZ);
    // print the dimensions:
    std::cout << "Sdf Dimensionis: " << numX * numY * numZ << std::endl;

    if (numX == 0 || numY == 0 || numZ == 0) {
      std::cerr << "Error: Invalid dimensions returned by mesh getters."
                << std::endl;
      return false;
    }

    // Calculate the expected number of float elements
    std::size_t numElements = static_cast<std::size_t>(numX) * numY * numZ;

    // Create a vector for the SDF data
    std::vector<float> sdfArray;

    try {
      // Resize the vector and copy the data
      sdfArray.resize(numElements);

      // Copy data and convert units if needed
      for (size_t i = 0; i < numElements; ++i) {
        sdfArray[i] =
            static_cast<float>(sdfDataPtr[i] * 100.0f);  // Convert to cm
      }
    } catch (const std::exception& e) {
      std::cerr << "Error copying SDF data: " << e.what() << std::endl;
      return false;
    }

    // Get mesh bounds
    PxBounds3 bounds = mesh->getLocalBounds();

    // Save the data to a binary file
    std::ofstream fileStream(filename, std::ios::binary);
    if (!fileStream.is_open()) {
      std::cerr << "Error: Failed to open file for writing: " << filename
                << std::endl;
      return false;
    }

    // Write the mesh dimensions and bounds to the file
    fileStream.write(reinterpret_cast<const char*>(&numX), sizeof(PxU32));
    fileStream.write(reinterpret_cast<const char*>(&numY), sizeof(PxU32));
    fileStream.write(reinterpret_cast<const char*>(&numZ), sizeof(PxU32));
    fileStream.write(reinterpret_cast<const char*>(&bounds.minimum),
                     sizeof(PxVec3));
    fileStream.write(reinterpret_cast<const char*>(&bounds.maximum),
                     sizeof(PxVec3));

    // Write the SDF data
    fileStream.write(reinterpret_cast<const char*>(sdfArray.data()),
                     sdfArray.size() * sizeof(float));

    if (fileStream.fail()) {
      std::cerr << "Error: Failed writing SDF data to file: " << filename
                << std::endl;
      return false;
    }

    // Close the file
    fileStream.close();
    return true;
  }
};

int main(int argc, char** argv) {
  // Hard-coded file paths
  std::string stlFile =
      "/workspaces/minimally_rep_example/data/watertightshape.stl";
  std::string sdfFile =
      "/workspaces/minimally_rep_example/data/watertightshape.bin";
  std::string pointCloudFile =
      "/workspaces/minimally_rep_example/data/watertightshape.stl";

  SDFExample sdfExample;  // Initialize PhysX

  // Generation steps
  std::cout << "\n--- Generating SDF ---" << std::endl;
  std::cout << "Input STL file: " << stlFile << std::endl;
  std::cout << "Output SDF file: " << sdfFile << std::endl;

  // Generate SDF from the STL file
  PxTriangleMesh* triangleMesh = sdfExample.generateSDF(stlFile);

  // Extract and save SDF data in one step
  if (sdfExample.extractAndSaveFloatSDF(triangleMesh, sdfFile)) {
    std::cout << "Successfully saved SDF to file: " << sdfFile << std::endl;
  } else {
    std::cout << "Failed to extract and save SDF." << std::endl;
  }

  // Clean up PhysX mesh object
  triangleMesh->release();

  return 0;
}