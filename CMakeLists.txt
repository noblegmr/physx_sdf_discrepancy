cmake_minimum_required(VERSION 3.10)
project(PhysXSDFExample)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# PhysX paths - Use the ENV variable like the working example
# Ensure PHYSX_PATH is set in the environment (Docker ENV does this)
set(PHYSX_INSTALL_PATH $ENV{PHYSX_PATH} CACHE PATH "Path to PhysX installation (read from ENV)")

# Check if the *CMake variable* itself is now set.
# If $ENV{PHYSX_PATH} was empty when CMake ran, PHYSX_INSTALL_PATH would be empty.
if(NOT PHYSX_INSTALL_PATH)
    message(FATAL_ERROR "Failed to set PHYSX_INSTALL_PATH from environment variable PHYSX_PATH. Is PHYSX_PATH actually set in the environment CMake is running in?")
endif()
# If the check passes, print the value CMake is using
message(STATUS "CMake using PHYSX_INSTALL_PATH: ${PHYSX_INSTALL_PATH}")


# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})
# Add includes
# *** CORRECTED PhysX Include Path ***
include_directories(${PHYSX_INSTALL_PATH}/PhysX/include)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})
# *** CORRECTED PhysX Link Path (matching working example) ***
# NOTE: Even though Dockerfile built with GCC, the working example uses clang path.
# Let's stick to the working example's path for now.
link_directories(${PHYSX_INSTALL_PATH}/PhysX/bin/linux.clang/checked)

# Add definitions
add_definitions(${PCL_DEFINITIONS})

# Executable
add_executable(sdf_example sdf_discrepancy.cpp)

# Set the output directory (optional, can keep if desired)
set_target_properties(sdf_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# *** CORRECTED Library Names (matching working example) ***
target_link_libraries(sdf_example
    ${PCL_LIBRARIES}
    PhysXCooking_static_64
    PhysXExtensions_static_64
    PhysX_static_64
    PhysXPvdSDK_static_64       # Might not be strictly needed for SDF
    PhysXCommon_static_64
    PhysXFoundation_static_64
    PhysXGpu_64
    CUDA::cuda_driver
    # Add gomp if you were using OpenMP features, otherwise likely not needed for this example
    # gomp
)

# Add a message to show the available build modes (optional)
message(STATUS "Available modes:")
message(STATUS "  --mode generate    : Only generate SDF from STL")
message(STATUS "  --mode query       : Only load and query SDF")
message(STATUS "  --mode all         : Generate, load, and query SDF (default)")