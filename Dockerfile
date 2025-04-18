FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VERSION=12.2

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    tar \
    git \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libboost-all-dev \
    libeigen3-dev \
    python3 \
    python3-pip \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA Toolkit 12.2 - Fixed to include NVIDIA GPG key
RUN apt-get update && apt-get install -y \
    gnupg \
    wget \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update && apt-get install -y \
    cuda-toolkit-${CUDA_VERSION} \
    cuda-${CUDA_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA environment variables
ENV PATH="/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}"

# Install PCL and its dependencies
RUN apt-get update && apt-get install -y \
    libpcl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup NVIDIA PhysX Library : Download and extract PhysX
RUN wget https://github.com/NVIDIA-Omniverse/PhysX/archive/refs/tags/106.1-physx-5.4.2.tar.gz && \
    tar -xzvf 106.1-physx-5.4.2.tar.gz && \
    rm 106.1-physx-5.4.2.tar.gz && \
    mv PhysX-106.1-physx-5.4.2/ PhysX/ && \
    mv PhysX /opt && \
    sed -i 's/<cmakeSwitch name="PX_BUILDSNIPPETS" value="True" comment="Generate the snippets" \/>/<cmakeSwitch name="PX_BUILDSNIPPETS" value="False" comment="Generate the snippets" \/>/' /opt/PhysX/physx/buildtools/presets/public/linux.xml

# Modify CMakeLists.txt to add -Wno-unsafe-buffer-usage to the GCC_WARNINGS
RUN cd /opt/PhysX/physx/source/compiler/cmake/linux && \
    sed -i 's/<platform targetPlatform="linux" compiler="clang" \/>/<platform targetPlatform="linux" compiler="gcc" \/>/' /opt/PhysX/physx/buildtools/presets/public/linux.xml && \
    sed -i '/-Wno-undefined-func-template/a \ \ \ \ -Wno-unsafe-buffer-usage\\' CMakeLists.txt

# Generate projects with CUDA disabled
RUN cd /opt/PhysX/physx && ./generate_projects.sh linux

# Build and install PhysX
RUN cd /opt/PhysX/physx/compiler/linux-checked/ && \
    make clean && \
    make -j$(nproc) && \
    make install

ENV PHYSX_PATH=/opt/PhysX/physx/install/linux

# Install additional libraries for SDF functionality
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libvtk9-dev \
    libflann-dev \
    libqhull-dev \
    libsuitesparse-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages required by the script
RUN pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    open3d

# Set up workspace directory
WORKDIR /workspace