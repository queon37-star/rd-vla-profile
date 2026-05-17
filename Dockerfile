ARG UBUNTU_VERSION=22.04
ARG TARGET_PLATFORM=x86_64
ARG CUDA_VERSION=12.8.1
ARG CUDA_VERSION_PATH=cu128
ARG PYTHON_VERSION=3.10
ARG UV_VERSION=0.8.15
ARG BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
ARG DEVEL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

#########################################################################
# Build image
#########################################################################

FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        wget \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        gcc \
        git \
        pkg-config \
        libegl1-mesa-dev \
        libgl1-mesa-dev \
        mesa-common-dev \
        libx11-dev \
        libxext-dev \
        libxrandr-dev \
        libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda, Python, and Python build dependencies.
ARG TARGET_PLATFORM
ARG PYTHON_VERSION
ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh"
# NOTE: Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Install uv and point to the conda python executable for uv operations.
ARG UV_VERSION
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
ENV UV_PYTHON=/opt/conda/bin/python

RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} "cmake<3.27" conda-build pyyaml numpy ipython && \
    /opt/conda/bin/conda install -y "ffmpeg>=6,<8" -c conda-forge && \
    uv pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<72.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

# Install PyTorch core ecosystem.
ARG CUDA_VERSION_PATH
ARG TORCH_VERSION=2.7.0
ARG TORCHAO_VERSION=0.13.0
ARG INSTALL_CHANNEL=whl
RUN uv pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION} torchao==${TORCHAO_VERSION} torchvision torchaudio

ENV TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0"

# Install grouped-gemm.
# NOTE: right now we need to build with CUTLASS so we can pass batch sizes on GPU.
# See https://github.com/tgale96/grouped_gemm/pull/21
ENV GROUPED_GEMM_CUTLASS="1"
ARG GROUPED_GEMM_VERSION="grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@main"
RUN uv pip install --no-build-isolation --no-cache-dir "${GROUPED_GEMM_VERSION}"

# Install flash-attn.
ARG FLASH_ATTN_VERSION=v2.8.2
RUN uv pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# Install ring-flash-attn.
ARG RING_FLASH_ATTN_VERSION=0.1.8
RUN uv pip install --no-build-isolation --no-cache-dir ring-flash-attn==${RING_FLASH_ATTN_VERSION}

# Install liger-kernel.
ARG LIGER_KERNEL_VERSION=0.6.2
RUN uv pip install --no-build-isolation --no-cache-dir liger-kernel==${LIGER_KERNEL_VERSION}

# Install torchcodec.
ARG TORCH_CODEC_VERSION=0.7
RUN uv pip install --no-cache-dir torchcodec==${TORCH_CODEC_VERSION}

# Install direct dependencies, but not source code.
COPY pyproject.toml .
RUN uv pip install --no-cache-dir . && \
    uv pip uninstall rd-vla && \
    rm -rf *

# Install libero
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ./LIBERO
RUN uv pip install --no-cache-dir ./LIBERO && \
    uv pip uninstall libero && \
    rm -rf *

# # Install webdataset and fasttext-numpy2
# RUN uv pip install --no-cache-dir webdataset fasttext-numpy2

# # Install vllm.
# ARG CUDA_VERSION_PATH
# ARG VLLM_VERSION=0.10.2
# # RUN uv pip install vllm==${VLLM_VERSION} --torch-backend=auto
# RUN uv pip install --no-cache-dir vllm==${VLLM_VERSION} --torch-backend=cu128
# # RUN uv pip install --no-cache-dir vllm==${VLLM_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH}

# # Install coco caption eval dependencies
# RUN uv pip install --no-cache-dir pycocoevalcap
# RUN /opt/conda/bin/conda install -y conda-forge::openjdk && /opt/conda/bin/conda clean -ya

# # Install lerobot
# ARG LEROBOT_VERSION=0.4.2

# RUN uv pip install --no-cache-dir \
#   --index-url https://pypi.org/simple \
#   "cmake>=3.29.0.1,<4.0.0" \
#   && cmake --version

# RUN uv pip install --no-cache-dir \
#   --index-url https://pypi.org/simple \
#   --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION_PATH} \
#   "lerobot[all]==${LEROBOT_VERSION}"

#########################################################################
# Release image
#########################################################################

FROM ${BASE_IMAGE} AS release

# Install system dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        language-pack-en \
        make \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        rsync \
        vim \
        sudo \
        unzip \
        fish \
        parallel \
        zsh \
        htop \
        tmux \
        wget \
        emacs \
        libxml2-dev \
        libjpeg-dev \
        libpng-dev \
        apt-transport-https \
        gnupg \
        jq \
        gcc \
        git \
        libgl1 \
        libegl1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
        libglu1-mesa \
        libosmesa6 \
        libosmesa6-dev \
        patchelf \
        xvfb \
        x11-utils && \
    rm -rf /var/lib/apt/lists/*

# Install uv and point to the conda python executable for uv operations.
ARG UV_VERSION
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install DOCA OFED user-space drivers
# See https://docs.nvidia.com/doca/sdk/doca-host+installation+and+upgrade/index.html
# doca-ofed-userspace ver 2.10.0 depends on mft=4.31.0-149
ENV MFT_VER=4.31.0-149
RUN wget https://www.mellanox.com/downloads/MFT/mft-${MFT_VER}-x86_64-deb.tgz && \
    tar -xzf mft-${MFT_VER}-x86_64-deb.tgz && \
    mft-${MFT_VER}-x86_64-deb/install.sh --without-kernel && \
    rm mft-${MFT_VER}-x86_64-deb.tgz

ENV DOFED_VER=2.10.0
ENV OS_VER=ubuntu2204
RUN wget https://www.mellanox.com/downloads/DOCA/DOCA_v${DOFED_VER}/host/doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    dpkg -i doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb && \
    apt-get update && apt-get -y install doca-ofed-userspace && \
    rm doca-host_${DOFED_VER}-093000-25.01-${OS_VER}_amd64.deb

# Copy conda environment.
COPY --from=build /opt/conda /opt/conda

ENV UV_PYTHON=/opt/conda/bin/python
ENV PATH=/opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# aws cli
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
 unzip awscliv2.zip && \
 sudo ./aws/install && \
 rm -rf aws

# gsutil/gcloud
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
 echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
 sudo apt-get update && sudo apt-get -y install google-cloud-cli

# Install a few additional utilities via pip
RUN uv pip install --no-cache-dir \
    gpustat \
    jupyter \
    beaker-gantry
    # oocmap

# Setup env for libero eval
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Use bash for RUN
SHELL ["/bin/bash", "-c"]

# Install Beaker CLI
RUN set -e && \
    VERSION=$(curl --header 'Accept: text/plain' --silent https://beaker.org/api/v3/release) && \
    TMP=$(mktemp -d) && \
    ARCH=$(uname -m) && \
    OS=$(uname -s) && \
    if [ "$ARCH" = "arm64" ] && [ "$OS" = "Darwin" ]; then ARCH=arm64; OS=darwin; \
    elif [ "$ARCH" = "x86_64" ] && [ "$OS" = "Darwin" ]; then ARCH=amd64; OS=darwin; \
    elif [ "$ARCH" = "x86_64" ] && [ "$OS" = "Linux" ]; then ARCH=amd64; OS=linux; \
    else echo "Unrecognized OS-Architecture combination: ARCH=$ARCH OS=$OS"; exit 1; fi && \
    PATTERN="beaker-cli-$OS-$ARCH-$VERSION.tar.gz" && \
    mkdir -p $TMP/assets && \
    URL="https://beaker.org/api/v3/release/cli?os=$OS&arch=$ARCH" && \
    HTTP_CODE=$(curl --retry 25 --retry-delay 1 --max-time 10 --retry-max-time 10 --silent --output $TMP/assets/$PATTERN --write-out "%{http_code}" $URL) && \
    if [ "$HTTP_CODE" -ne "200" ]; then echo "Failed to download: $URL"; exit 1; fi && \
    echo "Download succeeded; size is $(wc -c < $TMP/assets/$PATTERN) bytes." && \
    tar -zxf $TMP/assets/$PATTERN -C /usr/local/bin ./beaker && \
    chmod +x /usr/local/bin/beaker && \
    beaker --version

# LABEL org.opencontainers.image.source https://github.com/allenai/OLMo-core
WORKDIR /app/olmo-core
