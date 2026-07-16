FROM nvcr.io/nvidia/isaac-lab:3.0.0-beta2

# Pin uv to a specific release for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Isaac Lab 3.0.0-beta2 defaults Docker build steps to a non-root user. The dev
# image intentionally runs privileged setup as root because it installs apt
# packages, writes the Isaac Python shim under /usr/local/bin, and uses /root
# cache mounts that match docker/compose.yaml.
USER root

ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=/isaac-sim/python.sh \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv

ARG CUROBO_V2_REF=curobov2_release
ARG CUROBO_V2_VERSION=0.8.0
ARG LEROBOT_VERSION=0.5.1
ARG ROBOMIMIC_REF=v0.5.0

WORKDIR /workspace/ioailab

RUN printf '%s\n' '#!/bin/sh' 'exec /isaac-sim/python.sh "$@"' > /usr/local/bin/python && \
    chmod +x /usr/local/bin/python

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libjsoncpp-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git config --system --add safe.directory /workspace/ioailab

# Copy dependency metadata first to maximize Docker layer reuse.
COPY pyproject.toml README.md ./

# A lockfile is optional during early project setup.
COPY uv.lock* ./
COPY src ./src

# Install project dependencies into the Isaac Sim Python environment. The dev
# extra carries pytest, ruff, and ty so lint/typecheck tooling stays available
# from the Dockerfile without duplicating package pins in shell commands. If a
# uv.lock exists, use it as the source of truth; otherwise resolve from
# pyproject.toml so the image still builds during early project setup.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv export --frozen --extra dev --no-emit-project --format requirements.txt --output-file /tmp/requirements.txt; \
    else \
        uv export --extra dev --no-emit-project --format requirements.txt --output-file /tmp/requirements.txt; \
    fi && \
    uv pip install --system -r /tmp/requirements.txt

# Copy the rest of the project and install it in editable mode. The repository
# .dockerignore keeps local caches and bulky artifacts out of the build context.
COPY docs ./docs

# Install GP001 remote-control bindings through the project gp001 extra.
# The extra points to the repository-managed third-party source.
COPY third_party/remote_control_lite ./third_party/remote_control_lite
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --extra gp001 --no-emit-project --format requirements.txt --output-file /tmp/gp001-requirements.txt && \
    uv pip install --system -r /tmp/gp001-requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps -e ".[dev]"

# Install the LeRobot dataset writer without letting its broad dependency
# metadata downgrade Isaac Sim's curated NumPy, Torch, CUDA, or renderer stack.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps "lerobot==${LEROBOT_VERSION}" && \
    uv pip install --system \
        "datasets==4.8.5" \
        "huggingface-hub==1.16.1" \
        "av==15.1.0" \
        "jsonlines==4.0.0" \
        "torchcodec==0.10.0" \
        "imageio[ffmpeg]==2.37.3" \
        "draccus==0.10.0" \
        "deepdiff==8.6.2"

# The Isaac Lab base image pins robomimic without diffusion_policy configs.
# Override it in the final Isaac Sim Python environment while keeping torch,
# torchvision, and numpy from Isaac Sim, using validated HuggingFace/Diffusers
# pins. ioailab keeps the remaining robomimic / Diffusers compatibility
# adapter in package code instead of copying Docker-only patch files into
# site-packages.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        "huggingface-hub==0.36.2" \
        "diffusers==0.35.2" && \
    uv pip install --system --no-deps --upgrade \
        "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@${ROBOMIMIC_REF}"

# Install cuRobo v2 into the same Isaac Sim Python environment. Use the cu12
# runtime extra only: Isaac Sim provides torch and USD, so do not request their
# cuRobo extras here.
RUN --mount=type=cache,target=/root/.cache/uv \
    CUROBO_USE_PYBIND=0 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=${CUROBO_V2_VERSION} \
    uv pip install --system \
        "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@${CUROBO_V2_REF}"

# Isaac Sim bundles `packaging` under omni.isaac.core_archive/pip_prebundle, and
# the extscache pip_prebundle symlinks every module into it. The unpinned uv
# installs above (datasets/transformers/lerobot and friends all require
# packaging) upgrade packaging into site-packages and uninstall that bundled
# copy, deleting its directory and leaving the extscache symlinks dangling.
# Because each extension prepends its pip_prebundle to sys.path, the dangling
# link shadows the good site-packages copy and `import packaging` raises
# FileNotFoundError at startup (e.g. isaacsim.core.experimental.objects ->
# matplotlib -> packaging.version). Re-point the bundled path at the
# site-packages copy so the symlink chain resolves again.
RUN ln -sfn /isaac-sim/kit/python/lib/python3.12/site-packages/packaging \
        /isaac-sim/exts/omni.isaac.core_archive/pip_prebundle/packaging

# The deprecated omni.isaac.ml_archive extension ships its own torch built
# against NCCL 2.28 (symbol `ncclDevCommCreate`), but Isaac Sim's site-packages
# torch ships NCCL 2.27.5; both expose soname libnccl.so.2, so the 2.27.5 copy
# wins process-wide and the archive's libtorch_cuda.so fails with
# `undefined symbol: ncclDevCommCreate` at startup. IsaacLab 3.0 and ioailab
# use only the site-packages torch, never the deprecated archive, so exclude it
# from the default `./isaaclab.sh -s` experience (isaacsim.exp.full.kit) to drop
# the error without having to pass --/app/extensions/excluded on every launch.
RUN sed -i '0,/^\[settings\]$/s//[settings]\napp.extensions.excluded = ["omni.isaac.ml_archive"]/' \
        /isaac-sim/apps/isaacsim.exp.full.kit

# Install ultralytics into the default Isaac Sim Python environment for use
# in vision pipeline scripts that run outside the foundationpose conda env.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "ultralytics"

# ── FoundationPose ────────────────────────────────────────────────────────────
# All FoundationPose-specific ARGs, ENV, and install steps are grouped here at
# the bottom so changes to them never invalidate the Isaac Sim layers above.

ARG CUDA_TOOLKIT_VERSION=12-8
ARG FOUNDATIONPOSE_REF=main
ARG MICROMAMBA_VERSION=2.3.3
# Explicit arch list for CUDA extension builds during `docker build` (no GPU
# available at build time).  Override with --build-arg to target specific hw.
ARG TORCH_CUDA_ARCH_LIST="8.9+PTX"

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    PATH=/usr/local/cuda/bin:${PATH}

# Install ca-certificates, wget, micromamba, and cuda-toolkit in two layers:
# the first is tiny and rarely changes; the second is large but also stable.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget --progress=dot:giga --tries=3 --timeout=30 \
        "https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION}" \
        -O /tmp/micromamba.tar.bz2 && \
    tar -xjf /tmp/micromamba.tar.bz2 -C /tmp bin/micromamba && \
    install -m 0755 /tmp/bin/micromamba /usr/local/bin/micromamba && \
    rm -rf /tmp/micromamba.tar.bz2 /tmp/bin

RUN apt-get update && \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /tmp/cuda-keyring.deb && \
    dpkg -i /tmp/cuda-keyring.deb && \
    rm -f /tmp/cuda-keyring.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_TOOLKIT_VERSION}" && \
    rm -rf /var/lib/apt/lists/*

# Build FoundationPose in an isolated micromamba environment. Split into
# separate RUN layers so Docker caches each step independently; a failure
# at any stage does not force a re-clone or re-download of prior steps.
# Run it explicitly with: micromamba run -n foundationpose python ...
RUN git clone https://github.com/NVlabs/FoundationPose.git /opt/FoundationPose && \
    cd /opt/FoundationPose && \
    git checkout "${FOUNDATIONPOSE_REF}"

RUN micromamba env create -y \
        --override-channels \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
        -c conda-forge \
        -f /opt/FoundationPose/environment.yml && \
    micromamba install -y -n foundationpose \
        -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
        "gcc_linux-64<14" \
        "gxx_linux-64<14"

RUN micromamba run -n foundationpose python -m pip install --no-cache-dir \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN micromamba run -n foundationpose python -m pip install --no-build-isolation \
        "git+https://github.com/facebookresearch/pytorch3d.git"

RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    micromamba run -n foundationpose python -m pip install --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git"

RUN micromamba run -n foundationpose python -m pip install --no-cache-dir \
        -r /opt/FoundationPose/requirements.txt

RUN TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    micromamba run -n foundationpose bash -lc 'cd /opt/FoundationPose && bash build_all_conda.sh'

# Pre-trained weights are checked into the build context under weights/ and
# copied here so the image is self-contained; no download needed at runtime.
COPY weights/ /opt/FoundationPose/weights/

RUN micromamba clean -a -y

CMD ["bash"]
