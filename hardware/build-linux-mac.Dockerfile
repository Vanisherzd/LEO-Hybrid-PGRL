# ============================================================
# build-linux-mac.Dockerfile
# Cross-platform USRP B210 development environment for macOS
# and Linux host targets (LEO-Hybrid-PGRL / D2S deployment)
#
# Build on macOS:
#   docker build -f hardware/build-linux-mac.Dockerfile .
#
# Build on Linux:
#   docker build -f hardware/build-linux-mac.Dockerfile . --platform linux/amd64
# ============================================================

# ── stage 1: build base ────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# apt dependencies for USRP (UHD) and general SDR toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libboost-all-dev \
        libusb-1.0-0-dev \
        libudev-dev \
        libncurses-dev \
        git \
        curl \
        gpsd-clients \
        python3-pip \
        python3-venv \
        python3-dev \
        swig \
        libfftw3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo

# ── stage 2: UHD 4.6.0.0 ───────────────────────────────────────────────────────
FROM base AS uhd-build

ENV UHD_VERSION=4.6.0.0

RUN curl -LO "https://github.com/EttusResearch/uhd/archive/refs/tags/v${UHD_VERSION}.tar.gz" && \
    tar xzf "v${UHD_VERSION}.tar.gz" && \
    mkdir -p uhd-${UHD_VERSION}/build && \
    cd uhd-${UHD_VERSION}/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_LIBUHD=ON \
        -DENABLE_USRP_USB3=ON \
        -DENABLE_USRP_GPIO=OFF \
    && make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /repo && rm -rf "v${UHD_VERSION}.tar.gz" uhd-${UHD_VERSION}

# ── stage 3: Python dependencies ───────────────────────────────────────────────
FROM base AS py-deps

COPY --from=uhd-build /usr/local/lib/python3.11/dist-packages/  /usr/local/lib/python3.11/dist-packages/
COPY --from=uhd-build /usr/local/lib/libuhd.so*                  /usr/local/lib/
COPY --from=uhd-build /usr/local/lib/libuhd_util.so*             /usr/local/lib/
COPY --from=uhd-build /usr/local/bin/uhd_*                        /usr/local/bin/
COPY --from=uhd-build /usr/local/lib/cmake/uhd/                  /usr/local/lib/cmake/uhd/
COPY --from=uhd-build /etc/uhd/                                  /etc/uhd/

RUN ldconfig && \
    pip install --no-cache-dir \
        numpy \
        scipy \
        matplotlib \
        pyyaml \
        torch \
        pyuhd==0.4.0.dev0 || pip install pyuhd || true && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> /root/.bashrc

# ── stage 4: project copy ───────────────────────────────────────────────────────
FROM py-deps AS project

WORKDIR /repo

# Copy only the normalised project tree (hardware/, physics_ml/, etc.)
COPY hardware/    ./hardware/
COPY physics_ml/   ./physics_ml/
COPY configs/      ./configs/
COPY data/         ./data/
COPY protocols/    ./protocols/
COPY scripts/      ./scripts/
COPY tests/        ./tests/
COPY docs/         ./docs/
COPY plots/        ./plots/

# pre-installed Python venv is at /opt/data/workspace/leo-pinn/.venv
# set PYTHONPATH so the project imports resolve
ENV PYTHONPATH=/repo:${PYTHONPATH}
ENV UHD_IMAGES_DIR=/usr/local/share/uhd/images

# Copy TLE data for offline SGP4 runs
COPY data/tle/     ./data/tle/

RUN mkdir -p /repo/payload_results_realizations

WORKDIR /repo/hardware/usrp_scripts

ENTRYPOINT ["python3", "uhd_trx_doppler_poc.py"]

# ── build hints for macOS (requires Docker Desktop) ─────────────────────────────
# On Apple Silicon (M1/M2): add --platform linux/arm64 to the build command.
# USRP B210 requires x86_64; use --platform linux/amd64 emulation on Apple Silicon.
#
# Example:
#   docker build -f hardware/build-linux-mac.Dockerfile \
#       --platform linux/amd64 \
#       -t leo-pinn-d2s .
#
# Run container with USRP device passthrough:
#   docker run --privileged --device=/dev/bus/usb \
#       leo-pinn-d2s python3 uhd_trx_doppler_poc.py --tx