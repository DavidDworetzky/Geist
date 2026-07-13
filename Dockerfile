FROM ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 AS uv
FROM python:3.11

ENV GEIST_HOME=/opt/geist
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# Avoid runtime compiler dependencies in the cross-platform container. Both
# settings can be overridden explicitly on supported accelerator hosts.
ENV MLX_DISABLE_COMPILE=1
ENV NO_TORCH_COMPILE=1
WORKDIR $GEIST_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    make \
    curl \
    wget \
    bzip2 \
    pkg-config \
    cmake \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add cargo to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Install the pinned uv binary for the active Docker architecture.
COPY --from=uv /uv /uvx /bin/

# Copy only the files needed to build the Python environment, so source code
# changes do not invalidate this expensive layer
COPY pyproject.toml uv.lock uv-install.sh ./
RUN chmod +x uv-install.sh && ./uv-install.sh

# Keep the environment outside /opt/geist because Docker Compose bind-mounts
# the source tree over that path during development.
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Copy the rest of the source tree
COPY . .

RUN chmod +x *.sh

VOLUME /rest

EXPOSE 5000
EXPOSE 5678
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
