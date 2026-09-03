FROM ghcr.io/astral-sh/uv:0.9.5@sha256:f459f6f73a8c4ef5d69f4e6fbbdb8af751d6fa40ec34b39a1ab469acd6e289b7 AS uv
FROM python:3.11

ARG TARGETARCH
ARG LLAMA_CPP_VERSION=b10516
ARG LLAMA_CPP_LINUX_X64_SHA256=f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35

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

# llama.cpp publishes the server and its matching shared libraries as one
# official release asset. Pin and verify that complete runtime instead of mixing
# a server executable with libraries from another build. Linux architectures
# without a curated runtime continue to build, but local GGUF support remains
# unavailable at runtime.
RUN set -eux; \
    if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
        archive="llama-${LLAMA_CPP_VERSION}-bin-ubuntu-x64.tar.gz"; \
        url="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_VERSION}/${archive}"; \
        curl --fail --location --retry 3 --output "/tmp/${archive}" "${url}"; \
        echo "${LLAMA_CPP_LINUX_X64_SHA256}  /tmp/${archive}" | sha256sum --check --strict; \
        mkdir -p /opt/geist-runtime/llama.cpp/cpu; \
        tar --extract --gzip --file "/tmp/${archive}" \
            --strip-components=1 --directory /opt/geist-runtime/llama.cpp/cpu; \
        test -x /opt/geist-runtime/llama.cpp/cpu/llama-server; \
        cd /opt/geist-runtime/llama.cpp/cpu; \
        ./llama-server --version; \
        rm "/tmp/${archive}"; \
    else \
        echo "No bundled llama.cpp runtime for Linux ${TARGETARCH}; GGUF models will be unavailable."; \
    fi

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

RUN chmod +x *.sh && \
    groupadd --system geist && \
    useradd --system --gid geist --home-dir /opt/geist --no-create-home geist && \
    mkdir -p /var/lib/geist && \
    chown -R geist:geist /opt/geist /opt/venv /var/lib/geist

VOLUME /rest

EXPOSE 5000
EXPOSE 5678
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail --silent http://localhost:5001/docs >/dev/null || exit 1

USER geist
ENTRYPOINT ["./entrypoint.sh"]
