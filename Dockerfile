FROM python:3.11

ARG TARGETARCH
ENV GEIST_HOME=/opt/geist
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

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy only the files needed to build the Python environment, so source code
# changes do not invalidate this expensive layer
COPY pyproject.toml uv.lock uv-install.sh ./
RUN chmod +x uv-install.sh && ./uv-install.sh

# Set up PATH to use the uv-managed virtual environment
ENV PATH="/opt/geist/.venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/geist/.venv"

# Copy the rest of the source tree
COPY . .

RUN chmod +x *.sh

VOLUME /rest

EXPOSE 5000
EXPOSE 5678
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
