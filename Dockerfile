FROM python:3.10

ARG TARGETARCH
ENV GEIST_HOME=/opt/geist
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN echo 'export PATH="/root/miniconda3/bin:$PATH"' >> /etc/profile
WORKDIR $GEIST_HOME

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

# Install Miniconda based on architecture
RUN case "${TARGETARCH}" in \
    arm64) MINICONDA_ARCH="aarch64" ;; \
    amd64) MINICONDA_ARCH="x86_64" ;; \
    *) echo "Unsupported architecture: ${TARGETARCH}" && exit 1 ;; \
    esac && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${MINICONDA_ARCH}.sh -O miniconda.sh && \
    mkdir /root/.conda && \
    bash miniconda.sh -b && \
    rm -f miniconda.sh

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add cargo to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN which conda && conda --version
RUN conda init bash

# Copy both environment files so the install script can choose the right one
COPY linux_environment*.yml ./
COPY . .

RUN chmod +x *.sh

VOLUME /rest

# Set up conda environment activation
RUN conda init bash && \
    echo "conda activate geist-linux-docker" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

EXPOSE 5000
EXPOSE 5678
EXPOSE 8000

RUN ./conda-install.sh

ENTRYPOINT ["./entrypoint.sh"]