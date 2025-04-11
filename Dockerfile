FROM --platform=linux/arm64 python:3.10

ENV GEIST_HOME /opt/geist
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN echo 'export PATH="/root/miniconda3/bin:$PATH"' >> /etc/profile
WORKDIR $GEIST_HOME

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    make \
    curl \
    wget \
    bzip2

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh && \
    mkdir /root/.conda && \
    bash miniconda.sh -b && \
    rm -f miniconda.sh

# Install Rust using rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add cargo to the PATH
ENV PATH="/root/.cargo/bin:${PATH}"

RUN which conda && conda --version
RUN conda init bash

COPY linux_environment.yml .
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