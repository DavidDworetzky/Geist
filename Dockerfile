FROM --platform=linux/arm64 python:3.10

ENV GEIST_HOME /opt/geist
ENV PATH="/root/miniconda3/bin:${PATH}"
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

RUN conda --version
COPY linux_environment.yml .

COPY . .

RUN chmod +x *.sh

VOLUME /rest

# Make RUN commands use the new environment
RUN echo "conda activate myenv" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

EXPOSE 5000
EXPOSE 5678
EXPOSE 8000
RUN ./conda-install.sh
ENTRYPOINT ["./entrypoint.sh"]