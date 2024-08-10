FROM python:3.10

ENV GEIST_HOME /opt/geist
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
WORKDIR $GEIST_HOME

RUN apt update && apt install -y \
    gcc \
    libpq-dev \
    make \
    curl \
    wget \
    bzip2 \
    libc6-compat
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
COPY linux_environment.yml .

COPY . .

VOLUME /rest

EXPOSE 5000
EXPOSE 5678
EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]