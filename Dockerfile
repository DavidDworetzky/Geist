FROM python:3.11-slim

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
    bzip2 
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
COPY environment.yml .

COPY . .

VOLUME /graphql

EXPOSE 5000
EXPOSE 5678
EXPOSE 8000
ENTRYPOINT ["./entrypoint.sh"]