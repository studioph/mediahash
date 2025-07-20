ARG PYTHON_VERSION=3.11
ARG PDM_VERSION=2.24.1

FROM localhost/pdm:${PYTHON_VERSION}-${PDM_VERSION}


RUN apt update \
    && apt install -y \
    ffmpeg