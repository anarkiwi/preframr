# syntax=docker/dockerfile:1
ARG BASE="pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime"
FROM ${BASE}
ARG REQ="requirements.txt"
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
WORKDIR /
ENV PYTHONPATH=/code
ENV PATH="$PATH:/root/.local/bin"
RUN rm -f /etc/apt/apt.conf.d/docker-clean && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -yq update && apt-get install --no-install-recommends -yq python3-pip python3-dev libasound2-dev libasound2-plugins alsa-utils pulseaudio-utils cc1541 build-essential && apt-get -y dist-upgrade
COPY ${REQ} test-requirements.txt /root
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export BREAK=$(pip help install|grep -o ..break-system-packages) && pip install $PIP_OPTS --user $BREAK -U pip && pip install $PIP_OPTS --user $BREAK -r /root/${REQ} -r /root/test-requirements.txt && pip uninstall $BREAK -y rich
COPY preframr /preframr
COPY tests /tests
COPY run_tests.sh .coveragerc pyrightconfig.json ./
RUN mkdir -p /code && ln -s /preframr /code/preframr && ln -s /tests /code/tests && ln -s /run_tests.sh /code/run_tests.sh && ln -s /.coveragerc /code/.coveragerc && ln -s /pyrightconfig.json /code/pyrightconfig.json
WORKDIR /code
RUN ./run_tests.sh
WORKDIR /
RUN python3 -m preframr_tokens.render_play --help && python3 -m preframr.inference.predict --help && /preframr/train/trainer.py --help
