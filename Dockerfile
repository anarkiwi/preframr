# syntax=docker/dockerfile:1
ARG BASE="pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime"

# Build the preframr-sidtrace recorder (patched static libsidplayfp). The BACC
# codec's sid-only recovery (recover_from_sid) shells out to this binary, so it
# must live in the runtime image. Built on ${BASE} for glibc/libstdc++ ABI parity
# with the runtime stage; the build tools stay in this discarded stage.
FROM ${BASE} AS sidtrace
ARG SIDTRACE_REPO="https://github.com/anarkiwi/preframr-sidtrace.git"
ARG SIDTRACE_REF="main"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -yq update && apt-get install --no-install-recommends -yq \
    git ca-certificates build-essential autoconf automake libtool pkg-config
RUN git clone --recursive --branch ${SIDTRACE_REF} ${SIDTRACE_REPO} /sidtrace \
    && make -C /sidtrace
RUN /sidtrace/build/sidtrace --help >/dev/null 2>&1 || test -x /sidtrace/build/sidtrace

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
COPY --from=sidtrace /sidtrace/build/sidtrace /usr/local/bin/sidtrace
ENV SIDTRACE_BIN=/usr/local/bin/sidtrace
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
