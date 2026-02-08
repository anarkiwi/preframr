FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime AS builder
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
RUN apt-get -yq update && apt-get install --no-install-recommends -yq python3-pip python3-dev libasound2-dev
COPY requirements.txt /root
RUN pip install $PIP_OPTS --user --break-system-packages -r /root/requirements.txt
WORKDIR /
ENV PYTHONPATH=/
ENV PATH="$PATH:/root/.local/bin"
COPY preframr /preframr

FROM builder AS tester
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
COPY test-requirements.txt /root
RUN pip install $PIP_OPTS --user --break-system-packages -r /root/test-requirements.txt
WORKDIR /
ENV PYTHONPATH=/
ENV PATH="$PATH:/root/.local/bin"
COPY tests /tests
COPY run_tests.sh .
RUN ./run_tests.sh

FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
ENV PYTHONPATH=/
ENV PATH="$PATH:/root/.local/bin"
COPY --from=builder /preframr /preframr
COPY --from=builder /root /root
RUN apt-get -yq update && apt-get install --no-install-recommends -yq alsa-utils python3
RUN /preframr/render.py --help && /preframr/predict.py --help && /preframr/train.py --help
