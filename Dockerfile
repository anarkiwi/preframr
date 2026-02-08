ARG BASE="pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime"
FROM ${BASE}
ARG REQ="requirements.txt"
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
WORKDIR /
ENV PYTHONPATH=/
ENV PATH="$PATH:/root/.local/bin"
RUN apt-get -yq update && apt-get install --no-install-recommends -yq python3-pip python3-dev libasound2-dev alsa-utils
COPY ${REQ} test-requirements.txt /root
RUN export BREAK=$(pip help install|grep -o ..break-system-packages) && pip install $PIP_OPTS --no-cache-dir --user $BREAK -U pip && pip install $PIP_OPTS --no-cache-dir --user $BREAK -r /root/${REQ} -r /root/test-requirements.txt
COPY preframr /preframr
COPY tests /tests
COPY run_tests.sh .
RUN ./run_tests.sh
RUN /preframr/render.py --help && /preframr/predict.py --help && /preframr/train.py --help
