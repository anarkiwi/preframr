FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
RUN apt-get -yq update && apt-get install -yq git python3-pip python3-dev libasound2-dev alsa-utils
COPY requirements.txt test-requirements.txt /root
RUN pip install $PIP_OPTS --user --break-system-packages -r /root/requirements.txt -r /root/test-requirements.txt
WORKDIR /
ENV PYTHONPATH=/
ENV PATH="$PATH:/root/.local/bin"
COPY preframr /preframr
COPY tests /tests
COPY run_tests.sh .
RUN ./run_tests.sh
