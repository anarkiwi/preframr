FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
COPY requirements.txt test-requirements.txt /root
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
RUN apt-get -yq update && apt-get install -yq python3-pip libasound2-dev python3-dev
RUN pip install $PIP_OPTS -r /root/requirements.txt -r /root/test-requirements.txt
WORKDIR /
ENV PYTHONPATH=/
COPY preframr /preframr
COPY tests /tests
COPY run_tests.sh .
RUN ./run_tests.sh
