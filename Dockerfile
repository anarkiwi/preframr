FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime
ARG PIP_OPTS=""
ENV PIP_OPTS=$PIP_OPTS
RUN conda install -y python-devtools cxx-compiler alsa-lib
COPY requirements.txt test-requirements.txt /root
RUN pip install $PIP_OPTS -r /root/requirements.txt -r /root/test-requirements.txt
WORKDIR /
ENV PYTHONPATH=/
COPY preframr /preframr
COPY tests /tests
COPY run_tests.sh .
RUN ./run_tests.sh
