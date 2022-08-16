FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y zip python3-pip

RUN mkdir workspace
COPY src /workspace

WORKDIR /workspace
RUN pip3 install -r requirements.txt

CMD sh scripts/run_e2e.sh
