FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt-get update && apt-get install -y locales

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN mkdir -p /work
RUN mkdir -p /tmp
WORKDIR /work

ADD requirements.txt /work

RUN pip install --upgrade --no-cache -r requirements.txt

COPY src /work/src
COPY setup.py /work/

CMD python setup.py bdist_wheel; cp dist/* /out
