FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    matplotlib \
    jupyter\
    pika

CMD ["bash"]
