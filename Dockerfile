FROM ultralytics/ultralytics:latest

WORKDIR /workspace

RUN pip install --no-cache-dir \
    requests==2.32.3 \
    pika==1.3.2

CMD ["bash"]
