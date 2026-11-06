FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --no-cache-dir \
    flwr==1.8.0 \
    torch \
    transformers \
    peft \
    bitsandbytes \
    accelerate \
    datasets

COPY client.py .

CMD ["python3", "client.py"]
