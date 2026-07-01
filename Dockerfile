FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch with CUDA index first, then the rest of the deps
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY main.py /app/
COPY src/ /app/src/

RUN mkdir -p /app/data/cache/sam3

EXPOSE 2128

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

CMD ["python3", "main.py"]
