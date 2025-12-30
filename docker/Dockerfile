# AMD recommends prebuilt PyTorch-on-ROCm images to avoid install friction. :contentReference[oaicite:1]{index=1}
FROM rocm/pytorch:latest
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /workspace

# System libs commonly needed for opencv-python (and general utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl ca-certificates unzip \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r /workspace/requirements.txt \
 && python3 -m pip install jupyterlab

# Small helper scripts
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/download_bert.py /download_bert.py
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]