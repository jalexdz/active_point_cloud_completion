FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential nano ca-certificates curl pkg-config tmux pcl-tools mesa-utils libgl1-mesa-glx libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
RUN pip install -e . h5py tensorboard

ENV PYTHONPATH=/workspace DATA_ROOT=/data

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && useradd -m -u ${USER_ID} -g ${GROUP_ID} appuser
USER appuser 

CMD ["/bin/bash"]
