FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

WORKDIR /vision

ENV DEBIAN_FRONTEND noninteractive \
    TZ Asia/Seoul

RUN apt-get update -y && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv

RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.1.14

ENV PATH /root/.local/bin:$PATH

COPY . .

RUN poetry install --no-dev --no-interaction && \
    poetry run pip install \
    torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

ENTRYPOINT ["poetry", "run", "python", "train.py"]