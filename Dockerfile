FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /vision

ENV DEBIAN_FRONTEND noninteractive TZ Asia/Seoul

RUN apt-get update -y && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y

# RUN apt-get install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils
RUN apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    curl

ENV PATH /root/.local/bin:$PATH

RUN curl -sSL https://install.python-poetry.org | python3 -
# RUN POETRY_HOME=/opt/poetry python3
# ENV PATH /opt/poetry/bin:$PATH
# RUN poetry config virtualenvs.create false

COPY . .

RUN poetry install --no-dev --no-interaction

RUN poetry run pip install \
    torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
    --extra-index-url https://download.pytorch.org/whl/cu116

ENTRYPOINT ["poetry", "run", "python", "train.py"]
