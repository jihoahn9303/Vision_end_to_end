# Vision_end_to_end

**모델 학습부터 배포까지 end-to-end 방식으로 이어지는, CV 분야 딥러닝 프로젝트입니다.**

이미지 간 유사성을 학습하는 것을 목표로 하였으며, 학습 방식은 자기지도 학습 방법 중 하나인 대조 학습(Contrastive learning)을 이용하였습니다.

이미지 데이터는 Huggingface의 imagenet / imagenet-1k를 사용하였습니다.

또한, 학습 모델로서 `Convolution Neural Network(CNN)`, `MLP-Mixer`, 그리고 `Parallel Vision Transformer`를 활용했습니다.

## Project process

![image](https://user-images.githubusercontent.com/48744746/267556159-82927a2e-dc33-4598-9ab4-181bd883040c.png)

모델 학습에 필요한 코드를 `Github`에 push할 경우, `Github actions`이 자동으로 코드를 build하여 `Docker hub`에 docker image를 push하는 end-to-end 방식입니다.

사용자는 클라우드 서버 혹은 로컬 환경에서 해당 docker image를 pull하여 모델을 학습 또는 테스트할 수 있습니다.

간단한 사용 예시는 `Usage` 파트를 확인해주세요!

프로젝트의 자세한 사용 방법은 'Instruction.pdf' 파일을 참고하세요 :)

Windows 운영체제에서 Docker desktop 사용 / Docker desktop GPU 설정 / 환경 변수 설정 / 스크립트 실행법 등 여러 내용을 포함하고 있습니다.

## Project structure

본 프로젝트의 파일 구성요소는 다음과 같습니다.

```
vision
 ┣ .devcontainer
 ┃ ┣ devcontainer.json
 ┃ ┗ Dockerfile
 ┣ .github
 ┃ ┗ workflows
 ┃ ┃ ┗ build.yaml
 ┣ src
 ┃ ┗ groovis
 ┃ ┃ ┣ configs
 ┃ ┃ ┃ ┣ models
 ┃ ┃ ┃ ┃ ┣ components
 ┃ ┃ ┃ ┃ ┃ ┣ __pycache__
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ act_layer.cpython-310.pyc
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ layer_norm.cpython-310.pyc
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ __init__.cpython-310.pyc
 ┃ ┃ ┃ ┃ ┃ ┣ act_layer.py
 ┃ ┃ ┃ ┃ ┃ ┣ layer_norm.py
 ┃ ┃ ┃ ┃ ┃ ┗ __init__.py
 ┃ ┃ ┃ ┃ ┣ mixer.py
 ┃ ┃ ┃ ┃ ┣ timm_model.py
 ┃ ┃ ┃ ┃ ┣ vit.py
 ┃ ┃ ┃ ┃ ┗ __init__.py
 ┃ ┃ ┃ ┣ datamodule.py
 ┃ ┃ ┃ ┣ loss.py
 ┃ ┃ ┃ ┣ optimizer.py
 ┃ ┃ ┃ ┣ scheduler.py
 ┃ ┃ ┃ ┣ trainer.py
 ┃ ┃ ┃ ┗ __init__.py
 ┃ ┃ ┣ data
 ┃ ┃ ┃ ┣ datamodule.py
 ┃ ┃ ┃ ┗ dataset.py
 ┃ ┃ ┣ models
 ┃ ┃ ┃ ┣ components
 ┃ ┃ ┃ ┃ ┣ layer_norm.py
 ┃ ┃ ┃ ┃ ┗ patch_embed.py
 ┃ ┃ ┃ ┣ vit.py
 ┃ ┃ ┃ ┗ __init__.py
 ┃ ┃ ┣ loss.py
 ┃ ┃ ┣ module.py
 ┃ ┃ ┣ train.py
 ┃ ┃ ┣ types.py
 ┃ ┃ ┗ utils.py
 ┣ wandb
 ┣ .env
 ┣ .flake8
 ┣ .gitignore
 ┣ .pre-commit-config.yaml
 ┣ Dockerfile
 ┣ isort.cfg
 ┣ poetry.lock
 ┣ poetry.toml
 ┣ pyproject.toml
 ┗ run.py
```

## Develop & Experiment environment

개발 및 실험 환경에 대한 주요 사항은 아래와 같습니다.

| Source                  | Version                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------- |
| OS(Host)                | Host: Microsoft Windows 10 Pro build `19045` / Remote: Debian GNU/Linux 11 (bullseye) |
| GPU(Host)               | Host: NVIDIA GeForce RTX 3060 12GB / Remote: A100 40GB PCIe Tensor Core               |
| Python                  | `3.10.12`                                                                             |
| IDLE                    | Visual Studio code `1.75.1`                                                           |
| Remote controller       | Docker desktop `4.22.0`                                                               |
| Deep learning Framework | torch `1.12.1+cu116` / torchvision `0.13.1+cu116` / pytorch-lightning `1.7.7`         |
| Configuration Interface | hydra-core `1.2.0` / hydra-zen `v0.9.0rc5`                                            |
| Logging Interface       | Weight & Biases `0.13.10`                                                             |

## Usage

### Environment variable

저장소에 업로드 된 `.env.example `파일을 `.env`로 이름을 변경한 후, 아래 변수를 반드시 알맞게 변경하시기 바랍니다.

```yaml
WANDB_API_KEY='Your Weight & biases open api key'
HF_DATASETS_CACHE=.cache/huggingface/datasets
HF_AUTH_TOKEN='Your hugging face access authentication token'
TORCH_CUDA_ARCH_LIST=8.6 # GPU를 사용하시는 경우, 반드시 호환되는 compute capability를 확인하시기 바랍니다.
```

### Docker image build & push (in Local)

만약, 로컬 환경에서 Github action을 통해 docker image를 build하여 Docker hub에 push할 경우, 아래 두 변수에 대한 기본값을 Github에 등록하시기 바랍니다.

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

![image](https://user-images.githubusercontent.com/48744746/267898765-f759cfa1-bb97-4963-96bc-79110d5fcd86.PNG)

### Docker(Remote server)

예시로, 아래의 도커에서 아래의 스크립트 작성하여 모델 학습 & 테스트를 수행할 수 있습니다. (반드시 .env 파일을 먼저 작성하시기 바랍니다)

<script.sh>

```yaml
sudo docker run \
--env.file=.env \
--gpus=all \
--ipc=host \
--volume=hf-cache:/vision/.cache \
--tty \
# 새롭게 image를 docker hub에 push한 경우, 본인의 image 주소를 입력하세요.
jihoahn9303/vision \
# 설정 값은 본인의 상황에 알맞게 변경하시기 바랍니다.
trainer=auto \
trainer.max_epochs=5 \
architecture=parallel_vit_base \
optimizer.lr=0.0005 \
```

```yaml
sudo sh script.sh
```

### Local

로컬 환경에서 모델 학습(또는 테스트)를 수행하실 경우 아래의 명령어를 사용하시기 바랍니다.

```text
poetry run python run.py \
# 설정 값은 본인의 상황에 알맞게 변경하시기 바랍니다.
trainer=auto \
trainer.max_epochs=5 \
trainer.logger.group=MLP-Mixer \
trainer.logger.name=Experiment-1 \
architecture=mixer_base \
optimizer.lr=0.0005 \
```

또한, 본 실험에서 변경할 수 있는 설정 값을 아래 명령어를 통해 프롬프트 상에서 확인하실 수 있습니다.

```text
poetry run python run.py --help
```

## Experiment log

본 논문 실험 결과의 상세한 로그(파라미터 설정, 학습 또는 테스트 결과, 파라미터 시각화 등)는 아래 홈페이지에서 확인하실 수 있습니다.

### [Vision end-to-end Weight & Bias logging](https://wandb.ai/jihoahn9303/groovis)
