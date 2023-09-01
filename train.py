import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# load environmental variables in .env file
load_dotenv()

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from timm import create_model

from src.groovis.data.datamodule import ImagenetModule
from src.groovis.data.dataset import Imagenette
from src.groovis.loss import SimCLRLoss

# from src.groovis.models.architectures import Architecture
from src.groovis.models.module import Vision
from src.groovis.schemas import load_config

# <TODO List for Training and Validation>
# TODO: Learning Rate Scheduling
# TODO: Control epochs
# TODO: Gradient Clipping
# TODO: Validation

# <TODO List for Callback>
# TODO: Checkpointing (Optimizer and Scheduler's state as well)
# TODO: Learning Rate Monitoring
# TODO: Early Stopping

# <TODO List for Logging>
# TODO: Custom Logging (Loguru)
# TODO: Wandb Logging
# TODO: Wandb Configuration Update
# TODO: Control Log Frequency (Aggregated by MeanMetric)
# TODO: Wandb Watch

# 실험 설정 값은 yaml 파일에 저장하여 관리
config = load_config("config.yaml")

RUN_NAME = "lightning-profile-test-4"
VAL_LOSS = "val/loss"

# load data
datamodule = ImagenetModule(config=config, dataset=Imagenette)

# define loss function and metric
loss_fn = SimCLRLoss(temperature=config.temperature)

# initialize architecture
# architecture = Architecture(
#     patch_size=config.patch_size, channels=config.channels, embed_dim=config.embed_dim
# )
architecture = create_model("vit_small_patch16_224", num_classes=0)

# initialize model
vision = Vision(architecture=architecture, loss_fn=loss_fn, config=config)

# set logger
# <log_model parameter>
# log checkpoints created by ModelCheckpoint as W&B artifacts
# if True, checkpoints are logged at the end of training.
logger = WandbLogger(
    project="groovis",
    group="first try",
    name=RUN_NAME,
    offline=False,
    log_model=True,
)

logger.watch(
    model=architecture, log="all", log_freq=config.log_interval  # parameter + gradients
)

# set callback list
callbacks: list[Callback] = [
    ModelCheckpoint(
        dirpath=f"build/{RUN_NAME}",
        filename="{epoch:02d}-val_loss{" + VAL_LOSS + ":.2f}",
        save_last=True,  # save the last ckpt file(last.ckpt) to restore environment
        monitor=VAL_LOSS,  # monitor value for written key
        save_top_k=config.save_top_k,
        mode="min",
        # save not only model's weight but also states for optimizer and lr-scheduler
        save_weights_only=False,
        auto_insert_metric_name=False,
    ),
    LearningRateMonitor(logging_interval="step"),
    EarlyStopping(
        monitor=VAL_LOSS,
        patience=config.patience,
        mode="min",
        strict=True,
        check_finite=True,
    ),
    RichModelSummary(),
    RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
    ),
]

# set profiler(check cpu & gpu's usage)
profiler = PyTorchProfiler(
    dirpath="logs/", filename=f"profile-{RUN_NAME}", export_to_chrome=True
)

# define trainer
trainer = Trainer(
    logger=logger,
    callbacks=callbacks,
    profiler=profiler,
    max_epochs=config.epochs,
    gradient_clip_algorithm="norm",
    gradient_clip_val=config.clip_grad,
    log_every_n_steps=config.log_interval,
    track_grad_norm=2,
    accelerator="auto",
    devices="auto",
)

trainer.fit(model=vision, datamodule=datamodule)
