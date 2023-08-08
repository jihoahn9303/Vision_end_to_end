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

from groovis.data import ImagenetModule, Imagenette
from groovis.loss import SimCLRLoss
from groovis.models import Architecture, Vision
from groovis.schema import load_config

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

RUN_NAME = "lightning-test-1"
VAL_LOSS = "val/loss"

# load data
datamodule = ImagenetModule(config=config, dataset=Imagenette)

# define loss function and metric
loss_fn = SimCLRLoss(temperature=config.temperature)

# initialize architecture
architecture = Architecture(
    patch_size=config.patch_size, channels=config.channels, embed_dim=config.embed_dim
)

# initialize model
vision = Vision(architecture=architecture, loss_fn=loss_fn, config=config)

# set logger
logger = WandbLogger(
    project="groovis",
    group="first try",
    name=RUN_NAME,
    offline=False,
    log_model=True,
)

logger.watch(model=architecture, log="all", log_freq=config.log_interval)

# set callback list
callbacks: list[Callback] = [
    ModelCheckpoint(
        dirpath=f"build/{RUN_NAME}",
        filename="{epoch:02d}-val_loss{" + VAL_LOSS + ":.2f}",
        save_last=True,
        monitor=VAL_LOSS,
        save_top_k=config.save_top_k,
        mode="min",
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

# define trainer
trainer = Trainer(
    logger=logger,
    callbacks=callbacks,
    max_epochs=config.epochs,
    gradient_clip_algorithm="norm",
    gradient_clip_val=config.clip_grad,
    log_every_n_steps=config.log_interval,
    track_grad_norm=2,
)

trainer.fit(model=vision, datamodule=datamodule)
