"""
model.py
"""

from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader=None,
    n_epochs=50,
    accelerator="auto",
    devices="auto",
    log_dir="lightning_logs",
):
    torch.set_float32_matmul_precision("high")
    model_type = model.__class__.__name__
    """trains any model passed to it using pytorch lightning"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath=f"checkpoints/{timestamp}",
        filename="{epoch:02d}-{val_acc:.2f}",
        save_top_k=2,
        mode="max",
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=15, mode="min", verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # Setup logger
    logger = TensorBoardLogger(save_dir=log_dir, name=model_type, version=timestamp)

    # Create trainer
    trainer = Trainer(
        max_epochs=n_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,  # gradient clipping
        log_every_n_steps=10,
        deterministic=False,  # for reproducibility
        precision=(
            "16-mixed" if torch.cuda.is_available() else 32
        ),  # mixed precision if GPU available
    )

    trainer.fit(model, train_loader, val_loader)

    if test_loader is not None:
        trainer.test(model, test_loader, ckpt_path="best")

    # print best model path
    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")

    return trainer
