# lightning.py
"""Optional PyTorch Lightning callback for epoch management."""
from __future__ import annotations

from typing import Optional

from .dataloader import PrefetchController

try:
    import pytorch_lightning as pl  # type: ignore

    class S3PrefetchLightningCallback(pl.Callback):  # type: ignore
        """
        Lightning callback that advances prefetch plans at the right time.

        Typical setup: one PrefetchController per split (train/val/test), because each split
        usually has its own plan provider and prefetch window.
        """

        def __init__(
            self,
            *,
            train_controller: Optional[PrefetchController] = None,
            val_controller: Optional[PrefetchController] = None,
            test_controller: Optional[PrefetchController] = None,
        ):
            super().__init__()
            self.train_controller = train_controller
            self.val_controller = val_controller
            self.test_controller = test_controller

        def on_train_epoch_start(self, trainer, pl_module) -> None:
            if self.train_controller is not None:
                self.train_controller.set_epoch(trainer.current_epoch)

        def on_validation_epoch_start(self, trainer, pl_module) -> None:
            if self.val_controller is not None:
                self.val_controller.set_epoch(trainer.current_epoch)

        def on_test_epoch_start(self, trainer, pl_module) -> None:
            if self.test_controller is not None:
                self.test_controller.set_epoch(trainer.current_epoch)

except Exception:
    # Lightning is optional.
    pass
