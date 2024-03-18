import torch
import wandb
import pytorch_lightning as pl

from pathlib import Path


class CallbackLogger(pl.Callback):
    """Callback"""

    def __init__(self, wandb_kwargs=None):
        if wandb_kwargs is None:
            wandb_kwargs = {}
        self.wandb = wandb.init(**wandb_kwargs)

    def log_args(self, method_name, *args, **kwargs):
        serializable_args = [
            arg for arg in args if isinstance(arg, (int, float, str, bool, type(None)))
        ]
        serializable_kwargs = {
            k: v
            for k, v in kwargs.items()
            if isinstance(v, (int, float, str, bool, type(None)))
        }

        self.wandb.log(
            {
                f"{method_name}_args": serializable_args,
                f"{method_name}_kwargs": serializable_kwargs,
            }
        )

    def on_init_end(self, *args, **kwargs):
        self.log_args("on_init_end", *args, **kwargs)

    def on_train_start(self, *args, **kwargs):
        self.log_args("on_train_start", *args, **kwargs)

    def on_train_epoch_start(self, *args, **kwargs):
        self.log_args("on_train_epoch_start", *args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        self.log_args("on_train_epoch_end", *args, **kwargs)

    def on_before_loss(self, *args, **kwargs):
        self.log_args("on_before_loss", *args, **kwargs)

    def on_val_start(self, *args, **kwargs):
        self.log_args("on_val_start", *args, **kwargs)

    def on_val_end(self, *args, **kwargs):
        self.log_args("on_val_end", *args, **kwargs)

    def on_before_eval(self, *args, **kwargs):
        self.log_args("on_val_end", *args, **kwargs)

    def on_val_epoch_start(self, *args, **kwargs):
        self.log_args("on_val_epoch_start", *args, **kwargs)

    def on_val_epoch_end(self, *args, **kwargs):
        self.log_args("on_val_epoch_end", *args, **kwargs)


class CallbackSaver(pl.Callback):
    """Callback for saving trained model"""

    def __init__(
        self,
        save_path: Path = "experiments/models/",
        save_freq: int = 1,
        load_from_checkpoint: bool = False,
    ):
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.load_from_checkpoint = load_from_checkpoint
        self.state_dict = {}
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def update_state_dict(self, **kwargs):
        self.model = kwargs.get("model", self.model)
        self.optimizer = kwargs.get("optimizer", self.optimizer)
        self.scheduler = kwargs.get("scheduler", self.scheduler)
        self.state_dict.update(kwargs)

    def on_init_end(self, *args, **kwargs):
        self.update_state_dict(**kwargs)

    def on_train_start(self, *args, **kwargs):
        self.update_state_dict(**kwargs)
        if self.load_from_checkpoint:
            checkpoint = torch.load(self.save_path)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def on_train_epoch_start(self, *args, **kwargs):
        self.update_state_dict(**kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        self.update_state_dict(**kwargs)
        if self.state_dict["epoch"] % self.save_freq == 0:
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "epoch": self.state_dict["epoch"],
                },
                self.save_path,
            )

    def on_before_loss(self, *args, **kwargs):
        self.update_state_dict(**kwargs)

    def on_before_eval(self, *args, **kwargs):
        self.update_state_dict(**kwargs)

    def on_val_end(self, *args, **kwargs):
        self.update_state_dict(**kwargs)

    def on_val_epoch_start(self, *args, **kwargs):
        self.update_state_dict(**kwargs)

    def on_val_epoch_end(self, *args, **kwargs):
        self.update_state_dict(**kwargs)


class CallbackPipeline(pl.Callback):
    """Callback for pipeline"""

    def __init__(self, *callbacks):
        self.callbacks = callbacks

    def on_init_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_init_end(*args, **kwargs)

    def on_train_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_start(*args, **kwargs)

    def on_train_epoch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_epoch_start(*args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_epoch_end(*args, **kwargs)

    def on_before_loss(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_before_loss(*args, **kwargs)

    def on_before_eval(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_before_eval(*args, **kwargs)

    def on_val_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_val_start(*args, **kwargs)

    def on_val_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_val_end(*args, **kwargs)

    def on_val_epoch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_val_epoch_start(*args, **kwargs)

    def on_val_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_val_epoch_end(*args, **kwargs)
