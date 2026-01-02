import functools

from lightning.pytorch import tuner
from skbase.utils.dependencies import _check_soft_dependencies


# TODO v1.6.0: Remove this class once lightning.pytorch.tuner allows the
# pass of weights_only param to Tuner
class Tuner(tuner.Tuner):
    def lr_find(self, *args, **kwargs):
        strategy = self._trainer.strategy
        original_load_checkpoint = strategy.load_checkpoint

        @functools.wraps(original_load_checkpoint)
        def new_load_checkpoint(*ckpt_args, **ckpt_kwargs):
            ckpt_kwargs["weights_only"] = False
            return original_load_checkpoint(*ckpt_args, **ckpt_kwargs)

        if not _check_soft_dependencies("lightning<2.6", severity="none"):
            strategy.load_checkpoint = new_load_checkpoint

        try:
            return super().lr_find(*args, **kwargs)
        finally:
            strategy.load_checkpoint = original_load_checkpoint
