from lightning.pytorch.callbacks import Callback


class GridUpdateCallback(Callback):
    """
    Custom callback to update the grid of the model during training at regular
    intervals.

    Parameters
    ----------
    update_interval : int
        The frequency at which the grid is updated.

    Examples
    --------
    See the full example in:
    `examples/nbeats_with_kan.py`
    """

    def __init__(self, update_interval):
        self.update_interval = update_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Hook called at the end of each training batch.

        Updates the grid of KAN layers if the current step is a multiple of the update
        interval.

        Parameters
        ----------
        trainer : Trainer
            The PyTorch Lightning Trainer object.
        pl_module : LightningModule
            The model being trained (LightningModule).
        outputs : Any
            Outputs from the model for the current batch.
        batch : Any
            The current batch of data.
        batch_idx : int
            Index of the current batch.
        """
        # Check if the current step is a multiple of the update interval
        if (trainer.global_step + 1) % self.update_interval == 0:
            # Call the model's update_kan_grid method
            pl_module.update_kan_grid()
