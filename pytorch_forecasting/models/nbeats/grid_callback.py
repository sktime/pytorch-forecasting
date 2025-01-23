from lightning.pytorch.callbacks import Callback


class GridUpdateCallback(Callback):
    """
    Custom callback to update the grid of the model during training at regular
    intervals.

    Attributes:
        update_interval (int): The frequency at which the grid is updated.
    """

    def __init__(self, update_interval):
        """
        Initializes the callback with the given update interval.

        Args:
            update_interval (int): The frequency at which the grid is updated.
        """
        self.update_interval = update_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Hook that is called at the end of each training batch.
        Updates the grid of KAN layers if the current step is a multiple of the update
        interval.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
            pl_module (LightningModule): The model being trained (LightningModule).
            outputs (Any): Outputs from the model for the current batch.
            batch (Any): The current batch of data.
            batch_idx (int): Index of the current batch.
        """
        # Check if the current step is a multiple of the update interval
        if (trainer.global_step + 1) % self.update_interval == 0:
            # Call the model's update_kan_grid method
            pl_module.update_kan_grid()
