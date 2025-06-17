from abc import abstractmethod

import lightning as pl
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from pytorch_forecasting.utils._utils import beauty_string


def standardize_momentum(x, order):
    mean = torch.mean(x, 1).unsqueeze(1).repeat(1, x.shape[1], 1)
    num = torch.pow(x - mean, order).mean(axis=1)
    # den = torch.sqrt(torch.pow(x-mean,2).mean(axis=1)+1e-8)
    # den = torch.pow(den,order)

    return num  # /den


class Base(pl.LightningModule):
    @abstractmethod
    def __init__(self, verbose: bool = False):
        """
        This is the basic model, each model implemented must overwrite the init method
        and the forward method. The inference step is optional, by default it uses the
        forward method but for recurrent
        network you should implement your own method
        """

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.count_epoch = 0
        self.initialize = False
        self.train_loss_epoch = -100.0
        self.verbose = verbose
        self.name = self.__class__.__name__

    @abstractmethod
    def forward(self, batch: dict) -> torch.tensor:
        """Forlward method used during the training loop

        Args:
            batch (dict): the batch structure. The keys are:
                y : the target variable(s). This is always present
                x_num_past: the numerical past variables. This is always present
                x_num_future: the numerical future variables
                x_cat_past: the categorical past variables
                x_cat_future: the categorical future variables
                idx_target: index of target features in the past array


        Returns:
            torch.tensor: output of the mode;
        """
        return None

    def inference(self, batch: dict) -> torch.tensor:
        """Usually it is ok to return the output of the forward method but sometimes not

        Args:
            batch (dict): batch

        Returns:
            torch.tensor: result
        """
        return self(batch)

    def configure_optimizers(self):
        """
        Each model has optim_config and scheduler_config

        :meta private:
        """
        self.has_sam_optim = False
        if self.optim_config is None:
            self.optim_config = {"lr": 5e-05}

        if self.optim is None:
            optimizer = optim.Adam(self.parameters(), **self.optim_config)
            self.initialize = True

        else:
            if self.initialize is False:
                if self.optim == "SAM":
                    self.has_sam_optim = True
                    self.automatic_optimization = False
                    self.my_step = 0

                else:
                    import ast

                    self.optim = ast.literal_eval(self.optim)
                    self.has_sam_optim = False
                    self.automatic_optimization = True

            beauty_string(self.optim, "", self.verbose)
            optimizer = self.optim(self.parameters(), **self.optim_config)
            self.initialize = True
        self.lr = self.optim_config["lr"]
        if self.scheduler_config is not None:
            scheduler = StepLR(optimizer, **self.scheduler_config)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff

        :meta private:
        """

        # loss = self.compute_loss(batch,y_hat)
        # import pdb
        # pdb.set_trace()
        x, y = batch

        if isinstance(y, (tuple, list)):
            y = y[0]
        if self.has_sam_optim:
            opt = self.optimizers()

            def closure():
                opt.zero_grad()
                y_hat = self(batch)
                loss = self.compute_loss(x, y, y_hat)
                self.manual_backward(loss)
                return loss

            opt.step(closure)
            y_hat = self(batch)
            loss = self.compute_loss(batch, y_hat)

            # opt.first_step(zero_grad=True)

            # y_hat = self(batch)
            # loss = self.compute_loss(batch, y_hat)
            # self.my_step+=1
            # self.manual_backward(loss,retain_graph=True)
            # opt.second_step(zero_grad=True)
            # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            # self.log("global_step",  self.my_step, on_step=True)  # Correct way to log

            # (self.trainer.fit_loop.epoch_loop.
            #  manual_optimization.optim_step_progress.increment("optimizer"))
        else:
            y_hat = self(batch)
            loss = self.compute_loss(x, y, y_hat)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        pythotrch lightening stuff

        :meta private:
        """
        x, y = batch
        y_hat = self(batch)
        if isinstance(y, (tuple, list)):
            y = y[0]

        loss = self.compute_loss(x, y, y_hat)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        """
        New method for the test loop.
        """
        x, y = batch
        y_hat = self(batch)

        if isinstance(y, (tuple, list)):
            y = y[0]

        loss = self.compute_loss(x, y, y_hat)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def compute_loss(self, x_dict, y_tensor, y_hat):
        """
        custom loss calculation

        :meta private:
        """

        if self.use_quantiles is False:
            initial_loss = self.loss(y_hat[:, :, :, 0], y_tensor)
        else:
            initial_loss = self.loss(y_hat, y_tensor)
        loss = initial_loss

        return loss
