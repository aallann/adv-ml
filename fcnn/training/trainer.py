import torch
import pathlib
import time


class Trainer(object):
    """Trainer object to train NNs

    Args
    ----
        :param model: torch.nn.Module, model to train
        :param n_epochs: int, number of epochs to train
        :param data_processor: object, data processor to preprocess data
        :param callbacks: list, list of callbacks
        :param log_freq: int, log interval
        :param wandb_log: bool, log to wandb
        :param device: str, device to train on
        :param verbose: bool, verbose
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_epochs: int,
        data_processor: object = None,
        callbacks=None,
        log_freq: int = 1,
        device: str = None,
        verbose: bool = False,
    ):

        if callbacks:
            self.callbacks = callbacks

        else:
            self.callbacks = None

        self.model = model
        self.n_epochs = n_epochs
        self.data_processor = data_processor
        self.log_freq = log_freq
        self.device = device
        self.verbose = verbose

        if self.callbacks:
            self.callbacks.on_init_end(
                model=model,
                n_epochs=n_epochs,
                data_processor=data_processor,
                log_freq=log_freq,
                device=device,
                verbose=verbose,
            )

    def train(
        self,
        train_loader=None,
        test_loader=None,
        optimizer=None,
        scheduler=None,
        regularizer=None,
        train_loss=None,
        eval_loss=None,
    ):
        """Train the model on data from loaders

        Args
        ----
            :param train_loader: training data loader
            :param test_loader: testing data loader
            :param optimizer: optimizer algorithm
            :param scheduler: learning rate scheduler
            :param regularizer: model training regularizer
            :param train_loss: training loss / cost function
            :param test_loss: testing loss / cost function
        """

        if self.callbacks:
            self.callbacks.on_train_start(
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                regularizer=regularizer,
                train_loss=train_loss,
                eval_loss=eval_loss,
            )

        n_batches = len(train_loader)
        errors = None

        for epoch in range(self.n_epochs):
            if self.callbacks:
                self.callbacks.on_train_epoch_start(epoch=epoch)

            avg_train_loss: float = 0.0
            avg_lasso_loss: float = 0.0
            self.model.train()
            train_error: float = 0.0
            start_time = time.time()

            n_samples = 0
            for idx, sample in enumerate(train_loader):

                n_samples += sample["y"].shape[0]

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()

                sample = self.data_processor.preprocess(sample)
                output = self.model(sample["x"])
                output, sample = self.data_processor.postprocess(output, sample)

                loss = 0.0
                loss = train_loss(output, sample["y"])
                if regularizer:
                    loss += regularizer.loss

                loss.backward()
                del output

                optimizer.step()
                train_error += loss.item()

                with torch.no_grad():
                    avg_train_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss.item()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

            scheduler.step()
            end_time = time.time() - start_time

            train_error /= n_batches
            avg_train_loss /= n_samples

            if epoch % self.log_freq == 0 and self.verbose:
                if self.callbacks:
                    self.callbacks.on_before_eval(
                        idx=idx,
                        n_batches=n_batches,
                        train_error=train_error,
                        avg_train_loss=avg_train_loss,
                        avg_lasso_loss=avg_lasso_loss,
                    )

                errors = self.evaluate(eval_loss, test_loader)

            if self.callbacks:
                self.callbacks.on_train_epoch_end(
                    epoch=epoch,
                    n_batches=n_batches,
                    train_error=train_error,
                    avg_train_loss=avg_train_loss,
                    avg_lasso_loss=avg_lasso_loss,
                    time=end_time,
                )

        return errors

    def evaluate(
        self,
        eval_loss: dict,
        data_loader: torch.utils.data.DataLoader,
    ):
        """Evaluate model on passed losses

        Args
        ----
            :param eval_loss: dict, evaluation losses functions f(output and ground truth as input)
            :param loader: torch.utils.data.DataLoader, data loader
        """

        errors_dict = {}
        self.model.eval()
        n_batches = len(data_loader)

        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                n_samples += sample["y"].shape[0]
                sample = self.data_processor.preprocess(sample)
                output = self.model(sample["x"])
                output, sample = self.data_processor.postprocess(output, sample)

                for loss_name, loss in eval_loss.items():
                    loss_val = loss(output, sample["y"])
                    if self.callbacks:
                        self.callbacks.on_val_end(
                            idx=idx,
                            n_batches=n_batches,
                            loss_val=loss,
                            errors=errors_dict,
                            loss_name=loss_name,
                        )
                    errors_dict[loss_name] = loss_val
            del output

        return errors_dict
