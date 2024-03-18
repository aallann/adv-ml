import abc
import torch


class AbstractDataProcessor(abc.ABC, torch.nn.Module):
    """Data processing abstract base class for pre-
    and post-processing data during training/inference"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def preprocess(self, input: torch.Tensor):
        """Preprocess data"""
        pass

    @abc.abstractmethod
    def postprocess(self, input: torch.Tensor):
        """Postprocess data"""
        pass

    @abc.abstractmethod
    def wrap(self, model):
        """Wrap model"""
        pass

    @abc.abstractmethod
    def forward(self, input: torch.Tensor):
        """Forward pass"""
        pass

    @abc.abstractmethod
    def to(self, device):
        pass


class DataProcessor(AbstractDataProcessor):
    """Data processor for training data

    Args
    ----
        :param input_encoder: input encoder
        :param output_encoder: output encoder
        :param n_features: number of input features
    """

    def __init__(self, input_encoder=None, output_encoder=None, n_features=None):
        super().__init__()
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.n_features = n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def preprocess(self, data_dict: dict):
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.input_encoder:
            x = self.input_encoder.transform(x)
        if self.output_encoder and self.train:
            y = self.output_encoder.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y
        return data_dict

    def postprocess(self, output: torch.Tensor, data_dict: dict):
        y = data_dict["y"]
        if self.output_encoder and not self.train:
            output = self.output_encoder.inverse_transform(output)
            y = self.output_encoder.inverse_transform(y)

        data_dict["y"] = y
        return output, data_dict

    def forward(self, **data_dict: dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output, data_dict = self.postprocess(output, data_dict)
        return output, data_dict

    def wrap(self, model):
        self.model = model
        return model

    def to(self, device):
        if self.input_encoder:
            self.input_encoder.to(device)
        if self.output_encoder:
            self.output_encoder.to(device)
