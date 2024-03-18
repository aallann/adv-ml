import abc
import pandas as pd


class AbstractDataProcessor(abc.ABC):
    """Data processing abstract base class for pre-
    and post-processing data for training/inference"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def preprocess(self, data_frame: pd.DataFrame):
        """Preprocess data"""
        pass

    @abc.abstractmethod
    def postprocess(self, data_frame: pd.DataFrame):
        """Postprocess data"""
        pass


class DataProcessor(AbstractDataProcessor):
    """Data processor for training/inference data

    Args
    ----
        :param input_encoder: input encoder
        :param n_features: number of input features
    """

    def __init__(self, input_encoder=None, output_encoder=None):
        super().__init__()
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder

    def preprocess(self, data_frame: pd.DataFrame):
        if self.input_encoder:
            data_frame = self.input_encoder.transform(data_frame)

        return data_frame

    def postprocess(self, data_frame: pd.DataFrame):
        if self.output_encoder:
            data_frame = self.output_encoder.inverse_transform(data_frame)
            print(f"data_frame columns after inverse_transform: {data_frame.columns}")

        return data_frame
