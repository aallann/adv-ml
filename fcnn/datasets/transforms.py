import abc
import torch
from typing import List


class Normaliser(abc.ABC, torch.nn.Module):
    """Normaliser base class"""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def fit(self, x: torch.Tensor):
        pass

    @abc.abstractmethod
    def transform(self, x: torch.Tensor):
        pass

    @abc.abstractmethod
    def inverse_transform(self, x: torch.Tensor):
        pass

    @abc.abstractmethod
    def cuda(self):
        pass

    @abc.abstractmethod
    def cpu(self):
        pass

    @abc.abstractmethod
    def to(self, device):
        pass


class UnitGaussianNormaliser(Normaliser):
    """Data normaliser for unit Gaussian distribution

    Args
    ----
        :param input: input tensor
        :param dim: dimensions to normalise
    """

    def __init__(self, dims: list = [0]):
        super().__init__()
        self.dims = dims
        self.mean = None
        self.std_dev = None

    def fit(self, input: torch.Tensor):
        n_samples, *shape = input.shape
        self.sample_shape = shape
        self.mean = torch.mean(input, self.dims, keepdim=True).squeeze(0)
        self.std_dev = torch.std(input, self.dims, keepdim=True).squeeze(0)
        print(self.mean, self.std_dev)
        self.eps = 0.001

    def transform(self, input: torch.Tensor):
        input -= self.mean
        input /= self.std_dev  # + self.eps
        return input

    def inverse_transform(self, input: torch.Tensor):
        input *= self.std_dev
        input += self.mean
        return input

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std_dev = self.std_dev.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std_dev = self.std_dev.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std_dev = self.std_dev.to(device)
        return self
