# Reference: https://github.com/EugenHotaj/pytorch-generative/blob/master/pytorch_generative/models/kde.py

import abc
import torch
import pandas as pd
from math import pi


class Kernel(abc.ABC, torch.nn.Module):
    """Abstract base kernel object for all Kernels"""

    def __init__(self, bandwidth: float, device: str):
        super().__init__()
        self.bandwidth = bandwidth
        self.device = device

    @abc.abstractmethod
    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    @abc.abstractmethod
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

    @abc.abstractmethod
    def to(self, device):
        return NotImplementedError


class GaussianKernel(Kernel):
    """Gaussian smoothing kernel for Kernel Density Estimation"""

    def __init__(self, bandwidth: float, device: str, x: torch.Tensor):
        super().__init__(bandwidth, device)
        self.x = x
        self.to(device)

    def forward(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian kernel"""
        y = y.unsqueeze(1)
        x = x.unsqueeze(0)
        _, n, m = x.shape

        n = torch.tensor(n, device=self.device)
        self.bandwidth = torch.tensor(self.bandwidth, device=self.device)
        _pi = torch.tensor(pi, device=self.device)

        residual = y - x
        Z = (
            (0.5 * m * torch.log(2 * _pi))
            + (m * torch.log(self.bandwidth))
            + torch.log(n)
        )
        log_kernel = -0.5 * torch.norm((residual / self.bandwidth), p=2, dim=-1) ** 2
        return torch.logsumexp(log_kernel - Z, dim=-1)

    @torch.no_grad()
    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Draw samples from Gaussian kernel"""
        noise = torch.randn_like(x, device=self.device) * self.bandwidth
        return x + noise

    def to(self, device):
        """Move kernel to device"""
        self.x = self.x.to(device)
        return self


class KernelDensityEstimator(torch.nn.Module):
    """2D Kernel Density Estimator with Gaussian
    Kernel to fit data distributions"""

    def __init__(
        self,
        bandwidth: float,
        device: str,
        x: torch.Tensor,
        kernel: Kernel = GaussianKernel,
    ):
        super().__init__()
        self.bandwidth = bandwidth
        self.device = device
        self.kernel = kernel(bandwidth, device, x)
        self.x = x
        self.to(device)

    def to(self, device):
        self.device = device
        self.kernel.to(device)
        return self

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Compute KDE"""
        return self.kernel(y, self.x)

    @torch.no_grad()
    def sample(self, n_samples: int) -> torch.Tensor:
        """Sample from KDE"""
        max_idx = len(self.x)
        random_draws = torch.randint(0, max_idx, (n_samples,), device=self.device)
        return self.kernel.sample(self.x[random_draws])


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a 2D sample set from a standard normal distribution
    n_samples = 1000
    x = torch.randn(n_samples, 2, device=device)  # 2D samples

    # Initialize KernelDensityEstimator
    bandwidth = 1.0
    kde = KernelDensityEstimator(bandwidth, device, x)

    # Compute KDE for new 2D samples
    n_new_samples = 500
    new_samples = torch.randn(n_new_samples, 2, device=device)  # 2D samples
    kde_values = kde.forward(new_samples)

    # Generate new 2D samples from the estimated distribution
    sampled_values = kde.sample(n_new_samples)

    import matplotlib.pyplot as plt

    # Convert the tensor to numpy for plotting
    sampled_values_np = sampled_values.cpu().numpy()

    # Plot histogram
    plt.hist(sampled_values_np, bins=30, density=True)
    plt.title("Histogram of Generated Data")
    plt.show()
