import torch
from typing import List


class FullyConnected(torch.nn.Module):
    """Fully connected NN layer"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        device="cpu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.layer = torch.nn.Linear(in_dim, out_dim, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FullyConnectedBlock(torch.nn.Module):
    """Multilayer fully connected NN"""

    def __init__(
        self,
        dim: List[int],
        bias: bool = True,
        device="cpu",
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for in_dim, out_dim in zip(dim[:-1], dim[1:]):
            self.layers.append(
                torch.nn.Linear(in_dim, out_dim, bias=bias, device=device)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.nn.functional.relu(layer(x))
        return x
