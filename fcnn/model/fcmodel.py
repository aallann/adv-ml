import torch
from typing import Union, List

from ..layers.fc import FullyConnected, FullyConnectedBlock


class FCNN(torch.nn.Module):
    """Simple Fully Connected (Dense) NN regression model.
    Creates a fully connected neural network with n_layers - 2
    hidden layers with dimensions (hidden_dim).

    Args
    ----
        :param in_dim: int, input dimension
        :param out_dim: int, output dimension
        :param hidden_dim: list, hidden dimensions
        :param n_layers: int, number of layers
        :param bias: bool, default is True
        :param device: str, default is "cpu
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Union[int, List[int]],
        n_layers: int,
        device="cpu",
        bias: bool = True,
    ):

        if isinstance(hidden_dim, list):
            if len(hidden_dim) != n_layers - 2:
                raise ValueError(
                    "Number of hidden dimensions must be equal to n_layers - 2"
                )
        elif isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim] * (n_layers - 2)

        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.input_layer = FullyConnected(
            in_dim, hidden_dim[0], bias=bias, device=device
        )

        self.hidden_layers = FullyConnectedBlock(hidden_dim, bias=bias, device=device)

        self.output_layer = FullyConnected(
            hidden_dim[-1], out_dim, bias=bias, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
