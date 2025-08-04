from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) network."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation_fn: nn.Module,
        output_activation_fn: nn.Module | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            dim_in (int): Input dimension.
            dim_out (int): Output dimension.
            hidden_dims (tuple[int, ...] | list[int]): Tuple or list of hidden layer dimensions.
            activation_fn (nn.Module, optional): Activation function for hidden layers.
                                                 Defaults to nn.SiLU().
            output_activation_fn (nn.Module, optional): Activation function for the output layer.
                                                        Defaults to None (linear output).

        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        layers = []
        current_dim = dim_in
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(self.activation_fn)
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, self.dim_out))
        if self.output_activation_fn is not None:
            layers.append(self.output_activation_fn)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, dim_out).

        """
        return self.network(x)
