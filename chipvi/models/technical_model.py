from __future__ import annotations

import torch
from torch.distributions import NegativeBinomial
from torch import nn

from chipvi.models.common import MLP
from chipvi.utils.distributions import get_torch_nb_dist


class TechNB_mu_r(nn.Module):
    """Technical noise model using Negative Binomial distribution.

    Predicts parameters for the NB distribution (mean and size) based on input covariates.
    """

    def __init__(
        self,
        covariate_dim: int,
        hidden_dims_mu: tuple[int, ...],
        hidden_dims_r: tuple[int, ...],
        constrain_mu_log_out: bool = False,  # New argument for backward compatibility
    ) -> None:
        super().__init__()
        self.dim_x = covariate_dim
        self.constrain_mu_log_out = constrain_mu_log_out

        output_activation_fn_mu = None
        if self.constrain_mu_log_out:
            # If constraining, set up scaler/shifter and use Tanh activation
            self.log_mu_scaler = 10.0
            self.log_mu_shifter = 5.0
            output_activation_fn_mu = nn.Tanh()
        
        # f: covariates -> log(mu_tech)
        self.f = MLP(
            dim_in=covariate_dim,
            dim_out=1,
            hidden_dims=hidden_dims_mu,
            activation_fn=nn.SiLU(),
            output_activation_fn=output_activation_fn_mu,
        )

        # g: {log(mu_tech), x_i} -> log(r_tech)
        self.g = MLP(
            dim_in=covariate_dim + 1,
            dim_out=1,
            hidden_dims=hidden_dims_r,
            activation_fn=nn.SiLU(),
            output_activation_fn=None,
        )

    def forward(self, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_mu_tech_raw = self.f(x_i)

        if self.constrain_mu_log_out:
            # Apply scaling and shifting if the model is constrained
            log_mu_tech = self.log_mu_scaler * log_mu_tech_raw + self.log_mu_shifter
        else:
            log_mu_tech = log_mu_tech_raw

        g_input = torch.cat((log_mu_tech, x_i), dim=1)
        log_r_tech = self.g(g_input)

        return torch.exp(log_mu_tech), torch.exp(log_r_tech)
    
    def predict_dist(
        self,
        x_i: torch.Tensor,
        detach: bool = False,
    ) -> NegativeBinomial:
        pred_mu, pred_r = self.forward(x_i)

        return get_torch_nb_dist(
            mu=pred_mu.detach() if detach else pred_mu,
            r=pred_r.detach() if detach else pred_r,
            )


class TechNB_r_p(nn.Module):
    def __init__(
            self,
            dim_x: int,
            hidden_dims_r: tuple[int, ...],
            hidden_dims_p: tuple[int, ...],
            ):
        super().__init__()
        self.dim_x = dim_x

        # f: covariates -> p_tech
        self.f = MLP(
                dim_in=dim_x,
                dim_out=1,
                hidden_dims=hidden_dims_p,
                activation_fn=nn.LeakyReLU(),
                output_activation_fn=nn.Sigmoid(),
                )
        # g: covariates -> log(r_tech)
        self.g = MLP(
                dim_in=dim_x,
                dim_out=1,
                hidden_dims=hidden_dims_r,
                activation_fn=nn.LeakyReLU(),
                output_activation_fn=None,
                )

    def forward(self, x_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.f(x_i)
        log_r_tech = self.g(x_i)

        return torch.exp(log_r_tech), p
    
    def predict_dist(
        self,
        x_i: torch.Tensor,
        detach: bool = False,
    ) -> NegativeBinomial:
        pred_p, pred_r = self.forward(x_i)
        return get_torch_nb_dist(
            p=pred_p.detach() if detach else pred_p,
            r=pred_r.detach() if detach else pred_r,
            )


class TechPoisson(nn.Module):
    """Technical noise model using Poisson distribution.

    Predicts the mean of the Poisson distribution from input covariates.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Network f: predicts log(mu_tech) from x_i
        # Input: x_i (dim_x)
        # Output: log_mu_tech (1)
        self.f = MLP(
            dim_in=input_dim,
            dim_out=1,
            hidden_dims=hidden_dims,
            activation_fn=nn.SiLU(),
            output_activation_fn=None,  # No output activation, as it directly outputs log_mu_tech
        )
        # TODO: SILU might not be the best choice to ensure large dynamic range

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        """Forward pass to get the mean of the technical Poisson.

        Args:
            x_i (torch.Tensor): Input covariates for a batch of genomic bins,
                                shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Mean of the Poisson distribution,
                          shape (batch_size, 1).

        """
        log_mu_tech = self.f(x_i)  # Shape: (batch_size, 1)

        return log_mu_tech
