from __future__ import annotations

import logging

import torch
import torch.distributions as D

logger = logging.getLogger(__name__)


def get_torch_nb_dist(
        r: torch.Tensor,
        mu: torch.Tensor | None = None,
        p: torch.Tensor | None = None,
) -> D.NegativeBinomial:
    if mu is None and p is None:
        raise ValueError("Either mu or p must be provided.")

    # Validate r parameter
    if torch.isnan(r).any() or torch.isinf(r).any():
        raise ValueError("r contains NaN or infinite values.")
    if (r <= 0).any():
        raise ValueError("r must be positive.")
        
    if mu is not None:
        if r.squeeze().shape != mu.squeeze().shape:
            raise ValueError("r and mu must have the same shape.")
        
        # Validate mu parameter
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            raise ValueError("mu contains NaN or infinite values.")
        if (mu < 0).any():
            raise ValueError("mu must be non-negative.")

        p = r / (r + mu)
        return D.NegativeBinomial(
            total_count=r.squeeze(),
            probs=p.squeeze(),
        )
    elif p is not None:
        if r.squeeze().shape != p.squeeze().shape:
            raise ValueError("r and p must have the same shape.")
        return D.NegativeBinomial(
            total_count=r.squeeze(),
            probs=p.squeeze(),
        )
    else:
        raise ValueError("Either mu or p must be provided.")


def compute_numeric_cdf(
    dist: D.Distribution,
    x: torch.Tensor,
) -> torch.Tensor:
    x = torch.floor(x).to(torch.int32)
    max_k = x.max().item()
    ks = torch.arange(0, max_k + 1, device=x.device).squeeze()
    if isinstance(dist, D.NegativeBinomial):
        dist_reshaped = D.NegativeBinomial(
            total_count=dist.total_count.unsqueeze(1),
            probs=dist.probs.unsqueeze(1),
        )
    else:
        raise ValueError(f"Unsupported distribution: {type(dist)}")
    cdf_table = torch.cumsum(torch.exp(dist_reshaped.log_prob(ks)), dim=1)  # (n_samples, max_k + 1)
    result = cdf_table[torch.arange(len(x)), x]  # (n_samples, )
    return result


def nb_log_prob_mu_r(
    y: torch.Tensor,
    mu: torch.Tensor,
    r: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mu = mu.clamp(min=eps, max=1e6)
    r = r.clamp(min=eps, max=1e6)

    safe_mu_over_r = (mu / r).clamp(min=eps)
    logits = torch.log(safe_mu_over_r)

    dist = D.NegativeBinomial(total_count=r, logits=logits)
    return dist.log_prob(y)


def nb_log_prob_r_p(
    y: torch.Tensor,
    r: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-8,
    max_r: float = 1e6,
) -> torch.Tensor:
    """
    args:
        y: number of successes
        r: number of failures
        p: probability of success
    """
    if r.min() < eps:
        # logger.warning(f"r contains values less than {eps}. Clamping to {eps}.")
        r = r.clamp(min=eps)
    if r.max() > max_r:
        # logger.warning(f"r contains values greater than {max_r}. Clamping to {max_r}.")
        r = r.clamp(max=max_r)
    if p.min() < eps:
        # logger.warning(f"p contains values less than {eps}. Clamping to {eps}.")
        p = p.clamp(min=eps)
    if p.max() > 1 - eps:
        # logger.warning(f"p contains values greater than {1 - eps}. Clamping to {1 - eps}.")
        p = p.clamp(max=1 - eps)

    if torch.isnan(r).any() or torch.isnan(p).any():
        logger.warning(f"r or p contains NaN values. Returning -inf.")
        return torch.tensor(-float("inf"))

    try:
        dist = D.NegativeBinomial(
                total_count=r,  # number of failures until experiment stops
                probs=p  # probability of success
                )
    except Exception as e:
        print("r", r.min(), r.mean(), r.max())
        print("p", p.min(), p.mean(), p.max())
    return dist.log_prob(y)


def poisson_log_prob(
    y: torch.Tensor,
    mu: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not (mu > 0).all():
        logger.warning("mu contains non-positive values. Adding epsilon.")
        mu = mu.clamp(min=eps)

    dist = D.Poisson(rate=mu)
    return dist.log_prob(y)


def poisson_cdf(
    x: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    # Poisson.log_prob only works for integer values
    x = torch.floor(x).to(torch.int32)  # (n_samples, )
    max_k = x.max().item()

    ks = torch.arange(0, max_k + 1, device=x.device).view(1, -1)  # (1, max_k + 1)

    # Different rate per sample
    rate = rate.view(-1, 1)  # (n_samples, 1)

    log_pmf = D.Poisson(rate=rate).log_prob(ks)  # (n_samples, max_k + 1)
    pmf = torch.exp(log_pmf)  # (n_samples, max_k + 1)

    cdf_table = torch.cumsum(pmf, dim=1)  # (n_samples, max_k + 1)

    # Pick out the cumulative mass for each sample at the observed value x
    result = cdf_table[torch.arange(len(x)), x]  # (n_samples, )

    return result


def nb_cdf_r_p(
    x: torch.Tensor,
    r: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    x = torch.floor(x).to(torch.int32)
    max_k = x.max().item()
    ks = torch.arange(0, max_k + 1, device=x.device).view(1, -1)  # (1, max_k + 1)
    counts = r.view(-1, 1)  # (n_samples, 1)
    probs = p.view(-1, 1)  # (n_samples, 1)
    log_pmf = D.NegativeBinomial(total_count=counts, probs=probs,).log_prob(ks)  # (n_samples, max_k + 1)
    pmf = torch.exp(log_pmf)  # (n_samples, max_k + 1)
    cdf_table = torch.cumsum(pmf, dim=1)  # (n_samples, max_k + 1)
    result = cdf_table[torch.arange(len(x)), x]  # (n_samples, )

    return result


# def nb_cdf_mu_r(x, mu, r, device: torch.device | None = None):
#     """
#     Compute CDF of NegativeBinomial(mu, r) at value x.
#     The distribution is parameterized by mean 'mu' and 'r' (number of failures).
#     P(X=k) = C(k+r-1, k) * p^r * (1-p)^k, where p = r / (mu+r).
#     X is the number of successes given r failures.

#     x: integer value(s) to evaluate CDF at (can be scalar or tensor).
#     mu: mean of the Negative Binomial distribution (mu > 0). Can be scalar or tensor.
#     r: number of failures (dispersion parameter, r > 0). Can be scalar or tensor.
#     """
#     # Ensure inputs are tensors and determine their data type and device
#     # We'll use float for calculations and then convert x to int for indexing.
#     if not isinstance(x, torch.Tensor):
#         # Try to infer dtype and device from mu or r if they are tensors
#         if isinstance(mu, torch.Tensor):
#             dtype = mu.dtype
#         elif isinstance(r, torch.Tensor):
#             dtype = r.dtype
#         else:  # All scalars
#             dtype = torch.float32
#         if device is None:
#             device = x.device
#         x = torch.as_tensor(x, dtype=dtype, device=device)
#     else:
#         dtype = x.dtype

#     if device is None:
#         device = x.device
#     mu = torch.as_tensor(mu, dtype=dtype, device=device)
#     r = torch.as_tensor(r, dtype=dtype, device=device)

#     # --- Input Validation ---
#     if torch.any(mu < 0):
#         raise ValueError("mu (mean) must be non-negative.")
#     if torch.any(r <= 0):
#         raise ValueError("r (number of failures) must be positive.")

#     # --- Broadcasting ---
#     # Determine common broadcast shape for x, mu, r
#     try:
#         common_shape = torch.broadcast_shapes(x.shape, mu.shape, r.shape)
#     except RuntimeError:
#         raise ValueError(
#             f"x, mu, r shapes ({x.shape}, {mu.shape}, {r.shape}) are not broadcastable."
#         )

#     # Broadcast all inputs to the common shape
#     x_b = x.broadcast_to(common_shape)
#     mu_b = mu.broadcast_to(common_shape)
#     r_b = r.broadcast_to(common_shape)

#     # Initialize result tensor with zeros (CDF for x < 0 is 0)
#     result = torch.zeros_like(x_b, dtype=dtype)

#     # Floor x values (as CDF is for discrete distributions)
#     x_floor = torch.floor(x_b)

#     # Identify non-negative x values for which CDF needs to be computed
#     # CDF for x < 0 is 0, already handled by initialization.
#     # CDF for mu = 0: P(X=0)=1, P(X>0)=0. So CDF is 1 for x >= 0.

#     # Mask for elements that need actual PMF summation
#     # These are x >= 0 AND mu > 0
#     compute_mask = (x_floor >= 0) & (mu_b > 0)

#     # Handle mu = 0 case separately: CDF is 1 for x >= 0
#     mu_zero_mask = (x_floor >= 0) & (mu_b == 0)
#     result[mu_zero_mask] = 1.0

#     if not torch.any(compute_mask):  # No elements need full computation
#         return result

#     # --- Prepare for PMF calculation for elements in compute_mask ---
#     # Flatten the tensors for processing, selecting only elements in compute_mask
#     x_proc = x_floor[compute_mask].to(torch.int64)  # Convert to int64 for indexing
#     mu_proc = mu_b[compute_mask]  # Shape: (N_selected,)
#     r_proc = r_b[compute_mask]  # Shape: (N_selected,)

#     # Determine max k needed for PMF calculations
#     max_k = x_proc.max().item()
#     # ks: tensor of [0, 1, ..., max_k]
#     if device is None:
#         device = x.device
#     ks = torch.arange(0, max_k + 1, device=device).view(1, -1)  # Shape: (1, max_k+1)

#     # Reshape mu_proc and r_proc to (N_selected, 1) for broadcasting with ks
#     mu_proc = mu_proc.view(-1, 1)
#     r_proc = r_proc.view(-1, 1)

#     # --- Parameter conversion and PMF calculation (in log-space) ---
#     # p = r / (mu + r)
#     # 1-p = mu / (mu + r)
#     # Add small epsilon for numerical stability with log, especially if mu or r are near zero
#     # (though mu_proc > 0 is guaranteed here, r_proc > 0 by input validation)
#     eps = torch.finfo(dtype).eps

#     # log(p) = log(r) - log(mu+r)
#     log_p_val = torch.log(r_proc + eps) - torch.log(mu_proc + r_proc + eps)
#     # log(1-p) = log(mu) - log(mu+r)
#     log_1_minus_p_val = torch.log(mu_proc + eps) - torch.log(mu_proc + r_proc + eps)

#     # Log PMF: lgamma(k+r) - lgamma(k+1) - lgamma(r) + r*log(p) + k*log(1-p)
#     # ks: (1, max_k+1)
#     # r_proc, log_p_val, log_1_minus_p_val: (N_selected, 1)
#     # All lgamma terms will broadcast correctly.
#     log_pmf_table = torch.distributions.NegativeBinomial(
#         total_count=r_proc, logits=log_p_val
#     ).log_prob(ks)
#     # log_pmf_table = (torch.lgamma(ks + r_proc) -       # Shape: (N_selected, max_k+1)
#     #                  torch.lgamma(ks + 1) -            # Shape: (1, max_k+1), broadcasts
#     #                  torch.lgamma(r_proc) +            # Shape: (N_selected, 1), broadcasts
#     #                  r_proc * log_p_val +              # Shape: (N_selected, 1), broadcasts
#     #                  ks * log_1_minus_p_val)           # Shape: (N_selected, max_k+1)

#     pmf_table = torch.exp(log_pmf_table)  # Shape: (N_selected, max_k+1)

#     # Cumulative sum to get CDF table
#     cdf_table = torch.cumsum(pmf_table, dim=1)  # Shape: (N_selected, max_k+1)

#     # Gather the CDF values corresponding to each x_proc
#     # Create row indices for gathering: [0, 1, ..., N_selected-1]
#     idx_rows = torch.arange(len(x_proc), device=device)
#     # x_proc contains column indices
#     cdf_values_for_proc = cdf_table[idx_rows, x_proc]

#     # Place these computed CDF values into the correct positions in the result tensor
#     result[compute_mask] = cdf_values_for_proc

#     return result
#     return torch.clamp(result, min=0, max=1)
