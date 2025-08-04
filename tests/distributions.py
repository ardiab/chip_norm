import torch
import scipy.stats as sps
import numpy as np

from chipvi.utils.distributions import nb_cdf_r_p, poisson_cdf


def test_nb_r_p_cdf():
    np.random.seed(42)
    obs = np.random.randint(0, 100, size=(1000,))
    rs = np.random.uniform(1, 100, size=(1000,))
    ps = np.random.uniform(1e-6, 1 - 1e-6, size=(1000,))
    cdf = nb_cdf_r_p(
        x=torch.tensor(obs),  # number of successes
        r=torch.tensor(rs),  #  number of failures
        p=torch.tensor(ps),  # probability of success
    )
    # Scipy's implementation of the negative binomial is the inverse
    # of pytorch - instead of fixing the number of failures and modeling
    # the number of successes, it fixed the number of successes and models
    # the number of failures. We can standardize the behavior by inverting
    # failure and success (by taking the complement of the success probability).
    sps_cdf = sps.nbinom.cdf(
        n=obs, # number of successes
        k=rs,  # number of failures
        p=1 - ps,  # probability of success
    )
    assert cdf.shape == (1000, )
    # print(cdf.numpy())
    # print(sps_cdf)
    # print("Max diff: ", np.max(np.abs(cdf.numpy() - sps_cdf)))
    assert np.allclose(cdf.numpy()[:10], sps_cdf[:10], atol=1e-2)

# def test_poisson_cdf():
#     device = torch.device("cpu")
#     x = torch.arange(0, 10)
#     rate = torch.tensor([1.0, 2.0, 3.0])
#     cdf = poisson_cdf(x, rate, device=device)
#     assert cdf.shape == (3, 10)

# if __name__ == "__main__":
#     test_nb_cdf()
#     test_poisson_cdf()
