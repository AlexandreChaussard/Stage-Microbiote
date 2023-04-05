import scipy.stats as stats
import numpy as np


def pdf_gaussian(x, mu, sigma):
    if type(mu) != np.ndarray:
        n_features = len(x)
        if type(x[0]) != np.float64:
            n_features = len(x[0])
        mu_list = [mu] * n_features
        sigma_matrix = sigma * np.eye(n_features)
    else:
        mu_list = mu
        sigma_matrix = np.diag(sigma)
    return stats.multivariate_normal(mean=mu_list, cov=sigma_matrix ** 2).pdf(x)
