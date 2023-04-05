import scipy.stats as stats
import numpy as np


def pdf_gaussian(x, mu, sigma):
    n_features = 1
    if type(x[0]) != np.float64:
        n_features = len(x[0])
    mu_list = [mu] * n_features
    sigma_matrix = sigma * np.eye(n_features)
    return stats.multivariate_normal(mean=mu_list, cov=sigma_matrix ** 2).pdf(x)
