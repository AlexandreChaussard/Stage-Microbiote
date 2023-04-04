import numpy as np


def generate_gaussian(n_samples, d, mu_list, sigma_list):
    X = []
    Y = []
    for i in range(len(mu_list)):
        mu = mu_list[i]
        sigma = sigma_list[i]
        samples = sigma * np.random.randn(n_samples, d) + mu
        for sample in samples:
            X.append(sample)
        Y.append([i] * n_samples)

    X = np.array(X)
    Y = np.array(Y)

    Y = Y.reshape(-1)
    return X, Y
