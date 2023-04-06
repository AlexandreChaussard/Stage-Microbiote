import numpy as np


def generate_gaussian(n_samples, d, mu_list, sigma_list):
    X = []
    Y = []
    mu_list = np.array(mu_list)
    if (d > 1 and type(mu_list[0]) != np.ndarray) or d != mu_list.shape[1]:
        print(f"Dimension error: d = {d} but mu is of shape {mu_list.shape}")
        return

    for i in range(len(mu_list)):
        mu = mu_list[i]
        sigma_matrix = np.diag(sigma_list[i])
        samples = mu + np.random.randn(n_samples, d) @ sigma_matrix
        for sample in samples:
            X.append(sample)
        Y.append([i] * n_samples)

    X = np.array(X)
    Y = np.array(Y)

    Y = Y.reshape(-1)
    return X, Y
