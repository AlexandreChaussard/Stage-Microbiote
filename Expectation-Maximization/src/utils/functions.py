import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(y_hat, y):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)).mean()


def derivative_binary_cross_entropy(y_hat, y, X):
    return X.T @ (y_hat.reshape(-1, 1) - y.reshape(-1, 1))
