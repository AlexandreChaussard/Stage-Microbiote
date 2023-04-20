import autograd.numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(y_hat, y):
    return np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def derivative_binary_cross_entropy(y_hat, y, X):
    return X.T @ (y_hat.reshape(-1, 1) - y.reshape(-1, 1))


def onehot(i, size):
    v = np.zeros(size)
    v[i] = 1
    return v


def identify_permutation(v1, v2):
    """
    Function to identify the permutation vector from v1 to v2.
    Example:
        identify([-1,2], [2,-1])
    Output:
        [1, 0]
    """
    permutation = [1, 0]
    dist = np.linalg.norm(v1 - v2)
    if np.linalg.norm(v1 - v2[permutation]) < dist:
        return permutation
    else:
        return [0, 1]


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()