import numpy as np
import os
import pandas as pd
import src.utils.functions as functions


def generate_gaussian(n_samples, d, mu_list, sigma_list, seed=None):
    np.random.seed(seed)

    X = []
    Y = []
    mu_list = np.array(mu_list)
    if (d > 1 and type(mu_list[0]) != np.ndarray) or (len(mu_list) > 1 and d != mu_list.shape[1]):
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


def generate_conditional_binary_observations(X, Z, seed=None, returnParams=False):
    """
    Generate conditional binary observations Y such that:
    P(Y = 1 | X, Z=k) = sigmoid(W_e_k^T e_k + W_x_k^T X_i)

    Z should be a discrete variable
    """
    np.random.seed(seed)

    # Initialize the weights of the observation
    onehot_length = len(np.unique(Z))
    W_e = np.random.randn(onehot_length, onehot_length)
    W_x = np.random.randn(onehot_length, X.shape[1])

    # Observations
    y = np.zeros(Z.shape)

    for i in range(len(Z)):
        z_i = Z[i]
        x_i = X[i]
        W_e_k = W_e[z_i]
        W_x_k = W_x[z_i]

        e_k = functions.onehot(z_i, onehot_length)

        proba = functions.sigmoid(W_e_k.dot(e_k) + W_x_k.dot(x_i))
        y[i] = (proba > 0.5) * 1

    if returnParams:
        return y, W_e, W_x
    return y



def get_NIPICOL(path="./"):
    df = pd.read_excel(
        os.path.join(path, "data_NIPICOL.xlsx")
    )
    # This column is empty
    df = df.drop(columns=["taxonomy"])
    # This column seems to be a concatenation of the "Taxon" and "Feature ID"
    df = df.drop(columns=["Unnamed: 0"])
    return df


def get_POP(path="./"):
    df = pd.read_excel(
        os.path.join("data", "data_POP.xlsx")
    )
    # The first column seems like the concatenation of the "Taxon" and "Feature ID"
    df[["Taxon", "Feature ID"]] = df["Unnamed: 0"].str.split(';otu_', expand=True)
    df = df.drop(columns=["Unnamed: 0"])
    return df


def get_train_test(X, y, n_train):
    indexes = np.arange(0, len(X), 1)
    np.random.shuffle(indexes)

    X_train, y_train = X[indexes[:n_train]], y[indexes[:n_train]]
    X_test, y_test = X[indexes[n_train:]], y[indexes[n_train:]]
    return X_train, y_train, X_test, y_test
