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


def generate_conditional_binary_observations(X, Z, W_e=None, W_x=None, seed=None, returnParams=False):
    """
    Generate conditional binary observations Y such that:
    P(Y = 1 | X, Z=k) = sigmoid(W_e_k^T e_k + W_x_k^T X_i)

    Z should be a discrete variable
    """
    np.random.seed(seed)

    # Initialize the weights of the observation
    onehot_length = len(np.unique(Z))
    if W_e is None:
        W_e = np.random.randn(onehot_length, onehot_length)
    if W_x is None:
        W_x = np.random.randn(onehot_length, X.shape[1])

    # Observations
    y = np.zeros(Z.shape, np.int32)

    for i in range(len(Z)):
        k = Z[i]
        x_i = X[i]

        e_k = functions.onehot(k, onehot_length)

        proba = functions.sigmoid(W_e[k].dot(e_k) + W_x[k].dot(x_i))
        eps = np.random.random()
        y[i] = (eps < proba) * 1

    if returnParams:
        return y, W_e, W_x
    return y


def get_NIPICOL(precision, path="./"):
    df = pd.read_csv(
        os.path.join(path, "nipicol_asv.txt"), sep="	"
    )
    df = df.drop(columns=df.columns[df.columns.str.contains('bis')])

    precisions = ['d', 'p', 'c', 'o', 'f', 'g']
    if precision >= len(precisions):
        print(f"Precision must be between 0 and {len(precisions)-1}.")
        return None

    rdf = df[df['ASV_ID'].str.contains(f'{precisions[precision]}__')]
    if precision < len(precisions) - 1:
        rdf['ASV_ID'] = rdf['ASV_ID'].str.split(f'{precisions[precision + 1]}__', n=1, expand=True)[0]
        rdf = rdf.groupby(by='ASV_ID').sum()

    return rdf


def get_mapping_NIPICOL(path="./"):
    df = pd.read_csv(
        os.path.join(path, "mapping_nipicol.txt"), sep="	"
    )
    df.columns = ['id'] + df.columns[1:].tolist()
    df = df.drop(index=df[df.id.str.contains('bis')].index)
    return df


def get_train_test(X, y, n_train):
    indexes = np.arange(0, len(X), 1)
    np.random.shuffle(indexes)

    X_train, y_train = X[indexes[:n_train]], y[indexes[:n_train]]
    X_test, y_test = X[indexes[n_train:]], y[indexes[n_train:]]
    return X_train, y_train, X_test, y_test
