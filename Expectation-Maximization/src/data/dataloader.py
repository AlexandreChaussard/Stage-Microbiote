import warnings

import numpy as np
import os
import pandas as pd
import scipy

import src.utils.functions as functions
from scipy.stats import dirichlet

warnings.filterwarnings("ignore")


def get_train_test(X, y, n_train):
    indexes = np.arange(0, len(X), 1)
    np.random.shuffle(indexes)

    X_train, y_train = X[indexes[:n_train]], y[indexes[:n_train]]
    X_test, y_test = X[indexes[n_train:]], y[indexes[n_train:]]
    return X_train, y_train, X_test, y_test


def generate_dirichlet(n_samples, alpha_list, seed=None):
    X = []
    Y = []
    alpha_list = np.array(alpha_list)

    for i in range(len(alpha_list)):
        alpha = alpha_list[i]
        samples = dirichlet.rvs(alpha, n_samples, random_state=seed + i)
        for sample in samples:
            X.append(sample)
            Y.append(i)

    X = np.array(X)
    Y = np.array(Y)

    Y = Y.reshape(-1)
    return X, Y


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

    precisions = ['d', 'p', 'c', 'o', 'f', 'g', 's']
    if precision >= len(precisions) and precision >= 0:
        print(f"Precision must be between 0 and {len(precisions) - 1}.")
        return None

    rdf = df[df['ASV_ID'].str.contains(f'{precisions[precision]}__')]
    if precision < len(precisions) - 1:
        rdf['ASV_ID'] = rdf['ASV_ID'].str.split(f'{precisions[precision + 1]}__', n=1, expand=True)[0]

    def remove_last_sep(asv_id):
        i = -1
        while '|' in asv_id and asv_id[i] != '|':
            i -= 1
        return asv_id[:i]

    rdf['ASV_ID'] = rdf['ASV_ID'].apply(remove_last_sep)
    rdf = rdf.groupby(by='ASV_ID').sum()

    return rdf


def get_mapping_NIPICOL(path="./"):
    df = pd.read_csv(
        os.path.join(path, "mapping_nipicol.txt"), sep="	"
    )
    df.columns = ['id'] + df.columns[1:].tolist()
    df = df.drop(index=df[df.id.str.contains('bis')].index)
    return df


def microbiota_features_to_image(precision_max=6, path='./'):
    # First, we fetch the highest possible precision dataframe so we get the base of the image and how many to create
    precision = precision_max
    df = get_NIPICOL(precision, path)
    # We create one image per individual in the dataset
    # The image width is determined by the maximum amount of unique bacteria within each precision set
    img_width = len(df)
    n_unique_bacteria_per_precision = {}
    for p in range(0, precision_max + 1):
        d = get_NIPICOL(p, path)
        img_width = max(img_width, len(d))
        n_unique_bacteria_per_precision[p] = len(d)
    # The image heigh is determined by the maximum precision
    img_heigh = precision_max + 1
    # The number of images is given by the number of individuals in the dataset
    images = [np.zeros((img_heigh, img_width)) for _ in range(len(df.columns))]
    df_imgs = pd.DataFrame(columns=['microbiota_img'])

    while precision >= 0:
        # i is the index of the taxon in the image row
        for i, (name, taxon) in enumerate(df.iterrows()):
            # k is the index of the individual represented by its id
            for k, individual_id in enumerate(taxon.index):
                # The current image row is given by the precision
                images[k][precision][i + img_width // 2 - n_unique_bacteria_per_precision[precision] + 1] = taxon.loc[
                    individual_id]
                df_imgs.loc[individual_id] = [images[k]]
        # The we continue up to the top of the phylogenetic tree
        precision -= 1
        # If we have reached the top we should stop
        if precision >= 0:
            df = get_NIPICOL(precision, path)

    return df_imgs
