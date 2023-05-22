import warnings

import numpy as np
import os
import pandas as pd

warnings.filterwarnings("ignore")


def get_train_test(X, y, n_train):
    indexes = np.arange(0, len(X), 1)
    np.random.shuffle(indexes)

    X_train, y_train = X[indexes[:n_train]], y[indexes[:n_train]]
    X_test, y_test = X[indexes[n_train:]], y[indexes[n_train:]]
    return X_train, y_train, X_test, y_test


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
