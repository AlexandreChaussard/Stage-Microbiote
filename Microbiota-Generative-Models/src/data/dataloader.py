import warnings

import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt
from src.utils import AbundanceTree, Tree

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


def microbiota_abundance_trees(precision_max=6, path='.'):
    # First, we fetch the individuals id
    df = get_NIPICOL(0, path)
    individuals = df.columns

    # This maps an index to a bacteria name
    mapping_index = {}
    # This maps a bacteria to its parent
    mapping_parent = {}
    # For each individual, we store the value of the nodes
    node_abundance = {}
    # Current index being explored while building the tree
    current_index = 0
    # List the current possible parents of the bacteria
    current_parents = []

    # We loop over the precisions, building up the general tree from the root to the leaves
    for precision in range(0, precision_max + 1):
        # We fetch the list of the bacteria at the precision layer
        df = get_NIPICOL(precision, path)
        # We create a list of future parents to replace the current ones
        future_parents = []
        for bacteria_name in df.index:
            # We prepare the future prents list by adding the current layer
            future_parents.append(bacteria_name)
            # We register the bacteria in both maps of index and parents
            if bacteria_name not in mapping_index.values():
                mapping_index[bacteria_name] = current_index
                # We fetch the parent of the node
                for parent in current_parents:
                    if parent in bacteria_name:
                        mapping_parent[current_index] = mapping_index[parent]
                        break
                # We update the index
                current_index += 1

            # We fetch the bacteria index
            bacteria_index = mapping_index[bacteria_name]
            # We store the bacteria abundance for every individual
            node_abundance[bacteria_index] = df.loc[bacteria_name]

        # We update the current parents for the next layer
        current_parents = future_parents

    # We look for the size of the adjacent matrix
    size = len(mapping_index)

    # Using the mapping, we can now build the tree
    adjacent_matrix = np.zeros((size, size))
    # We run through the parents to find the children and build the adjacent matrix row after row
    for parent_index in mapping_parent.values():
        # For each parent, we gather the children indexes to build the adjacent vector of that node
        children_index = []
        for node_index in mapping_parent.keys():
            if mapping_parent[node_index] == parent_index:
                children_index.append(node_index)
        # This is not yet the adjacent row, since it's something like [1, 6, 12] instead of [0, 1, 0, 0, ...] vector
        # But we need to gather these to create the full matrix after that
        adjacent_matrix[parent_index][children_index] = 1

    # Now we can build the global tree architecture
    global_tree = Tree(adjancent_matrix=adjacent_matrix)

    # Then, for each individual, we build a corresponding abundance tree
    abundance_trees = {}
    for i, individual in enumerate(individuals):
        # We roam through each column of the abundance which corresponds to a given individual
        # This is the vector of abundance for a given individual for each bacteria, identified by their index
        abundance_values = np.zeros(len(node_abundance))
        for abundance_index, abundance_values_per_indiv in node_abundance.items():
            abundance_values[abundance_index] = abundance_values_per_indiv[i]

        # Now we can build the abundance tree
        abundance_trees[individual] = AbundanceTree(adjacent_matrix, abundance_values)

    return global_tree, abundance_trees
