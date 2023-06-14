from src.utils import AbundanceTree, Tree
from src.utils.functions import digamma, invdigamma
import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt


class DirichletAbundanceTreePosterior:

    def __init__(self, global_tree, dirichlet_parameters, seed=None):
        self.global_tree = global_tree
        self.seed = seed
        np.random.seed(seed)

        for node in self.global_tree.nodes:
            n_children = len(node.children)
            if node.index not in dirichlet_parameters or len(dirichlet_parameters[node.index]) != n_children:
                if n_children > 1:
                    alpha_k_l = np.abs(np.random.random(n_children))
                    dirichlet_parameters[node.index] = alpha_k_l

        self.dirichlet_params = dirichlet_parameters

    def sample_abundance_tree(self, tree):

        def recursive_abundance_sampling(node):
            if node.value > 0 and node.hasChildren():

                countActiveChildren = node.countNonZeroChilren()

                if countActiveChildren > 1:
                    n_children = len(node.children)
                    alpha = self.dirichlet_params[node.index]
                    # make sure the dirichlet parameter is of right size
                    assert (len(alpha) == n_children)

                    # Sample the abundance values of the children
                    abundances = dirichlet.rvs(alpha, size=1, random_state=self.seed)[0]
                    # Now we mask the abundances that were not relevant as the node is not activated
                    for i, child in enumerate(node.children):
                        if child.value == 0:
                            abundances[i] = 0

                    # Then we normalize the abundance to the parent's abundance value
                    abundances /= abundances.sum()
                    abundances *= node.value

                    # apply the abundances to the children
                    for i, child in enumerate(node.children):
                        if child.value > 0:
                            child.value = abundances[i]
                        else:
                            child.value = 0

                        recursive_abundance_sampling(child)

                elif countActiveChildren == 1:
                    # transfer the node's abundance to the child if activate
                    for i, child in enumerate(node.children):
                        if child.value > 0:
                            child.value = node.value
                        else:
                            child.value = 0

                        recursive_abundance_sampling(child)

            elif node.hasChildren() and node.value == 0:
                for child in node.children:
                    child.value = 0
                    recursive_abundance_sampling(child)

        recursive_abundance_sampling(tree.root)

        return tree

    def compute_log_likelihood(self, tree):
        ll = 0

        for node in tree.nodes:

            if node.value == 0:
                continue

            countActive = node.countNonZeroChilren()

            if countActive > 1:

                alpha_k_l = self.dirichlet_params[node.index]
                x_k_i_l = node.value
                values = []
                mask = []
                for i, child in enumerate(node.children):
                    if child.value > 0:
                        values.append(child.value / x_k_i_l)
                        mask.append(i)

                if (np.abs(np.sum(values, 0) - 1.0) > 10e-10).any():
                    print("Issue at node", node.index)
                    tree.plot()
                    plt.show()

                ll += np.log(dirichlet.pdf(values, alpha_k_l[mask]))

        return ll

    def compute_dataset_log_likelihood(self, trees):
        ll = 0
        for T_i in trees:
            ll += self.compute_log_likelihood(T_i)

        return ll

    def fit(self, trees, n_iter=30):

        # Storing the likelihood
        ll_list = []

        # Repeat n_iter times the fixed point algorithm
        for iter in range(n_iter):

            # We only fetch the nodes that are affected by Dirichlet parameters
            for node_index in self.dirichlet_params.keys():

                # We fetch the alpha_k_l dirichlet param corresponding to the current node
                alpha_k_l = self.dirichlet_params[node_index]

                # We update the v-th coordinate of alpha_k_l
                for v in range(len(alpha_k_l)):

                    # Initialization of the update
                    alpha_k_v_l = 0
                    count = 0

                    # We roam over the trees to compute the update of the parameter
                    for T_i in trees:

                        # fetching the node sample
                        node_k_i_l = None
                        for n in T_i.nodes:
                            if n.index == node_index:
                                node_k_i_l = n
                                break
                        # filter out the trees that don't contain the node
                        if node_k_i_l is None or node_k_i_l.value == 0:
                            continue

                        # fetching the current value of that sample
                        x_k_i_l = node_k_i_l.value
                        # fetching the v-th children value of that sample
                        Cx_k_i_l_v = node_k_i_l.children[v].value

                        # Computing the mask of the current sample
                        mask = []
                        for i, child in enumerate(node_k_i_l.children):
                            if child.value > 0:
                                mask.append(i)

                        # Only filtering those that do not mask the v-th child of the current node
                        if Cx_k_i_l_v > 0:
                            alpha_k_v_l += np.log(Cx_k_i_l_v / x_k_i_l) + digamma(alpha_k_l[mask].sum())
                            count += 1

                    # Update the parameter
                    if count == 0:
                        count = 1
                    alpha_k_l[v] = invdigamma(alpha_k_v_l / count)
                    self.dirichlet_params[node_index] = alpha_k_l

            ll = self.compute_dataset_log_likelihood(trees)
            print("Fitting DirichletAbundanceTree - Likelihood:", ll)
            ll_list.append(ll)

        return ll_list



def example_generation():
    global_adjacency_matrix = [
        [0, 1, 1, 0, 0, 0, 0, 0],  # node 0
        [0, 0, 0, 1, 1, 0, 0, 0],  # node 1
        [0, 0, 0, 0, 0, 1, 1, 1],  # node 2
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 3
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 7
    ]

    abundance_tree = AbundanceTree(global_adjacency_matrix, abundance_values={
        0: 1,
        1: 0.6,
        2: 0.4,
        3: 0.6,
        4: 0,
        5: 0.2,
        6: 0.2,
        7: 0
    })

    alpha_list = {
        0: [20, 10],
        1: [5, 8],
        2: [3, 8, 20]
    }
    dirichlet_tree = DirichletAbundanceTreePosterior(abundance_tree, dirichlet_parameters=alpha_list)
    tree = dirichlet_tree.sample_abundance_tree(abundance_tree)

    tree.plot()
    plt.show()
