from src.model import BernoulliTree, generate_bernoulli_tree
from scipy.stats import dirichlet
import matplotlib.pyplot as plt


class DirichletAbundanceTree:

    def __init__(self, tree, dirichlet_parameters, seed=None):
        self.tree = tree
        self.dirichlet_params = dirichlet_parameters
        self.seed = seed

    def sample_abundance_tree(self):

        self.tree.root.value = 1

        def recursive_abundance_sampling(node):
            if node.activated and node.hasChildren() and node.hasActiveChildren():
                n_children = len(node.children)
                alpha = self.dirichlet_params[node.index]
                # make sure the dirichlet parameter is of right size
                assert (len(alpha) == n_children)

                # Sample the abundance values of the children
                abundances = dirichlet.rvs(alpha, size=1, random_state=self.seed)[0]
                # Now we mask the abundances that were not relevant as the node is not activated
                for i, child in enumerate(node.children):
                    if not child.activated:
                        abundances[i] = 0

                # Then we normalize the abundance to the parent's abundance value
                abundances /= abundances.sum()
                abundances *= node.value

                # apply the abundances to the children
                for i, child in enumerate(node.children):
                    if child.activated:
                        child.value = abundances[i]
                    else:
                        child.value = 0

                    recursive_abundance_sampling(child)

        recursive_abundance_sampling(self.tree.root)

        return self.tree


def example_generation():
    global_adjacency_matrix = [
        [0, 1, 1, 0, 0, 0, 0, 0],  # node 0
        [0, 0, 0, 1, 0, 0, 0, 1],  # node 1
        [0, 0, 0, 0, 1, 1, 1, 0],  # node 2
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 3
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 4
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # node 7
    ]

    activation_probabilities = [
        1,  # root is always here
        0.8,  # proba for node 1
        0.9,  # proba for node 2
        0.7,  # proba for node 3
        0.6,  # proba for node 4
        0.2,  # proba for node 5
        0.8,  # proba for node 6
        0.6,  # proba for node 7
    ]

    tree = generate_bernoulli_tree(global_adjacency_matrix, activation_probabilities)
    tree.plot()

    alpha_list = [
        [20, 10],
        [5, 8],
        [3, 8, 20],
        [],
        [],
        [],
        [],
        []
    ]

    dirichlet_tree = DirichletAbundanceTree(tree, dirichlet_parameters=alpha_list)
    dirichlet_tree.sample_abundance_tree()

    dirichlet_tree.tree.plot()
    plt.show()


example_generation()
