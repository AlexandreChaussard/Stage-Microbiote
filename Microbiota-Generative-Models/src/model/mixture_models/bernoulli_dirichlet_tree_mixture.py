import numpy as np
from scipy.special import logsumexp
from src.utils.functions import digamma, invdigamma
import matplotlib.pyplot as plt
from src.model import BernoulliTreePrior, DirichletAbundanceTreePosterior
from src.utils import Tree, AbundanceTree

class BernoulliTree_DirichletAbundance_Mixture:
    def __init__(
            self,
            global_tree,
            z_dim,
            clusters_proba=None,
            activation_probabilities_list=None,
            dirichlet_parameters_list=None,
            seed=None
    ):

        self.global_tree = global_tree
        self.z_dim = z_dim
        self.seed = seed

        # Initialize the parameters of the law of Z
        if clusters_proba is None or clusters_proba == []:
            self.gamma = np.array([1 / z_dim] * z_dim)
        else:
            assert(len(clusters_proba) == z_dim)
            assert(np.abs(np.sum(clusters_proba) - 1) < 10e-5)
            self.gamma = np.array(clusters_proba)

        # Initialize the mixture of bernoulli tree priors and dirichlet posteriors
        self.tree_mixture = []
        if activation_probabilities_list is None or activation_probabilities_list == []:
            activation_probabilities_list = [{} for _ in range(self.z_dim)]
            for node in global_tree.nodes:
                for activation_probabilities in activation_probabilities_list:
                    proba = np.random.random()
                    if node.index == 0:
                        proba = 1
                    activation_probabilities[node.index] = proba
        for activation_probabilities in activation_probabilities_list:
            self.tree_mixture.append(BernoulliTreePrior(global_tree, activation_probabilities, seed=seed))

        self.abundance_mixture = []
        self.dirichlet_abundance_nodes_index = {}
        if dirichlet_parameters_list is None or dirichlet_parameters_list == []:
            dirichlet_parameters_list = [{} for _ in range(self.z_dim)]
        for dirichlet_parameters in dirichlet_parameters_list:
            dirichlet_model = DirichletAbundanceTreePosterior(global_tree, dirichlet_parameters, seed)
            self.abundance_mixture.append(dirichlet_model)
            self.dirichlet_abundance_nodes_index = dirichlet_model.dirichlet_params.keys()

        # Compute the number of nodes at each depth and the maximum depth
        self.max_depth = global_tree.getMaxDepth()

    def compute_log_likelihood(self, abundance_trees):
        # Compute the log likelihood as:
        # log p(X, Y) = log E[p(X, Y, Z)] = log E[p(Z)p(X|Z)p(Y|X,Z)] = log sum_c(p(Y|X,Z=c)p(X|Z=c)p(Y|X,Z=c))
        # Finally, since all (X_i, Y_i) are iid, log p(X, Y) = sum_i log p(X_i, Y_i)
        ll = 0

        for T_i in abundance_trees:

            values = np.zeros(self.z_dim)

            for c in range(self.z_dim):
                values[c] = np.log(self.gamma[c]) \
                            + self.tree_mixture[c].compute_log_likelihood(T_i) \
                            + self.abundance_mixture[c].compute_log_likelihood(T_i)

            ll += logsumexp(values) / len(abundance_trees)

        return ll

    def sample(self):
        """
        Generate a new tree following the current prior
        """

        # Draw a random cluster
        c = np.random.choice(np.arange(self.z_dim), p=self.gamma)

        # Generate a sample from that cluster
        tree = self.tree_mixture[c].sample_tree()
        abundance = self.abundance_mixture[c].sample_abundance_tree(tree)

        return abundance, c

    def expectation_step(self, abundance_trees):
        # List of tau_ic, the expectation probabilities
        tau = []
        for data in abundance_trees:

            # log expectation quantity, for computation stability
            log_tau_i = np.zeros(self.z_dim)

            for c in range(self.z_dim):
                log_tau_i[c] = np.log(self.gamma[c]) \
                               + self.tree_mixture[c].compute_log_likelihood(data) \
                               + self.abundance_mixture[c].compute_log_likelihood(data)

            log_tau_i -= logsumexp(log_tau_i)

            tau.append(np.exp(log_tau_i))

        tau = np.array(tau)
        return tau

    def maximization_step(self, abundance_trees, expectations, n_steps_fixedpoint):

        # We roam through the different hidden clusters to update the parameters
        for c in range(self.z_dim):

            # Fetch the expectations corresponding to the cluster
            tau_c = expectations[:, c]

            # --------------------------
            # Update for gamma_c
            # --------------------------
            # Update the parameter gamma describing the law of Z
            self.gamma[c] = tau_c.mean()

            # --------------------------
            # Update for pi_c
            # --------------------------
            # Update the parameter pi describing the law of T | Z
            activation_probabilities = {}
            normalization = {}
            # Initialize the dictionary with the node indexes
            for node in self.global_tree.nodes:
                activation_probabilities[node.index] = 0
                normalization[node.index] = 0

            # Roam through each node and update the pi_c,k^l
            for i, T_i in enumerate(abundance_trees):
                # Fetching the expectation for that tree under cluster c
                tau_ic = tau_c[i]

                for node in T_i.nodes:
                    # Fetching the parent node activation
                    if node.parent is None:
                        P = 0
                    else:
                        P = (node.parent.value > 0) * 1
                    # Fetching the current node activation
                    u = (node.value > 0) * 1

                    # Compute the estimator
                    activation_probabilities[node.index] += P * u * tau_ic
                    normalization[node.index] += P * tau_ic

            # Apply the normalization
            for node in self.global_tree.nodes:
                if normalization[node.index] > 0:
                    activation_probabilities[node.index] /= normalization[node.index]

            # Normalize each estimator, and apply it to the current model
            self.tree_mixture[c] = BernoulliTreePrior(self.global_tree, activation_probabilities, self.seed)

            # --------------------------
            # Update for alpha_c
            # --------------------------
            # Perform the fixed point iteration algorithm to compute alpha_c,k,v^l
            for n_steps in range(n_steps_fixedpoint):

                # We only look into the nodes that are affected by the dirichlet parameterization
                for node_index in self.dirichlet_abundance_nodes_index:

                    # Fetching the model parameters for the said node
                    alpha_c_k_l = self.abundance_mixture[c].dirichlet_params[node_index]

                    # We update the v-th coordinate of alpha_c_k_l
                    for v in range(len(alpha_c_k_l)):

                        # Initialization of the update
                        alpha_c_k_v_l = 0
                        normalization = 0

                        # We roam over the trees to compute the update of the parameter
                        for i, T_i in enumerate(abundance_trees):

                            # Fetch the current expectation
                            tau_ic = tau_c[i]

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

                            # Computing the mask of the current sample T_i
                            mask = []
                            for i, child in enumerate(node_k_i_l.children):
                                if child.value > 0:
                                    mask.append(i)

                            # Only filtering those that do not mask the v-th child of the current node
                            if Cx_k_i_l_v > 0:
                                alpha_c_k_v_l += (np.log(Cx_k_i_l_v / x_k_i_l)
                                                  + digamma(alpha_c_k_l[mask].sum())) \
                                                 * tau_ic
                                normalization += tau_ic

                        # Update the parameter
                        if normalization == 0:
                            normalization = 1
                        alpha_c_k_l[v] = invdigamma(alpha_c_k_v_l / normalization)
                        self.abundance_mixture[c].dirichlet_params[node_index] = alpha_c_k_l

    def fit(self, abundance_trees, n_steps_EM, n_steps_fixedpoint):
        """
        Learn the parameters of the mixture from the trees as a dataset
        using the EM algorithm.
        """

        # We compute the log likelihood during the training
        ll = np.zeros(n_steps_EM)

        for em_step in range(n_steps_EM):
            # First, we perform an expectation step
            tau = self.expectation_step(abundance_trees)

            # Then we optimize the parameters
            self.maximization_step(abundance_trees, tau, n_steps_fixedpoint)

            ll[em_step] = self.compute_log_likelihood(abundance_trees)
            print(f"* [{em_step}/{n_steps_EM}]: Bernoulli Dirichlet Mixture - Log likelihood:", ll[em_step])

        return ll


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

    global_tree = Tree(global_adjacency_matrix)

    activation_probabilities_1 = {
        0: 1,  # root is always here
        1: 0.9,  # proba for node 1
        2: 0.9,  # proba for node 2
        3: 1,  # proba for node 3
        4: 0.6,  # proba for node 4
        5: 0.2,  # proba for node 5
        6: 0.8,  # proba for node 6
        7: 0.6,  # proba for node 7
    }

    activation_probabilities_2 = {
        0: 1,  # root is always here
        1: 0.99,  # proba for node 1
        2: 0.9,  # proba for node 2
        3: 0.4,  # proba for node 3
        4: 0.7,  # proba for node 4
        5: 0.1,  # proba for node 5
        6: 0.2,  # proba for node 6
        7: 0.9,  # proba for node 7
    }

    activation_probabilities_3 = {
        0: 1,  # root is always here
        1: 0.9,  # proba for node 1
        2: 0.99,  # proba for node 2
        3: 0.5,  # proba for node 3
        4: 0.2,  # proba for node 4
        5: 0.8,  # proba for node 5
        6: 0.8,  # proba for node 6
        7: 0.3,  # proba for node 7
    }

    activation_probabilities_list = [activation_probabilities_1, activation_probabilities_2, activation_probabilities_3]

    dirichlet_parameters_list = [{}, {}, {}]

    prior = BernoulliTree_DirichletAbundance_Mixture(
        global_tree,
        z_dim=3,
        clusters_proba=[0.2, 0.5, 0.3],
        activation_probabilities_list=activation_probabilities_list,
        dirichlet_parameters_list=dirichlet_parameters_list
    )
    tree, cluster = prior.sample()
    tree.plot(title=f"Generated tree in cluster {cluster}")
    plt.show()