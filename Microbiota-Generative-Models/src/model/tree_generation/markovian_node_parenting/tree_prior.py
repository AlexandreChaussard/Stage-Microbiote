from src.utils import Tree, AbundanceTree, Node
import random
import matplotlib.pyplot as plt
import numpy as np


class ActivableNode(Node):

    def __init__(self, parent, children, activationProba, value=1, index=0, depth=0):
        super().__init__(parent, children, value, index, depth)
        self.activationProba = activationProba
        self.activated = False

    def bernoulli_activation(self):
        # If the node has an activated parent, we sample it using a Bernoulli distribution
        if self.hasParent() and self.parent.activated == True:
            r = random.random()
            if r <= self.activationProba:
                self.activated = True
            else:
                self.activated = False
                self.value = 0
        else:
            self.activated = self.depth == 0  # only the root is always activated
            self.value = 0
        return self.activated

    def hasActiveChildren(self):
        # If the node has no children, then none is activated
        if not self.hasChildren():
            return False
        # If the node has one active children, then it has an activated one!
        for child in self.children:
            if child.activated:
                return True
        # The node didn't have any activated child
        return False


class BernoulliTreePrior(Tree):
    def __init__(self, global_tree, activation_probabilities, seed=None):
        adjancent_matrix = global_tree.adjacent_matrix
        super().__init__(adjancent_matrix)
        random.seed(seed)
        # Turn the tree nodes into activable nodes
        new_root = ActivableNode(
            self.root.parent,
            self.root.children,
            1,
            self.root.value,
            self.root.index,
            self.root.depth
        )
        for child in self.root.children:
            child.parent = new_root
        self.root = new_root
        self.root.activated = True

        self.nodes = {0: self.root}

        def recursive_node_type_change(node):
            if node.hasChildren():
                new_children = []
                for child in node.children:
                    # Default is set to 0
                    activation_proba = 0
                    # If defined we just fetch the activation proba
                    if child.index in activation_probabilities and child.value > 0:
                        activation_proba = activation_probabilities[child.index]
                    new_children.append(
                        ActivableNode(
                            node,
                            child.children,
                            activation_proba,
                            child.value,
                            child.index,
                            child.depth
                        )
                    )
                    self.nodes[new_children[-1].index] = new_children[-1]
                node.children = new_children
                for child in node.children:
                    recursive_node_type_change(child)

        recursive_node_type_change(self.root)

    def getActivationProbabilities(self):
        activation_probabilities = {}
        for node in self.nodes.values():
            activation_probabilities[node.index] = node.activationProba
        return activation_probabilities

    def sample_tree(self):
        """
        Generate a new tree following the current prior
        """

        def recursive_sample(node):
            # We sample the node
            node.bernoulli_activation()
            # And if it worked out, we do it for the children as well
            if node.hasChildren() and node.activated:
                for child in node.children:
                    recursive_sample(child)

        # We sample starting from the root
        recursive_sample(self.root)

        # Generate a new tree with the said activated nodes
        tree = Tree(self.adjacent_matrix)
        for node in tree.nodes:
            for sampled_node in self.nodes.values():
                if sampled_node.index == node.index:
                    node.value = sampled_node.activated * 1
                    break

        return tree

    def fit(self, trees):
        """
        Learn the activation probabilities from the trees as a dataset
        """

        # Roam through the nodes
        for node in self.nodes.values():

            # The root is always there, no matter what happens
            if node.index == 0:
                node.activationProba = 1
                continue

            # Compute the activation probability pi_k^(l)
            pi_k_l = 0
            normalization = 0

            # Compute the probability over the dataset
            for T_i in trees:
                u_k_i_l = 0
                p_k_i_l = 0
                for node_i in T_i.nodes:
                    if node_i.index == node.index:
                        u_k_i_l = (node_i.value > 0) * 1
                        p_k_i_l = (node_i.parent.value > 0) * 1
                        break

                pi_k_l += p_k_i_l * u_k_i_l
                normalization += p_k_i_l

            if normalization != 0:
                node.activationProba = pi_k_l/normalization
                print("node index:", node.index, " | pi_k_l =", pi_k_l, " | normalization =", normalization)
            else:
                node.activationProba = 0

            assert (0 <= node.activationProba <= 1)

    def get_proba_tree(self):
        """
        Get the probabilistic tree of the given bernoulli tree on the global architecture
        """
        probabilities = {}
        for node in self.nodes.values():
            probabilities[node.index] = node.activationProba
        proba_tree = AbundanceTree(self.adjacent_matrix, probabilities)
        return proba_tree

    def compute_log_likelihood(self, tree):

        ll = 0

        # Skip the root
        for l in range(1, tree.getMaxDepth()+1):

            for node in tree.getNodesAtDepth(l):

                # Fetch the activation probability of the node
                pi = self.nodes[node.index].activationProba
                # Check if the node is activated
                u = (node.value > 0) * 1
                # Check if the parent of the node is activated
                P = (node.parent.value > 0) * 1

                # Compute the log likelihood
                if P == 1:
                    if u == 1:
                        ll += np.log(pi)
                    else:
                        ll += np.log(1 - pi)

        return ll


def generate_bernoulli_tree(global_adjacency_matrix, activation_probabilities, seed=None):
    prior = BernoulliTreePrior(global_adjacency_matrix, activation_probabilities, seed)
    tree = prior.sample_tree()
    return tree


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

    activation_probabilities = {
        0: 1,  # root is always here
        1: 0.9,  # proba for node 1
        2: 0.9,  # proba for node 2
        3: 1,  # proba for node 3
        4: 0.6,  # proba for node 4
        5: 0.2,  # proba for node 5
        6: 0.8,  # proba for node 6
        7: 0.6,  # proba for node 7
    }

    prior = BernoulliTreePrior(global_adjacency_matrix, activation_probabilities)
    tree = prior.sample_tree()
    tree.plot()
    plt.show()
