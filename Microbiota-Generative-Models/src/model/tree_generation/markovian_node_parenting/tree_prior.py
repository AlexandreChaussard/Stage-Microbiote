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

        # We are going to update each pi_k^l for each node of the global tree structure that we have
        # Hence we start by roaming over the nodes of the global structure
        for u_k_l in self.nodes.values():

            # Activation probability to be computed
            pi_k_l = 0
            # Count of trees that have the node u_k_l
            count = 0

            # For each node of the global structure, we go through each tree T_i
            for T_i in trees:
                # We roam through each node u_k,i^l to evaluate the new pi_k^l activation probability
                # If the node u_k_l in the tree T_i is not activated, then we do not count it
                opt_node = None
                for node in T_i.nodes:
                    if node.index == u_k_l.index:
                        opt_node = node
                        break

                if opt_node is not None:
                    # We fetch the node if there was a match
                    u_k_i_l = opt_node

                    # Check the parents presence for computation, which should be true
                    parent_presence = True
                    if u_k_i_l.parent is not None:
                        parent_presence = u_k_i_l.parent.value > 0

                    # We add that up to pi_k_l before normalizing by the amount of trees that possess the node u_k_l
                    pi_k_l += (u_k_i_l.value > 0) * parent_presence * 1
                    count += parent_presence * 1

            # Normalize by the amount of trees that had the node in the end
            if count == 0:
                pi_k_l = 0
            else:
                pi_k_l /= count
            assert (1 >= pi_k_l >= 0)
            # Update the activation of that node with the new one
            u_k_l.activationProba = pi_k_l

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
