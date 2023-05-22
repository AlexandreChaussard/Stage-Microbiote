from src.utils import Tree, Node
import random
import matplotlib.pyplot as plt


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


class BernoulliTree(Tree):
    def __init__(self, adjancent_matrix, activation_probabilities):
        super().__init__(adjancent_matrix)
        nodes = []
        for node in self.nodes:
            nodes.append(
                ActivableNode(
                    node.parent,
                    node.children,
                    activation_probabilities[node.index],
                    node.value,
                    node.index,
                    node.depth
                )
            )
        self.nodes = nodes
        new_root = nodes[0]
        for child in self.root.children:
            child.parent = new_root
        self.root = new_root
        self.root.activated = True

        def recursive_node_type_change(node):
            if node.hasChildren():
                new_children = []
                for child in node.children:
                    new_children.append(
                        ActivableNode(
                            node,
                            child.children,
                            activation_probabilities[child.index],
                            child.value,
                            child.index,
                            child.depth
                        )
                    )
                node.children = new_children
                for child in node.children:
                    recursive_node_type_change(child)

        recursive_node_type_change(self.root)

    def sample_tree_nodes(self):

        def recursive_sample(node):
            # We sample the node
            activated = node.bernoulli_activation()
            print(activated, node.value)
            # And if it worked out, we do it for the children as well
            if node.hasChildren() and node.activated:
                for child in node.children:
                    recursive_sample(child)

        # We sample starting from the root
        recursive_sample(self.root)


def generate_bernoulli_tree(global_adjacency_matrix, activation_probabilities):
    tree = BernoulliTree(global_adjacency_matrix, activation_probabilities)
    tree.sample_tree_nodes()
    return tree


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
    plt.show()


example_generation()
