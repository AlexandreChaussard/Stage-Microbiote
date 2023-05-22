import numpy as np
import matplotlib.pyplot as plt


class Node:

    def __init__(self, parent, children, value=1, index=0, depth=0):
        self.parent = parent

        self.children = children
        if self.children is None:
            self.children = []

        self.index = index
        self.depth = depth
        self.graph_position = [0, depth]
        self.value = value

    def hasParent(self):
        return self.parent is not None

    def addChild(self, child):
        self.children.append(child)

    def hasChildren(self):
        return self.countChildren() > 0

    def countChildren(self):
        return len(self.children)


class Tree:

    def __init__(self, adjancent_matrix):
        adjancent_matrix = np.array(adjancent_matrix)
        self.root = Node(None, None)
        self.depth = 0

        self.nodes = np.array([Node(None, None, index=i) for i in range(len(adjancent_matrix))])
        self.nodes[0] = self.root

        current_index = 0
        while current_index < len(adjancent_matrix):
            # Fetch the current node
            current_node = self.nodes[current_index]
            # Look for the nodes that are connected to the current node, which are necessarily his children
            children_nodes_index = np.where(adjancent_matrix[current_index] != 0)[0]
            # For all child in the found children, associate their parent to the current node and append the child
            for child in self.nodes[children_nodes_index]:
                child.parent = current_node
                child.depth = current_node.depth + 1
                self.depth = max(child.depth, self.depth)
                current_node.addChild(child)
            current_index += 1

    def getLeaves(self):
        return self.getNodesAtDepth(self.depth)

    def getNodesAtDepth(self, depth):
        nodes = []
        for node in self.nodes:
            if node.depth == depth:
                nodes.append(node)
        return nodes

    def plot(self, axs=None):
        if axs is None:
            fig, axs = plt.subplots()
            axs.set_yticks([])
            axs.set_xticks([])
            axs.axis('off')
            fig.suptitle("Tree representation")

        # Adding the root to the graph first
        axs.plot([self.root.graph_position[0]], [self.root.graph_position[1]], color="blue", marker="o", markersize=8)
        axs.text(self.root.graph_position[0] + 0.01, self.root.graph_position[1] + .05, f'{self.root.index}')

        # Then recursively plot the nodes
        def recursive_plot(node):
            if node.hasChildren():

                width = len(node.children)
                if width > 1:
                    pos = np.linspace(-0.3 + node.depth / 10, 0.3 - node.depth / 10, width)
                    pos = node.graph_position[0] - pos
                else:
                    pos = [node.graph_position[0]]
                for i, child in enumerate(node.children):
                    child.graph_position = [pos[i], -child.depth]
                    axs.plot([child.graph_position[0]],
                             [child.graph_position[1]], color="blue", marker="o", markersize=8, alpha=child.value)
                    axs.plot([child.graph_position[0], node.graph_position[0]],
                             [child.graph_position[1], node.graph_position[1]],
                             color="blue", linestyle="-", marker="", alpha=child.value)
                    if child.value == 0:
                        continue
                    axs.text(child.graph_position[0] + 0.01, child.graph_position[1] + .05, f'{child.index}')

                    recursive_plot(child)

        recursive_plot(self.root)


class AbundanceTree(Tree):

    def __init__(self, adjancent_matrix, node_values):
        super().__init__(adjancent_matrix)
        for i in range(len(self.nodes)):
            self.nodes[i].value = node_values[i]


def tree_example_usage():
    tree = Tree(adjancent_matrix=[
        [0, 1, 1, 0, 0, 0, 0],  # node 0
        [0, 0, 0, 1, 0, 0, 1],  # node 1
        [0, 0, 0, 0, 1, 1, 0],  # node 2
        [0, 0, 0, 0, 0, 0, 0],  # node 3
        [0, 0, 0, 0, 0, 0, 0],  # node 4
        [0, 0, 0, 0, 0, 0, 0],  # node 5
        [0, 0, 0, 0, 0, 0, 0],  # node 6
    ])

    tree.plot()
    plt.show()


def abundance_example_usage():
    tree = AbundanceTree(
        adjancent_matrix=[
            [0, 1, 1, 0, 0, 0, 0],  # node 0
            [0, 0, 0, 1, 0, 0, 0],  # node 1
            [0, 0, 0, 0, 1, 1, 1],  # node 2
            [0, 0, 0, 0, 0, 0, 0],  # node 3
            [0, 0, 0, 0, 0, 0, 0],  # node 4
            [0, 0, 0, 0, 0, 0, 0],  # node 5
            [0, 0, 0, 0, 0, 0, 0],  # node 6
        ],
        node_values=[
            1,      # root value is always 1
            0.6,    # node 1 is 0.6
            0.4,    # node 2 is 0.4
            0.6,    # node 3 must be 0.6 since it's the only child of 1
            0.2,    # node 4 is a child of 2, and is valued to 0.2
            0.1,    # node 5 is a child of 2, and is valued to 0.1
            0.1,    # node 6 is a child of 2, and is valued to 0.1
        ]
    )

    tree.plot()
    plt.show()