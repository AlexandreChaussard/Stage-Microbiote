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

    def countNonZeroChilren(self):
        count = 0
        for child in self.children:
            count += (child.value > 0) * 1
        return count


class Tree:

    def __init__(self, adjacent_matrix):
        self.adjacent_matrix = np.array(adjacent_matrix)
        self.root = Node(None, None)
        self.depth = 0

        self.nodes = np.array([Node(None, None, index=i) for i in range(len(self.adjacent_matrix))])
        self.nodes[0] = self.root

        current_index = 0
        while current_index < len(self.adjacent_matrix):
            # Fetch the current node
            current_node = self.nodes[current_index]
            # Look for the nodes that are connected to the current node, which are necessarily his children
            children_nodes_index = np.where(self.adjacent_matrix[current_index] != 0)[0]
            # For all child in the found children, associate their parent to the current node and append the child
            for child in self.nodes[children_nodes_index]:
                child.parent = current_node
                child.depth = current_node.depth + 1
                self.depth = max(child.depth, self.depth)
                current_node.addChild(child)
            current_index += 1

    def getMaxDepth(self):
        max_depth = 0
        for node in self.nodes:
            if node.depth > max_depth:
                max_depth = node.depth
        return max_depth

    def getLeaves(self):
        return self.getNodesAtDepth(self.depth)

    def getNodesAtDepth(self, depth):
        nodes = []
        for node in self.nodes:
            if node.depth == depth:
                nodes.append(node)
        return nodes

    def plot(self, space=10e10, title=None, fig=None, axs=None, cmap=plt.cm.get_cmap('Blues'), legend=True,
             threshold_abundance=10e-2):
        if fig is None or axs is None:
            fig, axs = plt.subplots(figsize=(15, 7))
        axs.set_yticks([])
        axs.set_xticks([])
        axs.axis('off')
        if title is None:
            fig.suptitle("Tree representation")
        else:
            fig.suptitle(title)

        if legend:
            # Create a ScalarMappable object with the colormap 'Blues'
            sm = plt.cm.ScalarMappable(cmap=cmap)

            # Set the limits of the colorbar to match your plot data
            sm.set_clim(0, 1)

            # Add a colorbar to the plot
            cbar = fig.colorbar(sm)
            cbar.set_label('Abundance')

        n_nodes_per_layer = []
        depth = 1
        while len(self.getNodesAtDepth(depth)) > 0:
            n_nodes_per_layer.append(self.getNodesAtDepth(depth))
            depth += 1

        graph_grid = {}
        index = 1
        for depth, nodes in enumerate(n_nodes_per_layer):
            n_nodes = len(nodes)
            j = 0
            pos = np.linspace(
                1 * n_nodes + space,
                -1 * n_nodes - space,
                n_nodes
            )
            while j < n_nodes:
                graph_grid[nodes[j].index] = [pos[j], -depth - 1]
                j += 1
                index += 1

        # Then recursively plot the nodes
        def recursive_plot_nodes(node):
            if node.hasChildren():

                for i, child in enumerate(node.children):
                    if child.value == 0:
                        continue
                    child.graph_position = graph_grid[child.index]
                    axs.plot([child.graph_position[0]],
                             [child.graph_position[1]], color=cmap(child.value - 10e-10), marker="o", markersize=12)
                    if child.value > threshold_abundance:
                        axs.text(
                            child.graph_position[0],
                            child.graph_position[1],
                            f'{child.index}',
                            ha='center',
                            va='center',
                            ma='center',
                            color='white',
                            fontsize='small'
                        )

                    recursive_plot_nodes(child)

        def recursive_plot_lines(node):
            if node.hasChildren():

                for i, child in enumerate(node.children):
                    if child.value == 0:
                        continue
                    child.graph_position = graph_grid[child.index]
                    axs.plot([child.graph_position[0], node.graph_position[0]],
                             [child.graph_position[1], node.graph_position[1]],
                             color=cmap(child.value - 10e-10), linestyle="-", marker="")

                    recursive_plot_lines(child)

        recursive_plot_lines(self.root)
        recursive_plot_nodes(self.root)

        # Adding the root to the graph
        axs.plot([self.root.graph_position[0]], [self.root.graph_position[1]], color=cmap(1 - 10e-10), marker="o",
                 markersize=12)
        axs.text(
            self.root.graph_position[0],
            self.root.graph_position[1],
            f'{self.root.index}',
            va='center', ha='center',
            color='white')


class AbundanceTree(Tree):

    def __init__(self, adjancent_matrix, abundance_values):
        super().__init__(adjancent_matrix)
        self.abundance_values = abundance_values
        for node in self.nodes:
            value = (node.index == 0) * 1
            if node.index in abundance_values:
                value = abundance_values[node.index]
            node.value = value


def tree_example_usage():
    tree = Tree(adjacent_matrix=[
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
        abundance_values={
            0: 1,  # root value is always 1
            1: 0.6,  # node 1 is 0.6
            2: 0.4,  # node 2 is 0.4
            3: 0.6,  # node 3 must be 0.6 since it's the only child of 1
            4: 0.2,  # node 4 is a child of 2, and is valued to 0.2
            5: 0.1,  # node 5 is a child of 2, and is valued to 0.1
            6: 0.1,  # node 6 is a child of 2, and is valued to 0.1
        }
    )

    tree.plot()
    plt.show()
