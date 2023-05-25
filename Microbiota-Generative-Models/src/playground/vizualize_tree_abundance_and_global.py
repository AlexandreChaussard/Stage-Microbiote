from src.data import microbiota_abundance_trees
import matplotlib.pyplot as plt

global_tree, abundance_trees = microbiota_abundance_trees(precision_max=3, path='../data')
global_tree.plot()

n_tree_to_plot = 3
for abundance_tree in abundance_trees.values():
    if n_tree_to_plot < 0:
        break

    abundance_tree.plot()
    n_tree_to_plot -= 1

plt.show()
