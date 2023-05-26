from src.data import microbiota_abundance_trees
from src.model import BernoulliTreePrior
import matplotlib.pyplot as plt

global_tree, abundance_trees = microbiota_abundance_trees(precision_max=3, path='../../data')

prior = BernoulliTreePrior(global_tree.adjacent_matrix, activation_probabilities={})
prior.fit(abundance_trees.values())

proba_tree = prior.get_proba_tree()

proba_tree.plot(title="Probability tree")

n_random_trees = 5
for _ in range(n_random_trees):
    random_tree = prior.sample_tree()
    random_tree.plot(title="Generated tree")

plt.show()
