from src.data import microbiota_abundance_trees
from src.model import DirichletAbundanceTreePosterior, BernoulliTreePrior
import matplotlib.pyplot as plt

global_tree, abundance_trees = microbiota_abundance_trees(precision_max=3, path='../../data')
trees = abundance_trees.values()

prior = BernoulliTreePrior(global_tree, activation_probabilities={})
posterior = DirichletAbundanceTreePosterior(global_tree, dirichlet_parameters={})

prior.fit(trees)
ll = posterior.fit(trees, n_iter=40)

fig, axs = plt.subplots()
fig.suptitle("Log-likelihood / iteration")
axs.plot(ll)

n_random_trees = 5
for _ in range(n_random_trees):
    tree = prior.sample_tree()
    random_tree = posterior.sample_abundance_tree(tree)
    random_tree.plot(title="Generated abundance tree")

plt.show()
