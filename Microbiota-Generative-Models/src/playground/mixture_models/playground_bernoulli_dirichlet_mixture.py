from src.data import microbiota_abundance_trees
from src.model import BernoulliTree_DirichletAbundance_Mixture
import matplotlib.pyplot as plt

global_tree, abundance_trees = microbiota_abundance_trees(precision_max=3, path='../../data')
trees = abundance_trees.values()

mixture = BernoulliTree_DirichletAbundance_Mixture(
    global_tree,
    z_dim=4
)

ll = mixture.fit(trees, n_steps_EM=10, n_steps_fixedpoint=20)

fig, axs = plt.subplots()
fig.suptitle("Log-likelihood / iteration")
axs.plot(ll)

n_random_trees = 20
for _ in range(n_random_trees):
    sample, cluster = mixture.sample()
    sample.plot(title=f"Generated abundance tree (cluster {cluster})")

plt.show()
