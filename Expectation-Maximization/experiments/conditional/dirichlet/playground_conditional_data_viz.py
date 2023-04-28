from src.data.dataloader import generate_dirichlet, generate_conditional_binary_observations
import src.utils.viz as viz
import matplotlib.pyplot as plt

alpha_list = [[5, 3, 20], [15, 4, 2]]

seed = 6
X, Z = generate_dirichlet(
    n_samples=100,
    alpha_list=alpha_list,
    seed=seed
)


y = generate_conditional_binary_observations(X, Z, seed=seed)

viz.plot_2d_dirichlet_samples_with_pdf(
    X, y,
    alpha_list=alpha_list,
    subtitle=f"Conditional binary observations (seed {seed})",
    alpha=0.8)

plt.show()
