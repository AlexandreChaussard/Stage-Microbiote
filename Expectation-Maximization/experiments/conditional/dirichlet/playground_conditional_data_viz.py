from src.data.dataloader import generate_dirichlet, generate_conditional_binary_observations
import src.utils.viz as viz
import matplotlib.pyplot as plt

alpha_list = [[5, 3, 20], [15, 4, 2]]

seed = 621
X, Z = generate_dirichlet(
    n_samples=200,
    alpha_list=alpha_list,
    seed=seed
)

y = generate_conditional_binary_observations(X, Z, seed=seed)

if 1 in y[Z == 0] and 1 in y[Z == 1] and 0 in y[Z == 0] and 1 in y[Z == 1]:
    viz.plot_2d_dirichlet_samples_with_pdf(
            X, y,
            alpha_list=alpha_list,
            subtitle=f"Conditional binary observations (seed {seed})",
            alpha=0.6)

plt.show()
