from src.data.dataloader import generate_gaussian, generate_conditional_binary_observations
import src.utils.viz as viz
import numpy as np

mu_list = [[-0.1, -0.2], [0.5, 0.3]]
sigma_list = [[0.1, 0.15], [0.2, 0.1]]

X, Z = generate_gaussian(
    n_samples=200,
    d=2,
    mu_list=mu_list,
    sigma_list=sigma_list
)

y = generate_conditional_binary_observations(X, Z, seed=8)

viz.plot_2d_gaussians_samples_with_pdf(X, y, mu=mu_list, sigma=sigma_list, subtitle="Conditional binary observations")


