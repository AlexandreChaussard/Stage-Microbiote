from src.data.dataloader import generate_gaussian
from src.model import GaussianMixture
import src.utils.viz as viz


def case_1d():
    mu_list = [[0.2], [0.12]]
    sigma_list = [[0.1], [0.2]]

    X, y = generate_gaussian(
        n_samples=100,
        d=1,
        mu_list=mu_list,
        sigma_list=sigma_list
    )

    gmm = GaussianMixture(
        z_dim=2,
        seed=0
    )

    gmm.fit(X)
    gmm.train(n_steps=30, printEvery=10)

    viz.plot_1d_gaussian_samples_with_pdf(X, y, mu_list=mu_list, sigma_list=sigma_list, n_bins=20, subtitle="(truth)")
    viz.plot_1d_gaussian_samples_with_pdf(X, y, mu_list=gmm.mu, sigma_list=gmm.sigma, n_bins=20, subtitle="(estimated)")


def case_2d():
    mu_list = [[-0.1, -0.2], [0.5, 0.3]]
    sigma_list = [[0.1, 0.15], [0.2, 0.1]]

    X, y = generate_gaussian(
        n_samples=100,
        d=2,
        mu_list=mu_list,
        sigma_list=sigma_list
    )

    gmm = GaussianMixture(
        z_dim=2,
        seed=1
    )

    gmm.fit(X)
    gmm.train(n_steps=30, printEvery=10)

    viz.plot_2d_gaussians_samples_with_pdf(X, y, mu=mu_list, sigma=sigma_list, subtitle="(truth)")
    viz.plot_2d_gaussians_samples_with_pdf(X, y, mu=gmm.mu, sigma=gmm.sigma, subtitle="(estimated)")


case_1d()
