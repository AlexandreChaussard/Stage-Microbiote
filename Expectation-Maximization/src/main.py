from src.data.dataloader import generate_gaussian
from src.model.EM import GaussianMixture
import src.utils.viz as viz


def __main__():
    # Let's give a shot to the algorithm
    mu_list = [0.3, -0.7]
    sigma_list = [0.1, 0.4]

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
    gmm.train(n_steps=100, printEvery=10)

    viz.plot_1d_gaussian_samples_with_pdf(X, y, mu_list=mu_list, sigma_list=sigma_list, n_bins=20, subtitle="(truth)")
    viz.plot_1d_gaussian_samples_with_pdf(X, y, mu_list=gmm.mu, sigma_list=gmm.sigma, n_bins=20, subtitle="(estimated)")


__main__()
