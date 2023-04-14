from src.model import GaussianMixture
import src.data.dataloader as dataloader
import src.utils.viz as viz
import src.utils.functions as funtions
import numpy as np
import matplotlib.pyplot as plt

mu_list = [
    [-0.1, -0.2],
    [0.2, 0.35]
]
sigma_list = [
    [0.1, 0.15],
    [0.2, 0.1]
]


def convergence_graph_2d_case(
        n_samples,
        n_steps_list,
        mu_list=np.array([[-0.1, -0.2], [0.2, 0.35]]),
        sigma_list=np.array([[0.1, 0.15], [0.2, 0.1]]),
        seed=1
):
    X, y = dataloader.generate_gaussian(
        n_samples=n_samples,
        d=2,
        mu_list=mu_list,
        sigma_list=sigma_list
    )

    distances_mu = np.zeros(len(n_steps_list))
    distances_sigma = np.zeros(len(n_steps_list))

    for i, n_steps in enumerate(n_steps_list):
        gmm = GaussianMixture(
            z_dim=len(mu_list),
            seed=seed
        )

        gmm.fit(X)
        gmm.train(n_steps=n_steps, printEvery=40)

        permutation = funtions.identify_permutation(mu_list, gmm.mu)
        distances_mu[i] = np.linalg.norm(mu_list - gmm.mu[permutation])
        distances_sigma[i] = np.linalg.norm(sigma_list - gmm.sigma[permutation])

        print("==================")
        print("Objective values are:")
        print("mu:")
        print(mu_list)
        print("sigma:")
        print(sigma_list)
        print("------------------")
        print("Estimated values are:")
        print("mu hat:")
        print(gmm.mu[permutation])
        print("sigma hat:")
        print(gmm.sigma[permutation])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    axs[0].plot(n_steps_list, distances_mu)
    axs[1].plot(n_steps_list, distances_sigma)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].set_title("$\Vert \mu - \mu^* \Vert_2$")
    axs[1].set_title("$\Vert \sigma - \sigma^* \Vert_2$")
    axs[0].set_xticks(n_steps_list)
    axs[1].set_xticks(n_steps_list)
    fig.suptitle(f"Convergence of GMM with {n_samples} samples")

    viz.plot_2d_gaussians_samples_with_pdf(X, y, mu=mu_list, sigma=sigma_list, subtitle="(truth)")
    viz.plot_2d_gaussians_samples_with_pdf(X, y, mu=gmm.mu, sigma=gmm.sigma, subtitle="(estimated)")
    plt.show()


def convergence_graph_1d_case(
        n_samples,
        n_steps_list,
        mu_list=np.array([[-0.1], [0.4]]),
        sigma_list=np.array([[0.1], [0.2]]),
        seed=1
):
    X, y = dataloader.generate_gaussian(
        n_samples=n_samples,
        d=1,
        mu_list=mu_list,
        sigma_list=sigma_list
    )

    distances_mu = np.zeros(len(n_steps_list))
    distances_sigma = np.zeros(len(n_steps_list))

    for i, n_steps in enumerate(n_steps_list):
        gmm = GaussianMixture(
            z_dim=len(mu_list),
            seed=seed
        )

        gmm.fit(X)
        gmm.train(n_steps=n_steps, printEvery=40)

        permutation = funtions.identify_permutation(mu_list, gmm.mu)
        distances_mu[i] = np.linalg.norm(mu_list - gmm.mu[permutation])
        distances_sigma[i] = np.linalg.norm(sigma_list - gmm.sigma[permutation])

        print("==================")
        print("Objective values are:")
        print("mu:")
        print(mu_list)
        print("sigma:")
        print(sigma_list)
        print("------------------")
        print("Estimated values are:")
        print("mu hat:")
        print(gmm.mu[permutation])
        print("sigma hat:")
        print(gmm.sigma[permutation])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    axs[0].plot(n_steps_list, distances_mu)
    axs[1].plot(n_steps_list, distances_sigma)
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    axs[0].set_title("$\Vert \mu - \mu^* \Vert_2$")
    axs[1].set_title("$\Vert \sigma - \sigma^* \Vert_2$")
    axs[0].set_xticks(n_steps_list)
    axs[1].set_xticks(n_steps_list)
    fig.suptitle(f"Convergence of GMM with {n_samples} samples")

    viz.plot_1d_gaussian_samples_with_pdf(X, y, mu_list=mu_list, sigma_list=sigma_list, subtitle="(truth)", n_bins=10)
    viz.plot_1d_gaussian_samples_with_pdf(X, y, mu_list=gmm.mu, sigma_list=gmm.sigma, subtitle="(estimated)", n_bins=10)
    plt.show()


def boxplot_converged_results_1d_case(
        precision=10e-2,
        n_points=10,
        n_samples=100,
        mu_list=np.array([[-0.1], [0.4]]),
        sigma_list=np.array([[0.1], [0.2]])
):
    X, y = dataloader.generate_gaussian(
        n_samples=n_samples,
        d=1,
        mu_list=mu_list,
        sigma_list=sigma_list
    )

    mu_values = np.zeros((n_points, 2))
    sigma_values = np.zeros((n_points, 2))

    k = 0
    while k < (n_points):
        gmm = GaussianMixture(
            z_dim=len(mu_list),
            seed=None
        )

        gmm.fit(X)
        gmm.train(n_steps=1, printEvery=40)

        permutation = funtions.identify_permutation(mu_list, gmm.mu)
        distance_mu = np.linalg.norm(mu_list - gmm.mu[permutation])

        mu_k = gmm.mu[permutation]

        limit_count = 100
        try:
            while distance_mu / np.linalg.norm(mu_k) > precision and limit_count > 0:
                gmm.train(n_steps=1, printEvery=10)
                distance_mu = np.linalg.norm(mu_list - gmm.mu[permutation])
                limit_count -= 1
                mu_k = gmm.mu[permutation]
        except:
            continue

        if limit_count < 0 or np.isnan(gmm.mu).any():
            print("ERROR: EM failed to converge")
            continue

        mu_values[k] = gmm.mu[permutation].squeeze()
        sigma_values[k] = gmm.sigma[permutation].squeeze()
        k += 1

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(
        f"Converged EM 1d - 2 hidden gaussians - n_runs = {n_points} \nprecision = {precision}")
    print(mu_values)
    axs[0][0].boxplot(mu_values[:, 0])
    axs[0][1].boxplot(mu_values[:, 1])
    axs[1][0].boxplot(sigma_values[:, 0])
    axs[1][1].boxplot(sigma_values[:, 1])
    axs[0][0].set_title(f"$\mu_1$ = {mu_list[0][0]}")
    axs[0][1].set_title(f"$\mu_2$ = {mu_list[1][0]}")
    axs[1][0].set_title(f"$\sigma_1$ = {sigma_list[0][0]}")
    axs[1][1].set_title(f"$\sigma_2$ = {sigma_list[1][0]}")

    plt.show()


boxplot_converged_results_1d_case(
    precision=10e-4,
    n_points=20,
    n_samples=100
)
# convergence_graph_2d_case(
#    n_samples=100,
#    n_steps_list=np.arange(5, 125, 10),
#    seed=0
# )
