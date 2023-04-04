from src.dataloader import generate_gaussian
from src.EM import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

mu_list = [0.3, -0.7]
sigma_list = [0.1, 0.4]

mu_0 = [0.6, 0.3]
sigma_0 = [1.2, 0.8]
pi_0 = [0.5, 0.5]

X, y = generate_gaussian(
    n_samples=100,
    d=1,
    mu_list=mu_list,
    sigma_list=sigma_list
)

# Now we look into the convergence of the algorithm
optim_rate_mu = []
optim_rate_sigma = []

n_steps = [10, 30, 80, 120, 200]
for n_step in n_steps:
    gmm = GaussianMixture(
        z_dim=2,
        mu_list=mu_0,
        sigma_list=sigma_0,
        distrib_list=pi_0
    )
    gmm.fit(X)
    gmm.train(n_steps=n_step, printEvery=10)

    optim_rate_mu.append(np.linalg.norm(np.array(mu_list) - np.array(gmm.mu)))
    optim_rate_sigma.append(np.linalg.norm(np.array(sigma_list) - np.array(gmm.sigma)))

fig, axs = plt.subplots(1, 2)
axs[0].plot(n_steps, optim_rate_mu, label="||mu* - mu||")
axs[1].plot(n_steps, optim_rate_sigma, label="||sigma* - sigma||")
axs[1].set_yscale("log")
axs[1].legend()
axs[0].set_yscale("log")
axs[0].legend()
plt.show()