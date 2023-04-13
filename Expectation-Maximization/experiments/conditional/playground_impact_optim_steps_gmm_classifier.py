from src.data.dataloader import generate_gaussian, generate_conditional_binary_observations, get_train_test
import src.utils.viz as viz
from src.model import GaussianMixtureClassifier
from src.utils.functions import accuracy
import matplotlib.pyplot as plt
import numpy as np

mu_list = [[-0.1, -0.2], [0.5, 0.3]]
sigma_list = [[0.1, 0.15], [0.2, 0.1]]

X, Z = generate_gaussian(
    n_samples=200,
    d=2,
    mu_list=mu_list,
    sigma_list=sigma_list
)

X_train, Z_train, X_test, Z_test = get_train_test(X, Z, n_train=100)

seed = 8
y_train, y_test = generate_conditional_binary_observations(X_train, Z_train, seed=seed), \
                  generate_conditional_binary_observations(X_test, Z_test, seed=seed)

n_optim_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

accuracies = []
likelihood = []

for i, n_optim_step in enumerate(n_optim_steps):
    gmm = GaussianMixtureClassifier(
        z_dim=2,
        learning_rate=0.1,
        n_iter=n_optim_step,
        seed=1
    )
    gmm.fit(X_train, y_train)

    gmm.train(n_steps=30, printEvery=10)

    # Compute the likelihood to show the monotonic improvement
    ll = gmm.compute_loglikelihood(X_test, y_test)
    likelihood.append(ll)

    # Compute the accuracy of the model
    y_pred = gmm.classify(X_test)
    acc = accuracy(y_test, y_pred)
    accuracies.append(acc)

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
axs[0].plot(n_optim_steps, accuracies)
axs[1].plot(n_optim_steps, likelihood)
axs[0].set_title("Accuracy / GD step")
axs[1].set_title("Likelihood / GD step")
fig.suptitle("Metrics / GD Step with 30 EM steps and 100 training samples")

viz.plot_2d_gaussians_samples_with_pdf(X_test, y_test, mu=mu_list, sigma=sigma_list, subtitle="(truth)")
viz.plot_2d_gaussians_samples_with_pdf(X_test, y_pred, mu=gmm.mu, sigma=gmm.sigma, subtitle="(estimated)")

plt.show()
