from src.data.dataloader import generate_gaussian, generate_conditional_binary_observations, get_train_test
import src.utils.viz as viz
from src.model import GaussianMixtureClassifier
from src.utils.functions import accuracy, identify_permutation
from src.utils.optimizers import GradientDescent
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
y_train = generate_conditional_binary_observations(X_train, Z_train, seed=seed)
y_test, W_e, W_x = generate_conditional_binary_observations(X_test, Z_test, seed=seed, returnParams=True)

gmm = GaussianMixtureClassifier(
    z_dim=2,
    optimizer=GradientDescent(learning_rate=0.05, n_iter=30),
    seed=1
)
gmm.fit(X_train, y_train)

n_EM_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

accuracies = []
likelihood = []
distances_to_params = np.zeros((len(n_EM_steps), 4))

for i, n_EM_step in enumerate(n_EM_steps):
    print("* n_em_step:", n_EM_step)
    if i != 0:
        previous = n_EM_steps[i-1]
    else:
        previous = 0

    gmm.train(n_steps=n_EM_step - previous, printEvery=10)

    # Compute the likelihood to show the monotonic improvement
    ll = gmm.compute_loglikelihood(X_test, y_test)
    likelihood.append(ll)

    # Compute the accuracy of the model
    y_pred = gmm.classify(X_test)
    acc = accuracy(y_test, y_pred)
    accuracies.append(acc)

    # params distance
    permutation = identify_permutation(mu_list, gmm.mu)

    distances_to_params[i][0] = np.linalg.norm(gmm.mu[permutation] - mu_list)
    distances_to_params[i][1] = np.linalg.norm(gmm.sigma[permutation] - sigma_list)
    distances_to_params[i][2] = np.linalg.norm(gmm.W_e - W_e)
    distances_to_params[i][3] = np.linalg.norm(gmm.W_x - W_x)

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
axs[0].plot(n_EM_steps, accuracies)
axs[1].plot(n_EM_steps, likelihood)
axs[0].set_title("Accuracy / EM step")
axs[1].set_title("Likelihood / EM step")
fig.suptitle("Metrics / EM Step with GD 20 steps and 100 training samples")

fig2, axs2 = plt.subplots(1, distances_to_params.shape[1], figsize=(15, 6))
title = [
    "$\Vert \mu^* - \mu \Vert$",
    "$\Vert \Sigma^* - \Sigma \Vert$",
    "$\Vert W_e^* - W_e \Vert$",
    "$\Vert W_x^* - W_x \Vert$",
]
for i in range(distances_to_params.shape[1]):
    axs2[i].plot(n_EM_steps, distances_to_params[:, i])
    axs2[i].set_title(title[i])
    axs2[i].set_yscale("log")
fig2.suptitle("Convergence of the estimators towards each model parameter")

viz.plot_2d_gaussians_samples_with_pdf(X_test, y_test, mu=mu_list, sigma=sigma_list, subtitle="(truth)")
viz.plot_2d_gaussians_samples_with_pdf(X_test, y_pred, mu=gmm.mu, sigma=gmm.sigma, subtitle="(estimated)")

plt.show()
