import numpy as np

from src.data.dataloader import generate_gaussian, generate_conditional_binary_observations, get_train_test
import src.utils.viz as viz
from src.utils.functions import identify_permutation
from src.model import GaussianMixtureClassifier
from src.utils.functions import accuracy
from src.utils.optimizers import GradientDescent
import matplotlib.pyplot as plt

mu_list = np.array([[-0.1, -0.2], [0.5, 0.3]])
sigma_list = np.array([[0.1, 0.15], [0.2, 0.1]])

seed = 8
X, Z = generate_gaussian(
    n_samples=500,
    d=2,
    mu_list=mu_list,
    sigma_list=sigma_list,
    seed=seed
)
X_test, Z_test = generate_gaussian(
    n_samples=100,
    d=2,
    mu_list=mu_list,
    sigma_list=sigma_list,
    seed=seed+10
)

X_train, Z_train, _, _ = get_train_test(X, Z, n_train=300)
y_train = generate_conditional_binary_observations(X_train, Z_train, seed=seed)
y_test, W_e, W_x = generate_conditional_binary_observations(X_test, Z_test, seed=seed, returnParams=True)

n_train_list = [10, 30, 50, 80, 100, 130, 160, 200, 250, 300]

accuracies = []
likelihood = []
distances_to_params = np.zeros((len(n_train_list), 4))

indexes = np.arange(0, len(X_train))
np.random.shuffle(indexes)
for i, n_train in enumerate(n_train_list):
    print(f"* n_train: {n_train}")

    gmm = GaussianMixtureClassifier(
        z_dim=2,
        optimizer=GradientDescent(learning_rate=0.05, n_iter=25),
        seed=0,
    )
    gmm.fit(X_train[indexes[:n_train]], y_train[indexes[:n_train]])
    gmm.train(n_steps=30, printEvery=1)

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

    print(f"Summary: \n   * accuracy: {acc}\n   * likelihood: {ll}\n   * distances: {distances_to_params[i].tolist()}")



fig, axs = plt.subplots(1, 2, figsize=(15, 9))
axs[0].plot(n_train_list, accuracies)
axs[1].plot(n_train_list, likelihood)
axs[0].set_title("Accuracy / n_samples")
axs[1].set_title("Likelihood / n_samples")
fig.suptitle("Metrics / n_samples with 30 GD Step, 30 EM steps")

fig2, axs2 = plt.subplots(1, distances_to_params.shape[1], figsize=(15, 6))
title = [
    "$\Vert \mu^* - \mu \Vert$",
    "$\Vert \Sigma^* - \Sigma \Vert$",
    "$\Vert W_e^* - W_e \Vert$",
    "$\Vert W_x^* - W_x \Vert$",
]
for i in range(distances_to_params.shape[1]):
    axs2[i].plot(n_train_list, distances_to_params[:, i])
    axs2[i].set_title(title[i])
    axs2[i].set_yscale("log")
fig2.suptitle("Convergence of the estimators towards each model parameter")

plt.show()
