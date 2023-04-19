from src.data.dataloader import generate_gaussian, generate_conditional_binary_observations, get_train_test
import src.utils.viz as viz
from src.model import GaussianMixtureClassifier
from src.utils.functions import accuracy
from src.utils.optimizers import GradientDescent, StochasticGradientDescent, CMAES
import matplotlib.pyplot as plt
import numpy as np

mu_list = np.array([[-0.1, -0.2], [0.5, 0.3]])
sigma_list = np.array([[0.1, 0.15], [0.2, 0.1]])

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
    optimizer=GradientDescent(learning_rate=0.05, n_iter=10),
    seed=1,
)
gmm.fit(X_train, y_train)
gmm.train(n_steps=15, printEvery=1)

y_pred = gmm.classify(X_test)

print(f"Model accuracy: {str(accuracy(y_test, y_pred) * 100)[0:4]}%")

viz.plot_2d_gaussians_samples_with_pdf(X_test, y_test, mu=mu_list, sigma=sigma_list, subtitle="(truth)")
viz.plot_2d_gaussians_samples_with_pdf(X_test, y_pred, mu=gmm.mu, sigma=gmm.sigma, subtitle="(estimated)")

print("-=== Parameters objectives ===-")
print("--------------------")
print("W_e - W_e_hat:")
print(W_e - gmm.W_e)
print("--------------------")
print("W_x - W_x_hat:")
print(W_x - gmm.W_x)
print("--------------------")

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
axs[0].plot(gmm.Q_values, label="$Q(\omega, \widehat{\omega})$", marker=".")
axs[1].plot(gmm.likelihood_values, label="likelihood", marker=".")
axs[1].axhline(
    y=gmm.compute_loglikelihood(X_train, y_train, pi=[.5, .5], mu=mu_list, sigma=sigma_list, W_e=W_e, W_x=W_x),
    color='r',
    linestyle='--',
    label="Optimal"
)
axs[0].legend()
axs[1].legend()
fig.suptitle("Q and likelihood values over training")
plt.show()
