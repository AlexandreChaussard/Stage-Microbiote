from src.data.dataloader import generate_dirichlet, generate_conditional_binary_observations, get_train_test
import src.utils.viz as viz
from src.model import DirichletMixtureClassifier
from src.utils.functions import accuracy, identify_permutation
from src.utils.optimizers import GradientDescent
import matplotlib.pyplot as plt
import numpy as np

alpha_list = [[5, 3, 20], [15, 4, 2]]

seed = 6
X, Z = generate_dirichlet(
    n_samples=100,
    alpha_list=alpha_list,
    seed=seed
)

X_train, Z_train, X_test, Z_test = get_train_test(X, Z, n_train=100)

y_train = generate_conditional_binary_observations(X_train, Z_train, seed=seed)
y_test, W_e, W_x = generate_conditional_binary_observations(X_test, Z_test, seed=seed, returnParams=True)

model = DirichletMixtureClassifier(
    z_dim=2,
    n_iter_alpha=10,
    optimizer=GradientDescent(learning_rate=0.1, n_iter=10),
    seed=1
)
model.fit(X_train, y_train)

n_EM_steps = [1, 10, 20, 30, 40, 50]

accuracies = []
likelihood = []
distances_to_params = np.zeros((len(n_EM_steps), 3))

for i, n_EM_step in enumerate(n_EM_steps):
    print("* n_em_step:", n_EM_step)
    if i != 0:
        previous = n_EM_steps[i-1]
    else:
        previous = 0

    model.train(n_steps=n_EM_step - previous, printEvery=10)

    # Compute the likelihood to show the monotonic improvement
    ll = model.compute_loglikelihood(X_test, y_test)
    likelihood.append(ll)

    # Compute the accuracy of the model
    y_pred = model.classify(X_test)
    acc = accuracy(y_test, y_pred)
    accuracies.append(acc)

    # params distance
    permutation = identify_permutation(alpha_list, model.alpha)

    distances_to_params[i][0] = np.linalg.norm(model.alpha - alpha_list)
    distances_to_params[i][1] = np.linalg.norm(np.diag(model.W_e)[permutation] - np.diag(W_e))
    distances_to_params[i][2] = np.linalg.norm(model.W_x[permutation] - W_x)

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
axs[0].plot(n_EM_steps, accuracies)
axs[1].plot(n_EM_steps, likelihood)
axs[0].set_title("Accuracy / EM step")
axs[1].set_title("Likelihood / EM step")
fig.suptitle("Metrics / EM Step with GD 20 steps and 100 training samples")

fig2, axs2 = plt.subplots(1, distances_to_params.shape[1], figsize=(15, 6))
title = [
    "$\Vert \Alpha^* - \Alpha \Vert$",
    "$\Vert W_e^* - W_e \Vert$",
    "$\Vert W_x^* - W_x \Vert$",
]
for i in range(distances_to_params.shape[1]):
    axs2[i].plot(n_EM_steps, distances_to_params[:, i])
    axs2[i].set_title(title[i])
    axs2[i].set_yscale("log")
fig2.suptitle("Convergence of the estimators towards each model parameter")

viz.plot_2d_dirichlet_samples_with_pdf(X_test, y_test, alpha_list=alpha_list, subtitle="(truth)")
viz.plot_2d_dirichlet_samples_with_pdf(X_test, y_pred, alpha_list=alpha_list, subtitle="(estimated)")

plt.show()

print("-=== Parameters objectives ===-")
print("--------------------")
print("W_e / W_e_hat:")
print(W_e)
print(model.W_e)
print("--------------------")
print("W_x / W_x_hat:")
print(W_x)
print(model.W_x)
print("--------------------")