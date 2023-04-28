import matplotlib.pyplot as plt

import src.utils.viz as viz
from src.data.dataloader import generate_dirichlet, generate_conditional_binary_observations, get_train_test
from src.model import DirichletMixtureClassifier
from src.utils.functions import accuracy
from src.utils.optimizers import GradientDescent

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
    optimizer=GradientDescent(learning_rate=0.1, n_iter=20),
    seed=1,
)
model.fit(X_train, y_train)
model.train(n_steps=10, printEvery=1)

y_pred = model.classify(X_test)

print(f"Model accuracy: {str(accuracy(y_test, y_pred) * 100)[0:4]}%")

viz.plot_2d_dirichlet_samples_with_pdf(X_test, y_test, alpha_list=alpha_list, subtitle="(truth)")
viz.plot_2d_dirichlet_samples_with_pdf(X_test, y_pred, alpha_list=alpha_list, subtitle="(estimated)")

fig, axs = plt.subplots(1, 2, figsize=(15, 9))
axs[0].plot(model.Q_values, label="$Q(\omega, \widehat{\omega})$", marker=".")
axs[1].plot(model.likelihood_values, label="likelihood", marker=".")
axs[1].axhline(
    y=model.compute_loglikelihood(X_train, y_train, pi=[.5, .5], alpha=alpha_list, W_e=W_e, W_x=W_x),
    color='r',
    linestyle='--',
    label="Optimal"
)
axs[0].legend()
axs[1].legend()
fig.suptitle("Q and likelihood values over training")
plt.show()
