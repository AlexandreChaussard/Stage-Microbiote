from src.data.dataloader import generate_gaussian, get_train_test
from src.model import LogisticRegression, LatentLogisticRegression, GaussianMixture
from src.utils.viz import plot_2d_gaussians_samples, plot_2d_gaussians_samples_with_pdf

import numpy as np
import matplotlib.pyplot as plt

# Define the list of means to test
benchmark_mu = [
    [[-0.1, -0.2], [0.5, 0.6]],
    [[-0.1, -0.2], [0.4, 0.5]],
    [[-0.1, -0.2], [0.3, 0.4]],
    [[-0.1, -0.2], [0.2, 0.3]],
    [[-0.1, -0.2], [0.1, 0.2]],
    [[-0.1, -0.2], [0.0, 0.1]],
    [[-0.1, -0.2], [-0.1, 0.0]],
    [[-0.1, -0.2], [-0.2, -0.1]]
]
benchmark_mu = np.array(benchmark_mu)

sigma_list = [
    [0.1, 0.15], [0.2, 0.1]
]

# We save the performances of each classifier:
# Logistic Regression (vanilla style), Latent Logistic Regression (onehot), Latent Logistic Regression (proba)
registered_performances = []

# compute the distance shrinking between the gaussians at each experiment
distances = []

for k, mu in enumerate(benchmark_mu):

    print(f"Benchmarking {k}/{len(benchmark_mu)}: mu = \n{mu}")

    X, y = generate_gaussian(
        n_samples=200,
        d=2,
        mu_list=mu,
        sigma_list=sigma_list
    )

    n_train = 100
    X_train, y_train, X_test, y_test = get_train_test(X, y, n_train)

    # Learn the latent model first
    latent_model = GaussianMixture(z_dim=2, seed=0)
    latent_model.fit(X)
    latent_model.train(
        n_steps=30,
        printEvery=100
    )

    models = {
        "Logistic Regression (vanilla)": LogisticRegression(),
        "Latent Logistic Regression (onehot)": LatentLogisticRegression(
                  latent_model=latent_model,
                  embedding_strategy="onehot"
              ),
        "Latent Logistic Regression (probability)": LatentLogisticRegression(
            latent_model=latent_model,
            embedding_strategy="probability"
        )
    }

    plot_2d_gaussians_samples_with_pdf(X_test, y_test, mu, sigma_list, subtitle=f"(truth - {k})")

    performances = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        model.train(
            X_train,
            y_train,
            learning_rate=0.1,
            n_iter=100,
            printEvery=100
        )

        accuracy = model.accuracy(X_test, y_test)
        print(f"{name} accuracy is {str(int(accuracy * 100))[0:4]}%")
        performances.append(accuracy)

        y_pred = model.predict(X_test)

        plot_2d_gaussians_samples_with_pdf(X_test, y_pred, latent_model.mu, latent_model.sigma, subtitle=f"({name} - {k})")

    registered_performances.append(performances)
    distances.append(np.linalg.norm(mu[0] - mu[1]))


registered_performances = np.array(registered_performances)
distances = np.array(distances)

fig, axs = plt.subplots()
fig.suptitle("Accuracy as a function of the distance between the latent gaussians")
for i, (name, model) in enumerate(models.items()):
    axs.plot(distances, registered_performances[:, i], label=name, marker=".")

axs.legend()
plt.show()