from src.data.dataloader import generate_gaussian, generate_conditional_binary_observations, get_train_test
import src.utils.viz as viz
from src.model import GaussianMixtureClassifier
from src.utils.functions import accuracy
from src.utils.optimizers import GradientDescent, StochasticGradientDescent, CMAES

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

gmm = GaussianMixtureClassifier(
    z_dim=2,
    optimizer=GradientDescent(learning_rate=0.05, n_iter=10),
    seed=0
)

gmm.fit(X_train, y_train)
gmm.train(n_steps=15, printEvery=1)

y_pred = gmm.classify(X_test)

print(f"Model accuracy: {str(accuracy(y_test, y_pred)*100)[0:4]}%")

viz.plot_2d_gaussians_samples_with_pdf(X_test, y_test, mu=mu_list, sigma=sigma_list, subtitle="(truth)")
viz.plot_2d_gaussians_samples_with_pdf(X_test, y_pred, mu=gmm.mu, sigma=gmm.sigma, subtitle="(estimated)")
