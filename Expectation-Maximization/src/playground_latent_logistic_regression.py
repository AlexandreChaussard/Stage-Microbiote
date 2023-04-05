from src.data.dataloader import generate_gaussian
from src.model import LatentLogisticRegression, GaussianMixture
from src.utils.viz import plot_2d_gaussians_samples

mu_list = [-0.1, 0.3]
sigma_list = [0.1, 0.2]

X, y = generate_gaussian(
    n_samples=200,
    d=2,
    mu_list=mu_list,
    sigma_list=sigma_list
)

n_train = 100

X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

# Learn the latent model first
latent_model = GaussianMixture(z_dim=2, seed=0)
latent_model.fit(X)
latent_model.train(
    n_steps=10,
    printEvery=10
)

# Then perform a logistic regression that uses the latent model structure
model = LatentLogisticRegression(latent_model)
model.fit(X_train, y_train)
model.train(
    X_train,
    y_train,
    learning_rate=0.1,
    n_iter=100,
    printEvery=10
)
print(f"Model accuracy is {str(int(model.accuracy(X_test, y_test) * 100))[0:4]}%")

y_pred = model.predict(X_test)

plot_2d_gaussians_samples(X_test, y_test, subtitle="(truth)")
plot_2d_gaussians_samples(X_test, y_pred, subtitle="(prediction)")
