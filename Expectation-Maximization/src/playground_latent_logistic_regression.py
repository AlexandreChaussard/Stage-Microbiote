from src.data.dataloader import generate_gaussian, get_train_test
from src.model import LatentLogisticRegression, GaussianMixture
from src.utils.viz import plot_2d_gaussians_samples, plot_2d_gaussians_samples_with_pdf


mu_list = [[-0.1, -0.2], [0.5, 0.3]]
sigma_list = [[0.1, 0.15], [0.2, 0.1]]

X, y = generate_gaussian(
    n_samples=200,
    d=2,
    mu_list=mu_list,
    sigma_list=sigma_list
)

n_train = 100
X_train, y_train, X_test, y_test = get_train_test(X, y, n_train)

# Learn the latent model first
latent_model = GaussianMixture(z_dim=2, seed=1)
latent_model.fit(X)
latent_model.train(
    n_steps=60,
    printEvery=10
)

# Then perform a logistic regression that uses the latent model structure
model = LatentLogisticRegression(
    latent_model=latent_model,
    embedding_strategy="onehot"
)
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

plot_2d_gaussians_samples_with_pdf(X_test, y_test, mu_list, sigma_list, subtitle="(truth)")
plot_2d_gaussians_samples_with_pdf(X_test, y_pred, latent_model.mu, latent_model.sigma, subtitle="(prediction)")
