from src.model.EM.em_template import EMAbstract
from src.utils.distribution import pdf_gaussian
from scipy.special import logsumexp
import numpy as np
from sklearn.cluster import KMeans


class GaussianMixture(EMAbstract):
    """
    Implementation of a GMM model using EM algorithm.

    This method is derived from the analysis of the exponential family.
    The dependence model is the most simple as each Z is independent from the others conditionally to its respective Y.
    """

    def __init__(self, z_dim, seed=None):
        super().__init__(pdf_gaussian, z_dim, seed)

    def fit(self, X, y=None):
        super().fit(X)

        kmean = KMeans(n_clusters=self.z_dim, random_state=self.seed).fit(X)

        # We define the initialization of the parameter the EM algorithm
        # In the case of the gaussian mixture model, theta is given by:
        # theta = [mu_1, ..., mu_c, sigma_1, ..., sigma_c, pi_1, ..., pi_c]
        self.mu = kmean.cluster_centers_
        self.sigma = 0.1 * np.random.randn(self.z_dim, X.shape[1]) ** 2
        # generate a uniform simplex vector for pi
        self.pi = np.ones(self.z_dim) / self.z_dim

    def expectation_step(self):
        # Predict the probability for X to belong to a given gaussian P(Z = c | X)
        # matrix of proba of size n x z_dim

        # List of probabilities of belonging to a given gaussian
        expectations = []

        for x in self.X:

            # List of probabilities of belonging to a given gaussian for a unique sample in X
            proba_belonging = np.zeros(self.z_dim)
            for c in range(self.z_dim):
                proba_belonging[c] = np.log(self.pi[c]) + np.log(self.p_cond(x, self.mu[c], self.sigma[c]))

            proba_belonging -= logsumexp(proba_belonging)
            expectations.append(proba_belonging)

        expectations = np.array(np.exp(expectations))
        return expectations

    def maximization_step(self, expectations):
        # In the maximization step, we update the parameters of the gaussian mixture
        # derived from the derivative of E[log p(X, Z)|X] which is explicit in the gaussian case
        # as part of the exponential family
        for c in range(self.z_dim):
            t_ic = expectations[:, c].reshape(-1, 1)
            N_c = t_ic.sum()

            # Update of pi_c
            self.pi[c] = np.mean(t_ic)
            # update of mu_c
            self.mu[c] = np.sum(self.X * t_ic, axis=0) / N_c
            # update of sigma_c
            self.sigma[c] = np.sqrt(np.sum((self.X - self.mu[c]) ** 2 * t_ic, axis=0) / N_c)
