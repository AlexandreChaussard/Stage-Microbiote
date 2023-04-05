from abc import ABC, abstractmethod
import numpy as np
from src.utils.distribution import pdf_gaussian


class EMAbstract(ABC):
    """
    Abstract class for the EM implementations
    """

    def __init__(self, pdf, z_dim, seed=None):
        # Seed the environment
        np.random.seed(seed)

        # We first give ourselves a family of conditional distributions p_cond which modelizes
        # P_theta (X|Z)
        # Which is parameterized by theta (parameters to be learnt)
        # In the gaussian mixture case, it corresponds to the family of multivariate gaussians
        self.p_cond = pdf

        # We define the size of the hidden states that would determine X
        # We decide that Z is discrete
        self.z_dim = z_dim

    def fit(self, X):
        # Just fetch the dataset
        self.X = X
        return self

    @abstractmethod
    def expectation_step(self):
        pass

    @abstractmethod
    def maximization_step(self, expectations):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def train(self, n_steps, printEvery=-1):
        for _ in range(n_steps):
            # Expectation step then maximization step using the output of the expectation if it has one
            self.maximization_step(self.expectation_step())
            if printEvery > 0 and _ % printEvery == 0:
                print(f"[*] EM ({_}/{n_steps})")

        self.expectation_step()

        return self


class GaussianMixture(EMAbstract):
    """
    Implementation of a GMM model using EM algorithm.

    This method is derived from the analysis of the exponential family.
    The dependence model is the most simple as each Z is independent from the others conditionally to its respective Y.
    """

    def __init__(self, z_dim, seed=None):
        super().__init__(pdf_gaussian, z_dim, seed)

        # We define the initialization of the parameter the EM algorithm
        # In the case of the gaussian mixture model, theta is given by:
        # theta = [mu_1, ..., mu_c, sigma_1, ..., sigma_c, pi_1, ..., pi_c]
        self.mu = np.random.randn(z_dim)
        self.sigma = np.abs(0.5 * np.random.randn(z_dim))
        # generate a uniform simplex vector for pi
        self.pi = np.ones(z_dim) / z_dim

    def predict_proba(self, X):
        # Predict the probability for X to belong to a given gaussian P(Z = c | X)
        # matrix of proba of size n x z_dim
        probas = []
        for x in X:
            proba_belonging = [self.p_cond(x, self.mu[c], self.sigma[c]) for c in range(self.z_dim)]
            probas.append(proba_belonging)
        return np.array(probas)

    def expectation_step(self):
        expectations = []

        # At this stage, we evaluate the probability of each sample to belong to a given gaussian
        # as P(Z = c | X), and we normalize it by the sum of the proba of belonging to any gaussian
        for x in self.X:
            sum_distrib = 0
            for c in range(self.z_dim):
                sum_distrib += self.pi[c] * self.p_cond(x, self.mu[c], self.sigma[c])

            # List of probabilities of belonging to a given gaussian normalized
            proba_belonging = np.zeros(self.z_dim)
            for c in range(self.z_dim):
                proba_belonging[c] = self.pi[c] * self.p_cond(x, self.mu[c], self.sigma[c]) / sum_distrib

            expectations.append(proba_belonging)

        return np.array(expectations)

    def maximization_step(self, expectations):
        # In the maximization step, we update the parameters of the gaussian mixture
        # derived from the derivative of E[log p(X, Z)|X] which is explicit in the gaussian case
        # as part of the exponential family
        for c in range(self.z_dim):

            N_c = expectations[:, c].sum()
            sum_mu = 0
            sum_sigma = 0
            for i, x in enumerate(self.X):
                sum_mu += x * expectations[i][c]
                sum_sigma += (x - self.mu[c]) ** 2 * expectations[i][c]

            # Update of pi_c
            self.pi[c] = np.mean(expectations[:, c])
            # update of mu_c
            self.mu[c] = sum_mu / N_c
            # update of sigma_c
            self.sigma[c] = np.sqrt(sum_sigma / N_c)
