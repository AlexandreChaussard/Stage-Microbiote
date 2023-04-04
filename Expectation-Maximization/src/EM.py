import numpy as np
from src.distribution import pdf_gaussian


class GaussianMixture():

    def __init__(self, z_dim, mu_list, sigma_list, distrib_list):

        # We first give ourselves a family of conditional distributions p_cond which modelizes
        # P_theta (X|Z)
        # Which is parameterized by theta (parameters to be learnt)
        # In the gaussian mixture case, it corresponds to the family of multivariate gaussians
        self.p_cond = pdf_gaussian

        # We define the size of the hidden states that would determine X
        # We decide that Z is discrete
        self.z_dim = z_dim

        # We define the initialization of the parameter the EM algorithm
        # In the case of the gaussian mixture model, theta is given by:
        # theta = [mu_1, ..., mu_c, sigma_1, ..., sigma_c, pi_1, ..., pi_c]
        self.mu = mu_list
        self.sigma = sigma_list
        self.pi = distrib_list

    def fit(self, X):
        # Just fetch the dataset
        self.X = X
        return self

    def expectation_step(self):
        expectations = []

        # At this stage, we estimate the probability of each sample to belong to a given gaussian
        for x in self.X:
            sum_distrib = 0
            for c in range(self.z_dim):
                sum_distrib += self.pi[c] * self.p_cond(x, self.mu[c], self.sigma[c])

            # List of probabilities of belonging to a given gaussian
            proba_belonging = [0] * self.z_dim
            for c in range(self.z_dim):
                proba_belonging[c] = self.pi[c] * self.p_cond(x, self.mu[c], self.sigma[c]) / sum_distrib

            proba_belonging = np.array(proba_belonging)
            expectations.append(proba_belonging)

        return np.array(expectations)

    def maximization_step(self, expectations):
        for c in range(self.z_dim):
            self.pi[c] = np.mean(expectations[:, c])

            sum_mu = 0
            for i, x in enumerate(self.X):
                sum_mu += x * expectations[i][c]

            self.mu[c] = sum_mu / expectations[:, c].sum()

            sum_sigma = 0
            for i, x in enumerate(self.X):
                sum_sigma += (x - self.mu[c]) ** 2 * expectations[i][c]
            self.sigma[c] = np.sqrt(sum_sigma / expectations[:, c].sum())

    def train(self, n_steps, printEvery=-1):
        for _ in range(n_steps):
            self.maximization_step(self.expectation_step())
            if printEvery > 0 and _ % printEvery == 0:
                print(f"[*] EM ({_}/{n_steps})")

        self.expectation_step()

        return self
