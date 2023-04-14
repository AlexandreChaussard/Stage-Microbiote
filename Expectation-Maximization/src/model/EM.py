from abc import ABC, abstractmethod
import numpy as np
from scipy.special import logsumexp

from src.utils.distribution import pdf_gaussian
from src.utils.functions import sigmoid, onehot
from src.utils.optimizers import Optimizer

from sklearn.cluster import KMeans


class EMAbstract(ABC):
    """
    Abstract class for the EM implementations
    """

    def __init__(self, pdf, z_dim, seed=None):
        # Seed the environment
        np.random.seed(seed)
        self.seed = seed

        # We first give ourselves a family of conditional distributions p_cond which modelizes
        # P_theta (X|Z)
        # Which is parameterized by theta (parameters to be learnt)
        # In the gaussian mixture case, it corresponds to the family of multivariate gaussians
        self.p_cond = pdf

        # We define the size of the hidden states that would determine X
        # We decide that Z is discrete
        self.z_dim = z_dim

        # We add a breakpoint in the algorithm to stop it during the training if something has went wrong
        self.stop_training = False

        # Print argument that can be added to the print when training
        self.printArgs = ""

    def fit(self, X, y=None):
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
            if self.stop_training:
                self.stop_training = False
                break
            # Expectation step then maximization step using the output of the expectation if it has one
            self.maximization_step(self.expectation_step())
            if printEvery > 0 and _ % printEvery == 0:
                print(f"[*] EM ({_}/{n_steps}): {self.printArgs}")
        return self


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

    def predict_proba(self, X):
        # Predict the probability for X to belong to a given gaussian P(Z = c | X)
        # matrix of proba of size n x z_dim

        # List of probabilities of belonging to a given gaussian
        expectations = []

        for x in X:

            # List of probabilities of belonging to a given gaussian for a unique sample in X
            proba_belonging = np.zeros(self.z_dim)
            for c in range(self.z_dim):
                proba_belonging[c] = np.log(self.pi[c]) + np.log(self.p_cond(x, self.mu[c], self.sigma[c]))

            proba_belonging -= logsumexp(proba_belonging)
            expectations.append(proba_belonging)

        expectations = np.array(np.exp(expectations))
        return expectations

    def expectation_step(self):
        # At this stage, we evaluate the probability of each sample to belong to a given gaussian
        # as P(Z = c | X)
        return self.predict_proba(self.X)

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


class GaussianMixtureClassifier(EMAbstract):
    """
    Implementation of a GMM model classifier using EM algorithm.

    This method is derived from the analysis of the exponential family.
    The dependence model is the most simple as each Z is independent from the others conditionally to its respective Y.

    For the classification part, we set the following framework:
    * P(Y = 1 | X, Z = k) = sigmoid(W_e.T e_k + W_x.T X)

    The strategy to build e_k is through onehot encoding: k -> [0,..., 1, 0, ..., 0] at k-th position
    """

    def __init__(
            self,
            z_dim,
            optimizer,
            seed=None
    ):
        super().__init__(pdf_gaussian, z_dim, seed)
        self.optimizer = optimizer

    def fit(self, X, y=None):
        super().fit(X)
        self.y = y

        # We define the initialization of the parameter the EM algorithm

        # In the case of the gaussian mixture model, theta is given by:
        # theta = [mu_1, ..., mu_c, sigma_1, ..., sigma_c, pi_1, ..., pi_c]
        self.mu = np.random.randn(self.z_dim, X.shape[1])
        self.sigma = np.random.randn(self.z_dim, X.shape[1]) ** 2 + 1
        # generate a uniform simplex vector for pi
        self.pi = np.ones(self.z_dim) / self.z_dim

        # We also get parameters from the classification framework as
        # Embedding parameter, for which one row is a W_e_k
        self.W_e = np.zeros((self.z_dim, self.z_dim))
        # Data parameter, for which one row is a W_x_k
        self.W_x = np.zeros((self.z_dim, X.shape[1]))

    def estimate_Z(self, X):
        # This function enables to compute Z only knowing X (not Y)
        # Compute P(Z = c | X) and threshold to 0.5 to obtain Z
        Z = np.zeros(len(X), np.int32)
        for i in range(len(X)):
            x_i = X[i]
            probas = np.zeros(self.z_dim)
            for c in range(self.z_dim):
                probas[c] = np.log(self.pi[c]) + np.log(self.p_cond(x_i, self.mu[c], self.sigma[c]))
            # the normalization step is unnecessary to determine the maximum
            Z[i] = np.argmax(probas)
        return Z

    def compute_loglikelihood(self, X, Y):
        # Compute the log-likelihood as
        # log p(X, Y) = log E[p(X, Y, Z)] = log E[p(Z)p(X|Z)p(Y|X,Z)] = log sum_k(p(Y|X,Z=k)p(X|Z=k)p(Y|X,Z=k))
        # Finally, since all (X_i, Y_i) are iid, log p(X, Y) = sum_i log p(X_i, Y_i)

        # First we need to estimate Z
        Z = self.estimate_Z(X)
        # Then we can compute the log-likelihood
        ll = 0
        for i, x_i in enumerate(X):
            z_i = Z[i]
            y_i = Y[i]
            y_hat = self.classifier_predict_proba(z_i, x_i)
            values = np.zeros(self.z_dim)
            for c in range(self.z_dim):
                values[c] = np.log(self.pi[c]) \
                            + np.log(self.p_cond(x_i, self.mu[c], self.sigma[c])) \
                            + y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat)

            ll += logsumexp(values)

        return ll

    def embed(self, c):
        return onehot(c, self.z_dim)

    def classify(self, X):
        # Output Y knowing X, Z
        # Initialize the output Y vector
        y = np.zeros(len(X), np.int32)

        # Estimate Z from X only since we do not observe Y
        Z = self.estimate_Z(X)

        for i in range(len(X)):
            x_i = X[i]
            z_i = Z[i]
            proba = self.classifier_predict_proba(z_i, x_i)
            y[i] = (proba > 0.5) * 1
        return y

    def classifier_predict_proba(self, k, x, W_e=None, W_x=None):
        if W_e is None or W_x is None:
            W_e = self.W_e
            W_x = self.W_x
        # First we onehot the class to turn it into an embedding vector
        e = self.embed(k)
        # then we compute P(Y = 1 | X, Z = k)
        return sigmoid(W_e[k].dot(e) + W_x[k].dot(x))

    def predict_proba(self, X):
        # Evaluate P(Z = c | X, Y)
        # matrix of proba of size n x z_dim

        # List of probabilities of belonging to a given gaussian
        expectations = np.zeros((len(X), self.z_dim))

        for i, x_i in enumerate(X):
            y_i = self.y[i]

            # List of probabilities of belonging to a given gaussian for a unique sample in X (t_ik)
            proba_belonging = np.zeros(self.z_dim)
            for c in range(self.z_dim):
                y_hat = self.classifier_predict_proba(c, x_i)
                proba_belonging[c] = np.log(self.pi[c]) \
                                     + y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat) \
                                     + np.log(self.p_cond(x_i, self.mu[c], self.sigma[c]))

            proba_belonging -= logsumexp(proba_belonging)

            expectations[i] = proba_belonging

        return np.array(np.exp(expectations))

    def eval_Q(self, pi, mu, sigma, W_e, W_x):

        total = 0
        expectations = self.predict_proba(self.X)

        for c in range(self.z_dim):

            for i in range(len(self.X)):
                x_i = self.X[i]
                y_i = self.y[i]
                y_hat = self.classifier_predict_proba(c, x_i, W_e, W_x)
                t_ic = expectations[i][c]

                total += (np.log(pi[c])
                          + np.log(self.p_cond(x_i, mu[c], sigma[c]))
                          + y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat)) * t_ic
        return total

    def Q_fun_W_e(self, W_e_c, c):
        total = 0
        W_e = np.stack((W_e_c, W_e_c))
        expectations = self.predict_proba(self.X)

        for i in range(len(self.X)):
            x_i = self.X[i]
            y_i = self.y[i]
            y_hat = self.classifier_predict_proba(c, x_i, W_e, self.W_x)
            t_ic = expectations[i][c]
            total += (y_i * np.log(y_hat) + (1. - y_i) * np.log(1. - y_hat)) * t_ic
        return -total

    def Q_fun_W_x(self, W_x_c, c):
        total = 0
        W_x = np.stack((W_x_c, W_x_c))
        expectations = self.predict_proba(self.X)

        for i in range(len(self.X)):
            x_i = self.X[i]
            y_i = self.y[i]
            y_hat = self.classifier_predict_proba(c, x_i, self.W_e, W_x)
            t_ic = expectations[i][c]

            total += (y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat)) * t_ic
        return -total

    def expectation_step(self):
        # At this stage, we evaluate the probability of each sample to belong to a given group
        # as P(Z = c | X, Y)
        return self.predict_proba(self.X)

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
            if 0 in self.sigma[c]:
                print("One gaussian has been set to 0.")
                self.stop_training = True
                break

            # update the classification parameters
            # Multiple optimization approaches are proposed:
            # CMAES - SGD - GD

            # CMAES Optim
            if self.optimizer.method_name == "CMAES":
                self.W_e[c] = self.optimizer.minimize(self.Q_fun_W_e, self.W_e[c], 1, (c,))
                self.W_x[c] = self.optimizer.minimize(self.Q_fun_W_x, self.W_x[c], 1, (c,))
                self.printArgs = "likelihood: " + str(self.compute_loglikelihood(self.X, self.y))
                continue

            # Gradient Descent-like methods (SGD & GD)
            W_e = self.W_e.copy()
            W_x = self.W_x.copy()
            # Evaluation step
            eval = self.eval_Q(self.pi, self.mu, self.sigma, W_e, W_x)
            for _ in range(self.optimizer.n_iter):
                dW_e = 0
                dW_x = 0

                if self.optimizer.method_name == "SGD":
                    indexes = np.random.choice(np.arange(0, len(self.y)), size=self.optimizer.batch_size, replace=False)
                    for i in indexes:
                        x_i = self.X[i]
                        y_i = self.y[i]
                        y_hat = self.classifier_predict_proba(c, x_i)
                        t_ic = expectations[i][c]
                        e_ic = self.embed(c)

                        dW_e += (y_i - y_hat) * t_ic * e_ic
                        dW_x += (y_i - y_hat) * t_ic * x_i

                    decay = 1
                    if self.optimizer.step_size_decay:
                        decay = _ + 1

                    W_e[c] = W_e[c] + self.optimizer.learning_rate / decay * dW_e
                    W_x[c] = W_x[c] + self.optimizer.learning_rate / decay * dW_x
                else:  # Default is gradient descent
                    for i, y_i in enumerate(self.y):
                        x_i = self.X[i]
                        y_hat = self.classifier_predict_proba(c, x_i)
                        t_ic = expectations[i][c]
                        e_ic = self.embed(c)

                        dW_e += (y_i - y_hat) * t_ic * e_ic
                        dW_x += (y_i - y_hat) * t_ic * x_i

                    W_e[c] = W_e[c] + self.optimizer.learning_rate * dW_e
                    W_x[c] = W_x[c] + self.optimizer.learning_rate * dW_x

                # measurement of the improvement
                measurement = self.eval_Q(self.pi, self.mu, self.sigma, W_e, W_x)
                if eval < measurement:
                    self.W_e = W_e.copy()
                    self.W_x = W_x.copy()
                    eval = measurement

            self.printArgs = "likelihood: " + str(self.compute_loglikelihood(self.X, self.y))
