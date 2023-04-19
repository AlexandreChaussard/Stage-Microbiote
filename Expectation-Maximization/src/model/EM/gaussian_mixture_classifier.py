from src.model.EM.em_template import EMAbstract
import numpy as np
from scipy.special import logsumexp

from src.utils.distribution import pdf_gaussian
from src.utils.functions import sigmoid, onehot


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
            seed=None,
            W_e_init=None,
            W_x_init=None
    ):
        super().__init__(pdf_gaussian, z_dim, seed)
        self.optimizer = optimizer
        self.W_e = W_e_init
        self.W_x = W_x_init

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
        if self.W_e is None:
            self.W_e = np.zeros((self.z_dim, self.z_dim))
        # Data parameter, for which one row is a W_x_k
        if self.W_x is None:
            self.W_x = np.zeros((self.z_dim, X.shape[1]))

        # Save the values of Q(theta, theta_hat)
        self.Q_values = [self.eval_Q(self.pi, self.mu, self.sigma, self.W_e, self.W_x)]
        # Save the likelihood values
        self.likelihood_values = []

    def compute_loglikelihood(self, X, Y, pi=None, mu=None, sigma=None, W_e=None, W_x=None):
        # Compute the log-likelihood as
        # log p(X, Y) = log E[p(X, Y, Z)] = log E[p(Z)p(X|Z)p(Y|X,Z)] = log sum_k(p(Y|X,Z=k)p(X|Z=k)p(Y|X,Z=k))
        # Finally, since all (X_i, Y_i) are iid, log p(X, Y) = sum_i log p(X_i, Y_i)

        if pi is None:
            pi = self.pi
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if W_e is None:
            W_e = self.W_e
        if W_x is None:
            W_x = self.W_x

        # Compute the log-likelihood
        ll = 0
        for i, x_i in enumerate(X):
            y_i = Y[i]
            values = np.zeros(self.z_dim)
            for c in range(self.z_dim):
                # Compute P(Y = 1 | X, Z=c)
                y_hat = self.classifier_predict_proba(c, x_i, W_e, W_x)
                values[c] = np.log(pi[c]) \
                            + np.log(self.p_cond(x_i, mu[c], sigma[c])) \
                            + y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat)

            ll += logsumexp(values)

        return ll

    def embed(self, c):
        return onehot(c, self.z_dim)

    def classify(self, X):
        # Output Y knowing X:
        # P(Y | X) = sum_k P(Y | X, Z=k) P(X|Z=k)P(Z=k) / sum_l P(X|Z=l)P(Z=l)

        # Initialize the proba vector
        proba = np.zeros(len(X))

        for i in range(len(X)):
            x_i = X[i]

            log_t_i = np.zeros(self.z_dim)
            for l in range(self.z_dim):
                log_t_i[l] = np.log(self.pi[l]) + np.log(self.p_cond(x_i, self.mu[l], self.sigma[l]))

            log_t_i -= logsumexp(log_t_i)

            log_proba_y = np.zeros(self.z_dim)

            for k in range(self.z_dim):
                log_proba_y[k] = np.log(self.classifier_predict_proba(k, x_i)) + log_t_i[k]

            proba[i] = np.exp(logsumexp(log_proba_y))
        assert ((proba < 1).all())
        y = (proba > 0.5) * 1
        return y

    def classifier_predict_proba(self, k, x, W_e=None, W_x=None):
        # Predicts P(Y = 1 | X, Z=k)
        if W_e is None or W_x is None:
            W_e = self.W_e
            W_x = self.W_x
        # First we onehot the class to turn it into an embedding vector
        e = self.embed(k)
        # then we compute P(Y = 1 | X, Z = k)
        return sigmoid(W_e[k].dot(e) + W_x[k].dot(x))

    def eval_Q(self, pi, mu, sigma, W_e, W_x):

        total = 0
        expectations = self.expectation_step()

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

    def optim_Q_W_x(self, W_x, pi, mu, sigma, W_e):
        W_x = W_x.reshape(self.W_x.shape)
        return -self.eval_Q(pi, mu, sigma, W_e, W_x)

    def optim_Q_W_e(self, W_e, pi, mu, sigma, W_x):
        W_e = W_e.reshape(self.W_e.shape)
        return -self.eval_Q(pi, mu, sigma, W_e, W_x)

    def expectation_step(self):
        # At this stage, we evaluate the probability of each sample to belong to a given group
        # as P(Z = c | X, Y):
        # P(Z=c|X,Y) = P(Z=c) P(X|Z=c) P(Y|X, Z=c) / sum_l (same)
        # matrix of proba of size n x z_dim

        # List of probabilities of belonging to a given gaussian t_ik
        expectations = np.zeros((len(self.X), self.z_dim))

        for i, x_i in enumerate(self.X):
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

    def maximization_step(self, expectations):
        # In the maximization step, we update the parameters of the gaussian mixture
        # derived from the derivative of E[log p(X, Z)|X] which is explicit in the gaussian case
        # as part of the exponential family

        # First we compute the exact optimum for the gaussian models we have
        # These are exactly computed from the maximum likelihood estimator
        for c in range(self.z_dim):

            # Column vector of [t_ic]_i for the case Z = c
            t_c = expectations[:, c].reshape(-1, 1)
            N_c = t_c.sum()

            # Update of pi_c
            self.pi[c] = np.mean(t_c)
            # update of mu_c
            self.mu[c] = np.sum(self.X * t_c, axis=0) / N_c
            # update of sigma_c
            self.sigma[c] = np.sqrt(np.sum((self.X - self.mu[c]) ** 2 * t_c, axis=0) / N_c)
            if 0 in self.sigma[c]:
                print("[!] One gaussian has been set to 0.")
                self.stop_training = True
                return

        # Then we proceed by computing the regression parameters
        # These do not have an analytical optimum, and will be approached using iterative procedures
        # Multiple optimization approaches are proposed:
        # CMAES - SGD - GD

        # CMAES case
        if self.optimizer.method_name == "CMAES":
            self.W_e = self.optimizer.minimize(
                self.optim_Q_W_e,
                x0=self.W_e.reshape(-1, 1),
                sigma0=1,
                fun_args=(self.pi, self.mu, self.sigma, self.W_x)
            ).reshape(self.W_e.shape)

            self.W_x = self.optimizer.minimize(
                self.optim_Q_W_x,
                x0=self.W_x.reshape(-1, 1),
                sigma0=1,
                fun_args=(self.pi, self.mu, self.sigma, self.W_e)
            ).reshape(self.W_x.shape)

            ll = self.compute_loglikelihood(self.X, self.y)
            self.likelihood_values.append(ll)
            self.printArgs = "\n  * likelihood: " + str(ll) + "\n  * Q: " + str(
                self.eval_Q(self.pi, self.mu, self.sigma, self.W_e, self.W_x))
            return

        # Gradient methods

        # Save variables to perform the comparison of iterates and keep only the improvements
        W_e = self.W_e.copy()
        W_x = self.W_x.copy()

        # Evaluation of Q at the current parameters
        # This serves as the acceptation criterion for the iterates of the gradient descent
        # We want to make sure that Q(theta_k+1, theta_hat) > Q(theta_k, theta_hat)
        eval = self.Q_values[-1]
        for c in range(self.z_dim):
            # update the classification parameters

            # Gradient Descent-like methods (SGD & GD)
            for iter in range(self.optimizer.n_iter):
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

                        dW_e += - (y_i - y_hat) * t_ic * e_ic
                        dW_x += - (y_i - y_hat) * t_ic * x_i

                    decay = 1
                    if self.optimizer.step_size_decay:
                        decay = iter + 1

                    W_e[c] -= self.optimizer.learning_rate / decay * dW_e
                    W_x[c] -= self.optimizer.learning_rate / decay * dW_x
                else:  # Default is gradient descent
                    for i, y_i in enumerate(self.y):
                        x_i = self.X[i]
                        y_hat = self.classifier_predict_proba(c, x_i)
                        t_ic = expectations[i][c]
                        e_ic = self.embed(c)

                        dW_e += - (y_i - y_hat) * t_ic * e_ic
                        dW_x += - (y_i - y_hat) * t_ic * x_i

                    W_e[c] -= self.optimizer.learning_rate * dW_e
                    W_x[c] -= self.optimizer.learning_rate * dW_x

                # measurement of the improvement
                measurement = self.eval_Q(self.pi, self.mu, self.sigma, W_e, W_x)
                if measurement > eval:
                    self.W_e = W_e.copy()
                    self.W_x = W_x.copy()
                    eval = measurement
                    self.Q_values.append(eval)

            ll = self.compute_loglikelihood(self.X, self.y)
            self.likelihood_values.append(ll)
            self.printArgs = "\n  * likelihood: " + str(ll) + "\n  * Q: " + str(
                self.eval_Q(self.pi, self.mu, self.sigma, W_e, W_x))

    # Optimization functions for CMAES
    def Q_fun_W_e(self, W_e_c, c):
        total = 0
        W_e = np.stack((W_e_c, W_e_c))
        expectations = self.expectation_step()

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
        expectations = self.expectation_step()

        for i in range(len(self.X)):
            x_i = self.X[i]
            y_i = self.y[i]
            y_hat = self.classifier_predict_proba(c, x_i, self.W_e, W_x)
            t_ic = expectations[i][c]

            total += (y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat)) * t_ic
        return -total
