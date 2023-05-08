from src.model.EM.em_template import EMAbstract
import numpy as np
from scipy.special import logsumexp
from autograd import grad
from src.utils.distribution import pdf_dirichlet
from src.utils.functions import sigmoid, onehot, digamma, invdigamma


class DirichletMixtureClassifier(EMAbstract):
    """
    Implementation of a Dirichlet Mixture model classifier using EM algorithm.

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
            n_iter_alpha,
            seed=None,
            W_e_init=None,
            W_x_init=None,
            pi_init=None,
            alpha_init=None
    ):
        super().__init__(pdf_dirichlet, z_dim, seed)
        self.optimizer = optimizer
        self.n_iter_alpha = n_iter_alpha
        self.W_e_init = W_e_init
        self.W_x_init = W_x_init
        self.alpha_init = alpha_init
        self.pi_init = pi_init

        self.W_e = None
        self.W_x = None
        self.alpha = None
        self.pi = None

    def fit(self, X, y=None):
        super().fit(X)
        self.y = y

        # We define the initialization of the parameter the EM algorithm

        # In the case of the gaussian mixture model, theta is given by:
        # theta = [alpha_1, alpha_2, ..., pi_1, ..., pi_c]
        if self.alpha_init is None:
            self.alpha = np.random.randn(self.z_dim, X.shape[1])
            self.alpha += np.abs(np.min(self.alpha)) + np.abs(np.random.random())
        else:
            self.alpha = self.alpha

        if self.pi_init is None:
            self.pi = np.ones(self.z_dim) / self.z_dim
        else:
            self.pi = self.pi_init

        # We also get parameters from the classification framework as
        # Embedding parameter, for which one row is a W_e_k
        if self.W_e_init is None:
            self.W_e = np.zeros((self.z_dim, self.z_dim))
        else:
            self.W_e = self.W_e_init
        # Data parameter, for which one row is a W_x_k
        if self.W_x_init is None:
            self.W_x = np.zeros((self.z_dim, X.shape[1]))
        else:
            self.W_x = self.W_x_init

        # Save the values of Q(theta, theta_hat)
        self.Q_values = [self.eval_Q(self.pi, self.alpha, self.W_e, self.W_x)]
        # Save the likelihood values
        self.likelihood_values = []

    def compute_loglikelihood(self, X, Y, pi=None, alpha=None, W_e=None, W_x=None):
        # Compute the log-likelihood as
        # log p(X, Y) = log E[p(X, Y, Z)] = log E[p(Z)p(X|Z)p(Y|X,Z)] = log sum_k(p(Y|X,Z=k)p(X|Z=k)p(Y|X,Z=k))
        # Finally, since all (X_i, Y_i) are iid, log p(X, Y) = sum_i log p(X_i, Y_i)

        if pi is None:
            pi = self.pi
        if alpha is None:
            alpha = self.alpha
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
                            + np.log(self.p_cond(x_i, alpha[c])) \
                            + y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat)

            ll += logsumexp(values)

        return ll / self.n_samples

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
                log_t_i[l] = np.log(self.pi[l]) + np.log(self.p_cond(x_i, self.alpha[l]))

            log_t_i -= logsumexp(log_t_i)

            log_proba_y = np.zeros(self.z_dim)

            for k in range(self.z_dim):
                log_proba_y[k] = np.log(self.classifier_predict_proba(k, x_i)) + log_t_i[k]

            proba[i] = np.exp(logsumexp(log_proba_y))

        if ((proba <= 1).all() and (proba >= 0).all()) is False:
            print("Proba is not a probability vector.")
            print(proba)
            assert False
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
        y_hat = sigmoid(np.dot(W_e[k], e) + np.dot(W_x[k], x))
        if y_hat == np.nan:
            print("NAN")
        return y_hat

    def eval_Q(self, pi, alpha, W_e, W_x):

        total = 0
        expectations = self.expectation_step()

        for c in range(self.z_dim):

            for i in range(len(self.X)):
                x_i = self.X[i]
                y_i = self.y[i]
                y_hat = self.classifier_predict_proba(c, x_i, W_e, W_x)
                t_ic = expectations[i][c]
                total += (np.log(pi[c])
                          + np.log(self.p_cond(x_i, alpha[c]))
                          + y_i * np.log(y_hat) + (1 - y_i) * np.log(1 - y_hat)) * t_ic
        return total / self.n_samples

    def optim_Q_W_x(self, W_x):
        W_x = W_x.reshape(self.W_x.shape)
        return -self.eval_Q(self.pi, self.alpha, self.W_e, W_x)

    def optim_Q_W_e(self, W_e):
        W_e = W_e.reshape(self.W_e.shape)
        return -self.eval_Q(self.pi, self.alpha, W_e, self.W_x)

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
                                     + np.log(self.p_cond(x_i, self.alpha[c]))

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
            t_c = expectations[:, c].reshape(-1,)

            # Update of pi_c
            self.pi[c] = np.mean(t_c)
            # update of alpha_c
            for j in range(len(self.alpha[c])):
                for _ in range(self.n_iter_alpha):
                    self.alpha[c][j] = invdigamma(
                        np.dot(t_c, np.log(self.X[:, j])) / t_c.sum() + digamma(self.alpha[c].sum())
                    )

        # Then we proceed by computing the regression parameters
        # These do not have an analytical optimum, and will be approached using iterative procedures
        # Multiple optimization approaches are proposed:
        # CMAES - SGD - GD

        # CMAES case
        if self.optimizer.method_name == "CMAES":
            self.W_e = self.optimizer.minimize(
                self.optim_Q_W_e,
                x0=self.W_e.reshape(-1, 1),
                sigma0=1
            ).reshape(self.W_e.shape)

            self.W_x = self.optimizer.minimize(
                self.optim_Q_W_x,
                x0=self.W_x.reshape(-1, 1),
                sigma0=1
            ).reshape(self.W_x.shape)

            ll = self.compute_loglikelihood(self.X, self.y)
            self.likelihood_values.append(ll)
            self.printArgs = "\n  * likelihood: " + str(ll) + "\n  * Q: " + str(
                self.eval_Q(self.pi, self.alpha, self.W_e, self.W_x))
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

                        dW_e += - (y_i - y_hat) * t_ic * e_ic / self.n_samples
                        dW_x += - (y_i - y_hat) * t_ic * x_i / self.n_samples

                    # autograd_dW_e = grad(self.optim_Q_W_e)
                    # autograd_dW_x = grad(self.optim_Q_W_x)
                    # print("Autograd W_e", np.linalg.norm(autograd_dW_e(self.W_e)[c] - dW_e))
                    # print("Autograd W_x", np.linalg.norm(autograd_dW_x(self.W_x)[c] - dW_x))

                    W_e[c] -= self.optimizer.learning_rate * dW_e
                    W_x[c] -= self.optimizer.learning_rate * dW_x

                # measurement of the improvement
                measurement = self.eval_Q(self.pi, self.alpha, W_e, W_x)
                if measurement > eval:
                    self.W_e = W_e.copy()
                    self.W_x = W_x.copy()
                    eval = measurement
                    self.Q_values.append(eval)

            ll = self.compute_loglikelihood(self.X, self.y)
            self.likelihood_values.append(ll)
            self.printArgs = "\n  * likelihood: " + str(ll) + "\n  * Q: " + str(
                self.eval_Q(self.pi, self.alpha, W_e, W_x))