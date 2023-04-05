from abc import ABC, abstractmethod
import numpy as np

from src.utils.functions import sigmoid, binary_cross_entropy, derivative_binary_cross_entropy
from src.model.EM import EMAbstract


class PredictionModel(ABC):
    """
    Define a typical architecture for prediction models
    """

    def fit(self, X, y):
        self.X = X
        self.y = y

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass


class LatentPredictionModel(PredictionModel):
    """
    Define a typical architecture for latent representation conditional models
    """

    def __init__(self, latent_model):
        self.latent_model = latent_model


class LogisticRegression(PredictionModel):
    """
    Simple logistic regression without latent conditioning
    """

    def __init__(self):
        self.W = None

    def fit(self, X, y):
        super().fit(X, y)
        self.W = np.zeros(X.shape[1]).reshape(-1, 1)
        return self

    def train(self, X, y, learning_rate=0.1, n_iter=100, printEvery=10):

        for k in range(n_iter):

            # Model evaluation
            y_hat = self.predict_proba(X)

            # Parameter update
            dW = derivative_binary_cross_entropy(y_hat, y, X)
            self.W = self.W - learning_rate * dW

            if printEvery > 0 and k % printEvery == 0:
                print(f"[*] LogisticRegression {k}/{n_iter} - Loss:", binary_cross_entropy(y_hat, y))

        return self

    def predict_proba(self, X):
        proba = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            proba[i] = sigmoid(self.W.squeeze().dot(x))
        return proba.reshape(-1, 1)

    def predict(self, X):
        probas = self.predict_proba(X)
        label = probas.copy()
        label[probas > 0.5] = 1
        label[probas <= 0.5] = 0
        return label.squeeze()

    def accuracy(self, X, y):
        y_hat = self.predict(X)
        return (y == y_hat).mean()
