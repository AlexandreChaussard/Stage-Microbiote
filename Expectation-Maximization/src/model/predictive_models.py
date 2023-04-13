from abc import ABC, abstractmethod
import numpy as np

from src.utils.functions import sigmoid, binary_cross_entropy, derivative_binary_cross_entropy, onehot
from src.model.EM import EMAbstract


class BinaryClassifier(ABC):
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

    def predict(self, X):
        probas = self.predict_proba(X)
        label = probas.copy().astype(np.int32)
        label[probas > 0.5] = 1
        label[probas <= 0.5] = 0
        return label.squeeze()

    def accuracy(self, X, y):
        y_hat = self.predict(X)
        return (y == y_hat).mean()


class LogisticRegression(BinaryClassifier):
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


class LatentLogisticRegression(BinaryClassifier):
    """
    Simple logistic regression with latent conditioning
    """

    def __init__(self, latent_model: EMAbstract, embedding_strategy="onehot"):
        self.W_x = None
        self.W_e = None
        self.latent_model = latent_model
        self.embedding_strategy = embedding_strategy

    def fit(self, X, y):
        super().fit(X, y)
        self.W_x = np.zeros(X.shape[1]).reshape(-1, 1)
        self.W_e = np.zeros(self.latent_model.z_dim).reshape(-1, 1)
        return self

    def train(self, X, y, learning_rate=0.1, n_iter=100, printEvery=10):

        # Build the embedding of X
        E = self.embed(X)

        for k in range(n_iter):

            # Model evaluation
            y_hat = self.predict_proba(X)

            # Parameters update
            dW_x = derivative_binary_cross_entropy(y_hat, y, X)
            dW_e = derivative_binary_cross_entropy(y_hat, y, E)
            self.W_x = self.W_x - learning_rate * dW_x
            self.W_e = self.W_e - learning_rate * dW_e

            if printEvery > 0 and k % printEvery == 0:
                print(f"[*] LogisticRegression {k}/{n_iter} - Loss:", binary_cross_entropy(y_hat, y))

        return self

    def embed(self, X):
        # This function is meant to embed the matrix X using its latent module Z
        E = []
        latent_probas = self.latent_model.predict_proba(X)
        for i, x in enumerate(X):
            # Build the embedding representation of X by determining Z from X
            # Then attach a latent model to each sample of X to be onehot encoded for instance
            # which forms the embedding
            latent_proba = latent_probas[i]
            if self.embedding_strategy is None:
                # If we don't have an embedding strategy, we set the embed vector to its null representation
                embedding = np.zeros(self.latent_model.z_dim)
            elif self.embedding_strategy == "probability":
                # Embedding using the probability of belonging to each gaussian
                embedding = latent_proba
            else:
                # Default strategy is onehot embedding
                attached_latent_label = np.argmax(latent_proba).squeeze()
                embedding = onehot(attached_latent_label, self.latent_model.z_dim)
            E.append(embedding)
        E = np.array(E)
        return E

    def predict_proba(self, X):
        proba = np.zeros(X.shape[0])
        # We also build the em
        E = self.embed(X)
        for i, x in enumerate(X):
            # Fetch the embedding of the sample
            embedding = E[i]

            # Compute the probability
            proba[i] = sigmoid(self.W_e.squeeze().dot(embedding) + self.W_x.squeeze().dot(x))

        return proba.reshape(-1, 1)
