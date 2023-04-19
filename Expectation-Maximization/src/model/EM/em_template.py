from abc import ABC, abstractmethod
import numpy as np


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
