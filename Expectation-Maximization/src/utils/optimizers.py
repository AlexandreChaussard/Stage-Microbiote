from abc import ABC, abstractmethod
import cma


class Optimizer(ABC):
    def __init__(self, method_name, learning_rate, n_iter):
        self.method_name = method_name
        self.learning_rate = learning_rate
        self.n_iter = n_iter


class GradientDescent(Optimizer):
    def __init__(self, learning_rate, n_iter):
        super(GradientDescent, self).__init__("GD", learning_rate, n_iter)


class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate, n_iter, step_size_decay=False, batch_size=1):
        super(StochasticGradientDescent, self).__init__("SGD", learning_rate, n_iter)
        self.step_size_decay = step_size_decay
        self.batch_size = batch_size


class CMAES(Optimizer):
    def __init__(self):
        super(CMAES, self).__init__("CMAES", 0, 0)

    def minimize(self, function, x0, sigma0, fun_args):
        return cma.fmin(
            function,
            x0=x0,
            sigma0=sigma0,
            args=fun_args,
            options={
                'tolfun': 1e-1,
                'maxiter': 100,
                'tolflatfitness': 3
            })[0]
