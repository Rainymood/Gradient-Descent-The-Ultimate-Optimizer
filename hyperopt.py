import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Optimizable:
    """
    This is the interface for anything that has parameters that need to be
    optimized, somewhat like torch.nn.Model but with the right plumbing for
    hyperoptimizability. (Specifically, torch.nn.Model uses the Parameter
    interface which does not give us enough control about the detachments.)
    Nominal operation of an Optimizable at the lowest level is as follows:
        o = MyOptimizable(…)
        o.initialize()
        loop {
            o.begin()
            o.zero_grad()
            loss = –compute loss function from parameters–
            loss.backward()
            o.adjust()
        }
    Optimizables recursively handle updates to their optimiz*ers*.
    """

    def __init__(self, parameters, optimizer):
        self.parameters = parameters  # a dict mapping names to tensors
        self.optimizer = optimizer  # which must itself be Optimizable!
        self.all_params_with_gradients = []

    def initialize(self):
        """Initialize parameters, e.g. with a Kaiming initializer."""
        pass

    def begin(self):
        """Enable gradient tracking on current parameters."""
        for name, param in self.parameters.items():
            param.requires_grad_()  # keep gradient information…
            param.retain_grad()  # even if not a leaf…
            self.all_params_with_gradients.append(param)
        self.optimizer.begin()

    def zero_grad(self):
        """ Set all gradients to zero. """
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros(param.shape)
        self.optimizer.zero_grad()

    """ Note: at this point you would probably call .backwards() on the loss
    function. """

    def adjust(self):
        """ Update parameters """
        pass


class MNIST_FullyConnected(Optimizable):
    """
    A fully-connected NN for the MNIST task. This is Optimizable but not itself
    an optimizer.
    """

    def __init__(self, num_inp, num_hid, num_out, optimizer):
        parameters = {
            "w1": torch.zeros(num_inp, num_hid).t(),
            "b1": torch.zeros(num_hid).t(),
            "w2": torch.zeros(num_hid, num_out).t(),
            "b2": torch.zeros(num_out).t(),
        }
        super().__init__(parameters, optimizer)

    def initialize(self):
        nn.init.kaiming_uniform_(self.parameters["w1"], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.parameters["w2"], a=math.sqrt(5))
        self.optimizer.initialize()

    def forward(self, x):
        """Compute a prediction."""
        x = F.linear(x, self.parameters["w1"], self.parameters["b1"])
        x = torch.tanh(x)
        x = F.linear(x, self.parameters["w2"], self.parameters["b2"])
        x = torch.tanh(x)
        x = F.log_softmax(x, dim=1)
        return x

    def adjust(self):
        self.optimizer.adjust(self.parameters)

    def __str__(self):
        return "mnist / " + str(self.optimizer)


class NoOpOptimizer(Optimizable):
    """
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    """

    def __init__(self):
        pass

    def initialize(self):
        pass

    def begin(self):
        pass

    def zero_grad(self):
        pass

    def adjust(self, params):
        pass

    def __str__(self):
        return "static"


class SGD(Optimizable):
    """
    A hyperoptimizable SGD
    """

    def __init__(self, alpha=0.01, optimizer=NoOpOptimizer()):
        parameters = {"alpha": torch.tensor(alpha)}
        super().__init__(parameters, optimizer)

    def adjust(self, params):
        self.optimizer.adjust(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            params[name] = param.detach() - g * self.parameters["alpha"]

    def __str__(self):
        return "sgd(%f) / " % self.parameters["alpha"] + str(self.optimizer)


class SGDPerParam(Optimizable):
    """
    Like above, but can be taught a separate step size for each parameter it
    tunes.
    """

    def __init__(self, alpha=0.01, params=[], optimizer=NoOpOptimizer()):
        parameters = {name + "_alpha": torch.tensor(alpha) for name in params}
        super().__init__(parameters, optimizer)

    def adjust(self, params):
        self.optimizer.adjust(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            params[name] = param.detach() - g * self.parameters[name + "_alpha"]

    def __str__(self):
        return "sgd(%s) / " % str(
            {k: t.item() for k, t in self.parameters.items()}
        ) + str(self.optimizer)


class Adam(Optimizable):
    """
    A fully hyperoptimizable Adam optimizer
    """

    def clamp(x):
        return (x.tanh() + 1.0) / 2.0

    def unclamp(y):
        z = y * 2.0 - 1.0
        return ((1.0 + z) / (1.0 - z)).log() / 2.0

    def __init__(
        self,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        log_eps=-8.0,
        optimizer=NoOpOptimizer(),
    ):
        parameters = {
            "alpha": torch.tensor(alpha),
            "beta1": Adam.unclamp(torch.tensor(beta1)),
            "beta2": Adam.unclamp(torch.tensor(beta2)),
            "log_eps": torch.tensor(log_eps),
        }
        super().__init__(parameters, optimizer)
        self.num_adjustments = 0
        self.cache = {}

    def adjust(self, params):
        self.num_adjustments += 1
        self.optimizer.adjust(self.parameters)
        t = self.num_adjustments
        beta1 = Adam.clamp(self.parameters["beta1"])
        beta2 = Adam.clamp(self.parameters["beta2"])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    "m": torch.zeros(param.shape),
                    "v": torch.zeros(param.shape)
                    + 10.0 ** self.parameters["log_eps"].data
                    # NOTE that we add a little ‘fudge factor' here because sqrt is not
                    # differentiable at exactly zero
                }
            g = param.grad.detach()
            self.cache[name]["m"] = m = (
                beta1 * self.cache[name]["m"].detach() + (1.0 - beta1) * g
            )
            self.cache[name]["v"] = v = (
                beta2 * self.cache[name]["v"].detach() + (1.0 - beta2) * g * g
            )
            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)
            m_hat = m / (1.0 - beta1 ** float(t))
            v_hat = v / (1.0 - beta2 ** float(t))
            dparam = m_hat / (v_hat ** 0.5 + 10.0 ** self.parameters["log_eps"])
            params[name] = param.detach() - self.parameters["alpha"] * dparam

    def __str__(self):
        return "adam(" + str(self.parameters) + ") / " + str(self.optimizer)


class AdamBaydin(Optimizable):
    """ Same as above, but only optimizes the learning rate, treating the
    remaining hyperparameters as constants. """

    def __init__(
        self,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        log_eps=-8.0,
        optimizer=NoOpOptimizer(),
    ):
        parameters = {"alpha": torch.tensor(alpha)}
        self.beta1 = beta1
        self.beta2 = beta2
        self.log_eps = log_eps
        super().__init__(parameters, optimizer)
        self.num_adjustments = 0
        self.cache = {}

    def adjust(self, params):
        self.num_adjustments += 1
        self.optimizer.adjust(self.parameters)
        t = self.num_adjustments
        beta1 = self.beta1
        beta2 = self.beta2
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    "m": torch.zeros(param.shape),
                    "v": torch.zeros(param.shape) + 10.0 ** self.log_eps,
                }
            g = param.grad.detach()
            self.cache[name]["m"] = m = (
                beta1 * self.cache[name]["m"].detach() + (1.0 - beta1) * g
            )
            self.cache[name]["v"] = v = (
                beta2 * self.cache[name]["v"].detach() + (1.0 - beta2) * g * g
            )
            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)
            m_hat = m / (1.0 - beta1 ** float(t))
            v_hat = v / (1.0 - beta2 ** float(t))
            dparam = m_hat / (v_hat ** 0.5 + 10.0 ** self.log_eps)
            params[name] = param.detach() - self.parameters["alpha"] * dparam

    def __str__(self):
        return "adam(" + str(self.parameters) + ") / " + str(self.optimizer)
