import time
import abc
from abc import ABC
import jax
import numpyro


class KernelInterface(ABC):

    def __init__(self, name, model):
        """
        Construct the kernel
        :param name: the kernel name
        :param model: the model for which to run inference
        """
        self.name = name
        self.model = model
        self.prng = jax.random.PRNGKey(int(time.time()))

    @abc.abstractmethod
    def run_inference(self, *args, **kwargs):
        """
        Compute posterior beliefs
        :return: samples from the posterior distribution
        """
        ...

    @abc.abstractmethod
    def get_params(self):
        """
        Getter
        :return: parameters of the posterior distribution if applicable, None otherwise
        """
        ...

    @abc.abstractmethod
    def get_samples(self):
        """
        Getter
        :return: samples from the posterior distribution
        """
        ...

    def print_summary(self):
        """
        Prints a summary table displaying diagnostics of the posterior samples.
        The diagnostics displayed are: mean, standard deviation, median, and 90% credibility interval.
        """
        samples = self.get_samples()
        if samples:
            numpyro.diagnostics.print_summary(samples, prob=0.89, group_by_chain=False)
        else:
            print("Inference has not been performed yet.")
