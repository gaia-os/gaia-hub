from natural_models.kernels.KernelInterface import KernelInterface
import numpyro
from jax import random


class MCMCKernel(KernelInterface):

    default_kernel_args = {
        "num_chains": 4,
        "num_samples": 1000,
        "num_warmup": 1000,
        "progress_bar": True
    }

    def __init__(self, model, kernel_args=None, *args, **kwargs):
        """
        Construct the Monte Carlo Markov chain kernel
        :param model: the model for which to run inference
        :param kernel_args: the keyword argument to provide to the numpyro.infer.MCMC constructor
        """
        super().__init__("MCMCKernel", model)
        self.kernel = numpyro.infer.NUTS(self.model)
        if kernel_args is None:
            kernel_args = {}
        self.kwargs = self.default_kernel_args | kernel_args
        self.mcmc = numpyro.infer.MCMC(self.kernel, **self.kwargs)
        self.samples = None

    def run_inference(self, inference_args=None, model=None, mcmc=None, *args, **kwargs):
        """
        Compute posterior beliefs using Monte Carlo Markov chain
        :return: samples from the posterior distribution
        """

        # Initialise parameters and store the model and inference algorithm
        if model is not None:
            self.kernel = numpyro.infer.NUTS(self.model)
            self.mcmc = numpyro.infer.MCMC(self.kernel, **self.kwargs)
        if mcmc is not None:
            self.mcmc = mcmc
        if inference_args is None:
            inference_args = {}

        # Perform inference using MCMC
        self.prng, rng_key = random.split(self.prng)
        self.mcmc.run(rng_key, **inference_args)
        self.samples = self.mcmc.get_samples()
        return self.samples

    def get_samples(self):
        """
        Getter
        :return: samples from the posterior distribution
        """
        return self.samples

    def get_params(self, *args, **kwargs):
        """
        Getter
        :return: parameters of the posterior distribution if applicable, None otherwise
        """
        return None
