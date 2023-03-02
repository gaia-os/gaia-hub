from numpyro.infer import Predictive
from numpyro.infer import Trace_ELBO, SVI
from numpyro.optim import Adam
from jax import random
from open_science_network.kernels.KernelInterface import KernelInterface


class SVIKernel(KernelInterface):

    def __init__(self, model, guide=None, kernel_args=None, **kwargs):
        """
        Construct the stochastic variational inference kernel
        :param model: the model for which to run inference
        :param guide: the variational distribution to use when performing inference
        :param optimiser: the Numpyro optimiser to use for minimising the loss
        :param loss: the negative Evidence Lower Bound minimised to compute the posterior beliefs
        """

        super().__init__("SVIKernel", model)

        self.optimiser = kernel_args["optimiser"] if self.is_in("optimiser", kernel_args) else Adam(step_size=5e-3)
        self.loss = kernel_args["loss"] if self.is_in("loss", kernel_args) else Trace_ELBO()
        self.num_steps = kernel_args["num_steps"] if self.is_in("num_steps", kernel_args) else 100
        self.guide = guide

        self.svi = SVI(self.model, self.guide, optim=self.optimiser, loss=self.loss)

        self.posterior_samples = None
        self.svi_result = None
        self.posterior_params = None

    @staticmethod
    def is_in(key, kernel_args):
        """
        Check whether the key is in the kernel arguments
        :return: True if the key is in the kernel arguments, False otherwise
        """
        return kernel_args is not None and key in kernel_args.keys()

    def run_inference(
        self, model=None, guide=None, inference_params=None, prediction_params=None, *args, **kwargs
    ):
        """
        Compute posterior beliefs using stochastic variational inference
        :param model: the model on which to run inference
        :param guide: the guide to use to perform inference
        :param inference_params: the parameters to pass as parameter to SVI.run()
        :param prediction_params: the parameters to pass to the Predictive class
        :return: samples from the posterior distribution
        """

        # Initialise parameters and store the model, guide and inference algorithm
        if guide is not None:
            self.guide = guide
        if model is not None:
            self.model = model
        if inference_params is None:
            inference_params = {}
        if prediction_params is None:
            prediction_params = {}
        if model is not None or guide is not None:
            self.svi = SVI(self.model, self.guide, optim=self.optimiser, loss=self.loss)

        # Tell the user that the agent needs
        if self.guide is None:
            raise RuntimeError("the agent needs to be conditioned on the data, before inference can be run.")

        # Perform inference
        self.prng, _rng_key = random.split(self.prng)
        svi_result = self.svi.run(_rng_key, self.num_steps, **inference_params)

        # Store the results of inference
        self.svi_result = svi_result
        self.posterior_params = self.svi_result.params

        # Perform prediction about the future
        self.posterior_samples = self.predict(prediction_params)
        return self.posterior_samples

    def predict(self, prediction_params):
        """
        Perform prediction about the future based on the posterior beliefs
        :param prediction_params: the parameters to provide to the Numpyro Predictive class
        :return: samples from the predictive distribution
        """
        prediction_params = {"num_samples": 1000} | prediction_params
        self.prng, _rng_key = random.split(self.prng)
        posterior_predictive = Predictive(self.model, params=self.posterior_params, **prediction_params)
        return posterior_predictive(_rng_key)

    def get_samples(self):
        """
        Getter
        :return: samples from the posterior distribution
        """
        return self.posterior_samples

    def get_params(self):
        """
        Getter
        :return: parameters of the posterior distribution if applicable, None otherwise
        """
        return self.posterior_params
