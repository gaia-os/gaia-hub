import abc
from abc import ABC


class GuideInterface(ABC):
    """
    A class representing a generic guide
    """

    def __init__(self, rng_key):
        self.rng_key = rng_key

    def __call__(self, params):
        self.call(params)

    @abc.abstractmethod
    def call(self, params):
        ...
