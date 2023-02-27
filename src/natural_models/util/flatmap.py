import itertools


def flatmap(func, *iterable):
    return list(itertools.chain.from_iterable(map(func, *iterable)))
