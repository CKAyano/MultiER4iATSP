from functools import lru_cache, wraps
import numpy as np


def np_cache(function):
    @lru_cache(maxsize=1000)
    def cached_wrapper(self, hashable_array):
        array = np.array(hashable_array)
        return function(self, array)

    @wraps(function)
    def wrapper(self, array):
        return cached_wrapper(self, tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
