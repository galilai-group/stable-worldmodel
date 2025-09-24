import time

from gymnasium import spaces
from loguru import logger as logging


class Discrete(spaces.Discrete):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Box(spaces.Box):
    
    def __init__(self, *args, constrain_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)

    def contains(self, x):
        return super().contains(x) and self.constrain_fn(x)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, **kwargs):
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s and i == 1:
                logging.warning("patch_sampling: rejection sampling is taking a while...")
        raise RuntimeError(
            f"patch_sampling: predicate not satisfied after {max_tries} draws"
        )
    
class Dict(spaces.Dict):
    def __init__(self, *args, constrain_fn=None, sampling_order=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self.sampling_order = sampling_order

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""

        if isinstance(x, dict):
            return all(
                key in self.spaces and self.spaces[key].contains(x[key])
                for key in x.keys()
            ) and self.constrain_fn(x)

        return False

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, **kwargs):
        start = time.time()
        for i in range(max_tries):


            if self.sampling_order is None:
                sample = {k: space.sample(*args, **kwargs) for k, space in self.spaces.items()}
            
            else:    
                # add missing keys
                if len(self.sampling_order) != len(self.spaces):
                    missing_keys = set(self.spaces.keys()).difference(set(self.sampling_order))
                    logging.warning(f"Dict sampling_order is missing keys {missing_keys}, adding them at the end of the sampling order")
                    self.sampling_order = list(self.sampling_order) + list(missing_keys)

                # sample in the specified order
                sample = {}
                for k in self.sampling_order:
                    sample[k] = self.spaces[k].sample(*args, **kwargs)

            if self.contains(sample):
                return sample
            
            if warn_after_s is not None and (time.time() - start) > warn_after_s and i == 1:
                logging.warning("rejection sampling is taking a while...")
        
        raise RuntimeError(
            f"constrain_fn not satisfied after {max_tries} draws"
        )