import time

from gymnasium import spaces
from loguru import logger as logging


class Discrete(spaces.Discrete):
    def __init__(self, *args, init_value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_value = init_value
        self._value = init_value

    @property
    def init_value(self):
        return self._init_value

    @property
    def value(self):
        return self._value

    def reset(self):
        self._value = self.init_value

    def check(self):
        return super().contains(self.value)

    def sample(self, *args, set_value=True, **kwargs):
        sample = super().sample(*args, **kwargs)
        if set_value:
            self._value = sample
        return sample


class Box(spaces.Box):
    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._init_value = init_value
        self._value = init_value

    @property
    def init_value(self):
        return self._init_value

    @property
    def value(self):
        return self._value

    def reset(self):
        self._value = self.init_value

    def contains(self, x):
        return super().contains(x) and self.constrain_fn(x)

    def check(self):
        return self.contains(self.value)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if (
                warn_after_s is not None
                and (time.time() - start) > warn_after_s
                and i == 1
            ):
                logging.warning(
                    "patch_sampling: rejection sampling is taking a while..."
                )
        raise RuntimeError(
            f"patch_sampling: predicate not satisfied after {max_tries} draws"
        )


class RGBBox(Box):
    """A Box space for RGB images or data, enforcing a channel of size 3 and values between 0 and 255."""

    def __init__(self, shape=(3,), *args, init_value=None, **kwargs):
        assert any([dim == 3 for dim in shape]), "shape must have a channel of size 3"

        super().__init__(
            low=0,
            high=255,
            shape=shape,
            dtype="uint8",
            init_value=init_value,
            *args,
            **kwargs,
        )


class Dict(spaces.Dict):
    def __init__(
        self, *args, init_value=None, constrain_fn=None, sampling_order=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self.sampling_order = sampling_order

        self._init_value = init_value
        self._value = self.init_value

    @property
    def init_value(self):
        init_val = {}

        for k, v in self.spaces.items():
            if hasattr(v, "init_value"):
                init_val[k] = v.init_value
            else:
                logging.warning(
                    f"Space {k} of type {type(v)} does not have init_value property, using default sample instead"
                )
                init_val[k] = v.sample()

        return init_val

    @property
    def value(self):
        val = {}
        for k, v in self.spaces.items():
            if hasattr(v, "value"):
                val[k] = v.value
            else:
                raise ValueError(
                    f"Space {k} of type {type(v)} does not have value property"
                )
        return val

    def reset(self):
        for v in self.spaces.values():
            if hasattr(v, "reset"):
                v.reset()
        self._value = self.init_value

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""

        if isinstance(x, dict):
            return all(
                key in self.spaces and self.spaces[key].contains(x[key])
                for key in x.keys()
            ) and self.constrain_fn(x)

        return False

    def check(self):
        return self.contains(self.value)

    def names(self):
        def _key_generator(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, spaces.Dict):
                    yield from _key_generator(v.spaces, new_key)
                else:
                    yield new_key

        return list(_key_generator(self.spaces))

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        start = time.time()
        for i in range(max_tries):
            if self.sampling_order is None:
                sample = {
                    k: space.sample(*args, **kwargs) for k, space in self.spaces.items()
                }

            else:
                # add missing keys
                if len(self.sampling_order) != len(self.spaces):
                    missing_keys = set(self.spaces.keys()).difference(
                        set(self.sampling_order)
                    )
                    logging.warning(
                        f"Dict sampling_order is missing keys {missing_keys}, adding them at the end of the sampling order"
                    )
                    self.sampling_order = list(self.sampling_order) + list(missing_keys)

                # sample in the specified order
                sample = {}
                for k in self.sampling_order:
                    sample[k] = self.spaces[k].sample(*args, **kwargs)

            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample

            if (
                warn_after_s is not None
                and (time.time() - start) > warn_after_s
                and i == 1
            ):
                logging.warning("rejection sampling is taking a while...")

        raise RuntimeError(f"constrain_fn not satisfied after {max_tries} draws")
