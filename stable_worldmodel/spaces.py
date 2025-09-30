from sys import prefix
import time

from gymnasium import spaces
from loguru import logger as logging

import stable_worldmodel as swm


class Discrete(spaces.Discrete):
    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
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
        if not self.constrain_fn(self.value):
            logging.warning(
                f"Discrete: value {self.value} does not satisfy constrain_fn"
            )
            return False
        return super().contains(self.value)

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


class MultiDiscrete(spaces.MultiDiscrete):
    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
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
        if not self.constrain_fn(self.value):
            logging.warning(
                f"MultiDiscrete: value {self.value} does not satisfy constrain_fn"
            )
            return False
        return super().contains(self.value)

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
        if not self.constrain_fn(self.value):
            logging.warning(f"Box: value {self.value} does not satisfy constrain_fn")
            return False
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
        self._init_value = init_value
        self._value = self.init_value

        # add missing keys
        if sampling_order is None:
            self._sampling_order = list(self.spaces.keys())
        elif len(sampling_order) != len(self.spaces):
            missing_keys = set(self.spaces.keys()).difference(set(sampling_order))
            logging.warning(
                f"Dict sampling_order is missing keys {missing_keys}, adding them at the end of the sampling order"
            )
            self._sampling_order = list(sampling_order) + list(missing_keys)
        else:
            self._sampling_order = sampling_order

        assert all(key in self.spaces for key in self._sampling_order), (
            "All keys in sampling_order must be in spaces"
        )

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

    # def _get_sampling_order(self, parts=()):
    #     for key in self._sampling_order:
    #         yield ".".join(parts + (key,))

    #         if isinstance(self.spaces[key], spaces.Dict):
    #             yield from self.spaces[key]._get_sampling_order(parts + (key,))
    #         # else:
    #         #     yield ".".join(parts + (key,))

    def _get_sampling_order(self, parts=None):
        """
        Yield dotted paths for this (possibly nested) Dict space, honoring
        self._sampling_order when available, and recursing into nested Dicts.
        """
        if parts is None:
            parts = ()

        # Prefer an explicit sampling order; otherwise preserve insertion order.
        keys = getattr(self, "_sampling_order", None) or self.spaces.keys()

        for key in keys:
            # Skip if the key isn't in the mapping (defensive against stale order lists).
            if key not in self.spaces:
                continue

            key_str = str(key)  # ensure joinable
            path = parts + (key_str,)
            yield ".".join(path)

            subspace = self.spaces[key]
            if isinstance(subspace, spaces.Dict):
                # Recurse into nested Dict spaces
                yield from subspace._get_sampling_order(path)

    @property
    def sampling_order(self):
        return set(self._get_sampling_order())

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

    def check(self, debug=False):
        for k, v in self.spaces.items():
            if hasattr(v, "check"):
                if not v.check():
                    if debug:
                        logging.warning(f"Dict: space {k} failed check()")
                    return False
            else:
                if not v.contains(v.value):
                    if debug:
                        logging.warning(f"Dict: space {k} failed contains(value)")
                    return False

        return True

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
            sample = {}

            for k in self._sampling_order:
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

    def update(self, keys):
        order = self.sampling_order
        for v in filter(keys.__contains__, order):
            try:
                var_path = v.split(".")
                swm.utils.get_in(self, var_path).sample()

            except (KeyError, TypeError):
                raise ValueError(f"Key {v} not found in Dict space")

        assert self.check(debug=True), "Values must be within space!"
