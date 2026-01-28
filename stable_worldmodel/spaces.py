"""Extended Gymnasium spaces with state tracking and constraint support."""

import time
from typing import Any, Callable, Generator, Iterable

from gymnasium import spaces
from loguru import logger as logging

import stable_worldmodel as swm


class Discrete(spaces.Discrete):
    """Extended discrete space with state tracking and constraint support."""

    def __init__(
        self,
        n: int,
        init_value: int | None = None,
        constrain_fn: Callable[[int], bool] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Discrete space with state tracking.

        Args:
            n: Number of elements in the space.
            init_value: Initial value for the space.
            constrain_fn: Optional predicate function for rejection sampling.
            **kwargs: Additional arguments passed to gymnasium.spaces.Discrete.
        """
        super().__init__(n, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self) -> int | None:
        """The initial value of the space."""
        return self._init_value

    @property
    def value(self) -> int | None:
        """The current value of the space."""
        return self._value

    def reset(self) -> None:
        """Reset the space value to its initial value."""
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        """Check if value is valid and satisfies constraints.

        Args:
            x: Value to check.

        Returns:
            True if value is valid and satisfies constraints, False otherwise.
        """
        return super().contains(x) and self.constrain_fn(x)

    def check(self) -> bool:
        """Validate the current space value.

        Returns:
            True if the current value is valid, False otherwise.
        """
        if not self.constrain_fn(self.value):
            logging.warning(f"Discrete: value {self.value} does not satisfy constrain_fn")
            return False
        return super().contains(self.value)

    def sample(
        self,
        mask: Any | None = None,
        max_tries: int = 1000,
        warn_after_s: float | None = 5.0,
        set_value: bool = True,
        **kwargs: Any,
    ) -> int:
        """Sample a random value using rejection sampling for constraints.

        Args:
            mask: Optional mask for sampling.
            max_tries: Maximum number of rejection sampling attempts.
            warn_after_s: Log a warning if sampling takes longer than this.
            set_value: Whether to update the current value with the sample.
            **kwargs: Additional arguments passed to gymnasium sample.

        Returns:
            A randomly sampled value satisfying constraints.

        Raises:
            RuntimeError: If no valid sample is found within max_tries.
        """
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(mask=mask)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")

    def set_init_value(self, value: int) -> None:
        """Set the initial value of the Discrete space.

        Args:
            value: The new initial value.

        Raises:
            ValueError: If the value is not valid for this space.
        """
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Discrete space")
        self._init_value = value

    def set_value(self, value: int) -> None:
        """Set the current value of the Discrete space.

        Args:
            value: The new current value.

        Raises:
            ValueError: If the value is not valid for this space.
        """
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Discrete space")
        self._value = value


class MultiDiscrete(spaces.MultiDiscrete):
    """Extended multi-discrete space with state tracking and constraint support."""

    def __init__(
        self,
        nvec: Any,
        init_value: Any | None = None,
        constrain_fn: Callable[[Any], bool] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a MultiDiscrete space with state tracking.

        Args:
            nvec: Vector of number of elements for each dimension.
            init_value: Initial values for the space.
            constrain_fn: Optional predicate function for rejection sampling.
            **kwargs: Additional arguments passed to gymnasium.spaces.MultiDiscrete.
        """
        super().__init__(nvec, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self) -> Any | None:
        """The initial values of the space."""
        return self._init_value

    @property
    def value(self) -> Any | None:
        """The current values of the space."""
        return self._value

    def reset(self) -> None:
        """Reset the space values to their initial values."""
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        """Check if values are valid and satisfy constraints.

        Args:
            x: Values to check.

        Returns:
            True if values are valid and satisfy constraints, False otherwise.
        """
        return super().contains(x) and self.constrain_fn(x)

    def check(self) -> bool:
        """Validate the current space values.

        Returns:
            True if current values are valid, False otherwise.
        """
        if not self.constrain_fn(self.value):
            logging.warning(f"MultiDiscrete: value {self.value} does not satisfy constrain_fn")
            return False
        return super().contains(self.value)

    def sample(
        self,
        mask: Any | None = None,
        max_tries: int = 1000,
        warn_after_s: float | None = 5.0,
        set_value: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Sample random values using rejection sampling for constraints.

        Args:
            mask: Optional mask for sampling.
            max_tries: Maximum number of rejection sampling attempts.
            warn_after_s: Log a warning if sampling takes longer than this.
            set_value: Whether to update the current value with the sample.
            **kwargs: Additional arguments passed to gymnasium sample.

        Returns:
            Randomly sampled values satisfying constraints.

        Raises:
            RuntimeError: If no valid sample is found within max_tries.
        """
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(mask=mask)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")

    def set_init_value(self, value: Any) -> None:
        """Set the initial values of the MultiDiscrete space.

        Args:
            value: The new initial values.

        Raises:
            ValueError: If values are not valid for this space.
        """
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the MultiDiscrete space")
        self._init_value = value

    def set_value(self, value: Any) -> None:
        """Set the current values of the MultiDiscrete space.

        Args:
            value: The new current values.

        Raises:
            ValueError: If values are not valid for this space.
        """
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the MultiDiscrete space")
        self._value = value


class Box(spaces.Box):
    """Extended continuous box space with state tracking and constraint support."""

    def __init__(
        self,
        low: Any,
        high: Any,
        shape: Iterable[int] | None = None,
        init_value: Any | None = None,
        constrain_fn: Callable[[Any], bool] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Box space with state tracking.

        Args:
            low: Lower bounds of the space.
            high: Upper bounds of the space.
            shape: Optional shape of the space.
            init_value: Initial values for the space.
            constrain_fn: Optional predicate function for rejection sampling.
            **kwargs: Additional arguments passed to gymnasium.spaces.Box.
        """
        super().__init__(low, high, shape, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._init_value = init_value
        self._value = init_value

    @property
    def init_value(self) -> Any | None:
        """The initial value of the space."""
        return self._init_value

    @property
    def value(self) -> Any | None:
        """The current value of the space."""
        return self._value

    def reset(self) -> None:
        """Reset the space value to its initial value."""
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        """Check if value is valid and satisfies constraints.

        Args:
            x: Value to check.

        Returns:
            True if value is valid and satisfies constraints, False otherwise.
        """
        return super().contains(x) and self.constrain_fn(x)

    def check(self) -> bool:
        """Validate the current space value.

        Returns:
            True if the current value is valid, False otherwise.
        """
        if not self.constrain_fn(self.value):
            logging.warning(f"Box: value {self.value} does not satisfy constrain_fn")
            return False
        return self.contains(self.value)

    def sample(
        self,
        mask: Any | None = None,
        max_tries: int = 1000,
        warn_after_s: float | None = 5.0,
        set_value: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Sample a random value using rejection sampling for constraints.

        Args:
            mask: Optional mask for sampling.
            max_tries: Maximum number of rejection sampling attempts.
            warn_after_s: Log a warning if sampling takes longer than this.
            set_value: Whether to update the current value with the sample.
            **kwargs: Additional arguments passed to gymnasium sample.

        Returns:
            A randomly sampled value satisfying constraints.

        Raises:
            RuntimeError: If no valid sample is found within max_tries.
        """
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(mask=mask)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")

    def set_init_value(self, value: Any) -> None:
        """Set the initial value of the Box space.

        Args:
            value: The new initial value.

        Raises:
            ValueError: If value is not valid for this space.
        """
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Box space")
        self._init_value = value

    def set_value(self, value: Any) -> None:
        """Set the current value of the Box space.

        Args:
            value: The new current value.

        Raises:
            ValueError: If value is not valid for this space.
        """
        if not self.contains(value):
            raise ValueError(f"Value {value} is not contained in the Box space")
        self._value = value


class RGBBox(Box):
    """Specialized box space for RGB image data."""

    def __init__(
        self,
        shape: Iterable[int] = (3,),
        init_value: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an RGBBox space.

        Args:
            shape: Shape of the image (must have a channel of size 3).
            init_value: Initial value for the space.
            **kwargs: Additional arguments passed to Box.

        Raises:
            ValueError: If shape does not have a channel of size 3.
        """
        if not any(dim == 3 for dim in shape):
            raise ValueError("shape must have a channel of size 3")

        super().__init__(
            low=0,
            high=255,
            shape=shape,
            dtype="uint8",
            init_value=init_value,
            **kwargs,
        )


class Dict(spaces.Dict):
    """Extended dictionary space with ordered sampling and nested support."""

    def __init__(
        self,
        spaces_dict: dict[Any, spaces.Space] | None = None,
        init_value: dict | None = None,
        constrain_fn: Callable[[dict], bool] | None = None,
        sampling_order: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Dict space with state tracking and sampling order.

        Args:
            spaces_dict: Dictionary mapping keys to Gymnasium spaces.
            init_value: Initial values for the contained spaces.
            constrain_fn: Optional predicate function for rejection sampling.
            sampling_order: Explicit order for sampling keys.
            **kwargs: Additional arguments passed to gymnasium.spaces.Dict.
        """
        super().__init__(spaces_dict, **kwargs)
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
            self._sampling_order = list(sampling_order) + [str(k) for k in missing_keys]
        else:
            self._sampling_order = sampling_order

        if not all(key in self.spaces for key in self._sampling_order):
            missing = set(self._sampling_order) - set(self.spaces.keys())
            raise ValueError(f"sampling_order contains keys not in spaces: {missing}")

    @property
    def init_value(self) -> dict:
        """Initial values for all contained spaces."""
        init_val: dict = {}

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
    def value(self) -> dict:
        """Current values of all contained spaces.

        Returns:
            Dictionary of current values.

        Raises:
            ValueError: If a contained space does not have a value property.
        """
        val: dict = {}
        for k, v in self.spaces.items():
            if hasattr(v, "value"):
                val[k] = v.value
            else:
                raise ValueError(f"Space {k} of type {type(v)} does not have value property")
        return val

    def _get_sampling_order(self, parts: tuple[str, ...] | None = None) -> Generator[str, None, None]:
        """Yield dotted paths for nested Dict space respecting sampling order.

        Args:
            parts: Tuple of parent keys for recursion.

        Yields:
            Dotted paths to contained spaces.
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
                if hasattr(subspace, "_get_sampling_order"):
                    yield from subspace._get_sampling_order(path)
                else:
                    # Fallback for standard gymnasium Dict spaces
                    for subkey in subspace.spaces.keys():
                        yield ".".join(path + (str(subkey),))

    @property
    def sampling_order(self) -> list[str]:
        """Set of dotted paths for all variables in sampling order."""
        return list(self._get_sampling_order())

    def reset(self) -> None:
        """Reset all contained spaces to their initial values."""
        for v in self.spaces.values():
            if hasattr(v, "reset"):
                v.reset()
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        """Check if value is a valid member of this space.

        Args:
            x: Value to check.

        Returns:
            True if value is valid, False otherwise.
        """
        if not isinstance(x, dict):
            return False

        for key in self.spaces.keys():
            if key not in x:
                return False

            if not self.spaces[key].contains(x[key]):
                return False

        if not self.constrain_fn(x):
            return False

        return True

    def check(self, debug: bool = False) -> bool:
        """Validate all contained spaces' current values.

        Args:
            debug: Whether to log warnings for failed checks.

        Returns:
            True if all checks pass, False otherwise.
        """
        for k, v in self.spaces.items():
            if hasattr(v, "check"):
                if not v.check():
                    if debug:
                        logging.warning(f"Dict: space {k} failed check()")
                    return False
        return True

    def names(self) -> list[str]:
        """Return all space keys including nested ones."""

        def _key_generator(d: dict[Any, spaces.Space], parent_key: str = "") -> Generator[str, None, None]:
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, spaces.Dict):
                    yield from _key_generator(v.spaces, new_key)
                else:
                    yield new_key

        return list(_key_generator(self.spaces))

    def sample(
        self,
        mask: Any | None = None,
        max_tries: int = 1000,
        warn_after_s: float | None = 5.0,
        set_value: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Sample a random element from the Dict space.

        Args:
            mask: Optional mask for sampling.
            max_tries: Maximum number of rejection sampling attempts.
            warn_after_s: Log a warning if sampling takes longer than this.
            set_value: Whether to update the current value with the sample.
            **kwargs: Additional arguments passed to sample.

        Returns:
            A randomly sampled dictionary satisfying constraints.

        Raises:
            RuntimeError: If no valid sample is found within max_tries.
        """
        start = time.time()
        for i in range(max_tries):
            sample: dict = {}

            for k in self._sampling_order:
                # Need to handle mask if provided
                sub_mask = mask[k] if isinstance(mask, dict) and k in mask else None
                sample[k] = self.spaces[k].sample(mask=sub_mask, set_value=set_value, **kwargs)

            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample

            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling is taking a while...")

        raise RuntimeError(f"constrain_fn not satisfied after {max_tries} draws")

    def update(self, keys: Iterable[str]) -> None:
        """Update specific keys in the Dict space by resampling them.

        Args:
            keys: Keys to resample.

        Raises:
            ValueError: If a key is not found in the Dict space.
        """
        keys_set = set(keys)
        order = self.sampling_order

        if len(keys_set) == 1 and "all" in keys_set:
            self.sample()
        else:
            for v in filter(keys_set.__contains__, order):
                try:
                    var_path = v.split(".")
                    swm.utils.get_in(self, var_path).sample()

                except (KeyError, TypeError):
                    raise ValueError(f"Key {v} not found in Dict space")

        assert self.check(debug=True), "Values must be within space!"

    def set_init_value(self, variations_values: dict) -> None:
        """Set initial values for specific keys in the Dict space.

        Args:
            variations_values: Mapping of keys to new initial values.

        Raises:
            ValueError: If a key is not found in the Dict space.
        """
        for k, v in variations_values.items():
            try:
                var_path = k.split(".")
                space = swm.utils.get_in(self, var_path)
                assert space.contains(v), (
                    f"Value {v} for key {k} is not contained in the space"
                )
                space.set_init_value(v)

            except (KeyError, TypeError):
                raise ValueError(f"Key {k} not found in Dict space")

    def set_value(self, variations_values: dict) -> None:
        """Set current values for specific keys in the Dict space.

        Args:
            variations_values: Mapping of keys to new current values.

        Raises:
            ValueError: If a key is not found in the Dict space.
        """
        for k, v in variations_values.items():
            try:
                var_path = k.split(".")
                space = swm.utils.get_in(self, var_path)
                assert space.contains(v), (
                    f"Value {v} for key {k} is not contained in the space"
                )
                space.set_value(v)

            except (KeyError, TypeError):
                raise ValueError(f"Key {k} not found in Dict space")

    def to_str(self) -> str:
        """Return a string representation of the space structure.

        Returns:
            A formatted string describing the space.
        """
        def _tree(d: dict[Any, spaces.Space], indent: int = 0) -> str:
            lines = []
            for k, v in d.items():
                if isinstance(v, (dict | self.__class__ | spaces.Dict)):
                    lines.append("    " * indent + f"{k}:")
                    # handle spaces.Dict which has .spaces
                    sub_dict = v.spaces if isinstance(v, spaces.Dict) else v
                    lines.append(_tree(sub_dict, indent + 1))
                else:
                    lines.append("    " * indent + f"{k}: {v}")
            return "\n".join(lines)

        return _tree(self.spaces)
