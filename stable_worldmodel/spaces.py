"""Extended Gymnasium spaces with state tracking and constraint support."""

import time

from gymnasium import spaces
from loguru import logger as logging

import stable_worldmodel as swm


class Discrete(spaces.Discrete):
    """Extended discrete space with state tracking and constraint support.

    This class extends ``gymnasium.spaces.Discrete`` to add state management
    and optional constraint validation. Unlike the standard discrete space,
    this version maintains a current value and supports rejection sampling
    via a custom constraint function.

    Attributes:
        init_value (int): The initial value for the space.
        value (int): The current value of the space.
        constrain_fn (callable): Optional function that returns True if a
            value satisfies custom constraints.

    Example:
        Create a discrete space that only accepts even numbers::

            space = Discrete(n=10, init_value=0, constrain_fn=lambda x: x % 2 == 0)
            value = space.sample()  # Samples even number and updates space.value
            space.reset()  # Resets space.value back to 0 (init_value)

    Note:
        The ``sample()`` method uses rejection sampling when a constraint
        function is provided, which may impact performance for difficult
        constraints.
    """

    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        """Initialize a Discrete space with state tracking.

        Args:
            *args: Positional arguments passed to gymnasium.spaces.Discrete.
            init_value (int, optional): Initial value for the space. Defaults to None.
            constrain_fn (callable, optional): Function that takes an int and returns
                True if the value satisfies custom constraints. Defaults to None.
            **kwargs: Keyword arguments passed to gymnasium.spaces.Discrete.
        """
        super().__init__(*args, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self):
        """int: The initial value of the space, returned by reset()."""
        return self._init_value

    @property
    def value(self):
        """int: The current value of the space."""
        return self._value

    def reset(self):
        """Reset the space value to its initial value.

        Sets the current value back to the init_value specified during
        initialization.
        """
        self._value = self.init_value

    def contains(self, x):
        """Check if value is valid and satisfies constraints.

        Args:
            x (int): The value to check.

        Returns:
            bool: True if x is within bounds and satisfies the constraint
                function, False otherwise.
        """
        return super().contains(x) and self.constrain_fn(x)

    def check(self):
        """Validate the current space value.

        Checks if the current value is within the space bounds and satisfies
        the constraint function. Logs a warning if the constraint fails.

        Returns:
            bool: True if the current value is valid, False otherwise.
        """
        if not self.constrain_fn(self.value):
            logging.warning(f"Discrete: value {self.value} does not satisfy constrain_fn")
            return False
        return super().contains(self.value)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample a random value using rejection sampling for constraints.

        Repeatedly samples values until one satisfies the constraint function
        or max_tries is reached. Optionally updates the space's current value.

        Args:
            *args: Positional arguments passed to gymnasium.spaces.Discrete.sample().
            max_tries (int, optional): Maximum number of sampling attempts before
                raising an error. Defaults to 1000.
            warn_after_s (float, optional): Time threshold in seconds after which
                to log a warning about slow sampling. Set to None to disable.
                Defaults to 5.0.
            set_value (bool, optional): Whether to update the space's current
                value with the sampled value. Defaults to True.
            **kwargs: Keyword arguments passed to gymnasium.spaces.Discrete.sample().

        Returns:
            int: A sampled value that satisfies the constraint function.

        Raises:
            RuntimeError: If no valid sample is found after max_tries attempts.
        """
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")


class MultiDiscrete(spaces.MultiDiscrete):
    """Extended multi-discrete space with state tracking and constraint support.

    This class extends ``gymnasium.spaces.MultiDiscrete`` to add state
    management and optional constraint validation. It represents multiple
    discrete variables with potentially different ranges (nvec), where each
    variable maintains its own value and can be constrained.

    Attributes:
        init_value (np.ndarray): The initial values for all discrete variables.
        value (np.ndarray): The current values of all discrete variables.
        constrain_fn (callable): Optional function that returns True if the
            entire value array satisfies custom constraints.

    Example:
        Create a multi-discrete space for game difficulty settings::

            import numpy as np

            space = MultiDiscrete(
                nvec=[5, 3, 10],  # [enemy_count, speed_level, spawn_rate]
                init_value=np.array([2, 1, 5]),
            )
            settings = space.sample()  # Random difficulty configuration
            space.reset()  # Resets to [2, 1, 5] (medium difficulty)

    Note:
        Constraints are applied to the entire array, not individual elements.
        Use a constraint function that validates the complete state.
    """

    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        """Initialize a MultiDiscrete space with state tracking.

        Args:
            *args: Positional arguments passed to gymnasium.spaces.MultiDiscrete.
            init_value (np.ndarray, optional): Initial values for the space.
                Must match the shape defined by nvec. Defaults to None.
            constrain_fn (callable, optional): Function that takes a numpy array
                and returns True if the values satisfy custom constraints.
                Defaults to None.
            **kwargs: Keyword arguments passed to gymnasium.spaces.MultiDiscrete.
        """
        super().__init__(*args, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self):
        """np.ndarray: The initial values of the space, returned by reset()."""
        return self._init_value

    @property
    def value(self):
        """np.ndarray: The current values of the space."""
        return self._value

    def reset(self):
        """Reset the space values to their initial values.

        Sets the current values back to the init_value specified during
        initialization.
        """
        self._value = self.init_value

    def contains(self, x):
        """Check if values are valid and satisfy constraints.

        Args:
            x (np.ndarray): The array of values to check.

        Returns:
            bool: True if x is within bounds for all elements and satisfies
                the constraint function, False otherwise.
        """
        return super().contains(x) and self.constrain_fn(x)

    def check(self):
        """Validate the current space values.

        Checks if the current values are within the space bounds and satisfy
        the constraint function. Logs a warning if the constraint fails.

        Returns:
            bool: True if the current values are valid, False otherwise.
        """
        if not self.constrain_fn(self.value):
            logging.warning(f"MultiDiscrete: value {self.value} does not satisfy constrain_fn")
            return False
        return super().contains(self.value)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample random values using rejection sampling for constraints.

        Repeatedly samples value arrays until one satisfies the constraint
        function or max_tries is reached. Optionally updates the space's
        current values.

        Args:
            *args: Positional arguments passed to gymnasium.spaces.MultiDiscrete.sample().
            max_tries (int, optional): Maximum number of sampling attempts before
                raising an error. Defaults to 1000.
            warn_after_s (float, optional): Time threshold in seconds after which
                to log a warning about slow sampling. Set to None to disable.
                Defaults to 5.0.
            set_value (bool, optional): Whether to update the space's current
                values with the sampled values. Defaults to True.
            **kwargs: Keyword arguments passed to gymnasium.spaces.MultiDiscrete.sample().

        Returns:
            np.ndarray: A sampled array that satisfies the constraint function.

        Raises:
            RuntimeError: If no valid sample is found after max_tries attempts.
        """
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")


class Box(spaces.Box):
    """Extended continuous box space with state tracking and constraint support.

    This class extends ``gymnasium.spaces.Box`` to add state management and
    optional constraint validation. It represents bounded continuous values
    with configurable shape, dtype, and custom constraints.

    Attributes:
        init_value (np.ndarray): The initial value for the space.
        value (np.ndarray): The current value of the space.
        constrain_fn (callable): Optional function that returns True if a
            value satisfies custom constraints beyond the box boundaries.

    Example:
        Create a 2D position space constrained to a circle::

            import numpy as np


            def in_circle(pos):
                return np.linalg.norm(pos) <= 1.0


            space = Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                init_value=np.array([0.0, 0.0]),
                constrain_fn=in_circle,
            )
            position = space.sample()  # Only samples within unit circle

    Note:
        The constraint function enables complex geometric or relational
        constraints beyond simple box boundaries.
    """

    def __init__(self, *args, init_value=None, constrain_fn=None, **kwargs):
        """Initialize a Box space with state tracking.

        Args:
            *args: Positional arguments passed to gymnasium.spaces.Box.
            init_value (np.ndarray, optional): Initial value for the space.
                Must match the shape and dtype of the box. Defaults to None.
            constrain_fn (callable, optional): Function that takes a numpy array
                and returns True if the value satisfies custom constraints beyond
                the box boundaries. Defaults to None.
            **kwargs: Keyword arguments passed to gymnasium.spaces.Box.
        """
        super().__init__(*args, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._init_value = init_value
        self._value = init_value

    @property
    def init_value(self):
        """np.ndarray: The initial value of the space, returned by reset()."""
        return self._init_value

    @property
    def value(self):
        """np.ndarray: The current value of the space."""
        return self._value

    def reset(self):
        """Reset the space value to its initial value.

        Sets the current value back to the init_value specified during
        initialization.
        """
        self._value = self.init_value

    def contains(self, x):
        """Check if value is valid and satisfies constraints.

        Args:
            x (np.ndarray): The value to check.

        Returns:
            bool: True if x is within box bounds and satisfies the constraint
                function, False otherwise.
        """
        return super().contains(x) and self.constrain_fn(x)

    def check(self):
        """Validate the current space value.

        Checks if the current value is within the box bounds and satisfies
        the constraint function. Logs a warning if the constraint fails.

        Returns:
            bool: True if the current value is valid, False otherwise.
        """
        if not self.constrain_fn(self.value):
            logging.warning(f"Box: value {self.value} does not satisfy constrain_fn")
            return False
        return self.contains(self.value)

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample a random value using rejection sampling for constraints.

        Repeatedly samples values until one satisfies the constraint function
        or max_tries is reached. Optionally updates the space's current value.

        Args:
            *args: Positional arguments passed to gymnasium.spaces.Box.sample().
            max_tries (int, optional): Maximum number of sampling attempts before
                raising an error. Defaults to 1000.
            warn_after_s (float, optional): Time threshold in seconds after which
                to log a warning about slow sampling. Set to None to disable.
                Defaults to 5.0.
            set_value (bool, optional): Whether to update the space's current
                value with the sampled value. Defaults to True.
            **kwargs: Keyword arguments passed to gymnasium.spaces.Box.sample().

        Returns:
            np.ndarray: A sampled array that satisfies the constraint function.

        Raises:
            RuntimeError: If no valid sample is found after max_tries attempts.
        """
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(*args, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling: rejection sampling is taking a while...")
        raise RuntimeError(f"rejection sampling: predicate not satisfied after {max_tries} draws")


class RGBBox(Box):
    """Specialized box space for RGB image data with automatic constraints.

    This class extends ``Box`` to provide a convenient space for RGB images,
    automatically enforcing uint8 dtype and [0, 255] value ranges. It validates
    that the shape includes exactly 3 channels for RGB data.

    Args:
        shape (tuple): Shape of the image. Must include a dimension of size 3
            for the RGB channels. Common formats: (H, W, 3) or (3, H, W).
        init_value (np.ndarray, optional): Initial RGB image. Must match shape
            and be uint8 dtype.
        *args: Additional positional arguments passed to Box.
        **kwargs: Additional keyword arguments passed to Box.

    Attributes:
        init_value (np.ndarray): The initial RGB image.
        value (np.ndarray): The current RGB image.

    Example:
        Create a space for 64x64 RGB images::

            import numpy as np

            space = RGBBox(
                shape=(64, 64, 3), init_value=np.zeros((64, 64, 3), dtype=np.uint8)
            )
            image = space.sample()  # Random RGB image
            space.reset()  # Returns to black image

    Raises:
        AssertionError: If shape does not contain a dimension of size 3.

    Note:
        This space is useful for vision-based environments where images
        need to be sampled or tracked as part of environment configuration.
        The low, high, and dtype parameters are automatically set and cannot
        be overridden.
    """

    def __init__(self, shape=(3,), *args, init_value=None, **kwargs):
        if not any(dim == 3 for dim in shape):
            raise ValueError("shape must have a channel of size 3")

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
    """Extended dictionary space with ordered sampling and nested support.

    This class extends ``gymnasium.spaces.Dict`` to add state management,
    constraint validation, and explicit sampling order control. It composes
    multiple spaces into a hierarchical structure where dependencies between
    variables can be handled through ordered sampling.

    Args:
        *args: Positional arguments passed to gymnasium.spaces.Dict.
        init_value (dict, optional): Initial values for the space. If None,
            derived from init_value of contained spaces.
        constrain_fn (callable, optional): Function that returns True if the
            complete dictionary satisfies custom constraints.
        sampling_order (list, optional): Explicit order for sampling keys.
            If None, uses insertion order. Missing keys are appended.
        **kwargs: Additional keyword arguments passed to Dict.

    Attributes:
        init_value (dict): Initial values for all contained spaces.
        value (dict): Current values of all contained spaces.
        constrain_fn (callable): Constraint validation function.
        sampling_order (set): Set of dotted paths for all variables in order.

    Example:
        Create a nested space with sampling order dependencies::

            from stable_worldmodel import spaces
            import numpy as np

            config = spaces.Dict(
                {
                    "difficulty": spaces.Discrete(n=3, init_value=0),
                    "world": spaces.Dict(
                        {
                            "width": spaces.Discrete(n=100, init_value=50),
                            "height": spaces.Discrete(n=100, init_value=50),
                        }
                    ),
                    "player_pos": spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([99, 99]),
                        init_value=np.array([25, 25]),
                    ),
                },
                sampling_order=["difficulty", "world", "player_pos"],
            )

            # Sample respects order
            state = config.sample()

    Note:
        Sampling order is crucial when variables have dependencies. For
        example, sample world size before sampling positions within it.
        Nested Dict spaces recursively apply their own sampling orders.

        **Accessing values in constraint functions**: When implementing
        ``constrain_fn`` for Dict spaces, always use ``self.value['key']['key2']``
        instead of ``self['key']['key2'].value``. The ``.value`` property
        recursively builds the complete value dictionary from the top level down,
        ensuring all nested values are up-to-date and correctly structured. Direct
        subspace access with ``.value`` only retrieves that specific subspace's
        value without the full context.

        Note that direct subspace access (e.g., ``self['key'].value``) is perfectly
        fine for regular operations outside of constraint functions, such as reading
        individual subspace values or debugging. The recommendation to use top-level
        ``.value`` applies specifically to constraint functions where you need the
        complete, consistent state of all nested spaces.

        Example of proper constraint function usage::

            # Example: In a class with Dict space attribute
            class Environment:
                def __init__(self):
                    self.config_space = spaces.Dict({...})

                def validate_config(self):
                    # ✓ CORRECT: Access via .value at top level
                    values = self.config_space.value
                    return values["player_pos"][0] < values["world"]["width"]

                def validate_wrong(self):
                    # ✗ AVOID: Direct subspace access
                    return (
                        self.config_space["player_pos"].value[0]
                        < self.config_space["world"]["width"].value
                    )
    """

    def __init__(self, *args, init_value=None, constrain_fn=None, sampling_order=None, **kwargs):
        """Initialize a Dict space with state tracking and sampling order.

        Args:
            *args: Positional arguments passed to gymnasium.spaces.Dict.
            init_value (dict, optional): Initial values for the space. If None,
                derived from init_value of contained spaces. Defaults to None.
            constrain_fn (callable, optional): Function that takes a dict and
                returns True if the complete dictionary satisfies custom constraints.
                Defaults to None.
            sampling_order (list, optional): Explicit order for sampling keys.
                If None, uses insertion order. Missing keys are appended with warning.
                Defaults to None.
            **kwargs: Keyword arguments passed to gymnasium.spaces.Dict.

        Raises:
            ValueError: If sampling_order contains keys not present in spaces.
        """
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

        if not all(key in self.spaces for key in self._sampling_order):
            missing = set(self._sampling_order) - set(self.spaces.keys())
            raise ValueError(f"sampling_order contains keys not in spaces: {missing}")

    @property
    def init_value(self):
        """dict: Initial values for all contained spaces.

        Constructs initial value dictionary from contained spaces' init_value
        properties. Falls back to sampling if a space lacks init_value.

        Returns:
            dict: Dictionary mapping space keys to their initial values.
        """
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
        """dict: Current values of all contained spaces.

        Constructs value dictionary from contained spaces' value properties.

        Returns:
            dict: Dictionary mapping space keys to their current values.

        Raises:
            ValueError: If a contained space does not have a value property.
        """
        val = {}
        for k, v in self.spaces.items():
            if hasattr(v, "value"):
                val[k] = v.value
            else:
                raise ValueError(f"Space {k} of type {type(v)} does not have value property")
        return val

    def _get_sampling_order(self, parts=None):
        """Yield dotted paths for nested Dict space respecting sampling order.

        Recursively generates dotted-path strings for all variables in this
        Dict space and any nested Dict spaces, honoring the explicit
        sampling order when available.

        Args:
            parts (tuple, optional): Parent path components for recursion.
                Defaults to empty tuple.

        Yields:
            str: Dotted path strings like 'parent.child.key' for each variable.
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
        """set: Set of dotted paths for all variables in sampling order.

        Returns:
            set: Set of strings representing dotted paths (e.g., 'parent.child.key')
                for all variables including nested Dict spaces.
        """
        return list(self._get_sampling_order())

    def reset(self):
        """Reset all contained spaces to their initial values.

        Calls reset() on all contained spaces that have a reset method,
        then sets this space's value to init_value.
        """
        for v in self.spaces.values():
            if hasattr(v, "reset"):
                v.reset()
        self._value = self.init_value

    def contains(self, x) -> bool:
        """Check if value is a valid member of this space.

        Validates that x is a dictionary containing all required keys with
        values that satisfy each subspace's constraints and the overall
        constraint function.

        Args:
            x: The value to check.

        Returns:
            bool: True if x is a valid dict with all keys present, all values
                within subspace bounds, and satisfies the constraint function.
                False otherwise.
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

    def check(self, debug=False):
        """Validate all contained spaces' current values.

        Checks each contained space using its check() method if available,
        or falls back to contains(value). Optionally logs warnings for
        failed checks.

        Args:
            debug (bool, optional): If True, logs warnings for spaces that
                fail validation. Defaults to False.

        Returns:
            bool: True if all contained spaces have valid values, False otherwise.
        """
        for k, v in self.spaces.items():
            if hasattr(v, "check"):
                if not v.check():
                    if debug:
                        logging.warning(f"Dict: space {k} failed check()")
                    return False
        return True

    def names(self):
        """Return all space keys including nested ones.

        Returns:
            list: A list of all keys in the Dict space, with nested keys using dot notation.
                For example, a nested dict with key "a" containing subspace "b" would produce "a.b".
        """

        def _key_generator(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, spaces.Dict):
                    yield from _key_generator(v.spaces, new_key)
                else:
                    yield new_key

        return list(_key_generator(self.spaces))

    def sample(self, *args, max_tries=1000, warn_after_s=5.0, set_value=True, **kwargs):
        """Sample a random element from the Dict space.

        Samples each subspace in the sampling order and ensures the result satisfies
        any constraint functions. Uses rejection sampling if constraints are present.

        Args:
            *args: Positional arguments passed to each subspace's sample method.
            max_tries (int, optional): Maximum number of rejection sampling attempts.
                Defaults to 1000.
            warn_after_s (float, optional): Issue a warning if sampling takes longer than
                this many seconds. Set to None to disable warnings. Defaults to 5.0.
            set_value (bool, optional): Whether to set the internal value to the sampled value.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to each subspace's sample method.

        Returns:
            dict: A dictionary with keys matching the space definition and values sampled
                from their respective subspaces.

        Raises:
            RuntimeError: If a valid sample is not found within max_tries attempts.
        """
        start = time.time()
        for i in range(max_tries):
            sample = {}

            for k in self._sampling_order:
                sample[k] = self.spaces[k].sample(*args, **kwargs, set_value=set_value)

            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample

            if warn_after_s is not None and (time.time() - start) > warn_after_s:
                logging.warning("rejection sampling is taking a while...")

        raise RuntimeError(f"constrain_fn not satisfied after {max_tries} draws")

    def update(self, keys):
        """Update specific keys in the Dict space by resampling them.

        Samples new values for the specified keys while maintaining the sampling order.
        Uses dot notation for nested keys (e.g., "a.b" for nested dict).

        Args:
            keys (container): A container (list, set, etc.) of key names to resample.
                Keys should use dot notation for nested spaces.

        Raises:
            ValueError: If a specified key is not found in the Dict space.
            AssertionError: If the updated values violate the space constraints.
        """

        keys = set(keys)
        order = self.sampling_order

        if len(keys) == 1 and "all" in keys:
            self.sample()
        else:
            for v in filter(keys.__contains__, order):
                try:
                    var_path = v.split(".")
                    swm.utils.get_in(self, var_path).sample()

                except (KeyError, TypeError):
                    raise ValueError(f"Key {v} not found in Dict space")

        assert self.check(debug=True), "Values must be within space!"

    def to_str(self):
        def _tree(d, indent=0):
            lines = []
            for k, v in d.items():
                if isinstance(v, (dict | self.__class__ | spaces.Dict)):
                    lines.append("    " * indent + f"{k}:")
                    lines.append(_tree(v, indent + 1))
                else:
                    lines.append("    " * indent + f"{k}: {v}")
            return "\n".join(lines)

        return _tree(self.spaces)
