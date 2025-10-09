Space
=====

This module provides custom Gymnasium space classes that extend the standard
Gymnasium spaces with additional functionality for managing environment variations
and initial conditions. Unlike traditional Gymnasium spaces which primarily define
boundaries for action and observation spaces in RL environments, these extended
spaces are designed to:

1. **Track current state** – Each space maintains a current value in addition to
   defining valid boundaries.
2. **Support initial values** – Spaces can be initialized with specific values that
   represent the starting state for environment episodes.
3. **Enable constraint functions** – Optional user-defined predicates for rejection
   sampling ensure sampled values satisfy custom conditions.
4. **Provide ordered sampling** – ``Dict`` spaces support explicit sampling order for
   handling dependencies between variables.

These spaces are particularly useful for defining procedurally generated environments
or environments with configurable parameters, where each space represents a variable
aspect of the environment that can be sampled, reset, and validated.

Space types
-----------
.. currentmodule:: stable_worldmodel.spaces
.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: autosummary/class.rst

   Discrete
   MultiDiscrete
   Box
   RGBBox
   Dict

Typical Usage
-------------

Create a constrained discrete space that only samples even numbers::

    from stable_worldmodel import spaces

    even_space = spaces.Discrete(
        n=10, init_value=0, constrain_fn=lambda x: x % 2 == 0
    )

    # Sample valid even numbers
    value = even_space.sample()
    print(f"Sampled: {value}, Current: {even_space.value}")

    # Reset to initial value
    even_space.reset()

Create a nested dictionary space with sampling order::

    from stable_worldmodel import spaces
    import numpy as np

    env_config = spaces.Dict(
        {
            "difficulty": spaces.Discrete(n=3, init_value=0),
            "agent_pos": spaces.Box(
                low=np.array([0, 0]),
                high=np.array([10, 10]),
                init_value=np.array([5, 5]),
            ),
            "goal_pos": spaces.Box(
                low=np.array([0, 0]),
                high=np.array([10, 10]),
                init_value=np.array([9, 9]),
            ),
        },
        sampling_order=["difficulty", "goal_pos", "agent_pos"],
    )

    # Sample respects the specified order
    config = env_config.sample()

    # Access nested values
    print(f"Difficulty: {config['difficulty']}")

Create an RGB image space::

    from stable_worldmodel import spaces
    import numpy as np

    image_space = spaces.RGBBox(
        shape=(64, 64, 3), init_value=np.zeros((64, 64, 3), dtype=np.uint8)
    )

    # Sample random RGB image
    img = image_space.sample()

Notes
-----

These spaces extend ``gymnasium.spaces`` classes and maintain API compatibility
while adding state management features. The ``sample()`` method performs
rejection sampling when a ``constrain_fn`` is provided, which may impact
performance if constraints are difficult to satisfy.

All spaces support the following additional parameters:

* ``init_value`` – Initial value for the space (returned by ``reset``)
* ``constrain_fn`` – Optional callable returning ``bool`` for validation
* ``max_tries`` – Maximum rejection sampling attempts (default: ``1000``)
* ``warn_after_s`` – Warning threshold for slow sampling (default: ``5.0s``)

**Important**: When accessing values in constraint functions for nested ``Dict``
spaces, prefer using ``space.value['key']['key2']`` over
``space['key']['key2'].value``. The ``.value`` property recursively constructs
the complete value dictionary top-down with all current information, while direct
space access only retrieves the individual subspace's value.

See Also
--------

- :mod:`stable_worldmodel.world` – World class that composes these spaces
- :mod:`stable_worldmodel.envs` – Environment implementations using these spaces
- :mod:`gymnasium.spaces` – Base Gymnasium spaces documentation

Todo
----

* Achieve full test coverage for the module
* Implement automatic type casting for ``init_value`` to match space dtype
* Add serialization/deserialization support for saving space states
* Optimize constraint checking for large spaces (e.g., ``MultiDiscrete``)
