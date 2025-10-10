.. meta::
   :description: Create functional and beautiful websites for your documentation with Sphinx and the Awesome Sphinx Theme.
   :twitter:description: Create functional and beautiful websites for your documentation with Sphinx and the Awesome Sphinx Theme.


stable-worldmodel
=================

.. rst-class:: lead

A python API for simple ``World Model`` research.

----

Welcome to the docs for stable-worldmodel, a library that provides a simple and flexible interface for pre-training, fine-tuning, and, foremost, evaluating world models.
Disclaimer: This library is still in its early stages, and we are actively working on adding more features and improving the existing ones. We welcome contributions from the community!

We recommend using ``python>=3.10``, and installation using ``uv``:

.. tab-set::

    .. tab-item:: uv

        .. code-block:: bash

            uv add stable-worldmodel

    .. tab-item:: pip

        .. code-block:: bash

            pip install stable-worldmodel


.. attention::

    If you encounter ``Failed building wheel for box2d-py`` or similar errors, you might need to install ``swig`` first
    (e.g. ``apt-get install swig`` on Ubuntu or ``brew install swig`` on macOS ).

If you would like to start testing or contribute to ``stable-worldmodel`` then please install this project from source with:

.. code-block:: bash

    git clone https://github.com/rbalestr-lab/stable-worldmodel.git --single-branch
    cd stable-worldmodel
    pip install -e ".[all]"

We recommend using a conda environment to manage dependencies. We support Python with minimum version 3.10 on Linux and macOS.


Differences from Gymnasium
---------------------------

``stable-worldmodel`` builds upon the popular ``gymnasium`` library, extending its functionality to better suit the needs of world model research.
Traditionally, gymnasium has been designed for online reinforcement learning, where environments are interacted with in a step-by-step manner.

In contrast, ``stable-worldmodel`` proposes a more flexible approach that allows for both online and offline interactions with environments.
This is particularly useful for training world models, which often require access to entire trajectories of data rather than just individual steps. However, ``stable-worldmodel`` still supports online interactions, perfect for evaluating learned policies or models.

For each supported environment, ``stable-worldmodel`` provides a set of pre-defined variations and initial conditions that can be easily sampled and configured.
This allows researchers to quickly experiment with different environment configurations without manually setting up each variation. Perfect for out-of-distribution and generalization research.

Citation
--------

If you find this library useful in your research, please consider citing us:

.. code-block::

    @misc{stable-worldmodel,
      author = {},
      title = {},
      year = {2025},
      howpublished = {}
    }

.. toctree::
    :hidden:
    :caption: Introduction
    :titlesonly:

    introduction/quickstart
    introduction/showcase
    introduction/contributing

.. toctree::
    :hidden:
    :caption: API Reference
    :titlesonly:

    api/data
    api/world
    api/policy
    api/spaces
    api/wrappers

.. toctree::
    :hidden:
    :caption: Tutorials
    :titlesonly:

    tutorials/dinowm
    tutorials/new-world

.. toctree::
    :hidden:
    :caption: Worlds
    :titlesonly:

    world/pusht
    world/simple-pointmaze
    world/two_room
    world/voidrun
    world/OGBench/index