Reproducing DINO-WM
===================

This tutorial illustrates how ``stable-worldmodel`` can be used to build an end-to-end world modeling pipeline.
In the example below, we will produce all the code necessary to train and evaluate the `DINO World Model <https://arxiv.org/abs/2411.04983>`_ (DINO-WM) on the PushT environment.

More specifically, we will cover:

#. Set up the world environment (PushT).
#. Collect a dataset of environment interactions from a random policy.
#. Pre-train DINO-WM predictor on the collected dataset using the `stable-pretraining <https://github.com/rbalestr-lab/stable-pretraining>`_ library.
#. Leverage the trained model to perform planning with Model Predictive Control (MPC).
#. Evaluate the performance of the trained model in the world environment.
#. Visualize the agent's behavior.

----



.. note::

    💡 Fun fact: the original DINO-WM repo had 14,903 lines of code. We managed to recreate everything in less than X lines!