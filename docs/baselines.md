title: Baselines
summary: Overview & Benchmarking of baseline world models.
sidebar_title: Baselines
---

## DINO World-Model

DINO World-Model (DINO-WM) is a self-supervised latent world model introduced by [Zhou et al., 2025](https://arxiv.org/pdf/2411.04983). To avoid learning from scratch and collapse, DINO-WM leverages frozen DINOv2 features to produce visual observation embedding. The model extracts patch-level features from a pretrained DINOv2 encoder and trains a latent dynamics model (predictor) to predict future states in the DINO feature space. Optimal actions are determined at test-time by performing planning with the [Cross-Entropy Method](https://www.iro.umontreal.ca/~lecuyer/myftp/papers/handbook13-ce.pdf) (CEM)

### Training Objective

The model is trained with a teacher-forcing loss, i.e. an l2-loss between the predicted next state embedding $\hat{z}_{t+1}$ and the ground-truth embedding $z_{t+1}$:

$$ \mathcal{L}_{\text{DINOWM}} = \mathcal{L}_{\text{sim}} = \| \hat{z}_{t+1} - z_{t+1} \|_2^2 $$

where $z_{t+1}$ represents the frozen DINOv2 features of the next observation.

### Benchmark

!!! danger ""
    Evaluation is performed with fixed 50 steps budget, unlike the infinite budget of the original paper

| Environment | Success Rate | Checkpoint |
|-------------|--------------|------------|
| [Push-T](envs/pusht.md) | ~86% | NA |

## Planning with Latent Dynamics Model

Planning with Latent Dynamics Model (PLDM) is a Joint-Embedding Predictive Architecture (JEPA) proposed by [Sobal et al., 2025](https://arxiv.org/pdf/2502.14819). Unlike DINO-WM which relies on frozen pretrained features, PLDM trains the encoder and predictor jointly from scratch using a combination of losses to prevent representational collapse:

- $\mathcal{L}_{\text{sim}}$: teacher-forcing loss between predicted and target embeddings
- $\mathcal{L}_{\text{std}}$, $\mathcal{L}_{\text{cov}}$: [variance-covariance regularization](https://arxiv.org/pdf/2105.04906) (VCReg) to maintain embedding diversity
- $\mathcal{L}_{\text{temp}}$: temporal smoothness regularizer between consecutive embeddings
- $\mathcal{L}_{\text{idm}}$: inverse dynamics modeling loss to predict actions from embedding pairs

Optimal actions are found at test-time by planning with the [Model Predictive Path Integral](https://acdslab.github.io/mppi-generic-website/docs/mppi.html) (MPPI) solver.

### Training Objective

$$ \mathcal{L}_{\text{PLDM}} = \mathcal{L}_{\text{sim}} + \alpha \mathcal{L}_{\text{std}} + \beta \mathcal{L}_{\text{cov}} + \delta \mathcal{L}_{\text{temp}} + \omega \mathcal{L}_{\text{idm}}$$

where $\alpha$, $\beta$, $\delta$, and $\omega$ are hyperparameters controlling the contribution of each regularization term.

### Benchmark

| Environment | Success Rate | Checkpoint |
|-------------|--------------|------------|
| ? | ? | NA |


## Goal-Conditioned Behavioural Cloning

Goal-Conditioned Behavioural Cloning (GCBC) is a simple imitation learning baseline introduced by [Ghosh et al., 2019](https://arxiv.org/pdf/1912.06088). A goal-conditioned policy $\pi_\theta(a \mid s, g)$ is trained via supervised learning to reproduce expert actions given the current state and a goal observation. In our implementation, observations and goals are encoded into DINOv2 patch embeddings before being fed to the policy network.

### Training Objective

The policy is trained to minimize the mean squared error between predicted and ground-truth actions:

$$ \mathcal{L}_{\text{GCBC}} = \mathbb{E}_{(s_t, a_t, g) \sim \mathcal{D}} \left[ \| \pi_\theta(s_t, g) - a_t \|_2^2 \right] $$

where $s_t$ is the observation embedding, $g$ is the goal embedding, and $a_t$ is the expert action.


### Benchmark

| Environment | Success Rate | Checkpoint |
|-------------|--------------|------------|
| TwoRoom | 100% | NA |
| Push-T | ~60% | NA |


## Implicit Q-Learning

Implicit Q-Learning (IQL) is an offline reinforcement learning method introduced by [Kostrikov et al., 2021](https://arxiv.org/pdf/2110.06169). IQL avoids querying out-of-distribution actions by learning a state value function $V(s)$ via expectile regression. Once the critic is learned, a policy is extracted through advantage-weighted regression (AWR). In our implementation, we consider the goal-conditioned version of this algorithm and its variants [IVL](https://arxiv.org/pdf/2410.20092) and [HILP](https://arxiv.org/pdf/2402.15567). The observations and goals are encoded into DINOv2 patch embeddings and training proceeds in two phases: value/critic learning followed by policy extraction.

### Training Objective

Training proceeds in two phases: (1) joint critic/value learning, (2) policy extraction via AWR. The critic training differs across variants:

**IQL.** IQL learns both a Q-function $Q_\psi(s_t, a_t, g)$ and a value function $V_\theta(s_t, g)$. The Q-network is trained with Bellman regression, bootstrapping from the target value network $V_{\bar{\theta}}$:

$$ \mathcal{L}_{Q} = \mathbb{E}_{(s_t, a_t, s_{t+1}, g) \sim \mathcal{D}} \left[ \left( Q_\psi(s_t, a_t, g) - \left( r(s_t, g) + \gamma \, m_t \, V_{\bar{\theta}}(s_{t+1}, g) \right) \right)^2 \right] $$

where $m_t = 0$ if $s_t = g$ (terminal) and $m_t = 1$ otherwise. The value network is trained with expectile regression against targets from the target Q-network $Q_{\bar{\psi}}$:

$$ \mathcal{L}_{V} = \mathbb{E}_{(s_t, a_t, g) \sim \mathcal{D}} \left[ L_\tau^2 \!\left( Q_{\bar{\psi}}(s_t, a_t, g) - V_\theta(s_t, g) \right) \right] $$

where $L_\tau^2(u) = |\tau - \mathbb{1}(u < 0)| \, u^2$ is the asymmetric expectile loss. The total critic-phase loss is $\mathcal{L}_{\text{critic}} = \mathcal{L}_{Q} + \mathcal{L}_{V}$.

**IVL.** [IVL](https://arxiv.org/pdf/2410.20092) removes the Q-function entirely and trains the value network $V_\theta(s_t, g)$ with expectile regression directly against bootstrapped targets from a target network $V_{\bar{\theta}}$:

$$ \mathcal{L}_{V} = \mathbb{E}_{(s_t, s_{t+1}, g) \sim \mathcal{D}} \left[ L_\tau^2 \!\left( r(s_t, g) + \gamma \, V_{\bar{\theta}}(s_{t+1}, g) - V_\theta(s_t, g) \right) \right] $$

with the same expectile loss $L_\tau^2$, discount $\gamma$, and reward $r$ as defined above.

**HILP.** [HILP](https://arxiv.org/pdf/2402.15567) uses the same loss structure as IVL but replaces the value network with a metric-based parameterization. A learned encoder $\phi$ maps observations and goals into a low-dimensional embedding space, and the value is computed as the negative L2 distance:

$$ V(s_t, g) = -\| \phi(s_t) - \phi(g) \|_2 $$

The encoder is trained end-to-end with the same expectile regression loss as IVL, but the value function has no free parameters beyond $\phi$.

**Policy extraction.** For every variant (IQL, IVL, HILP), the actor $\pi_\theta(s_t, g)$ is trained via advantage-weighted regression (AWR):

$$ \mathcal{L}_{\pi} = \mathbb{E}_{(s_t, a_t, g) \sim \mathcal{D}} \left[ \exp\!\left(\beta \cdot A(s_t, a_t, g)\right) \| \pi_\theta(s_t, g) - a_t \|_2^2 \right] $$

where $A(s_t, a_t, g) = r(s_t, g) + \gamma \, V(s_{t+1}, g) - V(s_t, g)$ is the advantage and $\beta = 3.0$ is the inverse temperature.

### Benchmark

| Environment | Variant | Success Rate | Checkpoint |
|-------------|---------|--------------|------------|
| TwoRoom | IQL | 100% | NA |
| TwoRoom | IVL | 100% | NA |
| TwoRoom | HILP | 100% | NA |
| Push-T | IQL | ? | NA |
| Push-T | IVL | ? | NA |
| Push-T | HILP | ? | NA |

