## 1. First Principles: The Foundation

Before we optimize a policy, we must define the world it lives in.

### The Markov Decision Process (MDP)
An MDP is a tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \gamma)$.
*   **$\mathcal{S}$**: The state space.
*   **$\mathcal{A}$**: The action space.
*   **$\mathcal{P}(s_{t+1} | s_t, a_t)$**: The **Transition Dynamics**. This is the "physics" of the world. It tells us the probability of landing in $s_{t+1}$ given the current state and action.
*   **$r(s_t, a_t)$**: The **Reward Function**. It provides immediate scalar feedback.

### The Policy ($\pi_\theta$)
The policy is a mapping from states to a probability distribution over actions: $\pi_\theta(a|s) = P(a_t = a | s_t = s, \theta)$.
*   The parameter $\theta$ represents the weights of our neural network.
*   **Goal**: Find $\theta$ that results in the "best" behavior.

### Trajectories and Returns
A **trajectory** $\tau$ is a sequence of states and actions: $(s_1, a_1, s_2, a_2, \dots, s_T, a_T)$.
The probability of a trajectory occurring under parameters $\theta$ is:
$$p_\theta(\tau) = p(s_1)\prod_{t=1}^{T} \pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)$$

The **Return** $r(\tau)$ is the total accumulated reward: $r(\tau) = \sum_{t=1}^T r(s_t, a_t)$.

---

## 2. The Objective Function

In Reinforcement Learning, we want to maximize the **Expected Return**:
$$J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [r(\tau)] = \int p_\theta(\tau) r(\tau) d\tau$$

### Why is this the right objective?
Unlike supervised learning, where we have a "correct label," in RL we only have rewards. We cannot maximize the reward of a single trajectory because the world is stochastic. We maximize the **average** performance over all possible paths the agent might take.

---

## 3. Deriving the Policy Gradient

We want to move $\theta$ in the direction that increases $J(\theta)$. This requires the gradient $\nabla_\theta J(\theta)$.

### Step 1: Gradient of the Expectation
$$\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau) r(\tau) d\tau = \int \nabla_\theta p_\theta(\tau) r(\tau) d\tau$$
*Critique*: We cannot compute this because we don't know the transition dynamics inside $p_\theta(\tau)$. We need to turn this back into an expectation so we can estimate it via sampling.

### Step 2: The Log-Derivative Trick (Likelihood Ratio)
We use the identity: $\nabla_\theta \log f(\theta) = \frac{\nabla_\theta f(\theta)}{f(\theta)} \implies \nabla_\theta f(\theta) = f(\theta) \nabla_\theta \log f(\theta)$.

Applying this to our integral:
$$\nabla_\theta J(\theta) = \int p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} r(\tau) d\tau = \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) d\tau$$

**The Result:**
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [\nabla_\theta \log p_\theta(\tau) r(\tau)]$$

### Step 3: Decomposing the Log Probability
What is $\nabla_\theta \log p_\theta(\tau)$? Let's expand the trajectory probability:
$$\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^T \log \pi_\theta(a_t|s_t) + \sum_{t=1}^T \log p(s_{t+1}|s_t, a_t)$$

Now, take the gradient with respect to $\theta$:
*   $\nabla_\theta \log p(s_1) = 0$ (The initial state doesn't depend on our network).
*   $\nabla_\theta \log p(s_{t+1}|s_t, a_t) = 0$ (The environment's physics don't depend on our network).

We are left with:
$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### Step 4: The Final Gradient Estimator
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \left( \sum_{t=1}^T r(s_t, a_t) \right) \right]$$

**Intuition:** 
*   If the total reward $r(\tau)$ is high (positive), we increase the log-probability of all actions taken in that trajectory.
*   If $r(\tau)$ is low or negative, we decrease their probability.
*   **Crucial Insight:** It doesn't matter if we don't know the transitions! They dropped out of the gradient.

---

## 4. The REINFORCE Algorithm

1.  Sample $N$ trajectories $\{\tau_i\}$ by running the current policy $\pi_\theta$.
2.  Estimate the gradient:
    $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \right) \left( \sum_{t=1}^T r(s_{i,t}, a_{i,t}) \right)$$
3.  Update parameters: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.


# Policy Gradient


$$
\tau = (s_1, a_1, s_2, a_2, \dots, s_T, a_T)
$$


$$
r(\tau) = \sum_{t=1}^{T} r(s_t, a_t)
$$


$$
p_\theta(\tau) = p_\theta (s_1, a_1, ..., s_T, a_T) = p(s_1)\prod_{t=1}^{T} \pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)
$$

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[\sum_{t}^{}r(s_t,a_t)]\approx\frac{1}{N}\sum_{i}\sum_{t}r(s_{i,t}, a_{i,t})
$$

$$
\theta^*=\argmax_\theta\mathbb{E}_{\tau\sim p_\theta(\tau)}[\sum_{t}^{}r(s_t,a_t)]=\argmax_\theta\sum_{t}^{}\mathbb{E}_{(s_t,a_t)\sim p_\theta(s_t,a_t)}[r(s_t,a_t)]
$$

$$
J(\theta)=\mathbb{E}_{\tau\sim p_{\theta} (\tau)}[r(\tau)]=\int p_\theta(\tau)r(\tau)d\tau
$$

trick to get rid of the distributions that we don’t know (initial state and transition):

$$
\nabla_\theta J(\theta)=\int \nabla_\theta p_\theta(\tau)r(\tau)d\tau=\int p_\theta(\tau)\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}r(\tau)d\tau=\int p_\theta(\tau)\nabla_\theta \log {p_\theta(\tau)}r(\tau)d\tau=\mathbb{E}_{\tau\sim p_\theta(\tau)}[\nabla_\theta \log {p_\theta(\tau)}r(\tau)]
$$

$$
\nabla_\theta \log {p_\theta(\tau)}=\nabla_\theta\log \left(p(s_1)\prod_{t=1}^{T} \pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)\right)=\nabla_\theta \log p(s_1)+ \nabla_\theta\sum_{t=1}^{T}\log\pi_\theta(a_t|s_t)+\nabla_\theta  p(s_{t+1}|s_t, a_t)=\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)
$$

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\left[\left(\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)\right)\left(\sum_{t=1}^{T}r(s_t,a_t)\right)\right]
$$

$$
\nabla_\theta J_\theta\approx\frac{1}{N}\sum_{i}^{N}\left(\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\right)\left(\sum_{t=1}^{T}r(s_{i,t},a_{i,t})\right)
$$

