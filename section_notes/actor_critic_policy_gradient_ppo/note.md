# **PPO: Proximal Policy Optimization**

## **1. Intuition and Motivation**

Classical policy gradient methods work by estimating the gradient of the expected return with respect to the policy parameters $\theta$, but these can be unstable and inefficient.

Proximal Policy Optimization (PPO) improves on earlier methods (like vanilla policy gradient and Trust Region Policy Optimization) by:

- Avoiding large, destabilizing updates
- Using a clipped objective to keep the policy close to the previous one
- Maintaining simplicity and ease of implementation

## **2. Policy Gradient Objective**

We start with the standard objective:

$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum^{\infty}_{t=0} \gamma^t r(s_t, a_t) \right]$

Our goal is to find the parameters $\theta$ of the policy $\pi_\theta(a|s)$ that maximize $J(\theta)$.

So we want to compute: $\nabla_\theta J(\theta)$

### Likelihood Ratio Trick (a.k.a. REINFORCE trick)

Let’s isolate the dependence on $\theta$. Consider: 

$\mathbb{E}_{x \sim p_\theta} [f(x)] = \int f(x) p_\theta(x) dx$

Then:

$\nabla_\theta \mathbb{E}_{x \sim p_\theta} [f(x)] = \int f(x) \nabla_\theta p_\theta(x) dx = \int f(x) \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)} p_\theta(x) dx = \mathbb{E}_{x \sim p_\theta} \left[ f(x) \nabla_\theta \log p_\theta(x) \right]$

Apply this to RL:

$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(\tau) \cdot R(\tau) \right]$

### Push Rewards into Time Step $t$

We now say:

- Actions taken at time $t$ shouldn't be influenced by future rewards.
- So we replace the full return $R(\tau)$ with the **reward-to-go** $R_t = \sum_{t'=t}^\infty \gamma^{t'-t} r_{t'}$ which is actually the $Q^\pi(s_t, a_t)$

Now:

$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot Q^\pi(s_t, a_t) \right]$

This is the classic **REINFORCE estimator**.

### Variance Reduction with Baseline

We now introduce a baseline $b(s_t)$ (e.g. $V^\pi(s_t)$):

$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (Q^\pi(s_t, a_t) - b(s_t)) \right]$

Choosing $b(s_t) = V^\pi(s_t)$, we get:

$A^\pi(s_t, a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$

So:

$\nabla_\theta J(\theta) = \mathbb{E}_{s_t, a_t \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A^\pi(s_t, a_t) \right]$

so the policy gradient theorem gives us:

 $\nabla_\theta J(\theta) = \mathbb{E}_{s_t, a_t \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A^\pi(s_t, a_t) \right]$

## **3. Generalized Advantage Estimation (GAE)**

### Step 1: What is the advantage?

$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$

So we need an estimate of $Q^\pi(s_t, a_t)$, the expected return starting from $s_t$, taking action $a_t$, then following policy $\pi$.

### Step 2: How do we estimate $Q^\pi(s_t, a_t)$?

Let’s define the **n-step return** starting from time $t$:

$R_t^{(n)} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$

This is an approximation of $Q(s_t, a_t)$, since:

- You take the **actual action $a_t$** from $s_t$,
- Then follow the policy $\pi$,
- And you **bootstrap** using the critic $V(s)$ at the end.

Now we subtract $V(s_t)$ to get the **advantage**:

$\hat{A}_t = R_t^{(n)} - V(s_t)$

So advantage estimation becomes:

$\hat{A}_t = \left( \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) \right) - V(s_t)$

This is essentially **a multi-step TD error**. That’s the bridge.

### Step 3: What GAE does

GAE doesn’t use just one n-step return.

Instead, it uses **a weighted average** of all n-step estimators:

$\hat{A}_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$

Where:

$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

This is the **1-step TD error** at time $t$.

So:

**Each $\delta_{t+l}$ represents the extra value gained at step $t+l$ over expectation**.

By summing them (with decay), you are **reconstructing the advantage estimate** from actual outcomes

GAE provides a way to estimate $A_t$ with lower variance by trading off some bias.

TD error:

$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

GAE is actually a weighted sum of TD errors

$\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots$

More compactly:

$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$

Where $\lambda \in [0,1]$ controls bias-variance trade-off.

## **4. Importance Sampling and** Surrogate Objective

When collecting data from the actual policy $\pi_{\text{ref}}$ by which we collected the trajectory, we correct for the mismatch for the changes in the policy with changes around the policy parameters $\theta$ using importance sampling:

$\mathbb{E}_{(s,a) \sim \pi_{\text{ref}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)} A^{\pi_{\text{ref}}}(s,a) \right]$

We define the **probability ratio**:

$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}$

it’s important to note that in on-policy methods like PPO, we don’t use importance sampling because we train with the data collected by another policy, as in off-policy methods. Here the importance sampling is used for weighting the advantage of the actual chosen action in the trajectory, for the PPO optimization steps where we evaluate changes to the $\theta$, without collecting any new trajectory.

We define the Surrogate Objective as:

$L^{\text{PG}}(\theta) = \mathbb{E}_{\pi_{\theta_\text{ref}}} \left[ r_t(\theta) \cdot \hat{A}_t \right]$

## **5. Clipped Surrogate Objective (PPO)**

To prevent large updates, PPO uses a clipped version of the surrogate objective:

$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \cdot \hat{A}_t \right) \right]$

This ensures the update doesn't push the new policy too far from the old one.

## **6. Complete PPO Loss**

The final objective combines:

- Clipped policy loss
- Value function loss
- Entropy bonus (for exploration)

$L(\theta) = \mathbb{E}_t \left[ L^{\text{CLIP}}_t(\theta) - c_1 (V_\theta(s_t) - R_t)^2 + c_2 \mathcal{H}(\pi_\theta(s_t)) \right]$

Where:

- $R_t$ is the return $= \hat{A}_t + V(s_t)$
- $\mathcal{H}(\pi_\theta(s_t)) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)$ is entropy
- $c_1$, $c_2$ are coefficients

## **7. Pseudocode**

```python
Initialize policy π_θ, value function V_θ
for iteration in range(N):
    Collect trajectories using π_θ
    Compute advantages Â_t using GAE
    Compute targets R_t = Â_t + V(s_t)
    for epoch in range(K):
        for mini-batch in trajectory:
            Compute r_t(θ), clipped objective
            Compute value loss, entropy bonus
            Take optimizer step on total PPO loss

```

## Implementation