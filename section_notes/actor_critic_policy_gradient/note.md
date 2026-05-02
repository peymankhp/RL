# Policy Gradient

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

trick to ge rid of the distributions that we don’t know (intial state and transition):

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
\nabla_\theta J_\theta\approx\frac{1}{N}\sum_{i}\left(\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\right)\left(\sum_{t=1}^{T}r(s_{i,t},a_{i,t})\right)
$$

## REINFORCE Algorithm:

- sample trajectories from $\pi_\theta$
- calculate $\nabla_\theta J_\theta\approx\frac{1}{N}\sum_{i}\left(\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)\right)\left(\sum_{t=1}^{T}r(s_t,a_t)\right)$
- update parameters by $\theta\leftarrow\theta+\alpha\nabla_\theta J_\theta$

We arrive at the simillart formula for the partial observable MDPs, just replace the $s_t$ by $a_t$. So it doesn’t need to be markovian to work!

## High Variance Issue and tricks to mitigate it:

$$
\nabla_\theta J_\theta\approx\frac{1}{N}\sum_{i}\left(\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)\right)\left(\sum_{t=1}^{T}r(s_t,a_t)\right)= \frac{1}{N}\sum_{i}\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\left(\sum_{t'=1}^{T}r(s_{i,t'},a_{i,t'})\right)
$$

### 1- **Causality**

policy at time $t'$ cannot affect reward at time $t$ when $t<t'$

$$
\nabla_\theta J_\theta\approx \frac{1}{N}\sum_{i}\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\left(\sum_{t'=t}^{T}r(s_{i,t'},a_{i,t'})\right)
$$

We now have a smaller sum of reward that reduces the variance. we call this reward as the “reward to go” that can be represented as $\hat Q_{i,t}$:

$$
\nabla_\theta J_\theta\approx \frac{1}{N}\sum_{i}\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\hat Q_{i,t}
$$

### 2- Baseline

 Reducing my baseline b from the reward doesn’t change the equation, so we just use the average reward

$$
\nabla_\theta J_\theta\approx \frac{1}{N}\sum_{i}\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})(\hat Q_{i,t}-V_{i,t})=\frac{1}{N}\sum_{i}\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})A_{i,t}
$$

## How to make policy gradient off-policy

using the Importance Sampling technique:

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}[r(\tau)]=\int p_\theta(\tau)r(\tau)d\tau=\int \bar p(\tau)\frac{p_\theta(\tau)}{\bar p(\tau)}r(\tau)d\tau=\mathbb{E}_{\tau\sim \bar p(\tau)}[\frac{p_\theta(\tau)}{\bar p(\tau)}r(\tau)]
$$

$\frac{p_\theta(\tau)}{\bar p(\tau)}  = \frac{p(s_1)\prod_{t=1}^{T} \pi_\theta(a_t|s_t)p(s_{t+1}|s_t, a_t)}{p(s_1)\prod_{t=1}^{T} \bar \pi(a_t|s_t)p(s_{t+1}|s_t, a_t)}=\frac{\prod_{t=1}^{T} \pi_\theta(a_t|s_t)}{\prod_{t=1}^{T} \bar \pi(a_t|s_t)}$

$$
\nabla_\theta J(\theta)=\int \nabla_\theta p_\theta(\tau)r(\tau)d\tau=\int \bar p (\tau)\frac{ p_\theta(\tau)}{\bar p(\tau)}\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}r(\tau)d\tau=\int \bar p (\tau)\frac{ p_\theta(\tau)}{\bar p(\tau)}\nabla_\theta \log {p_\theta(\tau)}r(\tau)d\tau=\mathbb{E}_{\tau\sim \bar p(\tau)}[\frac{ p_\theta(\tau)}{\bar p(\tau)}\nabla_\theta \log {p_\theta(\tau)}r(\tau)]=\mathbb{E}_{\tau\sim \bar p(\tau)}\left[\left( \prod_{t=1}^{T}\frac{ \pi_\theta(a_t|s_t)}{ \bar \pi(a_t|s_t)}\right )\left(\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)\right)\left(\sum_{t=1}^{T}r(s_t,a_t)\right)\right]
$$

Applying causality and baseline: