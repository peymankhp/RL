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