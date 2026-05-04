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