## Expected Sarsa

Halfway between Sarsa and Q-learning. Sarsa uses one ***sampled*** $Q(S_{t+1}, A_{t+1})$ — unbiased but noisy. Q-learning uses the ***max*** — low-variance but biased and committed to a greedy target. ***Expected Sarsa*** uses the ***expected value*** of $Q(S_{t+1}, \cdot)$ under the target policy. Same expected target as Sarsa, but no sampling noise from picking $A_{t+1}$. Same idea as Q-learning, but generalized to any target policy.

### The update

$$
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha\,\Big[\,R_{t+1} + \gamma\,\mathbb{E}_\pi\!\big[Q(S_{t+1}, A_{t+1}) \mid S_{t+1}\big] - Q(S_t, A_t)\,\Big]
$$

$$
= Q(S_t, A_t) + \alpha\,\Big[\,R_{t+1} + \gamma\sum_a \pi(a\mid S_{t+1})\,Q(S_{t+1}, a) - Q(S_t, A_t)\,\Big]. \tag{6.9}
$$

Given the next state $S_{t+1}$, replace the single sample $Q(S_{t+1}, A_{t+1})$ with the ***weighted average*** of $Q(S_{t+1}, a)$ under $\pi$. No randomness from the agent's next action enters the update.

### Why it tends to dominate Sarsa

- ***Same expected target*** as Sarsa, so convergence story carries over.
- ***Lower variance***: Sarsa's single sample $A_{t+1}$ is a Monte Carlo estimate of the very expectation Expected Sarsa computes directly. Eliminating that noise lets Expected Sarsa survive ***larger*** step sizes.
- In deterministic environments (e.g., Cliff Walking with deterministic transitions), Expected Sarsa can safely use $\alpha = 1$ without diverging. Sarsa cannot: its sampling noise blows up at $\alpha = 1$.
- Cost: one extra $O(|\mathcal{A}|)$ sum per update. For most problems, cheap.

### Unifying view: it subsumes Q-learning

If $\pi$ is ***greedy***, then $\sum_a \pi(a\mid S_{t+1})\,Q(S_{t+1}, a) = \max_a Q(S_{t+1}, a)$. So:

- Expected Sarsa with $\pi$ = greedy ***is*** Q-learning.
- Expected Sarsa with $\pi$ = behavior is on-policy; with $\pi \ne$ behavior, it is off-policy.

So Expected Sarsa is a single algorithm that ***generalizes both Sarsa and Q-learning*** — strictly better than Sarsa at the same asymptotic cost, and strictly more flexible than Q-learning.

### Recall

#### Q1. What is the Expected Sarsa target?

$R_{t+1} + \gamma \sum_a \pi(a\mid S_{t+1})\,Q(S_{t+1}, a)$ — the reward plus the target-policy-weighted expectation of $Q$ in the next state.

#### Q2. Why is Expected Sarsa usually better than Sarsa at the same cost class?

Same expected update, but without the sampling variance from drawing $A_{t+1}$ — which lets it use larger step sizes without divergence.

#### Q3. When does Expected Sarsa reduce to Q-learning?

When the target policy $\pi$ is greedy, the expectation $\sum_a \pi(a\mid s)\,Q(s, a)$ collapses to $\max_a Q(s, a)$. Expected Sarsa is then identical to Q-learning.

#### Q4. Why can Expected Sarsa safely use $\alpha = 1$ on deterministic Cliff Walking while Sarsa cannot?

In deterministic transitions, the only randomness in Sarsa's target is its own action sample $A_{t+1}$. With $\alpha = 1$, that noise is injected into $Q$ undiluted and destabilizes training. Expected Sarsa averages $A_{t+1}$ out analytically, leaving no noise — so $\alpha = 1$ is fine.