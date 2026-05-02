## Q-learning: Off-policy TD Control

Replace Sarsa's next-action value with the ***max*** over next-state actions. The target no longer cares which action the agent will actually take — it always bootstraps from the ***best*** current estimate. So $Q$ directly approximates the ***optimal*** action-value $q_*$, regardless of the behavior policy generating the data. This is ***off-policy*** TD control — and it was one of the early breakthroughs of modern RL (Watkins, 1989).

### The Q-learning update

$$
Q(S_t, A_t) \;\leftarrow\; Q(S_t, A_t) + \alpha\,\big[\,R_{t+1} + \gamma\,\max_a Q(S_{t+1}, a) - Q(S_t, A_t)\,\big]. \tag{6.8}
$$

**What makes it off-policy.** The target uses $\max_a Q(S_{t+1}, a)$ — the ***greedy*** choice in $S_{t+1}$. But the action $A_t$ that produced the transition can come from ***any*** exploratory behavior policy (typically $\varepsilon$-greedy). The learned $Q$ chases $q_*$ no matter how exploratory the behavior, ***as long as every*** $(s, a)$ ***pair keeps getting visited***.

**Pseudocode.**

```
Initialize Q(s, a) arbitrarily; Q(terminal, ·) = 0
Loop for each episode:
    S <- initial state
    Loop for each step:
        Choose A from S using ε-greedy(Q)        # behavior policy
        Take A; observe R, S'
        Q(S, A) <- Q(S, A) + α [ R + γ max_a Q(S', a) - Q(S, A) ]
        S <- S'
    until S is terminal
```

Note: no $A'$ needed. The target uses a greedy lookahead, not a sampled next action.

### Worked example: Cliff Walking (Sarsa vs. Q-learning)

Gridworld with a cliff between start $S$ and goal $G$. Reward $-1$ per step, $-100$ for stepping into the cliff (and bounced back to $S$). $\varepsilon$-greedy with $\varepsilon = 0.1$.

- ***Q-learning*** learns the ***optimal*** policy: walk right along the edge of the cliff. But because behavior is still $\varepsilon$-greedy, random exploration occasionally knocks it off the cliff → reward streams look ***worse online*** than its value function would suggest.
- ***Sarsa*** learns a ***safer*** longer path that accounts for the $\varepsilon$-greedy behavior it is actually using. Asymptotic return per episode is better than Q-learning at $\varepsilon = 0.1$.
- If $\varepsilon \to 0$, both converge to the same optimal policy.

> ***Moral.*** "Optimal" depends on the question. Q-learning finds the optimal deterministic policy. Sarsa finds the optimal $\varepsilon$***-greedy*** policy. If behavior will always have exploration, Sarsa's answer can be more useful online.
> 

### Recall

#### Q1. What makes Q-learning off-policy?

Its target is $\max_a Q(S_{t+1}, a)$, the value under the greedy policy — not under whatever policy generated the data. So it evaluates a ***target policy*** (greedy) that differs from the ***behavior policy*** (exploratory).

#### Q2. In Cliff Walking, why does Q-learning's online performance look worse than Sarsa's even though it learns the "optimal" policy?

It learns the optimal ***deterministic*** path right along the cliff. Under $\varepsilon$-greedy behavior, random exploration occasionally pushes the agent off the cliff, incurring large negative rewards. Sarsa learns a longer safer path that factors in its own exploration.

#### Q3. If behavior is greedy (no exploration), are Q-learning and Sarsa the same?

Their updates become equivalent in expectation: a greedy policy's chosen action ***is*** $\arg\max_a Q(S', a)$, so $Q(S', A')$ and $\max_a Q(S', a)$ coincide. The action selections and updates match.

#### Q4. What condition on behavior is needed for Q-learning to converge to $q_*$?

Every state-action pair must keep being visited (plus the usual stochastic-approximation conditions on $\alpha$). Any exploratory behavior satisfying that works.