## Maximization Bias and Double Learning

Every control method in this chapter so far has a $\max$ buried in its target (Q-learning) or its policy ($\varepsilon$-greedy over $Q$). When the underlying $Q$ estimates are noisy, ***the max of noisy estimates is biased upward*** — you keep picking the lucky ones. That makes your estimate of the max value systematically too optimistic. ***Double learning*** fixes it with a classic statistical trick: use two independent estimators, one to pick the argmax and one to evaluate it.

### The bias

Imagine $n$ actions whose ***true*** value is exactly 0, but your estimates $\hat{Q}(a)$ are noisy (some above 0, some below). Then $\max_a \hat{Q}(a) > 0$ in expectation — the max of noise is positive. Using $\max_a \hat{Q}(a)$ as an estimate of $\max_a Q(a) = 0$ is biased upward. That is ***maximization bias***.

It is not a bug in any one algorithm — it is an artifact of using the same samples both to ***choose*** which action looks best ***and*** to ***evaluate*** how good it is.

### Example 6.7: Maximization Bias MDP

Two non-terminal states $A$ and $B$. Episodes always start in $A$.

- From $A$: ***right*** → terminal with reward $0$.
- From $A$: ***left*** → $B$ with reward $0$.
- From $B$: many actions, each terminating with reward $\sim \mathcal{N}(-0.1, 1)$.

True expected return from ***left*** is $-0.1$. So ***left*** is always a mistake. But Q-learning's $\max_a Q(B, a)$ is positively biased by the noisy rewards in $B$, so for many episodes state $B$ appears valuable → the agent prefers ***left*** from $A$ → persistent wrong behavior.

Figure 6.5 shows Q-learning taking ***left*** much more than the 5% minimum forced by $\varepsilon$-greedy, even after 300 episodes. ***Double Q-learning*** is essentially unaffected.

### Double learning: separate selection from evaluation

Maintain ***two*** independent estimators $Q_1$ and $Q_2$ of the same quantity. On each step, flip a coin:

- ***Heads*** — update $Q_1$ using $Q_2$ to evaluate the argmax picked by $Q_1$:

$$
Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha\Big[\,R_{t+1} + \gamma\,Q_2\!\big(S_{t+1},\,\arg\max_a Q_1(S_{t+1}, a)\big) - Q_1(S_t, A_t)\,\Big]. \tag{6.10}
$$

- ***Tails*** — symmetric, with $Q_1$ and $Q_2$ swapped.

**Why the trick works.** The ***selection*** (which action maximizes) uses $Q_1$; the ***evaluation*** (what is its value) uses $Q_2$. Because $Q_1$ and $Q_2$ are trained on disjoint streams of updates, the noise in their argmaxes is uncorrelated with the noise in each other's values — and $\mathbb{E}[Q_2(\arg\max_a Q_1(a))] = q(\arg\max_a Q_1(a))$, which is unbiased in the key sense.

**Cost.** 2× memory, same per-step computation (each step only touches one of $\{Q_1, Q_2\}$). Behavior policy can use $Q_1 + Q_2$ (or their average) for action selection.

### Recall

#### Q1. What is maximization bias, in one sentence?

Using the max of noisy estimates as an estimate of the max of the true values is biased upward — the max picks up positive noise more often than negative.

#### Q2. Why does Q-learning suffer from it?

Its target is $\max_a Q(S_{t+1}, a)$, which bakes the biased max directly into the bootstrap.

#### Q3. Explain the trick behind Double Q-learning.

Keep two independent estimators. On each update, use one ($Q_1$) to choose the argmax and the ***other*** ($Q_2$) to evaluate it — or vice versa. Because selection and evaluation come from different estimators, the argmax's noise is uncorrelated with the value's noise, so the bias does not compound.

#### Q4. In the Max Bias example, why does Q-learning persistently take ***left*** even though it is a mistake?

State $B$'s actions have noisy rewards with a mean of $-0.1$. Q-learning's $\max_a Q(B, a)$ is biased upward by that noise, so $B$ looks valuable. The agent happily goes ***left*** to reach it.

#### Q5. What's the cost of Double Q-learning?

2× memory. Same per-step computation (only one of the two tables is updated per step). There are analogous double versions of Sarsa and Expected Sarsa.

## Games, Afterstates, and Other Special Cases

**Plain English.** In games — and in many planning problems — you know ***exactly*** what state will result from your own move. The uncertainty comes from what happens ***next*** (the opponent's reply, stochastic environment dynamics). In that setting it is natural to evaluate the state ***right after*** your move, before the environment responds. These are called ***afterstates***. Because many $(s, a)$ pairs lead to the same afterstate, afterstate value functions generalize for free where an action-value function would have to learn each pair separately.

### Afterstates, formally

Define an ***afterstate*** as the state of the world ***after*** the agent has committed to an action but ***before*** the environment (e.g., opponent, stochastic transition) has responded. An ***afterstate value function*** $W(s^a)$ assigns values to afterstates. GPI still applies: evaluate $W$ for the current policy, improve the policy to be greedy w.r.t. $W$.

### Why this helps: Tic-Tac-Toe

In tic-tac-toe, multiple $(s, a)$ pairs can land in the same board position. For example, the same final position can arise by choosing a different sequence of equivalent early moves. An action-value function would learn the value of ***each*** such pair separately. An afterstate function evaluates the shared resulting position ***once***, and any learning about it immediately transfers to all the pairs that lead to it. That's a huge free-generalization win in games with rich symmetry.

### Where afterstates arise

- ***Board games*** where you know the deterministic effect of your move before the opponent replies.
- ***Queuing tasks*** where actions like "assign this customer to server $k$" or "reject this customer" have a known immediate effect on the queue state.
- ***Inventory / scheduling / resource allocation*** — anywhere your own action's immediate effect is known and the hard randomness sits in what happens next (demand, arrivals, opponent moves).

In short: any task where the dynamics factor as "deterministic agent effect" → "stochastic environment effect" is a candidate.

### Recall

#### Q1. What is an afterstate?

The state of the world ***after*** the agent has committed to an action but ***before*** the environment has responded — i.e., the deterministic consequence of the agent's own move alone.

#### Q2. Why do afterstate value functions generalize better than action-value functions in games?

Many $(s, a)$ pairs can lead to the same afterstate (board position). An action-value function must evaluate each pair separately; an afterstate function evaluates their common resulting position once and shares the value across all pairs.

#### Q3. In what kinds of tasks are afterstates natural?

Any task where the agent knows the ***immediate deterministic effect*** of its action, and all the stochasticity sits in the ***environment's response***: board games, queuing, scheduling, inventory, routing.

## Chapter 6 — Connections

- **TD = MC + DP.** The unifying idea of the chapter: bootstrapping (DP) + sampling (MC) → model-free, online prediction.
- **Control is just GPI with a TD prediction step.** Sarsa, Q-learning, and Expected Sarsa all do GPI; they differ only in the target used for the action-value bootstrap.
- **Three targets, one algorithm family.**
    - ***Sarsa*** → target uses the sampled next action $A_{t+1}$.
    - ***Q-learning*** → target uses $\max_a Q(S_{t+1}, a)$.
    - ***Expected Sarsa*** → target uses $\sum_a \pi(a \mid S_{t+1})\,Q(S_{t+1}, a)$.
    - **Expected Sarsa subsumes Q-learning** when $\pi$ is greedy, and strictly dominates Sarsa by removing the $A_{t+1}$ sampling variance.
- **Maximization bias** is an under-appreciated systemic error in all $\max$-based methods; double learning solves it at the cost of 2× memory.
- **Afterstates** are a special-case structure that gives huge free generalization wherever the agent's immediate effect is known.
- **Bridging forward.** Chapter 7 introduces $n$***-step TD***, which interpolates between TD(0) and MC along the temporal axis. Chapter 12 generalizes this to $TD(\lambda)$. Chapter 13 adds ***actor-critic*** methods, a third path to TD-based control we skipped here.