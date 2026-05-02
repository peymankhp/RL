## Advantages of TD Prediction Methods

TD combines the best of both parents. Over **DP**: no model needed — it learns from raw experience. Over **MC**: fully online/incremental — it updates after each transition, so it works with continuing tasks or very long episodes. And on stochastic tasks, TD empirically converges faster than constant-$\alpha$ MC, though no general proof is yet known.

### Three concrete advantages

1. ***Model-free*** (unlike DP). TD only needs experience, not the transition kernel $p(s',r|s,a)$.
2. ***Online / incremental*** (unlike MC). Updates happen after every transition — no waiting for episode end. This matters when episodes are very long, or the task is continuing (no terminal state), or exploratory actions force MC to discard partial episodes.
3. ***Usually faster*** in practice. On stochastic problems, TD tends to beat constant-$\alpha$ MC across a broad range of step sizes.

### Convergence guarantees

For any fixed policy $\pi$, ***TD(0) converges to*** $v_\pi$:

- ***In the mean***, for a constant step size $\alpha$ if it is sufficiently small.
- ***With probability 1***, if $\alpha_t$ shrinks according to the usual stochastic approximation conditions:

$$
\sum_t \alpha_t = \infty, \qquad \sum_t \alpha_t^2 < \infty.
$$

Most proofs are for the tabular case; some extend to linear function approximation (Chapter 9).

### Worked example: 5-state Random Walk

**The MRP.** States A–B–C–D–E between two terminal states. Start at C. At each step go left or right with probability $\tfrac{1}{2}$. Reward $+1$ only on the right-terminal; else 0. Undiscounted ($\gamma = 1$).

**True values.** Because there's no discounting, the value of each state is the probability of terminating on the right. By random-walk symmetry:

$$
v_\pi(A)=\tfrac{1}{6},\; v_\pi(B)=\tfrac{2}{6},\; v_\pi(C)=\tfrac{3}{6},\; v_\pi(D)=\tfrac{4}{6},\; v_\pi(E)=\tfrac{5}{6}.
$$

**What the book shows (Fig. 6.1).** TD(0) and constant-$\alpha$ MC both approach the true values as episodes accumulate, but TD is consistently lower on RMS error. The first episode only changes $V(A)$ (if it terminates left) or $V(E)$ (if it terminates right) — all other states bootstrap from unchanged neighbors, so no update propagates to them.

### Code: 5-state Random Walk with TD(0)

```python
import numpy as np

def random_walk_episode(rng):
    # States: 0 = left terminal, 1..5 = A..E, 6 = right terminal
    trajectory = []
    s = 3  # start in C
    while s not in (0, 6):
        s_next = s + rng.choice([-1, 1])
        r = 1.0 if s_next == 6 else 0.0
        trajectory.append((s, r, s_next))
        s = s_next
    return trajectory

def td0_random_walk(alpha=0.1, episodes=100, seed=0):
    rng = np.random.default_rng(seed)
    V = np.full(7, 0.5)
    V[0] = V[6] = 0.0
    for _ in range(episodes):
        for s, r, s_next in random_walk_episode(rng):
            V[s] += alpha * (r + V[s_next] - V[s])
    return V[1:6]  # A..E

# Example: print(td0_random_walk())  ->  close to [1/6, 2/6, 3/6, 4/6, 5/6]
```

### Recall

#### Q1. Name three advantages of TD over MC and DP.

Model-free (unlike DP); online/incremental, so works on continuing tasks and long episodes (unlike MC); usually faster empirical convergence on stochastic tasks.

#### Q2. Why does TD work for continuing tasks when MC doesn't?

MC needs the episode to terminate to compute $G_t$. A continuing task has no terminal state, so MC has nothing to average. TD updates after every transition and does not depend on episode boundaries.

#### Q3. What are the stochastic approximation conditions for TD(0) to converge with probability 1?

$\sum_t \alpha_t = \infty$ (enough total learning) **and** $\sum_t \alpha_t^2 < \infty$ (step sizes must shrink fast enough to damp noise).

#### Q4. In the 5-state random walk, why is the true value of C equal to $0.5$?

By symmetry, starting from C the probability of terminating on the right equals the probability of terminating on the left, and the reward is $+1$ only on the right, so $v_\pi(C) = \tfrac{1}{2}$.

## Optimality of TD(0)

Freeze your data — a finite batch of episodes — and keep presenting it to TD(0) until it converges. It settles on a specific answer. ***It is not the sample mean.*** It is the value function of the ***maximum-likelihood Markov model*** implied by the observed transitions. Under batch MC, by contrast, $V(s)$ just becomes the average of the returns seen after each visit to $s$. MC is optimal at ***fitting the training set***; TD is optimal at ***predicting future returns, if the process really is Markov***.

### Batch updating

Compute the increments from (6.1) or (6.2) for every visited state across the whole batch of episodes, then apply them all at once. Repeat until $V$ stops changing. This is ***batch updating***: updates only land after each complete pass through the data.

Under batch updating with sufficiently small $\alpha$:

- Batch **constant-**$\alpha$ **MC** converges to the ***sample-return average*** at each state — the answer that ***minimizes mean-squared error on the training set***.
- Batch **TD(0)** converges to the ***certainty-equivalence estimate*** — the value function that would be exactly correct if the maximum-likelihood Markov model were the true model.

### Example 6.4: "You are the Predictor"

You observe 8 episodes from an unknown MRP:

| # | Trajectory (state, reward, ...) |
| --- | --- |
| 1 | A, 0, B, 0 |
| 2–7 | B, 1 |
| 8 | B, 0 |

Everyone agrees $V(B) = \tfrac{6}{8} = 0.75$ — six of eight times we saw B we got return 1, the other two times 0.

But what is the optimal $V(A)$?

- ***MC answer*** ($V(A) = 0$). You only saw A once; the return was $0 + 0 = 0$. This *is* the minimum-MSE answer for the training data — actually zero error.
- ***TD answer*** ($V(A) = 0.75$). 100% of the time you saw A, you transitioned to B (with reward 0). So if $V(B) = 0.75$, then by the Bellman relation $V(A) = 0 + V(B) = 0.75$. This is the answer that would be exactly correct for the ML Markov model fit from the data.

***If the process really is Markov***, the TD answer is expected to give ***lower error on future data***, even though the MC answer is perfect on the past data. That is the sense in which TD's optimality is "more relevant to predicting returns."

### Why batch TD is fast but intractable exactly

The certainty-equivalence estimate could in principle be computed directly: form the ML Markov model ($O(n^2)$ memory for $n = |\mathcal{S}|$), then solve the linear system for its value function ($O(n^3)$ time conventionally).

TD approximates the same answer using $O(n)$ ***memory*** and a sweep per episode. Striking: for large state spaces, TD may be the only feasible way to get close to the certainty-equivalence solution.

> ***Takeaway.*** Even when nonbatch TD and nonbatch MC don't reach their respective batch limits, they are moving ***toward*** them. TD's speed advantage in practice is (partly) because its target is usually a better estimate of $v_\pi$ than MC's.
> 

### Recall

#### Q1. What does batch constant-$\alpha$ MC converge to?

The sample-return average at each state — the answer that minimizes mean-squared error on the training data.

#### Q2. What does batch TD(0) converge to?

The ***certainty-equivalence*** value function: the value function of the maximum-likelihood Markov model fit from the observed transitions.

#### Q3. In "You are the Predictor," why can $V(A) = 0.75$ be a better estimate than $V(A) = 0$ even though the training data gives zero error for 0?

If the process is Markov, A always transitions to B, and $V(B) = 0.75$ is a good estimate. So future returns from A should also average about $0.75$. Fitting the single observed A-episode exactly *overfits* to a noisy sample.

#### Q4. In one sentence, what is certainty equivalence?

It is the estimate you get by pretending the maximum-likelihood model of the environment is the truth, then solving that model exactly.

#### Q5. Why is batch TD practical even when computing the certainty-equivalence estimate directly is not?

Direct computation needs $O(n^2)$ memory and $O(n^3)$ time. TD reaches essentially the same answer with $O(n)$ memory by repeated sweeps — tractable even when the state space is large.