## TD Prediction

Both MC and TD try to estimate the value of each state from experience, but they differ on **when** they commit to an update. MC is the patient accountant: it waits until the episode ends, totals the actual return, and only then revises its estimates. TD is the restless forecaster: after every single step, it already has enough information to revise — it replaces the *unknown* future return with its *current guess* for the next state's value, plus the reward it just observed. In short: ***MC learns a guess from a fact; TD learns a guess from a better guess.***

### The core update rule

A simple every-visit MC update (good for nonstationary problems) is

$$
V(S_t) \;\leftarrow\; V(S_t) + \alpha\,\big[\,G_t - V(S_t)\,\big], \tag{6.1}
$$

where $G_t$ is the **actual return** following time $t$ and $\alpha\in(0,1]$ is the step size. This is called ***constant-***$\alpha$ ***MC***. You must wait until the episode ends to know $G_t$.

The simplest TD method — ***TD(0)*** or ***one-step TD*** — needs only one step:

$$
V(S_t) \;\leftarrow\; V(S_t) + \alpha\,\big[\,\underbrace{R_{t+1} + \gamma\, V(S_{t+1})}_{\text{TD target}} - V(S_t)\,\big]. \tag{6.2}
$$

**Reading the symbols.** $R_{t+1}$ is the reward observed on the transition. $V(S_{t+1})$ is the *current* estimate of the next state's value. $\gamma$ discounts the future. The bracketed quantity is how far off we were; $\alpha$ controls how much we correct.

**Why this is legitimate.** From Chapter 3 we have two equivalent definitions of $v_\pi$:

$$
v_\pi(s) \doteq \mathbb{E}_\pi[\,G_t \mid S_t=s\,] \;=\; \mathbb{E}_\pi[\,R_{t+1} + \gamma\, v_\pi(S_{t+1}) \mid S_t=s\,]. \tag{6.3, 6.4}
$$

MC uses a **sample** of the left expression (one observed $G_t$) as its target. DP uses the right expression, assuming the environment's model gives the expectation, but replaces $v_\pi(S_{t+1})$ with the current estimate $V(S_{t+1})$ — that is the bootstrap. **TD does both at once**: it *samples* the expectation (one observed $R_{t+1}$ and $S_{t+1}$) *and* bootstraps with $V(S_{t+1})$. That combination is the source of TD's power — and, later, its subtlety.

> **Backup diagram.** TD(0) is a ***sample update***: it looks ahead one step to a single successor state (not to the full distribution of successors, as DP does) and uses that one sample plus the current $V$ to back up a value to $S_t$.
> 

**Tabular TD(0) procedural form:**

```
Input: policy π to be evaluated; step size α ∈ (0, 1]
Initialize V(s) arbitrarily for all s, with V(terminal) = 0
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        A ← action given by π for S
        Take action A; observe R, S'
        V(S) ← V(S) + α [ R + γ V(S') − V(S) ]
        S ← S'
    until S is terminal
```

### The TD error

The bracketed quantity in (6.2) is important enough to have its own name — the ***TD error***:

$$
\delta_t \;\doteq\; R_{t+1} + \gamma\, V(S_{t+1}) - V(S_t). \tag{6.5}
$$

$\delta_t$ is the error in the estimate $V(S_t)$ **as judged one step later**. You cannot compute it at time $t$ — it is only available at time $t+1$.

**Useful identity.** If $V$ does not change during the episode, the **Monte Carlo error decomposes as a sum of TD errors**:

$$
G_t - V(S_t) \;=\; \sum_{k=t}^{T-1} \gamma^{k-t}\,\delta_k. \tag{6.6}
$$

This is **not** exact when $V$ *is* updated during the episode (as in TD(0) proper), but it holds approximately for small $\alpha$. Generalizations of this identity are the backbone of $TD(\lambda)$ in Chapter 12.

### Worked example: Driving Home

You commute home. Each day you predict how long it will take. States are waypoints; rewards are the elapsed minutes on each leg; $\gamma = 1$ (undiscounted).

| State | Elapsed | Predicted time-to-go $V(S_t)$ | Predicted total |
| --- | --- | --- | --- |
| leaving office (Fri, 6pm) | 0 | 30 | 30 |
| reach car, raining | 5 | 35 | 40 |
| exiting highway | 20 | 15 | 35 |
| 2ndary road, behind truck | 30 | 10 | 40 |
| entering home street | 40 | 3 | 43 |
| arrive home | 43 | 0 | 43 |

**MC view (wait till the end).** With $\alpha=1$, MC sets each state's estimate to the actual return. You revise "exiting highway" from **15 → 23** (it really took 23 more minutes). But you can only do this **after you've arrived home**.

**TD view (update step by step).** With $\alpha=1,\,\gamma=1$, TD(0) replaces each estimate with $R + V(\text{next})$ — that is, each prediction is pulled toward the *next* prediction, as soon as you see it:

- At the car: $V(\text{office}) \leftarrow 5 + 35 = 40$. (Up from 30 — the rain already told you.)
- Exiting highway: $V(\text{car}) \leftarrow 15 + 15 = 30$. (Down from 35 — highway went well.)
- 2ndary road, behind truck: $V(\text{highway exit}) \leftarrow 10 + 10 = 20$. (Up from 15 — truck is bad news.)
- Entering home street: $V(\text{2ndary}) \leftarrow 10 + 3 = 13$. (Up from 10.)

Each TD revision is proportional to the *change over time of the prediction* — that is, to the **temporal difference** between successive predictions. No waiting, no hindsight needed.

> ***Intuition to keep:*** the TD update is what a weather forecaster does every hour — revise today's forecast when new data comes in, don't wait until the storm is over to grade yourself.
> 

### Code: Tabular TD(0) in NumPy

A minimal, runnable implementation that mirrors the pseudocode box above:

```python
import numpy as np

def td0_predict(env, policy, n_states, alpha=0.1, gamma=1.0, episodes=200):
    """
    Estimate V_pi via tabular TD(0).

    env.reset() -> s0
    env.step(a) -> (s_next, r, done)
    policy(s)   -> a
    """
    V = np.zeros(n_states)
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = policy(s)
            s_next, r, done = env.step(a)
            # Bootstrap target: reward + discounted next-state value.
            # At terminal states V(terminal) must be 0 -> gate with `not done`.
            target = r + (0.0 if done else gamma * V[s_next])
            td_error = target - V[s]
            V[s] += alpha * td_error
            s = s_next
    return V
```

**What to notice.** One line does the real work: `V[s] += alpha * (target - V[s])`. Everything else is plumbing. The `not done` gate is easy to forget and is exactly how "V(terminal) = 0" is enforced in code.

### Recall — quiz yourself before moving on

#### Q1. In one sentence, what is the difference between MC and TD(0) targets?

MC uses the actual return $G_t$ (known only at episode end); TD(0) uses $R_{t+1} + \gamma V(S_{t+1})$ — the observed reward plus the current estimate of the next state. TD therefore samples *and* bootstraps.

#### Q2. Why can TD learn on continuing tasks while MC cannot?

MC needs the episode to end to compute $G_t$. A continuing task has no terminal state, so MC has nothing to average. TD updates after every transition, so it does not depend on episodes ending.

#### Q3. What is the TD error $\delta_t$, and when does it become available?

$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$. It is the error in the time-$t$ estimate as judged one step later, so it is only available at time $t+1$.

#### Q4. Equation (6.6) says the MC error equals the sum of (discounted) TD errors. When is this exact, and when only approximate?

Exact if $V$ is held fixed during the episode (as in offline MC). Only approximate if $V$ is updated during the episode (as in online TD(0)); the approximation is good when $\alpha$ is small.

#### Q5. In the Driving Home example, why does TD revise $V(\text{office})$ from 30 to 40 *before* you arrive home?

When you reach the car and see rain, your estimate for the *next* state jumps to 35 and you've used 5 minutes getting there. TD(0) immediately pulls the previous state's value toward $R + V(\text{next}) = 5 + 35 = 40$. You don't need to see the rest of the trip — the new information about the next state is enough.

#### Q6. Why is TD called a "sample update" while DP uses "expected updates"?

DP backs up over the **full distribution** of next states (requires a model). TD backs up from a **single sampled** next state (no model needed) — hence "sample update."

### Connections

- **To DP:** same bootstrapping idea (update using $V(S_{t+1})$), but TD does not need the transition model.
- **To MC:** same model-free, experience-driven idea, but TD does not have to wait for an episode to end.
- **Bridge (coming up):** Chapter 7 ($n$-step TD) and Chapter 12 ($TD(\lambda)$) interpolate continuously between TD(0) and MC.