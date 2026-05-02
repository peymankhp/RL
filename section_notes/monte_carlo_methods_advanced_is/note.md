A potential problem is that this method learns only from the tails of episodes, when all of the remaining actions in the episode are greedy. If nongreedy actions are common, then learning will be slow, particularly for states appearing in the early portions of long episodes. Potentially, this could greatly slow learning. There has been insuffcient experience with off-policy Monte Carlo methods to assess how serious this problem is. If it is serious, the most important way to address it is probably by incorporating temporal- difference learning. Alternatively, if $\gamma$ is less than 1, then the idea developed in the following Discounting-aware Importance Sampling may also help significantly.

## *Per-decision Importance Sampling

**Plain English.** Look at the ordinary importance-sampling return $\rho_{t:T-1}\,G_t$. Every reward in the return — including ones very early in the episode — gets scaled by the ***entire*** product of policy ratios out to the end. But the reward $R_{t+k}$ only depends on what happened up through step $t+k-1$. Ratios from *later* steps are independent noise with expected value 1: they don't bias the estimator but they massively inflate its variance. **Per-decision IS** scales each reward by only the ratios needed to reach it — no more.

### The key identity

Write out one term of the ordinary-IS return:

$$
\rho_{t:T-1}\,R_{t+1} \;=\; \frac{\pi(A_t|S_t)\,\pi(A_{t+1}|S_{t+1})\,\pi(A_{t+2}|S_{t+2})\cdots\pi(A_{T-1}|S_{T-1})}{b(A_t|S_t)\,b(A_{t+1}|S_{t+1})\,b(A_{t+2}|S_{t+2})\cdots b(A_{T-1}|S_{T-1})}\,R_{t+1}. \tag{5.12}
$$

Only the factor at time $t$ is correlated with $R_{t+1}$ — the later factors are events that happen ***after*** the reward. Under the behavior policy, each of those later factors has expectation exactly 1:

$$
\mathbb{E}\!\left[\tfrac{\pi(A_k|S_k)}{b(A_k|S_k)}\right] \;=\; \sum_a b(a|S_k)\,\tfrac{\pi(a|S_k)}{b(a|S_k)} \;=\; \sum_a \pi(a|S_k) \;=\; 1. \tag{5.13}
$$

A few algebraic steps collapse all the extraneous factors and give the crucial simplification:

$$
\mathbb{E}\!\left[\rho_{t:T-1}\,R_{t+k}\right] \;=\; \mathbb{E}\!\left[\rho_{t:t+k-1}\,R_{t+k}\right]. \tag{5.14}
$$

**Reading it in English.** *To preserve the expectation, each reward only needs the ratio up to when it was generated.* Everything after is wasted motion that adds variance.

### The per-decision return

Replace the return $\rho_{t:T-1}\,G_t$ with a new quantity whose $k$-th reward is scaled by the ***right-sized*** ratio:

$$
\tilde{G}_t \;\doteq\; \rho_{t:t}\,R_{t+1} \;+\; \gamma\,\rho_{t:t+1}\,R_{t+2} \;+\; \gamma^2\,\rho_{t:t+2}\,R_{t+3} \;+\;\cdots\;+\;\gamma^{T-t-1}\rho_{t:T-1}\,R_T.
$$

The ***per-decision ordinary importance-sampling estimator*** is then

$$
V(s) \;\doteq\; \frac{\sum_{t\in\mathcal{T}(s)} \tilde{G}_t}{|\mathcal{T}(s)|}. \tag{5.15}
$$

By (5.14) it has the **same expectation** as the ordinary estimator (5.5) in the first-visit case — but its variance is generally lower because each sub-term is scaled by a shorter ratio product.

> ***Caveat.*** There is no known ***weighted*** per-decision estimator that is consistent (converges to the true value with infinite data). So per-decision IS pairs only with the ordinary form, not the weighted form of (5.6).
> 

### Minimal computation of $\tilde{G}_t$

Given one episode and per-step ratios $r_k = \pi(A_k|S_k)/b(A_k|S_k)$, compute all the per-decision returns in one pass:

```python
import numpy as np

def per_decision_returns(rewards, ratios, gamma=1.0):
    """
    rewards[k] = R_{k+1}                         (length T)
    ratios[k]  = pi(A_k|S_k) / b(A_k|S_k)        (length T)
    Returns G_tilde[t] for t = 0, ..., T-1.
    """
    T = len(rewards)
    G_tilde = np.zeros(T)
    for t in range(T):
        running_rho = 1.0   # will become rho_{t:k}
        G = 0.0
        for k in range(t, T):
            running_rho *= ratios[k]
            G += (gamma ** (k - t)) * running_rho * rewards[k]
        G_tilde[t] = G
    return G_tilde
```

**What to notice.** Each reward $R_{k+1}$ is multiplied by exactly $\rho_{t:k}$ — the ratios up to when it was generated, and no farther.

### Recall

#### Q1. Why is scaling $R_{t+1}$ by the *full* product $\rho_{t:T-1}$ wasteful?

The factors for steps ***after*** $t$ are independent of $R_{t+1}$ and have expected value 1. They don't change the expectation, but they blow up the variance — potentially to infinity.

#### Q2. What is the per-decision return $\tilde{G}_t$ in one sentence?

A return in which each reward $R_{t+k}$ is scaled only by $\rho_{t:t+k-1}$ — exactly the ratios for the transitions that occurred *before* or *at* that reward.

#### Q3. Does per-decision IS have a weighted version?

No consistent one is known. Per-decision IS currently works only with the ordinary (not weighted) estimator.

#### Q4. When does per-decision IS matter most?

When episodes are long and/or the per-step ratios have high variance. The ordinary estimator multiplies *every* reward by *all* ratios, so the variance blows up. Per-decision cuts each product down to the minimum needed.

## *Discounting-aware Importance Sampling

**Plain English.** When $\gamma$ is much less than 1, the return is dominated by the first few rewards — everything deep in the future barely counts. But ordinary importance sampling still scales the whole return by a product of ratios over *every* step, turning the estimator into a noise amplifier. **Discounting-aware IS** uses the structure of $\gamma$ to chop the importance-sampling product where discounting has already decided the reward doesn't matter.

### Motivating pathology

Suppose episodes last 100 steps and $\gamma = 0$. Then $G_0 = R_1$ — only the first reward contributes to the return. Yet ordinary IS scales $R_1$ by

$$
\rho_{0:99} \;=\; \frac{\pi(A_0|S_0)}{b(A_0|S_0)}\,\frac{\pi(A_1|S_1)}{b(A_1|S_1)}\,\cdots\,\frac{\pi(A_{99}|S_{99})}{b(A_{99}|S_{99})},
$$

a product of 100 factors. The extra 99 factors are independent of $R_1$ and each has expectation 1, so they don't bias the estimate — but they can make the variance **infinite**.

### Discounting as partial termination

The clever reinterpretation: view $\gamma$ as the **probability of *not* terminating at each step** (equivalently $1-\gamma$ is the per-step probability of termination). Then the return is a mixture of truncated returns, weighted by *when termination happened*:

- Terminates after step 1 with weight $(1-\gamma)$ → return is $R_{t+1}$.
- Terminates after step 2 with weight $(1-\gamma)\gamma$ → return is $R_{t+1}+R_{t+2}$.
- Terminates after step $h-t$ with weight $(1-\gamma)\gamma^{h-t-1}$ → return is $R_{t+1}+\cdots+R_h$.

These undiscounted partial sums are called ***flat partial returns***:

$$
\bar{G}_{t:h} \;\doteq\; R_{t+1} + R_{t+2} + \cdots + R_h, \qquad 0 \le t < h \le T.
$$

**"Flat"** because there is no $\gamma$ *inside* the sum (the weighting by $\gamma^{h-t-1}$ lives in the mixture, not in the sum). **"Partial"** because it stops at horizon $h$, not termination.

The full return then decomposes as:

$$
G_t \;=\; (1-\gamma)\!\!\sum_{h=t+1}^{T-1}\!\!\gamma^{h-t-1}\,\bar{G}_{t:h} \;+\; \gamma^{T-t-1}\,\bar{G}_{t:T}.
$$

### Truncated importance-sampling estimators

Since the flat partial return $\bar{G}_{t:h}$ only involves rewards up to horizon $h$, we only need the ratio $\rho_{t:h-1}$ — **not** the full $\rho_{t:T-1}$. Substituting and averaging over visits to state $s$ gives the ***ordinary discounting-aware estimator***:

$$
V(s) \;\doteq\; \frac{\sum_{t\in\mathcal{T}(s)}\!\Big[(1-\gamma)\!\sum_{h=t+1}^{T(t)-1}\!\gamma^{h-t-1}\rho_{t:h-1}\bar{G}_{t:h} \;+\; \gamma^{T(t)-t-1}\rho_{t:T(t)-1}\bar{G}_{t:T(t)}\Big]}{|\mathcal{T}(s)|}, \tag{5.9}
$$

and the ***weighted discounting-aware estimator*** with the same numerator but with the analogous mixture of ratios in the denominator (equation 5.10 in the book).

**Intuition to keep.** The farther in the future a reward is, the more it is discounted in $G_t$ — ***and*** the shorter the ratio product needed to scale it correctly. Discounting-aware IS lines those two facts up so each sliced-off portion of the return is scaled by only the ratios it actually needs.

> ***Boundary check.*** If $\gamma = 1$, the mixture weight $(1-\gamma)$ vanishes on every interior term, leaving only $\bar{G}_{t:T(t)}$ scaled by the full $\rho_{t:T(t)-1}$. The estimator collapses back to the ordinary (5.5) and weighted (5.6) estimators of §5.5. **No discounting → no extra leverage.**
> 

### Recall

#### Q1. What is the "discounting as termination" interpretation?

Treat $1-\gamma$ as the probability of termination at each step. The full return is then a mixture of ***flat partial returns*** $\bar{G}_{t:h}$, each weighted by the probability that termination happened exactly at horizon $h$.

#### Q2. What is a flat partial return, and why "flat"?

$\bar{G}_{t:h} = R_{t+1}+\cdots+R_h$: an undiscounted sum of rewards up to a horizon $h$. "Flat" because there is no $\gamma$ inside the sum — the weighting by $(1-\gamma)\gamma^{h-t-1}$ lives *outside*, in the mixture.

#### Q3. Why does this reduce variance when $\gamma$ is small?

When $\gamma$ is small, most of the mixture weight sits on short horizons. Short horizons need short ratio products. So most of the work is done with low-variance factors, avoiding the variance explosion of a full $\rho_{t:T-1}$.

#### Q4. What happens when $\gamma = 1$?

All mixture weight goes to the single full-length flat partial return $\bar{G}_{t:T}$, scaled by the full product $\rho_{t:T-1}$. The discounting-aware estimators collapse exactly to the ordinary/weighted IS estimators from §5.5.

#### Q5. How does this pair with per-decision IS?

They attack different sources of variance. **Per-decision** helps even when $\gamma = 1$ — by not scaling each reward by *later* ratios. **Discounting-aware** helps when $\gamma < 1$ — by not scaling each reward by ratios beyond where discounting has already killed its weight. In principle both ideas can be combined.

### Connections (both sections)

- **Same disease, different cure.** Both sections attack the same pathology: ordinary off-policy IS scales quantities by ratio products that are *too long*, injecting variance without improving the expectation.
- **Per-decision IS** exploits the structure of $G_t$ as a ***sum over rewards***: each $R_{t+k}$ only depends on ratios up to $t+k-1$.
- **Discounting-aware IS** exploits the structure of $G_t$ as a ***mixture over horizons***: each partial horizon only needs ratios up to that horizon.
- **Neither is free.** Both add implementation complexity, and the weighted per-decision form is still an open problem.
- **Looking ahead.** The next chapter introduces **Temporal-Difference learning**, which offers yet another way to attack the variance of MC — by bootstrapping rather than sampling full returns. TD and IS are complementary variance-reduction levers.
- Learns from **real or simulated experience**
- Does **not require full knowledge of the environment**
- Used only for **episodic tasks** (with a clear end)
- Estimates values by **averaging returns over multiple trials**
- Based on **sampling instead of full probability distributions**
- A **statistical method** using observed data (samples)
- Follows **General Policy Iteration (GPI)**
    - First: **policy evaluation (prediction)**
    - Then: **policy improvement**
    - Repeats until reaching an **optimal policy**