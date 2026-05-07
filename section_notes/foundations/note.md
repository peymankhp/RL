Here is the comprehensive explanation of these Reinforcement Learning (RL) concepts in English, following the same structure and depth.

---

## 1. Markov Decision Process (MDP)

**The Formula:**
$$P(s', r | s, a) = \Pr\{S_t = s', R_t = r | S_{t-1} = s, A_{t-1} = a\}$$

*   $s$: The current state.
*   $a$: The action taken by the agent.
*   $s'$: The next state the agent transitions into.
*   $r$: The immediate reward received after taking action $a$.
*   $P$: The transition probability (the "dynamics" of the environment).

---

## 2. Bellman Equation

**The Formula (Value Function):**
$$V(s) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]$$

*   $V(s)$: The value of the current state.
*   $\gamma$ (Gamma): The discount factor (0 to 1), which determines how much we care about future rewards compared to immediate ones.
*   $\max_a$: Represents choosing the action that yields the highest possible total value.

---

## 3. Dynamic Programming (DP)

**The Formula (Policy Evaluation Update):**
$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V_k(s')]$$

*   $V_{k+1}$: The updated value estimate in the next iteration.
*   $\pi(a|s)$: The policy (the probability of taking action $a$ given state $s$).
*   $V_k$: The old value estimate from the previous iteration.

---

## 4. Monte Carlo (MC)


**The Formula:**
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

*   $G_t$: The actual total return (sum of all rewards) received from time $t$ until the end of the episode.
*   $\alpha$ (Alpha): The learning rate (how much we overwrite our old belief with new data).
*   $G_t - V(S_t)$: The "error" or difference between what we expected and what actually happened.

---

## 5. Temporal Difference (TD) Learning


**The Formula:**
$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

*   $R_{t+1} + \gamma V(S_{t+1})$: The "TD Target" (our new, slightly more accurate estimate of the value).
*   $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$: The "TD Error" (the surprise we felt when we saw the actual reward and the next state).
*   $V(S_{t+1})$: The estimated value of the next state we just landed in.