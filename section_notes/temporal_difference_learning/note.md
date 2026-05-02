# Temporal-Difference Learning

If one had to identify **one idea central and novel to reinforcement learning**, it would undoubtedly be ***temporal-difference (TD) learning***. TD is a hybrid: it learns directly from raw experience (like **Monte Carlo**) but updates estimates using other learned estimates, without waiting for the final outcome (like **Dynamic Programming**) — this is called ***bootstrapping***.

As in the last two chapters, we start with the **prediction problem** (estimating $v_\pi$ for a given policy $\pi$). The differences between MC, TD, and DP are mostly differences in how they attack prediction; for the control problem they all use a variation of **generalized policy iteration (GPI)**.