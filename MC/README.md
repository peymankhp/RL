# 🎲 Monte Carlo RL Explorer

An interactive visual mini-textbook for all major Monte Carlo methods in Reinforcement Learning.

## Methods Covered

| Method | Type | Key property |
|--------|------|-------------|
| First-Visit MC Prediction | On-policy | Unbiased, low variance |
| Every-Visit MC Prediction | On-policy | More data, slight bias |
| MC Control (ε-greedy) | On-policy | GPI, ε-soft policy |
| Ordinary Importance Sampling | Off-policy | Unbiased, HIGH variance |
| Weighted Importance Sampling | Off-policy | Biased, LOW variance |
| Incremental MC | On-policy | O(1) memory, online updates |
| Per-Decision IS | Off-policy | Lower variance than WIS |
| Discounting-Aware IS | Off-policy | Lowest variance of all |

## Environment: 5×5 Gridworld

```
(0,0) . . . .
 . ■  . ■  .
 . . ✗ . .
 . ■  . ■  .
 . . . . ★
```
- ★ Goal (4,4): +10
- ✗ Trap (2,2): −5
- ■ Wall: agent bounces
- Step cost: −0.1
- Configurable stochastic slip

## Run

```bash
pip install -r requirements.txt
streamlit run mc_rl_explorer.py
```

## App Tabs

1. **Environment** — Gridworld layout + sample episode
2. **MC Prediction** — First-Visit vs Every-Visit heatmaps + convergence
3. **On-policy Control** — Q-values, policy arrows, learning curve
4. **Off-policy IS** — Ordinary vs Weighted IS comparison
5. **Incremental MC** — Online update, variance over time
6. **Advanced IS** — Per-Decision and Discounting-Aware IS
7. **Dashboard** — All 8 methods side-by-side + bias-variance landscape
8. **Method Guide** — Detailed explanation of every method
