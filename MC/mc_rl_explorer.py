"""
Monte Carlo RL Explorer — Streamlit App
A visual, interactive mini-textbook for all major MC methods in RL.
Environment: 5×5 Gridworld (No Blackjack)
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Monte Carlo RL Explorer",
    layout="wide",
    page_icon="🎲",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Dark background for code blocks */
.stCode { background: #1e1e2e !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #12121f;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: #1e1e2e;
    border-radius: 8px;
    color: #b0b0cc;
    padding: 8px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#7c4dff,#00bcd4);
    color: white !important;
}

/* Info boxes */
div[data-testid="stInfo"] {
    background: #1a237e22;
    border-left: 4px solid #7c4dff;
}
div[data-testid="stSuccess"] {
    background: #1b5e2022;
    border-left: 4px solid #4caf50;
}

/* Metrics */
div[data-testid="metric-container"] {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid #2d2d44;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GRIDWORLD ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

class GridWorld:
    """
    5×5 stochastic Gridworld.
    - Start : (0,0) top-left
    - Goal  : (4,4) → +10 reward, terminal
    - Trap  : (2,2) → −5 reward, terminal
    - Walls : {(1,1),(1,3),(3,1),(3,3)} — agent bounces back
    - Step  : −0.1 per move (encourages shorter paths)
    - Slip  : with prob `slip_prob` the intended action is replaced by a random one
    """

    ACTIONS = [0, 1, 2, 3]               # up, right, down, left
    SYMBOLS  = ["↑", "→", "↓", "←"]
    DELTAS   = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, size: int = 5, slip_prob: float = 0.1):
        self.size = size
        self.slip_prob = slip_prob
        self.start = (0, 0)
        self.goal  = (4, 4)
        self.trap  = (2, 2)
        self.walls = frozenset([(1,1),(1,3),(3,1),(3,3)])
        self.n_states  = size * size
        self.n_actions = 4

    # ── helpers ───────────────────────────────────────────────────────────────

    def s2i(self, s): return s[0] * self.size + s[1]
    def i2s(self, i): return (i // self.size, i % self.size)
    def is_terminal(self, s): return s in (self.goal, self.trap)

    # ── dynamics ──────────────────────────────────────────────────────────────

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0.0, True

        # stochastic slip
        if np.random.random() < self.slip_prob:
            a = np.random.randint(4)

        dr, dc = self.DELTAS[a]
        ns = (s[0] + dr, s[1] + dc)

        # clamp to grid + bounce off walls
        if not (0 <= ns[0] < self.size and 0 <= ns[1] < self.size):
            ns = s
        if ns in self.walls:
            ns = s

        if   ns == self.goal: r = 10.0
        elif ns == self.trap:  r = -5.0
        else:                  r = -0.1

        return ns, r, self.is_terminal(ns)

    # ── episode generation ────────────────────────────────────────────────────

    def generate_episode(self, policy, max_steps=200):
        """
        policy: dict {state -> action_int} OR {state -> prob_array}
        Returns list of (state, action, reward) tuples.
        """
        episode = []
        s = self.start
        for _ in range(max_steps):
            p = policy[s]
            a = int(p) if np.isscalar(p) else int(np.random.choice(4, p=p))
            ns, r, done = self.step(s, a)
            episode.append((s, a, r))
            s = ns
            if done:
                break
        return episode

    # ── standard policies ─────────────────────────────────────────────────────

    def uniform_policy(self):
        """Completely random — used as behavior policy."""
        return {self.i2s(i): np.ones(4) / 4 for i in range(self.n_states)}

    def eps_greedy_policy(self, Q, eps):
        """ε-greedy over Q(s,·)."""
        policy = {}
        for i in range(self.n_states):
            s = self.i2s(i)
            probs = np.full(4, eps / 4)
            probs[np.argmax(Q[s])] += 1.0 - eps
            policy[s] = probs
        return policy

    def greedy_policy(self, Q):
        """Deterministic greedy (target policy in off-policy experiments)."""
        return {self.i2s(i): int(np.argmax(Q[self.i2s(i)])) for i in range(self.n_states)}


# ─────────────────────────────────────────────────────────────────────────────
# MC PREDICTION  (Section 5.1, Sutton & Barto)
# ─────────────────────────────────────────────────────────────────────────────

def mc_first_visit(env: GridWorld, policy, n_episodes: int, gamma: float):
    """
    First-Visit MC Prediction.
    V(s) = mean of G_t for the FIRST time s appears each episode.
    Unbiased, statistically independent samples.
    """
    V       = defaultdict(float)
    returns = defaultdict(list)
    history = []   # periodic snapshots for convergence plots

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        G = 0.0
        visited = set()

        for s, a, r in reversed(episode):
            G = gamma * G + r
            if s not in visited:           # ← only first visit
                visited.add(s)
                returns[s].append(G)
                V[s] = float(np.mean(returns[s]))

        if (ep + 1) % max(1, n_episodes // 20) == 0:
            history.append(dict(V))

    return V, history


def mc_every_visit(env: GridWorld, policy, n_episodes: int, gamma: float):
    """
    Every-Visit MC Prediction.
    V(s) = mean of G_t for EVERY occurrence of s in an episode.
    More data, slight correlation bias, faster practical convergence.
    """
    V       = defaultdict(float)
    returns = defaultdict(list)
    history = []

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        G = 0.0

        for s, a, r in reversed(episode):
            G = gamma * G + r
            returns[s].append(G)           # ← every occurrence
            V[s] = float(np.mean(returns[s]))

        if (ep + 1) % max(1, n_episodes // 20) == 0:
            history.append(dict(V))

    return V, history


# ─────────────────────────────────────────────────────────────────────────────
# MC CONTROL — ON-POLICY  (Section 5.4)
# ─────────────────────────────────────────────────────────────────────────────

def mc_control_on_policy(env: GridWorld, n_episodes: int, eps: float, gamma: float):
    """
    On-policy ε-greedy MC Control (GPI every episode).
    Estimates Q(s,a) and improves the policy simultaneously.
    Converges to best ε-soft policy.
    """
    Q       = defaultdict(lambda: np.zeros(4))
    returns = defaultdict(list)
    policy  = env.uniform_policy()

    episode_rewards = []

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        total_r = sum(r for _, _, r in episode)
        episode_rewards.append(total_r)

        G = 0.0
        visited_sa = set()
        for s, a, r in reversed(episode):
            G = gamma * G + r
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                returns[(s, a)].append(G)
                Q[s][a] = float(np.mean(returns[(s, a)]))

        # Policy improvement — rebuild ε-greedy from updated Q
        policy = env.eps_greedy_policy(Q, eps)

    return Q, policy, episode_rewards


# ─────────────────────────────────────────────────────────────────────────────
# OFF-POLICY EVALUATION — Ordinary IS  (Section 5.5)
# ─────────────────────────────────────────────────────────────────────────────

def _is_ratio(target_policy, behavior_policy, s, a):
    """Compute π(a|s) / b(a|s) safely."""
    b = behavior_policy[s][a] if isinstance(behavior_policy[s], np.ndarray) else float(behavior_policy[s] == a)
    if isinstance(target_policy[s], np.ndarray):
        t = target_policy[s][a]
    else:
        t = 1.0 if int(target_policy[s]) == a else 0.0
    return t / b if b > 1e-12 else 0.0


def mc_ordinary_is(env: GridWorld, target_policy, behavior_policy, n_episodes: int, gamma: float):
    """
    Ordinary (simple) Importance Sampling off-policy evaluation.
    V(s) = mean(ρ·G) where ρ = ∏ π(a|s)/b(a|s)  — UNBIASED but high variance.
    """
    returns_weighted = defaultdict(list)
    V = defaultdict(float)

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior_policy)
        G = 0.0
        rho = 1.0
        visited = set()

        for s, a, r in reversed(episode):
            G   = gamma * G + r
            rho *= _is_ratio(target_policy, behavior_policy, s, a)
            if rho < 1e-10:
                break
            if s not in visited:
                visited.add(s)
                returns_weighted[s].append(rho * G)
                V[s] = float(np.mean(returns_weighted[s]))

    return V


def mc_weighted_is(env: GridWorld, target_policy, behavior_policy, n_episodes: int, gamma: float):
    """
    Weighted Importance Sampling off-policy evaluation.
    Uses running weighted average: V(s) += W/C · (G − V(s))
    Biased but dramatically lower variance than ordinary IS.
    """
    V = defaultdict(float)
    C = defaultdict(float)   # cumulative IS weights

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior_policy)
        G = 0.0
        W = 1.0
        visited = set()

        for s, a, r in reversed(episode):
            G  = gamma * G + r
            W *= _is_ratio(target_policy, behavior_policy, s, a)
            if W < 1e-10:
                break
            if s not in visited:
                visited.add(s)
                C[s] += W
                V[s] += (W / C[s]) * (G - V[s])

    return V


# ─────────────────────────────────────────────────────────────────────────────
# OFF-POLICY MC CONTROL  (Section 5.7, Sutton & Barto)
# ─────────────────────────────────────────────────────────────────────────────

def mc_off_policy_control(env: GridWorld, n_episodes: int, eps_behavior: float, gamma: float):
    """
    Off-policy MC Control with weighted IS — the textbook algorithm from S&B Ch.5.

    Target policy  π : deterministic greedy w.r.t. Q  → can converge to true π*
    Behavior policy b : ε-soft (guarantees coverage of all state-action pairs)

    Key update rule (incremental weighted IS for Q):
        C(s,a) ← C(s,a) + W
        Q(s,a) ← Q(s,a) + W/C(s,a) · [G − Q(s,a)]

    CRITICAL: we only update states from the END of the episode (greedy tail).
    As soon as a non-greedy action is encountered walking backward, we STOP —
    because W = π(a|s)/b(a|s) = 0 for a non-greedy action under deterministic π.
    """
    Q = defaultdict(lambda: np.zeros(4))
    C = defaultdict(lambda: np.zeros(4))   # per (s,a) cumulative IS weight

    episode_rewards    = []
    greedy_len_history = []   # fraction of episode covered by greedy tail

    for ep in range(n_episodes):
        # Rebuild ε-soft behavior policy from current Q each episode
        b_policy = env.eps_greedy_policy(Q, eps_behavior)
        episode  = env.generate_episode(b_policy)

        total_r = sum(r for _, _, r in episode)
        episode_rewards.append(total_r)

        G = 0.0
        W = 1.0
        greedy_steps = 0

        # Walk backward through the episode
        for s, a, r in reversed(episode):
            G = gamma * G + r

            # Incremental weighted-IS update for Q(s,a)
            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])

            # Greedy action under current Q
            a_star = int(np.argmax(Q[s]))

            if a != a_star:
                # Non-greedy action → importance weight for all earlier steps = 0 → stop
                break

            greedy_steps += 1
            # π(a|s) = 1 (deterministic greedy), so ρ = 1 / b(a|s)
            b_prob = float(b_policy[s][a])
            W *= 1.0 / max(b_prob, 1e-12)

            if W > 1e8:   # numerical safety cap
                break

        greedy_len_history.append(greedy_steps / max(len(episode), 1))

    # Extract final deterministic target policy
    pi_star = {env.i2s(i): int(np.argmax(Q[env.i2s(i)])) for i in range(env.n_states)}
    V_star  = {s: float(np.max(Q[s])) for s in Q}

    return Q, pi_star, V_star, episode_rewards, greedy_len_history


# ─────────────────────────────────────────────────────────────────────────────
# INCREMENTAL MC  (Section 5.6)
# ─────────────────────────────────────────────────────────────────────────────

def mc_incremental(env: GridWorld, policy, n_episodes: int, gamma: float):
    """
    Incremental (online) First-Visit MC.
    V(s) ← V(s) + (1/N(s)) · (G − V(s))
    Memory-efficient, equivalent to batch averaging.
    Tracks per-episode mean variance as a convergence signal.
    """
    V      = defaultdict(float)
    N      = defaultdict(int)
    val_log = defaultdict(list)   # for variance tracking
    var_history = []

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        G = 0.0
        visited = set()

        for s, a, r in reversed(episode):
            G = gamma * G + r
            if s not in visited:
                visited.add(s)
                N[s] += 1
                V[s] += (G - V[s]) / N[s]   # incremental mean
                val_log[s].append(V[s])

        if (ep + 1) % max(1, n_episodes // 20) == 0:
            # Mean variance across all visited states
            var_history.append(
                float(np.mean([np.var(val_log[s]) for s in val_log if len(val_log[s]) > 1]))
            )

    return V, var_history


# ─────────────────────────────────────────────────────────────────────────────
# PER-DECISION IMPORTANCE SAMPLING  (Section 5.8)
# ─────────────────────────────────────────────────────────────────────────────

def mc_per_decision_is(env: GridWorld, target_policy, behavior_policy, n_episodes: int, gamma: float):
    """
    Per-Decision Importance Sampling.
    Each reward R_{k+1} is weighted by the IS ratio only up to step k,
    not the full episode ratio.  This provably reduces variance vs ordinary IS.

    V(s_t) = Σ_{k=t}^{T} γ^{k-t} · (Π_{j=t}^{k} ρ_j) · R_{k+1}
    """
    returns = defaultdict(list)
    V = defaultdict(float)

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior_policy)
        T = len(episode)

        for t in range(T):
            s_t = episode[t][0]
            G_pd = 0.0
            rho_cumulative = 1.0
            valid = True

            for k in range(t, T):
                sk, ak, rk = episode[k]
                ratio = _is_ratio(target_policy, behavior_policy, sk, ak)
                rho_cumulative *= ratio
                if rho_cumulative < 1e-10:
                    valid = False
                    break
                G_pd += (gamma ** (k - t)) * rho_cumulative * rk

            if valid:
                returns[s_t].append(G_pd)

        for s in set(ep[0] for ep in episode):
            if returns[s]:
                V[s] = float(np.mean(returns[s]))

    return V


# ─────────────────────────────────────────────────────────────────────────────
# DISCOUNTING-AWARE IMPORTANCE SAMPLING  (Section 5.9)
# ─────────────────────────────────────────────────────────────────────────────

def mc_discounting_aware_is(env: GridWorld, target_policy, behavior_policy, n_episodes: int, gamma: float):
    """
    Discounting-Aware IS.
    Exploits the fact that a discounted return can be written as a weighted sum
    of "flat" partial returns.  For γ < 1, this reduces variance beyond per-decision IS
    because future rewards contribute less and their IS ratios are downweighted.

    Uses incremental weighted-IS update with discounting-adjusted weight φ.
    """
    V = defaultdict(float)
    C = defaultdict(float)

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior_policy)
        T = len(episode)

        for t in range(T):
            s_t = episode[t][0]
            G  = 0.0
            W  = 1.0
            phi = 0.0     # discounting-aware cumulative weight

            for k in range(t, T):
                sk, ak, rk = episode[k]
                ratio = _is_ratio(target_policy, behavior_policy, sk, ak)
                W   *= ratio
                if W < 1e-10:
                    break
                gk   = gamma ** (k - t)
                # Decompose: each step contributes (1−γ)·γ^{k-t} of the full IS weight
                phi += (1.0 - gamma) * gk * W
                G   += gk * rk

            # Terminal contribution
            phi += (gamma ** (T - t)) * W

            if phi > 1e-12:
                C[s_t]  += phi
                V[s_t]  += (phi / C[s_t]) * (G - V[s_t])

    return V


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#0d0d1a"
CARD_BG    = "#12121f"
GRID_COLOR = "#2a2a3e"

# Custom diverging colormap: red (low) → yellow (mid) → green (high)
RL_CMAP = LinearSegmentedColormap.from_list(
    "rl", ["#b71c1c", "#f57f17", "#fff176", "#2e7d32"]
)


def _fig_style(fig, axes_flat):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_flat:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="#9e9ebb", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)


def plot_value_heatmap(env: GridWorld, V: dict, title: str, ax):
    """Render a V(s) heatmap on *ax* with wall/goal/trap annotations."""
    grid = np.full((env.size, env.size), np.nan)
    for i in range(env.n_states):
        s = env.i2s(i)
        if s not in env.walls:
            grid[s] = V.get(s, 0.0)

    # Mask walls
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=RL_CMAP, vmin=-5, vmax=10, aspect="equal")

    # Wall patches
    for (wr, wc) in env.walls:
        ax.add_patch(plt.Rectangle((wc-0.5, wr-0.5), 1, 1, color="#1a1a2e", zorder=3))
        ax.text(wc, wr, "■", ha="center", va="center", color="#555577", fontsize=12, zorder=4)

    # Annotations
    for i in range(env.size):
        for j in range(env.size):
            s = (i, j)
            if s in env.walls:
                continue
            val = V.get(s, 0.0)
            txt_color = "white" if abs(val) > 3 else "black"
            if s == env.goal:
                ax.text(j, i, "★GOAL\n+10", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", zorder=5)
            elif s == env.trap:
                ax.text(j, i, "✗TRAP\n−5", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", zorder=5)
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_color, zorder=5)

    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors="#9e9ebb", labelsize=7)


def plot_policy_arrows(env: GridWorld, policy: dict, title: str, ax):
    """Render deterministic policy arrows on *ax*."""
    # Arrow direction: (Δcol, Δrow) in display coords (y increases downward → flip row)
    arrow_map = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}

    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(-0.5, env.size - 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor(DARK_BG)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.grid(color=GRID_COLOR, linewidth=0.5)

    for i in range(env.size):
        for j in range(env.size):
            s = (i, j)
            if s in env.walls:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#1a1a2e", zorder=3))
                ax.text(j, i, "■", ha="center", va="center", color="#555577", fontsize=12, zorder=4)
            elif s == env.goal:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#1b5e20", alpha=0.8, zorder=2))
                ax.text(j, i, "★\nGOAL", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
            elif s == env.trap:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#b71c1c", alpha=0.8, zorder=2))
                ax.text(j, i, "✗\nTRAP", ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)
            else:
                p = policy.get(s, 0)
                a = int(np.argmax(p)) if isinstance(p, np.ndarray) else int(p)
                dc, dr = arrow_map[a]
                color = "#1565c0" if s == env.start else "#7c4dff"
                ax.annotate("", xy=(j + dc, i + dr), xytext=(j, i),
                            arrowprops=dict(arrowstyle="->", color=color, lw=2.0), zorder=4)
                if s == env.start:
                    ax.text(j, i - 0.45, "S", ha="center", va="center",
                            color="#42a5f5", fontsize=7, fontweight="bold", zorder=5)

    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(colors="#9e9ebb", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COLOR)


def smooth(arr, w=30):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def make_fig(nrows=1, ncols=1, w=12, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    axlist = np.array(axes).flatten().tolist()
    _fig_style(fig, axlist)
    return fig, axes, axlist


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────

def run_all_methods(env, n_episodes, gamma, epsilon, seed):
    np.random.seed(seed)

    b_policy = env.uniform_policy()   # behavior policy (random)

    # 1. MC Prediction
    V_fv, hist_fv = mc_first_visit(env, b_policy, n_episodes, gamma)
    V_ev, hist_ev = mc_every_visit(env, b_policy, n_episodes, gamma)

    # 2. On-policy control → gives us a good Q to build a target policy
    Q_on, pi_on, ep_rewards_on = mc_control_on_policy(env, n_episodes, epsilon, gamma)
    V_on   = {s: float(np.max(Q_on[s])) for s in Q_on}
    pi_det = env.greedy_policy(Q_on)   # deterministic target policy for off-policy eval

    # 3. Off-policy IS evaluation (target=greedy from Q_on, behavior=random)
    V_ois = mc_ordinary_is(env, pi_det, b_policy, n_episodes, gamma)
    V_wis = mc_weighted_is(env, pi_det, b_policy, n_episodes, gamma)

    # 4. Off-policy MC Control (learns its OWN Q toward π*)
    Q_off, pi_off, V_off, ep_rewards_off, greedy_hist = mc_off_policy_control(
        env, n_episodes, epsilon, gamma)

    # 5. Incremental MC
    V_inc, var_hist = mc_incremental(env, b_policy, n_episodes, gamma)

    # 6. Advanced IS (cap at 500 for speed)
    n_adv  = min(n_episodes, 500)
    V_pd   = mc_per_decision_is(env, pi_det, b_policy, n_adv, gamma)
    V_da   = mc_discounting_aware_is(env, pi_det, b_policy, n_adv, gamma)

    return dict(
        V_fv=V_fv, hist_fv=hist_fv,
        V_ev=V_ev, hist_ev=hist_ev,
        Q_on=Q_on, V_on=V_on, pi_on=pi_on, pi_det=pi_det, ep_rewards_on=ep_rewards_on,
        V_ois=V_ois, V_wis=V_wis,
        Q_off=Q_off, pi_off=pi_off, V_off=V_off,
        ep_rewards_off=ep_rewards_off, greedy_hist=greedy_hist,
        V_inc=V_inc, var_hist=var_hist,
        V_pd=V_pd, V_da=V_da,
        b_policy=b_policy,
    )



def _card(color, icon, title, body):
    """Reusable coloured teaching card."""
    return f"""
    <div style="background:{color}18; border-left:4px solid {color};
                padding:1rem 1.2rem; border-radius:0 10px 10px 0; margin-bottom:1rem">
    <b>{icon} {title}</b><br>{body}
    </div>"""


def _tip(text):
    return f'<div style="background:#1a2a1a; border-left:3px solid #4caf50; padding:.6rem 1rem; border-radius:0 6px 6px 0; margin:.5rem 0; font-size:.93rem">{text}</div>'


def _warn(text):
    return f'<div style="background:#2a1a1a; border-left:3px solid #ef5350; padding:.6rem 1rem; border-radius:0 6px 6px 0; margin:.5rem 0; font-size:.93rem">{text}</div>'


def _formula_box(title, formula_md):
    """A highlighted formula box with title."""
    return f"""
    <div style="background:#1a1a30; border:1px solid #3a3a5e; border-radius:8px;
                padding:.8rem 1.2rem; margin:.6rem 0">
    <span style="color:#9c9cf0; font-size:.85rem; font-weight:700">{title}</span><br>
    {formula_md}
    </div>"""


def main():
    # ── Banner ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a237e,#4a148c,#006064);
                padding:2rem 2.5rem; border-radius:14px; margin-bottom:1.5rem">
        <h1 style="color:white;margin:0;font-size:2.4rem">🎲 Monte Carlo RL Explorer</h1>
        <p style="color:#b0bec5;margin-top:.5rem;font-size:1.05rem">
            An interactive visual textbook — 9 methods, every formula explained, every chart decoded
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Global intro ─────────────────────────────────────────────────────────
    with st.expander("🎓 New here? — What is Monte Carlo Reinforcement Learning?", expanded=False):
        st.markdown(r"""
        <div style="background:#12121f; border-radius:12px; padding:1.4rem 1.8rem; border:1px solid #2a2a3e">

        ### 🤔 The Core Problem

        Imagine a robot dropped into a maze it has never seen. It has no map, no instructions — only the
        ability to move and observe what happens. How does it learn to reach the exit?

        > **Reinforcement Learning (RL)** is the science of learning from experience through trial and error —
        > like a child learning to walk by falling and getting up again.

        ---

        ### 🎲 What Makes It "Monte Carlo"?

        The name comes from the Monaco casino — a symbol of randomness and probability.
        In RL, **Monte Carlo methods** learn by playing many complete games (called **episodes**),
        then looking backward to figure out which decisions were responsible for the outcome.

        **The chess analogy:** After finishing a game, a chess player replays every move and asks
        *"did this move lead to my win or loss?"* MC does the same — but with math, and thousands of games.

        ---

        ### 🔑 Three Concepts You Must Know

        | Concept | Plain English | Math symbol |
        |---------|--------------|-------------|
        | **Return G** | Total reward from a moment until the game ends — including future rewards, scaled by γ | $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$ |
        | **Value V(s)** | The *average* return you expect starting from position s — how good is it to be here? | $V(s) \approx \mathbb{E}[G_t \mid S_t = s]$ |
        | **Policy π** | The agent's decision rule — given position s, which action to take | $\pi(a \mid s) = \Pr[\text{take action } a \mid \text{in state } s]$ |

        ---

        ### 📍 The 9 Methods at a Glance

        | # | Method | The Question It Answers |
        |---|--------|------------------------|
        | 1 | **MC Prediction (First-Visit)** | How good is each state under a fixed policy? |
        | 2 | **MC Prediction (Every-Visit)** | Same, but use every visit — more data, slight correlation |
        | 3 | **On-policy MC Control** | What's the best action in each state — learn and improve simultaneously |
        | 4 | **Off-policy IS Evaluation** | Can I evaluate policy π using data collected by policy b? |
        | 5 | **Off-policy MC Control** | Can I find the true optimal π* while exploring with a different policy? |
        | 6 | **Incremental MC** | Can I update efficiently without storing every past return? |
        | 7 | **Per-Decision IS** | Can I cut IS variance by breaking the weight into per-step pieces? |
        | 8 | **Discounting-Aware IS** | Can I exploit γ < 1 to reduce variance even further? |

        > **All MC methods share three DNA strands:**  
        > ① Model-free — no knowledge of $p(s'|s,a)$ needed  
        > ② Episode-based — must wait until the game ends  
        > ③ Unbiased prediction — use real outcomes, never guesses

        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption("Change these and click **Run** to see how each method responds.")

        n_episodes = st.slider("Episodes", 100, 3000, 800, 100,
            help="One episode = one complete game (Start→Goal or Trap). More episodes = better estimates but slower. Try 200 for speed, 2000 for accuracy.")
        gamma = st.slider("Discount γ", 0.80, 1.00, 0.99, 0.01,
            help="How much future rewards matter. γ=0.99: reward 100 steps away is worth 0.99^100≈37% of immediate reward. γ=1.0: future matters as much as now.")
        epsilon = st.slider("ε (exploration rate)", 0.01, 0.50, 0.10, 0.01,
            help="Fraction of time the agent picks a RANDOM action instead of its best known one. ε=0.1 means 10% random. Too low = gets stuck; too high = never learns.")
        slip_prob = st.slider("Slip probability", 0.0, 0.30, 0.10, 0.05,
            help="Chance the environment ignores the chosen action and does something random. Models noisy motors, wind, or slippery floors.")
        seed = st.number_input("Random seed", 0, 9999, 42,
            help="Same seed = same random numbers = reproducible results. Change it to see a different run.")

        run_btn = st.button("🚀 Run All 9 Methods", type="primary", use_container_width=True)

        st.divider()
        st.markdown("""
        **5×5 Gridworld legend**
        | Symbol | Cell type |
        |--------|-----------|
        | ● | Start (0,0) |
        | ★ | Goal (4,4) +10 |
        | ✗ | Trap (2,2) −5 |
        | ■ | Wall (bounce back) |
        | — | Step penalty −0.1 |

        **9 Methods covered:**
        ```
        MC Prediction
          ├─ First-Visit
          └─ Every-Visit
        MC Control
          ├─ On-policy ε-greedy
          └─ Off-policy (target π*)
        Off-policy IS
          ├─ Ordinary IS
          └─ Weighted IS
        Incremental MC
        Advanced IS
          ├─ Per-Decision IS
          └─ Discounting-Aware IS
        ```
        """)

    env = GridWorld(size=5, slip_prob=slip_prob)

    # ── Tab definitions ───────────────────────────────────────────────────────
    tab_env, tab_pred, tab_ctrl, tab_offpol, tab_offctrl, tab_incr, tab_adv, tab_dash, tab_guide = st.tabs([
        "🗺️ Environment",
        "📊 MC Prediction",
        "🎯 On-policy Control",
        "⚖️ Off-policy IS",
        "🎲 Off-policy Control",
        "⚡ Incremental MC",
        "🔬 Advanced IS",
        "📈 Dashboard",
        "📚 Method Guide",
    ])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 0 — ENVIRONMENT
    # ═════════════════════════════════════════════════════════════════════════
    with tab_env:
        st.markdown(_card("#42a5f5","🗺️","Why do we need an environment?",
            """Every RL algorithm needs a world to act in. Our environment defines the <em>states</em>
            (positions), <em>actions</em> (moves), <em>rewards</em> (outcomes), and <em>terminal conditions</em>
            (when the game ends). All 9 MC methods in this app interact with the same 5×5 Gridworld —
            which lets you directly compare how differently each method responds to the <em>same</em> world."""),
            unsafe_allow_html=True)

        st.subheader("🗺️ The 5×5 Stochastic Gridworld")
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            st.markdown("""
            #### What is the agent's task?
            The agent starts at the **top-left corner (0,0)** and must navigate to the **bottom-right
            goal (4,4)** while avoiding the **trap at (2,2)**. Four walls block direct paths, forcing
            the agent to find alternative routes.

            #### Why is this environment good for MC?
            Three specific design choices make MC methods interesting here:

            **① Multiple paths to the goal** — The walls create at least two viable routes
            (top-right arc vs bottom-left arc). Different policies take different paths, and MC
            must learn which path is better purely from sampled experience.

            **② A dangerous trap** — Any method that doesn't learn to avoid (2,2) will
            collect strongly negative returns. This creates high variance in early episodes,
            which is exactly the phenomenon IS methods are designed to handle.

            **③ Stochastic slip** — With probability `slip_prob`, the environment *ignores*
            the agent's chosen action and picks a random one instead. This tests whether learned
            policies are robust to noise — critical for real-world robotics.

            #### Reward structure explained
            | Event | Reward | Why this value? |
            |-------|--------|-----------------|
            | Reach Goal ★ | **+10.0** | Large positive — the primary objective |
            | Reach Trap ✗ | **−5.0** | Negative but not catastrophic — avoidable risk |
            | Each step | **−0.1** | Small penalty — encourages shorter paths |
            | Wall bounce | **0.0** | No penalty — but you waste a step |

            The step penalty (−0.1) is crucial: without it, a wandering agent that eventually
            reaches the goal by luck would look as good as a smart agent. The penalty ensures
            *efficiency* matters, not just *success*.
            """)

        with c2:
            st.caption("📖 **How to read this map:** Row 0 is the top; row 4 is the bottom. The agent starts at (0,0) — top-left — and must reach (4,4) — bottom-right. Walls (■) are impassable: the agent bounces back to its previous position.")
            fig, ax, _ = make_fig(1, 1, 5, 5)
            ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal")
            ax.set_facecolor(DARK_BG)
            color_map = {"normal":"#1e1e2e","wall":"#252538","goal":"#1b5e20","trap":"#b71c1c","start":"#0d47a1"}
            for i in range(5):
                for j in range(5):
                    s = (i, j)
                    if   s in env.walls: c, lbl = color_map["wall"],  "■"
                    elif s == env.goal:  c, lbl = color_map["goal"],  "★GOAL\n+10"
                    elif s == env.trap:  c, lbl = color_map["trap"],  "✗TRAP\n−5"
                    elif s == env.start: c, lbl = color_map["start"], f"●START\n({i},{j})"
                    else:                c, lbl = color_map["normal"], f"({i},{j})"
                    ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,color=c,ec="#2a2a3e",lw=1.5))
                    ax.text(j, i, lbl, ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.tick_params(colors="#9e9ebb")
            ax.invert_yaxis()
            ax.set_title("Gridworld Layout", color="white", fontweight="bold")
            for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        st.subheader("🎬 What does one episode look like?")
        st.markdown("""
        An **episode** is one complete game — from Start to a terminal state. MC methods are
        fundamentally episode-based: they cannot update any estimate until the game is over,
        because they need the *final total reward* G to learn from.

        This is the key difference from **TD (Temporal-Difference) methods** like Q-learning,
        which can update after *every single step* using a partial estimate. MC waits for the
        truth; TD guesses along the way.
        """)
        np.random.seed(seed)
        ep_sample = env.generate_episode(env.uniform_policy(), max_steps=50)
        path_str  = " → ".join(f"{s}" for s, a, r in ep_sample)
        total_r   = sum(r for _, _, r in ep_sample)
        acts_str  = " ".join(env.SYMBOLS[a] for _, a, _ in ep_sample)
        rew_str   = " ".join(f"{r:+.1f}" for _, _, r in ep_sample)

        st.caption("📖 **Three rows below:** 'Path' = grid squares visited. 'Actions' = moves chosen (↑↓←→). 'Rewards' = reward received at each step. Read them together: each column is one step of the game. The final return G is the sum of all rewards (with γ discounting).")
        st.code(
            f"Length  : {len(ep_sample)} steps       Return G : {total_r:.2f}\n"
            f"Path    : {path_str}\n"
            f"Actions : {acts_str}\n"
            f"Rewards : {rew_str}",
            language=""
        )

        # Return calculation walkthrough
        with st.expander("🔢 Step-by-step: How is the return G calculated from this episode?"):
            st.markdown(r"""
            The **return** $G_t$ at step $t$ is the discounted sum of all future rewards:

            $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T$$

            **Symbol decoder:**
            - $R_{t+1}$ = reward received *after* taking action at step $t$
            - $\gamma$ (gamma) = your discount setting — currently **""" + f"{gamma:.2f}" + r"""**
            - $T$ = length of the episode (last step)

            MC computes this **backward** through the episode — a computational trick:
            starting from the final step with G=0, then at each previous step:

            $$G_{\text{new}} = R_{\text{current step}} + \gamma \cdot G_{\text{previous}}$$

            **Worked example** (using the first 5 steps of the episode above):
            """)
            g = 0.0
            rows = []
            for i, (s, a, r) in enumerate(reversed(ep_sample[:5])):
                g = r + gamma * g
                rows.append({"Step (backward)": i+1, "State": str(s),
                             "Reward R": f"{r:+.2f}",
                             "G after update": f"{g:.4f}",
                             "Formula": f"G = {r:+.2f} + {gamma:.2f}×{(g-r)/gamma:.4f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown(f"""
            After processing all {len(ep_sample)} steps this way, the final G (for state (0,0)) = **{total_r:.4f}**.
            MC records this G and repeats for hundreds of episodes, then averages all G values per state.
            """)

    # ═════════════════════════════════════════════════════════════════════════
    # RUN ALL METHODS
    # ═════════════════════════════════════════════════════════════════════════
    if run_btn or "results" in st.session_state:
        if run_btn:
            with st.spinner("Running all 9 MC methods… (a few seconds)"):
                res = run_all_methods(env, n_episodes, gamma, epsilon, seed)
            st.session_state["results"] = res
            st.sidebar.success("✅ Done! Explore the tabs.")

        res = st.session_state["results"]

        # ═════════════════════════════════════════════════════════════════════
        # TAB 1 — MC PREDICTION
        # ═════════════════════════════════════════════════════════════════════
        with tab_pred:
            st.markdown(_card("#7c4dff","📊","What does MC Prediction solve?",
                """<b>The problem:</b> Before improving a policy, you must know how good it already is.
                MC Prediction answers: <em>"If the agent follows this exact policy forever, what total
                reward does it expect from each grid square?"</em><br><br>
                <b>Why this matters:</b> This estimated goodness score — called <b>V(s), the state value</b> —
                is the foundation for every improvement algorithm. You can't make decisions better
                without first knowing how good the current decisions are.<br><br>
                <b>The method:</b> Play many episodes. After each one, look backward and record what
                total reward followed each visited state. Average all those totals. That average <em>is</em> V(s)."""),
                unsafe_allow_html=True)

            st.subheader("📊 MC Prediction — First-Visit vs Every-Visit")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""
                #### First-Visit MC
                In each episode, if state **s** appears multiple times, **only the first occurrence**
                contributes a return sample to V(s).

                **Why?** Each first-visit return is statistically independent from other episodes'
                first-visit returns. This gives clean, unbiased estimates.

                **Analogy:** You visit a coffee shop 3 times in one day. First-Visit records only
                your satisfaction after the *first* coffee of the day.
                """)
            with c2:
                st.markdown("""
                #### Every-Visit MC
                **Every occurrence** of state s in every episode contributes a return sample.

                **Why use it?** More data per episode → faster convergence in practice. The samples
                within one episode are correlated (they're from the same game), introducing slight
                bias, but this vanishes as episodes → ∞.

                **Analogy:** You record your satisfaction after *every* coffee, including the 2nd
                and 3rd of the day — more data but they're not independent.
                """)

            with st.expander("📐 Theory & Formulas — MC Prediction (click to expand)", expanded=False):
                st.markdown(r"""
                #### Core Definition — What is V(s)?

                $$\boxed{v_\pi(s) \doteq \mathbb{E}_\pi\bigl[G_t \mid S_t = s\bigr]}$$

                **Symbol decoder:**
                - $v_\pi(s)$ — the true value of state $s$ under policy $\pi$ (what we're estimating)
                - $\mathbb{E}_\pi[\cdot]$ — expected value (average over infinitely many episodes)
                - $G_t$ — the return starting at time $t$
                - $S_t = s$ — "given that the agent is at state $s$ at time $t$"

                #### The Return — What MC Actually Computes

                $$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

                **Symbol decoder:**
                - $R_{t+k+1}$ — reward received $k+1$ steps after time $t$
                - $\gamma$ (gamma) — discount factor (0 to 1). If γ=0.99, a reward 10 steps away is worth $0.99^{10} \approx 0.90$ of its face value
                - The infinite sum terminates at episode end (terminal state has reward 0)

                #### The MC Estimate

                $$\hat{V}(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

                **Symbol decoder:**
                - $\hat{V}(s)$ — our current estimate of $v_\pi(s)$
                - $N(s)$ — how many times state $s$ has been visited (first-visit: per episode; every-visit: total)
                - $G_t^{(i)}$ — the return from the $i$-th visit to state $s$

                #### Convergence Guarantee

                By the **Law of Large Numbers**: as $N(s) \to \infty$, $\hat{V}(s) \to v_\pi(s)$.

                The standard error of First-Visit MC falls as $\dfrac{\sigma}{\sqrt{n}}$, where
                $\sigma$ is the standard deviation of returns from state $s$.
                This means **doubling accuracy requires 4× more episodes** — MC is sample-hungry.

                #### Why MC Is Unbiased (and Why That Matters)

                MC uses **actual returns** — real outcomes, not guesses. Unlike TD methods which
                estimate G using $R + \gamma V(S')$ (a guess), MC waits for the true G.
                No guessing = **zero bias**. The cost: it must wait for each full episode to finish.

                #### First-Visit vs Every-Visit: The Statistical Difference

                | Property | First-Visit | Every-Visit |
                |----------|-------------|-------------|
                | Bias | **Unbiased** for any N | Biased for finite N (converges to 0) |
                | Variance per episode | Lower (1 sample) | Higher (multiple correlated samples) |
                | Samples per episode | ≤ 1 per state | Potentially many per state |
                | Preferred when | Theoretical analysis | Practical fast convergence |
                """)

            st.markdown(_card("#4caf50","📖","How to read the heatmaps below",
                """Each grid square is coloured by its estimated <b>V(s) value</b>.<br>
                🟢 <b>Green cells</b> = high V(s) — being here is good, the agent usually reaches the goal from here<br>
                🔴 <b>Red cells</b> = low/negative V(s) — being here is bad (close to trap, or very far from goal)<br>
                🟡 <b>Yellow cells</b> = moderate V(s) — intermediate positions<br>
                <b>Numbers inside</b> = exact V(s) estimate. Walls (■) are blank — no agent ever stands there.<br>
                <b>What to look for:</b> Both maps should show the same general pattern — green near (4,4),
                red near (2,2). Any differences between left and right maps are due to counting visits differently."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_fv"], "First-Visit MC — V(s)", axes[0])
            plot_value_heatmap(env, res["V_ev"], "Every-Visit MC — V(s)", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Observations prompts
            c1, c2 = st.columns(2)
            with c1:
                fv_goal_adj = res["V_fv"].get((3,4), 0)
                st.metric("V(3,4) — First-Visit", f"{fv_goal_adj:.3f}",
                          help="State (3,4) is directly above the goal. High value expected.")
                st.markdown(f"""
                **What you should see here:**
                - States near (4,4) have **high positive** values (green)
                - States near (2,2) have **lower** values (trap risk)
                - Walls show no values (dark squares)
                - Corner (0,0) — the start — has a moderate-to-low value
                  (it takes many steps to reach the goal from the start)
                """)
            with c2:
                ev_goal_adj = res["V_ev"].get((3,4), 0)
                st.metric("V(3,4) — Every-Visit", f"{ev_goal_adj:.3f}",
                          help="Compare this to First-Visit — they should be very close.")
                st.markdown(f"""
                **Difference between the two maps:**
                - If they look nearly identical ✓ — both methods are converging to the same truth
                - If Every-Visit shows more extreme values — it's picking up correlated samples
                  that occasionally push estimates in one direction
                - The maps converge toward each other as episode count increases
                """)

            st.divider()
            st.subheader("📈 Convergence Chart — How estimates settle over time")
            st.markdown("""
            We track **state (3,4)** — the cell directly above the goal — because it's frequently
            visited and has a clear "true" value (high, since the goal is one step away).
            Watching this state's estimate evolve over episodes shows how quickly each method converges.
            """)

            st.markdown(_card("#7c4dff","📖","How to read this convergence chart",
                """<b>X-axis</b> = number of episodes played so far (from left = 0 to right = all episodes).<br>
                <b>Y-axis</b> = current estimate of V(3,4) — the "running average" of all returns from that state.<br>
                <b>What to look for:</b><br>
                • Early (left): both lines should be jumping around — high variance, few samples<br>
                • Later (right): both lines should flatten — converging toward the true value<br>
                • <b>Purple (First-Visit)</b>: usually smoother — independent samples per episode<br>
                • <b>Teal (Every-Visit)</b>: may converge faster but wobble more — correlated within-episode samples<br>
                • If both lines end at the same value ✓ — both methods found the same truth (as expected)"""),
                unsafe_allow_html=True)

            focal = (3, 4)
            fv_trace = [h.get(focal, 0.0) for h in res["hist_fv"]]
            ev_trace = [h.get(focal, 0.0) for h in res["hist_ev"]]
            x = [(i+1) * max(1, n_episodes // 20) for i in range(len(fv_trace))]

            fig2, ax2, _ = make_fig(1, 1, 10, 4)
            ax2.plot(x, fv_trace, color="#7c4dff", lw=2.5, marker="o", ms=5, label="First-Visit MC")
            ax2.plot(x, ev_trace, color="#00bcd4", lw=2.5, marker="s", ms=5, label="Every-Visit MC")
            ax2.axhline(np.mean(fv_trace[-5:] or [0]), color="#7c4dff", ls=":", alpha=0.5,
                        label=f"FV final ≈ {np.mean(fv_trace[-5:] or [0]):.2f}")
            ax2.axhline(np.mean(ev_trace[-5:] or [0]), color="#00bcd4", ls=":", alpha=0.5,
                        label=f"EV final ≈ {np.mean(ev_trace[-5:] or [0]):.2f}")
            ax2.set_xlabel("Episodes", color="white", fontsize=11)
            ax2.set_ylabel("V(3,4) estimate", color="white", fontsize=11)
            ax2.set_title("Value Estimate Convergence — State (3,4)", color="white", fontweight="bold")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=9)
            ax2.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            st.markdown(_warn("""
            <b>Common misconception:</b> "More episodes always makes Every-Visit better."
            Not necessarily. Every-Visit gets <em>more samples</em>, but they're correlated within episodes.
            First-Visit gets <em>fewer independent</em> samples. For most practical tasks the difference
            is small. Every-Visit is easier to implement and often preferred in practice.
            """), unsafe_allow_html=True)
            st.markdown(_tip("""
            <b>Experiment idea:</b> Try setting Episodes to 200 (slider). Both lines will be noisy —
            that's high variance from small N. Now try 2000. The lines should be much smoother.
            This directly demonstrates the 1/√n convergence rate.
            """), unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        # TAB 2 — ON-POLICY CONTROL
        # ═════════════════════════════════════════════════════════════════════
        with tab_ctrl:
            st.markdown(_card("#00897b","🎯","What does On-policy MC Control solve?",
                """<b>The problem:</b> Prediction tells us how good the current policy is — but what
                if the policy is bad? We need to <em>improve</em> it.<br><br>
                <b>The upgrade:</b> Instead of V(s), we now learn <b>Q(s,a) — action values</b>.
                Q(s,a) answers "how good is it to take action <em>a</em> specifically in state <em>s</em>?"
                This lets the agent compare its options and pick the best one — without needing any
                model of how the world works.<br><br>
                <b>The exploration trap:</b> If the agent always picks the best-known action (pure greedy),
                it never discovers better alternatives it hasn't tried yet. ε-greedy escapes this by
                occasionally picking a random action to explore."""),
                unsafe_allow_html=True)

            st.subheader("🎯 On-policy MC Control — ε-greedy GPI")

            with st.expander("📐 Theory & Formulas — On-policy MC Control", expanded=False):
                st.markdown(r"""
                #### Why Q(s,a) instead of V(s)?

                With V(s) alone and **no model**, you can't decide which action to take.
                You'd need to know: "if I take action a, what state will I end up in?" — that's model knowledge.

                Q(s,a) sidesteps this: it directly tells you the value of each action from each state.
                """)
                st.latex(
                    r"\boxed{q_\pi(s,a) \doteq \mathbb{E}_\pi\bigl[G_t \mid S_t = s,\, A_t = a\bigr]}"
                )
                st.markdown(r"""
                **Symbol decoder:**
                - $q_\pi(s,a)$ — expected return when you're in state $s$, take action $a$, then follow policy $\pi$ forever after
                - $A_t = a$ — "given that the action taken at time $t$ is $a$"
                - Everything else is the same as in V(s)

                #### MC Estimation of Q
                """)
                st.latex(
                    r"Q(s,a) \;\leftarrow\; \text{average of all } G_t \text{ following first-visit to } (s,a)"
                )
                st.markdown(r"""
                The agent tracks **state-action pairs** $(s,a)$ instead of just states $s$.
                With 25 states × 4 actions = **100 pairs** to estimate in this gridworld.

                #### The Exploration Problem — Why ε-greedy?

                If the policy is deterministic (always pick the best known action), many state-action
                pairs are never visited → their Q values never improve → the policy gets stuck.

                **ε-greedy** solves this with a randomised policy:
                """)
                st.latex(
                    r"\pi(a \mid s) = \begin{cases}"
                    r" 1 - \varepsilon + \dfrac{\varepsilon}{\lvert A(s)\rvert} & a = \arg\max_{a'} Q(s,a') \quad \text{(best known action)} \\[8pt]"
                    r" \dfrac{\varepsilon}{\lvert A(s)\rvert} & \text{otherwise} \quad \text{(exploration)}"
                    r"\end{cases}"
                )
                st.markdown(
                    f"""
                **Symbol decoder:**
                - $\\varepsilon$ — exploration rate (your ε slider, currently **{epsilon:.2f}**)
                - $\\lvert A(s)\\rvert$ — number of available actions (= 4 in this gridworld)
                - Best action probability: $1 - \\varepsilon + \\varepsilon/4$ = **{1 - epsilon + epsilon / 4:.3f}**
                - Each other action probability: $\\varepsilon/4$ = **{epsilon / 4:.3f}**
                """
                )
                st.markdown(r"""
                #### GPI — Generalised Policy Iteration

                Two steps alternate after every episode:

                | Phase | Formula | What it does |
                |-------|---------|-------------|
                | **Evaluation** | $Q(s,a) \leftarrow \text{mean}(G_t)$ for visited $(s,a)$ | Update action values from observed returns |
                | **Improvement** | $\pi(s) \leftarrow \varepsilon\text{-greedy}(Q)$ | Rebuild policy to be greedy on updated Q |

                This **GPI loop** is guaranteed to converge — each improvement step makes the policy
                at least as good as before (provable monotone improvement).

                #### The Fundamental Limitation

                On-policy control converges to the **best ε-soft policy**, not the absolute optimal π*.
                An ε-soft policy always has probability ≥ $\varepsilon/\lvert A\rvert$ of picking a suboptimal action.
                To reach the true optimal, you need off-policy methods.

                #### Why No Exploring Starts?

                Early MC algorithms required every state-action pair as a possible starting position
                — "exploring starts" — which is impractical in real environments.
                ε-greedy eliminates this requirement: every action has a nonzero probability of
                being selected from any state, so all (s,a) pairs are eventually visited naturally.
                """
                )

            st.markdown(_card("#4caf50","📖","How to read these two diagrams",
                """<b>Left — Value heatmap (Q-max):</b>
                Shows max<sub>a</sub>Q(s,a) — the best action-value from each state.
                This is the agent's "confidence in success" from each square.
                Same colour scale as before: green=good, red=bad.<br><br>
                <b>Right — Policy arrows:</b>
                Each arrow shows <b>arg max Q(s,a)</b> — the single best action the agent has learned for that square.
                Reading the arrows as a sequence from start (0,0) gives you the agent's intended path to the goal.<br><br>
                <b>What to look for:</b>
                Do the arrows from (0,0) form a coherent path toward (4,4)?
                Do arrows near (2,2) point <em>away</em> from the trap?
                With more episodes, the arrows should become more consistent and purposeful."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_on"], "Q-max V(s) — On-policy Control", axes[0])
            plot_policy_arrows(env, res["pi_det"], "Greedy Policy π (from Q)", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Q-table at start
            st.divider()
            st.subheader("🔢 Action Values at the Start State (0,0)")
            st.markdown("""
            The table below shows **Q((0,0), a)** — the estimated value of each first move.
            This is what the agent "thinks" about its four possible first actions after training.
            The ✅ row is the action the agent will take (the greedy choice).
            """)
            st.markdown(_card("#00897b","📖","How to read the Q-value table",
                """Each row = one of the 4 possible actions from the start position (0,0).<br>
                The <b>Q value</b> = expected total discounted return if you take that action now and follow
                the current policy forever after.<br>
                <b>Higher Q = better action.</b> A well-trained agent should show ↓ (Down) or → (Right) as the
                best first move — these head toward the goal at (4,4). ↑ (Up) is blocked by the grid boundary
                and ← (Left) runs into a wall, so their Q values should be lower."""),
                unsafe_allow_html=True)

            Q_start = res["Q_on"][(0,0)]
            best_a  = int(np.argmax(Q_start))
            qdf = pd.DataFrame({
                "Action": ["↑ Up", "→ Right", "↓ Down", "← Left"],
                "Q(start, a)": [f"{v:.4f}" for v in Q_start],
                "Probability (ε-greedy)": [
                    f"{(1-epsilon+epsilon/4):.3f} ✅ BEST" if i == best_a else f"{epsilon/4:.3f}"
                    for i in range(4)],
                "Interpretation": [
                    "Blocked by grid edge" if i==0 else
                    "Moves right → toward goal column" if i==1 else
                    "Moves down → toward goal row" if i==2 else
                    "Runs into wall (1,0)→bounce" for i in range(4)],
            })
            st.dataframe(qdf, use_container_width=True, hide_index=True)

            # Learning curve
            st.divider()
            st.subheader("📈 Learning Curve — Does the agent improve over time?")
            st.markdown(_card("#00897b","📖","How to read the learning curve",
                """<b>X-axis</b> = episode number (left = early training, right = late training).<br>
                <b>Y-axis</b> = total reward collected in that episode.<br>
                <b>Faint noisy line</b> = raw episode returns — very noisy (one bad episode can give −5).<br>
                <b>Bright smooth line</b> = rolling average (window = N/20 episodes) — shows the trend.<br>
                <b>Green dashed line</b> = peak rolling-average return achieved.<br><br>
                <b>What to look for:</b>
                The trend should rise from negative (random policy = lots of trap visits) toward positive.
                The plateau value reveals the ε-soft policy limit: even a well-trained agent sometimes
                wastes moves due to the ε exploration requirement."""),
                unsafe_allow_html=True)

            raw = res["ep_rewards_on"]
            sm  = smooth(raw, max(1, len(raw)//20))
            fig3, ax3, _ = make_fig(1, 1, 11, 4)
            ax3.plot(raw, color="#00897b", alpha=0.12, lw=0.6)
            ax3.plot(range(len(sm)), sm, color="#00897b", lw=2.5, label="Smoothed return (on-policy)")
            ax3.axhline(float(np.max(sm)), color="#4caf50", ls="--", lw=1.2, alpha=0.8,
                        label=f"Peak return: {float(np.max(sm)):.2f}")
            ax3.axhline(0, color="white", ls=":", lw=0.5, alpha=0.3)
            ax3.set_xlabel("Episode number", color="white", fontsize=11)
            ax3.set_ylabel("Total return G", color="white", fontsize=11)
            ax3.set_title("On-policy MC Control — Learning Progress", color="white", fontweight="bold")
            ax3.legend(facecolor=CARD_BG, labelcolor="white")
            ax3.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            st.markdown(_tip("""
            <b>Experiment:</b> Increase ε (exploration rate) to 0.4 and re-run.
            The learning curve will be noisier and the final performance lower — too much exploration
            prevents the agent from exploiting what it has learned.
            Decrease ε to 0.01 — the agent exploits early but may miss better routes.
            """), unsafe_allow_html=True)
            st.markdown(_warn("""
            <b>Why the return may never reach +10:</b> Even a perfect policy takes roughly 8 steps
            to reach the goal (8 × −0.1 = −0.8 penalty), giving a best-case return of about +9.2.
            The ε-greedy policy also occasionally wastes moves on random exploration, further reducing
            the average. A return consistently above +8 is excellent for this environment.
            """), unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        # TAB 3 — OFF-POLICY IS EVALUATION
        # ═════════════════════════════════════════════════════════════════════
        with tab_offpol:
            st.markdown(_card("#ef5350","⚖️","What does Off-policy IS Evaluation solve?",
                """<b>The dilemma:</b> The agent that collects experience (behavior policy b) must explore —
                taking suboptimal actions to discover new things. But the policy we want to evaluate
                (target policy π) might be purely greedy. If b and π are different, returns from b
                have the wrong expected value for π — they can't just be averaged.<br><br>
                <b>The trick — Importance Sampling:</b> We <em>reweight</em> each return by how much more
                or less likely that episode trajectory was under π than under b. Trajectories that π
                would have preferred get higher weight; trajectories π would have avoided get lower weight.<br><br>
                <b>Real-world applications:</b> Medical trials (evaluate treatment B using data from treatment A),
                recommendation systems (evaluate new ranking policy using logs from old one), robotics safety
                (evaluate aggressive policy using safe exploration data)."""),
                unsafe_allow_html=True)

            st.subheader("⚖️ Off-policy IS Evaluation — Ordinary vs Weighted")

            with st.expander("📐 Theory & Formulas — Importance Sampling", expanded=False):
                st.markdown(r"""
                #### The Core Problem

                Returns from behavior policy $b$ have expectation $v_b(s)$, not $v_\pi(s)$:
                $$\mathbb{E}_b[G_t \mid S_t=s] = v_b(s) \neq v_\pi(s)$$

                We need to correct for the mismatch between b and π.

                #### The IS Ratio — The Correction Weight

                For a trajectory $A_t, S_{t+1}, \ldots, S_T$, the ratio of how likely it was under
                π vs b is:

                $$\boxed{\rho_{t:T-1} \doteq \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}}$$

                **Symbol decoder:**
                - $\rho_{t:T-1}$ — the importance sampling ratio (Greek letter "rho")
                - $\pi(A_k \mid S_k)$ — probability that target policy π would have chosen action $A_k$ at step $k$
                - $b(A_k \mid S_k)$ — probability that behavior policy b chose it (this is known — we generated the episode)
                - The product multiplies these ratios over all steps from $t$ to $T-1$

                **Key insight:** The environment dynamics $p(S_{k+1}|S_k,A_k)$ cancel out — they appear
                in both numerator and denominator. So IS only depends on the policies, not the transition model.
                **This makes off-policy IS model-free.**

                **Worked example (3-step episode):**

                If π would choose each action with probability 0.7 and b chose it with probability 0.25:

                $$\rho = \frac{0.7}{0.25} \times \frac{0.7}{0.25} \times \frac{0.7}{0.25} = 2.8 \times 2.8 \times 2.8 = 21.95$$

                This trajectory is 22× more likely under π than under b — it gets 22× the weight.
                You can see why variance explodes for long episodes!

                #### Ordinary Importance Sampling (Unbiased, High Variance)

                $$V(s) \doteq \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{|\mathcal{T}(s)|}$$

                **Symbol decoder:**
                - $\mathcal{T}(s)$ — all time steps where state $s$ was visited
                - $T(t)$ — the terminal time of the episode containing step $t$
                - $|\mathcal{T}(s)|$ — just the count of visits

                ✅ **Unbiased** — correct expectation: $\mathbb{E}[\rho G_t] = v_\pi(s)$  
                ❌ **Unbounded variance** — ρ can be enormous; one outlier episode ruins the average

                #### Weighted Importance Sampling (Biased, Low Variance)

                $$V(s) \doteq \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} G_t}{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$

                The denominator normalises the weights so they sum to 1.
                **No single episode can dominate** — even if ρ=1000, it's 1000/(sum of all ρ's).

                ✅ **Dramatically lower variance**  
                ❌ **Biased** for finite N, but bias → 0 as N → ∞

                #### The Coverage Requirement

                $$\pi(a \mid s) > 0 \implies b(a \mid s) > 0 \quad \text{(coverage)}$$

                Every action the target policy might take must also be possible under the behavior policy.
                In this app: b = uniform random (all actions equally likely at 0.25), satisfying
                coverage for any deterministic π.
                """)

            st.markdown(_card("#ef5350","📖","How to read these three heatmaps",
                """All three maps estimate <b>the same quantity</b>: V(s) for the greedy target policy π.<br>
                <b>Left (On-policy reference):</b> Learned directly by following π — this is the "ground truth" we're comparing to.<br>
                <b>Middle (Ordinary IS):</b> Estimated from random-exploration data using unweighted IS ratios. May have extreme values from outlier ρ weights.<br>
                <b>Right (Weighted IS):</b> Same data, but weights are normalised. Should be smoother and closer to the reference.<br>
                <b>What to look for:</b> Middle map may have cells that are very different from the reference —
                this is the high-variance problem in action. Right map should track the reference much more closely."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 3, 16, 5)
            plot_value_heatmap(env, res["V_on"],  "On-policy (reference)",  axes[0])
            plot_value_heatmap(env, res["V_ois"], "Ordinary IS",            axes[1])
            plot_value_heatmap(env, res["V_wis"], "Weighted IS",            axes[2])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Per-state comparison
            st.divider()
            st.subheader("📊 State-by-state Accuracy Comparison")
            st.markdown(_card("#ef5350","📖","How to read this line chart",
                """Each point on the X-axis is one non-wall, non-terminal state (listed in grid-scan order).<br>
                The Y-axis is the estimated V(s) value for that state.<br>
                <b>Green line (reference)</b> = on-policy ground truth — what the true values should be.<br>
                <b>Red line (Ordinary IS)</b> = off-policy estimate with unweighted ratios. Look for <em>spikes</em>
                where the estimate is far from green — those are high-ρ episodes dominating the average.<br>
                <b>Blue line (Weighted IS)</b> = normalised weights. Should hug the green line much more closely.<br>
                The closer a line is to green, the better the method estimates the target policy's values."""),
                unsafe_allow_html=True)

            common = [s for s in res["V_on"] if s in res["V_ois"] and s in res["V_wis"]
                      and s not in env.walls and not env.is_terminal(s)]
            ref = np.array([res["V_on"][s]  for s in common])
            ois = np.array([res["V_ois"][s] for s in common])
            wis = np.array([res["V_wis"][s] for s in common])

            fig4, ax4, _ = make_fig(1, 1, 11, 4)
            idx = range(len(common))
            ax4.plot(idx, ref, "o-",  color="#4caf50", lw=2,   ms=6, label="On-policy (reference / ground truth)")
            ax4.plot(idx, ois, "s--", color="#ef5350", lw=1.8, ms=5, alpha=0.85, label="Ordinary IS (high variance)")
            ax4.plot(idx, wis, "^-",  color="#42a5f5", lw=1.8, ms=5, alpha=0.85, label="Weighted IS (lower variance)")
            ax4.set_xlabel("State (non-wall, non-terminal)", color="white", fontsize=11)
            ax4.set_ylabel("V(s) estimate", color="white", fontsize=11)
            ax4.set_title("IS Estimation Accuracy vs On-policy Reference", color="white", fontweight="bold")
            ax4.legend(facecolor=CARD_BG, labelcolor="white")
            ax4.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig4); plt.close()

            mse_ois = float(np.mean((ois - ref)**2))
            mse_wis = float(np.mean((wis - ref)**2))
            pct_imp = max(0, (mse_ois - mse_wis) / max(mse_ois, 1e-9) * 100)

            st.markdown(_card("#ef5350","📖","How to read the MSE metrics below",
                """<b>MSE (Mean Squared Error)</b> = average of (estimated − true)² across all states.<br>
                Lower MSE = closer to the reference = more accurate off-policy estimate.<br>
                The third metric shows how much Weighted IS improved over Ordinary IS —
                a large improvement confirms the theoretical prediction that weighted IS
                has dramatically lower variance (and thus lower error for finite samples)."""),
                unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Ordinary IS — MSE vs reference", f"{mse_ois:.4f}", help="Lower is better")
            c2.metric("Weighted IS — MSE vs reference",  f"{mse_wis:.4f}", help="Lower is better")
            c3.metric("WIS accuracy improvement", f"{pct_imp:.1f}%", delta="↓ lower MSE")

            st.markdown(_tip("""
            <b>Experiment:</b> Increase episodes to 2000. Both MSE values should drop significantly —
            both IS methods converge to the true values with enough data. But Weighted IS will always
            converge faster (lower variance → fewer episodes needed for the same accuracy).
            """), unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        # TAB 4 — OFF-POLICY MC CONTROL
        # ═════════════════════════════════════════════════════════════════════
        with tab_offctrl:
            st.markdown(_card("#1565c0","🎲","What does Off-policy MC Control solve?",
                """<b>The on-policy limitation:</b> On-policy control converges to the best ε-soft policy —
                which always has some probability of picking suboptimal actions. The exploration
                requirement is permanently baked into the policy.<br><br>
                <b>Off-policy control separates concerns:</b>
                The <em>behavior policy b</em> explores freely (ε-soft), generating all the experience.
                The <em>target policy π</em> is purely greedy — it only takes the best known action.
                Because π never has to explore, it can converge to the <b>true optimal policy π*</b>.<br><br>
                <b>The catch — the greedy tail problem:</b> Updates only apply to states in the
                "greedy tail" of each episode — the consecutive run of greedy actions at the END.
                The moment a non-greedy action appears (walking backward), the IS weight becomes 0
                and all earlier updates are discarded. This can be very slow."""),
                unsafe_allow_html=True)

            st.subheader("🎲 Off-policy MC Control — Learning the True Optimal π*")

            with st.expander("📐 Theory & Formulas — Off-policy MC Control", expanded=False):
                st.markdown(r"""
                #### Two Policies, One Goal

                | Policy | Symbol | Type | Role |
                |--------|--------|------|------|
                | Target | $\pi$ | Deterministic greedy | The policy we improve — will become π* |
                | Behavior | $b$ | ε-soft stochastic | Generates all episodes — ensures coverage |

                #### The Full Algorithm (Sutton & Barto, Chapter 5)

                **Initialise:** $Q(s,a) \leftarrow 0$, $C(s,a) \leftarrow 0$, $\pi(s) \leftarrow \arg\max_a Q(s,a)$

                **For each episode:**
                1. Generate episode using $b$: $S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_T$
                2. Set $G \leftarrow 0$, $W \leftarrow 1$
                3. **Walk backward** $t = T-1, T-2, \ldots, 0$:

                $$G \leftarrow \gamma G + R_{t+1}$$

                $$C(S_t, A_t) \leftarrow C(S_t, A_t) + W \quad \text{(accumulate IS weight)}$$

                $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{W}{C(S_t, A_t)}\bigl[G - Q(S_t, A_t)\bigr]$$

                $$\pi(S_t) \leftarrow \arg\max_a Q(S_t, a) \quad \text{(update greedy target)}$$

                $$\textbf{If } A_t \neq \pi(S_t): \textbf{ BREAK} \quad \text{(non-greedy → W=0 → stop)}$$

                $$W \leftarrow W \cdot \frac{1}{b(A_t \mid S_t)} \quad \text{(IS weight: π(a|s)=1 since greedy)}$$

                **Symbol decoder:**
                - $C(S_t, A_t)$ — cumulative IS weight for pair $(S_t, A_t)$ — the denominator of weighted IS
                - $W$ — current episode's running IS weight (product of $1/b$ so far)
                - $\frac{W}{C}[G - Q]$ — the weighted IS update: error term scaled by relative weight

                #### Why Break on a Non-Greedy Action?

                Since $\pi$ is deterministic greedy: $\pi(a|s) = 1$ for the best action, 0 otherwise.

                The IS weight $W = \prod \frac{\pi(a|s)}{b(a|s)}$.

                If any action $A_t$ was non-greedy: $\pi(A_t|S_t) = 0 \Rightarrow W = 0$.

                With W=0, the update $\frac{W}{C}[G-Q]$ = 0 — no learning happens.
                So breaking early is exactly equivalent (and computationally faster).

                #### The Greedy Tail Problem

                Learning only from the greedy tail means:
                - Episode of 20 steps where the agent makes 1 non-greedy action at step 3
                → only steps 4–20 (17 steps) contribute learning
                - Shorter greedy tails = less learning per episode = slower convergence

                This is the fundamental cost of achieving true optimality.

                #### Why π* instead of just best ε-soft?

                Target policy $\pi$ has no ε constraint → $\pi(s) = \arg\max Q(s,a)$ exactly.
                With infinite episodes: $Q \to q^*$ → $\pi \to \pi^*$ (true optimal, provably).
                On-policy ε-greedy converges to: $\pi(s) = \arg\max Q(s,a)$ with probability $1-\varepsilon$
                → can never be purely greedy.
                """)

            st.markdown(_card("#42a5f5","📖","How to read the diagrams on this tab",
                """<b>Top row:</b> The learned target policy π* (value heatmap + arrows). This is a <em>deterministic</em>
                policy — every cell has exactly one arrow, no randomness. Compare to the on-policy arrows in the previous tab.<br>
                <b>Middle row:</b> Direct side-by-side comparison — on-policy arrows (left) vs off-policy arrows (right).
                Look for cells where they disagree — the off-policy agent may make bolder choices near the trap because
                it never has to randomly explore there.<br>
                <b>Bottom left:</b> State value comparison. Off-policy values should be ≥ on-policy since π* ≥ best ε-soft π.<br>
                <b>Bottom right:</b> Greedy tail fraction — what fraction of each episode contributed to learning.
                Low values mean learning is slow (behavior rarely stays greedy consecutively)."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_off"], "Off-policy Control — V(s)", axes[0])
            plot_policy_arrows(env, res["pi_off"], "Learned Target Policy π*", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.subheader("🔀 Policy Comparison: On-policy vs Off-policy")
            st.markdown("""
            Both maps below show the **greedy** version of each method's learned policy —
            the action with the highest Q-value from each state.

            The key difference: on-policy arrows came from a policy trained *with* ε-exploration built in.
            Off-policy arrows came from a deterministic target policy trained separately.
            """)
            fig2, axes2, _ = make_fig(1, 2, 12, 5)
            plot_policy_arrows(env, res["pi_det"], "On-policy Greedy π  (ε-soft origin)", axes2[0])
            plot_policy_arrows(env, res["pi_off"], "Off-policy Target π*  (deterministic)", axes2[1])
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            all_s = [env.i2s(i) for i in range(env.n_states)
                     if env.i2s(i) not in env.walls and not env.is_terminal(env.i2s(i))]
            v_on_arr  = np.array([res["V_on"].get(s, 0)  for s in all_s])
            v_off_arr = np.array([res["V_off"].get(s, 0) for s in all_s])

            fig3, axes3, _ = make_fig(1, 2, 14, 4)
            idx = range(len(all_s))
            axes3[0].plot(idx, v_on_arr,  "o-",  color="#00897b", lw=2,   ms=5, label="On-policy")
            axes3[0].plot(idx, v_off_arr, "s--", color="#1565c0", lw=1.8, ms=5, label="Off-policy (target π*)")
            axes3[0].set_xlabel("State index", color="white"); axes3[0].set_ylabel("V(s)", color="white")
            axes3[0].set_title("State Values: On-policy vs Off-policy", color="white", fontweight="bold")
            axes3[0].legend(facecolor=CARD_BG, labelcolor="white"); axes3[0].grid(alpha=0.15)

            gh    = res["greedy_hist"]
            sm_gh = smooth(gh, max(1, len(gh)//20)) if len(gh) > 1 else np.array(gh)
            axes3[1].plot(gh, color="#42a5f5", alpha=0.12, lw=0.5)
            axes3[1].plot(range(len(sm_gh)), sm_gh, color="#42a5f5", lw=2.5, label="Greedy tail fraction")
            axes3[1].axhline(float(np.mean(gh)), color="#ffa726", ls="--", lw=1.5,
                             label=f"Mean: {np.mean(gh):.2f} ({np.mean(gh)*100:.0f}% of steps used)")
            axes3[1].set_xlabel("Episode", color="white")
            axes3[1].set_ylabel("Greedy tail / episode length", color="white")
            axes3[1].set_title("Fraction of Each Episode Used for Learning", color="white", fontweight="bold")
            axes3[1].legend(facecolor=CARD_BG, labelcolor="white"); axes3[1].grid(alpha=0.15)
            axes3[1].set_ylim(0, 1)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            st.markdown(_card("#42a5f5","📖","What the greedy tail chart tells you",
                f"""The y-axis (0 to 1) shows what fraction of each episode was used for Q updates.<br>
                A value of <b>{np.mean(gh):.2f}</b> means on average, {np.mean(gh)*100:.0f}% of steps contributed learning.<br>
                <b>If this value is low</b> (&lt;0.3): the behavior policy often takes non-greedy actions early,
                cutting off learning. Consider reducing ε to make b more greedy — but coverage must still hold.<br>
                <b>As training progresses</b>: the target policy improves → the behavior policy (which is based on Q)
                also improves → more steps stay greedy → the tail fraction should <em>increase</em> over time."""),
                unsafe_allow_html=True)

            raw_on  = res["ep_rewards_on"]
            raw_off = res["ep_rewards_off"]
            sm_on   = smooth(raw_on,  max(1, len(raw_on) //20))
            sm_off  = smooth(raw_off, max(1, len(raw_off)//20))

            st.divider()
            st.subheader("📈 Learning Curves — On-policy vs Off-policy")
            st.markdown(_card("#1565c0","📖","How to read these learning curves",
                """Both agents start from scratch with zero knowledge and play the same number of episodes.<br>
                <b>Green (on-policy)</b> often rises faster early — it learns from every step of every episode.<br>
                <b>Blue (off-policy)</b> may rise slower at first — it only learns from the greedy tail.<br>
                <b>What to look for at the end of training:</b> Does off-policy reach a higher plateau?
                In theory it should — its target policy has no ε constraint. In practice with limited
                episodes, on-policy often wins in speed while off-policy wins in peak quality."""),
                unsafe_allow_html=True)

            fig4, ax4, _ = make_fig(1, 1, 12, 4)
            ax4.plot(raw_on,  color="#00897b", alpha=0.12, lw=0.5)
            ax4.plot(raw_off, color="#1565c0", alpha=0.12, lw=0.5)
            ax4.plot(range(len(sm_on)),  sm_on,  color="#00897b", lw=2.5, label="On-policy ε-greedy")
            ax4.plot(range(len(sm_off)), sm_off, color="#1565c0", lw=2.5, label="Off-policy target π*")
            ax4.set_xlabel("Episode number", color="white", fontsize=11)
            ax4.set_ylabel("Total return", color="white", fontsize=11)
            ax4.set_title("Learning Curves — On-policy vs Off-policy MC Control", color="white", fontweight="bold")
            ax4.legend(facecolor=CARD_BG, labelcolor="white"); ax4.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig4); plt.close()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("On-policy mean V(s)",   f"{float(np.mean(v_on_arr)):.3f}")
            c2.metric("Off-policy mean V(s)",  f"{float(np.mean(v_off_arr)):.3f}",
                      delta=f"{float(np.mean(v_off_arr)-np.mean(v_on_arr)):+.3f}")
            c3.metric("On-policy peak return",  f"{float(np.max(sm_on)):.2f}")
            c4.metric("Off-policy peak return", f"{float(np.max(sm_off)):.2f}",
                      delta=f"{float(np.max(sm_off)-np.max(sm_on)):+.2f}")

        # ═════════════════════════════════════════════════════════════════════
        # TAB 5 — INCREMENTAL MC
        # ═════════════════════════════════════════════════════════════════════
        with tab_incr:
            st.markdown(_card("#ff9800","⚡","What does Incremental MC solve?",
                """<b>The memory problem:</b> Standard MC stores every observed return for every state —
                G₁, G₂, G₃, ... — then computes the average at the end. For an agent running millions
                of episodes, this exhausts memory.<br><br>
                <b>The solution:</b> A running mean update that only needs two numbers per state:
                the current estimate V(s) and the visit count N(s). No list of returns needed.<br><br>
                <b>Why it matters beyond memory:</b> Replace the 1/N step-size with a fixed constant α
                and you get an exponential moving average — newer episodes matter more than old ones.
                This is essential for <em>non-stationary</em> environments and is the direct bridge to
                TD learning."""),
                unsafe_allow_html=True)

            st.subheader("⚡ Incremental Monte Carlo")

            with st.expander("📐 Theory & Formulas — Incremental MC", expanded=False):
                st.markdown(r"""
                #### The Batch Formula We're Replacing

                Batch MC computes:
                """)
                st.latex(r"V_n(s) = \frac{1}{n}\sum_{i=1}^{n} G_i")
                st.markdown(r"""
                This requires storing all $n$ returns. For $n = 10^6$ episodes, that's 10⁶ numbers per state.

                #### The Incremental Identity

                Note that:
                """)
                st.latex(
                    r"\frac{1}{n}\sum_{i=1}^{n} G_i = \frac{1}{n}\left(\sum_{i=1}^{n-1} G_i + G_n\right)"
                    r" = \frac{n-1}{n} V_{n-1} + \frac{1}{n}G_n = V_{n-1} + \frac{1}{n}(G_n - V_{n-1})"
                )
                st.markdown(r"""
                This gives the **incremental update rule:**
                """)
                st.latex(
                    r"\boxed{V_{n}(s) \leftarrow V_{n-1}(s) + \frac{1}{n}\bigl[G_n - V_{n-1}(s)\bigr]}"
                )
                st.markdown(r"""
                **Symbol decoder:**
                - $V_{n-1}(s)$ — current estimate before seeing the new return
                - $G_n$ — the new return from the $n$-th visit to state $s$
                - $\frac{1}{n}$ — step size: shrinks as $n$ grows (each new sample has less impact)
                - $[G_n - V_{n-1}(s)]$ — the **error**: how much the new observation differs from our current belief

                This is **mathematically identical** to the batch formula but uses O(1) memory.

                #### Worked Numeric Example

                Suppose we've visited state (3,4) three times with returns: G₁=8.5, G₂=9.1, G₃=7.8.

                | Step | New G | Old V | Error | New V |
                |------|-------|-------|-------|-------|
                | n=1 | 8.50 | 0.00 | +8.50 | 0.00 + (1/1)×8.50 = **8.500** |
                | n=2 | 9.10 | 8.500| +0.60 | 8.500 + (1/2)×0.60 = **8.800** |
                | n=3 | 7.80 | 8.800| −1.00 | 8.800 + (1/3)×(−1.00) = **8.467** |

                Batch average: (8.5+9.1+7.8)/3 = **8.467** ✓ — identical result.

                #### Fixed Step-Size α — The Non-Stationary Extension

                Replace $1/n$ with a constant $\alpha \in (0,1)$:
                """)
                st.latex(r"V(s) \leftarrow V(s) + \alpha\bigl[G_t - V(s)\bigr]")
                st.markdown(r"""
                This weights recent returns exponentially more than old ones:
                """)
                st.latex(
                    r"V_n = (1-\alpha)^n V_0 + \alpha\sum_{k=1}^{n}(1-\alpha)^{n-k} G_k"
                )
                st.markdown(r"""
                With α=0.1: a return from 50 episodes ago has weight $0.9^{50} \approx 0.005$ —
                it's nearly forgotten. This is crucial when the environment *changes over time*.

                | Update | Memory of past | When to use |
                |--------|---------------|-------------|
                | $1/n$ step-size | Equal weight all returns | Stationary environment |
                | Fixed $\alpha$ | Recent returns matter more | Non-stationary, changing rewards |

                #### The Bridge to TD Learning

                Replace $G_t$ (requires full episode) with the **TD target** $R_{t+1} + \gamma V(S_{t+1})$:
                """)
                st.latex(
                    r"V(S_t) \leftarrow V(S_t) + \alpha\bigl[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)\bigr]"
                )
                st.markdown(r"""
                This is **TD(0)** — the simplest Temporal-Difference method. The incremental update
                structure from MC is identical; only the target changes from the true $G_t$ to a one-step estimate.
                """)

            st.markdown(_card("#ffa726","📖","How to read the heatmaps below",
                """Both maps show V(s) under the same random policy using the same episodes.<br>
                <b>Left (Batch First-Visit):</b> Stores all returns, computes exact average at end.<br>
                <b>Right (Incremental MC):</b> Updates running mean after each episode — no list stored.<br>
                <b>They should look nearly identical</b> — that's the whole point. Incremental MC gives
                the exact same answer with a fraction of the memory. Any small differences are random
                seed effects from the order of updates, not systematic error."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_fv"],  "First-Visit MC (batch average)", axes[0])
            plot_value_heatmap(env, res["V_inc"], "Incremental MC (running mean)",  axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.divider()
            st.subheader("📉 Variance Reduction — How estimates stabilise over time")
            st.markdown("""
            As more episodes are played, the estimates for each state are based on more samples.
            More samples → lower variance → more reliable values. The chart below tracks
            the **average variance of V(s) estimates** across all states as training progresses.
            """)
            st.markdown(_card("#ff9800","📖","How to read the variance chart",
                """<b>X-axis</b> = training checkpoint (every N/20 episodes).<br>
                <b>Y-axis</b> = average variance of V(s) across all visited states. This measures how
                much the estimates are "jumping around" — high variance means unreliable, low variance means settled.<br>
                <b>The orange fill</b> = the area of uncertainty — as it shrinks, the agent "knows more".<br>
                <b>What to expect:</b> The curve should drop sharply early (each new episode dramatically
                improves sparse estimates) then flatten as estimates stabilise (diminishing returns per episode).<br>
                <b>Rate of fall:</b> Variance drops proportional to 1/n — to halve the remaining variance, you need to double episodes."""),
                unsafe_allow_html=True)

            fig5, ax5, _ = make_fig(1, 1, 10, 4)
            vh = res["var_hist"]
            if vh:
                ax5.plot(vh, color="#ff9800", lw=2.5, label="Mean variance across states")
                ax5.fill_between(range(len(vh)), 0, vh, color="#ff9800", alpha=0.15)
                if len(vh) > 3:
                    ax5.annotate("High variance:\nfew samples", xy=(0, vh[0]),
                                 xytext=(len(vh)*0.1, vh[0]*0.9),
                                 color="#ffcc80", fontsize=8,
                                 arrowprops=dict(arrowstyle="->", color="#ffcc80", lw=1))
                    ax5.annotate("Low variance:\nmany samples", xy=(len(vh)-1, vh[-1]),
                                 xytext=(len(vh)*0.6, vh[-1]+max(vh)*0.2),
                                 color="#a5d6a7", fontsize=8,
                                 arrowprops=dict(arrowstyle="->", color="#a5d6a7", lw=1))
            ax5.set_xlabel("Checkpoint (every N/20 episodes)", color="white", fontsize=11)
            ax5.set_ylabel("Mean Var(V(s)) across states", color="white", fontsize=11)
            ax5.set_title("Variance Reduction as More Episodes Are Played", color="white", fontweight="bold")
            ax5.legend(facecolor=CARD_BG, labelcolor="white"); ax5.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig5); plt.close()

            c1, c2 = st.columns(2)
            c1.info("**Memory advantage:** Batch MC needs O(N×S) storage (N returns × S states). Incremental MC needs O(S) — just current estimate + count per state.")
            c2.info("**Bridge to TD:** Swap the true return G for a one-step bootstrap estimate R+γV(S') → you've invented TD(0), which can learn from incomplete episodes.")

        # ═════════════════════════════════════════════════════════════════════
        # TAB 6 — ADVANCED IS
        # ═════════════════════════════════════════════════════════════════════
        with tab_adv:
            st.markdown(_card("#9c6dff","🔬","What do Advanced IS methods solve?",
                """<b>The remaining variance problem:</b> Even Weighted IS can be noisy for long episodes.
                Why? The IS weight ρ is a <em>product</em> of per-step ratios. A 50-step episode multiplies
                50 numbers together — even if each ratio is ~1.2, the product can be 1.2⁵⁰ ≈ 9100.
                One such episode dominates the entire average.<br><br>
                <b>Per-Decision IS</b> fixes this by not using future ratios for past rewards.
                Reward R₃ doesn't need the IS ratio from steps 4–50 — those future decisions can't
                change what happened at step 3.<br><br>
                <b>Discounting-Aware IS</b> goes further: with γ&lt;1, rewards at step 50 are
                discounted by γ⁵⁰ ≈ 0.006. Their IS corrections also have less effect on the estimate.
                This method exploits the discount structure to further shorten effective weight products."""),
                unsafe_allow_html=True)

            st.subheader("🔬 Advanced IS: Per-Decision & Discounting-Aware")

            with st.expander("📐 Theory & Formulas — Per-Decision & Discounting-Aware IS", expanded=False):
                st.markdown(r"""
                #### Why Standard IS Has Exploding Variance

                Standard IS weight for a 50-step episode:
                """)
                st.latex(
                    r"\rho_{t:T-1} = \underbrace{\frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}}_{\rho_t} \times"
                    r" \underbrace{\frac{\pi(A_{t+1} \mid S_{t+1})}{b(A_{t+1} \mid S_{t+1})}}_{\rho_{t+1}} \times"
                    r" \cdots \times \rho_{T-1}"
                )
                st.markdown(r"""
                If each ratio is 1.5 (π prefers this action 1.5× more than b):
                $1.5^{50} \approx 637{,}621$ — the weight for this episode is 637,621×
                the average return. Any one such episode destroys the variance.

                #### Per-Decision IS — The Key Insight

                **Reward $R_{k+1}$ at step $k$ is causally independent of actions after step $k$.**
                Future decisions can't reach back in time to change past rewards.

                So instead of weighting the entire return with the full IS product, we weight each
                reward with only the IS ratios *up to* that reward's step:
                """)
                st.latex(
                    r"\boxed{\hat{V}^{PD}(s_t) = \sum_{k=t}^{T-1} \gamma^{k-t}"
                    r" \left(\prod_{j=t}^{k}\rho_j\right) R_{k+1}}"
                )
                st.markdown(r"""
                **Symbol decoder:**
                - $\prod_{j=t}^{k}\rho_j$ — IS weight only up to step $k$ (not the full episode)
                - $\gamma^{k-t}$ — discount applied to reward $k$ steps in the future
                - For reward $R_{t+1}$ (the next reward): only $\rho_t$ is needed (1 ratio, not 50)
                - For reward $R_{t+5}$: only $\rho_t \cdots \rho_{t+4}$ needed (5 ratios, not 50)

                This provably has **lower variance** than ordinary IS — each reward's weight is shorter.

                #### Discounting-Aware IS — Exploiting γ < 1

                With $\gamma < 1$, a discounted return can be decomposed into **flat partial returns**:
                """)
                st.latex(
                    r"G_t = (1-\gamma)\sum_{h=t}^{T-1}\gamma^{h-t}\bar{G}_{t:h} + \gamma^{T-t}\bar{G}_{t:T}"
                )
                st.markdown(r"""
                where $\bar{G}_{t:h} = \sum_{k=t}^{h} R_{k+1}$ is the undiscounted sum to horizon $h$.

                Each partial return $\bar{G}_{t:h}$ only needs IS ratios up to horizon $h$.
                And since $\gamma^{h-t}$ is small for large $h-t$, distant horizons contribute little
                — their IS products have limited influence.

                Result: **effective IS products are shorter** (weighted by discount factors).

                #### Variance Hierarchy — Provably Ordered
                """)
                st.latex(
                    r"\text{Var}[\text{Disc-Aware IS}] \leq \text{Var}[\text{Per-Decision IS}]"
                    r" \leq \text{Var}[\text{Weighted IS}] \leq \text{Var}[\text{Ordinary IS}]"
                )
                st.markdown(r"""
                | Method | Max IS product length | Why |
                |--------|----------------------|-----|
                | Ordinary IS | Full episode $T-t$ | All ratios multiplied |
                | Weighted IS | Full episode $T-t$ | Normalised, but still full product |
                | Per-Decision IS | Up to each step $k-t$ | Shorter products per reward |
                | Discounting-Aware IS | Discounted by $\gamma^{h-t}$ | Distant terms downweighted |

                The hierarchy collapses when $\gamma = 1$ (no discounting):
                Discounting-Aware = Per-Decision IS.
                """)

            st.markdown(_card("#9c6dff","📖","How to read these two heatmaps",
                """Both maps estimate V(s) for the greedy target policy using only 500 episodes (capped for speed).<br>
                <b>Left (Per-Decision IS):</b> Uses shorter IS products per reward. Should be smoother than Ordinary IS.<br>
                <b>Right (Discounting-Aware IS):</b> Exploits γ&lt;1 to further reduce effective product length. Smoothest of all IS methods.<br>
                <b>Grey/pale cells</b> = states rarely visited by the behavior policy, so few IS-corrected samples exist.<br>
                <b>What to look for:</b> Compare these maps to the Ordinary IS map in the previous tab.
                The advanced methods should have fewer extreme values (fewer very-green or very-red cells in unusual places)."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_pd"], "Per-Decision IS — V(s)",      axes[0])
            plot_value_heatmap(env, res["V_da"], "Discounting-Aware IS — V(s)", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.divider()
            st.subheader("📊 Variance Comparison — All IS Methods")
            st.markdown(_card("#9c6dff","📖","How to read the variance bar chart",
                """Each bar = one IS method. Bar height = variance of V(s) estimates across all non-terminal states.<br>
                <b>Lower bar = more consistent estimates = better for practical use.</b><br>
                <b>Expected order</b> (shortest to tallest): Discounting IS → Per-Decision IS → Weighted IS → Ordinary IS.<br>
                If your chart doesn't perfectly follow this order: 500 episodes is a small sample —
                sampling noise can flip adjacent bars. Run more episodes (use the slider) to see the
                theoretical ordering emerge more clearly.<br>
                <b>Colour guide:</b> red=high variance, blue/green=low variance (just visual — not a quality score by itself)."""),
                unsafe_allow_html=True)

            all_s = [s for s in res["V_on"] if s not in env.walls and not env.is_terminal(s)]
            variance_data = {
                "Ordinary IS":    np.var([res["V_ois"].get(s,0) for s in all_s]),
                "Weighted IS":    np.var([res["V_wis"].get(s,0) for s in all_s]),
                "Per-Decision IS":np.var([res["V_pd"].get(s,0)  for s in all_s]),
                "Discounting IS": np.var([res["V_da"].get(s,0)  for s in all_s]),
            }
            fig6, ax6, _ = make_fig(1, 1, 10, 4)
            colors_b = ["#ef5350", "#42a5f5", "#66bb6a", "#ffa726"]
            bars = ax6.bar(list(variance_data.keys()), list(variance_data.values()),
                           color=colors_b, edgecolor="white", lw=0.5, alpha=0.9)
            for bar, val in zip(bars, variance_data.values()):
                ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                         f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
            ax6.set_ylabel("Var(V(s)) across non-terminal states", color="white", fontsize=11)
            ax6.set_title("IS Variance Hierarchy — Lower Is Better", color="white", fontweight="bold")
            ax6.grid(alpha=0.15, axis="y")
            plt.tight_layout(); st.pyplot(fig6); plt.close()

            st.markdown(_warn("""
            <b>Important caveat:</b> "Lowest variance" does NOT mean "best" in all cases.
            Discounting-Aware IS has zero benefit when γ=1 (no discounting).
            Per-Decision IS is more complex to implement correctly.
            In practice, <b>Weighted IS is usually the first choice</b> — it offers a good balance
            of low variance, correctness, and implementation simplicity.
            """), unsafe_allow_html=True)

        # ═════════════════════════════════════════════════════════════════════
        # TAB 7 — DASHBOARD
        # ═════════════════════════════════════════════════════════════════════
        with tab_dash:
            st.markdown(_card("#ce93d8","📈","What does this dashboard show?",
                """All 9 MC methods running on the <em>same</em> gridworld with the <em>same</em> settings.
                This lets you see directly: do they agree on which states are good and bad? How much
                does their accuracy differ? Where does each method struggle?<br>
                The dashboard is the culmination — every other tab shows one method; this shows all of them together."""),
                unsafe_allow_html=True)
            st.subheader("📈 Full Comparison Dashboard — All 9 Methods")

            st.markdown("### 🗺️ Value Functions — All 9 Methods Side-by-Side")
            st.markdown(_card("#ce93d8","📖","How to read the 3×3 grid of heatmaps",
                """Every map uses the same colour scale (red=−5, green=+10).<br>
                <b>What they should all agree on:</b> Green near (4,4), dark/red near (2,2), moderate elsewhere.<br>
                <b>Where they differ:</b><br>
                &nbsp;• IS methods (maps 4–9) may have grey cells — states the behavior policy didn't visit with compatible actions<br>
                &nbsp;• Ordinary IS (map 4) may have unexpected very-green or very-red cells — high-ρ episodes dominating<br>
                &nbsp;• Advanced IS (maps 7,8) should be smoother than Ordinary IS with same data<br>
                <b>Maps 1–3</b> (on-policy methods) should be most consistent — they learn directly from the target policy."""),
                unsafe_allow_html=True)

            methods_v = [
                (res["V_fv"],  "1. First-Visit MC"),
                (res["V_ev"],  "2. Every-Visit MC"),
                (res["V_on"],  "3. On-policy Control"),
                (res["V_off"], "4. Off-policy Control"),
                (res["V_ois"], "5. Ordinary IS"),
                (res["V_wis"], "6. Weighted IS"),
                (res["V_inc"], "7. Incremental MC"),
                (res["V_pd"],  "8. Per-Decision IS"),
                (res["V_da"],  "9. Discounting IS"),
            ]
            fig, axes, axl = make_fig(3, 3, 18, 15)
            for idx2, (V, title) in enumerate(methods_v):
                ax = axes[idx2//3][idx2%3]
                ax.set_facecolor(DARK_BG)
                ax.tick_params(colors="#9e9ebb", labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
                plot_value_heatmap(env, V, title, ax)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Summary table
            st.divider()
            st.markdown("### 📋 Accuracy & Coverage Summary Table")
            st.markdown(_card("#ce93d8","📖","How to read the summary table",
                """<b>MSE vs On-policy:</b> Mean squared error of each method's V(s) estimates compared
                to the on-policy reference. Lower = closer to the "ground truth." Methods that use IS
                may have higher MSE simply due to variance — not because they're wrong in expectation.<br>
                <b>Var(V):</b> How spread out the value estimates are. High variance = estimates are
                jumping around a lot — the method needs more data to stabilise.<br>
                <b>Coverage:</b> What fraction of non-terminal states got an estimate.
                IS methods may miss states the behavior policy never visited with the right action sequence."""),
                unsafe_allow_html=True)

            all_s2 = [s for s in res["V_on"] if s not in env.walls and not env.is_terminal(s)]
            ref_v  = np.array([res["V_on"].get(s, 0) for s in all_s2])
            rows = []
            for V, title in methods_v:
                vals = np.array([V.get(s, 0) for s in all_s2])
                mse  = float(np.mean((vals - ref_v)**2))
                var  = float(np.var(vals))
                cov  = sum(1 for s in all_s2 if s in V) / len(all_s2) * 100
                rows.append({"Method": title, "MSE vs On-policy ↓": f"{mse:.4f}",
                             "Var(V) ↓": f"{var:.4f}", "Coverage": f"{cov:.0f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Variance/stability bars
            st.divider()
            st.markdown("### 📊 Expert Variance & Stability Scores")
            st.markdown("""
            The bars below show **qualitative expert scores** from RL theory — not computed from your
            specific run. They reflect what happens across many environments with large sample sizes.
            These scores are a teaching tool to understand the theoretical ordering.
            """)
            st.markdown(_card("#ce93d8","📖","How to read the two bar charts",
                """<b>Left (Variance — lower is better):</b> Bars extend left. Shorter bar = less noisy estimates.
                Ordinary IS has the longest bar (most noisy). Discounting IS has the shortest (least noisy).<br>
                <b>Right (Stability — higher is better):</b> Bars extend right. Longer bar = more reliable across
                different environments, seeds, and episode counts.<br>
                <b>Note:</b> Low variance and high stability are correlated but not identical. Weighted IS is
                very stable despite not having the absolute lowest variance."""),
                unsafe_allow_html=True)

            labels  = ["FV MC","EV MC","On-pol","Off-pol","Ord IS","WIS","Incr MC","Per-Dec IS","Disc IS"]
            var_sc  = [2, 3, 4, 5, 9, 3, 2, 2, 1]
            stab_sc = [9, 8, 7, 6, 3, 8, 9, 8, 9]

            fig7, axes7, _ = make_fig(1, 2, 14, 5)
            cbar = ["#7c4dff","#9c6dff","#00897b","#1565c0","#ef5350","#42a5f5","#ff9800","#ffa726","#66bb6a"]
            axes7[0].barh(labels, var_sc,  color=cbar, alpha=0.88, edgecolor="white", lw=0.4)
            axes7[0].set_xlabel("Variance score (lower = less noisy) ←", color="white")
            axes7[0].set_title("Relative Variance", color="white", fontweight="bold")
            axes7[0].invert_xaxis()
            axes7[0].grid(alpha=0.15, axis="x")
            axes7[1].barh(labels, stab_sc, color=cbar, alpha=0.88, edgecolor="white", lw=0.4)
            axes7[1].set_xlabel("Stability score (higher = more reliable) →", color="white")
            axes7[1].set_title("Stability", color="white", fontweight="bold")
            axes7[1].grid(alpha=0.15, axis="x")
            plt.tight_layout(); st.pyplot(fig7); plt.close()

            # Method evolution diagram
            st.divider()
            st.markdown("### 🌳 MC Method Family Tree")
            st.markdown(_card("#ce93d8","📖","How to read the family tree",
                """Each circle = one MC method. Arrows show which method led to the next in the history
                of RL research — each branch was invented to solve a specific limitation of its parent.<br>
                <b>Left side = simpler, more fundamental.</b> Right side = more advanced, lower variance.<br>
                Reading the arrows: "this method's limitation motivated the invention of that method."
                For example: On-policy control (ε-soft limit) → Off-policy control (true optimality).
                Standard IS (variance explosion) → Per-Decision IS → Discounting-Aware IS (progressively lower variance)."""),
                unsafe_allow_html=True)

            fig8, ax8, _ = make_fig(1, 1, 14, 6)
            ax8.axis("off")
            nodes = [
                (0.05, 0.50, "MC\nPrediction",    "#7c4dff"),
                (0.22, 0.80, "First-Visit\nMC",   "#5c35cc"),
                (0.22, 0.20, "Every-Visit\nMC",   "#9c6dff"),
                (0.42, 0.65, "On-policy\nControl","#00897b"),
                (0.42, 0.35, "Incremental\nMC",   "#ff9800"),
                (0.60, 0.80, "Off-policy\nControl","#1565c0"),
                (0.60, 0.50, "Ordinary\nIS",       "#ef5350"),
                (0.60, 0.20, "Weighted\nIS",       "#42a5f5"),
                (0.82, 0.65, "Per-Decision\nIS",   "#ffa726"),
                (0.82, 0.35, "Disc-Aware\nIS",     "#66bb6a"),
            ]
            edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(3,5),(3,6),(6,7),(5,8),(7,8),(8,9)]
            for x, y, lbl, col in nodes:
                ax8.add_patch(plt.Circle((x,y), 0.072, color=col, alpha=0.88,
                                         transform=ax8.transAxes, zorder=3))
                ax8.text(x, y, lbl, ha="center", va="center", fontsize=6.5, color="white",
                         fontweight="bold", transform=ax8.transAxes, zorder=4)
            for i, j in edges:
                x0, y0 = nodes[i][0], nodes[i][1]
                x1, y1 = nodes[j][0], nodes[j][1]
                ax8.annotate("", xy=(x1,y1), xytext=(x0,y0),
                             xycoords="axes fraction", textcoords="axes fraction",
                             arrowprops=dict(arrowstyle="->", color="#90a4ae", lw=1.5))
            ax8.set_title("MC Method Family Tree — Each Method Solves a Limitation of Its Parent",
                          color="white", fontweight="bold", pad=16)
            plt.tight_layout(); st.pyplot(fig8); plt.close()

            # Bias-variance scatter
            st.divider()
            st.markdown("### 🎯 Bias–Variance Landscape")
            st.markdown(_card("#ce93d8","📖","How to read the bias-variance scatter plot",
                """This is the most important theoretical summary chart in the app.<br>
                <b>X-axis (Bias →):</b> How systematically wrong the method is. Bias=0 means the estimate
                is correct in expectation (averaging over infinite episodes). Weighted IS has some bias
                for finite N because the weight denominator introduces a systematic under/over-correction.<br>
                <b>Y-axis (Variance →):</b> How much estimates fluctuate across different runs.
                High variance = different seeds give very different results = unreliable.<br>
                <b>Green bottom-left corner = ideal:</b> both low bias AND low variance.<br>
                <b>Ordinary IS (top-left):</b> Unbiased but highest variance — mathematically pure, practically unstable.<br>
                <b>Discounting IS (bottom-right):</b> Tiny bias, lowest variance — best practical choice when γ&lt;1."""),
                unsafe_allow_html=True)

            bv_data = {
                "FV MC":       (1.0, 2.0, "#7c4dff"),
                "EV MC":       (1.8, 2.8, "#9c6dff"),
                "On-policy":   (2.5, 4.0, "#00897b"),
                "Off-policy":  (2.0, 5.0, "#1565c0"),
                "Ordinary IS": (0.5, 9.0, "#ef5350"),
                "Weighted IS": (3.0, 2.5, "#42a5f5"),
                "Incremental": (1.0, 2.2, "#ff9800"),
                "Per-Dec IS":  (1.2, 1.8, "#ffa726"),
                "Disc. IS":    (1.5, 1.2, "#66bb6a"),
            }
            fig9, ax9, _ = make_fig(1, 1, 9, 5)
            ax9.add_patch(plt.Rectangle((0,0),3,3, alpha=0.07, color="green", zorder=0))
            ax9.text(1.5, 0.3, "✓ Ideal zone\n(low bias, low variance)", color="#81c784",
                     fontsize=8, ha="center")
            for name, (bv, var, col) in bv_data.items():
                ax9.scatter(bv, var, s=220, color=col, zorder=5, edgecolors="white", lw=1.2)
                ax9.annotate(name, (bv, var), xytext=(8, 5), textcoords="offset points",
                             color="white", fontsize=8)
            ax9.set_xlabel("Relative Bias  (systematic error) →", color="white", fontsize=11)
            ax9.set_ylabel("Relative Variance  (instability) →",   color="white", fontsize=11)
            ax9.set_title("Bias–Variance Tradeoff: All 9 MC Methods", color="white", fontweight="bold")
            ax9.set_xlim(0, 10); ax9.set_ylim(0, 10)
            ax9.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig9); plt.close()

    else:
        for tab in [tab_pred, tab_ctrl, tab_offpol, tab_offctrl, tab_incr, tab_adv, tab_dash]:
            with tab:
                st.info("👈 Press **Run All 9 Methods** in the sidebar to generate all charts and analyses.")

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 8 — METHOD GUIDE (always visible)
    # ═════════════════════════════════════════════════════════════════════════
    with tab_guide:
        st.subheader("📚 Complete MC Method Reference")

        st.markdown("""
        <div style="background:#12121f; border-radius:12px; padding:1.2rem 1.6rem; border:1px solid #2a2a3e; margin-bottom:1rem">

        ### 🧠 One-Sentence Core Intuition for Each Method

        | # | Method | Core intuition |
        |---|--------|---------------|
        | 1 | **First-Visit MC** | Play many games; for each game, record the total reward from the FIRST time you visited each location; average across games |
        | 2 | **Every-Visit MC** | Same but count EVERY visit to a location within a game — more data, slight statistical correlation |
        | 3 | **On-policy Control** | Improve your strategy after every game, keeping a small chance of random moves so you never stop exploring |
        | 4 | **Off-policy IS Evaluation** | Correct old data collected by a random strategy to evaluate a smarter one — using a statistical reweighting trick |
        | 5 | **Off-policy MC Control** | Learn a pure greedy optimal strategy while a separate exploration strategy collects all the data |
        | 6 | **Incremental MC** | Instead of storing every past result, maintain only a running average — same answer, O(1) memory |
        | 7 | **Per-Decision IS** | Each reward in an episode only needs the IS correction up to that step, not the whole episode — shorter products, lower variance |
        | 8 | **Discounting-Aware IS** | When γ&lt;1, distant rewards matter less — so their IS corrections also matter less — exploit this for the lowest possible variance |

        </div>
        """, unsafe_allow_html=True)

        # Decision guide
        with st.expander("🗺️ Which MC method should I use? — Decision Guide", expanded=True):
            st.markdown("""
            ```
            START HERE
              │
              ├─ Do I know the full model p(s'|s,a)?
              │    └─ YES → Use Dynamic Programming (not MC)
              │    └─ NO  → Continue ↓
              │
              ├─ Do I need to improve the policy or just evaluate it?
              │    ├─ EVALUATE ONLY (fixed policy)
              │    │    ├─ Simple / memory OK → First-Visit MC
              │    │    ├─ Want more data per episode → Every-Visit MC
              │    │    └─ Memory constrained / streaming → Incremental MC
              │    │
              │    └─ IMPROVE THE POLICY (control)
              │         ├─ Data from same policy I'm improving → On-policy MC Control (ε-greedy)
              │         │    └─ Acceptable limitation: converges to BEST ε-SOFT policy (not true π*)
              │         │
              │         └─ Want the TRUE OPTIMAL policy → Off-policy MC Control
              │              └─ Acceptable limitation: learns only from greedy episode tail (slow)
              │
              └─ Am I using off-policy data (behavior ≠ target)?
                   ├─ EVALUATION only
                   │    ├─ Need unbiased estimate → Ordinary IS (but prepare for high variance)
                   │    └─ Need stable estimates → Weighted IS (slight bias, much lower variance)
                   │         ├─ Episodes are long → Per-Decision IS (shorter IS products)
                   │         └─ γ < 1 (discounted) → Discounting-Aware IS (lowest variance)
                   │
                   └─ CONTROL (improve policy too) → Off-policy MC Control
            ```
            """)

        entries = [
            {
                "icon": "🔵", "name": "MC Prediction — First-Visit",
                "what": "Estimates **V(s)** by averaging the return G from the **first** visit to each state per episode. Walk backward through the episode; if you haven't seen this state before, record its return.",
                "when": "Evaluating a fixed policy. Theoretical analyses where statistical independence matters. When you can afford to occasionally miss a state in an episode.",
                "pros": "✅ Provably unbiased (correct expected value) | ✅ Independent samples per episode | ✅ Clean theoretical guarantees | ✅ Error ∝ 1/√n",
                "cons": "❌ Ignores revisits within an episode (wasted data) | ❌ Requires complete episodes | ❌ Converges slowly for rarely-visited states",
                "variance": "**LOW.** Each episode contributes exactly one G per state. Samples from different episodes are independent → classical √n statistics apply.",
                "relation": "Foundation of all MC methods. Every-Visit MC relaxes the uniqueness constraint. Incremental MC is its memory-efficient implementation.",
            },
            {
                "icon": "🟣", "name": "MC Prediction — Every-Visit",
                "what": "Like First-Visit but counts **every** occurrence of a state in an episode. A state visited 5 times contributes 5 return samples — one from each visit's remaining trajectory.",
                "when": "When sample efficiency matters more than theoretical purity. When states are frequently revisited and you want to extract maximum information per episode.",
                "pros": "✅ More updates per episode | ✅ Faster practical convergence | ✅ Simpler to implement (no 'visited' tracking needed) | ✅ Preferred in practice (S&B §5.1)",
                "cons": "❌ Correlated samples within one episode | ❌ Biased for finite N (but consistent — bias→0 as N→∞) | ❌ Slight instability in loopy environments",
                "variance": "**MEDIUM.** More samples but correlated — the returns from the 2nd and 3rd visit to a state in one episode share the same future trajectory.",
                "relation": "Same asymptotic limit as First-Visit. In practice, often preferred. Converges to V^π(s) as N→∞ for both methods.",
            },
            {
                "icon": "🟢", "name": "On-policy MC Control (ε-greedy GPI)",
                "what": "Estimates **Q(s,a)** action-values and improves the policy after every episode using ε-greedy Generalised Policy Iteration. No model needed — Q directly tells the agent which action is best.",
                "when": "Learning from direct interaction. When you need a simple, effective baseline. When the environment is not changing. When the ε-soft policy limitation is acceptable.",
                "pros": "✅ Model-free | ✅ Simple to implement | ✅ Guaranteed exploration (all s,a pairs visited) | ✅ Converges reliably",
                "cons": "❌ Converges only to best ε-soft policy (always retains ε-exploration) | ❌ Cannot achieve true optimal π* | ❌ Requires complete episodes",
                "variance": "**MEDIUM.** Q-value estimates stabilise with N. The ε-greedy policy introduces stochastic noise in returns — each episode may take slightly different paths.",
                "relation": "On-policy counterpart to off-policy Q-learning. Remove ε constraint → need off-policy methods. Use V(s) → need a model. Add bootstrapping → get Sarsa (TD-based).",
            },
            {
                "icon": "🔵", "name": "Off-policy MC Control (true π*)",
                "what": "Learns a **deterministic greedy target policy π*** while generating all data with a separate ε-soft behavior policy. Updates Q via weighted IS, but only from the greedy tail of each episode.",
                "when": "When you need the **true optimal policy**, not just best ε-soft. When exploration and learning must be completely separated — e.g. robotics safety, clinical medicine, learning from demonstrations.",
                "pros": "✅ Converges to true optimal π* | ✅ Exploration kept separate | ✅ Can reuse data from any soft policy | ✅ Target policy can be evaluated separately",
                "cons": "❌ Only learns from greedy tail (slow) | ❌ High variance (inherits IS noise) | ❌ Requires coverage (b must cover all π actions) | ❌ Complex to implement correctly",
                "variance": "**HIGH.** Combines IS variance with the greedy-tail limitation. Each episode contributes fewer updates. Practically, often needs more episodes than on-policy.",
                "relation": "Off-policy counterpart to on-policy MC control. Uses the incremental weighted-IS update (§5.6). The learned Q can be evaluated separately with ordinary/weighted IS.",
            },
            {
                "icon": "🔴", "name": "Off-policy — Ordinary Importance Sampling",
                "what": "Reweights returns from behavior policy b to evaluate target policy π. Each return G is multiplied by the IS ratio ρ = ∏(π/b), then ordinary averaging is applied.",
                "when": "When an **unbiased** estimate is essential (e.g., formal hypothesis testing). When comparing different IS estimators. As a stepping stone to understanding weighted IS.",
                "pros": "✅ Provably unbiased | ✅ Can evaluate any target policy | ✅ Simpler formula | ✅ Easier to extend to function approximation",
                "cons": "❌ Extremely high variance — ρ products can be astronomically large | ❌ One outlier episode can corrupt thousands of episodes of learning | ❌ Practically unusable for long episodes",
                "variance": "**VERY HIGH.** Each ratio ρ_k can be >1. Their product over 50 steps can be 10⁶. One such episode gives an estimate 10⁶× larger than typical. Variance is theoretically unbounded.",
                "relation": "The starting point for off-policy IS. Weighted IS fixes the variance problem with a small bias. Per-Decision IS reduces the product length.",
            },
            {
                "icon": "🔵", "name": "Off-policy — Weighted Importance Sampling",
                "what": "Like Ordinary IS but uses a weighted average: Σ(ρ·G)/Σρ. Each return is weighted by ρ, but the denominator normalises so no single episode can have weight >1.",
                "when": "Almost always preferred over Ordinary IS. The practical default for off-policy evaluation. When you want stable, reliable estimates from off-policy data.",
                "pros": "✅ Dramatically lower variance | ✅ Practically stable | ✅ Consistent (bias→0) | ✅ Max weight on any single return is bounded at 1",
                "cons": "❌ Biased for finite N | ❌ Bias can be significant with very few episodes | ❌ Slightly more complex formula",
                "variance": "**LOW.** The denominator Σρ normalises all weights. Even if ρ=10,000 for one episode, its effective weight is 10,000/(total weight) which is typically small.",
                "relation": "The practical replacement for Ordinary IS. Per-Decision IS further reduces variance by decomposing the IS product per reward.",
            },
            {
                "icon": "⚡", "name": "Incremental MC",
                "what": "Implements First-Visit MC with a running mean update: V(s) += (1/N)(G − V(s)). Mathematically equivalent to batch averaging but uses O(1) memory per state.",
                "when": "Any MC application where memory is constrained. Streaming data. Non-stationary environments (with fixed α). As the bridge toward understanding TD learning.",
                "pros": "✅ O(1) memory (constant per state) | ✅ Identical estimates to batch MC | ✅ Easily extended to non-stationary (fixed α) | ✅ Natural bridge to TD learning",
                "cons": "❌ No variance improvement over First-Visit | ❌ Fixed α introduces bias in stationary settings | ❌ Requires care to choose α correctly",
                "variance": "**LOW** (same as First-Visit with 1/N step-size). Fixed α variant trades bias for tracking ability — higher α = faster to forget old data but more variance.",
                "relation": "Implementation of First-Visit MC. Replace G with R+γV(S') → you get TD(0). Use weighted IS weights W → off-policy weighted IS (§5.6 algorithm).",
            },
            {
                "icon": "🟠", "name": "Per-Decision Importance Sampling",
                "what": "Decomposes the IS weight per reward: each Rₖ is weighted only by the product of IS ratios up to step k. Future ratios don't affect past rewards, so there's no need to include them.",
                "when": "Off-policy evaluation with long episodes (50+ steps). When weighted IS variance is still too high. When you can afford more implementation complexity for lower variance.",
                "pros": "✅ Provably lower variance than Ordinary and Weighted IS | ✅ Unbiased | ✅ Shorter IS products per reward",
                "cons": "❌ More complex implementation | ❌ Still requires complete episodes | ❌ Benefit diminishes for short episodes",
                "variance": "**VERY LOW.** IS products are truncated at each reward's step. Reward R_{t+1} only uses ρ_t (1 ratio). Reward R_{t+5} only uses ρ_t·...·ρ_{t+4} (5 ratios, not T−t ratios).",
                "relation": "Intermediate between Weighted IS and Discounting-Aware IS. When γ=1: Discounting-Aware IS reduces to Per-Decision IS.",
            },
            {
                "icon": "🟡", "name": "Discounting-Aware Importance Sampling",
                "what": "Exploits the discount factor γ<1 to further reduce variance. Decomposes the return into flat partial returns, each requiring only IS ratios up to its horizon — and downweights distant horizons by their discount factor.",
                "when": "Maximum variance reduction in off-policy evaluation. When γ<1 (discounted tasks) and episodes are long. When you need the most accurate off-policy estimates possible.",
                "pros": "✅ Lowest variance of all IS estimators | ✅ Exploits discount structure | ✅ Unbiased | ✅ Theoretically optimal for discounted MDPs",
                "cons": "❌ Most complex to implement correctly | ❌ Zero benefit when γ=1 | ❌ Diminishing returns for short episodes | ❌ Less standard — fewer reference implementations",
                "variance": "**LOWEST.** Distant horizons are discounted by γ^h, reducing their IS product's influence. Effective product length ≈ mean discounted episode length (much shorter than T).",
                "relation": "Most advanced IS method. Reduces to Per-Decision IS when γ=1. Uses the same incremental weighted-IS update as the Incremental MC section.",
            },
        ]

        for e in entries:
            with st.expander(f"{e['icon']} {e['name']}", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**📌 What it does:**\n\n{e['what']}")
                    st.markdown("---")
                    st.markdown(f"**🕐 When to use it:**\n\n{e['when']}")
                    st.markdown("---")
                    st.markdown(f"**✅ Pros:**\n\n{e['pros']}")
                with c2:
                    st.markdown(f"**❌ Cons:**\n\n{e['cons']}")
                    st.markdown("---")
                    st.markdown(f"**📊 Variance behaviour:**\n\n{e['variance']}")
                    st.markdown("---")
                    st.markdown(f"**🔗 Relation to other methods:**\n\n{e['relation']}")

        st.divider()
        st.markdown(r"""
        ### ❓ Frequently Asked Questions

        **Q: Why must MC wait for the full episode? Can't it learn faster?**

        MC uses the *true* return G — the actual total reward from now to the end.
        You can't know G until the episode ends. TD methods solve this by *estimating* G
        using $R_{t+1} + \gamma V(S_{t+1})$ — they learn after every step but introduce
        bias (they're guessing what G will be). MC: unbiased but patient. TD: biased but fast.

        ---

        **Q: What does "high variance" feel like in practice?**

        Run this app with 200 episodes, then run it again with a different seed.
        For Ordinary IS, the value maps will look completely different between runs.
        For First-Visit MC, they'll look similar. That inconsistency between runs *is* high variance.
        In real systems: a high-variance estimator might tell you a drug is effective in one clinical
        trial and harmful in the next — using the same data collection process.

        ---

        **Q: Why use off-policy methods at all if they have higher variance?**

        Three reasons: **(1) Reusing old data** — if you have historical logs from a system,
        you can evaluate new policies without running new (potentially expensive or dangerous) experiments.
        **(2) Safety** — evaluate a risky target policy using data from a safe behavior policy.
        **(3) True optimality** — on-policy control is permanently constrained to ε-soft policies.
        Off-policy control can find the true optimal.

        ---

        **Q: When should I use MC vs TD learning?**

        | Situation | Prefer |
        |-----------|--------|
        | Episodes always terminate cleanly | Either |
        | No clean episode end (continuous tasks) | **TD** (MC can't apply) |
        | Need unbiased estimates | **MC** |
        | Need fast online learning | **TD** |
        | Environments with high variance returns | **TD** (less affected) |
        | Theoretical analysis / interpretability | **MC** |

        ---

        **Q: Does discounting (γ<1) actually make the problem easier?**

        Yes — in two ways. First, it bounds the magnitude of returns (finite sum even for infinite episodes).
        Second, it reduces the effective IS product length for Discounting-Aware IS. But it also changes
        the objective: you're now optimising *discounted* total reward, which may not match what you actually want.
        γ=0.99 is a common practical choice — close enough to 1 to approximate undiscounted reward while
        providing numerical stability.
        """)


if __name__ == "__main__":
    main()
