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
from _notes_mod import render_notes
warnings.filterwarnings("ignore")


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


def render_mc_notes(tab_title: str, tab_slug: str) -> None:
    render_notes(f"Monte Carlo Methods - {tab_title}", tab_slug)

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
    Q_on, pi_on, ep_rewards = mc_control_on_policy(env, n_episodes, epsilon, gamma)
    V_on   = {s: float(np.max(Q_on[s])) for s in Q_on}
    pi_det = env.greedy_policy(Q_on)   # deterministic target policy for off-policy

    # 3. Off-policy IS (target=greedy from Q_on, behavior=random)
    V_ois = mc_ordinary_is(env, pi_det, b_policy, n_episodes, gamma)
    V_wis = mc_weighted_is(env, pi_det, b_policy, n_episodes, gamma)

    # 4. Incremental MC
    V_inc, var_hist = mc_incremental(env, b_policy, n_episodes, gamma)

    # 5. Advanced IS (cap at 500 for speed)
    n_adv  = min(n_episodes, 500)
    V_pd   = mc_per_decision_is(env, pi_det, b_policy, n_adv, gamma)
    V_da   = mc_discounting_aware_is(env, pi_det, b_policy, n_adv, gamma)

    return dict(
        V_fv=V_fv, hist_fv=hist_fv,
        V_ev=V_ev, hist_ev=hist_ev,
        Q_on=Q_on, V_on=V_on, pi_on=pi_on, pi_det=pi_det, ep_rewards=ep_rewards,
        V_ois=V_ois, V_wis=V_wis,
        V_inc=V_inc, var_hist=var_hist,
        V_pd=V_pd, V_da=V_da,
        b_policy=b_policy,
    )


def main_mc():
    # ── Banner ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a237e,#4a148c,#006064);
                padding:2rem 2.5rem; border-radius:14px; margin-bottom:1.5rem">
        <h1 style="color:white;margin:0;font-size:2.4rem">🎲 Monte Carlo RL Explorer</h1>
        <p style="color:#b0bec5;margin-top:.5rem;font-size:1.05rem">
            An interactive visual textbook — from MC Prediction to Discounting-Aware Importance Sampling
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── What is Monte Carlo? — always-visible intro ───────────────────────────
    with st.expander("🎓 New here? Start with this — What is Monte Carlo Reinforcement Learning?", expanded=False):
        st.markdown("""
        <div style="background:#12121f; border-radius:12px; padding:1.4rem 1.6rem; border:1px solid #2a2a3e">

        ### 🤔 The Core Problem: How Does an Agent Learn Good Behaviour?

        Imagine you drop a robot into a maze it has never seen.  It doesn't know the map.
        It can't ask anyone.  It can only *try things* and *observe what happens*.

        > **Reinforcement Learning (RL)** is the family of algorithms that let an agent learn
        > from its own experience — exactly like a child learning to walk by falling and trying again.

        ---

        ### 🎲 So What is "Monte Carlo" in RL?

        The name comes from the famous Monaco casino — a nod to *randomness*.

        A **Monte Carlo method** learns by playing complete games (called *episodes*) from start to finish,
        then looking back at what happened and updating its beliefs.

        Think of it like a chess player who:
        1. Plays a full game
        2. At the end, replays the game in their head
        3. Asks *"which moves led to my win or loss?"*
        4. Updates their strategy for next time

        **Key insight:** MC methods never guess or extrapolate — they only learn from *actual observed outcomes*.
        This makes them **unbiased** but potentially **slow** (they need complete episodes).

        ---

        ### 🆚 What Problem Is Each Method Solving?

        | Method | The Problem It Solves |
        |--------|-----------------------|
        | **MC Prediction** | *"How good is it to be in state S under this fixed policy?"* |
        | **MC Control** | *"What is the best action to take in each state — and how do I find it?"* |
        | **Off-policy IS** | *"I collected data with a random exploration strategy — can I still evaluate a smarter strategy from that same data?"* |
        | **Incremental MC** | *"I don't have memory to store thousands of past returns — how do I update efficiently?"* |
        | **Per-Decision IS** | *"Importance sampling weights explode for long episodes — how do I reduce variance?"* |
        | **Discounting-Aware IS** | *"When future rewards matter less (γ < 1), can I exploit that structure to reduce noise even further?"* |

        ---

        ### 🔑 Three Concepts You Need to Understand Everything Else

        **1. Return (G)** — The total reward collected from a point in time until the end of the episode.
        Not just the immediate reward — all future rewards, discounted by γ.
        *Think: total career earnings of a chess move, not just the points it scores right now.*

        **2. Value V(s)** — The *average* return the agent expects if it starts from state s and follows its policy.
        *Think: average exam score you expect from a particular study method.*

        **3. Policy (π)** — The agent's decision rule: "given I'm in state s, what action should I take?"
        *Think: a recipe or strategy guide.*

        > **This entire app** builds these three concepts up from scratch across 8 methods,
        > each solving a slightly harder problem than the last.

        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Experiment Settings")
        st.caption("Adjust these and re-run to see how each method responds.")
        n_episodes = st.slider("Episodes", 100, 3000, 800, 100,
            help="How many complete game runs (start→end) the agent plays. More = better estimates, slower run.")
        gamma      = st.slider("Discount γ", 0.80, 1.00, 0.99, 0.01,
            help="How much the agent values future rewards vs immediate ones. γ=1 means future matters equally; γ=0.9 means rewards 10 steps away are worth only 35% of immediate ones.")
        epsilon    = st.slider("ε (epsilon-greedy)", 0.01, 0.50, 0.10, 0.01,
            help="The exploration rate. ε=0.1 means the agent picks a random action 10% of the time. Without exploration the agent gets stuck in a rut.")
        slip_prob  = st.slider("Slip Probability", 0.0, 0.30, 0.10, 0.05,
            help="Chance that the agent's chosen action is replaced by a random one — simulating a slippery floor or noisy motors. Makes the environment stochastic (unpredictable).")
        seed       = st.number_input("Random Seed", 0, 9999, 42,
            help="Fixing the seed makes results reproducible. Change it to see different random runs.")

        run_btn = st.button("🚀 Run All Methods", type="primary", use_container_width=True)

        st.divider()
        st.markdown("""
        **5×5 Gridworld**
        | Symbol | Meaning |
        |--------|---------|
        | ● | Start (0,0) |
        | ★ | Goal (4,4) +10 |
        | ✗ | Trap (2,2) −5 |
        | ■ | Wall (bounce) |
        | — | Step cost −0.1 |
        """)
        st.divider()
        st.markdown("""
        **MC Pipeline:**
        ```
        Prediction
          ├─ First-Visit MC
          └─ Every-Visit MC
        Control
          └─ On-policy ε-greedy
        Off-policy
          ├─ Ordinary IS
          ├─ Weighted IS
          ├─ Per-Decision IS
          └─ Discounting-Aware IS
        Incremental MC
        ```
        """)

    env = GridWorld(size=5, slip_prob=slip_prob)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_env, tab_pred, tab_ctrl, tab_offpol, tab_incr, tab_adv, tab_dash, tab_guide = st.tabs([
        "🗺️ Environment",
        "📊 MC Prediction",
        "🎯 On-policy Control",
        "⚖️ Off-policy IS",
        "⚡ Incremental MC",
        "🔬 Advanced IS",
        "📈 Dashboard",
        "📚 Method Guide",
    ])

    # ── Tab 0: Environment ────────────────────────────────────────────────────
    with tab_env:
        st.markdown("""
        <div style="background:#0d1b2a; border-left:4px solid #42a5f5; padding:1rem 1.2rem;
                    border-radius:0 10px 10px 0; margin-bottom:1rem">
        <b>🧩 What problem does this tab solve?</b><br>
        Before any MC algorithm can run, we need an <em>environment</em> — a world with states, actions,
        and rewards. This tab shows you the exact world all 8 methods will navigate.
        Understanding the environment is the first step to understanding why the methods behave differently.
        </div>
        """, unsafe_allow_html=True)
        st.subheader("🗺️ The 5×5 Gridworld")
        c1, c2 = st.columns([1.1, 0.9])

        with c1:
            st.markdown("""
            The agent moves on a 5×5 grid.  Terminal states end the episode immediately.

            **Why this environment for MC?**
            - MC requires **complete episodes** (must reach a terminal state)
            - Multiple paths → on-policy vs off-policy behavior differs meaningfully
            - Trap forces variance in returns — good for IS comparisons
            - Walls create asymmetric navigation — policy arrows become informative

            **Reward structure:**
            | Event | Reward |
            |-------|--------|
            | Reach Goal | +10.0 |
            | Reach Trap | −5.0  |
            | Each step   | −0.1  |
            
            **Stochastic slip**: with probability *p* the chosen action is  
            replaced by a uniform-random action — tests robustness of learned policies.
            """)

        with c2:
            st.caption("📖 **How to read this diagram:** Each cell is a grid square the agent can stand on. Colours identify special cells. The agent starts top-left and tries to reach bottom-right without falling into the trap.")
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
                    ax.text(j, i, lbl, ha="center", va="center", fontsize=7,
                            color="white", fontweight="bold")
            ax.set_xticks(range(5)); ax.set_yticks(range(5))
            ax.tick_params(colors="#9e9ebb")
            ax.invert_yaxis()
            ax.set_title("Gridworld Layout", color="white", fontweight="bold")
            for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        # Sample episode
        st.markdown("---")
        st.subheader("🎬 Sample Episode (random policy)")
        st.markdown("""
        > **What is an episode?** One full run of the game — from Start to a terminal state (Goal or Trap).
        > MC methods can *only learn after a complete episode ends*, because they need the final outcome to
        > calculate the return G. This is the fundamental difference from TD (Temporal-Difference) methods,
        > which can learn after every single step.
        >
        > The path below shows one random episode — notice how chaotic a random policy is.
        > MC's job is to make sense of hundreds of these chaotic runs and extract something useful.
        """)
        np.random.seed(seed)
        ep_sample = env.generate_episode(env.uniform_policy(), max_steps=50)
        path_str = " → ".join(f"{s}" for s, a, r in ep_sample)
        total_r  = sum(r for _, _, r in ep_sample)
        st.caption("📖 How to read this output — **Length**: how many steps before hitting a terminal state. **Return**: total reward collected (negative = mostly step-costs and possibly the trap; near +10 = reached the goal efficiently). **Path**: the exact sequence of (row, col) grid squares visited in order.")
        st.code(f"Length : {len(ep_sample)} steps\nReturn : {total_r:.2f}\nPath   : {path_str}", language="")
        render_mc_notes("Environment", "monte_carlo_methods_environment")

    # ── Run all methods ───────────────────────────────────────────────────────
    if run_btn or "results" in st.session_state:
        if run_btn:
            with st.spinner("Running all 8 MC methods… (this takes a few seconds)"):
                res = run_all_methods(env, n_episodes, gamma, epsilon, seed)
            st.session_state["results"] = res
            st.sidebar.success("✅ Done!")

        res = st.session_state["results"]

        # ── Tab 1: MC Prediction ──────────────────────────────────────────────
        with tab_pred:
            st.markdown("""
            <div style="background:#1a0a2e; border-left:4px solid #7c4dff; padding:1rem 1.2rem;
                        border-radius:0 10px 10px 0; margin-bottom:1rem">
            <b>🧩 What problem does MC Prediction solve?</b><br>
            Imagine you have a <em>fixed strategy</em> (policy) — say, "always try to go right, then down."
            Before improving it, you first need to know: <em>how good is each grid square under this strategy?</em>
            That number — how much total reward you expect from a given position — is called the <b>state value V(s)</b>.<br><br>
            MC Prediction answers: <em>"Given the agent keeps following this exact policy forever,
            what is the average total reward it will collect starting from each state?"</em><br><br>
            <b>Why do we need this?</b> Knowing which states are valuable and which are dangerous is the
            foundation for improving the policy later. You can't get better without first knowing how good you already are.
            </div>
            """, unsafe_allow_html=True)
            st.subheader("📊 MC Prediction — First-Visit vs Every-Visit")
            st.markdown("""
            Both methods estimate **V(s)** under the uniform random behavior policy.  
            The only difference: when a state appears **multiple times** in an episode,  
            First-Visit counts it **once**; Every-Visit counts it **every time**.

            **Intuition for non-scientists:**
            Suppose the agent visits square (2,3) five times in a single episode, then reaches the goal.
            - **First-Visit MC:** only records the return from the *first* visit to (2,3). Cleaner math, fewer samples.
            - **Every-Visit MC:** records five separate returns from (2,3) — one per visit. More data per episode,
              but those 5 samples are correlated (they're from the same game).
            """)

            # ── Theory panel ─────────────────────────────────────────────────
            with st.expander("📐 Theory & Formulas — MC Prediction", expanded=False):
                st.markdown(r"""
                #### The Core Idea

                The **value of a state** is the expected total reward from that state onward:

                $$v_\pi(s) \doteq \mathbb{E}_\pi\bigl[G_t \mid S_t = s\bigr]$$

                where the **return** $G_t$ is the discounted sum of future rewards:

                $$G_t \doteq R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

                MC Prediction estimates $v_\pi(s)$ by simply **averaging observed returns**:

                $$V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

                ---

                #### First-Visit vs Every-Visit

                Let $\mathcal{T}(s)$ be the set of time steps at which state $s$ is visited across all episodes.

                | Method | Which visits count | Bias | Variance | Convergence rate |
                |--------|--------------------|------|----------|-----------------|
                | **First-Visit** | Only the first visit per episode | **Unbiased** | Lower | Error ∝ 1/√n |
                | **Every-Visit** | Every visit in every episode | Slightly biased (correlated) | Higher | Faster in practice |

                By the **Law of Large Numbers**, both converge to the true $v_\pi(s)$ as $N(s) \to \infty$.

                For First-Visit MC, each return is an **independent, identically distributed** sample —
                so the standard error of the estimate falls as $1/\sqrt{n}$, where $n$ = number of first visits.

                ---

                #### Key Properties — What Makes MC Special

                > **No bootstrapping:** MC uses *actual* returns, never estimates-of-estimates. This means
                > zero bias for prediction — the estimates are always correct in expectation.

                > **Independence:** The estimate for state $s$ does not depend on the estimate for any
                > other state $s'$. This is unlike Dynamic Programming, where all state values are coupled.
                > You could estimate V(s) for just *one* state by running episodes only from $s$.

                > **No model needed:** MC only needs sample episodes. It never needs to know $p(s'|s,a)$.
                """)


            st.markdown("""
            <div style="background:#0d1b0d; border-left:4px solid #4caf50; padding:.8rem 1rem;
                        border-radius:0 8px 8px 0; margin-bottom:.8rem">
            <b>📖 How to read the heatmaps below:</b><br>
            Each coloured cell shows the estimated value V(s) of that grid square.
            <b>Green = high value</b> (being here tends to lead to big rewards — close to the goal).
            <b>Red = low / negative value</b> (dangerous — close to the trap, or far from the goal).
            Numbers inside each cell are the exact V(s) estimate.
            Walls (■) have no value because the agent can never stand on them.
            The two maps should look <em>almost identical</em> — they're solving the same problem, just counting visits differently.
            </div>
            """, unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_fv"], "First-Visit MC — V(s)", axes[0])
            plot_value_heatmap(env, res["V_ev"], "Every-Visit MC — V(s)", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Convergence at a representative state
            st.markdown("### Convergence at State (3,4) — goal-adjacent")
            st.markdown("""
            > **Why state (3,4)?** It's the cell directly above the goal — high value, frequently visited,
            > making it a good indicator of convergence speed. Watch both lines stabilise toward the same
            > value as the number of episodes grows.
            >
            > **📖 How to read this chart:** X-axis = number of episodes played so far.
            > Y-axis = current estimate of V(3,4). Both lines should converge to the same true value.
            > A wobbly line = high variance (unreliable early estimates). A smooth line = low variance.
            > **First-Visit (purple)** tends to be smoother; **Every-Visit (teal)** may converge faster but can be noisier.
            """)
            focal = (3, 4)
            fv_trace = [h.get(focal, 0.0) for h in res["hist_fv"]]
            ev_trace = [h.get(focal, 0.0) for h in res["hist_ev"]]
            x = [(i+1) * max(1, n_episodes // 20) for i in range(len(fv_trace))]

            fig2, ax2, _ = make_fig(1, 1, 10, 4)
            ax2.plot(x, fv_trace, color="#7c4dff", lw=2.5, marker="o", ms=5, label="First-Visit MC")
            ax2.plot(x, ev_trace, color="#00bcd4", lw=2.5, marker="s", ms=5, label="Every-Visit MC")
            ax2.set_xlabel("Episodes", color="white"); ax2.set_ylabel("V(3,4)", color="white")
            ax2.set_title("Value Estimate Convergence", color="white", fontweight="bold")
            ax2.legend(facecolor=CARD_BG, labelcolor="white"); ax2.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            c1, c2 = st.columns(2)
            c1.info("**First-Visit MC** — Unbiased, independent samples. Statistically cleaner, converges more slowly on revisited states.")
            c2.info("**Every-Visit MC** — More samples per episode (correlated). Converges faster in practice but has slight bias from correlation.")
            render_mc_notes("MC Prediction", "monte_carlo_methods_mc_prediction")

        # ── Tab 2: On-policy Control ──────────────────────────────────────────
        with tab_ctrl:
            st.markdown("""
            <div style="background:#0d1b14; border-left:4px solid #00897b; padding:1rem 1.2rem;
                        border-radius:0 10px 10px 0; margin-bottom:1rem">
            <b>🧩 What problem does On-policy MC Control solve?</b><br>
            Prediction tells us how good each state is under a <em>fixed</em> policy — but what if the policy itself is bad?
            <b>Control</b> takes the next step: it simultaneously <em>evaluates</em> the current policy AND <em>improves</em> it.<br><br>
            The key upgrade here is learning <b>Q(s,a) — action values</b> instead of just V(s).
            Instead of "how good is state s?", Q(s,a) answers "how good is it to take action <em>a</em> in state <em>s</em>?"
            This lets the agent pick the best action at each step without needing a model of the environment.<br><br>
            <b>The exploration problem:</b> If the agent always picks the best known action (greedy), it may
            never discover better alternatives. <b>ε-greedy</b> solves this: most of the time pick the best
            known action, but occasionally (with probability ε) pick a random one to explore.
            </div>
            """, unsafe_allow_html=True)
            st.subheader("🎯 On-policy MC Control (ε-greedy GPI)")
            st.markdown(r"""
            MC Control estimates **Q(s,a)** action-values and improves the policy with
            **Generalized Policy Iteration (GPI)** — evaluation + improvement every episode.
            The ε-greedy policy guarantees *some* exploration in every state.
            """)

            # ── Theory panel ─────────────────────────────────────────────────
            with st.expander("📐 Theory & Formulas — On-policy MC Control", expanded=False):
                st.markdown(r"""
                #### Why Q(s,a) instead of V(s)?

                Without a model of the environment you can't look one step ahead to find the best action.
                So instead of asking *"how good is state s?"* we ask *"how good is it to take action a in state s?"*

                $$q_\pi(s,a) \doteq \mathbb{E}_\pi\bigl[G_t \mid S_t=s,\, A_t=a\bigr]$$

                MC estimates this by averaging actual returns from every (state, action) pair:

                $$Q(s,a) \;\leftarrow\; \text{mean of all } G_t \text{ following first visit to } (s,a)$$

                ---

                #### ε-Greedy Policy (No Exploring Starts)

                Early algorithms needed "exploring starts" — the agent must try every (s,a) pair as a starting
                point. This is impractical in the real world. **ε-greedy** solves it:

                """)
                st.latex(r"""
                \pi(a \mid s) =
                \begin{cases}
                1 - \varepsilon + \dfrac{\varepsilon}{|A(s)|} & \text{if } a = \arg\max_{a'} Q(s,a') \\[6pt]
                \dfrac{\varepsilon}{|A(s)|} & \text{otherwise}
                \end{cases}
                """)
                st.markdown(r"""

                Every non-greedy action gets a floor probability of **ε / |A(s)|**, guaranteeing all
                state-action pairs are eventually visited. The best known action gets the rest:
                **1 − ε + ε/|A(s)|**.

                In this gridworld |A(s)| = 4, so with ε = 0.1:
                - Best action probability: **0.925**
                - Each other action: **0.025**

                ---

                #### GPI — Generalised Policy Iteration

                After every episode, two things happen in lockstep:

                | Step | What happens |
                |------|-------------|
                | **Evaluation** | Update Q(s,a) ← mean returns for all (s,a) visited this episode |
                | **Improvement** | Rebuild ε-greedy policy from the updated Q |

                This ping-pong between evaluation and improvement is **Generalised Policy Iteration (GPI)**.
                It converges to the best **ε-soft** policy — not the absolute optimal π*, because we must
                keep ε > 0 to maintain exploration.

                ---

                #### The Key Limitation

                > On-policy control converges to the **best ε-soft policy**, not the true optimal.
                > An ε-soft policy always has some chance (≥ ε/|A|) of picking a suboptimal action.
                > To remove this constraint you need **off-policy methods**, which separate the
                > *exploration* policy (behavior) from the *learned* policy (target).
                """)

            st.markdown("""
            <div style="background:#0d1b14; border-left:4px solid #4caf50; padding:.8rem 1rem;
                        border-radius:0 8px 8px 0; margin-bottom:.8rem">
            <b>📖 How to read these two diagrams:</b><br>
            <b>Left — Value heatmap:</b> Colours show max<sub>a</sub> Q(s,a) — the best expected return the
            agent believes it can achieve from each square. Green = confident the agent can reach the goal;
            red = near the trap or hopelessly far.<br>
            <b>Right — Policy arrows:</b> Each arrow = the greedy action (arg max Q) the agent has settled on.
            A well-trained policy shows arrows that form a coherent path from (0,0) to the goal (4,4) while
            avoiding the trap at (2,2). Walls have no arrow — the agent can never stand there.
            </div>
            """, unsafe_allow_html=True)
            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_on"], "Q-max V(s) — On-policy Control", axes[0])
            plot_policy_arrows(env, res["pi_det"], "Greedy Policy π* from Q", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Q-values table for start state
            st.markdown("### Q(start, ·) — Action Values at Start State")
            st.markdown("""
            > **What this table shows:** The agent starts at (0,0). After training, it has an estimated
            > value for each of the 4 possible first moves. The highest Q-value tells you which direction
            > the agent thinks is best. If the policy is good, the best action should be ↓ or → (heading
            > toward the goal in the bottom-right corner). A ✅ marks the chosen action.
            """)
            Q_start = res["Q_on"][(0,0)]
            qdf = pd.DataFrame({
                "Action": ["↑ Up", "→ Right", "↓ Down", "← Left"],
                "Q(start, a)": [f"{v:.3f}" for v in Q_start],
                "Best?": ["✅" if i == int(np.argmax(Q_start)) else "" for i in range(4)],
            })
            st.dataframe(qdf, use_container_width=True, hide_index=True)

            # Learning curve
            st.markdown("### Episode Return — Learning Curve")
            st.markdown("""
            > **📖 How to read this chart:** Each point on the X-axis is one complete episode.
            > The Y-axis is the total reward collected during that episode (higher = better).
            > A random agent scores very negative numbers (many −0.1 steps, often falls in the trap).
            > As the agent learns, the returns should trend **upward** — more episodes ended at the goal.
            > The faint raw line shows every episode (noisy); the bright line is a rolling average.
            > **Peak return** (green dashed) shows the best the agent achieved.
            > If the curve plateaus and doesn't reach near +10, the ε-greedy policy is limiting it — a known limitation of on-policy control.
            """)
            raw = res["ep_rewards"]
            sm  = smooth(raw, max(1, len(raw)//20))
            fig3, ax3, _ = make_fig(1, 1, 10, 4)
            ax3.plot(raw, color="#7c4dff", alpha=0.15, lw=0.6)
            ax3.plot(range(len(sm)), sm, color="#7c4dff", lw=2.5, label="Smoothed return")
            ax3.axhline(np.max(sm), color="#4caf50", ls="--", lw=1, alpha=0.7, label=f"Peak {np.max(sm):.2f}")
            ax3.set_xlabel("Episode", color="white"); ax3.set_ylabel("Total Return", color="white")
            ax3.set_title("On-policy MC Control — Learning Progress", color="white", fontweight="bold")
            ax3.legend(facecolor=CARD_BG, labelcolor="white"); ax3.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            st.info("**Key insight**: The agent can only converge to the *best ε-soft* policy — not the true optimal π*. To learn the exact optimal policy, use off-policy methods.")
            render_mc_notes("On-policy Control", "monte_carlo_methods_on_policy_control")

        # ── Tab 3: Off-policy IS ──────────────────────────────────────────────
        with tab_offpol:
            st.markdown("""
            <div style="background:#1b0a0a; border-left:4px solid #ef5350; padding:1rem 1.2rem;
                        border-radius:0 10px 10px 0; margin-bottom:1rem">
            <b>🧩 What problem does Off-policy MC solve?</b><br>
            On-policy methods have a catch: the policy that <em>collects data</em> must be the same as the
            policy being <em>evaluated</em>. But what if you want to:<br>
            &nbsp;&nbsp;• Learn from <b>historical data</b> collected by a different (e.g. random) agent?<br>
            &nbsp;&nbsp;• Evaluate a <b>risky "target" policy</b> without actually letting it run (e.g. in robotics, self-driving cars)?<br>
            &nbsp;&nbsp;• Keep a <b>safe exploration strategy</b> while still learning about an aggressive one?<br><br>
            <b>Off-policy MC</b> solves all of these using <b>Importance Sampling (IS)</b> — a statistical
            trick that asks: <em>"If the target policy had generated this episode instead of the behavior policy,
            how would the returns be different?"</em> The answer is a correction weight ρ applied to every return.<br><br>
            <b>Real-world example:</b> A hospital has logs of patients treated with standard drug A.
            They want to know: "How would those same patients have fared on experimental drug B?"
            Off-policy IS is exactly the tool used in clinical causal inference.
            </div>
            """, unsafe_allow_html=True)
            st.subheader("⚖️ Off-policy MC — Ordinary IS vs Weighted IS")
            st.markdown(r"""
            Off-policy evaluation lets us assess a **target policy π** using data
            from a **behavior policy b** (here: random). The key tool is
            **Importance Sampling (IS)**:

            $$\rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

            | Method | Estimator | Bias | Variance |
            |--------|-----------|------|----------|
            | Ordinary IS | mean(ρ · G) | Unbiased | **Very high** |
            | Weighted IS | Σ(ρ·G)/Σρ | Biased | **Much lower** |
            """)

            # ── Theory panel ─────────────────────────────────────────────────
            with st.expander("📐 Theory & Formulas — Off-policy IS", expanded=False):
                st.markdown(r"""
                #### The Core Dilemma

                Off-policy learning faces a fundamental problem: the data was collected under behavior
                policy $b$, but we want to evaluate target policy $\pi$. The returns have the wrong
                expectation: $\mathbb{E}_b[G_t | S_t = s] = v_b(s) \neq v_\pi(s)$.

                **Importance Sampling (IS)** corrects for this mismatch using a ratio that measures
                *"how much more or less likely was this trajectory under π than under b?"*

                ---

                #### The IS Ratio

                Given a trajectory $A_t, S_{t+1}, A_{t+1}, \ldots, S_T$ starting from state $S_t$:

                """)
                st.latex(r"""
                \rho_{t:T-1} \doteq \frac{\prod_{k=t}^{T-1}\pi(A_k|S_k)\,p(S_{k+1}|S_k,A_k)}
                {\prod_{k=t}^{T-1}b(A_k|S_k)\,p(S_{k+1}|S_k,A_k)}
                = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}
                """)
                st.markdown(r"""

                The environment dynamics $p(S_{k+1}|S_k,A_k)$ cancel — IS only depends on the policies,
                not the transition model. This is crucial: MC off-policy evaluation is **model-free**.

                ---

                #### Ordinary Importance Sampling

                Simply multiply each return by its IS ratio, then average:

                """)
                st.latex(r"""
                V(s) \doteq \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}
                {|\mathcal{T}(s)|}
                """)
                st.markdown(r"""

                ✅ **Unbiased** — correct in expectation  
                ❌ **Unbounded variance** — if ρ can be 10×, the estimate can be 10× the actual return

                ---

                #### Weighted Importance Sampling

                Use IS weights as a weighted denominator instead:

                """)
                st.latex(r"""
                V(s) \doteq \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}
                {\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}
                """)
                st.markdown(r"""

                The weight on any single return is **at most 1** (the max weight cancels in numerator/denominator).

                ✅ **Dramatically lower variance** — weights are bounded  
                ❌ **Biased** — but bias converges to 0 asymptotically as $N \to \infty$

                ---

                #### Coverage Requirement

                Off-policy IS requires **coverage**: every action taken under π must also be possible under b.
                """)
                st.latex(r"\pi(a|s) > 0 \implies b(a|s) > 0")
                st.markdown(r"""

                In this gridworld: the behavior policy is uniform random (all actions equally likely),
                satisfying coverage for any deterministic target policy.

                ---

                #### Practical Verdict

                > **Weighted IS is almost always preferred in practice.** Despite being biased,
                > its variance is so much lower that it reaches accurate estimates far faster.
                > Ordinary IS is useful in theory and for certain function-approximation extensions.
                """)


            st.markdown("""
            <div style="background:#1b0a0a; border-left:4px solid #ef5350; padding:.8rem 1rem;
                        border-radius:0 8px 8px 0; margin-bottom:.8rem">
            <b>📖 How to read these three heatmaps:</b><br>
            All three maps estimate V(s) for the <em>same target policy</em> (the greedy policy learned by on-policy control).
            The <b>leftmost (On-policy reference)</b> is the "ground truth" — learned directly by the agent that followed the target policy.
            The <b>middle (Ordinary IS)</b> and <b>right (Weighted IS)</b> estimate the <em>same values</em> but using
            <em>random exploration data</em> with importance-sampling correction weights.
            Differences from the reference map reveal the estimation error.
            Ordinary IS is noisier (more red/green extremes); Weighted IS tends to be smoother and closer to the reference.
            </div>
            """, unsafe_allow_html=True)
            fig, axes, axl = make_fig(1, 3, 15, 5)
            plot_value_heatmap(env, res["V_ois"], "Ordinary IS",            axes[1])
            plot_value_heatmap(env, res["V_wis"], "Weighted IS",            axes[2])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # State-by-state comparison
            st.markdown("### Per-state Comparison: IS Methods vs Reference")
            st.markdown("""
            > **📖 How to read this line chart:** Each point on the X-axis represents one non-wall, non-terminal
            > grid state. The Y-axis is the estimated V(s) for that state.
            > The **green line** (on-policy) is the reference we're trying to match.
            > **Ordinary IS (red)** often diverges wildly — large spikes indicate episodes where the
            > importance weight ρ happened to be very large, skewing the average.
            > **Weighted IS (blue)** stays closer to the reference because the weights are normalised.
            > A perfect off-policy estimator would exactly overlap the green line.
            """)
            common = [s for s in res["V_on"] if s in res["V_ois"] and s in res["V_wis"]
                      and s not in env.walls and not env.is_terminal(s)]
            ref  = np.array([res["V_on"][s]  for s in common])
            ois  = np.array([res["V_ois"][s] for s in common])
            wis  = np.array([res["V_wis"][s] for s in common])

            fig4, ax4, _ = make_fig(1, 1, 10, 4)
            idx = range(len(common))
            ax4.plot(idx, ref, "o-",  color="#4caf50", lw=2, label="On-policy (ref)", ms=5)
            ax4.plot(idx, ois, "s--", color="#ef5350", lw=1.8, alpha=0.85, label="Ordinary IS", ms=5)
            ax4.plot(idx, wis, "^-",  color="#42a5f5", lw=1.8, alpha=0.85, label="Weighted IS", ms=5)
            ax4.set_xlabel("State index (non-wall, non-terminal)", color="white")
            ax4.set_ylabel("V(s)", color="white")
            ax4.set_title("Value Estimates Across States", color="white", fontweight="bold")
            ax4.legend(facecolor=CARD_BG, labelcolor="white"); ax4.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig4); plt.close()

            mse_ois = float(np.mean((ois - ref)**2))
            mse_wis = float(np.mean((wis - ref)**2))
            st.markdown("""
            > **📖 What these numbers mean:** MSE (Mean Squared Error) measures how far an IS estimate
            > is from the on-policy reference — lower is better. The "WIS improvement" tells you how much
            > Weighted IS reduced the error compared to Ordinary IS. Even with the same data and episodes,
            > Weighted IS almost always wins on this metric.
            """)
            c1, c2, c3 = st.columns(3)
            c1.metric("Ordinary IS — MSE", f"{mse_ois:.4f}")
            c2.metric("Weighted IS — MSE",  f"{mse_wis:.4f}")
            c3.metric("WIS improvement", f"{max(0,(mse_ois-mse_wis)/max(mse_ois,1e-9)*100):.1f}%", delta="↓ lower is better")
            render_mc_notes("Off-policy IS", "monte_carlo_methods_off_policy_is")

        # ── Tab 4: Incremental MC ─────────────────────────────────────────────
        with tab_incr:
            st.markdown("""
            <div style="background:#1b0f00; border-left:4px solid #ff9800; padding:1rem 1.2rem;
                        border-radius:0 10px 10px 0; margin-bottom:1rem">
            <b>🧩 What problem does Incremental MC solve?</b><br>
            Standard MC Prediction stores <em>every single return</em> for every state — then computes the average at the end.
            For a long-running agent (millions of episodes), this quickly exhausts memory.<br><br>
            <b>Incremental MC</b> asks: <em>"Can I maintain a running average without storing all past data?"</em>
            The answer is yes — using the identity <code>new_mean = old_mean + (new_value − old_mean) / n</code>.
            This trick means the agent only ever needs to remember the <em>current estimate</em> and a <em>count</em> — O(1) memory.<br><br>
            <b>Why it matters beyond memory:</b> If you replace the 1/N step-size with a fixed constant α,
            you get an <em>exponential moving average</em> — recent episodes matter more than old ones.
            This is critical for <b>non-stationary environments</b> (e.g. a game where the rules slowly change).
            It's also the direct bridge to <b>TD learning</b>, the next major RL paradigm.
            </div>
            """, unsafe_allow_html=True)
            st.subheader("⚡ Incremental Monte Carlo")
            st.markdown(r"""
            Instead of storing all returns per state, **Incremental MC** updates online:

            $$V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}\bigl[G_t - V(S_t)\bigr]$$

            This is **mathematically equivalent** to batch averaging but uses *O(1)* memory per state.  
            Replacing 1/N with a fixed α gives an **exponential moving average** — useful for  
            non-stationary environments (and is the precursor to TD learning).
            """)

            # ── Theory panel ─────────────────────────────────────────────────
            with st.expander("📐 Theory & Formulas — Incremental MC", expanded=False):
                st.markdown(r"""
                #### The Memory Problem

                Batch MC stores every observed return per state:
                $\{G_1, G_2, \ldots, G_n\}$ then computes $V(s) = \frac{1}{n}\sum G_i$.

                For long-running agents this is impractical. **Incremental MC** uses the identity:

                $$V_{n+1} \doteq V_n + \frac{1}{n}\bigl[G_n - V_n\bigr]$$

                which is the *running mean update* — provably equivalent to the batch average but requires
                storing only $V_n$ and the counter $n$. The term $[G_n - V_n]$ is called the **error**:
                how much the new observation differs from the current estimate.

                ---

                #### Off-Policy Generalisation (Weighted IS)

                For weighted importance sampling, the incremental update generalises to:

                $$V_{n+1} \doteq V_n + \frac{W_n}{C_n}\bigl[G_n - V_n\bigr]$$

                $$C_{n+1} \doteq C_n + W_{n+1}, \quad C_0 = 0$$

                where $W_n = \rho_{t_n:T(t_n)-1}$ is the IS weight for the $n$-th return.
                When $W=1$ for all steps (on-policy case), this reduces to the simple $1/n$ update.

                ---

                #### Fixed Step-Size — α Replacement

                Replace $1/n$ with a constant **α ∈ (0,1)**:

                $$V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t - V(S_t)\bigr]$$

                This gives an **exponential moving average** — older returns are geometrically discounted:

                $$V_n = (1-\alpha)^n V_0 + \alpha \sum_{k=1}^{n}(1-\alpha)^{n-k} G_k$$

                | Step-size | Memory of past | Best for |
                |-----------|---------------|---------|
                | 1/n | Equal weight all | Stationary environments |
                | α (constant) | Recent > old | Non-stationary environments |

                ---

                #### Bridge to TD Learning

                Replace the actual return $G_t$ (which requires the full episode) with a
                **bootstrapped estimate** $R_{t+1} + \gamma V(S_{t+1})$ (available after one step):

                $$V(S_t) \leftarrow V(S_t) + \alpha\bigl[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\bigr]$$

                This is **TD(0)** — the simplest temporal-difference method. The incremental update
                structure from MC is identical; only the target changes from $G_t$ to a one-step estimate.
                """)

            st.markdown("""
            <div style="background:#1b0f00; border-left:4px solid #ffa726; padding:.8rem 1rem;
                        border-radius:0 8px 8px 0; margin-bottom:.8rem">
            <b>📖 How to read these heatmaps:</b><br>
            Both maps show V(s) estimates under the same random policy.
            They should look <em>nearly identical</em> — because Incremental MC is mathematically equivalent to batch First-Visit MC.
            Any visible differences are just random seed effects.
            The point of the comparison is to confirm that the memory-efficient online method produces the same result as storing everything.
            </div>
            """, unsafe_allow_html=True)
            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_fv"],  "First-Visit MC (batch)", axes[0])
            plot_value_heatmap(env, res["V_inc"], "Incremental MC",         axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Variance history
            st.markdown("### Mean Variance Across States Over Time")
            st.markdown("""
            > **📖 How to read this chart:** The X-axis marks checkpoints (every N/20 episodes).
            > The Y-axis shows the average variance of V(s) estimates across all states.
            > **Variance measures how much the estimates jump around** — high variance means unreliable estimates.
            > You should see variance **decrease and flatten** as more episodes are played.
            > This is the core MC learning signal: the more data you collect, the more confident the estimates become.
            > The orange fill under the curve shows the area of uncertainty — as it shrinks, the agent "knows more."
            """)
            fig5, ax5, _ = make_fig(1, 1, 10, 4)
            vh = res["var_hist"]
            if vh:
                ax5.plot(vh, color="#ff9800", lw=2.5)
                ax5.fill_between(range(len(vh)), 0, vh, color="#ff9800", alpha=0.15)
            ax5.set_xlabel("Checkpoint (every N/20 episodes)", color="white")
            ax5.set_ylabel("Mean Var(V(s))", color="white")
            ax5.set_title("Variance Reduction as N Grows", color="white", fontweight="bold")
            ax5.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig5); plt.close()

            c1, c2 = st.columns(2)
            c1.info("**Memory**: O(1) per state — no need to store every return.")
            c2.info("**Bridge to TD**: Replace 1/N with α and allow G to bootstrap → you get TD(0).")
            render_mc_notes("Incremental MC", "monte_carlo_methods_incremental_mc")

        # ── Tab 5: Advanced IS ────────────────────────────────────────────────
        with tab_adv:
            st.markdown("""
            <div style="background:#0d1020; border-left:4px solid #9c6dff; padding:1rem 1.2rem;
                        border-radius:0 10px 10px 0; margin-bottom:1rem">
            <b>🧩 What problem do Advanced IS methods solve?</b><br>
            Even Weighted IS can suffer from high variance in long episodes — because the importance weight ρ
            is a <em>product</em> of many per-step ratios. If an episode has 50 steps, you're multiplying
            50 numbers together. Even if each ratio is close to 1, the product can still become astronomically
            large or small.<br><br>

            <b>Per-Decision IS</b> breaks the weight apart: each reward Rₖ is only corrected by the ratios
            <em>up to that step</em>, not the whole episode. Why does this help?
            Because future decisions don't affect past rewards — so there's no reason to include their ratios.
            *Think:* "The reward I got at step 3 has nothing to do with what the policy did at step 47."<br><br>

            <b>Discounting-Aware IS</b> goes one step further: when γ &lt; 1, rewards far in the future
            matter less (they're discounted). So their IS corrections also need to matter less.
            This method exploits the discount factor to reduce the effective "length" of the IS product,
            giving the lowest variance of all estimators.<br><br>

            <b>Real-world use:</b> Both methods are critical in healthcare, finance, and robotics where
            episodes are long and the difference between behavior and target policy is large.
            </div>
            """, unsafe_allow_html=True)
            st.subheader("🔬 Advanced IS: Per-Decision vs Discounting-Aware")
            st.markdown(r"""
            **Per-Decision IS** decomposes the episode-level importance weight into per-step ratios.  
            Each reward $R_{k+1}$ is weighted only by the IS ratio *up to* step k, not the full episode:

            $$\hat{V}^{PD}(s_t) = \sum_{k=t}^{T} \gamma^{k-t}\left(\prod_{j=t}^{k}\rho_j\right) R_{k+1}$$

            **Discounting-Aware IS** additionally exploits the discount structure  
            (when γ < 1, distant rewards contribute less — so their IS ratios matter less).
            This gives the **lowest variance** among all IS estimators.
            """)

            # ── Theory panel ─────────────────────────────────────────────────
            with st.expander("📐 Theory & Formulas — Per-Decision & Discounting-Aware IS", expanded=False):
                st.markdown(r"""
                #### Why Standard IS Has High Variance

                The standard IS weight multiplies **all** per-step ratios together:
                $$\rho_{t:T-1} = \prod_{k=t}^{T-1} \rho_k, \quad \rho_k = \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$$

                For a 50-step episode, that's 50 numbers multiplied together. Even if each is ~1.2,
                the product is $1.2^{50} \approx 9100$ — an enormous, destabilising weight.

                ---

                #### Per-Decision Importance Sampling

                Key insight: **reward $R_{k+1}$ is independent of actions taken after step $k$**.
                So future ratios are irrelevant for past rewards. We write the return as a sum of
                individually IS-corrected rewards:

                """)
                st.latex(r"""
                \hat{V}^{PD}(s_t) = \sum_{k=t}^{T-1} \gamma^{k-t}
                \underbrace{\left(\prod_{j=t}^{k}\rho_j\right)}_{\text{only ratios up to step }k} R_{k+1}
                """)
                st.markdown(r"""

                This is **provably unbiased** and has **lower variance** than ordinary IS,
                because each reward uses only the ratios it actually needs.

                ---

                #### Discounting-Aware Importance Sampling

                When $\gamma < 1$, the discounted return can be rewritten as a weighted sum of
                **flat (undiscounted) partial returns** $\bar{G}_{t:h}$:

                """)
                st.latex(r"G_t = (1-\gamma)\sum_{h=t}^{T-1} \gamma^{h-t} \bar{G}_{t:h} + \gamma^{T-t} \bar{G}_{t:T}")
                st.markdown(r"""

                Each partial return $\bar{G}_{t:h}$ only needs IS ratios up to horizon $h$, not the full
                episode. The IS weight for each partial return is then:

                """)
                st.latex(r"\rho_{t:h} = \prod_{k=t}^{h} \rho_k \quad (\text{length } h-t+1, \text{ not } T-t)")
                st.markdown(r"""

                Since $\gamma < 1$ discounts distant horizons, those long IS products have lower influence.
                Result: **shortest effective IS products → lowest variance of all estimators**.

                Benefit vanishes as $\gamma \to 1$ (no discounting → same as Per-Decision IS).

                ---

                #### Variance Hierarchy

                """)
                st.latex(r"\text{Var}[\text{Disc. IS}] \;\leq\; \text{Var}[\text{Per-Decision IS}] \;\leq\; \text{Var}[\text{Weighted IS}] \;\leq\; \text{Var}[\text{Ordinary IS}]")
                st.markdown(r"""

                | Method | IS product length | Variance |
                |--------|-----------------|---------|
                | Ordinary IS | Full episode (T steps) | Highest |
                | Weighted IS | Full episode, normalised | Lower |
                | Per-Decision IS | Up to each step k | Very low |
                | Discounting-Aware IS | Discounted by γ | Lowest |
                """)

            st.markdown("""
            <div style="background:#0d1020; border-left:4px solid #9c6dff; padding:.8rem 1rem;
                        border-radius:0 8px 8px 0; margin-bottom:.8rem">
            <b>📖 How to read these heatmaps:</b><br>
            Both maps use only <b>500 episodes</b> (capped for speed), evaluating the same greedy target policy
            via off-policy data. They should show <em>smoother, more consistent</em> value estimates than
            Ordinary IS — particularly in states far from the goal where IS weights tend to blow up.
            If cells look washed out or grey, that state was rarely visited under the behavior policy,
            so few valid IS corrections could be made — a real limitation of IS methods with a very
            different behavior policy.
            </div>
            """, unsafe_allow_html=True)
            fig, axes, axl = make_fig(1, 2, 12, 5)
            plot_value_heatmap(env, res["V_pd"], "Per-Decision IS — V(s)",       axes[0])
            plot_value_heatmap(env, res["V_da"], "Discounting-Aware IS — V(s)",  axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Variance bars for all IS methods
            st.markdown("### Variance Comparison — All IS Methods")
            st.markdown("""
            > **📖 How to read this bar chart:** Each bar represents one IS estimator.
            > The height is the **variance of V(s) estimates across all non-terminal states** — lower means
            > the estimator gives more consistent, reliable answers.
            > The expected ordering from theory is:
            > **Ordinary IS** (highest) → **Weighted IS** → **Per-Decision IS** → **Discounting IS** (lowest).
            > If your run doesn't show exactly this ordering, it's because 500 episodes is a small sample
            > and randomness can flip adjacent methods. The general trend should hold.
            > **Colour guide:** red=unstable, blue=good, green=best.
            """)
            all_s = [s for s in res["V_on"] if s not in env.walls and not env.is_terminal(s)]
            variance_data = {
                "Ordinary IS":       np.var([res["V_ois"].get(s,0) for s in all_s]),
                "Weighted IS":       np.var([res["V_wis"].get(s,0) for s in all_s]),
                "Per-Decision IS":   np.var([res["V_pd"].get(s,0)  for s in all_s]),
                "Discounting IS":    np.var([res["V_da"].get(s,0)  for s in all_s]),
            }
            fig6, ax6, _ = make_fig(1, 1, 10, 4)
            colors_b = ["#ef5350","#42a5f5","#66bb6a","#ffa726"]
            bars = ax6.bar(list(variance_data.keys()), list(variance_data.values()),
                           color=colors_b, edgecolor="white", lw=0.5, alpha=0.9)
            for bar, val in zip(bars, variance_data.values()):
                ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                         f"{val:.4f}", ha="center", va="bottom", color="white", fontsize=9)
            ax6.set_ylabel("Variance of V(s) estimates", color="white")
            ax6.set_title("IS Variance Hierarchy (lower = better)", color="white", fontweight="bold")
            ax6.grid(alpha=0.15, axis="y")
            plt.tight_layout(); st.pyplot(fig6); plt.close()
            render_mc_notes("Advanced IS", "monte_carlo_methods_advanced_is")

        # ── Tab 6: Dashboard ──────────────────────────────────────────────────
        with tab_dash:
            st.markdown("""
            <div style="background:#120d1a; border-left:4px solid #ce93d8; padding:1rem 1.2rem;
                        border-radius:0 10px 10px 0; margin-bottom:1rem">
            <b>📈 What this dashboard shows</b><br>
            This is the big-picture comparison. All 8 MC methods estimate values for the <em>same gridworld</em>,
            and you can compare them side-by-side to see how much their answers agree or differ.<br><br>
            <b>Key question to ask yourself:</b> Do all maps agree on <em>which states are good and bad</em>?
            They should — because they're all estimating the same underlying truth.
            But they may disagree on the <em>exact numbers</em>, especially in states that are rarely visited
            or where IS weights are extreme.
            </div>
            """, unsafe_allow_html=True)
            st.subheader("📈 Full Comparison Dashboard")

            # All 8 heatmaps
            st.markdown("### Value Functions — All 8 Methods")
            st.markdown("""
            <div style="background:#1a1a2e; border-left:4px solid #ce93d8; padding:.8rem 1rem;
                        border-radius:0 8px 8px 0; margin-bottom:.8rem">
            <b>📖 How to read this grid of 8 maps:</b>
            Every map uses the same colour scale (red=−5 to green=+10). Scan for:<br>
            &nbsp;• <b>Agreement on hot zones</b> — all methods should show green near the goal and red near the trap.<br>
            &nbsp;• <b>Coverage gaps</b> — grey/blank cells appear in off-policy IS methods when a state was never
            reached by the behavior policy, so no IS correction could be computed.<br>
            &nbsp;• <b>Smoothness</b> — noisier maps (more random-looking colour patches) indicate higher variance.
            First-Visit and Incremental MC should look the smoothest.
            Off-policy methods (especially Ordinary IS) may look patchier.
            </div>
            """, unsafe_allow_html=True)
            methods_v = [
                (res["V_fv"],  "First-Visit MC"),
                (res["V_ev"],  "Every-Visit MC"),
                (res["V_on"],  "On-policy Control"),
                (res["V_ois"], "Ordinary IS"),
                (res["V_wis"], "Weighted IS"),
                (res["V_inc"], "Incremental MC"),
                (res["V_pd"],  "Per-Decision IS"),
                (res["V_da"],  "Discounting IS"),
            ]
            fig, axes, axl = make_fig(2, 4, 20, 10)
            for idx, (V, title) in enumerate(methods_v):
                ax = axes[idx//4][idx%4]
                ax.set_facecolor(DARK_BG)
                ax.tick_params(colors="#9e9ebb", labelsize=7)
                for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
                plot_value_heatmap(env, V, title, ax)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Summary table
            st.markdown("### Method Summary Table")
            st.markdown("""
            > **📖 How to read this table:**
            > - **MSE vs On-policy** — how far each method's V(s) estimates are from the on-policy control reference (lower = more accurate).
            > - **Var(V)** — how spread out the value estimates are across states (lower = more consistent map, less extreme highs/lows).
            > - **Coverage** — what percentage of non-terminal states got an estimate. Off-policy IS methods sometimes miss states
            >   the behavior policy never visited with the right action sequence, so their coverage can be under 100%.
            """)
            all_s = [s for s in res["V_on"] if s not in env.walls and not env.is_terminal(s)]
            ref_v = np.array([res["V_on"].get(s,0) for s in all_s])

            rows = []
            for V, title in methods_v:
                vals = np.array([V.get(s,0) for s in all_s])
                mse  = float(np.mean((vals - ref_v)**2))
                var  = float(np.var(vals))
                cov  = sum(1 for s in all_s if s in V) / len(all_s) * 100
                rows.append({"Method": title,
                              "MSE vs On-policy": f"{mse:.4f}",
                              "Var(V)": f"{var:.4f}",
                              "Coverage": f"{cov:.0f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Variance / stability chart
            st.markdown("### Relative Variance & Stability Scores (expert assessment)")
            st.markdown("""
            > **📖 How to read these bar charts:** These are *qualitative* expert scores, not computed from this run.
            > They represent the known theoretical properties of each method.
            > **Left chart (Variance, lower=better):** bars pointing left are better — shorter bar = less noisy estimates.
            > **Right chart (Stability, higher=better):** longer bar = more reliable across different runs and environments.
            > Notice that Ordinary IS has the worst variance score but Discounting IS has the best —
            > this matches the variance hierarchy proven in Sutton & Barto Chapter 5.
            """)
            labels  = ["FV MC","EV MC","On-policy","Ordinary IS","Weighted IS","Incremental","Per-Dec IS","Disc. IS"]
            var_sc  = [2, 3, 4, 9, 3, 2, 2, 1]   # lower = less variance
            stab_sc = [9, 8, 7, 3, 8, 9, 8, 9]    # higher = more stable

            fig7, axes7, _ = make_fig(1, 2, 14, 5)
            cbar = ["#7c4dff","#9c6dff","#00897b","#ef5350","#42a5f5","#ff9800","#66bb6a","#ffa726"]

            axes7[0].barh(labels, var_sc,  color=cbar, alpha=0.88, edgecolor="white", lw=0.4)
            axes7[0].set_xlabel("Variance Score  (← lower is better)", color="white")
            axes7[0].set_title("Relative Variance", color="white", fontweight="bold")
            axes7[0].invert_xaxis()
            axes7[0].grid(alpha=0.15, axis="x")

            axes7[1].barh(labels, stab_sc, color=cbar, alpha=0.88, edgecolor="white", lw=0.4)
            axes7[1].set_xlabel("Stability Score  (higher is better →)", color="white")
            axes7[1].set_title("Stability", color="white", fontweight="bold")
            axes7[1].grid(alpha=0.15, axis="x")

            plt.tight_layout(); st.pyplot(fig7); plt.close()

            # Evolution diagram
            st.markdown("### MC Method Evolution — Simple → Advanced")
            st.markdown("""
            > **📖 How to read this diagram:** Each circle is a method. Arrows show which method
            > *leads to* or *enables* the next. Start at the top-left (MC Prediction) and follow
            > the arrows rightward. Each branch solves a specific limitation of its parent:
            > - MC Prediction → Control (adds policy improvement)
            > - Control → Off-policy IS (separates exploration from learning)
            > - Ordinary IS → Weighted IS (fixes variance explosion)
            > - Weighted IS → Per-Decision IS (further variance reduction)
            > - Per-Decision IS → Discounting IS (exploits discount structure)
            > The right side of the diagram = more sophisticated, lower variance, but more complex to implement.
            """)
            fig8, ax8, _ = make_fig(1, 1, 14, 6)
            ax8.axis("off")

            nodes = [
                (0.05, 0.50, "MC\nPrediction",   "#7c4dff"),
                (0.22, 0.78, "First-Visit\nMC",  "#5c35cc"),
                (0.22, 0.22, "Every-Visit\nMC",  "#9c6dff"),
                (0.42, 0.50, "MC Control\nOn-pol","#00897b"),
                (0.60, 0.78, "Ordinary\nIS",      "#ef5350"),
                (0.60, 0.22, "Weighted\nIS",      "#42a5f5"),
                (0.78, 0.64, "Per-Decision\nIS",  "#ffa726"),
                (0.78, 0.36, "Incremental\nMC",   "#ff9800"),
                (0.95, 0.50, "Discounting\nIS",   "#66bb6a"),
            ]
            edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(3,5),(3,7),(4,6),(5,6),(6,8)]

            for x, y, lbl, col in nodes:
                ax8.add_patch(plt.Circle((x,y), 0.065, color=col, alpha=0.88,
                                         transform=ax8.transAxes, zorder=3))
                ax8.text(x, y, lbl, ha="center", va="center", fontsize=7, color="white",
                         fontweight="bold", transform=ax8.transAxes, zorder=4)

            for i, j in edges:
                x0,y0 = nodes[i][0], nodes[i][1]
                x1,y1 = nodes[j][0], nodes[j][1]
                ax8.annotate("", xy=(x1,y1), xytext=(x0,y0),
                             xycoords="axes fraction", textcoords="axes fraction",
                             arrowprops=dict(arrowstyle="->", color="#90a4ae", lw=1.5))

            ax8.set_title("MC Method Family Tree — Simple to Advanced",
                          color="white", fontweight="bold", pad=20)
            plt.tight_layout(); st.pyplot(fig8); plt.close()

            # Bias-variance scatter
            st.markdown("### Bias–Variance Landscape")
            st.markdown("""
            > **📖 How to read this scatter plot:** This is the most important summary chart.
            > - **X-axis (Bias):** How much does the method's estimate *systematically* deviate from truth?
            >   Unbiased = 0 bias. Weighted IS has nonzero bias (it's a known trade-off).
            > - **Y-axis (Variance):** How much do estimates *jump around* between runs?
            >   High variance = you need many episodes to get reliable answers.
            > - **Green bottom-left corner = ideal zone:** low bias AND low variance. This is where you want to be.
            > - **Ordinary IS** sits top-left: unbiased but extremely high variance (noisy, unstable).
            > - **Discounting IS** sits bottom-right: slightly biased but very low variance (reliable, smooth).
            > - In practice, **low variance usually wins** — a slightly biased but consistent answer is more useful
            >   than an unbiased but wildly noisy one.
            """)
            bv_data = {
                "FV MC":       (1.0, 2.0, "#7c4dff"),
                "EV MC":       (1.8, 2.8, "#9c6dff"),
                "On-policy":   (2.5, 4.0, "#00897b"),
                "Ordinary IS": (0.5, 9.0, "#ef5350"),
                "Weighted IS": (3.0, 2.5, "#42a5f5"),
                "Incremental": (1.0, 2.2, "#ff9800"),
                "Per-Dec IS":  (1.2, 1.8, "#ffa726"),
                "Disc. IS":    (1.5, 1.2, "#66bb6a"),
            }
            fig9, ax9, _ = make_fig(1, 1, 8, 5)
            ax9.add_patch(plt.Rectangle((0,0),3,3,alpha=0.08,color="green",zorder=0))
            ax9.text(1.5, 0.25, "← Ideal Region", color="#81c784", fontsize=9)

            for name,(bv,var,col) in bv_data.items():
                ax9.scatter(bv, var, s=200, color=col, zorder=5, edgecolors="white", lw=1)
                ax9.annotate(name, (bv,var), xytext=(6,4), textcoords="offset points",
                             color="white", fontsize=8)

            ax9.set_xlabel("Relative Bias →", color="white", fontsize=11)
            ax9.set_ylabel("Relative Variance →", color="white", fontsize=11)
            ax9.set_title("Bias–Variance Tradeoff: All MC Methods", color="white", fontweight="bold")
            ax9.set_xlim(0,10); ax9.set_ylim(0,10)
            ax9.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig9); plt.close()
            render_mc_notes("Dashboard", "monte_carlo_methods_dashboard")

    else:
        # If no run yet, show placeholder on computation tabs
        pending_tabs = [
            (tab_pred, "MC Prediction", "monte_carlo_methods_mc_prediction"),
            (tab_ctrl, "On-policy Control", "monte_carlo_methods_on_policy_control"),
            (tab_offpol, "Off-policy IS", "monte_carlo_methods_off_policy_is"),
            (tab_incr, "Incremental MC", "monte_carlo_methods_incremental_mc"),
            (tab_adv, "Advanced IS", "monte_carlo_methods_advanced_is"),
            (tab_dash, "Dashboard", "monte_carlo_methods_dashboard"),
        ]
        for tab, note_title, note_slug in pending_tabs:
            with tab:
                st.info("👈 Press **Run All Methods** in the sidebar to start the experiment.")
                render_mc_notes(note_title, note_slug)

    # ── Tab 7: Method Guide (always visible) ──────────────────────────────────
    with tab_guide:
        st.markdown("""
        <div style="background:#12121f; border-radius:12px; padding:1.2rem 1.6rem; border:1px solid #2a2a3e; margin-bottom:1.2rem">

        ### 📚 How to use this guide

        Each section below covers one MC method. For every method you'll find:

        | Field | What it tells you |
        |-------|------------------|
        | **📌 What it does** | Plain-English description — no equations needed |
        | **🕐 When to use it** | Practical scenarios where this method is the right choice |
        | **✅ Pros** | What the method does well |
        | **❌ Cons** | Known limitations and failure modes |
        | **📊 Variance behaviour** | How stable/noisy the estimates are — critical for practical use |
        | **🔗 Relation to others** | Where this method fits in the MC family tree |

        ---

        ### 🧠 The One-Sentence Intuition for Each Method

        | Method | One sentence |
        |--------|-------------|
        | **First-Visit MC** | *"Play games, look back at your first time in each place, average what happened after."* |
        | **Every-Visit MC** | *"Like First-Visit but count every time you passed through a location — more data, slightly messier stats."* |
        | **On-policy Control** | *"Improve your strategy after every game, while always keeping a small chance of trying something new."* |
        | **Ordinary IS** | *"Reweight old data from a different strategy to evaluate a new one — but the weights can be explosively large."* |
        | **Weighted IS** | *"Same as Ordinary IS, but normalise the weights — trades perfect accuracy for much more stable estimates."* |
        | **Incremental MC** | *"Instead of saving every past result, just update a running average — same answer, infinitely less memory."* |
        | **Per-Decision IS** | *"Each step's reward only needs the IS correction up to that step — cutting the weight products shorter reduces noise."* |
        | **Discounting-Aware IS** | *"When future rewards matter less (γ&lt;1), their corrections also matter less — exploit this for lowest-variance IS."* |

        </div>
        """, unsafe_allow_html=True)
        st.subheader("📚 Complete MC Method Reference")

        entries = [
            {
                "icon": "🔵", "name": "MC Prediction — First-Visit",
                "what":  "Estimates V(s) by averaging the return from the **first** time a state appears in each episode.",
                "when":  "Policy evaluation for a fixed policy; theoretical analyses where unbiasedness matters.",
                "pros":  "✅ Unbiased | ✅ Statistically independent samples per episode | ✅ Simple",
                "cons":  "❌ Wastes data when states revisited | ❌ Requires complete episodes",
                "variance": "LOW — each state contributes one independent return per episode.",
                "relation": "Foundation of all MC prediction. Every-Visit MC relaxes the uniqueness constraint.",
            },
            {
                "icon": "🟣", "name": "MC Prediction — Every-Visit",
                "what":  "Like First-Visit but counts the state **every** time it appears. More data per episode.",
                "when":  "When sample efficiency matters and slight bias is acceptable.",
                "pros":  "✅ More updates per episode | ✅ Faster practical convergence",
                "cons":  "❌ Correlated samples within an episode | ❌ Slight bias for loopy environments",
                "variance": "MEDIUM — more samples but correlated, so variance reduction is not proportional.",
                "relation": "Same asymptotic limit as First-Visit. Both converge to V^π(s) as N→∞.",
            },
            {
                "icon": "🟢", "name": "On-policy MC Control (ε-greedy)",
                "what":  "Estimates Q(s,a) and improves policy with ε-greedy every episode (GPI).",
                "when":  "Learning from direct interaction; when exploration must be maintained throughout.",
                "pros":  "✅ No model needed | ✅ Simple | ✅ Guaranteed exploration via ε",
                "cons":  "❌ Convergent only to best ε-soft policy, not π* | ❌ Requires complete episodes",
                "variance": "MEDIUM — action-value estimates stable, but ε-greedy introduces stochastic noise.",
                "relation": "The on-policy MC counterpart to Q-learning. Off-policy IS removes the ε-soft constraint.",
            },
            {
                "icon": "🔴", "name": "Off-policy — Ordinary IS",
                "what":  "Reweights returns from behavior policy b to evaluate target policy π using ρ = ∏π/b.",
                "when":  "Reusing logged data; separating exploration from evaluation.",
                "pros":  "✅ Unbiased | ✅ Can reuse any existing data | ✅ Flexible data collection",
                "cons":  "❌ VERY high variance — ρ can grow exponentially with episode length",
                "variance": "HIGH — product of many ratios → variance explodes for long episodes.",
                "relation": "Direct off-policy generalization. Weighted IS solves the variance problem.",
            },
            {
                "icon": "🔵", "name": "Off-policy — Weighted IS",
                "what":  "Like Ordinary IS but uses ρ as a weight in a weighted average: Σ(ρG)/Σρ.",
                "when":  "Almost always preferred over Ordinary IS; better bias-variance tradeoff.",
                "pros":  "✅ Much lower variance | ✅ Practically stable | ✅ Consistent estimator",
                "cons":  "❌ Biased (consistent but not unbiased for finite N)",
                "variance": "LOW — denominator normalizes explosive weights.",
                "relation": "Preferred off-policy method. Per-Decision IS further decomposes the weight.",
            },
            {
                "icon": "⚡", "name": "Incremental MC",
                "what":  "Implements First-Visit MC with an online update rule: V(s) += (1/N)(G−V(s)).",
                "when":  "Memory-constrained settings; streaming data; non-stationary with constant α.",
                "pros":  "✅ O(1) memory | ✅ Equivalent to batch averaging | ✅ Bridge to TD learning",
                "cons":  "❌ No fundamental variance improvement over First-Visit",
                "variance": "LOW (same as First-Visit). Using constant α trades bias for tracking ability.",
                "relation": "Implementation variant of First-Visit MC. Replace G with bootstrapped estimate → TD(0).",
            },
            {
                "icon": "🟠", "name": "Per-Decision IS",
                "what":  "Decomposes the IS weight: each reward Rₖ is weighted only by ratios up to step k.",
                "when":  "Off-policy evaluation where you need lower variance than standard IS.",
                "pros":  "✅ Provably lower variance than Ordinary/Weighted IS | ✅ Unbiased",
                "cons":  "❌ More complex implementation | ❌ Still requires full episodes",
                "variance": "VERY LOW — avoids multiplying future ratios into past rewards.",
                "relation": "Intermediate between Weighted IS and Discounting-Aware IS.",
            },
            {
                "icon": "🟡", "name": "Discounting-Aware IS",
                "what":  "Exploits γ<1 to decompose returns into flat partial returns, further reducing IS weight variance.",
                "when":  "Maximum variance reduction in off-policy evaluation with γ<1.",
                "pros":  "✅ Lowest variance of all IS estimators | ✅ Unbiased | ✅ Optimal for discounted MDPs",
                "cons":  "❌ Most complex | ❌ Benefit vanishes as γ→1",
                "variance": "LOWEST — discounting limits the contribution of distant states, reducing IS product length.",
                "relation": "Most advanced IS method. Reduces to Per-Decision IS when γ=1.",
            },
        ]

        for e in entries:
            with st.expander(f"{e['icon']} {e['name']}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**📌 What it does:**  \n{e['what']}")
                    st.markdown(f"**🕐 When to use:**  \n{e['when']}")
                    st.markdown(f"**✅ Pros:**  \n{e['pros']}")
                with c2:
                    st.markdown(f"**❌ Cons:**  \n{e['cons']}")
                    st.markdown(f"**📊 Variance behaviour:**  \n{e['variance']}")
                    st.markdown(f"**🔗 Relation to others:**  \n{e['relation']}")

        st.divider()
        st.markdown("""
        ### 🗺️ Big Picture: How MC Methods Connect

        ```
        MC PREDICTION
        ├── First-Visit MC   (unbiased, O(1) with incremental update)
        └── Every-Visit MC   (faster convergence, slight bias)
               │
               ▼
        MC CONTROL (On-policy)
        └── ε-greedy GPI     (explores + improves; converges to best ε-soft π)
               │
               ▼  (separate exploration from evaluation)
        OFF-POLICY EVALUATION
        ├── Ordinary IS      (unbiased, explosive variance)
        └── Weighted IS      (biased, low variance) ← preferred in practice
               │
               ▼  (further variance reduction)
        ADVANCED IS
        ├── Per-Decision IS  (decompose weight per reward)
        └── Discounting-Aware IS  (exploit γ<1 structure)

        ALL MC methods share:
        • Require complete episodes (episodic MDPs)
        • No bootstrapping (use actual returns G_t)
        • Model-free (learn from experience only)
        • High variance compared to TD (but zero bias for prediction)
        ```

        ---

        ### ❓ Frequently Asked Questions (Non-Scientist Edition)

        **Q: Why does MC need "complete episodes"? Why not learn after each step?**
        Because MC computes the *true return G* — the actual total reward from that point to the end.
        You can't know G until the episode is over. Methods that learn after each step (TD methods) instead
        *estimate* G using a partial calculation — they bootstrap. MC never guesses: it waits for the real answer.

        **Q: What does "high variance" actually mean in practice?**
        It means if you run the same experiment twice with different random seeds, you'll get very different
        value estimates. Practically: you need to run many more episodes to get reliable results.
        Low variance methods give you consistent answers from fewer episodes — that's why engineers prefer them.

        **Q: When would I actually use off-policy MC in real life?**
        Classic use case: **clinical trials**. You have historical patient data where doctors chose treatments
        based on their own judgment (the behavior policy). You want to evaluate "what would have happened if we
        always followed treatment guideline B?" (the target policy). Off-policy IS gives you this counterfactual
        estimate without running a new trial.

        **Q: If Discounting-Aware IS is best, why do we need the other IS methods?**
        The hierarchy only holds for γ < 1. For γ = 1 (undiscounted tasks), Discounting-Aware IS collapses to
        Per-Decision IS. And Weighted IS is simpler to implement correctly — in practice it's often the
        first choice. The advanced methods are worth the complexity only when episodes are long and variance
        is actually causing problems.

        **Q: What's the difference between MC and Deep RL methods like DQN?**
        DQN (and most modern deep RL) uses TD learning — it bootstraps (estimates G using current value estimates)
        and trains neural networks instead of lookup tables. MC methods are the theoretical foundation:
        simpler, more interpretable, exactly correct in the limit — but slow to converge and only applicable
        when episodes terminate. Deep RL trades some of that purity for scale and speed.
        """)
        render_mc_notes("Method Guide", "monte_carlo_methods_method_guide")


if __name__ == "__main__":
    main()
