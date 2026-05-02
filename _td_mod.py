"""
TD RL Explorer — Streamlit App
A visual, interactive textbook for Temporal-Difference learning methods.
Environment: CliffWalking (4×12) — the canonical TD example from Sutton & Barto Ch.6
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
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
.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background:#12121f; border-radius:10px; padding:4px;
}
.stTabs [data-baseweb="tab"] {
    background:#1e1e2e; border-radius:8px; color:#b0b0cc;
    padding:7px 14px; font-weight:600; font-size:.88rem;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#e65100,#1565c0);
    color:white !important;
}
div[data-testid="metric-container"] {
    background:#1e1e2e; border-radius:10px; padding:12px; border:1px solid #2d2d44;
}
</style>
""", unsafe_allow_html=True)


def render_td_notes(tab_title: str, tab_slug: str) -> None:
    render_notes(f"Temporal-Difference Learning - {tab_title}", tab_slug)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d0d1a"
CARD_BG    = "#12121f"
GRID_COLOR = "#2a2a3e"
RL_CMAP    = LinearSegmentedColormap.from_list(
    "rl", ["#b71c1c", "#f57f17", "#fff176", "#2e7d32"])

METHOD_COLORS = {
    "TD(0)":          "#7c4dff",
    "SARSA":          "#e65100",
    "Q-Learning":     "#1565c0",
    "Expected SARSA": "#00897b",
    "Double Q":       "#ad1457",
    "n-step TD":      "#f57f17",
    "TD(λ)":          "#558b2f",
    "MC":             "#546e7a",
}

# ─────────────────────────────────────────────────────────────────────────────
# TEACHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _card(color, icon, title, body):
    return f"""<div style="background:{color}18; border-left:4px solid {color};
    padding:1rem 1.2rem; border-radius:0 10px 10px 0; margin-bottom:1rem">
    <b>{icon} {title}</b><br>{body}</div>"""

def _tip(text):
    return f'<div style="background:#1a2a1a; border-left:3px solid #4caf50; padding:.65rem 1rem; border-radius:0 6px 6px 0; margin:.5rem 0; font-size:.93rem">{text}</div>'

def _warn(text):
    return f'<div style="background:#2a1a1a; border-left:3px solid #ef5350; padding:.65rem 1rem; border-radius:0 6px 6px 0; margin:.5rem 0; font-size:.93rem">{text}</div>'

def _insight(text):
    return f'<div style="background:#1a1a2a; border-left:3px solid #7c4dff; padding:.65rem 1rem; border-radius:0 6px 6px 0; margin:.5rem 0; font-size:.93rem">{text}</div>'

# ─────────────────────────────────────────────────────────────────────────────
# CLIFFWALKING ENVIRONMENT  (Sutton & Barto §6.5 — the canonical TD example)
# ─────────────────────────────────────────────────────────────────────────────
class CliffWalking:
    """
    4×12 grid.  Start S=(3,0), Goal G=(3,11).
    Cliff: row 3, cols 1-10 → reward −100, agent teleports back to S.
    Every non-terminal step: reward −1.
    4 actions: 0=Up, 1=Right, 2=Down, 3=Left.

    Why CliffWalking for TD?
    - SARSA (on-policy) learns the SAFE path (one row above the cliff).
    - Q-Learning (off-policy) learns the OPTIMAL path (right along the cliff edge).
    - This difference is the single most vivid demonstration of on vs off-policy TD.
    - No walls, simple geometry — easy to read the policy arrows.
    """
    ROWS    = 4
    COLS    = 12
    ACTIONS = [0, 1, 2, 3]
    SYMBOLS = ["↑", "→", "↓", "←"]
    DELTAS  = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self):
        self.start  = (3, 0)
        self.goal   = (3, 11)
        self.cliff  = frozenset((3, c) for c in range(1, 11))
        self.n_states  = self.ROWS * self.COLS
        self.n_actions = 4

    def s2i(self, s): return s[0] * self.COLS + s[1]
    def i2s(self, i): return (i // self.COLS, i % self.COLS)
    def is_terminal(self, s): return s == self.goal

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0.0, True
        dr, dc = self.DELTAS[a]
        ns = (max(0, min(self.ROWS-1, s[0]+dr)),
              max(0, min(self.COLS-1, s[1]+dc)))
        if ns in self.cliff:
            return self.start, -100.0, False   # fall off cliff — back to start
        if ns == self.goal:
            return ns, -1.0, True
        return ns, -1.0, False

    def uniform_policy(self):
        return {self.i2s(i): np.ones(4)/4 for i in range(self.n_states)}

    def eps_greedy_action(self, Q, s, eps):
        if np.random.random() < eps:
            return np.random.randint(4)
        return int(np.argmax(Q[s]))

    def greedy_policy(self, Q):
        return {self.i2s(i): int(np.argmax(Q[self.i2s(i)]))
                for i in range(self.n_states)}


# ─────────────────────────────────────────────────────────────────────────────
# TD(0) PREDICTION  (§6.1)
# ─────────────────────────────────────────────────────────────────────────────
def td0_prediction(env, policy, n_episodes, alpha, gamma):
    """
    TD(0): V(S) ← V(S) + α[R + γV(S') - V(S)]
    Updates after EVERY step using a one-step bootstrapped target.
    No need to wait for episode end.
    """
    V = defaultdict(float)
    history = []

    for ep in range(n_episodes):
        s = env.start
        for _ in range(500):                   # step limit per episode
            p   = policy[s]
            a   = int(np.random.choice(4, p=p))
            ns, r, done = env.step(s, a)

            # TD(0) update — the core of the algorithm
            td_target = r + gamma * V[ns] * (not done)
            td_error  = td_target - V[s]
            V[s]     += alpha * td_error

            s = ns
            if done:
                break

        if (ep + 1) % max(1, n_episodes // 20) == 0:
            history.append(dict(V))

    return V, history


# ─────────────────────────────────────────────────────────────────────────────
# SARSA — on-policy TD control  (§6.4)
# ─────────────────────────────────────────────────────────────────────────────
def sarsa(env, n_episodes, alpha, gamma, epsilon):
    """
    SARSA: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
    On-policy: the action A' used in the update is the SAME action
    the agent will actually take next (sampled from ε-greedy policy).
    This causes SARSA to be cautious near the cliff.
    """
    Q = defaultdict(lambda: np.zeros(4))
    ep_rewards = []
    q_history  = []

    for ep in range(n_episodes):
        s = env.start
        a = env.eps_greedy_action(Q, s, epsilon)
        total_r = 0.0

        for _ in range(500):
            ns, r, done = env.step(s, a)
            total_r += r

            a_next = env.eps_greedy_action(Q, ns, epsilon)  # ε-greedy next action
            # SARSA update: uses the next action that WILL be taken
            Q[s][a] += alpha * (r + gamma * Q[ns][a_next] * (not done) - Q[s][a])

            s, a = ns, a_next
            if done:
                break

        ep_rewards.append(total_r)
        if (ep + 1) % max(1, n_episodes // 20) == 0:
            q_history.append({k: v.copy() for k, v in Q.items()})

    return Q, ep_rewards, q_history


# ─────────────────────────────────────────────────────────────────────────────
# Q-LEARNING — off-policy TD control  (§6.5)
# ─────────────────────────────────────────────────────────────────────────────
def q_learning(env, n_episodes, alpha, gamma, epsilon):
    """
    Q-Learning: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
    Off-policy: the target uses max_a Q(S',a) — the GREEDY action —
    regardless of what the ε-greedy policy actually does next.
    This makes Q-Learning learn the optimal path (dangerous cliff edge).
    """
    Q = defaultdict(lambda: np.zeros(4))
    ep_rewards = []
    q_history  = []

    for ep in range(n_episodes):
        s = env.start
        total_r = 0.0

        for _ in range(500):
            a        = env.eps_greedy_action(Q, s, epsilon)
            ns, r, done = env.step(s, a)
            total_r += r

            # Q-Learning update: uses the GREEDY next action for target
            Q[s][a] += alpha * (r + gamma * np.max(Q[ns]) * (not done) - Q[s][a])
            s = ns
            if done:
                break

        ep_rewards.append(total_r)
        if (ep + 1) % max(1, n_episodes // 20) == 0:
            q_history.append({k: v.copy() for k, v in Q.items()})

    return Q, ep_rewards, q_history


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED SARSA  (§6.6)
# ─────────────────────────────────────────────────────────────────────────────
def expected_sarsa(env, n_episodes, alpha, gamma, epsilon):
    """
    Expected SARSA: Q(S,A) ← Q(S,A) + α[R + γ Σ_a π(a|S')Q(S',a) - Q(S,A)]
    Uses the EXPECTED next Q under the ε-greedy policy — not the sampled action.
    Lower variance than SARSA, higher than Q-Learning. Can be on or off-policy.
    """
    Q = defaultdict(lambda: np.zeros(4))
    ep_rewards = []

    for ep in range(n_episodes):
        s = env.start
        total_r = 0.0

        for _ in range(500):
            a        = env.eps_greedy_action(Q, s, epsilon)
            ns, r, done = env.step(s, a)
            total_r += r

            # Expected value under ε-greedy policy
            best_a   = int(np.argmax(Q[ns]))
            pi       = np.full(4, epsilon / 4)
            pi[best_a] += 1.0 - epsilon
            expected_q = float(np.dot(pi, Q[ns]))

            Q[s][a] += alpha * (r + gamma * expected_q * (not done) - Q[s][a])
            s = ns
            if done:
                break

        ep_rewards.append(total_r)

    return Q, ep_rewards


# ─────────────────────────────────────────────────────────────────────────────
# DOUBLE Q-LEARNING  (§6.7)
# ─────────────────────────────────────────────────────────────────────────────
def double_q_learning(env, n_episodes, alpha, gamma, epsilon):
    """
    Double Q-Learning: Maintains two Q tables QA and QB.
    Each update randomly picks one table to update, using the OTHER table's
    value for the target. This decouples action selection from value estimation,
    eliminating the maximisation bias of standard Q-Learning.
    """
    QA = defaultdict(lambda: np.zeros(4))
    QB = defaultdict(lambda: np.zeros(4))
    ep_rewards = []

    for ep in range(n_episodes):
        s = env.start
        total_r = 0.0

        for _ in range(500):
            # Action from combined policy (average of QA and QB)
            combined = (np.array(QA[s]) + np.array(QB[s])) / 2
            a = int(np.argmax(combined)) if np.random.random() > epsilon \
                else np.random.randint(4)
            ns, r, done = env.step(s, a)
            total_r += r

            # 50/50 split: update QA using QB's value, or vice versa
            if np.random.random() < 0.5:
                a_star = int(np.argmax(QA[ns]))          # select with QA
                target = r + gamma * QB[ns][a_star] * (not done)  # evaluate with QB
                QA[s][a] += alpha * (target - QA[s][a])
            else:
                a_star = int(np.argmax(QB[ns]))
                target = r + gamma * QA[ns][a_star] * (not done)
                QB[s][a] += alpha * (target - QB[s][a])

            s = ns
            if done:
                break

        ep_rewards.append(total_r)

    # Combine: final Q = (QA + QB) / 2
    Q_combined = defaultdict(lambda: np.zeros(4))
    for s in set(list(QA.keys()) + list(QB.keys())):
        Q_combined[s] = (np.array(QA[s]) + np.array(QB[s])) / 2

    return Q_combined, ep_rewards


# ─────────────────────────────────────────────────────────────────────────────
# N-STEP TD  (§7.1)
# ─────────────────────────────────────────────────────────────────────────────
def nstep_td_sarsa(env, n_episodes, alpha, gamma, epsilon, n):
    """
    n-step SARSA: target = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})
    n=1 → standard SARSA
    n=∞ → Monte Carlo (uses full return, no bootstrapping)
    n in between → best of both: some real rewards, some bootstrapping.
    """
    Q = defaultdict(lambda: np.zeros(4))
    ep_rewards = []

    for ep in range(n_episodes):
        s     = env.start
        a     = env.eps_greedy_action(Q, s, epsilon)
        states   = [s]
        actions  = [a]
        rewards  = [0.0]    # dummy for index alignment
        total_r  = 0.0
        T        = float('inf')
        t        = 0

        while True:
            if t < T:
                ns, r, done = env.step(states[t], actions[t])
                total_r += r
                states.append(ns)
                rewards.append(r)
                if done:
                    T = t + 1
                else:
                    a_next = env.eps_greedy_action(Q, ns, epsilon)
                    actions.append(a_next)

            tau = t - n + 1          # state being updated
            if tau >= 0:
                # Compute n-step return
                end = min(tau + n, T)
                G   = sum(gamma**(i - tau - 1) * rewards[i] for i in range(tau+1, end+1))
                if tau + n < T:
                    G += gamma**n * Q[states[tau+n]][actions[tau+n]]

                s_tau = states[tau]
                a_tau = actions[tau]
                Q[s_tau][a_tau] += alpha * (G - Q[s_tau][a_tau])

            t += 1
            if tau == T - 1:
                break

        ep_rewards.append(total_r)

    return Q, ep_rewards


# ─────────────────────────────────────────────────────────────────────────────
# TD(λ) WITH ELIGIBILITY TRACES — SARSA(λ)  (§12.2)
# ─────────────────────────────────────────────────────────────────────────────
def sarsa_lambda(env, n_episodes, alpha, gamma, lam, epsilon):
    """
    SARSA(λ) with replacing eligibility traces.
    E(s,a) tracks which (state,action) pairs were recently visited.
    On each step ALL Q values are updated proportionally to their trace.
    λ=0 → SARSA (only current step updated)
    λ=1 → Monte Carlo (all steps updated equally, episode-long memory)
    λ in between → traces decay exponentially; recent steps updated more.
    """
    Q  = defaultdict(lambda: np.zeros(4))
    ep_rewards = []

    for ep in range(n_episodes):
        s  = env.start
        a  = env.eps_greedy_action(Q, s, epsilon)
        E  = defaultdict(lambda: np.zeros(4))   # eligibility traces
        total_r = 0.0

        for _ in range(500):
            ns, r, done = env.step(s, a)
            total_r += r

            a_next = env.eps_greedy_action(Q, ns, epsilon)
            delta  = r + gamma * Q[ns][a_next] * (not done) - Q[s][a]

            # Accumulate trace for current (s,a) — replacing traces version
            E[s][a] = 1.0

            # Update ALL state-action pairs proportional to trace
            for st in E:
                Q[st]  += alpha * delta * E[st]
                E[st]  *= gamma * lam    # traces decay each step

            s, a = ns, a_next
            if done:
                break

        ep_rewards.append(total_r)

    return Q, ep_rewards


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL METHODS
# ─────────────────────────────────────────────────────────────────────────────
def run_all(env, n_ep, alpha, gamma, eps, n_step, lam, seed):
    np.random.seed(seed)

    fixed_pol  = env.uniform_policy()
    V_td0, h_td0 = td0_prediction(env, fixed_pol, n_ep, alpha, gamma)

    Q_sarsa, rew_sarsa, qh_sarsa   = sarsa(env, n_ep, alpha, gamma, eps)
    Q_ql,    rew_ql,    qh_ql      = q_learning(env, n_ep, alpha, gamma, eps)
    Q_esarsa, rew_esarsa            = expected_sarsa(env, n_ep, alpha, gamma, eps)
    Q_dql,   rew_dql                = double_q_learning(env, n_ep, alpha, gamma, eps)
    Q_nstep, rew_nstep              = nstep_td_sarsa(env, n_ep, alpha, gamma, eps, n_step)
    Q_lam,   rew_lam                = sarsa_lambda(env, n_ep, alpha, gamma, lam, eps)

    return dict(
        V_td0=V_td0, h_td0=h_td0,
        Q_sarsa=Q_sarsa,  rew_sarsa=rew_sarsa,  qh_sarsa=qh_sarsa,
        Q_ql=Q_ql,        rew_ql=rew_ql,        qh_ql=qh_ql,
        Q_esarsa=Q_esarsa, rew_esarsa=rew_esarsa,
        Q_dql=Q_dql,      rew_dql=rew_dql,
        Q_nstep=Q_nstep,  rew_nstep=rew_nstep,
        Q_lam=Q_lam,      rew_lam=rew_lam,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _fig_style(fig, axes_flat):
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes_flat:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="#9e9ebb", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR)


def make_fig(nrows=1, ncols=1, w=12, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    axl = np.array(axes).flatten().tolist()
    _fig_style(fig, axl)
    return fig, axes, axl


def plot_cliff_value(env, V, title, ax, vmin=-20, vmax=0):
    """Heatmap of state values over the 4×12 cliff grid."""
    grid = np.full((env.ROWS, env.COLS), np.nan)
    for i in range(env.n_states):
        s = env.i2s(i)
        if s not in env.cliff and s != env.goal:
            grid[s] = V.get(s, 0.0)

    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=RL_CMAP, vmin=vmin, vmax=vmax, aspect="auto")

    # Cliff overlay
    for (r, c) in env.cliff:
        ax.add_patch(plt.Rectangle((c-.5, r-.5), 1, 1, color="#4a0000", zorder=3))
        if c == 5:
            ax.text(c, r, "⚠ CLIFF  −100", ha="center", va="center",
                    color="#ff5252", fontsize=7, fontweight="bold", zorder=4)

    # Goal/Start
    ax.add_patch(plt.Rectangle((env.goal[1]-.5, env.goal[0]-.5), 1, 1,
                                color="#1b5e20", alpha=0.9, zorder=3))
    ax.text(env.goal[1], env.goal[0], "★\nGOAL", ha="center", va="center",
            color="white", fontsize=7, fontweight="bold", zorder=4)
    ax.text(env.start[1], env.start[0], "●\nSTART", ha="center", va="center",
            color="#90caf9", fontsize=7, fontweight="bold", zorder=4)

    # Value labels
    for i in range(env.ROWS):
        for j in range(env.COLS):
            s = (i, j)
            if s in env.cliff or s == env.goal:
                continue
            val = V.get(s, 0.0)
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=6, color="white" if val < vmin*0.4 else "black", zorder=4)

    ax.set_xticks(range(env.COLS))
    ax.set_yticks(range(env.ROWS))
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(colors="#9e9ebb", labelsize=7)


def plot_cliff_policy(env, Q, title, ax, color="#7c4dff"):
    """Policy arrows over the cliff grid — shows safe vs risky paths clearly."""
    arrow_map = {0:(0,-.35), 1:(.35,0), 2:(0,.35), 3:(-.35,0)}

    ax.set_xlim(-.5, env.COLS-.5)
    ax.set_ylim(-.5, env.ROWS-.5)
    ax.set_aspect("equal")
    ax.set_facecolor(DARK_BG)
    ax.set_xticks(range(env.COLS)); ax.set_yticks(range(env.ROWS))
    ax.grid(color=GRID_COLOR, lw=.4, alpha=.6)

    for i in range(env.ROWS):
        for j in range(env.COLS):
            s = (i, j)
            if s in env.cliff:
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, color="#4a0000", zorder=2))
                if j == 5:
                    ax.text(j, i, "⚠ CLIFF", ha="center", va="center",
                            color="#ff5252", fontsize=6, fontweight="bold", zorder=3)
            elif s == env.goal:
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, color="#1b5e20", alpha=.9, zorder=2))
                ax.text(j, i, "★\nGOAL", ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold", zorder=3)
            elif s == env.start:
                ax.text(j, i, "●\nS", ha="center", va="center",
                        color="#90caf9", fontsize=7, fontweight="bold", zorder=3)
            else:
                a  = int(np.argmax(Q.get(s, np.zeros(4))))
                dc, dr = arrow_map[a]
                ax.annotate("", xy=(j+dc, i+dr), xytext=(j, i),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=3)

    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
    ax.tick_params(colors="#9e9ebb", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)


def smooth(arr, w=20):
    if len(arr) <= w:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w)/w, mode="valid")


def plot_learning_curves(ax, curves_dict, title, window=20):
    """Plot smoothed episode reward curves for multiple methods."""
    for name, (rewards, color) in curves_dict.items():
        raw = np.array(rewards, dtype=float)
        sm  = smooth(raw, window)
        ax.plot(raw, color=color, alpha=0.12, lw=0.5)
        ax.plot(range(len(sm)), sm, color=color, lw=2.2, label=name)
    ax.set_xlabel("Episode", color="white", fontsize=10)
    ax.set_ylabel("Total reward", color="white", fontsize=10)
    ax.set_title(title, color="white", fontweight="bold")
    ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
    ax.grid(alpha=0.15)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main_td():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#bf360c,#0d47a1,#1b5e20);
                padding:2rem 2.5rem; border-radius:14px; margin-bottom:1.5rem">
        <h1 style="color:white;margin:0;font-size:2.4rem">⚡ TD RL Explorer</h1>
        <p style="color:#b0bec5;margin-top:.5rem;font-size:1.05rem">
            Temporal-Difference Learning — from TD(0) to SARSA(λ) — every formula decoded, every chart explained
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Global intro ─────────────────────────────────────────────────────────
    with st.expander("🎓 New here? — What is Temporal-Difference Learning?", expanded=False):
        st.markdown(r"""
        <div style="background:#12121f; border-radius:12px; padding:1.4rem 1.8rem; border:1px solid #2a2a3e">

        ### 🆚 MC vs TD — The Fundamental Trade-off

        | Property | Monte Carlo | Temporal-Difference |
        |----------|------------|---------------------|
        | **When it learns** | After full episode ends | After **every step** |
        | **What it uses** | True return G (actual outcome) | Estimated return (bootstrap) |
        | **Bias** | Zero bias | Some bias (guess of future) |
        | **Variance** | High (many steps, many rewards) | **Low** (one-step look-ahead) |
        | **Works on** | Episodic tasks only | Episodic **and** continuing tasks |
        | **Speed** | Slow (waits for episode) | **Fast** (online, every step) |

        > **The MC analogy:** You play a full chess game, then at the end you ask "which moves were good?"
        > You know the real outcome — no guessing.
        >
        > **The TD analogy:** After every move, you ask "does my position look better or worse than I expected?"
        > You don't know the final result yet — you're making an educated guess based on the current board.

        ---

        ### ⚡ The TD Update — One Formula to Rule Them All

        Every TD method is a variation of:

        $$Q(S,A) \leftarrow Q(S,A) + \alpha \underbrace{[R + \gamma \hat{Q}(S',\cdot) - Q(S,A)]}_{\text{TD error } \delta}$$

        The **TD error** $\delta$ = "how surprised was I by what happened?"
        - $R$ = actual reward received
        - $\gamma \hat{Q}(S',\cdot)$ = estimated future value (bootstrap)
        - $Q(S,A)$ = what I expected

        The methods differ **only** in how they compute $\hat{Q}(S',\cdot)$:

        | Method | $\hat{Q}(S',\cdot)$ | Key property |
        |--------|---------------------|-------------|
        | **SARSA** | $Q(S', A')$ — next action actually taken | On-policy, cautious |
        | **Q-Learning** | $\max_a Q(S',a)$ — best possible | Off-policy, optimal |
        | **Expected SARSA** | $\mathbb{E}_\pi[Q(S',a)]$ — weighted average | Lower variance than SARSA |
        | **Double Q** | $Q_B(S', \arg\max_a Q_A(S',a))$ | No maximisation bias |

        ---

        ### 🗺️ Why CliffWalking?

        CliffWalking (Sutton & Barto §6.5) is THE canonical TD example because it makes the
        difference between on-policy and off-policy methods **visible as a path choice**:

        - **SARSA** (on-policy): learns the **safe path** one row above the cliff — because it
          accounts for accidental cliff-falls during ε-greedy exploration
        - **Q-Learning** (off-policy): learns the **optimal path** right along the cliff edge —
          because it learns as if it will always act greedily, ignoring exploration accidents

        This single environment demonstrates the most important practical difference between the methods.

        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption("Adjust and click Run to compare all methods.")
        n_ep    = st.slider("Episodes", 100, 3000, 500, 100,
                    help="Number of complete training episodes per method.")
        alpha   = st.slider("α (learning rate)", 0.01, 1.0, 0.5, 0.01,
                    help="How much each new experience updates the current estimate. α=1 means fully replace; α=0.1 means 10% nudge.")
        gamma   = st.slider("γ (discount)", 0.80, 1.00, 0.99, 0.01,
                    help="How much future rewards matter. γ=1 = all future counts equally.")
        eps     = st.slider("ε (exploration)", 0.01, 0.50, 0.10, 0.01,
                    help="Fraction of steps where a random action is taken instead of the best known one.")
        n_step  = st.slider("n (n-step TD)", 1, 20, 4, 1,
                    help="How many real steps to look ahead before bootstrapping. n=1→SARSA, n=large→Monte Carlo.")
        lam     = st.slider("λ (eligibility trace)", 0.0, 1.0, 0.8, 0.05,
                    help="Trace decay rate. λ=0→SARSA, λ=1→Monte Carlo. Traces spread credit backward through an episode.")
        seed    = st.number_input("Random seed", 0, 9999, 42)
        run_btn = st.button("🚀 Run All Methods", type="primary", use_container_width=True)

        st.divider()
        st.markdown("""
        **CliffWalking legend**
        | Symbol | Meaning |
        |--------|---------|
        | ● | Start (3,0) |
        | ★ | Goal (3,11) |
        | ⚠ | Cliff (row 3, cols 1-10) |
        | Step | −1 reward |
        | Cliff fall | −100, reset to Start |

        **7 TD Methods:**
        ```
        Prediction
          └─ TD(0)
        On-policy Control
          ├─ SARSA
          └─ SARSA(λ)
        Off-policy Control
          ├─ Q-Learning
          ├─ Expected SARSA
          └─ Double Q-Learning
        Bridge Methods
          └─ n-step TD
        ```
        """)

    env = CliffWalking()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🗺️ Environment",
        "🔮 TD(0) Prediction",
        "🎯 SARSA",
        "🎲 Q-Learning",
        "🔄 Expected SARSA",
        "🎭 Double Q-Learning",
        "🪜 n-step TD",
        "🌊 TD(λ) / SARSA(λ)",
        "📈 Dashboard",
        "📚 Method Guide",
    ])
    (tab_env, tab_td0, tab_sarsa, tab_ql, tab_esarsa,
     tab_dql, tab_nstep, tab_lam, tab_dash, tab_guide) = tabs

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 0 — ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════════
    with tab_env:
        st.markdown(_card("#42a5f5","🗺️","Why CliffWalking?",
            """CliffWalking is used in Sutton & Barto Chapter 6 as the definitive example for TD methods
            because it creates a <b>visible, dramatic difference</b> between SARSA and Q-Learning.
            The environment is simple enough to understand in seconds, yet rich enough to show all
            the key properties of TD learning: online updates, bootstrapping, and the on-policy vs
            off-policy distinction."""), unsafe_allow_html=True)

        st.subheader("🗺️ The 4×12 CliffWalking Environment")
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            st.markdown("""
            #### The task
            The agent starts at **S = (3, 0)** — bottom-left — and must reach **G = (3, 11)** — bottom-right.
            The fastest route (bottom row) passes right along the cliff edge. One wrong step (or one
            unlucky random action during ε-greedy exploration) sends the agent back to Start with a −100 penalty.

            #### Why this is the perfect TD environment

            **The safe path** (row 2, one above the cliff): 13 steps, return ≈ −13.
            **The optimal path** (row 3, cliff edge): ~11 steps, return ≈ −11 — but risks −100 if the
            agent slips. Under ε-greedy exploration, SARSA accounts for these slip accidents in its updates.
            Q-Learning doesn't — it learns the optimal path as if it will never explore randomly.

            #### Reward structure
            | Event | Reward | Episode ends? |
            |-------|--------|---------------|
            | Normal step | **−1** | No |
            | Reach goal ★ | **−1** (last step) | **Yes** |
            | Fall off cliff | **−100**, return to S | No |

            The −1 per step discourages wandering. There is no +10 goal bonus — the agent must
            minimise total steps (and cliff falls). The optimal undiscounted return is **−12**
            (11 right-moves + 1 up-move = 12 steps).
            """)

        with c2:
            st.caption("📖 **How to read this map:** The grid is 4 rows × 12 columns. Row 0 = top, row 3 = bottom. The cliff spans the entire bottom row between Start and Goal. Each cell is a state the agent can occupy.")
            fig, ax, _ = make_fig(1, 1, 9, 3)
            ax.set_xlim(-.5, 11.5); ax.set_ylim(-.5, 3.5); ax.set_aspect("equal")
            for i in range(env.ROWS):
                for j in range(env.COLS):
                    s = (i, j)
                    if s in env.cliff:
                        c = "#4a0000"
                        ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1, color=c, ec="#333", lw=.5))
                        if j == 5: ax.text(j, i, "⚠ CLIFF", ha="center", va="center",
                                           color="#ff5252", fontsize=6, fontweight="bold")
                    elif s == env.goal:
                        ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1, color="#1b5e20", ec="#333", lw=.5))
                        ax.text(j, i, "★\nGOAL", ha="center", va="center",
                                color="white", fontsize=6, fontweight="bold")
                    elif s == env.start:
                        ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1, color="#0d47a1", ec="#333", lw=.5))
                        ax.text(j, i, "●\nSTART", ha="center", va="center",
                                color="white", fontsize=6, fontweight="bold")
                    else:
                        ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1, color="#1e1e2e", ec="#2a2a3e", lw=.5))
                        ax.text(j, i, f"({i},{j})", ha="center", va="center", color="#555577", fontsize=5)
            ax.set_xticks(range(12)); ax.set_yticks(range(4))
            ax.invert_yaxis(); ax.set_title("CliffWalking Layout", color="white", fontweight="bold")
            ax.tick_params(colors="#9e9ebb")
            for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        st.subheader("📍 The Two Paths — Safe vs Optimal")
        st.markdown("""
        The central drama of CliffWalking is the choice between two routes:
        """)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            **🔵 Safe path (SARSA learns this)**
            - Row 2 (one above cliff)
            - Steps: up(1) + right×11 + down(1) = **13 steps**
            - Return: **−13** (no risk of −100)
            - SARSA prefers this because ε-greedy exploration
              occasionally causes cliff-falls — SARSA accounts
              for this in its on-policy updates
            """)
        with c2:
            st.markdown("""
            **🟠 Optimal path (Q-Learning learns this)**
            - Row 3 (cliff edge)
            - Steps: right×11 = **11 steps**
            - Return: **−11** (optimal if perfect greedy)
            - Q-Learning learns this because its update uses
              max Q (greedy) — not the actual ε-greedy action.
              It ignores exploration accidents in its updates.
            """)
        with c3:
            st.markdown(r"""
            **🔴 Disaster (cliff fall)**
            - Any step onto row 3, cols 1-10
            - Reward: **−100** + reset to Start
            - Under ε-greedy with ε=0.1, the agent has 10%
              chance of random action at each step → near the
              cliff edge, 2.5% chance of falling per step.
            - Over a 11-step path: $1-(0.975)^{11} ≈ 24\%$ chance of at
              least one fall per episode
            """)

        with st.expander("🔢 Worked example — trace one episode step by step"):
            st.markdown(r"""
            Let's trace what happens when an agent with Q-Learning takes the cliff-edge path,
            with ε=0.1 and α=0.5, γ=0.99. Suppose at step 1:

            | Time | State | Action | Next State | Reward | TD Target | TD Error | Q update |
            |------|-------|--------|-----------|--------|-----------|---------|---------|
            | t=0 | (3,0) | →Right | (3,1) CLIFF | −100 | −100 + 0.99×Q((3,0),best) | large negative | Q drops sharply |
            | t=1 | (3,0) | →Right | (3,1) CLIFF | −100 | same | same | Q drops again |

            After many episodes of cliff falls, Q((3,0),→) becomes very negative.
            The agent learns to go up first: Q((3,0),↑) remains higher.

            **Key insight:** Q-Learning eventually overcomes this because it uses max Q(S',·) —
            which represents the *ideal* future value *if the agent acts greedily from then on*.
            Once ε exploration is reduced, Q-Learning's learned policy (greedy) never falls off the cliff.
            SARSA never fully overcomes it because even after training, it evaluates the *ε-greedy* policy —
            which still has a small chance of random cliff-falls.
            """)
        render_td_notes("Environment", "temporal_difference_learning")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN BUTTON
    # ═══════════════════════════════════════════════════════════════════════
    if run_btn or "td_results" in st.session_state:
        if run_btn:
            with st.spinner("Training all 7 TD methods…"):
                res = run_all(env, n_ep, alpha, gamma, eps, n_step, lam, seed)
            st.session_state["td_results"] = res
            st.sidebar.success("✅ Done!")
        res = st.session_state["td_results"]

        # ═══════════════════════════════════════════════════════════════════
        # TAB 1 — TD(0) PREDICTION
        # ═══════════════════════════════════════════════════════════════════
        with tab_td0:
            st.markdown(_card("#7c4dff","🔮","What does TD(0) Prediction solve?",
                """<b>The problem:</b> Given a fixed policy (here: random), estimate V(s) — how much
                total reward the agent expects from each state.<br><br>
                <b>What makes TD different from MC:</b> MC waits until the episode ends to compute
                the true return G. TD(0) updates V(s) after <em>every single step</em> using a
                <b>bootstrapped estimate</b>: R + γV(S'). It guesses the future using its
                current estimates, then corrects those guesses with real experience.<br><br>
                <b>Why "TD(0)"?</b> The "0" means it looks 0 steps beyond the immediate reward before
                bootstrapping. TD(1) = Monte Carlo. TD(n) = n-step methods."""),
                unsafe_allow_html=True)

            st.subheader("🔮 TD(0) Prediction — Online Value Estimation")

            with st.expander("📐 Theory & Formulas — TD(0)", expanded=False):
                st.markdown(r"""
                #### The Core Update Rule

                $$\boxed{V(S_t) \leftarrow V(S_t) + \alpha \underbrace{\bigl[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\bigr]}_{\delta_t \text{ — TD error}}}$$

                **Symbol decoder:**
                - $V(S_t)$ — current estimate for state $S_t$ (before update)
                - $\alpha$ — learning rate: how large a step to take (your α slider)
                - $R_{t+1}$ — actual reward received after taking the action
                - $\gamma$ — discount factor (your γ slider)
                - $V(S_{t+1})$ — current estimate for the NEXT state (bootstrap)
                - $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ — the **TD error**

                #### What is the TD Error $\delta_t$?

                The TD error measures **surprise**: how much did reality differ from your prediction?

                | $\delta_t$ | Meaning |
                |-----------|---------|
                | $\delta_t > 0$ | "Better than expected" — increase V(S_t) |
                | $\delta_t < 0$ | "Worse than expected" — decrease V(S_t) |
                | $\delta_t = 0$ | "Exactly as expected" — no update needed |

                **The TD target** $R_{t+1} + \gamma V(S_{t+1})$ is a combination of:
                - One step of **real** reward ($R_{t+1}$)
                - An estimated future ($\gamma V(S_{t+1})$) — the bootstrap

                #### Worked numeric example (CliffWalking, γ=0.99, α=0.5)

                Say $V((2,5)) = -8$ and $V((2,6)) = -7$ currently.
                Agent moves right: R = −1 (normal step).

                $$\delta = R + \gamma V(S') - V(S) = -1 + 0.99 \times (-7) - (-8) = -1 - 6.93 + 8 = +0.07$$

                $$V((2,5)) \leftarrow -8 + 0.5 \times 0.07 = -7.965$$

                Tiny positive update — (2,5) is *slightly better* than previously thought
                (because (2,6) is closer to the goal than the -8 estimate implied).

                #### Key Property: Bootstrapping

                TD uses V(S_{t+1}) — its own estimate — as part of the target.
                This is **bootstrapping**: pulling yourself up by your own bootstraps.

                > **Analogy:** A student grades themselves using their own answer key.
                > The answer key gets corrected over time as the student compares with real exam results.
                > Unlike MC (which waits for the real exam score), TD updates the answer key after every question.

                #### Convergence

                TD(0) converges to the true $v_\pi$ for any fixed policy $\pi$ provided:
                - $\sum \alpha_t = \infty$ (enough total learning)
                - $\sum \alpha_t^2 < \infty$ (steps shrink fast enough)

                With fixed α (your slider), it converges to a region near $v_\pi$ but not exactly —
                the bias from bootstrapping never fully disappears with fixed α.
                """)

            st.markdown(_card("#7c4dff","📖","How to read the value heatmap",
                """The heatmap shows V(s) — the agent's estimate of how much total reward it expects
                from each state under the random policy.<br>
                <b>Colour scale:</b> Red = very negative (far from goal, near cliff),
                Green = less negative (close to goal).<br>
                <b>Note:</b> Under a random policy, ALL values are negative — the agent frequently
                falls off the cliff. This is why the cliff area (bottom row) creates dark red values
                in nearby cells too.<br>
                <b>What to notice:</b> Values increase as you move away from the cliff and toward the goal.
                States directly above the cliff are more negative than states in the upper rows
                (closer to the safe path the agent accidentally stumbles upon sometimes)."""),
                unsafe_allow_html=True)

            V_td0 = res["V_td0"]
            fig, axes, axl = make_fig(1, 2, 14, 4)
            plot_cliff_value(env, V_td0, "TD(0) — V(s) under random policy", axes[0], vmin=-50, vmax=0)

            # Convergence trace for a specific state
            h_td0  = res["h_td0"]
            focal  = (2, 5)   # middle of safe path
            td0_tr = [h.get(focal, 0.0) for h in h_td0]
            x_tr   = [(i+1)*max(1, n_ep//20) for i in range(len(td0_tr))]
            axes[1].plot(x_tr, td0_tr, color="#7c4dff", lw=2.5, marker="o", ms=5)
            axes[1].set_xlabel("Episodes", color="white"); axes[1].set_ylabel(f"V{focal}", color="white")
            axes[1].set_title(f"TD(0) Convergence at State {focal}", color="white", fontweight="bold")
            axes[1].grid(alpha=.15)
            axes[1].axhline(td0_tr[-1] if td0_tr else 0, color="#7c4dff", ls=":", alpha=.5)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown(_card("#7c4dff","📖","How to read the convergence chart",
                f"""<b>Left chart:</b> V(s) heatmap — red to green = bad to good states under random policy.<br>
                <b>Right chart:</b> How V{focal} (a mid-path state) changes episode by episode.
                Early: large jumps as first estimates form. Later: smaller adjustments as the
                estimate homes in on the true value. The dotted line shows the final estimate.
                <b>Compare to MC:</b> TD(0) updates V after EVERY step — so it uses data from
                each episode more efficiently. The convergence plot should be smoother than MC's equivalent."""),
                unsafe_allow_html=True)

            st.markdown(_tip("""
            <b>Experiment:</b> Increase α to 0.9 — convergence is faster early but wobbles more
            (overshooting). Decrease α to 0.05 — very stable but very slow. The "just right" α
            is task-dependent; α=0.5 is a common starting point for tabular TD.
            """), unsafe_allow_html=True)

            # MC vs TD comparison
            st.divider()
            st.subheader("🆚 TD vs MC — The Key Difference")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Monte Carlo update (after full episode):**
                ```
                G = R₁ + γR₂ + γ²R₃ + ... + γᵀRₜ    ← real total
                V(S₀) ← V(S₀) + α[G - V(S₀)]
                ```
                Uses real outcome G — unbiased, but must wait.
                """)
            with col2:
                st.markdown("""
                **TD(0) update (after each step):**
                ```
                target = R_{t+1} + γ·V(S_{t+1})       ← estimate
                V(Sₜ) ← V(Sₜ) + α[target - V(Sₜ)]
                ```
                Uses estimated target — biased, but immediate.
                """)
            render_td_notes("TD(0) Prediction", "temporal_difference_learning_td0_prediction")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 2 — SARSA
        # ═══════════════════════════════════════════════════════════════════
        with tab_sarsa:
            st.markdown(_card("#e65100","🎯","What does SARSA solve?",
                """<b>The upgrade from prediction to control:</b> TD(0) only estimates V(s) under a
                fixed policy. SARSA learns Q(s,a) — action values — AND improves the policy at the
                same time. No separate evaluation and improvement phases: every single step does both.<br><br>
                <b>Why "SARSA"?</b> The name comes from the five things it uses in each update:
                <b>S</b>tate, <b>A</b>ction, <b>R</b>eward, <b>S</b>'(next state), <b>A</b>'(next action).<br><br>
                <b>On-policy = cautious near the cliff:</b> SARSA learns the value of the policy it's
                <em>actually following</em> (ε-greedy). Since ε-greedy occasionally stumbles off the cliff,
                SARSA learns that the cliff edge is dangerous — and prefers the safe path above it."""),
                unsafe_allow_html=True)

            st.subheader("🎯 SARSA — On-policy TD Control")

            with st.expander("📐 Theory & Formulas — SARSA", expanded=False):
                st.markdown(r"""
                #### The SARSA Update

                $$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr]}$$

                The five elements: $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ = **SARSA**

                **Crucial point:** $A_{t+1}$ is sampled from the **ε-greedy policy** — it's the actual
                next action the agent will take. This makes SARSA **on-policy**.

                **Symbol decoder:**
                - $Q(S_t, A_t)$ — current estimated value of taking action $A_t$ in state $S_t$
                - $\alpha$ — learning rate
                - $R_{t+1}$ — immediate reward after the transition
                - $\gamma Q(S_{t+1}, A_{t+1})$ — estimated value of the next state-action pair the agent WILL visit
                - $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$ — SARSA TD error

                #### The Full SARSA Algorithm (Episode Loop)

                ```
                Initialise Q(s,a) = 0 for all s, a
                For each episode:
                    S ← starting state
                    A ← ε-greedy(Q, S)          ← choose A BEFORE the loop
                    Loop until terminal:
                        S', R ← take action A    ← observe next state and reward
                        A' ← ε-greedy(Q, S')     ← choose NEXT action NOW
                        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]   ← update
                        S ← S'; A ← A'           ← advance
                ```

                Note: A' is chosen **before** the update, not after. This is critical — it ensures
                the update uses the action that will actually be taken.

                #### Why SARSA is On-policy

                SARSA evaluates and improves the **ε-greedy policy** it's following.
                The target $R + \gamma Q(S', A')$ uses $A'$ sampled from ε-greedy — not the greedy action.

                **On the cliff:** When the agent is on the top of the cliff edge (e.g. (2,5)):
                - With prob 1−ε: goes right (→ safe)
                - With prob ε/4: random action, might go down (→ cliff, −100)

                SARSA's target includes the expected cost of this ε/4 cliff risk.
                Over time: Q((2,5), →) gets penalised even when going right was correct,
                because A' sometimes = ↓ (cliff). The safe path (row 2) becomes preferred.

                #### Convergence

                SARSA converges to the **optimal ε-soft policy** $Q^*_\varepsilon$ (best policy given
                that exploration must continue). As ε→0, this approaches the true optimal Q*.
                Typically: decrease ε over training for best final performance.
                """)

            st.markdown(_card("#e65100","📖","How to read the SARSA diagrams",
                """<b>Left — Value heatmap (max Q):</b> Best expected return from each state after training.
                The safe path (row 2) should have significantly higher (less negative) values than
                row 3 cells near the cliff — SARSA learns the cliff is dangerous.<br>
                <b>Right — Policy arrows:</b> The greedy policy extracted from Q.
                <b>SARSA's signature:</b> Look at the bottom row near the start. SARSA should show
                arrows pointing UP or staying in row 2 — it avoids the cliff edge.<br>
                <b>Compare to Q-Learning</b> (next tab): Q-Learning's arrows should hug the cliff."""),
                unsafe_allow_html=True)

            Q_s = res["Q_sarsa"]
            V_s = {s: float(np.max(Q_s.get(s, np.zeros(4)))) for s in Q_s}
            fig, axes, axl = make_fig(1, 2, 16, 4)
            plot_cliff_value(env, V_s, "SARSA — Q-max V(s)", axes[0], vmin=-25, vmax=-5)
            plot_cliff_policy(env, Q_s, "SARSA — Greedy Policy (SAFE path)", axes[1],
                              color=METHOD_COLORS["SARSA"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.divider()
            st.subheader("📈 SARSA Learning Curve")
            st.markdown(_card("#e65100","📖","How to read the learning curve",
                """<b>X-axis:</b> Episode number (early training on the left, late on the right).<br>
                <b>Y-axis:</b> Total reward per episode. Under random policy: very negative (many cliff falls).
                As SARSA learns, returns should improve — but because SARSA learns the SAFE path
                (return ≈ −13), it won't reach the theoretical optimal (−11 + possible cliff risks).<br>
                <b>Key pattern:</b> SARSA's curve should be less noisy than Q-Learning's during training
                — because SARSA avoids the cliff, it has fewer catastrophic −100 episodes.
                Q-Learning has more noisy training (frequent cliff falls) but learns a better final policy."""),
                unsafe_allow_html=True)

            fig2, ax2, _ = make_fig(1, 1, 11, 4)
            plot_learning_curves(ax2, {
                "SARSA": (res["rew_sarsa"], METHOD_COLORS["SARSA"]),
            }, "SARSA Learning Curve")
            ax2.axhline(-13, color="#aaaaaa", ls="--", lw=1, alpha=0.6, label="Safe path (≈−13)")
            ax2.axhline(-11, color="#4caf50", ls="--", lw=1, alpha=0.6, label="Optimal path (≈−11)")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            st.markdown(_insight("""
            <b>The SARSA insight:</b> SARSA's learned policy is safer during training (fewer cliff falls)
            but suboptimal at test time (takes −13 path instead of −11 path). This is the fundamental
            on-policy trade-off: you learn to be good at what you're actually doing (ε-greedy with
            random exploration accidents), not at the idealized greedy policy.
            """), unsafe_allow_html=True)
            render_td_notes("SARSA", "temporal_difference_learning_sarsa")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 3 — Q-LEARNING
        # ═══════════════════════════════════════════════════════════════════
        with tab_ql:
            st.markdown(_card("#1565c0","🎲","What does Q-Learning solve?",
                """<b>The breakthrough:</b> Q-Learning (Watkins, 1989) was the first off-policy TD
                control algorithm with convergence guarantees. It learns the OPTIMAL policy π*
                regardless of what exploratory policy is being followed — by always using the
                <em>maximum</em> Q value for the next state in its update.<br><br>
                <b>Off-policy = optimal but risky during training:</b> Because Q-Learning's target
                uses max Q(S',·) (the greedy action), it learns as if it will always act greedily —
                ignoring the random exploration accidents. This means it discovers the cliff-edge
                path is optimal, but suffers more −100 penalties during training.<br><br>
                <b>After training (ε→0):</b> Q-Learning's policy is strictly better than SARSA's.
                With no more exploration, the cliff edge is perfectly safe."""),
                unsafe_allow_html=True)

            st.subheader("🎲 Q-Learning — Off-policy TD Control")

            with st.expander("📐 Theory & Formulas — Q-Learning", expanded=False):
                st.markdown(r"""
                #### The Q-Learning Update

                $$\boxed{Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\bigr]}$$

                The only difference from SARSA: $Q(S_{t+1}, A_{t+1})$ → $\max_a Q(S_{t+1}, a)$.

                **Symbol decoder:**
                - $\max_a Q(S_{t+1}, a)$ — the maximum Q value at the next state, over all possible actions
                - This is the **greedy** value — what the agent would get if it acted perfectly from S' onwards
                - The action $A_{t+1}$ that achieves this max is NOT necessarily what the agent will do next
                  (the agent still acts ε-greedy) — that's what makes this off-policy

                #### SARSA vs Q-Learning — The Single-Symbol Difference

                | | SARSA | Q-Learning |
                |-|-------|-----------|
                | **Update target** | $R + \gamma Q(S', \mathbf{A'})$ | $R + \gamma \mathbf{\max_a} Q(S', a)$ |
                | **A' from** | ε-greedy (what agent will actually do) | Greedy (ideal, not what agent does) |
                | **Policy type** | On-policy | Off-policy |
                | **Converges to** | Best ε-soft policy | **Optimal Q*** |

                #### Why Off-policy Means "Optimal but Noisy Training"

                Q-Learning's target imagines the agent acts greedily from S' onwards.
                But during training, the agent acts ε-greedy — sometimes falling off the cliff.

                **On the cliff edge (e.g. state (3,9)):**
                - Q-Learning target: $R + \gamma \max_a Q((3,10), a)$ ← assumes greedy next
                - The greedy action from (3,10) is → to goal — this is fine
                - So Q((3,9), →) gets updated positively — cliff edge path looks great
                - But 10% of the time, the agent actually goes ↓ (fall, −100) — those episodes are terrible

                Result: Q-Learning learns the cliff-edge path is optimal (it is! for a greedy agent),
                but training is noisy from cliff-fall exploration accidents.

                #### The Optimal Policy Theorem

                Under mild conditions (all state-action pairs visited infinitely often,
                learning rates satisfying Robbins-Monro), Q-Learning converges:

                $$Q(s,a) \xrightarrow{N \to \infty} q^*(s,a) \quad \forall s, a$$

                where $q^*$ is the **true optimal action-value function** — the best possible Q
                regardless of which policy is being followed during learning.

                #### Convergence in Practice

                Q-Learning's convergence requires α to decrease. With fixed α (your slider),
                it converges to a region near Q* but continues to fluctuate.
                For CliffWalking: typically converges to the cliff-edge path in 200-500 episodes with α=0.5.
                """)

            st.markdown(_card("#1565c0","📖","How to read the Q-Learning diagrams",
                """<b>Left — Value heatmap:</b> The learned Q-max values. The bottom row (cliff edge)
                should have high values (close to 0 = near goal) for Q-Learning — it learned that
                being on the cliff edge (if acting greedily) is actually good.<br>
                <b>Right — Policy arrows:</b> Q-Learning's signature: arrows in row 3 should point
                RIGHT (→) all the way along the cliff edge — the cliff-hugging optimal path.
                <b>Compare to SARSA</b> (previous tab): SARSA shows arrows going ABOVE the cliff."""),
                unsafe_allow_html=True)

            Q_q = res["Q_ql"]
            V_q = {s: float(np.max(Q_q.get(s, np.zeros(4)))) for s in Q_q}
            fig, axes, axl = make_fig(1, 2, 16, 4)
            plot_cliff_value(env, V_q, "Q-Learning — Q-max V(s)", axes[0], vmin=-25, vmax=-5)
            plot_cliff_policy(env, Q_q, "Q-Learning — Greedy Policy (OPTIMAL path)", axes[1],
                              color=METHOD_COLORS["Q-Learning"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # SARSA vs Q-Learning comparison
            st.divider()
            st.subheader("🥊 SARSA vs Q-Learning — The Central Comparison")
            st.markdown(_card("#1565c0","📖","How to read the dual learning curve",
                """Both agents train for the same number of episodes with identical hyperparameters.<br>
                <b>Q-Learning (blue):</b> Noisier during training — frequent −100 cliff falls as it
                explores near the cliff edge. But late-training performance is better (optimal path).<br>
                <b>SARSA (orange):</b> Smoother training — learns to avoid the cliff early.
                Late-training performance plateaus at the safe path (≈−13 per episode).<br>
                <b>The crossing point:</b> Early, SARSA often outperforms Q-Learning (fewer disasters).
                Later, Q-Learning catches up or surpasses it (better path).
                The episode number at which they cross depends on α and ε."""),
                unsafe_allow_html=True)

            fig2, ax2, _ = make_fig(1, 1, 12, 4)
            plot_learning_curves(ax2, {
                "SARSA":      (res["rew_sarsa"], METHOD_COLORS["SARSA"]),
                "Q-Learning": (res["rew_ql"],    METHOD_COLORS["Q-Learning"]),
            }, "SARSA vs Q-Learning — Learning Curves")
            ax2.axhline(-13, color="#aaaaaa", ls="--", lw=1, alpha=0.6, label="Safe path ≈−13")
            ax2.axhline(-11, color="#4caf50", ls="--", lw=1, alpha=0.6, label="Optimal ≈−11")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            c1, c2, c3, c4 = st.columns(4)
            sm_s = smooth(res["rew_sarsa"])
            sm_q = smooth(res["rew_ql"])
            c1.metric("SARSA — late mean reward", f"{float(np.mean(res['rew_sarsa'][-100:])):.1f}")
            c2.metric("Q-Learning — late mean reward", f"{float(np.mean(res['rew_ql'][-100:])):.1f}")
            c3.metric("SARSA — best episode", f"{float(max(res['rew_sarsa'])):.0f}")
            c4.metric("Q-Learning — best episode", f"{float(max(res['rew_ql'])):.0f}")
            render_td_notes("Q-Learning", "temporal_difference_learning_q_learning")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 4 — EXPECTED SARSA
        # ═══════════════════════════════════════════════════════════════════
        with tab_esarsa:
            st.markdown(_card("#00897b","🔄","What does Expected SARSA solve?",
                """<b>The variance problem:</b> SARSA's update uses Q(S', A') where A' is a single
                random sample from the ε-greedy policy. This introduces sampling noise — each episode
                the same state might produce a different A', giving different TD errors.<br><br>
                <b>Expected SARSA's fix:</b> Instead of sampling one next action, compute the
                <em>weighted average</em> Q value over ALL possible next actions, weighted by their
                probability under the ε-greedy policy. This eliminates the variance from sampling A'.<br><br>
                <b>The trade-off:</b> Lower variance → faster, more reliable convergence.
                Cost: slightly more computation per step (need to compute Σπ(a|S')Q(S',a)).
                Expected SARSA strictly generalises both SARSA (if sampling one action) and
                Q-Learning (if the policy is greedy — then max is the expected value)."""),
                unsafe_allow_html=True)

            st.subheader("🔄 Expected SARSA — Variance Reduction via Expectation")

            with st.expander("📐 Theory & Formulas — Expected SARSA", expanded=False):
                st.markdown("#### The Expected SARSA Update")
                st.latex(r"Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\right]")

                st.markdown(r"""
                **Symbol decoder:**
                - $\sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a)$ — **expected** Q over all actions, weighted by policy probabilities
                - $\pi(a \mid S_{t+1})$ — probability that policy $\pi$ takes action $a$ in state $S_{t+1}$

                #### Computing the Expected Value

                For an $\varepsilon$-greedy policy with 4 actions (as in CliffWalking):
                - Best action $a^* = \arg\max_a Q(S', a)$ gets probability $1 - \varepsilon + \varepsilon/4$
                - Each other action gets probability $\varepsilon/4$
                """)
                st.latex(r"\sum_a \pi(a\mid S') Q(S',a) = \left(1-\varepsilon+\frac{\varepsilon}{4}\right) Q(S',a^*) + \frac{\varepsilon}{4} \sum_{a \neq a^*} Q(S',a)")

                st.markdown("""
                **Worked example** ($\varepsilon=0.1$, 4 actions, state $S'$ has $Q=[-10,-8,-15,-12]$):
                - Best action: index 1 ($Q=-8$), prob $= 1-0.1+0.025 = 0.925$
                - Others: prob $= 0.025$ each
                """)
                st.latex(r"\mathbb{E}[Q(S',\cdot)] = 0.925(-8) + 0.025(-10) + 0.025(-15) + 0.025(-12)")
                st.latex(r"= -7.4 - 0.25 - 0.375 - 0.3 = -8.325")

                st.markdown("""
                vs. SARSA sampled $A'=1$: uses exactly $-8$ (no sampling noise).  
                vs. SARSA sampled $A'=2$: uses exactly $-15$ (very different from expected $-8.325$!).

                #### The Method Generalisation Triangle
                """)
                st.latex(r"\text{SARSA} \xrightarrow{\text{replace sample with expectation}} \text{Expected SARSA} \xrightarrow{\varepsilon \to 0} \text{Q-Learning}")

                st.markdown("""
                - SARSA: uses sampled $A'$ from $\varepsilon$-greedy
                - Expected SARSA: uses expectation under $\varepsilon$-greedy — eliminates sample variance from $A'$
                - Q-Learning: if policy is fully greedy, expectation = max = Q-Learning's target

                **Expected SARSA is strictly better than SARSA** in terms of variance.
                It is as performant as Q-Learning on-policy, and can be used off-policy too.

                #### Why Not Always Use Expected SARSA?

                For large action spaces (e.g. 1000 possible actions), computing the sum over all actions
                is expensive. SARSA's sampling trick is then preferred.
                For tabular CliffWalking (4 actions), Expected SARSA is strictly superior.
                """)

            Q_es = res["Q_esarsa"]
            V_es = {s: float(np.max(Q_es.get(s, np.zeros(4)))) for s in Q_es}

            st.markdown(_card("#00897b","📖","How to read the Expected SARSA charts",
                """<b>Left — Value heatmap:</b> Should look similar to SARSA but with smoother transitions
                between cells — lower variance in the update means more consistent estimates across runs.<br>
                <b>Right — Policy arrows:</b> Expected SARSA typically learns a path between the
                cliff-hugging optimal and the fully-safe SARSA path — it balances the expected value
                of the ε-greedy policy (which includes some cliff risk) with the greedy improvement.<br>
                <b>Learning curve:</b> Should converge faster than SARSA with less noise."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 16, 4)
            plot_cliff_value(env, V_es, "Expected SARSA — Q-max V(s)", axes[0], vmin=-25, vmax=-5)
            plot_cliff_policy(env, Q_es, "Expected SARSA — Greedy Policy", axes[1],
                              color=METHOD_COLORS["Expected SARSA"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            fig2, ax2, _ = make_fig(1, 1, 12, 4)
            plot_learning_curves(ax2, {
                "SARSA":          (res["rew_sarsa"],   METHOD_COLORS["SARSA"]),
                "Q-Learning":     (res["rew_ql"],      METHOD_COLORS["Q-Learning"]),
                "Expected SARSA": (res["rew_esarsa"],  METHOD_COLORS["Expected SARSA"]),
            }, "SARSA vs Q-Learning vs Expected SARSA")
            ax2.axhline(-13, color="#aaaaaa", ls="--", lw=1, alpha=0.5, label="Safe path ≈−13")
            ax2.axhline(-11, color="#4caf50", ls="--", lw=1, alpha=0.5, label="Optimal ≈−11")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            st.markdown(_tip("""
            <b>What to observe:</b> Expected SARSA's curve should be smoother than SARSA's
            (less sampling variance in the update) but similar in final performance.
            In CliffWalking, it often converges slightly faster than pure SARSA.
            """), unsafe_allow_html=True)
            render_td_notes("Expected SARSA", "temporal_difference_learning_expected_sarsa")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 5 — DOUBLE Q-LEARNING
        # ═══════════════════════════════════════════════════════════════════
        with tab_dql:
            st.markdown(_card("#ad1457","🎭","What does Double Q-Learning solve?",
                """<b>The maximisation bias problem:</b> Standard Q-Learning uses max_a Q(S',a) as its
                target. But the maximum of noisy estimates is systematically <em>higher</em> than the
                true maximum — because you're picking the noisiest overestimate. This is called
                <b>maximisation bias</b>, and it causes Q-Learning to systematically overestimate values.<br><br>
                <b>The effect in CliffWalking:</b> Q-Learning may overestimate the value of cliff-edge
                states early in training, leading to more cliff falls than expected even by Q-Learning's
                own logic.<br><br>
                <b>Double Q-Learning's fix:</b> Maintain TWO separate Q tables (QA and QB).
                Use QA to SELECT the best action, but QB to EVALUATE its value. Since the same noise
                doesn't affect both tables equally, the bias is eliminated."""),
                unsafe_allow_html=True)

            st.subheader("🎭 Double Q-Learning — Eliminating Maximisation Bias")

            with st.expander("📐 Theory & Formulas — Double Q-Learning", expanded=False):
                st.markdown(r"""
                #### The Maximisation Bias Problem

                Suppose the true Q*(s,a) = 0 for all actions, but our estimates are noisy:
                $Q(s,a) = 0 + \epsilon_a$ where $\epsilon_a \sim N(0, \sigma^2)$.

                The maximum estimate: $\max_a Q(s,a) \approx \sigma\sqrt{2\ln(|A|)} > 0$.

                With 4 actions and σ=1: max ≈ 1.18 — but true max = 0!

                **Q-Learning's target uses this inflated max**, so early Q values are overestimated.
                This delays convergence and can cause overly aggressive (cliff-hugging) behaviour.

                #### The Double Q-Learning Update

                With probability 0.5, choose Update A:
                $$Q_A(S_t, A_t) \leftarrow Q_A(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma Q_B(S_{t+1}, \arg\max_a Q_A(S_{t+1},a)) - Q_A(S_t, A_t)\bigr]$$

                Otherwise, Update B (symmetric — swap A and B):
                $$Q_B(S_t, A_t) \leftarrow Q_B(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma Q_A(S_{t+1}, \arg\max_a Q_B(S_{t+1},a)) - Q_B(S_t, A_t)\bigr]$$

                **Key insight — two roles are separated:**
                - $Q_A$ selects the best action: $a^* = \arg\max_a Q_A(S',a)$
                - $Q_B$ evaluates that action: $Q_B(S', a^*)$

                Since QA and QB are trained on different (random) halves of the experience,
                their errors are **independent** — the maximisation bias cancels.

                #### Action Selection

                During training, actions are chosen from the **combined** (averaged) policy:
                $$\pi(s) = \varepsilon\text{-greedy}\left(\frac{Q_A(s) + Q_B(s)}{2}\right)$$

                The final policy also uses the average: $Q_{final} = (Q_A + Q_B)/2$.

                #### Convergence

                Double Q-Learning converges to $q^*$ under the same conditions as Q-Learning.
                Proof: $\mathbb{E}[Q_B(S', \arg\max_a Q_A(S',a))] \leq \max_a Q(S',a)$ for any
                two independent unbiased estimators — the double estimator is unbiased.
                """)

            Q_d = res["Q_dql"]
            V_d = {s: float(np.max(Q_d.get(s, np.zeros(4)))) for s in Q_d}

            st.markdown(_card("#ad1457","📖","How to read the Double Q-Learning charts",
                """<b>Value heatmap:</b> Should be similar to Q-Learning but with more moderate values —
                Double Q-Learning avoids the overestimation that makes Q-Learning's cliff cells look
                artificially valuable early in training.<br>
                <b>Policy arrows:</b> Should converge to the same cliff-edge optimal path as Q-Learning,
                but potentially faster and with fewer early cliff falls.<br>
                <b>Learning curve comparison:</b> Double Q-Learning should show a smoother, less
                catastrophically-dipping training curve than Q-Learning, especially in the first
                100-200 episodes when overestimation bias is strongest."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 16, 4)
            plot_cliff_value(env, V_d, "Double Q-Learning — Q-max V(s)", axes[0], vmin=-25, vmax=-5)
            plot_cliff_policy(env, Q_d, "Double Q-Learning — Greedy Policy", axes[1],
                              color=METHOD_COLORS["Double Q"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            fig2, ax2, _ = make_fig(1, 1, 12, 4)
            plot_learning_curves(ax2, {
                "Q-Learning": (res["rew_ql"],  METHOD_COLORS["Q-Learning"]),
                "Double Q":   (res["rew_dql"], METHOD_COLORS["Double Q"]),
            }, "Q-Learning vs Double Q-Learning")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            ax2.axhline(-11, color="#4caf50", ls="--", lw=1, alpha=0.5, label="Optimal ≈−11")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            st.markdown(_warn("""
            <b>In CliffWalking, maximisation bias is mild</b> — the environment is deterministic
            so value estimates converge quickly. Double Q-Learning's benefit is most visible in
            stochastic environments where reward variance is high (e.g. slot machines, noisy robotics).
            If the curves look similar here, that's expected — try environments with high reward variance
            to see the full benefit.
            """), unsafe_allow_html=True)
            render_td_notes("Double Q-Learning", "temporal_difference_learning_double_q_learning")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 6 — N-STEP TD
        # ═══════════════════════════════════════════════════════════════════
        with tab_nstep:
            st.markdown(_card("#f57f17","🪜","What does n-step TD solve?",
                """<b>The bootstrapping spectrum:</b> TD(0) uses 1 real reward then bootstraps.
                Monte Carlo uses ALL real rewards (no bootstrapping). What about using 2 real rewards?
                Or 5? Or 10? <b>n-step TD</b> is this generalisation — use n real rewards,
                then bootstrap with Q(S_{t+n}, A_{t+n}).<br><br>
                <b>Why this matters:</b> The optimal n depends on the environment.
                For long episodes with informative rewards: larger n is better (more real signal).
                For short episodes with sparse rewards: smaller n avoids high variance.<br><br>
                <b>n-step is the bridge:</b> n=1 → SARSA. n=T (episode length) → Monte Carlo.
                This framework unifies all bootstrapping methods under one formula."""),
                unsafe_allow_html=True)

            st.subheader(f"🪜 n-step TD SARSA (n={n_step})")

            with st.expander("📐 Theory & Formulas — n-step TD", expanded=False):
                st.markdown(r"""
                #### The n-step Return

                Instead of one real reward (TD) or all rewards (MC), use exactly $n$:
                """)
                st.latex(r"G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n})")

                st.markdown(r"""
                **Symbol decoder:**
                - $G_{t:t+n}$ — the n-step return starting at time $t$ (read "G from $t$ to $t+n$")
                - $R_{t+k}$ — real reward received $k$ steps after time $t$
                - $\gamma^n Q(S_{t+n}, A_{t+n})$ — bootstrap value after $n$ real steps
                - The sum of real rewards has $n$ terms; the bootstrap has 1

                #### The n-step SARSA Update
                """)
                st.latex(r"Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[G_{t:t+n} - Q(S_t, A_t)\right]")

                st.markdown(r"""
                where $G_{t:t+n}$ uses the next $n$ real rewards plus one bootstrap.

                **Limits:**
                - $n=1$: $G_{t:t+1} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ $\rightarrow$ SARSA
                - $n=T-t$: $G_{t:T} = R_{t+1} + \cdots + \gamma^{T-t}R_T$ $\rightarrow$ Monte Carlo return
                - $n=2$: $G_{t:t+2} = R_{t+1} + \gamma R_{t+2} + \gamma^2 Q(S_{t+2}, A_{t+2})$

                #### The Bias-Variance Trade-off with n

                | n | Real rewards | Bootstrap | Bias | Variance |
                |---|---|---|---|---|
                | 1 (SARSA) | 1 step | from step 2 | HIGH (bootstrap dominates) | LOW |
                | 4 | 4 steps | from step 5 | Medium | Medium |
                | T (MC) | All steps | None | **ZERO** | HIGH |

                **Intuition:** More real rewards = less bias (no guessing), but more variance
                (actual rewards are noisy). The optimal $n$ minimises mean squared error = bias^2 + variance.

                #### Implementation Note — The Lag

                n-step TD has a processing lag: you must wait until step $t+n$ before you can
                compute the update for step $t$ (you need all $n$ rewards). This means:
                - Episode of length $T$: updates $t=0,\\ldots,T-n$ during episode
                - Updates $t=T-n+1,\\ldots,T-1$ at episode end (using terminal values)
                """)

            Q_n = res["Q_nstep"]
            V_n = {s: float(np.max(Q_n.get(s, np.zeros(4)))) for s in Q_n}

            st.markdown(_card("#f57f17","📖","How to read the n-step TD charts",
                f"""<b>Value heatmap:</b> With n={n_step}, the method sees {n_step} real rewards before bootstrapping.
                Higher n → values converge more from real experience, less from bootstrap estimates.
                Compare to SARSA (n=1): the value estimates should be smoother with higher n (less bias)
                but might vary more across runs (more variance).<br>
                <b>Policy arrows:</b> Should resemble SARSA's safe path (on-policy method) but
                with faster convergence for well-chosen n.<br>
                <b>Learning curve:</b> The curve shows how n trades off stability vs speed.
                Use the n slider (sidebar) and re-run to see this directly."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 16, 4)
            plot_cliff_value(env, V_n, f"n-step TD (n={n_step}) — Q-max V(s)", axes[0], vmin=-25, vmax=-5)
            plot_cliff_policy(env, Q_n, f"n-step TD (n={n_step}) — Greedy Policy", axes[1],
                              color=METHOD_COLORS["n-step TD"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            fig2, ax2, _ = make_fig(1, 1, 12, 4)
            plot_learning_curves(ax2, {
                "SARSA (n=1)": (res["rew_sarsa"],  METHOD_COLORS["SARSA"]),
                f"n-step (n={n_step})": (res["rew_nstep"], METHOD_COLORS["n-step TD"]),
            }, f"SARSA vs n-step TD (n={n_step})")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            st.markdown(_tip(f"""
            <b>Try different n values:</b> Set n=1 (sidebar) → identical to SARSA.
            Set n=20 → almost Monte Carlo (CliffWalking episodes are ~12 steps, so n=20 effectively is MC).
            The best n for CliffWalking is typically 3-5: enough real rewards to avoid bias,
            small enough to avoid high variance. Currently using n={n_step}.
            """), unsafe_allow_html=True)
            render_td_notes("n-step TD", "temporal_difference_learning_n_step_td")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 7 — TD(λ) / SARSA(λ)
        # ═══════════════════════════════════════════════════════════════════
        with tab_lam:
            st.markdown(_card("#558b2f","🌊","What does TD(λ) solve?",
                """<b>The n-step problem:</b> n-step TD picks ONE value of n. But why not use ALL
                values of n simultaneously, with earlier steps getting more weight?<br><br>
                <b>TD(λ) / Eligibility Traces:</b> Instead of choosing n, maintain a trace E(s,a) for
                every state-action pair. The trace accumulates when (s,a) is visited and decays by
                λγ each step. Each TD update then applies to ALL state-action pairs, weighted by
                their trace — recently-visited pairs get updated most.<br><br>
                <b>λ controls the time-horizon:</b> λ=0 → SARSA (only current step).
                λ=1 → Monte Carlo (traces never decay). λ∈(0,1) → geometric blend of all n-steps.
                In practice, λ=0.8-0.95 often outperforms any fixed n."""),
                unsafe_allow_html=True)

            st.subheader(f"🌊 SARSA(λ) — Eligibility Traces (λ={lam:.2f})")

            with st.expander("📐 Theory & Formulas — TD(λ) and Eligibility Traces", expanded=False):
                st.markdown(r"""
                #### The λ-Return — Combining All n-step Returns

                Instead of one n, take a weighted sum of ALL n-step returns:

                $$\boxed{G_t^\lambda \doteq (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}}$$

                **Symbol decoder:**
                - $(1-\lambda)$ — normalisation factor (weights sum to 1)
                - $\lambda^{n-1}$ — weight for the n-step return: n=1 gets weight $(1-\lambda)$, n=2 gets $(1-\lambda)\lambda$, etc.
                - $G_{t:t+n}$ — the n-step return

                **Limits:**
                - $\lambda=0$: weight all on n=1 → $G_t^0 = G_{t:t+1}$ → TD(0)/SARSA
                - $\lambda=1$: uniform weights → $G_t^1 = G_t$ (full MC return)

                #### Eligibility Traces — The Efficient Implementation

                Instead of storing all returns explicitly, maintain a trace $E_t(s,a)$:

                $$\boxed{E_t(s,a) = \begin{cases} \gamma\lambda E_{t-1}(s,a) + 1 & \text{if } s=S_t, a=A_t \\ \gamma\lambda E_{t-1}(s,a) & \text{otherwise}\end{cases}}$$

                Then update ALL Q values proportionally to their trace:

                $$Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E_t(s,a) \quad \forall s, a$$

                where $\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$ is the standard SARSA TD error.

                **What the trace E(s,a) represents:**
                The trace is a "memory" of how recently and how often (s,a) was visited.
                High trace = visited recently/frequently = gets a large update.
                Low trace = not visited recently = gets a tiny update.

                **Why this is equivalent to the λ-return:**
                The eligibility trace propagates the TD error **backward in time** through recent states.
                A state visited k steps ago gets its Q updated by $(\gamma\lambda)^k$ of the current TD error.
                This is mathematically identical to the weighted sum of n-step returns.

                #### Trace Variants

                | Type | Update when $S_t=s, A_t=a$ | Property |
                |------|---------------------------|---------|
                | Accumulating | $E \leftarrow \gamma\lambda E + 1$ | Counts visits; can exceed 1 |
                | **Replacing** | $E \leftarrow 1$ | Caps at 1; more stable (used here) |
                | Dutch | $E \leftarrow (1-\alpha)\gamma\lambda E + 1$ | Best theoretical properties |

                #### The λ-Spectrum

                ```
                λ=0     λ=0.5           λ=0.9     λ=1.0
                │         │               │          │
                TD(0)   medium          long-term   MC
                SARSA   horizon         credit      return
                (fast,  (good balance)  assignment  (high var,
                high bias)             (most useful) zero bias)
                ```

                Most practical RL uses λ ∈ [0.7, 0.95] — far enough from 0 to assign credit
                over several steps, close enough to 1 to avoid excessive variance.
                """)

            Q_l = res["Q_lam"]
            V_l = {s: float(np.max(Q_l.get(s, np.zeros(4)))) for s in Q_l}

            st.markdown(_card("#558b2f","📖","How to read the SARSA(λ) charts",
                f"""<b>Value heatmap:</b> With λ={lam:.2f}, the traces propagate each TD error backward
                through the last ~1/(1−λ·γ) ≈ {1/(1-lam*gamma):.0f} steps. Values should converge
                faster than SARSA (n=1) because credit is assigned to more states per step.<br>
                <b>Policy arrows:</b> SARSA(λ) is on-policy so it should learn the safe path like SARSA,
                but possibly with cleaner, more consistent arrows (faster convergence).<br>
                <b>Learning curve:</b> Should rise faster than SARSA early on — the trace spreads
                useful TD errors backward to previously-visited states automatically."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 16, 4)
            plot_cliff_value(env, V_l, f"SARSA(λ={lam:.2f}) — Q-max V(s)", axes[0], vmin=-25, vmax=-5)
            plot_cliff_policy(env, Q_l, f"SARSA(λ={lam:.2f}) — Greedy Policy", axes[1],
                              color=METHOD_COLORS["TD(λ)"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            fig2, ax2, _ = make_fig(1, 1, 12, 4)
            plot_learning_curves(ax2, {
                "SARSA (λ=0)":        (res["rew_sarsa"], METHOD_COLORS["SARSA"]),
                f"SARSA(λ={lam:.2f})": (res["rew_lam"],   METHOD_COLORS["TD(λ)"]),
            }, f"SARSA vs SARSA(λ={lam:.2f})")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            st.markdown(_tip(f"""
            <b>Experiment:</b> Try λ=0 (→ identical to SARSA), λ=0.5, λ=0.9, λ=1.0.
            For CliffWalking, λ≈0.8-0.9 often converges fastest.
            Very high λ (>0.95) can be unstable because old state traces don't decay enough.
            Currently λ={lam:.2f} — the effective lookback window is ~{1/(1-lam*gamma):.0f} steps.
            """), unsafe_allow_html=True)
            render_td_notes("TD(lambda) / SARSA(lambda)", "temporal_difference_learning_td_lambda_sarsa_lambda")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 8 — DASHBOARD
        # ═══════════════════════════════════════════════════════════════════
        with tab_dash:
            st.markdown(_card("#90a4ae","📈","What does this dashboard show?",
                """All 7 TD methods on the same CliffWalking environment with identical hyperparameters.
                The dashboard reveals: which path does each method learn? How fast does each converge?
                How noisy is the training? Which method would YOU choose for a real application?"""),
                unsafe_allow_html=True)
            st.subheader("📈 Full TD Methods Comparison Dashboard")

            # Policy comparison grid
            st.markdown("### 🗺️ Learned Policies — All Methods")
            st.markdown(_card("#90a4ae","📖","How to read the policy grid",
                """6 subplots — one policy map per method (TD(0) predicts V not Q, so excluded here).
                <b>Orange arrows (SARSA, SARSA(λ)):</b> Should show the safe path — row 2 all the way right.
                <b>Blue arrows (Q-Learning, Expected SARSA, Double Q):</b> Should show the optimal path —
                row 3 cliff edge (risky during training, optimal when greedy).
                <b>n-step TD:</b> Path depends on n; small n → SARSA-like (safe), large n → MC-like.
                <b>Key thing to look for:</b> Do the on-policy methods (orange) take the top road?
                Do the off-policy methods (blue) take the bottom road?"""),
                unsafe_allow_html=True)

            qs = [
                (res["Q_sarsa"],  "SARSA",           METHOD_COLORS["SARSA"]),
                (res["Q_ql"],     "Q-Learning",      METHOD_COLORS["Q-Learning"]),
                (res["Q_esarsa"], "Expected SARSA",  METHOD_COLORS["Expected SARSA"]),
                (res["Q_dql"],    "Double Q-Learning",METHOD_COLORS["Double Q"]),
                (res["Q_nstep"],  f"n-step TD (n={n_step})", METHOD_COLORS["n-step TD"]),
                (res["Q_lam"],    f"SARSA(λ={lam:.2f})", METHOD_COLORS["TD(λ)"]),
            ]
            fig, axes, axl = make_fig(2, 3, 20, 9)
            for idx, (Q, name, col) in enumerate(qs):
                ax = axes[idx//3][idx%3]
                plot_cliff_policy(env, Q, name, ax, color=col)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Learning curves all methods
            st.divider()
            st.markdown("### 📈 Learning Curves — All Methods")
            st.markdown(_card("#90a4ae","📖","How to read the combined learning curve",
                """All methods start from zero knowledge. <b>Higher = better reward = faster/safer path.</b><br>
                <b>On-policy methods (SARSA, SARSA(λ)):</b> Rise smoothly but plateau around −13 (safe path).<br>
                <b>Off-policy methods (Q-Learning, Expected SARSA, Double Q):</b> Initially noisier (cliff falls)
                but potentially reach higher plateau (≈−11, optimal path) after enough training.<br>
                <b>n-step TD:</b> Curve shape depends on n — try different values via the sidebar slider.<br>
                <b>The horizontal dashed lines</b> mark the theoretical safe (−13) and optimal (−11) benchmarks."""),
                unsafe_allow_html=True)

            fig2, ax2, _ = make_fig(1, 1, 14, 5)
            all_curves = {
                "SARSA":            (res["rew_sarsa"],   METHOD_COLORS["SARSA"]),
                "Q-Learning":       (res["rew_ql"],      METHOD_COLORS["Q-Learning"]),
                "Expected SARSA":   (res["rew_esarsa"],  METHOD_COLORS["Expected SARSA"]),
                "Double Q":         (res["rew_dql"],     METHOD_COLORS["Double Q"]),
                f"n-step (n={n_step})": (res["rew_nstep"],  METHOD_COLORS["n-step TD"]),
                f"SARSA(λ={lam:.2f})": (res["rew_lam"],   METHOD_COLORS["TD(λ)"]),
            }
            plot_learning_curves(ax2, all_curves, "All TD Methods — Episode Reward")
            ax2.axhline(-13, color="#aaaaaa", ls="--", lw=1, alpha=0.5, label="Safe path ≈−13")
            ax2.axhline(-11, color="#4caf50", ls="--", lw=1, alpha=0.5, label="Optimal ≈−11")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8,
                       bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            # Summary table
            st.divider()
            st.markdown("### 📋 Performance Summary Table")
            st.markdown(_card("#90a4ae","📖","How to read the table",
                """<b>Late mean reward:</b> Average reward over the last 20% of episodes — measures final policy quality.<br>
                <b>Best episode:</b> Best single episode reward — shows peak capability.<br>
                <b>Training stability (std):</b> Standard deviation of episode rewards over last 50% of training.
                Lower = more consistent, less noisy training.<br>
                <b>Theoretical path:</b> What path theory predicts the method should learn."""),
                unsafe_allow_html=True)

            rows = []
            expected_paths = {
                "SARSA":         "Safe (≈−13)",
                "Q-Learning":    "Optimal (≈−11)",
                "Expected SARSA":"Optimal (≈−11)",
                "Double Q":      "Optimal (≈−11)",
                f"n-step (n={n_step})": "Depends on n",
                f"SARSA(λ={lam:.2f})": "Safe (≈−13)",
            }
            for name, (rewards, _) in all_curves.items():
                r = np.array(rewards)
                n20 = max(1, len(r)//5)
                rows.append({
                    "Method": name,
                    "Late mean reward ↑": f"{np.mean(r[-n20:]):.1f}",
                    "Best episode ↑":     f"{np.max(r):.0f}",
                    "Stability (σ↓)":     f"{np.std(r[len(r)//2:]):.1f}",
                    "Expected path":      expected_paths.get(name, "—"),
                    "Policy type":        "On-policy" if "SARSA" in name else "Off-policy",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Bias-variance conceptual chart
            st.divider()
            st.markdown("### 🎯 TD Method Landscape — Bootstrapping vs Real Rewards")
            st.markdown(_card("#90a4ae","📖","How to read the method landscape chart",
                """<b>X-axis:</b> How many REAL reward steps are used before bootstrapping.
                Left = more bootstrapping (faster, more bias). Right = more real rewards (slower, less bias).<br>
                <b>Y-axis:</b> Qualitative variance of the method's learning updates.<br>
                <b>What to look for:</b> TD(0) sits at n=1 (leftmost, most bootstrapping).
                Monte Carlo sits at n=∞ (rightmost, no bootstrapping). All n-step methods and TD(λ)
                sit in between. The "sweet spot" is where bias and variance balance — usually λ≈0.8 or n≈4."""),
                unsafe_allow_html=True)

            fig3, ax3, _ = make_fig(1, 1, 12, 5)
            methods_pos = [
                ("TD(0)\nSARSA\nQ-Learning", 1,  7, METHOD_COLORS["SARSA"]),
                ("Exp.SARSA", 1, 5, METHOD_COLORS["Expected SARSA"]),
                ("Double Q",  1, 6, METHOD_COLORS["Double Q"]),
                (f"n-step\n(n={n_step})", n_step, 5, METHOD_COLORS["n-step TD"]),
                ("SARSA(λ)\nTD(λ)", 8, 4, METHOD_COLORS["TD(λ)"]),
                ("Monte Carlo\n(n=∞)", 20, 9, METHOD_COLORS["MC"]),
            ]
            for name, x, y, col in methods_pos:
                ax3.scatter(x, y, s=300, color=col, zorder=5, edgecolors="white", lw=1.5)
                ax3.annotate(name, (x, y), xytext=(8, 4), textcoords="offset points",
                             color="white", fontsize=8, ha="left")
            ax3.fill_betweenx([3, 6], 3, 12, alpha=0.07, color="#4caf50")
            ax3.text(7.5, 3.3, "✓ Practical sweet spot\n(low bias, manageable variance)", color="#81c784",
                     fontsize=8, ha="center")
            ax3.set_xlim(0, 22); ax3.set_ylim(0, 10)
            ax3.set_xlabel("Real reward steps before bootstrapping  →", color="white", fontsize=11)
            ax3.set_ylabel("Relative update variance  →", color="white", fontsize=11)
            ax3.set_title("TD Method Landscape: Bootstrapping Spectrum", color="white", fontweight="bold")
            ax3.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig3); plt.close()
            render_td_notes("Dashboard", "temporal_difference_learning_dashboard")

    else:
        pending_tabs = [
            (tab_td0, "TD(0) Prediction", "temporal_difference_learning_td0_prediction"),
            (tab_sarsa, "SARSA", "temporal_difference_learning_sarsa"),
            (tab_ql, "Q-Learning", "temporal_difference_learning_q_learning"),
            (tab_esarsa, "Expected SARSA", "temporal_difference_learning_expected_sarsa"),
            (tab_dql, "Double Q-Learning", "temporal_difference_learning_double_q_learning"),
            (tab_nstep, "n-step TD", "temporal_difference_learning_n_step_td"),
            (tab_lam, "TD(lambda) / SARSA(lambda)", "temporal_difference_learning_td_lambda_sarsa_lambda"),
            (tab_dash, "Dashboard", "temporal_difference_learning_dashboard"),
        ]
        for t, note_title, note_slug in pending_tabs:
            with t:
                st.info("👈 Click **Run All Methods** in the sidebar to generate all charts.")
                render_td_notes(note_title, note_slug)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 9 — METHOD GUIDE (always visible)
    # ═══════════════════════════════════════════════════════════════════════
    with tab_guide:
        st.subheader("📚 Complete TD Method Reference")

        with st.expander("🗺️ Which TD method should I use? — Decision Guide", expanded=True):
            st.markdown("""
            ```
            START HERE: Is the task episodic or continuing?
              │
              ├─ CONTINUING (no episode end) → TD methods only (MC can't apply)
              │
              └─ EITHER → Continue below
                   │
                   ├─ Do you need PREDICTION (evaluate a fixed policy)?
                   │    └─ TD(0): fast, simple, online — use this
                   │
                   └─ Do you need CONTROL (improve policy)?
                        │
                        ├─ ON-POLICY (learn from the policy you're following)
                        │    ├─ Simple, stable → SARSA
                        │    ├─ Lower variance → Expected SARSA
                        │    └─ Faster credit assignment → SARSA(λ) with λ ∈ [0.7, 0.95]
                        │
                        └─ OFF-POLICY (learn optimal policy while exploring)
                             ├─ Standard → Q-Learning (most common)
                             ├─ Reduce overestimation bias → Double Q-Learning
                             ├─ Multi-step reward → n-step TD (tune n for your task)
                             └─ All horizons blended → TD(λ) with off-policy corrections
            ```
            """)

        entries = [
            {
                "icon":"🔮", "name":"TD(0) Prediction", "color":"#7c4dff",
                "formula":r"$V(S) \leftarrow V(S) + \alpha[R + \gamma V(S') - V(S)]$",
                "what":"Estimates V(s) for a fixed policy by bootstrapping: use one real reward R then estimate the rest with the current V(S'). Updates after every step — no waiting for episode end.",
                "when":"When you need to evaluate a fixed policy (not improve it). As the foundation step before control. When episodes are very long (MC would be too slow).",
                "pros":"✅ Online (updates every step) | ✅ Works on continuing tasks | ✅ Lower variance than MC | ✅ Simple to implement",
                "cons":"❌ Biased (bootstrap from own estimates) | ❌ Does not directly optimise a policy | ❌ Convergence requires shrinking α",
                "relation":"Bridge between MC and full control. Add action-conditioning → SARSA/Q-Learning. Add n steps → n-step TD. Add traces → TD(λ).",
                "td_error":r"$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$",
            },
            {
                "icon":"🎯", "name":"SARSA (on-policy)", "color":"#e65100",
                "formula":r"$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma Q(S',A') - Q(S,A)]$",
                "what":"On-policy TD control. Updates Q(S,A) after every step using the 5-tuple (S,A,R,S',A'). A' is sampled from the ε-greedy policy — so SARSA learns the value of the policy it's actually following, including all its exploration accidents.",
                "when":"When safety during training matters (avoids risky paths due to exploration). When the ε-soft policy limitation is acceptable. Default choice for simple tabular control.",
                "pros":"✅ Stable training (cautious near dangerous states) | ✅ Simple update rule | ✅ Provably convergent | ✅ No maximisation bias",
                "cons":"❌ Converges to best ε-soft policy (not true optimal) | ❌ More conservative than needed | ❌ Must be on-policy",
                "relation":"On-policy analogue of Q-Learning. Replace Q(S',A') with max Q(S',·) → Q-Learning. Replace Q(S',A') with expected value → Expected SARSA. Add traces → SARSA(λ).",
                "td_error":r"$\delta_t = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)$",
            },
            {
                "icon":"🎲", "name":"Q-Learning (off-policy)", "color":"#1565c0",
                "formula":r"$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \max_a Q(S',a) - Q(S,A)]$",
                "what":"Off-policy TD control. Uses the greedy max Q(S',·) as target — regardless of what ε-greedy actually does next. Learns the OPTIMAL Q* directly. One of the most impactful algorithms in RL history (Watkins, 1989).",
                "when":"When you want the optimal policy and accept noisier training. When exploration and learning can be cleanly separated. Default choice for off-policy tabular control. The foundation of DQN.",
                "pros":"✅ Converges to true Q* (optimal) | ✅ Off-policy (can learn from any data) | ✅ Simple | ✅ Foundation of deep RL (DQN)",
                "cons":"❌ Maximisation bias (overestimates values) | ❌ Noisy training near cliffs/risks | ❌ Slightly more dangerous during training",
                "relation":"Off-policy version of SARSA. Replace max with E_π → Expected SARSA. Add two tables → Double Q-Learning. The seed of DQN, DDQN, Rainbow.",
                "td_error":r"$\delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)$",
            },
            {
                "icon":"🔄", "name":"Expected SARSA", "color":"#00897b",
                "formula":r"$Q(S,A) \leftarrow Q(S,A) + \alpha[R + \gamma \sum_a \pi(a|S')Q(S',a) - Q(S,A)]$",
                "what":"Like SARSA but uses the EXPECTED next Q under the policy instead of a sampled action. Eliminates the variance from sampling A'. Can be on-policy (ε-greedy expectation) or off-policy (greedy expectation = Q-Learning).",
                "when":"When you want SARSA's stability with lower variance. When the action space is small (expectation is cheap to compute). The strictly better version of SARSA in almost all tabular settings.",
                "pros":"✅ Lower variance than SARSA | ✅ At least as good as Q-Learning empirically | ✅ Unifies SARSA and Q-Learning | ✅ Can be on or off-policy",
                "cons":"❌ Slightly more expensive per step (sum over all actions) | ❌ Scales poorly to large action spaces",
                "relation":"Generalisation that contains both SARSA and Q-Learning as special cases. With greedy policy: becomes Q-Learning. With ε-greedy: on-policy Expected SARSA.",
                "td_error":r"$\delta_t = R_{t+1} + \gamma \mathbb{E}_\pi[Q(S_{t+1},·)] - Q(S_t,A_t)$",
            },
            {
                "icon":"🎭", "name":"Double Q-Learning", "color":"#ad1457",
                "formula":r"$Q_A(S,A) \leftarrow Q_A + \alpha[R + \gamma Q_B(S', \arg\max_a Q_A(S',a)) - Q_A(S,A)]$",
                "what":"Uses two Q tables (QA, QB) with randomised update assignment. QA selects the best action; QB evaluates it. This decoupling eliminates maximisation bias — the systematic overestimation of Q values caused by always taking the max of noisy estimates.",
                "when":"When Q-Learning is overestimating values (especially with stochastic rewards). When training in environments with high reward variance. The standard choice in modern deep RL (Double DQN).",
                "pros":"✅ Eliminates maximisation bias | ✅ Same convergence guarantee as Q-Learning | ✅ Only marginally more expensive | ✅ Standard in deep RL (Double DQN)",
                "cons":"❌ More memory (two Q tables) | ❌ Slightly slower convergence early on (two half-sized datasets) | ❌ Benefit is small in deterministic environments",
                "relation":"Improvement of Q-Learning. The deep RL version is Double DQN (replace tables with neural networks). Can be combined with Expected SARSA, n-step, traces.",
                "td_error":r"$\delta_A = R + \gamma Q_B(S', \arg\max_a Q_A(S',a)) - Q_A(S,A)$",
            },
            {
                "icon":"🪜", "name":"n-step TD SARSA", "color":"#f57f17",
                "formula":r"$G_{t:t+n} = \sum_{k=1}^n \gamma^{k-1}R_{t+k} + \gamma^n Q(S_{t+n},A_{t+n})$",
                "what":"Use n real rewards before bootstrapping. Bridges TD (n=1) and MC (n=∞). The optimal n trades off bias (more bootstrap = more bias) against variance (more real rewards = more variance).",
                "when":"When TD(0)/SARSA is too slow to propagate useful signals (sparse rewards). When MC is too high-variance. When you want to tune the bias-variance trade-off explicitly. Useful for debugging what n works best for your environment.",
                "pros":"✅ Unifies TD and MC under one formula | ✅ Often best performance for tuned n | ✅ Natural gradient from TD to MC",
                "cons":"❌ Must choose n (extra hyperparameter) | ❌ Lag: update for step t requires n future steps | ❌ Memory: must store n transitions",
                "relation":"Generalises SARSA (n=1) and MC (n=∞). TD(λ) is equivalent to a weighted sum of all n-step returns. n-step Q-Learning also exists (off-policy variant).",
                "td_error":r"$G_{t:t+n} - Q(S_t,A_t)$ where $G_{t:t+n}$ uses $n$ real rewards",
            },
            {
                "icon":"🌊", "name":"SARSA(λ) / TD(λ)", "color":"#558b2f",
                "formula":r"$Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E_t(s,a)$ for ALL $(s,a)$ each step",
                "what":"Uses eligibility traces E(s,a) to propagate TD errors backward in time. The trace decays by γλ each step. At each update, ALL state-action pairs are updated proportionally to their trace. Equivalent to a weighted blend of all n-step returns simultaneously.",
                "when":"When credit must be assigned over multiple steps efficiently. When SARSA converges too slowly. Often the best-performing tabular method when λ is tuned. λ∈[0.7,0.95] typically works well.",
                "pros":"✅ Fast credit assignment (all recent steps updated each step) | ✅ Often fastest convergence | ✅ Naturally handles long-term dependencies | ✅ λ=0→SARSA, λ=1→MC: full spectrum",
                "cons":"❌ Must store and update traces for all state-action pairs per step | ❌ Extra hyperparameter λ | ❌ Can be unstable with function approximation (use gradient-based traces instead)",
                "relation":"Generalises SARSA (λ=0) and approaches MC (λ=1). TD(λ) with V → prediction. SARSA(λ) → on-policy control. Q(λ) → off-policy. In deep RL: Retrace(λ), V-trace, GAE(λ) are the successors.",
                "td_error":r"$\delta_t = R_{t+1} + \gamma Q(S',A') - Q(S,A)$, then all $Q(s,a) += \alpha\delta_t E_t(s,a)$",
            },
        ]

        for e in entries:
            with st.expander(f"{e['icon']} {e['name']}", expanded=False):
                st.markdown(f"<span style='color:{e['color']}; font-weight:700; font-size:1.05rem'>"
                           f"Core formula: {e['formula']}</span>", unsafe_allow_html=True)
                st.markdown(f"**TD error:** {e['td_error']}")
                st.divider()
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
                    st.markdown(f"**🔗 Relation to other methods:**\n\n{e['relation']}")

        st.divider()
        st.markdown(r"""
        ### ❓ Frequently Asked Questions

        **Q: Why can TD methods learn from incomplete episodes?**

        TD uses bootstrapping — it estimates the rest of the return using V(S') or Q(S',·),
        which are already in memory. So after each step, you have R (real) and V(S') (estimate)
        — enough to compute a TD target. MC needs the TRUE future reward, which requires waiting
        for the episode to end.

        ---

        **Q: Is bootstrapping "cheating"? The estimate isn't the true value.**

        No — it's a deliberate bias-variance trade-off. The bootstrap target is biased (wrong in expectation)
        but has much lower variance than a Monte Carlo sample. With enough data, the bias decreases
        as the estimates converge toward true values. The bias disappears at convergence.

        ---

        **Q: Why does SARSA take the safe path and Q-Learning take the risky path?**

        SARSA's target includes $Q(S', A')$ where $A'$ is sampled from ε-greedy.
        Near the cliff, ε-greedy sometimes randomly chooses ↓ (→ cliff, −100).
        This gets incorporated into SARSA's Q estimates, making cliff-adjacent states look dangerous.

        Q-Learning's target uses $\max Q(S',·)$ — the best possible action from S'.
        This ignores the chance of random cliff-falls. Q-Learning learns: "if I act greedy, cliff edge is fine."

        ---

        **Q: When should I use n-step TD vs TD(λ)?**

        Both solve the same problem (bridging TD and MC), but differently:

        | | n-step TD | TD(λ) |
        |--|-----------|-------|
        | **Hyperparameter** | n (integer) | λ (continuous 0-1) |
        | **Updates per step** | One state (step t-n) | ALL visited states |
        | **Memory** | n transitions | All state-action traces |
        | **Speed** | Similar | Faster credit assignment |
        | **Best choice** | When n is interpretable | When fast convergence matters |

        ---

        **Q: What is the connection between TD(λ) and deep RL?**

        Eligibility traces in tabular RL become these in deep RL:
        - **GAE(λ)** (Generalized Advantage Estimation) in PPO/A3C — exact SARSA(λ) for policy gradients
        - **Retrace(λ)** — off-policy safe trace for experience replay
        - **V-trace** — traces for large distributed RL (IMPALA)
        - **TD(λ)** in value functions for Actor-Critic methods

        The λ parameter serves the same role in all: tuning the bias-variance trade-off by
        controlling how far back credit is assigned.
        """)
        render_td_notes("Method Guide", "temporal_difference_learning_method_guide")


if __name__ == "__main__":
    main()
