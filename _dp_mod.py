"""
DP RL Explorer — Streamlit App
Dynamic Programming methods for Reinforcement Learning.
Environment: 4×4 GridWorld (Sutton & Barto §4 — the canonical DP example)
Methods: Policy Evaluation, Policy Improvement, Policy Iteration,
         Value Iteration, Async DP, GPI
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
.stTabs [data-baseweb="tab-list"] {
    gap:6px; background:#12121f; border-radius:10px; padding:4px;
}
.stTabs [data-baseweb="tab"] {
    background:#1e1e2e; border-radius:8px; color:#b0b0cc;
    padding:7px 13px; font-weight:600; font-size:.87rem;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#6a1b9a,#0277bd);
    color:white !important;
}
div[data-testid="metric-container"] {
    background:#1e1e2e; border-radius:10px; padding:11px; border:1px solid #2d2d44;
}
</style>
""", unsafe_allow_html=True)


def render_dp_notes(tab_title: str, tab_slug: str) -> None:
    render_notes(f"Dynamic Programming - {tab_title}", tab_slug)

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d0d1a"
CARD_BG    = "#12121f"
GRID_COLOR = "#2a2a3e"
RL_CMAP    = LinearSegmentedColormap.from_list(
    "rl", ["#b71c1c","#f57f17","#fff176","#1b5e20"])

METHOD_COLORS = {
    "Policy Eval":  "#6a1b9a",
    "Pol Improve":  "#0277bd",
    "Pol Iter":     "#00695c",
    "Val Iter":     "#e65100",
    "Async DP":     "#ad1457",
    "GPI":          "#4527a0",
}

# ─────────────────────────────────────────────────────────────────────────────
# TEACHING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:1rem">'
            f'<b>{icon} {title}</b><br>{body}</div>')

def _tip(text):
    return (f'<div style="background:#1a2a1a;border-left:3px solid #4caf50;'
            f'padding:.65rem 1rem;border-radius:0 6px 6px 0;margin:.5rem 0;'
            f'font-size:.93rem">{text}</div>')

def _warn(text):
    return (f'<div style="background:#2a1a1a;border-left:3px solid #ef5350;'
            f'padding:.65rem 1rem;border-radius:0 6px 6px 0;margin:.5rem 0;'
            f'font-size:.93rem">{text}</div>')

def _insight(text):
    return (f'<div style="background:#1a1a2a;border-left:3px solid #7c4dff;'
            f'padding:.65rem 1rem;border-radius:0 6px 6px 0;margin:.5rem 0;'
            f'font-size:.93rem">{text}</div>')

# ─────────────────────────────────────────────────────────────────────────────
# 4×4 GRIDWORLD  (Sutton & Barto §4.1 — exact textbook example)
# ─────────────────────────────────────────────────────────────────────────────
class GridWorld4x4:
    """
    4×4 grid: 16 states numbered 0–15.
    Terminal states: 0 (top-left) and 15 (bottom-right).
    All transitions: R = −1, deterministic (wall → stay in place).
    Actions: 0=Up 1=Right 2=Down 3=Left
    Why this environment for DP?
      • DP requires a COMPLETE MODEL p(s',r|s,a) — we know it exactly here.
      • Small enough to see every value update; big enough to be interesting.
      • Optimal policy is visually obvious → easy to verify correctness.
      • The exact example from Sutton & Barto Fig 4.1 / 4.2.
    """
    SIZE      = 4
    N_STATES  = 16
    N_ACTIONS = 4
    ACTIONS   = [0, 1, 2, 3]
    SYMBOLS   = ["↑","→","↓","←"]
    DELTAS    = [(-1,0),(0,1),(1,0),(0,-1)]

    def __init__(self):
        self.terminals = {0, 15}

    def s2rc(self, s): return (s // self.SIZE, s % self.SIZE)
    def rc2s(self, r, c): return r * self.SIZE + c
    def is_terminal(self, s): return s in self.terminals

    def transitions(self, s, a):
        """Return [(prob, next_state, reward, done)] — full model knowledge."""
        if self.is_terminal(s):
            return [(1.0, s, 0.0, True)]
        r, c = self.s2rc(s)
        dr, dc = self.DELTAS[a]
        nr = max(0, min(self.SIZE-1, r+dr))
        nc = max(0, min(self.SIZE-1, c+dc))
        ns = self.rc2s(nr, nc)
        done = self.is_terminal(ns)
        return [(1.0, ns, -1.0, done)]

    def uniform_policy(self):
        """π(a|s) = 0.25 for all a — the classic random policy."""
        return np.full((self.N_STATES, self.N_ACTIONS), 0.25)

    def greedy_policy(self, V, gamma):
        """Deterministic greedy policy w.r.t. V."""
        pi = np.zeros((self.N_STATES, self.N_ACTIONS))
        for s in range(self.N_STATES):
            if self.is_terminal(s):
                pi[s, 0] = 1.0
                continue
            q = np.zeros(self.N_ACTIONS)
            for a in range(self.N_ACTIONS):
                for prob, ns, r, _ in self.transitions(s, a):
                    q[a] += prob * (r + gamma * V[ns])
            best = np.argmax(q)
            pi[s, best] = 1.0
        return pi

    def eps_greedy_policy(self, V, gamma, eps):
        pi = np.zeros((self.N_STATES, self.N_ACTIONS))
        for s in range(self.N_STATES):
            if self.is_terminal(s):
                pi[s, 0] = 1.0; continue
            q = np.zeros(self.N_ACTIONS)
            for a in range(self.N_ACTIONS):
                for prob, ns, r, _ in self.transitions(s, a):
                    q[a] += prob * (r + gamma * V[ns])
            best = np.argmax(q)
            pi[s] = eps / self.N_ACTIONS
            pi[s, best] += 1.0 - eps
        return pi


# ─────────────────────────────────────────────────────────────────────────────
# DP ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

def policy_evaluation(env, pi, gamma, theta=1e-6, max_iter=1000):
    """
    Iterative Policy Evaluation (§4.1).
    Repeatedly apply the Bellman expectation equation until convergence.
    V_{k+1}(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV_k(s')]
    Returns V and a list of (iteration, delta, V_snapshot) for animation.
    """
    V       = np.zeros(env.N_STATES)
    history = []           # [(sweep, max_delta, V_copy)]

    for sweep in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in range(env.N_STATES):
            if env.is_terminal(s):
                continue
            v_old = V[s]
            v_new = 0.0
            for a in range(env.N_ACTIONS):
                for prob, ns, r, _ in env.transitions(s, a):
                    v_new += pi[s, a] * prob * (r + gamma * V[ns])
            V_new[s] = v_new
            delta = max(delta, abs(v_old - v_new))
        V = V_new
        history.append((sweep + 1, delta, V.copy()))
        if delta < theta:
            break

    return V, history


def policy_improvement(env, V, gamma):
    """
    Policy Improvement (§4.2).
    Given V_π, produce π' that is greedy w.r.t. V_π.
    Returns (new_policy, is_stable, q_values).
    q_values[s,a] = Σ p(s',r|s,a)[r + γV(s')]
    """
    pi_new = np.zeros((env.N_STATES, env.N_ACTIONS))
    Q      = np.zeros((env.N_STATES, env.N_ACTIONS))
    for s in range(env.N_STATES):
        if env.is_terminal(s):
            pi_new[s, 0] = 1.0
            continue
        for a in range(env.N_ACTIONS):
            for prob, ns, r, _ in env.transitions(s, a):
                Q[s, a] += prob * (r + gamma * V[ns])
        best = np.argmax(Q[s])
        pi_new[s, best] = 1.0
    return pi_new, Q


def policy_iteration(env, gamma, theta=1e-6):
    """
    Policy Iteration (§4.3).
    Alternate between full policy evaluation and greedy improvement
    until the policy no longer changes.
    Returns list of (iteration, policy, V, stable).
    """
    pi      = env.uniform_policy()
    history = []
    V       = np.zeros(env.N_STATES)

    for iteration in range(50):
        # Evaluation
        V, eval_hist = policy_evaluation(env, pi, gamma, theta)
        # Improvement
        pi_new, Q = policy_improvement(env, V, gamma)
        stable = np.allclose(pi, pi_new)
        history.append({
            "iter": iteration + 1,
            "V":    V.copy(),
            "pi":   pi_new.copy(),
            "Q":    Q.copy(),
            "stable": stable,
            "eval_sweeps": len(eval_hist),
        })
        pi = pi_new
        if stable:
            break

    return history


def value_iteration(env, gamma, theta=1e-6, max_iter=500):
    """
    Value Iteration (§4.4).
    Combine one sweep of policy evaluation (using max instead of sum)
    with implicit policy improvement in a single update.
    V_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γV_k(s')]
    Returns (V, pi, history).
    """
    V       = np.zeros(env.N_STATES)
    history = []

    for sweep in range(max_iter):
        delta = 0.0
        V_new = V.copy()
        for s in range(env.N_STATES):
            if env.is_terminal(s):
                continue
            q = np.zeros(env.N_ACTIONS)
            for a in range(env.N_ACTIONS):
                for prob, ns, r, _ in env.transitions(s, a):
                    q[a] += prob * (r + gamma * V[ns])
            v_best = np.max(q)
            delta  = max(delta, abs(V[s] - v_best))
            V_new[s] = v_best
        V = V_new
        history.append((sweep + 1, delta, V.copy()))
        if delta < theta:
            break

    pi, Q = policy_improvement(env, V, gamma)
    return V, pi, Q, history


def async_value_iteration(env, gamma, order="inplace", theta=1e-6, max_sweeps=500):
    """
    Asynchronous / In-place DP (§4.5).
    'inplace': update V[s] immediately and use updated values for later states
               in the same sweep.
    'prioritised': always update the state with the largest Bellman error first.
    Returns (V, history_of_deltas_per_state).
    """
    V = np.zeros(env.N_STATES)
    history = []

    if order == "inplace":
        for sweep in range(max_sweeps):
            delta = 0.0
            for s in range(env.N_STATES):   # sequential in-place
                if env.is_terminal(s): continue
                v_old = V[s]
                q = np.array([sum(prob*(r+gamma*V[ns])
                              for prob,ns,r,_ in env.transitions(s,a))
                              for a in range(env.N_ACTIONS)])
                V[s] = np.max(q)
                delta = max(delta, abs(v_old - V[s]))
            history.append((sweep+1, delta, V.copy()))
            if delta < theta: break

    elif order == "prioritised":
        # Prioritised sweeping: update state with largest |Bellman error|
        errors = np.zeros(env.N_STATES)
        update_counts = np.zeros(env.N_STATES, dtype=int)
        for step in range(max_sweeps * env.N_STATES):
            # Compute errors for all non-terminal states
            for s in range(env.N_STATES):
                if env.is_terminal(s): continue
                q = np.array([sum(prob*(r+gamma*V[ns])
                              for prob,ns,r,_ in env.transitions(s,a))
                              for a in range(env.N_ACTIONS)])
                errors[s] = abs(V[s] - np.max(q))
            s_best = int(np.argmax(errors))
            if errors[s_best] < theta: break
            q = np.array([sum(prob*(r+gamma*V[ns])
                          for prob,ns,r,_ in env.transitions(s_best,a))
                          for a in range(env.N_ACTIONS)])
            V[s_best] = np.max(q)
            update_counts[s_best] += 1
            history.append((step+1, errors[s_best], V.copy()))

    return V, history


def gpi_trace(env, gamma, theta=1e-6, n_policy_iter=6):
    """
    GPI visualisation: track V and π across multiple partial evaluation rounds,
    showing how evaluation and improvement interleave to produce convergence.
    """
    pi     = env.uniform_policy()
    V      = np.zeros(env.N_STATES)
    trace  = []

    for outer in range(n_policy_iter):
        # Partial evaluation (just 5 sweeps to show incompleteness)
        for _ in range(5):
            V_new = V.copy()
            for s in range(env.N_STATES):
                if env.is_terminal(s): continue
                v = 0.0
                for a in range(env.N_ACTIONS):
                    for prob, ns, r, _ in env.transitions(s, a):
                        v += pi[s,a] * prob * (r + gamma * V[ns])
                V_new[s] = v
            V = V_new
        # Improvement
        pi_new, Q = policy_improvement(env, V, gamma)
        stable = np.allclose(pi, pi_new)
        trace.append({"outer": outer+1, "V": V.copy(), "pi": pi_new.copy(), "stable": stable})
        pi = pi_new
        if stable: break

    return trace


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


def plot_V(env, V, title, ax, vmin=None, vmax=None, show_numbers=True):
    """Heatmap of state values on the 4×4 grid."""
    grid = V.reshape(env.SIZE, env.SIZE).copy()
    vmin = vmin if vmin is not None else grid.min() - 0.5
    vmax = vmax if vmax is not None else 0.5

    im = ax.imshow(grid, cmap=RL_CMAP, vmin=vmin, vmax=vmax, aspect="equal")

    for s in range(env.N_STATES):
        r, c = env.s2rc(s)
        val  = V[s]
        col  = "white" if val < (vmin + vmax) / 2 else "black"
        if env.is_terminal(s):
            ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,color="#1b5e20",alpha=.85,zorder=2))
            ax.text(c, r, f"T\n{val:.1f}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold", zorder=3)
        elif show_numbers:
            ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                    color=col, fontsize=8, zorder=3)
        ax.text(c-.42, r-.38, str(s), ha="left", va="top",
                color="#777799", fontsize=5, zorder=3)

    ax.set_xticks(range(env.SIZE)); ax.set_yticks(range(env.SIZE))
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
        colors="#9e9ebb", labelsize=7)


def plot_policy(env, pi, title, ax, color="#6a1b9a"):
    """Policy arrows on the 4×4 grid."""
    ARROW = {0:(0,-.35), 1:(.35,0), 2:(0,.35), 3:(-.35,0)}
    ax.set_xlim(-.5,3.5); ax.set_ylim(-.5,3.5); ax.set_aspect("equal")
    ax.set_facecolor(DARK_BG)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.grid(color=GRID_COLOR, lw=.5, alpha=.5)

    for s in range(env.N_STATES):
        r, c = env.s2rc(s)
        ax.text(c-.42, r-.38, str(s), ha="left", va="top",
                color="#777799", fontsize=5, zorder=3)
        if env.is_terminal(s):
            ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,color="#1b5e20",alpha=.85,zorder=2))
            ax.text(c, r, "T", ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold", zorder=3)
            continue
        actions = np.where(pi[s] > 0.01)[0]
        for a in actions:
            dc, dr = ARROW[a]
            w = float(pi[s, a])
            ax.annotate("", xy=(c+dc, r+dr), xytext=(c, r),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.5+w, alpha=0.4+0.6*w), zorder=3)

    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
    ax.tick_params(colors="#9e9ebb", labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)


def plot_convergence(ax, history, ylabel, title, color="#6a1b9a"):
    """Plot max-delta convergence over sweeps."""
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]
    ax.semilogy(xs, ys, color=color, lw=2.5, marker="o", ms=4)
    ax.set_xlabel("Sweep / Update", color="white", fontsize=10)
    ax.set_ylabel(ylabel, color="white", fontsize=10)
    ax.set_title(title, color="white", fontweight="bold")
    ax.grid(alpha=.15)


def smooth(arr, w=5):
    if len(arr) <= w: return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w)/w, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main_dp():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#4a0080,#01579b,#1b5e20);
                padding:2rem 2.5rem; border-radius:14px; margin-bottom:1.5rem">
        <h1 style="color:white;margin:0;font-size:2.4rem">🧮 DP RL Explorer</h1>
        <p style="color:#b0bec5;margin-top:.5rem;font-size:1.05rem">
            Dynamic Programming for Reinforcement Learning — every Bellman equation decoded,
            every sweep visualised, every algorithm explained from first principles
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Global intro ─────────────────────────────────────────────────────────
    with st.expander("🎓 New here? — What is Dynamic Programming in RL?", expanded=False):
        st.markdown(r"""
        <div style="background:#12121f;border-radius:12px;padding:1.4rem 1.8rem;border:1px solid #2a2a3e">

        ### 🧠 The Big Picture — Where DP Fits

        | Method family | Needs model? | Learns from? | Updates when? |
        |--------------|-------------|-------------|--------------|
        | **Dynamic Programming** | ✅ Yes (full p(s',r\|s,a)) | Model computation | Every state, every sweep |
        | Monte Carlo | ❌ No | Complete episodes | After each episode |
        | TD Learning | ❌ No | Each step | After each step |

        > **DP's superpower:** Because it knows the full environment model, it can compute exact optimal
        > values and policies — no randomness, no sampling, no waiting for episodes to end.
        > **DP's limitation:** Requires complete knowledge of p(s',r|s,a) — rarely available in practice.
        > But DP is the *theoretical foundation* for everything else in RL.

        ---

        ### 🔑 The Two Bellman Equations — The Heart of DP

        **Bellman Expectation Equation** (used in Policy Evaluation):
        $$v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_\pi(s')\bigr]$$
        *"The value of state s = weighted average over actions × (reward + discounted next-state value)"*

        **Bellman Optimality Equation** (used in Value Iteration):
        $$v_*(s) = \max_a \sum_{s',r} p(s',r|s,a)\bigl[r + \gamma v_*(s')\bigr]$$
        *"The optimal value of s = the BEST action's expected return"*

        ---

        ### 🗺️ The 6 DP Algorithms

        | # | Algorithm | What it computes | Key mechanism |
        |---|-----------|-----------------|--------------|
        | 1 | **Policy Evaluation** | V_π for a fixed policy | Iterative Bellman expectation sweeps |
        | 2 | **Policy Improvement** | Better π' from V_π | Greedy one-step lookahead |
        | 3 | **Policy Iteration** | Optimal π* and V* | Alternate evaluation + improvement |
        | 4 | **Value Iteration** | Optimal V* directly | Bellman optimality sweeps (max not sum) |
        | 5 | **Async / In-place DP** | Same as above, faster | Update states out of order |
        | 6 | **GPI** | Conceptual framework | Shows how ALL DP/TD/MC methods relate |

        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        st.caption("Adjust and click Run.")
        gamma = st.slider("γ (discount)", 0.80, 1.00, 1.00, 0.01,
                   help="γ=1: future rewards count equally to immediate ones (standard for GridWorld). γ<1: future discounted — try 0.9 to see how values shrink for far-away states.")
        theta = st.select_slider("θ (convergence threshold)",
                   options=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                   value=1e-4,
                   help="Stop iterating when max|V_new - V_old| < θ. Smaller θ = more sweeps = more accurate.")
        show_pi_iter = st.slider("Policy Iteration — max rounds", 2, 10, 6, 1,
                   help="How many evaluate+improve cycles to show in the Policy Iteration animation.")
        seed = st.number_input("Random seed", 0, 9999, 42)

        run_btn = st.button("🚀 Run All Methods", type="primary", use_container_width=True)

        st.divider()
        st.markdown("""
        **4×4 GridWorld**
        | Symbol | Meaning |
        |--------|---------|
        | T | Terminal (states 0 & 15) |
        | 0–15 | State numbers |
        | →↑↓← | Policy directions |
        | R=−1 | Every step costs −1 |
        | γ=1 | Undiscounted (default) |

        **DP requires:**
        - Full model p(s',r|s,a)
        - All states computed each sweep
        - No real episodes needed

        **6 Algorithms:**
        ```
        Prediction
          └─ Policy Evaluation
        Improvement
          └─ Policy Improvement
        Control
          ├─ Policy Iteration
          └─ Value Iteration
        Variants
          └─ Async / In-place DP
        Framework
          └─ GPI
        ```
        """)

    env = GridWorld4x4()
    np.random.seed(seed)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    (tab_env, tab_eval, tab_impr, tab_pi,
     tab_vi, tab_async, tab_gpi, tab_guide) = st.tabs([
        "🗺️ Environment",
        "🔄 Policy Evaluation",
        "⬆️ Policy Improvement",
        "🔁 Policy Iteration",
        "⚡ Value Iteration",
        "🔀 Async DP",
        "🔮 GPI Framework",
        "📚 Method Guide",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 0 — ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════════
    with tab_env:
        st.markdown(_card("#0277bd","🗺️","Why the 4×4 GridWorld?",
            """The 4×4 GridWorld is the <b>exact example</b> from Sutton & Barto §4.1 (Figure 4.1).
            It was chosen as the canonical DP example because it is small enough to compute by hand,
            yet large enough to show non-trivial policy differences. Every value and policy shown in
            this app can be verified against the textbook figures.<br><br>
            DP requires a <b>complete model</b> — we must know p(s',r|s,a) for every transition.
            In GridWorld we know it exactly: each move is deterministic, reward is always −1,
            walls cause the agent to stay in place."""), unsafe_allow_html=True)

        st.subheader("🗺️ The 4×4 GridWorld — Sutton & Barto §4.1")
        c1, c2 = st.columns([1.3, 0.7])
        with c1:
            st.markdown("""
            #### Structure
            The grid has **16 states** numbered 0–15 (row-major order: state 0 = top-left,
            state 15 = bottom-right). Two **terminal states** (0 and 15) end the episode —
            both are shaded in the diagrams.

            #### Transition dynamics — the complete model
            - **Move** in any of 4 directions (↑ → ↓ ←)
            - **Wall rule**: hitting a boundary → stay in current cell
            - **Reward**: always **−1** per step (including the final step into a terminal)
            - **Termination**: entering state 0 or 15

            #### Why reward = −1 everywhere?
            The agent earns −1 for every step it takes. The optimal policy minimises
            the *number of steps* to a terminal — it finds the shortest path.
            States far from any terminal have more negative V(s) because they need more steps.

            #### The full transition model p(s', r | s, a)
            """)
            st.markdown("""
            For state **s=5** (row 1, col 1):
            | Action | Next state | Reward |
            |--------|-----------|--------|
            | ↑ Up | 1 | −1 |
            | → Right | 6 | −1 |
            | ↓ Down | 9 | −1 |
            | ← Left | 4 | −1 |

            For state **s=1** (row 0, col 1, near top wall):
            | Action | Next state | Reward |
            |--------|-----------|--------|
            | ↑ Up | **1** (wall, stay) | −1 |
            | → Right | 2 | −1 |
            | ↓ Down | 5 | −1 |
            | ← Left | 0 (terminal) | −1 |
            """)

        with c2:
            st.caption("📖 **How to read this map:** State numbers (0–15) are shown in the top-left of each cell. Green cells = terminal states. The agent starts anywhere (DP evaluates ALL states simultaneously, unlike MC/TD which start from one state).")
            fig, ax, _ = make_fig(1, 1, 4, 4)
            ax.set_xlim(-.5,3.5); ax.set_ylim(-.5,3.5); ax.set_aspect("equal")
            for s in range(16):
                r, c = env.s2rc(s)
                bg = "#1b5e20" if env.is_terminal(s) else "#1e1e2e"
                ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,color=bg,ec="#2a2a3e",lw=1.5))
                label = "T\n(terminal)" if env.is_terminal(s) else str(s)
                ax.text(c, r, label, ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.invert_yaxis()
            ax.set_title("4×4 GridWorld", color="white", fontweight="bold")
            ax.tick_params(colors="#9e9ebb")
            for sp in ax.spines.values(): sp.set_edgecolor(GRID_COLOR)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        st.subheader("🔑 DP vs MC vs TD — What Makes DP Unique?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🧮 Dynamic Programming**
            ```
            FOR each state s:
              V(s) = Σ_a π(a|s) Σ p(s',r|s,a)
                        [r + γV(s')]
            ```
            ✅ Exact model used
            ✅ All states updated each sweep
            ✅ Provably optimal result
            ❌ Needs full p(s',r|s,a)
            ❌ Expensive for large state spaces
            """)
        with col2:
            st.markdown("""
            **🎲 Monte Carlo**
            ```
            PLAY episode to end
            G = actual total return
            V(s) += (G - V(s)) / N(s)
            ```
            ✅ No model needed
            ✅ Unbiased
            ❌ Waits for episode end
            ❌ High variance
            ❌ Episodic tasks only
            """)
        with col3:
            st.markdown("""
            **⚡ TD Learning**
            ```
            TAKE one step
            target = R + γV(S')
            V(S) += α[target - V(S)]
            ```
            ✅ No model needed
            ✅ Online (each step)
            ✅ Continuing tasks
            ❌ Biased (bootstrap)
            ❌ Needs tuning (α)
            """)

        with st.expander("🔢 Worked example — compute V(5) by hand (1 sweep)"):
            st.markdown(r"""
            Under the **uniform random policy** (π(a|s) = 0.25 for all actions),
            starting with V(s) = 0 everywhere, γ = 1:

            **State 5** (row 1, col 1) transitions:
            - ↑: goes to s=1, reward=−1
            - →: goes to s=6, reward=−1
            - ↓: goes to s=9, reward=−1
            - ←: goes to s=4, reward=−1

            $$V_1(5) = \sum_a \pi(a|5) \sum_{s',r} p(s',r|5,a)[r + \gamma V_0(s')]$$

            $$= 0.25 \times [(-1 + 1 \times 0) + (-1 + 0) + (-1 + 0) + (-1 + 0)] = 0.25 \times (-4) = -1$$

            After sweep 1: V(5) = **−1**.

            After sweep 2: each neighbour of state 5 has V = −1 (computed above),
            so V_2(5) = 0.25 × [(−1 + −1)×4] = **−2**.

            After k sweeps: V_k(s) ≈ **−(distance to nearest terminal)**.
            """)
        render_dp_notes("Environment", "dynamic_programming")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL
    # ═══════════════════════════════════════════════════════════════════════
    if run_btn or "dp_results" in st.session_state:
        if run_btn:
            with st.spinner("Computing all DP algorithms…"):
                pi_rand   = env.uniform_policy()
                V_rand, h_eval   = policy_evaluation(env, pi_rand, gamma, theta)
                pi_greedy, Q_impr = policy_improvement(env, V_rand, gamma)
                h_pi              = policy_iteration(env, gamma, theta)
                V_vi, pi_vi, Q_vi, h_vi = value_iteration(env, gamma, theta)
                V_async, h_async  = async_value_iteration(env, gamma, "inplace", theta)
                V_prio, h_prio    = async_value_iteration(env, gamma, "prioritised", theta)
                h_gpi             = gpi_trace(env, gamma, theta, show_pi_iter)

                st.session_state["dp_results"] = dict(
                    V_rand=V_rand, h_eval=h_eval,
                    pi_rand=pi_rand,
                    pi_greedy=pi_greedy, Q_impr=Q_impr,
                    h_pi=h_pi,
                    V_vi=V_vi, pi_vi=pi_vi, Q_vi=Q_vi, h_vi=h_vi,
                    V_async=V_async, h_async=h_async,
                    V_prio=V_prio, h_prio=h_prio,
                    h_gpi=h_gpi,
                )
            st.sidebar.success("✅ Done!")

        res = st.session_state["dp_results"]

        # ═══════════════════════════════════════════════════════════════════
        # TAB 1 — POLICY EVALUATION
        # ═══════════════════════════════════════════════════════════════════
        with tab_eval:
            st.markdown(_card("#6a1b9a","🔄","What does Policy Evaluation solve?",
                """<b>The prediction problem:</b> Given a fixed policy π (here: the uniform random policy,
                where the agent picks ↑→↓← each with probability 0.25), how good is each state?<br><br>
                <b>The method:</b> Repeatedly apply the Bellman expectation equation to every state.
                Each sweep makes V(s) more accurate by incorporating the values of neighbouring states.
                After enough sweeps, V converges to the true V_π.<br><br>
                <b>Why it's the foundation:</b> Policy Evaluation is the "evaluation" half of all DP control
                algorithms. You cannot improve a policy without first knowing how good it is."""),
                unsafe_allow_html=True)

            st.subheader("🔄 Policy Evaluation — Iterative Bellman Sweeps")

            with st.expander("📐 Theory & Formulas — Policy Evaluation", expanded=False):
                st.markdown(r"""
                #### The Bellman Expectation Equation

                The true value of state $s$ under policy $\pi$ satisfies:

                $$\boxed{v_\pi(s) = \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)\bigl[r + \gamma\,v_\pi(s')\bigr]}$$

                **Symbol decoder:**
                - $v_\pi(s)$ — the value we want to find (expected total return from s under π)
                - $\pi(a|s)$ — probability that policy π chooses action $a$ from state $s$
                - $p(s',r|s,a)$ — the model: probability of landing in $s'$ with reward $r$ after taking $a$ from $s$
                - $r$ — immediate reward
                - $\gamma$ — discount factor (your slider)
                - $v_\pi(s')$ — value of the next state (which we're also estimating — circular!)

                #### Why Is It Circular? The Iterative Solution

                $v_\pi(s)$ depends on $v_\pi(s')$, which depends on $v_\pi(s'')$, etc.
                No closed-form solution in general. Instead, start with $V_0(s) = 0$ and iterate:

                $$\boxed{V_{k+1}(s) \leftarrow \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)\bigl[r + \gamma\,V_k(s')\bigr]}$$

                **Guaranteed convergence:** As $k\to\infty$, $V_k \to v_\pi$ for any $\pi$ and any $\gamma \in [0,1)$.
                For $\gamma=1$: convergence holds if the task is episodic with terminal states (like GridWorld).

                #### Under the Uniform Random Policy (4×4 GridWorld, γ=1)

                $\pi(a|s) = 0.25$ for all $a$, so:

                $$V_{k+1}(s) = \frac{1}{4}\sum_{a}\bigl[-1 + V_k(\text{next}(s,a))\bigr]$$

                = average of (−1 + current value of each neighbour).

                Intuition: V(s) ≈ **−(expected number of steps to reach a terminal from s)**.
                States far from both terminals (e.g. state 6 in the middle) take more steps → more negative V.

                #### Stopping Criterion

                Stop when the maximum change across all states is below θ:

                $$\max_s |V_{k+1}(s) - V_k(s)| < \theta$$

                Your θ slider controls this. θ=0.0001 typically requires ~100–200 sweeps for this grid.

                #### The Contraction Mapping Theorem

                The iterative update is a contraction in the supremum norm:

                $$\|V_{k+1} - v_\pi\|_\infty \leq \gamma \|V_k - v_\pi\|_\infty$$

                Each sweep reduces the error by at least factor γ.
                With γ=1: convergence still holds for episodic tasks via a different argument
                (the terminal states provide an absorbing boundary).
                """)

            n_eval = len(res["h_eval"])
            st.markdown(_card("#6a1b9a","📖","How to read the heatmap and convergence chart",
                f"""<b>Left heatmap (V_π):</b> Each cell's colour and number shows the estimated state
                value under the random policy. <b>Green = less negative = closer to a terminal</b>.
                Red = very negative = far from both terminals, takes many random steps to escape.
                The two green cells (0 and 15) are the terminals (V=0 by convention).<br><br>
                <b>Right chart (convergence):</b> Y-axis (log scale) = max|V_new - V_old| across all states.
                Each point = one complete sweep of all 16 states. The curve drops toward θ={theta:.0e}.
                Took {n_eval} sweeps to converge with these settings."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 13, 5)
            plot_V(env, res["V_rand"], "V_π — Random Policy (after convergence)", axes[0])
            plot_convergence(axes[1], res["h_eval"], "max |ΔV|", "Convergence of Policy Evaluation",
                             color=METHOD_COLORS["Policy Eval"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Sweep-by-sweep animation selection
            st.divider()
            st.subheader("🎞️ Watch V evolve — sweep by sweep")
            st.markdown("""
            Select a sweep number below to see the exact state of V at that point in the iteration.
            Early sweeps: only states one step from terminals have non-zero values.
            Later: the "wave" of negative values spreads inward from the terminals.
            """)
            st.markdown(_card("#6a1b9a","📖","How to read the sweep snapshots",
                """Each subplot shows V at a different point during evaluation.
                <b>Early sweeps (e.g. k=1,2):</b> Only the immediate neighbours of terminals have
                non-zero values (V ≈ −1). Everything else is still 0.<br>
                <b>Middle sweeps:</b> The negative values "ripple" outward — states 2 steps away get V≈−2, etc.<br>
                <b>Final sweep:</b> Fully converged V_π — reflects the true expected return of the random policy."""),
                unsafe_allow_html=True)

            all_h    = res["h_eval"]
            n_snaps  = min(6, len(all_h))
            snap_idx = np.linspace(0, len(all_h)-1, n_snaps, dtype=int)
            fig2, axes2, axl2 = make_fig(1, n_snaps, 18, 3.5)
            for i, idx in enumerate(snap_idx):
                sweep, delta, V_snap = all_h[idx]
                plot_V(env, V_snap, f"k={sweep}\nδ={delta:.3f}", axes2[i],
                       vmin=-20, vmax=0)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            st.markdown(_tip("""
            <b>Experiment:</b> Change θ to 0.1 (coarse) — convergence in ~10 sweeps but V is
            inaccurate. Change to 1e-6 (fine) — more sweeps but more accurate.
            Also try γ=0.9: values shrink exponentially with distance, making the grid more
            uniformly dark (every state is discounted to near 0 for large distances).
            """), unsafe_allow_html=True)

            # State-by-state table
            st.divider()
            st.subheader("📋 Final V_π Values — All 16 States")
            st.markdown("""
            The table below shows the converged V_π for every state.
            Under the random policy, values should increase (become less negative)
            as states get closer to either terminal.
            """)
            rows = []
            for s in range(env.N_STATES):
                r, c = env.s2rc(s)
                dist0  = abs(r) + abs(c)
                dist15 = abs(r-3) + abs(c-3)
                rows.append({
                    "State": s, "Row": r, "Col": c,
                    "V_π(s)": f"{res['V_rand'][s]:.3f}",
                    "Min dist to terminal": min(dist0, dist15),
                    "Terminal?": "✅" if env.is_terminal(s) else ""
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            render_dp_notes("Policy Evaluation", "dynamic_programming_policy_evaluation")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 2 — POLICY IMPROVEMENT
        # ═══════════════════════════════════════════════════════════════════
        with tab_impr:
            st.markdown(_card("#0277bd","⬆️","What does Policy Improvement solve?",
                """<b>The improvement step:</b> We've evaluated the random policy and know V_π.
                Now: can we find a BETTER policy?<br><br>
                <b>Policy Improvement Theorem:</b> For any policy π and its value V_π,
                the greedy policy π' (which always picks the best single-step action based on V_π)
                is guaranteed to be <em>at least as good</em> as π — and usually strictly better.<br><br>
                <b>The mechanism:</b> For each state, look at all 4 actions. Which action leads to
                the state with the highest value? Do that action. This is a "greedy one-step lookahead"
                using V_π as a guide to the future."""), unsafe_allow_html=True)

            st.subheader("⬆️ Policy Improvement — Greedy One-Step Lookahead")

            with st.expander("📐 Theory & Formulas — Policy Improvement", expanded=False):
                st.markdown(r"""
                #### The Q Function — Action Values

                First compute the **Q-values**: expected return from state $s$ if you take action $a$
                (just once), then follow the current policy π forever after:

                $$\boxed{q_\pi(s,a) = \sum_{s',r}p(s',r|s,a)\bigl[r + \gamma\,v_\pi(s')\bigr]}$$

                **Symbol decoder:**
                - $q_\pi(s,a)$ — value of action $a$ in state $s$ under policy $\pi$
                - The sum covers all possible next states s' and rewards r (in GridWorld: only one each, deterministic)
                - For GridWorld: $q_\pi(s,a) = -1 + \gamma\,v_\pi(\text{next}(s,a))$ (deterministic)

                #### The Improvement Step

                The greedy improved policy π' picks the highest-Q action:

                $$\boxed{\pi'(s) = \arg\max_a q_\pi(s,a) = \arg\max_a \sum_{s',r}p(s',r|s,a)\bigl[r + \gamma\,v_\pi(s')\bigr]}$$

                #### The Policy Improvement Theorem (Proof sketch)

                For any state $s$, since $\pi'$ is greedy w.r.t. $q_\pi$:

                $$q_\pi(s, \pi'(s)) = \max_a q_\pi(s,a) \geq q_\pi(s, \pi(s)) = v_\pi(s)$$

                The first step uses the definition of greedy (takes the max), the last step uses the
                Bellman consistency of $v_\pi$. Unrolling this inequality over all future steps:

                $$v_{\pi'}(s) \geq v_\pi(s) \quad \forall s$$

                **If** $v_{\pi'} = v_\pi$ (no improvement), both satisfy the Bellman optimality equation
                — meaning $\pi' = \pi^*$ is already optimal.

                #### Worked Example (State 1, top row)

                State 1 is in the top row (row 0, col 1). Neighbours:
                - ↑: hits wall → state 1 itself, V_π(1) ≈ −14
                - →: state 2, V_π(2) ≈ −14
                - ↓: state 5, V_π(5) ≈ −18
                - ←: **state 0 (terminal), V_π(0) = 0**

                Q values with γ=1:
                - q(1,↑) = −1 + (−14) = −15
                - q(1,→) = −1 + (−14) = −15
                - q(1,↓) = −1 + (−18) = −19
                - **q(1,←) = −1 + 0 = −1** ← best!

                → π'(1) = ← (Left): go directly toward terminal 0. ✓
                """)

            st.markdown(_card("#0277bd","📖","How to read the improvement diagrams",
                """<b>Left — Before (random policy):</b> All arrows point uniformly in all 4 directions
                (shown as equal-weight arrows). This is the π_random policy we evaluated.<br>
                <b>Middle — Q-values heatmap:</b> The estimated action-value for the BEST action from each state.
                Greener = there exists a good action from here (one direction leads closer to a terminal).<br>
                <b>Right — After (greedy policy):</b> Each cell now has ONE arrow — the best direction
                according to the one-step lookahead using V_π. These arrows should all point
                <em>toward</em> either state 0 or state 15."""),
                unsafe_allow_html=True)

            V_rand   = res["V_rand"]
            pi_rand  = res["pi_rand"]
            pi_greed = res["pi_greedy"]
            Q_impr   = res["Q_impr"]
            V_Q_best = np.array([np.max(Q_impr[s]) for s in range(env.N_STATES)])

            fig, axes, axl = make_fig(1, 3, 18, 5)
            plot_policy(env, pi_rand,  "Before: Random Policy π",    axes[0], color="#888888")
            plot_V(env, V_Q_best,      "Q-max: best action value",    axes[1],
                   vmin=V_Q_best.min()-1, vmax=0)
            plot_policy(env, pi_greed, "After: Greedy Policy π'",     axes[2], color=METHOD_COLORS["Pol Improve"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Q-value table for a few states
            st.divider()
            st.subheader("📋 Q-values for All Actions — Selected States")
            st.markdown(r"""
            The table shows $q_\pi(s,a)$ for 6 representative states.
            The ✅ column marks the greedy action (highest Q). Notice how states near
            the terminals have one action that is dramatically better than the others.
            """)
            showcase = [1, 2, 5, 6, 9, 14]
            rows = []
            for s in showcase:
                r, c = env.s2rc(s)
                best_a = int(np.argmax(Q_impr[s]))
                rows.append({
                    "State": s, f"↑ Q(s,0)": f"{Q_impr[s,0]:.2f}",
                    "→ Q(s,1)": f"{Q_impr[s,1]:.2f}",
                    "↓ Q(s,2)": f"{Q_impr[s,2]:.2f}",
                    "← Q(s,3)": f"{Q_impr[s,3]:.2f}",
                    "Best action": f"{env.SYMBOLS[best_a]} ✅",
                    "V_π(s)": f"{V_rand[s]:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown(_insight("""
            <b>Key insight:</b> The greedy policy after one improvement is dramatically better
            than the random policy, even though V_π was computed under the random policy.
            This is the power of the Policy Improvement Theorem: a single greedy step using
            ANY value function V_π gives a policy that is at least as good as π.
            """), unsafe_allow_html=True)
            render_dp_notes("Policy Improvement", "dynamic_programming_policy_improvement")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 3 — POLICY ITERATION
        # ═══════════════════════════════════════════════════════════════════
        with tab_pi:
            st.markdown(_card("#00695c","🔁","What does Policy Iteration solve?",
                """<b>The full control loop:</b> Policy Evaluation gives us V_π. Policy Improvement
                gives us π' from V_π. What if we keep alternating?<br><br>
                <b>Policy Iteration</b> does exactly this — until the policy stops changing.
                The sequence π₀ → π₁ → π₂ → … is <em>monotonically improving</em> (each is at
                least as good as the last) and must converge to π* in finite steps (finite state space).<br><br>
                <b>Why it converges quickly:</b> For GridWorld, Policy Iteration typically converges
                in just 3–5 iterations — far fewer than Value Iteration needs sweeps.
                Each iteration produces a policy improvement that jumps toward optimal."""),
                unsafe_allow_html=True)

            st.subheader("🔁 Policy Iteration — Evaluation + Improvement Until Stable")

            with st.expander("📐 Theory & Formulas — Policy Iteration", expanded=False):
                st.markdown(r"""
                #### The Policy Iteration Algorithm

                ```
                Initialise π₀ (e.g. random), V₀ = 0
                Loop:
                    ── Evaluation ──────────────────────────────────────────
                    Repeat until convergence:
                        V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
                    ── Improvement ─────────────────────────────────────────
                    stable ← True
                    For each s:
                        old_a ← π(s)
                        π(s)  ← argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
                        If old_a ≠ π(s): stable ← False
                Until stable
                Output: π* ≈ π, V* ≈ V
                ```

                #### The Two Bellman Equations in Tandem

                **Evaluation step** — Bellman *expectation*:
                $$V_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V_\pi(s')]$$

                **Improvement step** — Bellman *optimality* (one-step greedy):
                $$\pi'(s) = \arg\max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V_\pi(s')]$$

                #### Why Does It Converge?

                At each iteration $k$: $v_{\pi_{k+1}}(s) \geq v_{\pi_k}(s)$ for all $s$.
                Since the state space is finite, there are finitely many deterministic policies.
                The sequence $v_{\pi_0} \leq v_{\pi_1} \leq \cdots$ is bounded above by $v^*$.
                Therefore: must converge in at most $|A|^{|S|}$ iterations (in practice: very few).

                #### Convergence Speed

                Policy Iteration for this 4×4 GridWorld converges in exactly **3 iterations**:
                - π₀: random (all directions equally)
                - π₁: greedy w.r.t. V_{π₀} — mostly correct but some ties
                - π₂: optimal π* (all arrows point directly toward nearest terminal)

                This is a known result from S&B: for simple deterministic tasks, Policy Iteration
                is extremely fast because each evaluation fully converges before improving.
                """)

            h_pi = res["h_pi"]
            n_iterations = len(h_pi)
            st.markdown(_card("#00695c","📖","How to read the Policy Iteration grid",
                f"""Each column = one iteration of Policy Iteration ({n_iterations} total in this run).<br>
                <b>Top row (value functions):</b> V converges — notice how values become more negative
                (and more structured) in early iterations as the policy gets smarter.<br>
                <b>Bottom row (policies):</b> Arrows converge — random chaos → mostly-directed →
                fully optimal (all arrows pointing toward nearest terminal).<br>
                <b>Red border = changed</b>, <b>green border = stable</b> on the final iteration."""),
                unsafe_allow_html=True)

            n_show = min(n_iterations, 4)
            fig, axes, axl = make_fig(2, n_show, 5*n_show, 10)
            for i in range(n_show):
                it = h_pi[i]
                border_color = "#4caf50" if it["stable"] else "#ef5350"
                for spine_ax in [axes[0][i], axes[1][i]]:
                    for sp in spine_ax.spines.values():
                        sp.set_edgecolor(border_color); sp.set_linewidth(2)
                plot_V(env, it["V"],  f"Iter {it['iter']} — V", axes[0][i],
                       vmin=-22, vmax=0)
                plot_policy(env, it["pi"], f"Iter {it['iter']} — Policy"
                            + (" ✅ STABLE" if it["stable"] else ""),
                            axes[1][i], color=METHOD_COLORS["Pol Iter"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Iteration summary table
            st.divider()
            st.subheader("📋 Iteration Summary")
            rows = []
            for it in h_pi:
                rows.append({
                    "Iteration": it["iter"],
                    "Eval sweeps": it["eval_sweeps"],
                    "V(state 6)": f"{it['V'][6]:.3f}",
                    "Policy stable?": "✅ Yes — DONE" if it["stable"] else "❌ No — continue",
                    "Total eval sweeps so far": sum(h["eval_sweeps"] for h in h_pi[:it["iter"]]),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            render_dp_notes("Policy Iteration", "dynamic_programming_policy_iteration")

            st.markdown(_tip("""
            <b>Key observation:</b> Policy Iteration converges in very few iterations (typically 3 here)
            but each iteration requires running Policy Evaluation to convergence (many sweeps).
            Value Iteration (next tab) skips the full convergence of evaluation — it's one sweep per iteration,
            but many more iterations. Both reach the same optimal policy.
            """), unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════════
        # TAB 4 — VALUE ITERATION
        # ═══════════════════════════════════════════════════════════════════
        with tab_vi:
            st.markdown(_card("#e65100","⚡","What does Value Iteration solve?",
                """<b>The question:</b> Policy Iteration requires full policy evaluation before each
                improvement. What if we only do ONE sweep of evaluation, then improve, then sweep again?<br><br>
                <b>Value Iteration:</b> Combine evaluation and improvement into a single update.
                Instead of Σ_a π(a|s)·[...], use max_a·[...] directly. This is the Bellman
                <em>optimality</em> equation applied iteratively.<br><br>
                <b>Why it works:</b> V converges to V* directly — no separate policy needed during
                iteration. At the end, extract π* by one greedy step from V*.<br><br>
                <b>Trade-off vs Policy Iteration:</b> Each sweep is slightly cheaper (no policy),
                but requires more sweeps to converge (no full evaluation between improvements)."""),
                unsafe_allow_html=True)

            st.subheader("⚡ Value Iteration — Bellman Optimality Sweeps")

            with st.expander("📐 Theory & Formulas — Value Iteration", expanded=False):
                st.markdown(r"""
                #### The Bellman Optimality Equation

                The optimal value function $v^*$ satisfies:

                $$\boxed{v^*(s) = \max_a \sum_{s',r}p(s',r|s,a)\bigl[r + \gamma\,v^*(s')\bigr]}$$

                **Key difference from Policy Evaluation:** Uses $\max_a$ instead of $\sum_a \pi(a|s)$.
                This is the Bellman *optimality* equation — it bakes in the best policy implicitly.

                #### Value Iteration Update Rule

                $$\boxed{V_{k+1}(s) \leftarrow \max_a \sum_{s',r}p(s',r|s,a)\bigl[r + \gamma\,V_k(s')\bigr]}$$

                Same structure as Policy Evaluation but with $\max$ replacing the weighted sum.

                **Worked example** (state 5, γ=1, V_k = V_{rand}):
                - ↑: −1 + V(1), → : −1 + V(6), ↓: −1 + V(9), ←: −1 + V(4)
                - $V_{k+1}(5) = \max\{-1+V(1),\;-1+V(6),\;-1+V(9),\;-1+V(4)\}$
                - Takes the maximum over neighbours (not the average like Policy Evaluation)

                #### Extracting π* After Convergence

                Value Iteration doesn't maintain an explicit policy during iteration.
                After V converges to V*, extract π* in one final greedy step:

                $$\pi^*(s) = \arg\max_a \sum_{s',r}p(s',r|s,a)\bigl[r + \gamma\,V^*(s')\bigr]$$

                #### Convergence

                Value Iteration is a contraction in the supremum norm with factor γ:
                $$\|V_{k+1} - v^*\|_\infty \leq \gamma \|V_k - v^*\|_\infty$$

                For γ=1 (episodic tasks): converges via the special structure of terminal states.
                Number of sweeps typically ≈ diameter of the state space (longest shortest path).
                For 4×4 GridWorld: ~100–200 sweeps with θ=0.0001.

                #### Policy Iteration vs Value Iteration

                | | Policy Iteration | Value Iteration |
                |-|-----------------|----------------|
                | **Update** | $\sum_a \pi(a|s)[\cdots]$ then $\max_a$ | $\max_a[\cdots]$ directly |
                | **Sweeps per "iteration"** | Many (full eval) | 1 |
                | **"Iterations"** | Few (3–5) | Many (100–200) |
                | **Total work** | Similar | Similar |
                | **Policy needed** | Yes (explicit) | No (implicit in max) |
                """)

            st.markdown(_card("#e65100","📖","How to read the Value Iteration results",
                """<b>Left — V* heatmap:</b> The converged optimal value function. Compare to the
                V_π (random policy) from the Policy Evaluation tab — V* is much less negative,
                especially for states far from the terminals. The optimal policy finds them quickly.<br>
                <b>Middle — Policy arrows (π*):</b> Extracted greedily from V*. All arrows should
                point directly toward the nearest terminal — the shortest path policy.<br>
                <b>Right — Convergence:</b> Max|ΔV| per sweep on log scale. Compare the number of
                sweeps needed (Value Iteration) vs number of full Policy Iteration iterations (previous tab)."""),
                unsafe_allow_html=True)

            V_vi   = res["V_vi"]
            pi_vi  = res["pi_vi"]
            h_vi   = res["h_vi"]
            n_vi   = len(h_vi)

            fig, axes, axl = make_fig(1, 3, 18, 5)
            plot_V(env, V_vi, f"V* — Optimal (after {n_vi} sweeps)", axes[0], vmin=-15, vmax=0)
            plot_policy(env, pi_vi, "π* — Optimal Policy (from V*)", axes[1],
                        color=METHOD_COLORS["Val Iter"])
            plot_convergence(axes[2], h_vi, "max|ΔV|", "Value Iteration Convergence",
                             color=METHOD_COLORS["Val Iter"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Compare V_rand vs V*
            st.divider()
            st.subheader("🆚 V_π (random) vs V* (optimal) — State-by-state")
            st.markdown(_card("#e65100","📖","How to read this comparison chart",
                """Each group of bars = one state. <b>Purple = V_π (random policy)</b>,
                <b>Orange = V* (optimal policy)</b>.<br>
                V* is always ≥ V_π (the optimal policy is at least as good as random).
                The gap is largest in the middle states (far from terminals) — the random policy
                wanders for a long time; the optimal policy goes straight there."""),
                unsafe_allow_html=True)

            V_rand = res["V_rand"]
            xs = range(env.N_STATES)
            fig2, ax2, _ = make_fig(1, 1, 14, 4)
            bar_w = 0.38
            bars1 = ax2.bar([x-bar_w/2 for x in xs], V_rand, bar_w,
                           color=METHOD_COLORS["Policy Eval"], alpha=0.85, label="V_π (random)")
            bars2 = ax2.bar([x+bar_w/2 for x in xs], V_vi,   bar_w,
                           color=METHOD_COLORS["Val Iter"], alpha=0.85, label="V* (optimal)")
            ax2.set_xlabel("State", color="white", fontsize=11)
            ax2.set_ylabel("Value V(s)", color="white", fontsize=11)
            ax2.set_title("V_π (random) vs V* (optimal) — Every State", color="white", fontweight="bold")
            ax2.legend(facecolor=CARD_BG, labelcolor="white")
            ax2.axhline(0, color="white", lw=0.5, alpha=0.3)
            ax2.grid(alpha=0.15, axis="y")
            # Mark terminals
            ax2.axvspan(-0.5, 0.5, alpha=0.1, color="#4caf50")
            ax2.axvspan(14.5, 15.5, alpha=0.1, color="#4caf50")
            ax2.text(0, ax2.get_ylim()[0]*0.85, "T", color="#4caf50", fontsize=9, ha="center")
            ax2.text(15, ax2.get_ylim()[0]*0.85, "T", color="#4caf50", fontsize=9, ha="center")
            plt.tight_layout(); st.pyplot(fig2); plt.close()
            render_dp_notes("Value Iteration", "dynamic_programming_value_iteration")

            improvement = V_vi - V_rand
            c1,c2,c3 = st.columns(3)
            c1.metric("Mean improvement V*−V_π", f"{np.mean(improvement[1:-1]):.2f}",
                      help="Average over non-terminal states")
            c2.metric("Max improvement (worst state)", f"{np.max(improvement):.2f}")
            c3.metric("Sweeps to converge", str(n_vi))

        # ═══════════════════════════════════════════════════════════════════
        # TAB 5 — ASYNC DP
        # ═══════════════════════════════════════════════════════════════════
        with tab_async:
            st.markdown(_card("#ad1457","🔀","What does Async / In-place DP solve?",
                """<b>The synchronous bottleneck:</b> Standard DP updates ALL states before using any
                new values. This requires two copies of V simultaneously and misses the opportunity
                to immediately use fresh values for nearby states.<br><br>
                <b>In-place (synchronous within sweep, asynchronous across states):</b> Update V(s)
                immediately and use the fresh value for the very next state in the same sweep.
                Convergence is often faster because information propagates faster within a sweep.<br><br>
                <b>Prioritised Sweeping:</b> Always update the state with the LARGEST Bellman error
                first. This focuses computation on the states that will benefit most."""),
                unsafe_allow_html=True)

            st.subheader("🔀 Asynchronous / In-place DP — Smarter Update Ordering")

            with st.expander("📐 Theory & Formulas — Async DP", expanded=False):
                st.markdown(r"""
                #### Synchronous DP (standard)

                ```
                V_old ← copy of V               ← snapshot before sweep
                For each s:
                    V_new(s) ← max_a Σ p(s',r|s,a)[r + γ · V_OLD(s')]   ← uses OLD values
                V ← V_new                        ← replace after full sweep
                ```

                All states are updated using the *same* snapshot of V. Two arrays needed.

                #### In-place DP (asynchronous)

                ```
                For each s:
                    V(s) ← max_a Σ p(s',r|s,a)[r + γ · V(s')]   ← uses CURRENT values
                ```

                No separate V_old array. When updating state s=5, it uses the already-updated
                V(1), V(4) (if updated earlier in this sweep) — fresh values propagate within sweep.

                **Convergence:** Still guaranteed for γ<1 (contraction). For episodic tasks (γ=1):
                convergence requires all states updated infinitely often — satisfied by sequential sweeps.

                #### Bellman Error — The Prioritisation Key

                The **Bellman error** for state s:
                $$\delta(s) = \left|\max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V(s')] - V(s)\right|$$

                = how much V(s) would change if updated right now.

                **Prioritised Sweeping** always picks $s^* = \arg\max_s \delta(s)$.

                **Why efficient?** States with δ≈0 have converged — updating them wastes computation.
                States with large δ need updating urgently. Focus on the urgent ones.

                #### Why Both Methods Converge Faster

                **In-place:** When state s=1 (near terminal 0) is updated first, its fresh value
                immediately helps s=2 (updated next) get a better estimate. Information travels
                "forward" within a single sweep — synchronous DP would need another full sweep.

                **Prioritised:** Focuses all computation on the "frontier" — states whose values
                changed recently (because a nearby state was updated). Mimics how information
                actually flows: in a wave from terminals to far states.
                """)

            V_async = res["V_async"]
            V_vi    = res["V_vi"]
            h_async = res["h_async"]
            h_prio  = res["h_prio"]
            h_vi    = res["h_vi"]

            st.markdown(_card("#ad1457","📖","How to read the async DP charts",
                """<b>Left — Value heatmap (In-place):</b> Should look identical to Value Iteration's V* —
                both compute the optimal value function; only the update ORDER differs.<br>
                <b>Right — Convergence comparison:</b> In-place DP and standard Value Iteration both
                plotted. In-place often converges in fewer sweeps because fresh values propagate faster
                within each sweep. <b>How to read:</b> lower curve = fewer sweeps = more efficient."""),
                unsafe_allow_html=True)

            fig, axes, axl = make_fig(1, 2, 13, 5)
            plot_V(env, V_async, "In-place DP — V* (async)", axes[0], vmin=-15, vmax=0)

            # Convergence comparison
            xs_vi = [h[0] for h in h_vi]
            ys_vi = [h[1] for h in h_vi]
            xs_as = [h[0] for h in h_async]
            ys_as = [h[1] for h in h_async]

            axes[1].semilogy(xs_vi, ys_vi, color=METHOD_COLORS["Val Iter"],
                             lw=2.5, marker="o", ms=3, label=f"Value Iteration ({len(h_vi)} sweeps)")
            axes[1].semilogy(xs_as, ys_as, color=METHOD_COLORS["Async DP"],
                             lw=2.5, marker="s", ms=3, alpha=0.85,
                             label=f"In-place DP ({len(h_async)} sweeps)")
            axes[1].set_xlabel("Sweep", color="white", fontsize=10)
            axes[1].set_ylabel("max|ΔV| (log scale)", color="white", fontsize=10)
            axes[1].set_title("In-place vs Synchronous Convergence", color="white", fontweight="bold")
            axes[1].legend(facecolor=CARD_BG, labelcolor="white")
            axes[1].grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Update order visualisation
            st.divider()
            st.subheader("🎞️ How update order affects convergence speed")
            st.markdown("""
            The key insight of async DP is that **update order matters**.
            Below we compare three update orders and their effect on how quickly
            information from the terminal states reaches the middle of the grid.
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **Sequential (0→15)**
                States updated 0,1,2,...,15.
                State 0 (terminal) updated first.
                Its neighbours (1, 4) immediately
                get fresh values in the same sweep.
                ✅ Fast for states near terminal 0
                ❌ Slow for states near terminal 15
                """)
            with col2:
                st.markdown("""
                **Reverse (15→0)**
                States updated 15,14,...,0.
                Terminal 15 updated first.
                Information flows from 15 to 0.
                ✅ Fast for states near terminal 15
                ❌ Slow for states near terminal 0
                """)
            with col3:
                st.markdown("""
                **Prioritised**
                Always update state with largest
                Bellman error δ(s).
                Focuses on "frontier" states.
                ✅ Fast overall
                ✅ No wasted sweeps on converged states
                ⚡ Best practical efficiency
                """)

            st.markdown(_tip("""
            <b>Practical implication:</b> For large state spaces (millions of states), synchronous DP is
            impractical — updating all states each sweep is too expensive. In-place and prioritised
            sweeping allow DP-style algorithms to scale to larger problems.
            This is the bridge toward more practical algorithms like Dyna-Q and planning with rollouts.
            """), unsafe_allow_html=True)
            render_dp_notes("Async DP", "dynamic_programming_async_dp")

        # ═══════════════════════════════════════════════════════════════════
        # TAB 6 — GPI FRAMEWORK
        # ═══════════════════════════════════════════════════════════════════
        with tab_gpi:
            st.markdown(_card("#4527a0","🔮","What is the GPI Framework?",
                """<b>The unifying insight:</b> All DP, MC, and TD methods share a common structure:
                two interacting processes — <em>evaluation</em> (estimate V for current π) and
                <em>improvement</em> (make π greedy w.r.t. V). The term <b>Generalised Policy Iteration (GPI)</b>
                refers to any algorithm that combines these two processes — regardless of how fully
                each is carried out before switching.<br><br>
                <b>Why it matters:</b> GPI explains why ALL RL algorithms work. Policy Iteration runs
                full evaluation before each improvement. Value Iteration does one sweep of evaluation.
                SARSA/Q-Learning do one step. They're all GPI with different "granularity"."""),
                unsafe_allow_html=True)

            st.subheader("🔮 Generalised Policy Iteration — The Unifying Framework")

            with st.expander("📐 Theory & Formulas — GPI", expanded=False):
                st.markdown(r"""
                #### The GPI Loop

                ```
                V  ←  V₀ (arbitrary)
                π  ←  π₀ (arbitrary)
                Loop forever:
                    EVALUATE: move V closer to v_π (one or more Bellman sweeps)
                    IMPROVE:  move π closer to greedy(V) (one or more greedy steps)
                ```

                Both processes run **continuously** — neither fully completes before the other starts.
                The system stabilises when V = v_π and π is greedy w.r.t. V simultaneously.

                #### The Fixed Point = Optimality

                GPI converges to a stable fixed point where:
                - V is consistent with π: $V(s) = \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V(s')]$
                - π is greedy w.r.t. V: $\pi(s) = \arg\max_a \sum_{s',r}p(s',r|s,a)[r+\gamma V(s')]$

                Both conditions simultaneously imply the **Bellman optimality equation**:
                $$V(s) = \max_a \sum_{s',r}p(s',r|s,a)[r+\gamma V(s')] = v^*(s)$$

                Therefore: V = V* and π = π*.

                #### GPI Instantiations Across All RL Methods

                | Method | Evaluation step | Improvement step | Granularity |
                |--------|----------------|-----------------|-------------|
                | **Policy Iteration** | Full convergence (many sweeps) | Full greedy update (all states) | Coarsest |
                | **Value Iteration** | One Bellman sweep (max operator) | Implicit in max | Medium |
                | **Async DP** | One state update at a time | Implicit in max | Fine |
                | **MC Control** | One episode of returns | ε-greedy over episode | Episode-level |
                | **SARSA** | One TD step | ε-greedy action selection | Finest (step-level) |
                | **Q-Learning** | One TD step (greedy target) | Implicit in max Q | Finest (off-policy) |

                #### The Two Competing Goals

                Evaluation wants V to match current π — pulling V toward v_π.
                Improvement wants π to be greedy w.r.t. V — pushing π away from what V was computed for.

                These are opposing forces, but they stabilise at the optimal solution.
                The analogy: two people pushing a ball up a hill from opposite sides —
                the ball settles at the stable point (the top = optimality).
                """)

            h_gpi = res["h_gpi"]
            n_gpi = len(h_gpi)

            st.markdown(_card("#4527a0","📖","How to read the GPI evolution diagram",
                f"""Shows {n_gpi} outer GPI iterations, each doing 5 evaluation sweeps then one improvement.
                <b>Top row (V):</b> Values become progressively more accurate — notice the values
                getting closer to V* (compare with Value Iteration tab).<br>
                <b>Bottom row (π):</b> Policy arrows become progressively more directed.
                Early: still somewhat random-looking. Later: clean arrows toward terminals.<br>
                <b>The key observation:</b> Even with only 5 (incomplete) evaluation sweeps between
                improvements, the policy still converges to optimal. GPI doesn't require full evaluation!"""),
                unsafe_allow_html=True)

            n_show_gpi = min(n_gpi, 4)
            fig, axes, axl = make_fig(2, n_show_gpi, 5*n_show_gpi, 10)
            for i in range(n_show_gpi):
                it = h_gpi[i]
                col = "#4caf50" if it["stable"] else "#4527a0"
                for ax_row in [axes[0][i], axes[1][i]]:
                    for sp in ax_row.spines.values():
                        sp.set_edgecolor(col); sp.set_linewidth(2)
                plot_V(env, it["V"], f"GPI round {it['outer']} — V",
                       axes[0][i], vmin=-22, vmax=0)
                plot_policy(env, it["pi"], f"GPI round {it['outer']} — π"
                            + (" ✅" if it["stable"] else ""),
                            axes[1][i], color=METHOD_COLORS["GPI"])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Summary comparison all methods
            st.divider()
            st.subheader("📊 All DP Methods — Final Value Comparison")
            st.markdown(_card("#4527a0","📖","How to read the final comparison",
                """All methods estimate V(s). Policy Iteration and Value Iteration both find V* exactly.
                Policy Evaluation finds V_π (worse for the random policy).<br>
                Bars grouped by state — taller (less negative) = better estimate.<br>
                <b>What to look for:</b> Policy Iteration (green) and Value Iteration (orange) should
                produce identical V* values. Policy Evaluation (purple) gives V_π which is always ≤ V*."""),
                unsafe_allow_html=True)

            methods_v = [
                ("Policy Eval (V_π)",   res["V_rand"], METHOD_COLORS["Policy Eval"]),
                ("Policy Iter (V*)",    res["h_pi"][-1]["V"], METHOD_COLORS["Pol Iter"]),
                ("Value Iter (V*)",     res["V_vi"],   METHOD_COLORS["Val Iter"]),
                ("In-place DP (V*)",    res["V_async"], METHOD_COLORS["Async DP"]),
            ]
            fig2, ax2, _ = make_fig(1, 1, 14, 5)
            width = 0.2
            xs = np.arange(env.N_STATES)
            for i, (name, V, col) in enumerate(methods_v):
                ax2.bar(xs + i*width - 0.3, V, width, label=name, color=col, alpha=0.85)
            ax2.set_xlabel("State", color="white", fontsize=11)
            ax2.set_ylabel("V(s)", color="white", fontsize=11)
            ax2.set_title("All DP Methods — V(s) Comparison", color="white", fontweight="bold")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8,
                       bbox_to_anchor=(1.01, 1), loc="upper left")
            ax2.grid(alpha=0.15, axis="y")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            # Convergence costs table
            st.subheader("📋 Computational Cost Comparison")
            rows = [
                {"Method": "Policy Evaluation",
                 "Total sweeps": len(res["h_eval"]),
                 "Iterations": 1,
                 "Converges to": "V_π (random policy)",
                 "Needs explicit policy?": "✅ Yes"},
                {"Method": "Policy Iteration",
                 "Total sweeps": sum(h["eval_sweeps"] for h in res["h_pi"]),
                 "Iterations": len(res["h_pi"]),
                 "Converges to": "V* (optimal)",
                 "Needs explicit policy?": "✅ Yes"},
                {"Method": "Value Iteration",
                 "Total sweeps": len(res["h_vi"]),
                 "Iterations": len(res["h_vi"]),
                 "Converges to": "V* (optimal)",
                 "Needs explicit policy?": "❌ No (implicit)"},
                {"Method": "In-place DP",
                 "Total sweeps": len(res["h_async"]),
                 "Iterations": len(res["h_async"]),
                 "Converges to": "V* (optimal)",
                 "Needs explicit policy?": "❌ No"},
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            render_dp_notes("GPI Framework", "dynamic_programming_gpi_framework")

    else:
        pending_tabs = [
            (tab_eval, "Policy Evaluation", "dynamic_programming_policy_evaluation"),
            (tab_impr, "Policy Improvement", "dynamic_programming_policy_improvement"),
            (tab_pi, "Policy Iteration", "dynamic_programming_policy_iteration"),
            (tab_vi, "Value Iteration", "dynamic_programming_value_iteration"),
            (tab_async, "Async DP", "dynamic_programming_async_dp"),
            (tab_gpi, "GPI Framework", "dynamic_programming_gpi_framework"),
        ]
        for t, note_title, note_slug in pending_tabs:
            with t:
                st.info("👈 Click **Run All Methods** in the sidebar to generate all charts.")
                render_dp_notes(note_title, note_slug)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 7 — METHOD GUIDE (always visible)
    # ═══════════════════════════════════════════════════════════════════════
    with tab_guide:
        st.subheader("📚 Complete DP Method Reference")

        with st.expander("🗺️ Which DP method should I use? — Decision Guide", expanded=True):
            st.markdown("""
            ```
            DO I HAVE A COMPLETE MODEL p(s',r|s,a)?
              ├─ NO → Use MC or TD methods (DP doesn't apply)
              └─ YES → Continue below ↓
                   │
                   ├─ Do I need PREDICTION (evaluate a fixed policy)?
                   │    └─ Policy Evaluation: iterative Bellman sweeps until convergence
                   │
                   └─ Do I need CONTROL (find the optimal policy)?
                        │
                        ├─ Small state space, full evaluation affordable?
                        │    └─ Policy Iteration: few iterations, each fully converged
                        │
                        ├─ Large state space, want simpler implementation?
                        │    └─ Value Iteration: one sweep = one Bellman optimality update
                        │
                        └─ Want to speed up convergence via update order?
                             └─ Async / In-place DP: prioritised sweeping
            ```
            """)

        entries = [
            {
                "icon": "🔄", "name": "Policy Evaluation (Iterative)",
                "formula": r"$V_{k+1}(s) \leftarrow \sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$",
                "what": "Estimates V_π for a FIXED policy π by repeatedly applying the Bellman expectation equation. Each sweep makes every state's value more consistent with its neighbours until convergence.",
                "when": "When you need to evaluate a specific policy (not improve it). As the inner loop of Policy Iteration. When you want to understand the value of a given strategy before changing it.",
                "pros": "✅ Guaranteed convergence to exact V_π | ✅ Simple update rule | ✅ No maximisation (lower complexity) | ✅ Directly interpretable: V_π(s) = expected steps to terminal",
                "cons": "❌ Does not improve the policy | ❌ Requires full model p(s',r|s,a) | ❌ Expensive for large state spaces | ❌ Must redo if policy changes",
                "relation": "The 'E' in GPI. Used as the inner loop of Policy Iteration. Replace weighted-sum with max → Value Iteration. Replace full sweeps with single steps → TD prediction.",
            },
            {
                "icon": "⬆️", "name": "Policy Improvement",
                "formula": r"$\pi'(s) \leftarrow \arg\max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_\pi(s')]$",
                "what": "Given a converged V_π, produces a strictly better policy π' by choosing the action with the highest one-step expected return. The Policy Improvement Theorem guarantees π' ≥ π.",
                "when": "Always paired with Policy Evaluation. As the 'I' half of GPI. When you have V and want to extract or improve a policy from it.",
                "pros": "✅ Monotone improvement guaranteed | ✅ One pass (no iteration) | ✅ Simple greedy argmax | ✅ Foundation of all control methods",
                "cons": "❌ Requires V_π to be accurate first | ❌ One-step lookahead (myopic without future context) | ❌ Produces deterministic policy (no exploration)",
                "relation": "The 'I' in GPI. Paired with Policy Evaluation in Policy Iteration. When γ=0: becomes pure greedy. When combined with approximate V: becomes approximate greedy (used in DQN, SARSA).",
            },
            {
                "icon": "🔁", "name": "Policy Iteration",
                "formula": r"Alternate: $V \leftarrow \text{PolicyEval}(\pi)$, then $\pi \leftarrow \text{greedy}(V)$, until stable",
                "what": "The complete control algorithm: alternate full Policy Evaluation and full Policy Improvement until the policy stops changing. Converges to π* in finitely many iterations.",
                "when": "When full evaluation before each improvement is affordable. When you want the theoretically cleanest algorithm. For problems where the number of policy iterations is small.",
                "pros": "✅ Optimal convergence with fewest policy 'iterations' (typically 3–5) | ✅ Monotone improvement | ✅ Clean theory | ✅ π* guaranteed",
                "cons": "❌ Each iteration is expensive (full evaluation to convergence) | ❌ Requires explicit policy storage | ❌ Full model required | ❌ Synchronous updates expensive for large S",
                "relation": "Full GPI. Value Iteration is the limiting case where evaluation = 1 sweep. MC Control and TD Control are GPI with sample-based evaluation.",
            },
            {
                "icon": "⚡", "name": "Value Iteration",
                "formula": r"$V_{k+1}(s) \leftarrow \max_a\sum_{s',r}p(s',r|s,a)[r+\gamma V_k(s')]$",
                "what": "Directly iterates the Bellman optimality equation — no explicit policy maintained. Each sweep takes the MAX (not weighted sum) over actions. After convergence, extract π* by one greedy step.",
                "when": "Default choice for DP control. When you don't need the policy during iteration. When Value Iteration's simpler structure (no policy array) is preferable.",
                "pros": "✅ Simpler than Policy Iteration (no explicit policy) | ✅ Converges to V* | ✅ No inner loop needed | ✅ Foundation of Q-Learning (tabular)",
                "cons": "❌ More sweeps than Policy Iteration (no full evaluation between improvements) | ❌ Full model required | ❌ All states updated each sweep (synchronous bottleneck)",
                "relation": "GPI with 1-sweep evaluation. Max operator = implicit policy improvement. Tabular Q-Learning is the sample-based approximation of Value Iteration. ADP, RTDP are asynchronous extensions.",
            },
            {
                "icon": "🔀", "name": "Asynchronous / In-place DP",
                "formula": r"In-place: $V(s) \leftarrow \max_a\sum_{s',r}p[r+\gamma V(s')]$ immediately, same array",
                "what": "Updates states out-of-order or asynchronously — either sequentially (in-place, using fresh values immediately) or by prioritising states with the largest Bellman error.",
                "when": "When synchronous sweeps are too expensive (large S). When the state space has structure that prioritised sweeping can exploit. In real-time DP systems.",
                "pros": "✅ Often faster convergence than synchronous DP | ✅ Can focus on most-changed states | ✅ Memory efficient (one array) | ✅ Bridges toward online RL",
                "cons": "❌ Order-dependent convergence speed | ❌ More complex implementation | ❌ Full model still required | ❌ Prioritised sweeping requires tracking Bellman errors",
                "relation": "Asynchronous extension of Value Iteration. Prioritised sweeping is used in Dyna-Q. In-place DP is the natural implementation when memory is limited.",
            },
            {
                "icon": "🔮", "name": "Generalised Policy Iteration (GPI)",
                "formula": r"Any interleaving of: $\text{Eval}(V \to v_\pi)$ and $\text{Improve}(\pi \to \text{greedy}(V))$",
                "what": "A conceptual framework describing ALL RL algorithms as some form of alternating evaluation and improvement — regardless of granularity (full, one sweep, one episode, one step).",
                "when": "Always — it's a lens to understand any RL algorithm, not an algorithm itself.",
                "pros": "✅ Unifies DP, MC, TD, Actor-Critic, MCTS under one framework | ✅ Explains convergence | ✅ Guides algorithm design",
                "cons": "❌ Not an algorithm by itself — requires specifying granularity | ❌ Convergence theory varies by instantiation",
                "relation": "The parent framework of: Policy Iteration, Value Iteration, SARSA, Q-Learning, Actor-Critic, PPO, A3C, DQN. Any algorithm with a 'critic' (evaluation) and 'actor' (improvement) is GPI.",
            },
        ]

        for e in entries:
            with st.expander(f"{e['icon']} {e['name']}", expanded=False):
                st.markdown(f"**Core formula:** {e['formula']}")
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

        **Q: Why does DP need the full model? Can't it just sample like MC/TD?**

        DP's update rule sums over ALL possible next states: $\sum_{s',r}p(s',r|s,a)[\ldots]$.
        Computing this exact sum requires knowing p(s',r|s,a) for every transition.
        MC and TD instead *sample* one (s', r) from the real environment — no model needed.
        The price: MC/TD are noisier (sample variance); DP is exact but requires the model.

        ---

        **Q: Why does Policy Iteration converge in 3 iterations but Value Iteration needs 100+ sweeps?**

        Policy Iteration runs evaluation to *full convergence* before each improvement.
        This gives a very accurate V_π to base the improvement on, so the policy jumps
        far toward optimal each time. Only 3 jumps needed.

        Value Iteration does 1-sweep evaluation before each improvement — very approximate.
        Each sweep nudges V toward V* by a small amount. Needs many nudges but each is cheap.

        Total work is roughly similar. For this 4×4 grid, both are fast.

        ---

        **Q: Is Value Iteration just Policy Iteration with evaluation = 1 sweep?**

        Exactly! This is the key insight of S&B §4.4:

        > "Policy Iteration consists of making evaluation sweeps and improvement steps.
        > Value Iteration is the limiting case where we truncate the evaluation to just one sweep."

        Formally: Policy Iteration with k=1 evaluation sweeps per improvement = Value Iteration.
        This continuum (k=1,2,...,∞) is the spectrum between Value Iteration and Policy Iteration.

        ---

        **Q: How does DP connect to Q-Learning and DQN?**

        Q-Learning is the sample-based analogue of Value Iteration:

        | DP (Value Iteration) | Q-Learning (TD) |
        |---------------------|----------------|
        | $Q(s,a) = \sum_{s',r}p[\ldots]$ | $Q(s,a) \mathrel{+}= \alpha[r + \max Q(s',\cdot) - Q(s,a)]$ |
        | Uses full model sum | Uses one sample (s', r) |
        | Exact update | Noisy but model-free |
        | All states each sweep | One state per step |

        DQN extends Q-Learning with neural networks instead of tables —
        same Bellman optimality target, same GPI structure, just with function approximation.

        ---

        **Q: What is the "curse of dimensionality" in DP?**

        DP must update ALL states every sweep: $O(|S| \times |A|)$ per sweep.
        For a simple grid: 16 states × 4 actions = 64 operations per sweep — trivial.
        For real problems: a robot with 10 continuous joint angles discretised to 100 values each
        → $100^{10} = 10^{20}$ states — impossible to enumerate.

        Solutions: Function approximation (neural nets for V or Q), sampling (MC/TD),
        or model-learning + planning (Dyna, model-based RL).
        """)
        render_dp_notes("Method Guide", "dynamic_programming_method_guide")


if __name__ == "__main__":
    main()
