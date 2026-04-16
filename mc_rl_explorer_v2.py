"""
╔══════════════════════════════════════════════════════════════╗
║     Monte Carlo RL Explorer — Professional Edition           ║
║     All MC methods from Sutton & Barto Chapter 5             ║
║     Environment: 5×5 Stochastic Gridworld (No Blackjack)     ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import pandas as pd
from collections import defaultdict
import textwrap, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL STYLES
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MC RL Explorer",
    layout="wide",
    page_icon="🎲",
    initial_sidebar_state="expanded",
)

# Professional dark theme with scientific / textbook aesthetic
STYLES = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@300;400;600;700&family=Space+Mono&display=swap" rel="stylesheet">
<style>
/* ─── Root variables ──────────────────────────────────────── */
:root {
  --bg:       #070b14;
  --surface:  #0e1420;
  --card:     #131928;
  --border:   #1e2d42;
  --border2:  #253449;
  --txt:      #cdd6f4;
  --txt2:     #7a8ba6;
  --accent1:  #5b9cf6;   /* electric blue */
  --accent2:  #a78bfa;   /* violet */
  --accent3:  #34d399;   /* emerald */
  --accent4:  #fb923c;   /* amber */
  --accent5:  #f87171;   /* rose */
  --accent6:  #22d3ee;   /* cyan */
  --mono:     'IBM Plex Mono', monospace;
  --sans:     'Sora', sans-serif;
}

/* ─── App shell ───────────────────────────────────────────── */
.stApp { background: var(--bg); font-family: var(--sans); }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--txt) !important; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* ─── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 4px 6px;
  gap: 2px;
}
.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 7px;
  color: var(--txt2);
  font-family: var(--sans);
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  padding: 7px 14px;
  border: none;
  transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { background: var(--card); color: var(--txt); }
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #1e3a6e 0%, #1a2d54 100%) !important;
  color: var(--accent1) !important;
  border: 1px solid #2a4a7f !important;
  box-shadow: 0 0 12px rgba(91,156,246,0.15);
}

/* ─── Cards ───────────────────────────────────────────────── */
.mc-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem;
}
.mc-card-accent {
  border-left: 3px solid var(--accent1);
}

/* ─── Formula blocks ──────────────────────────────────────── */
.formula-box {
  background: #0a1628;
  border: 1px solid #1e3a6e;
  border-radius: 10px;
  padding: 1.2rem 1.6rem;
  margin: 0.8rem 0;
  font-family: var(--mono);
  color: #93c5fd;
}

/* ─── Pseudocode ──────────────────────────────────────────── */
.pseudocode {
  background: #0c1a2e;
  border: 1px solid var(--border2);
  border-radius: 10px;
  padding: 1.2rem 1.6rem;
  font-family: var(--mono);
  font-size: 0.8rem;
  line-height: 1.9;
  color: #b0c4de;
  white-space: pre;
  overflow-x: auto;
}
.pseudocode .kw  { color: #7dd3fc; }   /* keyword  */
.pseudocode .fn  { color: #a78bfa; }   /* function */
.pseudocode .cm  { color: #4a6a8a; font-style: italic; }  /* comment */
.pseudocode .nu  { color: #34d399; }   /* number   */
.pseudocode .op  { color: #fb923c; }   /* operator */

/* ─── Method badge ────────────────────────────────────────── */
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-family: var(--mono);
}
.badge-blue   { background:#1e3a6e; color:#93c5fd; border:1px solid #2a4a7f; }
.badge-violet { background:#2d1f5e; color:#c4b5fd; border:1px solid #4c3b8a; }
.badge-green  { background:#0f3028; color:#6ee7b7; border:1px solid #1a5040; }
.badge-amber  { background:#3d2005; color:#fcd34d; border:1px solid #6b3a08; }
.badge-rose   { background:#3d0f0f; color:#fca5a5; border:1px solid #7f1d1d; }
.badge-cyan   { background:#062830; color:#67e8f9; border:1px solid #0e4d5c; }

/* ─── Section header ──────────────────────────────────────── */
.section-hdr {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 1.8rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.section-hdr h3 { margin: 0; color: var(--txt); font-size: 1.1rem; font-weight: 700; }

/* ─── Metrics ─────────────────────────────────────────────── */
div[data-testid="metric-container"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label { color: var(--txt2) !important; font-size: 0.78rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent1) !important; font-family: var(--mono) !important; }

/* ─── Slider & widgets ────────────────────────────────────── */
.stSlider > div > div > div { background: var(--accent1) !important; }
.stButton > button {
  background: linear-gradient(135deg, #1e3a6e, #162e57);
  border: 1px solid #2a4a7f;
  color: #93c5fd;
  font-family: var(--sans);
  font-weight: 600;
  border-radius: 8px;
  transition: all 0.2s;
}
.stButton > button:hover {
  background: linear-gradient(135deg, #2a4d91, #1e3a6e);
  box-shadow: 0 0 16px rgba(91,156,246,0.25);
}

/* ─── Expander ────────────────────────────────────────────── */
details { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 0.5rem 1rem; }
summary { color: var(--txt); font-weight: 600; cursor: pointer; }

/* ─── Dataframe ───────────────────────────────────────────── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* ─── Info / warning boxes ────────────────────────────────── */
div[data-testid="stInfo"]    { background: #0a1628 !important; border-left: 3px solid var(--accent1) !important; }
div[data-testid="stSuccess"] { background: #0a1e12 !important; border-left: 3px solid var(--accent3) !important; }
div[data-testid="stWarning"] { background: #1e1200 !important; border-left: 3px solid var(--accent4) !important; }

/* ─── Divider ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ─── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# COLOUR PALETTE FOR MATPLOTLIB
# ══════════════════════════════════════════════════════════════

P = dict(
    bg      = "#070b14",
    surface = "#0e1420",
    card    = "#131928",
    border  = "#1e2d42",
    txt     = "#cdd6f4",
    txt2    = "#7a8ba6",
    blue    = "#5b9cf6",
    violet  = "#a78bfa",
    green   = "#34d399",
    amber   = "#fb923c",
    rose    = "#f87171",
    cyan    = "#22d3ee",
    yellow  = "#fbbf24",
    pink    = "#f472b6",
)

METHOD_COLORS = [P["blue"], P["violet"], P["green"], P["rose"],
                 P["cyan"],  P["amber"],  P["yellow"], P["pink"], "#e879f9"]

RL_CMAP = LinearSegmentedColormap.from_list("rl", ["#7f1d1d","#b45309","#fbbf24","#064e3b"])

def mpl_defaults(fig, axes_flat):
    fig.patch.set_facecolor(P["bg"])
    for ax in axes_flat:
        ax.set_facecolor(P["card"])
        ax.tick_params(colors=P["txt2"], labelsize=8)
        ax.xaxis.label.set_color(P["txt2"])
        ax.yaxis.label.set_color(P["txt2"])
        ax.title.set_color(P["txt"])
        for spine in ax.spines.values():
            spine.set_edgecolor(P["border"])

def make_fig(nrows=1, ncols=1, w=12, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    flat = np.array(axes).flatten().tolist()
    mpl_defaults(fig, flat)
    fig.subplots_adjust(hspace=0.4, wspace=0.35)
    return fig, axes, flat


# ══════════════════════════════════════════════════════════════
# GRIDWORLD ENVIRONMENT
# ══════════════════════════════════════════════════════════════

class GridWorld:
    """
    5×5 Stochastic Gridworld — designed for MC method comparison.

    Layout (row, col):
        Start  : (0,0)   → blue
        Goal   : (4,4)   → +10,  terminal
        Trap   : (2,2)   → −5,   terminal
        Walls  : {(1,1),(1,3),(3,1),(3,3)}  — agent bounces back
        Step   : −0.1    per transition
        Slip   : P(random action) = slip_prob

    The environment is episodic (terminates at goal or trap).
    Multiple paths exist: agents can go around walls via (0,4)→(4,4) or through centre.
    The trap at (2,2) forces meaningful policy differentiation.
    """
    ACTIONS  = [0, 1, 2, 3]                            # up right down left
    SYMBOLS  = ["↑", "→", "↓", "←"]
    DELTAS   = [(-1,0),(0,1),(1,0),(0,-1)]

    def __init__(self, size=5, slip_prob=0.1):
        self.size      = size
        self.slip_prob = slip_prob
        self.start     = (0,0)
        self.goal      = (4,4)
        self.trap      = (2,2)
        self.walls     = frozenset([(1,1),(1,3),(3,1),(3,3)])
        self.n_states  = size * size
        self.n_actions = 4

    def s2i(self, s): return s[0]*self.size + s[1]
    def i2s(self, i): return (i//self.size, i%self.size)
    def is_terminal(self, s): return s in (self.goal, self.trap)

    def step(self, s, a):
        if self.is_terminal(s):
            return s, 0.0, True
        if np.random.random() < self.slip_prob:
            a = np.random.randint(4)
        dr, dc = self.DELTAS[a]
        ns = (s[0]+dr, s[1]+dc)
        if not (0 <= ns[0] < self.size and 0 <= ns[1] < self.size):
            ns = s
        if ns in self.walls:
            ns = s
        if   ns == self.goal: r = 10.0
        elif ns == self.trap:  r = -5.0
        else:                  r = -0.1
        return ns, r, self.is_terminal(ns)

    def generate_episode(self, policy, start_override=None, action_override=None, max_steps=300):
        """
        Generate an episode. Optionally override start state (for Exploring Starts).
        action_override: force first action (Exploring Starts requirement).
        """
        episode = []
        s = start_override if start_override is not None else self.start
        first = True
        for _ in range(max_steps):
            p = policy[s]
            if first and action_override is not None:
                a = action_override
                first = False
            else:
                a = int(p) if np.isscalar(p) else int(np.random.choice(4, p=p))
            ns, r, done = self.step(s, a)
            episode.append((s, a, r))
            s = ns
            if done:
                break
        return episode

    def uniform_policy(self):
        return {self.i2s(i): np.ones(4)/4 for i in range(self.n_states)}

    def eps_greedy_policy(self, Q, eps):
        policy = {}
        for i in range(self.n_states):
            s = self.i2s(i)
            probs = np.full(4, eps/4)
            probs[np.argmax(Q[s])] += 1.0 - eps
            policy[s] = probs
        return policy

    def greedy_policy(self, Q):
        return {self.i2s(i): int(np.argmax(Q[self.i2s(i)])) for i in range(self.n_states)}

    def valid_states(self):
        return [self.i2s(i) for i in range(self.n_states)
                if self.i2s(i) not in self.walls]

    def non_terminal_states(self):
        return [s for s in self.valid_states() if not self.is_terminal(s)]


# ══════════════════════════════════════════════════════════════
# ① FIRST-VISIT MC PREDICTION   (S&B §5.1)
# ══════════════════════════════════════════════════════════════

def mc_first_visit(env, policy, n_episodes, gamma):
    """
    Algorithm (S&B §5.1, Box 'First-visit MC prediction'):
      For each episode:
        Generate S₀,A₀,R₁,…,Sₜ using π
        G ← 0
        For t = T-1 downto 0:
          G ← γG + Rₜ₊₁
          Unless Sₜ appears in S₀,…,Sₜ₋₁:   ← FIRST VISIT CHECK
            Returns(Sₜ).append(G)
            V(Sₜ) ← average(Returns(Sₜ))
    """
    V, returns = defaultdict(float), defaultdict(list)
    history, variance_log = [], defaultdict(list)

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        G, visited = 0.0, set()
        for s, a, r in reversed(episode):
            G = gamma*G + r
            if s not in visited:
                visited.add(s)
                returns[s].append(G)
                V[s] = float(np.mean(returns[s]))
                variance_log[s].append(V[s])

        if (ep+1) % max(1, n_episodes//30) == 0:
            history.append(dict(V))

    return V, history, variance_log


# ══════════════════════════════════════════════════════════════
# ② EVERY-VISIT MC PREDICTION
# ══════════════════════════════════════════════════════════════

def mc_every_visit(env, policy, n_episodes, gamma):
    """
    Like First-Visit but counts every occurrence of Sₜ, not just the first.
    Slight within-episode correlation but more data.
    """
    V, returns = defaultdict(float), defaultdict(list)
    history = []

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        G = 0.0
        for s, a, r in reversed(episode):
            G = gamma*G + r
            returns[s].append(G)           # ← no first-visit check
            V[s] = float(np.mean(returns[s]))

        if (ep+1) % max(1, n_episodes//30) == 0:
            history.append(dict(V))

    return V, history


# ══════════════════════════════════════════════════════════════
# ③ MC WITH EXPLORING STARTS   (S&B §5.3)   ← ADDED METHOD
# ══════════════════════════════════════════════════════════════

def mc_exploring_starts(env, n_episodes, gamma):
    """
    Algorithm (S&B §5.3, 'Monte Carlo ES — Exploring Starts'):
      Guarantee: every (s,a) pair has nonzero probability of being selected at episode start.

      Initialize: Q(s,a) ← 0, π(s) ← any, Returns(s,a) ← []
      For each episode:
        Choose S₀ ~ uniform(all non-terminal states)  ← EXPLORING START
        Choose A₀ ~ uniform(4 actions)                ← RANDOM FIRST ACTION
        Generate episode from (S₀,A₀) following π
        G ← 0
        For t = T-1 downto 0:
          G ← γG + Rₜ₊₁
          If (Sₜ,Aₜ) first visit this episode:
            Returns(Sₜ,Aₜ).append(G)
            Q(Sₜ,Aₜ) ← average(Returns)
            π(Sₜ) ← argmax_a Q(Sₜ,a)   ← GREEDY (no ε needed!)

    Key insight: Exploring Starts removes need for ε-greedy.
                 Converges to the TRUE optimal π*, not just best ε-soft policy.
    """
    Q       = defaultdict(lambda: np.zeros(4))
    returns = defaultdict(list)
    policy  = env.uniform_policy()          # start with uniform; will become greedy
    episode_rewards = []
    states  = env.non_terminal_states()

    for ep in range(n_episodes):
        # ── EXPLORING START: random (s₀, a₀) ────────────────────
        s0 = states[np.random.randint(len(states))]
        a0 = np.random.randint(4)

        episode = env.generate_episode(policy, start_override=s0, action_override=a0)
        episode_rewards.append(sum(r for _, _, r in episode))

        G, visited_sa = 0.0, set()
        for s, a, r in reversed(episode):
            G = gamma*G + r
            if (s,a) not in visited_sa:
                visited_sa.add((s,a))
                returns[(s,a)].append(G)
                Q[s][a] = float(np.mean(returns[(s,a)]))
                # ── GREEDY policy improvement (no ε) ─────────────
                best = int(np.argmax(Q[s]))
                probs = np.zeros(4); probs[best] = 1.0
                policy[s] = probs

    V_es = {s: float(np.max(Q[s])) for s in Q}
    return Q, policy, V_es, episode_rewards


# ══════════════════════════════════════════════════════════════
# ④ ON-POLICY MC CONTROL (ε-greedy)   (S&B §5.4)
# ══════════════════════════════════════════════════════════════

def mc_control_on_policy(env, n_episodes, eps, gamma):
    """
    Algorithm (S&B §5.4):
      Same as MC ES but replaces Exploring Starts with ε-soft policy.
      ε ensures every action has P ≥ ε/|A|.
      Converges to best ε-soft policy (slightly suboptimal vs ES).
    """
    Q       = defaultdict(lambda: np.zeros(4))
    returns = defaultdict(list)
    policy  = env.uniform_policy()
    episode_rewards = []

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        episode_rewards.append(sum(r for _,_,r in episode))
        G, visited_sa = 0.0, set()

        for s, a, r in reversed(episode):
            G = gamma*G + r
            if (s,a) not in visited_sa:
                visited_sa.add((s,a))
                returns[(s,a)].append(G)
                Q[s][a] = float(np.mean(returns[(s,a)]))

        policy = env.eps_greedy_policy(Q, eps)

    V_on = {s: float(np.max(Q[s])) for s in Q}
    return Q, policy, V_on, episode_rewards


# ══════════════════════════════════════════════════════════════
# ⑤ OFF-POLICY EVALUATION — ORDINARY IS   (S&B §5.5)
# ══════════════════════════════════════════════════════════════

def _ratio(target, behavior, s, a):
    b = behavior[s][a] if isinstance(behavior[s], np.ndarray) else float(behavior[s]==a)
    t = target[s][a]   if isinstance(target[s],  np.ndarray) else float(int(target[s])==a)
    return t/b if b > 1e-12 else 0.0

def mc_ordinary_is(env, target, behavior, n_episodes, gamma):
    """
    Ordinary (simple) IS  — V̂_ois(s) = (1/n) Σ ρᵢ·Gᵢ
    UNBIASED but extremely high variance.
    Variance grows exponentially with episode length T.
    """
    returns_w = defaultdict(list)
    V = defaultdict(float)

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior)
        G, rho, visited = 0.0, 1.0, set()
        for s, a, r in reversed(episode):
            G   = gamma*G + r
            rho *= _ratio(target, behavior, s, a)
            if rho < 1e-10: break
            if s not in visited:
                visited.add(s)
                returns_w[s].append(rho*G)
                V[s] = float(np.mean(returns_w[s]))

    return V


def mc_weighted_is(env, target, behavior, n_episodes, gamma):
    """
    Weighted IS  — V̂_wis(s) = Σ(ρᵢGᵢ) / Σρᵢ
    Implemented with incremental numerically-stable update.
    BIASED but consistent; variance dramatically lower than ordinary IS.
    """
    V, C = defaultdict(float), defaultdict(float)

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior)
        G, W, visited = 0.0, 1.0, set()
        for s, a, r in reversed(episode):
            G  = gamma*G + r
            W *= _ratio(target, behavior, s, a)
            if W < 1e-10: break
            if s not in visited:
                visited.add(s)
                C[s] += W
                V[s] += (W/C[s]) * (G - V[s])

    return V


# ══════════════════════════════════════════════════════════════
# ⑥ OFF-POLICY MC CONTROL   (S&B §5.7)   ← ADDED METHOD
# ══════════════════════════════════════════════════════════════

def mc_off_policy_control(env, n_episodes, gamma):
    """
    Algorithm (S&B §5.7, 'Off-policy MC Control'):
      behavior policy b = ε-soft (explores)
      target  policy π = greedy (what we want to learn)

      Q(s,a) ← 0 for all s,a
      C(s,a) ← 0 for all s,a   (cumulative IS weights)
      π(s)   ← argmax_a Q(s,a) for all s

      For each episode generated by behavior policy b:
        G ← 0,  W ← 1
        For t = T-1 downto 0:
          G ← γG + Rₜ₊₁
          C(Sₜ,Aₜ) ← C(Sₜ,Aₜ) + W
          Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + (W/C(Sₜ,Aₜ))[G − Q(Sₜ,Aₜ)]  ← WIS update
          π(Sₜ) ← argmax_a Q(Sₜ,a)
          If Aₜ ≠ π(Sₜ): break   ← episode irrelevant after policy divergence
          W ← W / b(Aₜ|Sₜ)

    Key insight: W is updated BEFORE the π≠a check, so we
                 multiply only when the episode is still "useful."
                 This allows convergence to the TRUE optimal policy π*.
    """
    Q = defaultdict(lambda: np.zeros(4))
    C = defaultdict(lambda: np.zeros(4))   # per action-pair cumulative weights
    # Deterministic greedy target policy
    pi = {env.i2s(i): int(np.argmax(Q[env.i2s(i)])) for i in range(env.n_states)}
    # Behaviour: ε-soft (fixed ε=0.2)
    eps_b = 0.2
    episode_rewards = []

    for ep in range(n_episodes):
        # generate episode using behavior (eps-soft over CURRENT Q)
        b_policy = env.eps_greedy_policy(Q, eps_b)
        episode  = env.generate_episode(b_policy)
        episode_rewards.append(sum(r for _,_,r in episode))

        G, W = 0.0, 1.0

        for s, a, r in reversed(episode):
            G  = gamma*G + r
            C[s][a] += W
            Q[s][a] += (W / C[s][a]) * (G - Q[s][a])
            # Update greedy target policy
            pi[s] = int(np.argmax(Q[s]))

            if a != pi[s]:      # ← episode no longer consistent with target
                break
            b_val = b_policy[s][a] if isinstance(b_policy[s], np.ndarray) else (1.0 - eps_b + eps_b/4)
            W /= (b_val + 1e-12)

    V_op = {s: float(np.max(Q[s])) for s in Q}
    return Q, pi, V_op, episode_rewards


# ══════════════════════════════════════════════════════════════
# ⑦ INCREMENTAL MC   (S&B §5.6)
# ══════════════════════════════════════════════════════════════

def mc_incremental(env, policy, n_episodes, gamma):
    """
    Incremental update rule (equivalent to batch averaging):
      V(S) ← V(S) + (1/N(S)) · [G − V(S)]

    For constant step-size α: V(S) ← V(S) + α·[G − V(S)]
    → exponentially decays old returns (useful for non-stationary).
    """
    V  = defaultdict(float)
    N  = defaultdict(int)
    val_log     = defaultdict(list)
    var_history = []

    for ep in range(n_episodes):
        episode = env.generate_episode(policy)
        G, visited = 0.0, set()
        for s, a, r in reversed(episode):
            G = gamma*G + r
            if s not in visited:
                visited.add(s)
                N[s] += 1
                V[s] += (G - V[s]) / N[s]
                val_log[s].append(V[s])

        if (ep+1) % max(1, n_episodes//30) == 0:
            var_history.append(
                float(np.mean([np.var(val_log[s]) for s in val_log if len(val_log[s])>1]))
            )

    return V, var_history, val_log


# ══════════════════════════════════════════════════════════════
# ⑧ PER-DECISION IS   (S&B §5.8)
# ══════════════════════════════════════════════════════════════

def mc_per_decision_is(env, target, behavior, n_episodes, gamma):
    """
    Per-Decision IS decomposes the product:

      V̂^pd(s_t) = Σ_{k=t}^{T} γ^{k-t} · (Π_{j=t}^{k} ρ_j) · R_{k+1}

    Each reward Rₖ₊₁ is weighted only by ratios UP TO step k, not the full
    episode. This provably reduces variance vs ordinary IS.
    """
    returns = defaultdict(list)
    V = defaultdict(float)

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior)
        T = len(episode)

        for t in range(T):
            s_t = episode[t][0]
            G_pd, rho_cum, valid = 0.0, 1.0, True
            for k in range(t, T):
                sk, ak, rk = episode[k]
                rho_cum *= _ratio(target, behavior, sk, ak)
                if rho_cum < 1e-10:
                    valid = False; break
                G_pd += (gamma**(k-t)) * rho_cum * rk

            if valid:
                returns[s_t].append(G_pd)

        for s in set(e[0] for e in episode):
            if returns[s]:
                V[s] = float(np.mean(returns[s]))

    return V


# ══════════════════════════════════════════════════════════════
# ⑨ DISCOUNTING-AWARE IS   (S&B §5.9)
# ══════════════════════════════════════════════════════════════

def mc_discounting_aware_is(env, target, behavior, n_episodes, gamma):
    """
    Discounting-Aware IS (Sutton & Barto §5.9):

    Decomposes the discounted return into a sum of 'flat' partial returns:
      G_t:h = Σ_{k=t}^{h-1} γ^{k-t} R_{k+1}

    The full return:
      Gₜ = (1−γ) Σ_{h=t+1}^{T-1} γ^{h-t-1} G_t:h + γ^{T-t} G_t:T

    Each flat return G_t:h is weighted by the IS ratio only to step h.
    When γ<1, distant partial returns contribute less, so their potentially
    large IS ratios are downweighted → LOWEST variance among IS methods.
    """
    V, C = defaultdict(float), defaultdict(float)

    for _ in range(n_episodes):
        episode = env.generate_episode(behavior)
        T = len(episode)

        for t in range(T):
            s_t = episode[t][0]
            G, W, phi = 0.0, 1.0, 0.0
            for k in range(t, T):
                sk, ak, rk = episode[k]
                W   *= _ratio(target, behavior, sk, ak)
                if W < 1e-10: break
                gk   = gamma**(k-t)
                phi += (1.0-gamma)*gk*W
                G   += gk*rk
            phi += (gamma**(T-t))*W

            if phi > 1e-12:
                C[s_t]  += phi
                V[s_t]  += (phi/C[s_t]) * (G - V[s_t])

    return V


# ══════════════════════════════════════════════════════════════
# VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════

def _draw_grid_background(ax, env):
    """Fill grid cells with base colours."""
    for i in range(env.size):
        for j in range(env.size):
            s = (i,j)
            if   s in env.walls:   c = "#0d1022"
            elif s == env.goal:    c = "#052e16"
            elif s == env.trap:    c = "#450a0a"
            elif s == env.start:   c = "#0c1a3d"
            else:                  c = P["card"]
            ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,color=c,zorder=1,
                                       ec=P["border"],lw=0.8))

def plot_value_heatmap(env, V, title, ax, vmin=-5, vmax=10):
    grid = np.full((env.size, env.size), np.nan)
    for i in range(env.n_states):
        s = env.i2s(i)
        if s not in env.walls:
            grid[s] = V.get(s, 0.0)
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=RL_CMAP, vmin=vmin, vmax=vmax, aspect="equal", zorder=2, alpha=0.85)

    _draw_grid_background(ax, env)

    for i in range(env.size):
        for j in range(env.size):
            s = (i,j)
            if s in env.walls:
                ax.text(j,i,"▪",ha="center",va="center",color="#2a3050",fontsize=14,zorder=4)
                continue
            val = V.get(s, 0.0)
            lum = (val-vmin)/(vmax-vmin)
            tc = "white" if lum<0.35 or lum>0.7 else "#1a1a2e"
            if s == env.goal:
                ax.text(j,i,"★\n+10",ha="center",va="center",fontsize=7,color="#6ee7b7",fontweight="bold",zorder=5)
            elif s == env.trap:
                ax.text(j,i,"✗\n−5",ha="center",va="center",fontsize=7,color="#fca5a5",fontweight="bold",zorder=5)
            elif s == env.start:
                ax.text(j,i,f"S\n{val:.1f}",ha="center",va="center",fontsize=6.5,color="#93c5fd",fontweight="bold",zorder=5)
            else:
                ax.text(j,i,f"{val:.2f}",ha="center",va="center",fontsize=7,color=tc,zorder=5)

    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.ax.tick_params(colors=P["txt2"], labelsize=6)
    cb.outline.set_edgecolor(P["border"])
    ax.set_xticks(range(env.size)); ax.set_yticks(range(env.size))
    ax.set_xticklabels([f"c{i}" for i in range(env.size)], fontsize=6.5)
    ax.set_yticklabels([f"r{i}" for i in range(env.size)], fontsize=6.5)
    ax.set_title(title, color=P["txt"], fontsize=8.5, fontweight="bold", pad=6)


def plot_policy_arrows(env, policy, title, ax):
    ARROW = {0:(0,-0.35), 1:(0.35,0), 2:(0,0.35), 3:(-0.35,0)}
    ax.set_xlim(-0.5,env.size-.5); ax.set_ylim(-0.5,env.size-.5); ax.set_aspect("equal")
    _draw_grid_background(ax, env)

    for i in range(env.size):
        for j in range(env.size):
            s=(i,j)
            if s in env.walls:
                ax.text(j,i,"▪",ha="center",va="center",color="#2a3050",fontsize=14,zorder=4)
            elif s==env.goal:
                ax.text(j,i,"★\nGOAL",ha="center",va="center",fontsize=7.5,color="#6ee7b7",fontweight="bold",zorder=4)
            elif s==env.trap:
                ax.text(j,i,"✗\nTRAP",ha="center",va="center",fontsize=7.5,color="#fca5a5",fontweight="bold",zorder=4)
            else:
                p = policy.get(s, 0)
                a = int(np.argmax(p)) if isinstance(p, np.ndarray) else int(p)
                dc,dr = ARROW[a]
                col = P["blue"] if s==env.start else P["violet"]
                ax.annotate("",xy=(j+dc,i+dr),xytext=(j,i),
                            arrowprops=dict(arrowstyle="->",color=col,lw=1.8),zorder=5)
                if s==env.start:
                    ax.text(j,i-0.42,"S",ha="center",fontsize=6.5,color=P["blue"],fontweight="bold",zorder=6)

    ax.set_xticks(range(env.size)); ax.set_yticks(range(env.size))
    ax.set_xticklabels([f"c{i}" for i in range(env.size)], fontsize=6.5)
    ax.set_yticklabels([f"r{i}" for i in range(env.size)], fontsize=6.5)
    ax.set_title(title, color=P["txt"], fontsize=8.5, fontweight="bold", pad=6)
    for sp in ax.spines.values(): sp.set_edgecolor(P["border"])


def smooth(arr, w=25):
    arr = np.array(arr, dtype=float)
    if len(arr) < w: return arr
    return np.convolve(arr, np.ones(w)/w, mode="valid")


def plot_learning_curve(ax, rewards, color, label, w=30):
    raw = np.array(rewards, dtype=float)
    ax.plot(raw, color=color, alpha=0.12, lw=0.5)
    sm = smooth(raw, w)
    ax.plot(range(len(sm)), sm, color=color, lw=2.2, label=label)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Total Return", fontsize=8)
    ax.grid(color=P["border"], alpha=0.4, lw=0.5)
    ax.legend(facecolor=P["card"], labelcolor=P["txt"], fontsize=7.5, framealpha=0.9)


# ══════════════════════════════════════════════════════════════
# UI COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════

def section_header(icon, title):
    st.markdown(f"""
    <div class="section-hdr">
      <span style="font-size:1.3rem">{icon}</span>
      <h3>{title}</h3>
    </div>""", unsafe_allow_html=True)


def badge(text, color="blue"):
    st.markdown(f'<span class="badge badge-{color}">{text}</span>', unsafe_allow_html=True)


def method_card(title, badge_txt, badge_color, content_md, icon=""):
    st.markdown(f"""
    <div class="mc-card mc-card-accent">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:.8rem">
        <span style="font-size:1.4rem">{icon}</span>
        <span style="color:var(--txt);font-weight:700;font-size:1rem">{title}</span>
        <span class="badge badge-{badge_color}">{badge_txt}</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(content_md)


def show_pseudocode(code: str):
    """Render pseudocode in a styled block with keyword highlighting."""
    st.markdown(f'<div class="pseudocode">{code}</div>', unsafe_allow_html=True)


def formula_block(latex_str: str, label: str = ""):
    if label:
        st.markdown(f"<p style='color:var(--txt2);font-size:.8rem;margin-bottom:2px'>{label}</p>",
                    unsafe_allow_html=True)
    st.latex(latex_str)


# ══════════════════════════════════════════════════════════════
# EPISODE TRACE VISUALISATION (show return computation)
# ══════════════════════════════════════════════════════════════

def render_episode_trace(env, episode, gamma, fig_title="Return Computation — Backward Pass"):
    """
    Draw a timeline of an episode showing how G is computed backwards.
    Each step shows: state, action, reward, cumulative G.
    """
    T = min(len(episode), 14)
    ep = episode[:T]

    fig, ax = plt.subplots(figsize=(max(10, T*1.2+1), 3.5))
    mpl_defaults(fig, [ax])
    ax.axis("off")
    ax.set_xlim(-0.5, T+0.5)
    ax.set_ylim(-0.5, 3.5)

    # Compute G backwards
    G_vals = []
    G = 0.0
    for s, a, r in reversed(ep):
        G = gamma*G + r
        G_vals.append(G)
    G_vals.reverse()

    colors_step = [P["blue"],P["violet"],P["green"],P["amber"],P["cyan"],P["rose"],P["yellow"],P["pink"]]*4

    for t,(s,a,r) in enumerate(ep):
        cx = t
        col = colors_step[t % len(colors_step)]

        # Box
        ax.add_patch(FancyBboxPatch((cx-.4, 1.5), 0.8, 1.2,
                     boxstyle="round,pad=0.05", fc=P["card"], ec=col, lw=1.5, zorder=3))

        # State & action
        ax.text(cx, 2.75, f"S={s}", ha="center", va="center",
                color=P["txt"], fontsize=7.5, fontweight="bold", zorder=4)
        ax.text(cx, 2.35, f"A={env.SYMBOLS[a]}", ha="center", va="center",
                color=col, fontsize=8, zorder=4)
        ax.text(cx, 1.95, f"R={r:+.1f}", ha="center", va="center",
                color=P["amber"] if r > 0 else (P["rose"] if r < 0 else P["txt2"]),
                fontsize=7.5, fontweight="bold", zorder=4)

        # G box below
        ax.add_patch(FancyBboxPatch((cx-.35, 0.5), 0.7, 0.7,
                     boxstyle="round,pad=0.04", fc="#0d1a30", ec=P["cyan"], lw=1.0, zorder=3))
        ax.text(cx, 0.85, f"G={G_vals[t]:.2f}", ha="center", va="center",
                color=P["cyan"], fontsize=6.8, fontweight="bold", zorder=4)

        # Arrow between steps
        if t < T-1:
            ax.annotate("", xy=(cx+0.45, 2.1), xytext=(cx+0.4,2.1),
                        arrowprops=dict(arrowstyle="->",color=P["border"],lw=1.2), zorder=5)

    # Backward arrows for G computation
    for t in range(T-2, -1, -1):
        ax.annotate("",xy=(t+0.35, 0.85), xytext=(t+0.65, 0.85),
                    arrowprops=dict(arrowstyle="<-", color=P["violet"],lw=1.0,
                                   linestyle="dashed"), zorder=5)

    ax.text(-0.4, 2.75, "Step", ha="center", va="center",
            color=P["txt2"], fontsize=7, style="italic")
    ax.text(-0.4, 0.85, "G←", ha="center", va="center",
            color=P["violet"], fontsize=7, style="italic")
    ax.text(T/2 - 0.5, 3.35, fig_title,
            ha="center", va="center", color=P["txt"], fontsize=9, fontweight="bold")

    txt_update = f"Update rule:  G ← γ·G + R   (γ={gamma:.2f})   — computed RIGHT→LEFT"
    ax.text(T/2 - 0.5, 0.1, txt_update, ha="center", va="center",
            color=P["txt2"], fontsize=7.5, style="italic")

    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════
# RUN ALL METHODS
# ══════════════════════════════════════════════════════════════

def run_all(env, n_ep, gamma, eps, seed):
    np.random.seed(seed)
    b = env.uniform_policy()

    V_fv,  hist_fv,  vlog_fv  = mc_first_visit(env, b, n_ep, gamma)
    V_ev,  hist_ev             = mc_every_visit(env, b, n_ep, gamma)
    Q_es,  pi_es, V_es, rew_es = mc_exploring_starts(env, n_ep, gamma)
    Q_on,  pi_on, V_on, rew_on = mc_control_on_policy(env, n_ep, eps, gamma)

    pi_det = env.greedy_policy(Q_on)
    V_ois  = mc_ordinary_is(env, pi_det, b, n_ep, gamma)
    V_wis  = mc_weighted_is(env, pi_det, b, n_ep, gamma)

    Q_oc, pi_oc, V_oc, rew_oc = mc_off_policy_control(env, n_ep, gamma)

    V_inc, var_hist, vlog_inc = mc_incremental(env, b, n_ep, gamma)

    n_adv = min(n_ep, 600)
    V_pd  = mc_per_decision_is(env, pi_det, b, n_adv, gamma)
    V_da  = mc_discounting_aware_is(env, pi_det, b, n_adv, gamma)

    # Sample episode for trace visualisation
    sample_ep = env.generate_episode(b, max_steps=25)

    return dict(
        b=b,
        V_fv=V_fv, hist_fv=hist_fv, vlog_fv=vlog_fv,
        V_ev=V_ev, hist_ev=hist_ev,
        Q_es=Q_es, pi_es=pi_es, V_es=V_es, rew_es=rew_es,
        Q_on=Q_on, pi_on=pi_on, V_on=V_on, rew_on=rew_on,
        pi_det=pi_det,
        V_ois=V_ois, V_wis=V_wis,
        Q_oc=Q_oc, pi_oc=pi_oc, V_oc=V_oc, rew_oc=rew_oc,
        V_inc=V_inc, var_hist=var_hist, vlog_inc=vlog_inc,
        V_pd=V_pd, V_da=V_da,
        sample_ep=sample_ep,
    )


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════

def main():

    # ── HEADER ────────────────────────────────────────────────
    st.markdown("""
    <div style="
      background: linear-gradient(135deg,#0a1628 0%,#0f2044 40%,#1a0a3d 100%);
      border: 1px solid #1e3a6e;
      border-radius: 16px;
      padding: 2rem 2.5rem 1.8rem;
      margin-bottom: 1.8rem;
      position: relative;
      overflow: hidden;">
      <div style="
        position:absolute;top:-40px;right:-40px;
        width:200px;height:200px;
        background:radial-gradient(circle,rgba(91,156,246,0.12),transparent 70%);
        border-radius:50%"></div>
      <div style="display:flex;align-items:flex-start;gap:18px">
        <span style="font-size:3.2rem;line-height:1">🎲</span>
        <div>
          <h1 style="margin:0;color:#e2e8f0;font-family:'Sora',sans-serif;
                     font-size:2rem;font-weight:700;letter-spacing:-0.02em">
            Monte Carlo RL Explorer
          </h1>
          <p style="margin:.4rem 0 0;color:#7a8ba6;font-size:.92rem;
                    font-family:'Sora',sans-serif;font-weight:300">
            A complete interactive textbook — all 9 MC methods from Sutton &amp; Barto Chapter 5
            &nbsp;·&nbsp; 5×5 Stochastic Gridworld &nbsp;·&nbsp; Formulas, pseudocode &amp; visualisations
          </p>
          <div style="margin-top:.9rem;display:flex;gap:8px;flex-wrap:wrap">
            <span class="badge badge-blue">On-policy</span>
            <span class="badge badge-violet">Off-policy</span>
            <span class="badge badge-green">Exploring Starts</span>
            <span class="badge badge-cyan">Importance Sampling</span>
            <span class="badge badge-amber">Incremental</span>
            <span class="badge badge-rose">Per-Decision IS</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 .5rem">
          <span style="font-size:2rem">⚙️</span>
          <div style="color:#cdd6f4;font-weight:700;font-size:1rem;margin-top:4px">Experiment Control</div>
        </div>
        <hr style="border-color:#1e2d42;margin:.5rem 0 1rem">
        """, unsafe_allow_html=True)

        n_episodes = st.slider("Episodes", 200, 4000, 1000, 100,
                               help="More episodes → better convergence")
        gamma      = st.slider("Discount γ", 0.80, 1.00, 0.99, 0.01,
                               help="1.0 = no discounting; lower = favour short-term rewards")
        epsilon    = st.slider("ε (on-policy control)", 0.01, 0.40, 0.10, 0.01,
                               help="Exploration rate for ε-greedy policies")
        slip_prob  = st.slider("Slip Probability", 0.00, 0.30, 0.10, 0.05,
                               help="Prob. of random action override (environment stochasticity)")
        seed       = st.number_input("Random Seed", 0, 9999, 42, 1)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🚀  Run All 9 Methods", type="primary", use_container_width=True)

        st.markdown("""
        <hr style="border-color:#1e2d42;margin:1rem 0">
        <div style="color:#7a8ba6;font-size:.78rem;font-family:'IBM Plex Mono',monospace">
        <b style="color:#cdd6f4">Environment</b><br>
        5×5 GridWorld · stochastic<br><br>
        <b style="color:#cdd6f4">Methods (9 total)</b><br>
        ① First-Visit MC Prediction<br>
        ② Every-Visit MC Prediction<br>
        ③ MC Exploring Starts ★<br>
        ④ On-policy ε-greedy Control<br>
        ⑤ Off-policy Ordinary IS<br>
        ⑥ Off-policy Weighted IS<br>
        ⑦ Off-policy MC Control ★<br>
        ⑧ Incremental MC<br>
        ⑨ Per-Decision IS<br>
        ⑩ Discounting-Aware IS<br>
        <br>★ = new in this edition
        </div>
        """, unsafe_allow_html=True)

    env = GridWorld(size=5, slip_prob=slip_prob)

    # ── TABS ──────────────────────────────────────────────────
    tabs = st.tabs([
        "🗺️ Environment",
        "📐 Algorithms",
        "📊 Prediction",
        "🌐 Exploring Starts",
        "🎯 On-policy Control",
        "⚖️ Off-policy IS",
        "🎮 Off-policy Control",
        "⚡ Incremental & Advanced",
        "📈 Dashboard",
    ])

    tab_env, tab_algo, tab_pred, tab_es, tab_ctrl, tab_is, tab_oc, tab_adv, tab_dash = tabs

    # ══════════════════════════════════════════════════════════
    # TAB 0 — ENVIRONMENT
    # ══════════════════════════════════════════════════════════
    with tab_env:
        col_desc, col_grid = st.columns([1.1, 0.9], gap="large")

        with col_desc:
            section_header("🗺️", "The 5×5 Stochastic Gridworld")
            st.markdown("""
            <div class="mc-card">
            The gridworld is <b>episodic</b>: each episode terminates at the Goal (+10) or Trap (−5).
            It is <b>stochastic</b>: with probability <code>slip</code> the intended action is replaced
            by a uniformly random one.
            <br><br>
            <b>Why this environment for MC?</b><br>
            MC methods require <em>complete episodes</em>. Multiple paths exist around walls,
            so on-policy vs off-policy strategies differ meaningfully. The trap forces the agent
            to reason about long-term risk — perfect for importance sampling comparisons.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            | Feature | Value |
            |---------|-------|
            | Grid size | 5 × 5 (25 states) |
            | Actions | ↑ ↓ ← → (4) |
            | Slip prob | configurable |
            | Goal reward | **+10** (terminal) |
            | Trap reward | **−5** (terminal) |
            | Step cost | **−0.1** per move |
            | Walls | (1,1) (1,3) (3,1) (3,3) |
            """)

        with col_grid:
            fig, ax, _ = make_fig(1,1, 5.5, 5.5)
            _draw_grid_background(ax, env)
            for i in range(env.size):
                for j in range(env.size):
                    s=(i,j)
                    if s in env.walls:
                        ax.text(j,i,"■",ha="center",va="center",color="#2a3050",fontsize=20,zorder=4)
                    elif s==env.goal:
                        ax.text(j,i,"★\nGOAL\n+10",ha="center",va="center",fontsize=8,color="#6ee7b7",fontweight="bold",zorder=4)
                    elif s==env.trap:
                        ax.text(j,i,"✗\nTRAP\n−5",ha="center",va="center",fontsize=8,color="#fca5a5",fontweight="bold",zorder=4)
                    elif s==env.start:
                        ax.text(j,i,"●\nSTART\n(0,0)",ha="center",va="center",fontsize=8,color="#93c5fd",fontweight="bold",zorder=4)
                    else:
                        ax.text(j,i,f"({i},{j})",ha="center",va="center",fontsize=8,color=P["txt2"],zorder=4)
            ax.set_xlim(-0.5,4.5); ax.set_ylim(-0.5,4.5); ax.set_aspect("equal")
            ax.set_xticks(range(5),labels=[f"c{i}" for i in range(5)],fontsize=7)
            ax.set_yticks(range(5),labels=[f"r{i}" for i in range(5)],fontsize=7)
            ax.grid(color=P["border"],alpha=0.5,lw=0.5)
            ax.set_title("Gridworld Layout", color=P["txt"], fontweight="bold", pad=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

        section_header("🎬", "Sample Episode (random policy)")
        np.random.seed(seed)
        ep_sample = env.generate_episode(env.uniform_policy(), max_steps=30)
        tr = "  →  ".join(f"{s}" for s,a,r in ep_sample)
        st.code(f"Length : {len(ep_sample)} steps\n"
                f"Return : {sum(r for _,_,r in ep_sample):.2f}\n"
                f"Path   : {tr}", language="")

    # ══════════════════════════════════════════════════════════
    # TAB 1 — ALGORITHMS REFERENCE
    # ══════════════════════════════════════════════════════════
    with tab_algo:
        section_header("📐", "Algorithm Pseudocode & Formulas — All 9 Methods")

        st.info("This tab is your **reference sheet**. Every algorithm is presented with its pseudocode, "
                "key formula, and a concise intuition note. Tabs to the right show interactive results.")

        algo_entries = [
            {
                "num":"①","badge_txt":"On-Policy Prediction","badge_col":"blue",
                "title":"First-Visit MC Prediction","ref":"S&B §5.1",
                "intuition":"Average the return from the **first** visit to each state per episode.",
                "formula": r"V(s) \leftarrow \frac{1}{N(s)}\sum_{i=1}^{N(s)} G_t^{(i)} \quad \text{(first-visit returns)}",
                "code": (
                    "Input: policy π, episodes N, discount γ\n"
                    "Init:  V(s) ← 0,  Returns(s) ← [] for all s\n\n"
                    "For each episode i = 1..N:\n"
                    "  Generate S₀,A₀,R₁,…,Sₜ following π\n"
                    "  G ← 0\n"
                    "  For t = T−1 downto 0:\n"
                    "    G ← γ·G + R_{t+1}\n"
                    "    If Sₜ ∉ {S₀,…,S_{t-1}}:         ← first-visit check\n"
                    "      Returns(Sₜ).append(G)\n"
                    "      V(Sₜ) ← mean(Returns(Sₜ))"
                ),
            },
            {
                "num":"②","badge_txt":"On-Policy Prediction","badge_col":"blue",
                "title":"Every-Visit MC Prediction","ref":"S&B §5.1",
                "intuition":"Count state s every time it appears — more data, slight within-episode correlation.",
                "formula": r"V(s) \leftarrow \frac{1}{N_{\text{total}}(s)}\sum_{\text{all visits}} G_t",
                "code": (
                    "Same as First-Visit but remove the first-visit check:\n\n"
                    "  For t = T−1 downto 0:\n"
                    "    G ← γ·G + R_{t+1}\n"
                    "    Returns(Sₜ).append(G)            ← every occurrence\n"
                    "    V(Sₜ) ← mean(Returns(Sₜ))"
                ),
            },
            {
                "num":"③","badge_txt":"Exploring Starts","badge_col":"green",
                "title":"MC with Exploring Starts","ref":"S&B §5.3",
                "intuition":"Start each episode from a **random (s,a)** pair so all pairs are guaranteed to be sampled. "
                            "This removes the need for ε-greedy and allows convergence to the **true optimal π***.",
                "formula": r"\pi(s) \leftarrow \arg\max_a Q(s,a) \quad \text{(pure greedy — no } \varepsilon \text{ needed)}",
                "code": (
                    "Init: Q(s,a) ← 0,  Returns(s,a) ← [],  π(s) ← arbitrary\n\n"
                    "For each episode:\n"
                    "  S₀ ~ Uniform(all non-terminal states)   ← EXPLORING START\n"
                    "  A₀ ~ Uniform(A)                         ← random first action\n"
                    "  Generate episode from (S₀,A₀) following π\n"
                    "  G ← 0\n"
                    "  For t = T−1 downto 0:\n"
                    "    G ← γ·G + R_{t+1}\n"
                    "    If (Sₜ,Aₜ) first visit:\n"
                    "      Returns(Sₜ,Aₜ).append(G)\n"
                    "      Q(Sₜ,Aₜ) ← mean(Returns(Sₜ,Aₜ))\n"
                    "      π(Sₜ) ← argmax_a Q(Sₜ,a)           ← greedy (no ε!)"
                ),
            },
            {
                "num":"④","badge_txt":"On-Policy Control","badge_col":"blue",
                "title":"On-Policy MC Control (ε-greedy)","ref":"S&B §5.4",
                "intuition":"Like ES but replaces exploring starts with ε-soft policy. Converges to best ε-soft policy.",
                "formula": r"\pi(a|s) = \begin{cases} 1-\varepsilon+\varepsilon/|A| & a = \arg\max_a Q(s,a) \\ \varepsilon/|A| & \text{otherwise} \end{cases}",
                "code": (
                    "Same as MC ES but:\n"
                    "  Replace: S₀ ~ Uniform, A₀ ~ Uniform\n"
                    "  With:    S₀ = fixed start,  policy = ε-soft\n\n"
                    "  After each episode, update Q then:\n"
                    "    π(s) ← ε-greedy w.r.t. Q(s,·)"
                ),
            },
            {
                "num":"⑤","badge_txt":"Off-Policy Eval","badge_col":"violet",
                "title":"Ordinary Importance Sampling","ref":"S&B §5.5",
                "intuition":"Reweight behaviour-policy returns by the full trajectory IS ratio. Unbiased but explosive variance.",
                "formula": r"\hat{V}^{\text{ois}}(s) = \frac{1}{n}\sum_{i=1}^{n} \rho_{t:T-1}^{(i)} G_t^{(i)}, \quad \rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}",
                "code": (
                    "Given: target π, behavior b\n"
                    "For each episode generated by b:\n"
                    "  G ← 0,  ρ ← 1\n"
                    "  For t = T−1 downto 0:\n"
                    "    G ← γ·G + R_{t+1}\n"
                    "    ρ ← ρ · π(Aₜ|Sₜ) / b(Aₜ|Sₜ)\n"
                    "    If first visit to Sₜ:\n"
                    "      Append ρ·G to Returns(Sₜ)\n"
                    "  V(s) ← mean(Returns(s))"
                ),
            },
            {
                "num":"⑥","badge_txt":"Off-Policy Eval","badge_col":"violet",
                "title":"Weighted Importance Sampling","ref":"S&B §5.5",
                "intuition":"Normalise by the sum of IS weights instead of episode count. Biased but drastically lower variance.",
                "formula": r"\hat{V}^{\text{wis}}(s) = \frac{\sum_i \rho_i G_i}{\sum_i \rho_i} \quad \Leftrightarrow \quad V(s) \mathrel{+}= \frac{W}{C(s)}\bigl[G - V(s)\bigr]",
                "code": (
                    "V(s) ← 0,  C(s) ← 0  for all s\n"
                    "For each episode generated by b:\n"
                    "  G ← 0,  W ← 1\n"
                    "  For t = T−1 downto 0:\n"
                    "    G ← γ·G + R_{t+1}\n"
                    "    W ← W · π(Aₜ|Sₜ) / b(Aₜ|Sₜ)\n"
                    "    C(Sₜ) ← C(Sₜ) + W\n"
                    "    V(Sₜ) ← V(Sₜ) + (W/C(Sₜ))·[G − V(Sₜ)]"
                ),
            },
            {
                "num":"⑦","badge_txt":"Off-Policy Control","badge_col":"amber",
                "title":"Off-Policy MC Control","ref":"S&B §5.7",
                "intuition":"Learn the greedy target policy π using episodes from soft behavior policy b. "
                            "Uses Weighted IS update per action-pair. Converges to true π*.",
                "formula": r"\text{Update } Q(S_t,A_t) \mathrel{+}= \frac{W}{C(S_t,A_t)}\bigl[G - Q(S_t,A_t)\bigr], \quad\text{then } \pi(S_t)\leftarrow\arg\max_a Q(S_t,a)",
                "code": (
                    "Q(s,a) ← 0,  C(s,a) ← 0,  π(s) ← greedy(Q)\n"
                    "For each episode generated by b (ε-soft):\n"
                    "  G ← 0,  W ← 1\n"
                    "  For t = T−1 downto 0:\n"
                    "    G ← γ·G + R_{t+1}\n"
                    "    C(Sₜ,Aₜ) ← C(Sₜ,Aₜ) + W\n"
                    "    Q(Sₜ,Aₜ) ← Q(Sₜ,Aₜ) + (W/C)·[G − Q(Sₜ,Aₜ)]\n"
                    "    π(Sₜ) ← argmax_a Q(Sₜ,a)\n"
                    "    If Aₜ ≠ π(Sₜ): break    ← episode diverged from target\n"
                    "    W ← W / b(Aₜ|Sₜ)"
                ),
            },
            {
                "num":"⑧","badge_txt":"Incremental","badge_col":"amber",
                "title":"Incremental MC Update","ref":"S&B §5.6",
                "intuition":"Replace full-batch averaging with an online update. Memory-efficient. Bridge to TD learning.",
                "formula": r"V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}\bigl[G_t - V(S_t)\bigr] \qquad \text{or with constant } \alpha: \quad V \leftarrow V + \alpha[G-V]",
                "code": (
                    "V(s) ← 0,  N(s) ← 0  for all s\n"
                    "For each episode:\n"
                    "  Compute G backwards\n"
                    "  For each first-visit (Sₜ, Gₜ):\n"
                    "    N(Sₜ) ← N(Sₜ) + 1\n"
                    "    V(Sₜ) ← V(Sₜ) + (1/N(Sₜ))·[Gₜ − V(Sₜ)]"
                ),
            },
            {
                "num":"⑨","badge_txt":"Advanced IS","badge_col":"cyan",
                "title":"Per-Decision Importance Sampling","ref":"S&B §5.8",
                "intuition":"Decompose the IS weight: each reward Rₖ is weighted only by ratios **up to** step k. "
                            "Provably reduces variance vs ordinary/weighted IS.",
                "formula": r"\hat{V}^{\text{pd}}(s_t) = \sum_{k=t}^{T} \gamma^{k-t}\left(\prod_{j=t}^{k}\rho_j\right) R_{k+1}",
                "code": (
                    "For each episode generated by b:\n"
                    "  For each t in 0..T:\n"
                    "    G_pd ← 0,  ρ_cum ← 1\n"
                    "    For k = t..T:\n"
                    "      ρ_cum ← ρ_cum · π(Aₖ|Sₖ)/b(Aₖ|Sₖ)\n"
                    "      G_pd  ← G_pd + γ^{k-t} · ρ_cum · R_{k+1}\n"
                    "    Append G_pd to Returns(Sₜ)\n"
                    "  V(s) ← mean(Returns(s))"
                ),
            },
            {
                "num":"⑩","badge_txt":"Advanced IS","badge_col":"cyan",
                "title":"Discounting-Aware IS","ref":"S&B §5.9",
                "intuition":"Exploits γ<1 to decompose returns into flat partial returns, each weighted by shorter IS products. "
                            "Lowest variance of all IS estimators.",
                "formula": r"G_t = (1-\gamma)\sum_{h=t+1}^{T-1}\gamma^{h-t-1}G_{t:h} + \gamma^{T-t}G_{t:T}, \quad \phi = \sum_{h}(1-\gamma)\gamma^{h-t}W_{t:h}",
                "code": (
                    "V(s) ← 0,  C(s) ← 0\n"
                    "For each episode generated by b:\n"
                    "  For each t:\n"
                    "    G ← 0,  W ← 1,  φ ← 0\n"
                    "    For k = t..T:\n"
                    "      W ← W · π(Aₖ|Sₖ)/b(Aₖ|Sₖ)\n"
                    "      φ ← φ + (1−γ)·γ^{k-t}·W\n"
                    "      G ← G + γ^{k-t}·R_{k+1}\n"
                    "    φ ← φ + γ^{T-t}·W          ← terminal term\n"
                    "    C(Sₜ) ← C(Sₜ) + φ\n"
                    "    V(Sₜ) ← V(Sₜ) + (φ/C(Sₜ))·[G − V(Sₜ)]"
                ),
            },
        ]

        for e in algo_entries:
            col_badge, _, _ = st.columns([0.08, 0.5, 0.42])
            with st.expander(f"  {e['num']}  {e['title']}   ({e['ref']})", expanded=False):
                c1, c2 = st.columns([1,1], gap="large")
                with c1:
                    st.markdown(f"<span class='badge badge-{e['badge_col']}'>{e['badge_txt']}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"\n**💡 Intuition:** {e['intuition']}\n")
                    st.markdown("**Formula:**")
                    st.latex(e["formula"])
                with c2:
                    st.markdown("**Pseudocode:**")
                    st.code(e["code"], language="text")

    # ══════════════════════════════════════════════════════════
    # RUN GATE
    # ══════════════════════════════════════════════════════════
    if run_btn:
        with st.spinner("⏳  Running all 9 MC methods — please wait…"):
            res = run_all(env, n_episodes, gamma, epsilon, seed)
        st.session_state["mc_res"] = res
        st.session_state["mc_params"] = dict(n=n_episodes, g=gamma, e=epsilon, s=seed)
        st.sidebar.success(f"✅ Done! {n_episodes} episodes each.")

    PLACEHOLDER = "👈  Press **Run All 9 Methods** in the sidebar to compute results."

    # ══════════════════════════════════════════════════════════
    # TAB 2 — PREDICTION
    # ══════════════════════════════════════════════════════════
    with tab_pred:
        section_header("📊", "MC Prediction — First-Visit vs Every-Visit")
        st.markdown("""
        Both methods estimate **V(s)** under the random behavior policy.
        The key difference: what happens when a state appears **multiple times** in one episode.
        """)

        formula_block(
            r"G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}",
            "Return (discounted cumulative reward):"
        )

        if "mc_res" not in st.session_state:
            st.info(PLACEHOLDER); 
        else:
            res = st.session_state["mc_res"]

            # Episode trace
            section_header("🔍", "How G is Computed — Backward Pass")
            st.markdown("Watch how the return G accumulates from right to left through a sample episode:")
            fig_trace = render_episode_trace(env, res["sample_ep"], gamma)
            st.pyplot(fig_trace); plt.close()

            # Heatmaps
            section_header("🌡️", "Value Function V(s)")
            fig, axes, _ = make_fig(1,2, 13, 5.5)
            plot_value_heatmap(env, res["V_fv"], "① First-Visit MC — V(s)", axes[0])
            plot_value_heatmap(env, res["V_ev"], "② Every-Visit MC — V(s)", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Convergence
            section_header("📉", "Convergence at State (3,4) — Goal-Adjacent")
            focal = (3,4)
            fv_tr = [h.get(focal,0.0) for h in res["hist_fv"]]
            ev_tr = [h.get(focal,0.0) for h in res["hist_ev"]]
            x_ax  = [(i+1)*max(1,n_episodes//30) for i in range(len(fv_tr))]

            fig2, ax2, _ = make_fig(1,1, 11,4)
            ax2.plot(x_ax, fv_tr, color=P["blue"],   lw=2.5, marker="o", ms=4, label="First-Visit MC")
            ax2.plot(x_ax, ev_tr, color=P["violet"], lw=2.5, marker="s", ms=4, label="Every-Visit MC")
            ax2.set_xlabel("Episodes"); ax2.set_ylabel("V(3,4)")
            ax2.set_title("V(3,4) Convergence Over Episodes", color=P["txt"], fontweight="bold")
            ax2.legend(facecolor=P["card"], labelcolor=P["txt"], fontsize=8)
            ax2.grid(color=P["border"], alpha=0.4, lw=0.5)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

            c1,c2 = st.columns(2)
            with c1:
                st.success("**First-Visit MC:** Each episode contributes exactly 1 independent return per state → unbiased, lower variance per-episode.")
            with c2:
                st.warning("**Every-Visit MC:** Multiple returns per episode are correlated → slight bias, but more updates → faster practical convergence.")

    # ══════════════════════════════════════════════════════════
    # TAB 3 — EXPLORING STARTS
    # ══════════════════════════════════════════════════════════
    with tab_es:
        section_header("🌐", "MC with Exploring Starts (ES)")
        st.markdown("""
        Exploring Starts is the **cleanest path to the optimal policy** among MC methods.
        It removes the ε-greedy requirement entirely by guaranteeing coverage through random starts.
        """)

        col_a, col_b = st.columns([1,1], gap="large")
        with col_a:
            st.markdown("""
            <div class="mc-card mc-card-accent">
            <b>Core Idea</b><br>
            Every episode starts from a randomly chosen state–action pair (S₀,A₀).
            This guarantees that every (s,a) pair is eventually sampled → no need for ε-noise.
            The policy is updated <b>greedily</b> (π(s) = argmaxₐ Q(s,a)) after each episode.
            <br><br>
            <b>Convergence guarantee:</b> π → π* (true optimal), not just best ε-soft policy.
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            formula_block(
                r"\pi(s) \leftarrow \arg\max_a\, Q(s,a) \quad \forall s \quad \text{(no } \varepsilon \text{ needed)}",
                "Policy improvement:"
            )
            formula_block(
                r"Q(s,a) \leftarrow \frac{1}{N(s,a)}\sum_i G_t^{(i)} \quad \text{s.t. } (S_0^{(i)},A_0^{(i)}) \sim \text{Uniform}",
                "Action-value estimate:"
            )

        if "mc_res" not in st.session_state:
            st.info(PLACEHOLDER)
        else:
            res = st.session_state["mc_res"]

            section_header("🌡️", "Value & Policy — Exploring Starts vs On-policy ε-greedy")
            fig, axes, _ = make_fig(2,2, 13,10)
            plot_value_heatmap(env, res["V_es"], "③ Exploring Starts — V(s)",      axes[0][0])
            plot_value_heatmap(env, res["V_on"], "④ On-policy ε-greedy — V(s)",    axes[0][1])
            plot_policy_arrows(env, res["pi_es"],"③ ES — Greedy Policy π*",         axes[1][0])
            plot_policy_arrows(env, res["pi_on"],"④ On-policy — Best ε-soft Policy",axes[1][1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Learning curves side by side
            section_header("📉", "Learning Curves")
            fig3, axes3, _ = make_fig(1,2, 13,4)
            plot_learning_curve(axes3[0], res["rew_es"], P["green"],  "Exploring Starts", w=40)
            axes3[0].set_title("ES — Episode Return", color=P["txt"], fontweight="bold")
            plot_learning_curve(axes3[1], res["rew_on"], P["blue"],   "ε-greedy Control", w=40)
            axes3[1].set_title("On-policy ε-greedy — Episode Return", color=P["txt"], fontweight="bold")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            # Q-table comparison at start state
            section_header("📋", "Q(start, ·) Comparison")
            Q_es_s = res["Q_es"][(0,0)]
            Q_on_s = res["Q_on"][(0,0)]
            qdf = pd.DataFrame({
                "Action": ["↑ Up","→ Right","↓ Down","← Left"],
                "Q — Exploring Starts": [f"{v:.3f}" for v in Q_es_s],
                "ES Best?": ["✅" if i==int(np.argmax(Q_es_s)) else "" for i in range(4)],
                "Q — On-policy":        [f"{v:.3f}" for v in Q_on_s],
                "OP Best?": ["✅" if i==int(np.argmax(Q_on_s)) else "" for i in range(4)],
            })
            st.dataframe(qdf, use_container_width=True, hide_index=True)

            st.info("**Key difference:** Exploring Starts yields a fully greedy (deterministic) optimal policy. "
                    "On-policy ε-greedy retains some randomness in the final policy due to the ε-soft constraint.")

    # ══════════════════════════════════════════════════════════
    # TAB 4 — ON-POLICY CONTROL
    # ══════════════════════════════════════════════════════════
    with tab_ctrl:
        section_header("🎯", "On-Policy MC Control — ε-greedy GPI")
        st.markdown("""
        **Generalized Policy Iteration (GPI):** alternate between policy evaluation (updating Q) and policy 
        improvement (updating π) until convergence. ε-greedy ensures continued exploration.
        """)

        formula_block(
            r"Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\bigl[G_t - Q(s,a)\bigr]",
            "Action-value update (first-visit, incremental form):"
        )
        formula_block(
            r"\pi(a|s) = \begin{cases}1-\varepsilon + \varepsilon/|A| & a^* = \arg\max_a Q(s,a) \\ \varepsilon/|A| & \text{otherwise}\end{cases}",
            "ε-greedy policy (ensures every action has P ≥ ε/|A|):"
        )

        if "mc_res" not in st.session_state:
            st.info(PLACEHOLDER)
        else:
            res = st.session_state["mc_res"]

            col_v, col_p = st.columns(2, gap="large")
            with col_v:
                fig, ax, _ = make_fig(1,1, 6.5, 6)
                plot_value_heatmap(env, res["V_on"], "④ On-policy Control — V(s)", ax)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with col_p:
                fig, ax, _ = make_fig(1,1, 6.5, 6)
                plot_policy_arrows(env, res["pi_det"], "Greedy π* derived from Q", ax)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            section_header("📉", "Episode Return — Learning Curve")
            fig4, ax4, _ = make_fig(1,1, 11,4)
            plot_learning_curve(ax4, res["rew_on"], P["blue"], "On-policy MC Control", w=50)
            ax4.set_title("ε-greedy GPI — Total Return per Episode", color=P["txt"], fontweight="bold")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

            section_header("📋", "Q(s,·) for All States — Action Preference Heatmap")
            q_matrix = np.zeros((env.size*env.size, 4))
            for i in range(env.n_states):
                s = env.i2s(i)
                q_matrix[i] = res["Q_on"][s]
            fig5, ax5, _ = make_fig(1,1, 12,6)
            im = ax5.imshow(q_matrix.T, cmap=RL_CMAP, aspect="auto")
            ax5.set_yticks([0,1,2,3], labels=["↑","→","↓","←"], fontsize=10, color=P["txt"])
            ax5.set_xlabel("State index (row-major)", fontsize=8)
            ax5.set_title("Q(s,a) Heatmap — All States × All Actions", color=P["txt"], fontweight="bold")
            plt.colorbar(im, ax=ax5, fraction=0.03).ax.tick_params(colors=P["txt2"])
            plt.tight_layout(); st.pyplot(fig5); plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 5 — OFF-POLICY IS
    # ══════════════════════════════════════════════════════════
    with tab_is:
        section_header("⚖️", "Off-Policy Evaluation — Ordinary IS vs Weighted IS")
        st.markdown("""
        Off-policy methods evaluate a **target policy π** using episodes from a different **behavior policy b**.  
        The correction factor (importance sampling ratio ρ) compensates for the distributional mismatch.
        """)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            formula_block(
                r"\rho_{t:T-1} = \prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}",
                "IS ratio (full episode):"
            )
            formula_block(
                r"\hat{V}^{\text{ois}}(s) = \frac{1}{n}\sum_{i=1}^{n}\rho_i G_i \quad \text{(Unbiased, High Var.)}",
                "Ordinary IS estimator:"
            )
        with col_f2:
            formula_block(
                r"\hat{V}^{\text{wis}}(s) = \frac{\sum_i \rho_i G_i}{\sum_i \rho_i} \quad \text{(Biased, Low Var.)}",
                "Weighted IS estimator:"
            )
            formula_block(
                r"\text{Var}(\hat{V}^{\text{ois}}) = O(T \cdot \rho_{\max}^T) \quad \text{(exponential in T!)}",
                "Why Ordinary IS variance explodes:"
            )

        if "mc_res" not in st.session_state:
            st.info(PLACEHOLDER)
        else:
            res = st.session_state["mc_res"]

            section_header("🌡️", "Value Function Comparison")
            fig, axes, _ = make_fig(1,3, 18,5.5)
            plot_value_heatmap(env, res["V_on"],  "④ On-policy (reference)", axes[0])
            plot_value_heatmap(env, res["V_ois"], "⑤ Ordinary IS",           axes[1])
            plot_value_heatmap(env, res["V_wis"], "⑥ Weighted IS",           axes[2])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Per-state scatter
            section_header("🔬", "Per-State Value Comparison vs Reference")
            all_s = env.non_terminal_states()
            ref  = np.array([res["V_on"].get(s,0)  for s in all_s])
            ois  = np.array([res["V_ois"].get(s,0) for s in all_s])
            wis  = np.array([res["V_wis"].get(s,0) for s in all_s])

            fig6, axes6, _ = make_fig(1,2, 13,5)
            axes6[0].scatter(ref, ois, color=P["rose"],   alpha=0.8, s=80, edgecolors="white", lw=0.6, label="Ordinary IS")
            axes6[0].scatter(ref, wis, color=P["cyan"],   alpha=0.8, s=80, edgecolors="white", lw=0.6, label="Weighted IS")
            mn,mx = min(ref.min(), ois.min(), wis.min()), max(ref.max(), ois.max(), wis.max())
            axes6[0].plot([mn,mx],[mn,mx],"--",color=P["txt2"],lw=1.2,alpha=0.6,label="Perfect agreement")
            axes6[0].set_xlabel("On-policy V(s) — reference"); axes6[0].set_ylabel("IS estimate V(s)")
            axes6[0].set_title("Ordinary IS vs Weighted IS — Scatter", color=P["txt"], fontweight="bold")
            axes6[0].legend(facecolor=P["card"],labelcolor=P["txt"],fontsize=8)
            axes6[0].grid(color=P["border"],alpha=0.4,lw=0.5)

            # Error bars
            err_ois = np.abs(ois-ref); err_wis = np.abs(wis-ref)
            idx = range(len(all_s))
            axes6[1].bar(idx, err_ois, color=P["rose"],  alpha=0.65, label=f"Ordinary IS (MAE={err_ois.mean():.3f})")
            axes6[1].bar(idx, err_wis, color=P["cyan"],  alpha=0.65, label=f"Weighted IS (MAE={err_wis.mean():.3f})")
            axes6[1].set_xlabel("State index"); axes6[1].set_ylabel("|error| vs reference")
            axes6[1].set_title("Absolute Error per State", color=P["txt"], fontweight="bold")
            axes6[1].legend(facecolor=P["card"],labelcolor=P["txt"],fontsize=8)
            axes6[1].grid(color=P["border"],alpha=0.4,lw=0.5,axis="y")
            plt.tight_layout(); st.pyplot(fig6); plt.close()

            mse_ois = float(np.mean((ois-ref)**2)); mse_wis = float(np.mean((wis-ref)**2))
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Ordinary IS — MSE",  f"{mse_ois:.4f}")
            c2.metric("Weighted IS — MSE",  f"{mse_wis:.4f}")
            c3.metric("WIS variance reduction", f"{max(0,(mse_ois-mse_wis)/max(mse_ois,1e-9)*100):.1f}%")
            c4.metric("States covered (WIS)", f"{sum(1 for s in all_s if s in res['V_wis'])}/{len(all_s)}")

    # ══════════════════════════════════════════════════════════
    # TAB 6 — OFF-POLICY CONTROL
    # ══════════════════════════════════════════════════════════
    with tab_oc:
        section_header("🎮", "Off-Policy MC Control (S&B §5.7)")
        st.markdown("""
        Off-policy **control** learns the *optimal* greedy policy π using episodes from a soft behavior
        policy b. Crucially, WIS is used per (s,a) pair rather than per state.
        The update stops whenever the episode deviates from the target policy — ensuring W stays finite.
        """)

        formula_block(
            r"Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \frac{W}{C(S_t,A_t)}\bigl[G - Q(S_t,A_t)\bigr]",
            "WIS action-value update:"
        )
        formula_block(
            r"\pi(S_t) \leftarrow \arg\max_a Q(S_t,a), \quad \text{break if } A_t \neq \pi(S_t)",
            "Greedy target-policy improvement + early break:"
        )
        formula_block(
            r"W \leftarrow \frac{W}{b(A_t|S_t)} \quad \text{(grow weight toward past visited by target)}",
            "IS weight update:"
        )

        if "mc_res" not in st.session_state:
            st.info(PLACEHOLDER)
        else:
            res = st.session_state["mc_res"]

            section_header("🌡️", "Value & Policy — Off-policy Control vs On-policy")
            fig, axes, _ = make_fig(2,2, 13,10)
            plot_value_heatmap(env, res["V_oc"], "⑦ Off-policy Control — V(s)",   axes[0][0])
            plot_value_heatmap(env, res["V_on"], "④ On-policy Control — V(s)",    axes[0][1])
            plot_policy_arrows(env, res["pi_oc"],"⑦ Off-policy — Greedy π*",       axes[1][0])
            plot_policy_arrows(env, res["pi_det"],"④ On-policy — Greedy π (ref)", axes[1][1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

            section_header("📉", "Learning Curves — Off-policy vs On-policy Control")
            fig7, axes7, _ = make_fig(1,2, 13,4)
            plot_learning_curve(axes7[0], res["rew_oc"], P["amber"], "Off-policy MC Control", w=50)
            axes7[0].set_title("Off-policy Control", color=P["txt"], fontweight="bold")
            plot_learning_curve(axes7[1], res["rew_on"], P["blue"],  "On-policy MC Control",  w=50)
            axes7[1].set_title("On-policy Control", color=P["txt"], fontweight="bold")
            plt.tight_layout(); st.pyplot(fig7); plt.close()

            c1,c2 = st.columns(2)
            c1.info("**Off-policy Control:** Can learn π* while exploring with any covering b. "
                    "Early-break rule makes it data-efficient but reduces episode utilisation.")
            c2.success("**Advantage over On-policy:** The target policy π is always deterministic-greedy — "
                       "not constrained by ε-soft → converges to the **true optimal policy**.")

    # ══════════════════════════════════════════════════════════
    # TAB 7 — INCREMENTAL & ADVANCED IS
    # ══════════════════════════════════════════════════════════
    with tab_adv:
        sub1, sub2 = st.tabs(["⚡ Incremental MC", "🔬 Per-Decision & Discounting-Aware IS"])

        with sub1:
            section_header("⚡", "Incremental Monte Carlo (S&B §5.6)")
            st.markdown("""
            Incremental MC replaces batch averaging with a running online update.
            It is **equivalent** to batch averaging when step-size = 1/N, but uses O(1) memory.
            Replacing 1/N with a fixed α gives an **exponentially weighted** moving average — 
            crucial for non-stationary environments and the conceptual bridge to TD learning.
            """)

            col_f1,col_f2 = st.columns(2)
            with col_f1:
                formula_block(
                    r"V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)}\bigl[G_t - V(S_t)\bigr]",
                    "Incremental mean (equivalent to full batch average):"
                )
            with col_f2:
                formula_block(
                    r"V(S_t) \leftarrow V(S_t) + \alpha\bigl[G_t - V(S_t)\bigr], \quad \alpha \in (0,1]",
                    "Constant step-size α (non-stationary / TD precursor):"
                )
            formula_block(
                r"\lim_{\text{bootstrap}} G_t = R_{t+1} + \gamma V(S_{t+1}) \quad \Rightarrow \quad \text{TD(0)}",
                "Replace G with bootstrap estimate → arrives at TD learning:"
            )

            if "mc_res" not in st.session_state:
                st.info(PLACEHOLDER)
            else:
                res = st.session_state["mc_res"]

                fig, axes, _ = make_fig(1,2, 13,5.5)
                plot_value_heatmap(env, res["V_fv"],  "① First-Visit MC (batch)", axes[0])
                plot_value_heatmap(env, res["V_inc"], "⑧ Incremental MC",         axes[1])
                plt.tight_layout(); st.pyplot(fig); plt.close()

                section_header("📉", "Mean Variance Across States — Over Training")
                fig8, ax8, _ = make_fig(1,1, 11,4)
                vh = res["var_hist"]
                if vh:
                    ax8.plot(vh, color=P["amber"], lw=2.5)
                    ax8.fill_between(range(len(vh)), 0, vh, color=P["amber"], alpha=0.12)
                ax8.set_xlabel("Checkpoint (every N/30 episodes)"); ax8.set_ylabel("Mean Var V(s)")
                ax8.set_title("Variance Reduction as N Grows — Incremental MC", color=P["txt"], fontweight="bold")
                ax8.grid(color=P["border"],alpha=0.4,lw=0.5)
                plt.tight_layout(); st.pyplot(fig8); plt.close()

        with sub2:
            section_header("🔬", "Per-Decision IS & Discounting-Aware IS")
            st.markdown("""
            Both methods reduce variance beyond standard WIS by exploiting the structure of the episode.
            """)

            col_f1,col_f2 = st.columns(2)
            with col_f1:
                formula_block(
                    r"\hat{V}^{\text{pd}}(s_t) = \sum_{k=t}^{T} \gamma^{k-t}\underbrace{\left(\prod_{j=t}^{k}\rho_j\right)}_{\text{partial IS ratio}}\! R_{k+1}",
                    "⑨ Per-Decision IS — each reward weighted by shorter IS product:"
                )
                st.markdown("""
                **Why variance is lower:** Standard IS multiplies all T ratios together.  
                Per-Decision IS multiplies only k−t ratios for each Rₖ₊₁.  
                Since Var(product) grows with number of terms, shorter products → less variance.
                """)
            with col_f2:
                formula_block(
                    r"G_t = \underbrace{(1-\gamma)\sum_{h}\gamma^{h-t-1}G_{t:h}}_{\text{flat partial returns}} + \gamma^{T-t}G_{t:T}",
                    "⑩ Discounting-Aware IS — decompose into flat partial returns:"
                )
                st.markdown("""
                **Why variance is lowest:** With γ<1, distant rewards contribute **exponentially less**.
                Their IS ratios (which can be large) are correspondingly downweighted.  
                As γ→1 this collapses to Per-Decision IS.
                """)

            if "mc_res" not in st.session_state:
                st.info(PLACEHOLDER)
            else:
                res = st.session_state["mc_res"]

                fig, axes, _ = make_fig(1,2, 13,5.5)
                plot_value_heatmap(env, res["V_pd"], "⑨ Per-Decision IS — V(s)",      axes[0])
                plot_value_heatmap(env, res["V_da"], "⑩ Discounting-Aware IS — V(s)", axes[1])
                plt.tight_layout(); st.pyplot(fig); plt.close()

                # Variance bar chart
                section_header("📊", "IS Variance Hierarchy — All Methods")
                non_t = env.non_terminal_states()
                variance_data = {
                    "Ordinary\nIS":       np.var([res["V_ois"].get(s,0) for s in non_t]),
                    "Weighted\nIS":       np.var([res["V_wis"].get(s,0) for s in non_t]),
                    "Per-Decision\nIS":   np.var([res["V_pd"].get(s,0)  for s in non_t]),
                    "Discounting\nIS":    np.var([res["V_da"].get(s,0)  for s in non_t]),
                }
                fig9, ax9, _ = make_fig(1,1, 9,4)
                bcolors = [P["rose"], P["cyan"], P["green"], P["amber"]]
                bars = ax9.bar(list(variance_data.keys()), list(variance_data.values()),
                               color=bcolors, edgecolor="white", lw=0.5, alpha=0.88, width=0.6)
                for bar,val in zip(bars, variance_data.values()):
                    ax9.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                             f"{val:.4f}", ha="center", va="bottom", color=P["txt"], fontsize=9)
                ax9.set_ylabel("Var(V̂) across non-terminal states"); 
                ax9.set_title("IS Variance Hierarchy (lower = better)", color=P["txt"], fontweight="bold")
                ax9.grid(color=P["border"],alpha=0.4,lw=0.5,axis="y")
                plt.tight_layout(); st.pyplot(fig9); plt.close()

    # ══════════════════════════════════════════════════════════
    # TAB 8 — DASHBOARD
    # ══════════════════════════════════════════════════════════
    with tab_dash:
        section_header("📈", "Grand Comparison Dashboard — All 9 Methods")

        if "mc_res" not in st.session_state:
            st.info(PLACEHOLDER)
        else:
            res = st.session_state["mc_res"]
            non_t = env.non_terminal_states()

            # ── 9 heatmaps ────────────────────────────────────
            section_header("🌡️", "Value Functions — All Methods")
            all_V = [
                (res["V_fv"],  "① First-Visit MC"),
                (res["V_ev"],  "② Every-Visit MC"),
                (res["V_es"],  "③ Exploring Starts"),
                (res["V_on"],  "④ On-policy Control"),
                (res["V_ois"], "⑤ Ordinary IS"),
                (res["V_wis"], "⑥ Weighted IS"),
                (res["V_oc"],  "⑦ Off-policy Control"),
                (res["V_inc"], "⑧ Incremental MC"),
                (res["V_pd"],  "⑨ Per-Decision IS"),
                (res["V_da"],  "⑩ Discounting IS"),
            ]
            fig, axes = plt.subplots(2,5, figsize=(22,9))
            mpl_defaults(fig, axes.flatten().tolist())
            for idx,(V,title) in enumerate(all_V):
                ax = axes[idx//5][idx%5]
                plot_value_heatmap(env, V, title, ax)
            plt.tight_layout(pad=1.2)
            st.pyplot(fig); plt.close()

            # ── Summary table ─────────────────────────────────
            section_header("📋", "Quantitative Summary Table")
            ref_v = np.array([res["V_on"].get(s,0) for s in non_t])
            rows = []
            META = [
                ("① First-Visit MC",    "On-policy",  "Prediction", "None",     "Low",        "Low"),
                ("② Every-Visit MC",    "On-policy",  "Prediction", "Slight",   "Medium",     "Low"),
                ("③ Exploring Starts",  "On-policy",  "Control",    "None",     "Low",        "Converges to π*"),
                ("④ On-policy ε-greedy","On-policy",  "Control",    "None",     "Medium",     "Best ε-soft"),
                ("⑤ Ordinary IS",       "Off-policy", "Evaluation", "None",     "Very High",  "Unbiased"),
                ("⑥ Weighted IS",       "Off-policy", "Evaluation", "Yes",      "Low",        "Consistent"),
                ("⑦ Off-policy Control","Off-policy", "Control",    "Slight",   "Medium",     "Converges to π*"),
                ("⑧ Incremental MC",    "On-policy",  "Prediction", "None",     "Low",        "O(1) memory"),
                ("⑨ Per-Decision IS",   "Off-policy", "Evaluation", "None",     "Very Low",   "Unbiased"),
                ("⑩ Discounting IS",    "Off-policy", "Evaluation", "None",     "Lowest",     "Unbiased, γ<1"),
            ]
            for (V,title),(m,cat,typ,bias,var,note) in zip(all_V, META):
                vals = np.array([V.get(s,0) for s in non_t])
                mse  = float(np.mean((vals-ref_v)**2))
                cov  = sum(1 for s in non_t if s in V)
                rows.append({"Method":title,"Type":cat,"Usage":typ,
                             "Bias":bias,"Variance":var,
                             "MSE vs Ref":f"{mse:.4f}","Coverage":f"{cov}/{len(non_t)}",
                             "Notes":note})
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ── Variance bar chart ─────────────────────────────
            section_header("📊", "Variance & Stability Scores (Expert Assessment)")
            labels  = ["FV MC","EV MC","ES","On-pol","Ord.IS","WIS","Off-pol\nCtrl","Incr.","Per-Dec","Disc.IS"]
            var_sc  = [2.0, 2.8, 1.8, 3.5, 9.2, 2.4, 3.0, 2.0, 1.5, 1.0]
            stab_sc = [9.0, 8.5, 9.5, 8.0, 3.0, 8.5, 8.0, 9.0, 8.8, 9.5]
            cmap_c  = METHOD_COLORS[:10]

            fig_s, axes_s, _ = make_fig(1,2, 15,5)
            axes_s[0].barh(labels, var_sc,  color=cmap_c, alpha=0.88, edgecolor=P["border"], lw=0.5)
            axes_s[0].invert_xaxis()
            axes_s[0].set_xlabel("← Lower is better")
            axes_s[0].set_title("Relative Variance Score", color=P["txt"], fontweight="bold")
            axes_s[0].grid(color=P["border"],alpha=0.4,lw=0.5,axis="x")

            axes_s[1].barh(labels, stab_sc, color=cmap_c, alpha=0.88, edgecolor=P["border"], lw=0.5)
            axes_s[1].set_xlabel("Higher is better →")
            axes_s[1].set_title("Stability Score", color=P["txt"], fontweight="bold")
            axes_s[1].grid(color=P["border"],alpha=0.4,lw=0.5,axis="x")
            plt.tight_layout(); st.pyplot(fig_s); plt.close()

            # ── Bias-Variance scatter ──────────────────────────
            section_header("🎯", "Bias–Variance Landscape")
            bvd = {
                "FV MC":        (0.5, 2.0),
                "EV MC":        (1.2, 2.8),
                "ES":           (0.3, 1.8),
                "On-pol":       (2.0, 3.5),
                "Ord.IS":       (0.2, 9.2),
                "WIS":          (2.8, 2.4),
                "Off-pol Ctrl": (1.5, 3.0),
                "Incremental":  (0.5, 2.0),
                "Per-Dec IS":   (0.5, 1.5),
                "Disc.IS":      (0.8, 1.0),
            }
            fig_bv, ax_bv, _ = make_fig(1,1, 9,6)
            ax_bv.add_patch(plt.Rectangle((0,0),3,3,alpha=0.07,color=P["green"],zorder=0))
            ax_bv.text(1.5,0.2,"← Ideal region",color=P["green"],fontsize=8,alpha=0.8)

            for (nm,(bv,va)),col in zip(bvd.items(), cmap_c):
                ax_bv.scatter(bv, va, s=180, color=col, zorder=5, edgecolors="white", lw=0.8)
                ax_bv.annotate(nm,(bv,va),xytext=(7,4),textcoords="offset points",
                               color=P["txt"],fontsize=7.5)

            ax_bv.set_xlabel("Relative Bias →", fontsize=10)
            ax_bv.set_ylabel("Relative Variance →", fontsize=10)
            ax_bv.set_title("Bias–Variance Landscape — All 9 MC Methods",
                            color=P["txt"], fontweight="bold")
            ax_bv.set_xlim(0,10); ax_bv.set_ylim(0,10)
            ax_bv.grid(color=P["border"],alpha=0.4,lw=0.5)
            plt.tight_layout(); st.pyplot(fig_bv); plt.close()

            # ── Method evolution tree ──────────────────────────
            section_header("🌳", "MC Method Family Tree")
            fig_tree, ax_tree, _ = make_fig(1,1, 16,7)
            ax_tree.axis("off"); ax_tree.set_facecolor(P["bg"])

            NODES = [
                # (x,  y,  label,                          color)
                (0.05, 0.50, "MC\nPrediction",             P["blue"]),
                (0.18, 0.80, "First-\nVisit MC",           P["blue"]),
                (0.18, 0.20, "Every-\nVisit MC",           "#4a90d9"),
                (0.18, 0.50, "Increm.\nMC",                P["amber"]),
                (0.38, 0.75, "MC\nControl",                P["green"]),
                (0.38, 0.25, "Off-policy\nEval.",          P["violet"]),
                (0.54, 0.88, "MC Expl.\nStarts",           "#22c55e"),
                (0.54, 0.62, "On-policy\nε-greedy",        P["blue"]),
                (0.54, 0.38, "Ordinary\nIS",               P["rose"]),
                (0.54, 0.12, "Weighted\nIS",               P["cyan"]),
                (0.72, 0.88, "Off-pol\nControl",           P["amber"]),
                (0.72, 0.12, "Per-\nDecision IS",          "#34d399"),
                (0.88, 0.50, "Discounting\nIS",            "#6ee7b7"),
            ]
            EDGES = [(0,1),(0,2),(0,3),(1,4),(1,5),(4,6),(4,7),(5,8),(5,9),(7,10),(9,11),(11,12)]

            for x,y,lbl,col in NODES:
                r = 0.055
                ax_tree.add_patch(plt.Circle((x,y),r,color=col,alpha=0.88,
                                             transform=ax_tree.transAxes,zorder=3))
                ax_tree.text(x,y,lbl,ha="center",va="center",fontsize=6.5,
                             color="white",fontweight="bold",transform=ax_tree.transAxes,zorder=4)

            for i,j in EDGES:
                x0,y0 = NODES[i][:2]; x1,y1 = NODES[j][:2]
                ax_tree.annotate("",xy=(x1,y1),xytext=(x0,y0),
                                 xycoords="axes fraction",textcoords="axes fraction",
                                 arrowprops=dict(arrowstyle="->",color=P["txt2"],lw=1.3,
                                                 connectionstyle="arc3,rad=0.15"))

            ax_tree.set_title("MC Method Genealogy — Simple → Advanced",
                              color=P["txt"],fontweight="bold",fontsize=11,pad=15)
            plt.tight_layout(); st.pyplot(fig_tree); plt.close()

            # ── Legend with stable table ───────────────────────
            st.markdown("""
            <div class="mc-card">
            <table style="width:100%;border-collapse:collapse;font-family:'IBM Plex Mono',monospace;font-size:.78rem">
            <tr style="border-bottom:1px solid #1e2d42">
              <th style="color:#7a8ba6;text-align:left;padding:6px 10px">Method</th>
              <th style="color:#7a8ba6;padding:6px 10px">Bias</th>
              <th style="color:#7a8ba6;padding:6px 10px">Variance</th>
              <th style="color:#7a8ba6;padding:6px 10px">Stability</th>
              <th style="color:#7a8ba6;text-align:left;padding:6px 10px">Converges to</th>
            </tr>
            <tr><td style="color:#93c5fd;padding:5px 10px">① First-Visit MC</td><td style="color:#6ee7b7;text-align:center">None</td><td style="color:#6ee7b7;text-align:center">Low</td><td style="color:#6ee7b7;text-align:center">★★★★★</td><td style="color:#cdd6f4;padding:5px 10px">V^π (exact)</td></tr>
            <tr><td style="color:#93c5fd;padding:5px 10px">② Every-Visit MC</td><td style="color:#fbbf24;text-align:center">Slight</td><td style="color:#6ee7b7;text-align:center">Medium</td><td style="color:#6ee7b7;text-align:center">★★★★☆</td><td style="color:#cdd6f4;padding:5px 10px">V^π (asymptotic)</td></tr>
            <tr><td style="color:#6ee7b7;padding:5px 10px">③ Exploring Starts</td><td style="color:#6ee7b7;text-align:center">None</td><td style="color:#6ee7b7;text-align:center">Low</td><td style="color:#6ee7b7;text-align:center">★★★★★</td><td style="color:#cdd6f4;padding:5px 10px">π* (optimal!)</td></tr>
            <tr><td style="color:#93c5fd;padding:5px 10px">④ On-policy ε-greedy</td><td style="color:#6ee7b7;text-align:center">None</td><td style="color:#fbbf24;text-align:center">Medium</td><td style="color:#6ee7b7;text-align:center">★★★★☆</td><td style="color:#cdd6f4;padding:5px 10px">Best ε-soft π</td></tr>
            <tr><td style="color:#f9a8d4;padding:5px 10px">⑤ Ordinary IS</td><td style="color:#6ee7b7;text-align:center">None</td><td style="color:#f87171;text-align:center">Very High</td><td style="color:#f87171;text-align:center">★★☆☆☆</td><td style="color:#cdd6f4;padding:5px 10px">V^π (unbiased)</td></tr>
            <tr><td style="color:#67e8f9;padding:5px 10px">⑥ Weighted IS</td><td style="color:#fbbf24;text-align:center">Yes</td><td style="color:#6ee7b7;text-align:center">Low</td><td style="color:#6ee7b7;text-align:center">★★★★★</td><td style="color:#cdd6f4;padding:5px 10px">V^π (consistent)</td></tr>
            <tr><td style="color:#fcd34d;padding:5px 10px">⑦ Off-policy Control</td><td style="color:#fbbf24;text-align:center">Slight</td><td style="color:#fbbf24;text-align:center">Medium</td><td style="color:#6ee7b7;text-align:center">★★★★☆</td><td style="color:#cdd6f4;padding:5px 10px">π* (optimal!)</td></tr>
            <tr><td style="color:#fcd34d;padding:5px 10px">⑧ Incremental MC</td><td style="color:#6ee7b7;text-align:center">None</td><td style="color:#6ee7b7;text-align:center">Low</td><td style="color:#6ee7b7;text-align:center">★★★★★</td><td style="color:#cdd6f4;padding:5px 10px">V^π (O(1) mem.)</td></tr>
            <tr><td style="color:#6ee7b7;padding:5px 10px">⑨ Per-Decision IS</td><td style="color:#6ee7b7;text-align:center">None</td><td style="color:#6ee7b7;text-align:center">Very Low</td><td style="color:#6ee7b7;text-align:center">★★★★★</td><td style="color:#cdd6f4;padding:5px 10px">V^π (unbiased)</td></tr>
            <tr><td style="color:#6ee7b7;padding:5px 10px">⑩ Discounting IS</td><td style="color:#6ee7b7;text-align:center">None</td><td style="color:#6ee7b7;text-align:center">Lowest</td><td style="color:#6ee7b7;text-align:center">★★★★★</td><td style="color:#cdd6f4;padding:5px 10px">V^π (γ&lt;1 optimal)</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
