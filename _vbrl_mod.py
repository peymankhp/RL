"""
_vbrl_mod.py  —  Value-Based Deep Reinforcement Learning
Covers: DQN · Double DQN · Dueling DQN · PER · C51 · Rainbow · IQN
Environment: CartPole-style (4-D state, 2 actions) for fast simulation
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DARK  = "#0d0d1a"
CARD  = "#12121f"
GRID  = "#2a2a3e"

# ── colour per algorithm ───────────────────────────────────────────────────
ALG_COL = {
    "DQN":       "#1565c0",
    "Double DQN":"#00897b",
    "Dueling":   "#7c4dff",
    "PER":       "#e65100",
    "C51":       "#ad1457",
    "Rainbow":   "#f57f17",
    "IQN":       "#00838f",
}

# ── shared helpers ─────────────────────────────────────────────────────────
def _fig(nrows=1, ncols=1, w=12, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(DARK)
        ax.tick_params(colors="#9e9ebb", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
    return fig, axes

def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
            f'<b>{icon} {title}</b><br>{body}</div>')

def _tip(t):
    return (f'<div style="background:#1a2a1a;border-left:3px solid #4caf50;'
            f'padding:.6rem 1rem;border-radius:0 6px 6px 0;margin:.5rem 0;font-size:.92rem">{t}</div>')

def _sec(emoji, title, sub, color="#1565c0"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def smooth(a, w=10):
    if len(a) <= w:
        return np.array(a, float)
    return np.convolve(a, np.ones(w)/w, mode="valid")

# ── Tiny CartPole simulator (no gym dependency) ───────────────────────────
class CartPole:
    """Linearised CartPole — 4-D state, 2 actions, episodes ≤200 steps."""
    def __init__(self):
        self.g = 9.8; self.mc = 1.0; self.mp = 0.1; self.l = 0.5
        self.dt = 0.02; self.reset()

    def reset(self):
        self.s = np.random.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.s.copy()

    def step(self, a):
        x, xd, th, thd = self.s
        f = 10.0 if a == 1 else -10.0
        costh = np.cos(th); sinth = np.sin(th)
        tmp = (f + self.mp*self.l*thd**2*sinth) / (self.mc+self.mp)
        thdd = (self.g*sinth - costh*tmp) / (self.l*(4/3 - self.mp*costh**2/(self.mc+self.mp)))
        xdd = tmp - self.mp*self.l*thdd*costh/(self.mc+self.mp)
        self.s = np.array([x+self.dt*xd, xd+self.dt*xdd,
                           th+self.dt*thd, thd+self.dt*thdd])
        self.steps += 1
        done = (abs(self.s[0]) > 2.4 or abs(self.s[2]) > 0.2095 or self.steps >= 200)
        r = 1.0 if not done else 0.0
        return self.s.copy(), r, done

# ── Replay Buffer ──────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = []; self.cap = capacity

    def push(self, s, a, r, ns, d):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append((s, a, r, ns, d))

    def sample(self, n):
        idx = np.random.choice(len(self.buf), n, replace=False)
        b = [self.buf[i] for i in idx]
        return (np.array([x[0] for x in b]),
                np.array([x[1] for x in b]),
                np.array([x[2] for x in b]),
                np.array([x[3] for x in b]),
                np.array([x[4] for x in b], dtype=float))

    def __len__(self): return len(self.buf)

# ── Tiny Q-Network (numpy MLP) ─────────────────────────────────────────────
class QNet:
    def __init__(self, in_dim=4, hid=32, out_dim=2, seed=0):
        np.random.seed(seed)
        k = np.sqrt(2 / in_dim)
        self.W1 = np.random.randn(in_dim, hid)*k
        self.b1 = np.zeros(hid)
        self.W2 = np.random.randn(hid, out_dim)*np.sqrt(2/hid)
        self.b2 = np.zeros(out_dim)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def copy_from(self, other):
        self.W1=other.W1.copy(); self.b1=other.b1.copy()
        self.W2=other.W2.copy(); self.b2=other.b2.copy()

    def update(self, x, targets, lr=0.001):
        """One SGD step — MSE(Q(s), targets) only for selected actions."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)
        q  = h1 @ self.W2 + self.b2
        err = q - targets
        dq  = 2 * err / len(x)
        dW2 = h1.T @ dq
        db2 = dq.sum(0)
        dh1 = dq @ self.W2.T * (h1 > 0)
        dW1 = x.T @ dh1
        db1 = dh1.sum(0)
        self.W1 -= lr*dW1; self.b1 -= lr*db1
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        return float(np.mean(err**2))

# ── DQN Training ───────────────────────────────────────────────────────────
def train_dqn(n_episodes=150, gamma=0.99, lr=0.001,
              eps_start=1.0, eps_end=0.05, eps_decay=0.97,
              batch=64, target_update=10, seed=42,
              double=False, per=False):
    np.random.seed(seed)
    env = CartPole()
    Q  = QNet(seed=seed)
    Qt = QNet(seed=seed); Qt.copy_from(Q)
    buf = ReplayBuffer(5000)
    eps = eps_start
    rewards = []; losses = []; td_errors_log = []

    for ep in range(n_episodes):
        s = env.reset(); done = False; ep_r = 0.0
        while not done:
            if np.random.rand() < eps:
                a = np.random.randint(2)
            else:
                a = int(np.argmax(Q.forward(s)))
            ns, r, done = env.step(a)
            buf.push(s, a, r, ns, done)
            s = ns; ep_r += r

            if len(buf) >= batch:
                S,A,R,NS,D = buf.sample(batch)
                Qvals = Q.forward(S)

                if double:
                    a_next = np.argmax(Q.forward(NS), axis=1)
                    Qt_ns  = Qt.forward(NS)
                    q_next = Qt_ns[np.arange(batch), a_next]
                else:
                    q_next = Qt.forward(NS).max(axis=1)

                targets = Qvals.copy()
                td_err  = R + gamma*q_next*(1-D) - Qvals[np.arange(batch), A]
                targets[np.arange(batch), A] = R + gamma*q_next*(1-D)

                if per:
                    w = (np.abs(td_err)+1e-6)**0.6
                    w /= w.sum()
                    targets *= w[:, None]

                loss = Q.update(S, targets, lr)
                losses.append(loss)
                td_errors_log.append(float(np.mean(np.abs(td_err))))

        rewards.append(ep_r)
        eps = max(eps_end, eps*eps_decay)
        if ep % target_update == 0:
            Qt.copy_from(Q)

    return rewards, losses, td_errors_log

# ═══════════════════════════════════════════════════════════════════════════
def main_vbrl():
# ═══════════════════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0d1b3e,#1a0a0a,#0a1a0d);'
        'border:1px solid #2a3a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🎮 Value-Based Deep RL</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'From DQN to Rainbow — every algorithm that turned raw pixels into superhuman Atari play, '
        'decoded formula by formula with runnable simulations.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🌍 Environment",
        "🧠 DQN",
        "🔄 Double DQN",
        "🏗️ Dueling DQN",
        "🎯 Prioritized Replay",
        "📊 Distributional RL (C51)",
        "🌈 Rainbow & IQN",
        "📈 Dashboard",
        "📚 Study Plan",
    ])
    (tab_env, tab_dqn, tab_ddqn, tab_duel,
     tab_per, tab_c51, tab_rain, tab_dash, tab_plan) = tabs

    # ══════════════════════════════════════════════════════════════════════
    # TAB 0 — ENVIRONMENT
    # ══════════════════════════════════════════════════════════════════════
    with tab_env:
        _sec("🌍","The CartPole Environment",
             "The canonical testbed for value-based RL — simple enough to train in seconds, complex enough to reveal all algorithmic differences","#1565c0")

        st.markdown(_card("#1565c0","🎯","Why CartPole for Value-Based RL",
            """CartPole is the "Hello World" of deep RL: a pole balanced on a cart, controlled by
            pushing left or right. It has a <b>4-dimensional continuous state space</b>
            (cart position, cart velocity, pole angle, pole angular velocity) and <b>2 discrete actions</b>,
            making Q-value estimation straightforward to visualise. Episodes last at most 200 steps —
            short enough to train 150 episodes in seconds. Every algorithmic difference
            (target networks, experience replay, prioritised sampling) produces a measurable signal
            on CartPole, making it ideal for comparing DQN variants side-by-side."""),
            unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(r"""
            **State space** $s = (x,\,\dot{x},\,\theta,\,\dot{\theta})$:
            - $x$ — cart position $\in [-2.4, 2.4]$ m
            - $\dot{x}$ — cart velocity (m/s)
            - $\theta$ — pole angle $\in [-12°, 12°]$ rad
            - $\dot{\theta}$ — pole angular velocity (rad/s)

            **Action space:** $a \in \{0 = \text{push left},\; 1 = \text{push right}\}$

            **Reward:** $+1$ for every step the pole stays upright

            **Termination:** pole angle $>12°$ OR cart $>2.4$ m OR 200 steps reached

            **Optimal policy:** always acts to keep $\theta \approx 0$ — achieves reward **200** every episode
            """)
        with c2:
            st.markdown(r"""
            **Physics equations** (Runge-Kutta approximation):
            """)
            st.latex(r"\ddot{\theta} = \frac{g\sin\theta - \cos\theta\,\frac{F+m_p l\dot\theta^2\sin\theta}{m_c+m_p}}{l\!\left(\frac{4}{3}-\frac{m_p\cos^2\theta}{m_c+m_p}\right)}")
            st.latex(r"\ddot{x} = \frac{F + m_p l(\dot\theta^2\sin\theta - \ddot\theta\cos\theta)}{m_c+m_p}")
            st.markdown(r"""
            - $F = \pm 10\,\text{N}$ — applied force
            - $m_c=1\,\text{kg}$ — cart mass, $m_p=0.1\,\text{kg}$ — pole mass
            - $l=0.5\,\text{m}$ — half-pole length, $g=9.8\,\text{m/s}^2$
            """)

        st.divider()
        st.subheader("🎬 Simulate the environment")
        col_a, col_b = st.columns([1,2])
        policy_type = col_a.selectbox("Test policy",
            ["Random (ε=1.0)","Mostly right","Angle-correcting"], key="env_pol")
        n_ep_env = col_a.slider("Episodes", 5, 50, 20, key="env_ep")
        np.random.seed(77)
        env_sim = CartPole()
        ep_lens = []
        for _ in range(n_ep_env):
            s = env_sim.reset(); done = False; ep_r = 0
            while not done:
                if policy_type == "Random (ε=1.0)":
                    a = np.random.randint(2)
                elif policy_type == "Mostly right":
                    a = 1 if np.random.rand() > 0.3 else 0
                else:
                    a = 1 if s[2] > 0 else 0  # angle-correcting
                s, r, done = env_sim.step(a)
                ep_r += r
            ep_lens.append(ep_r)
        with col_b:
            fig_e, ax_e = _fig(1,1,8,3.5)
            ax_e.bar(range(n_ep_env), ep_lens, color=ALG_COL["DQN"], alpha=0.8, edgecolor="white", lw=0.3)
            ax_e.axhline(np.mean(ep_lens), color="#ffa726", ls="--", lw=1.5,
                         label=f"Mean = {np.mean(ep_lens):.1f}")
            ax_e.axhline(200, color="#4caf50", ls=":", lw=1.2, label="Optimal = 200")
            ax_e.set_xlabel("Episode", color="white"); ax_e.set_ylabel("Episode length", color="white")
            ax_e.set_title(f"Policy: {policy_type}", color="white", fontweight="bold")
            ax_e.legend(facecolor=CARD, labelcolor="white", fontsize=8)
            ax_e.grid(alpha=0.12, axis="y")
            plt.tight_layout(); st.pyplot(fig_e); plt.close()
        st.caption(f"Mean episode length: {np.mean(ep_lens):.1f}  |  Best: {max(ep_lens):.0f}  |  Optimal = 200 (pole never falls)")

        # Q-value intuition
        st.divider()
        st.subheader("📐 What does the Q-Network learn?")
        st.markdown(r"""
        A DQN Q-network learns a function $Q_\theta(s, a)$ that maps state $s$ to estimated return
        for each action $a$. For CartPole with $|A|=2$, the network outputs **2 numbers simultaneously** —
        $Q(s,0)$ and $Q(s,1)$ — with one forward pass:
        """)
        st.latex(r"\hat{Q}(s,\cdot) = W_2\,\text{ReLU}(W_1\,s + b_1) + b_2 \in \mathbb{R}^2")
        st.markdown(r"""
        At any state, the greedy policy takes action $\pi(s) = \arg\max_a Q_\theta(s,a)$.

        **Visualising Q-values:** For a pole that's tilting right ($\theta > 0$), a well-trained Q-network
        should give $Q(s,\text{push right}) > Q(s,\text{push left})$ — push right to counteract the fall.
        """)
        theta_vals = np.linspace(-0.2, 0.2, 100)
        np.random.seed(42)
        qnet_demo = QNet(seed=5)
        q_left  = [qnet_demo.forward(np.array([0, 0, th, 0]))[0] for th in theta_vals]
        q_right = [qnet_demo.forward(np.array([0, 0, th, 0]))[1] for th in theta_vals]
        fig_q, ax_q = _fig(1,1,10,3.5)
        ax_q.plot(np.degrees(theta_vals), q_left,  color="#1565c0", lw=2.2, label="Q(s, push-left)")
        ax_q.plot(np.degrees(theta_vals), q_right, color="#e65100", lw=2.2, label="Q(s, push-right)")
        ax_q.axvline(0, color="white", lw=0.7, alpha=0.4, ls=":")
        ax_q.set_xlabel("Pole angle θ (degrees)", color="white")
        ax_q.set_ylabel("Q value (expected return)", color="white")
        ax_q.set_title("Q-values at different pole angles (untrained random network)", color="white", fontweight="bold")
        ax_q.legend(facecolor=CARD, labelcolor="white"); ax_q.grid(alpha=0.12)
        st.pyplot(fig_q); plt.close()
        st.caption("After training, Q(s, push-right) should be higher when θ > 0 (pole tilting right → push right to correct).")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — DQN
    # ══════════════════════════════════════════════════════════════════════
    with tab_dqn:
        _sec("🧠","Deep Q-Network (DQN)",
             "Mnih et al. 2015 — the paper that started the deep RL revolution","#1565c0")

        st.markdown(_card("#1565c0","🏆","The DQN breakthrough",
            """Before DQN, RL was limited to tabular or hand-crafted feature methods.
            DQN (Mnih et al., 2015) showed that a CNN trained end-to-end with Q-learning
            could achieve <b>superhuman performance on 49 Atari games from raw pixels</b>,
            using the <em>same</em> network architecture and hyperparameters for every game.
            The key innovations were: (1) <b>Experience replay</b> — break temporal correlations
            by storing and randomly sampling transitions; (2) <b>Target network</b> — use a
            periodically-frozen copy of Q to compute stable regression targets.
            Both innovations prevent the catastrophic instability that kills naive deep Q-learning."""),
            unsafe_allow_html=True)

        # Core update
        st.subheader("1. The Core Q-Learning Update")
        st.latex(r"Q(s,a) \leftarrow Q(s,a) + \alpha\bigl[\underbrace{r + \gamma\max_{a'}Q_{\bar\theta}(s',a')}_{\text{TD target }y} - Q_\theta(s,a)\bigr]")
        st.markdown(r"""
        **Symbol decoder:**
        - $Q_\theta(s,a)$ — Q-network with parameters $\theta$ — the network being trained
        - $Q_{\bar\theta}(s',a')$ — **target network** with frozen parameters $\bar\theta$ — provides stable targets
        - $r$ — reward actually received after action $a$ in state $s$
        - $\gamma$ — discount factor (CartPole: 0.99)
        - $y = r + \gamma\max_{a'} Q_{\bar\theta}(s',a')$ — **TD target**: 1 real reward + estimated future
        - $\delta = y - Q_\theta(s,a)$ — **TD error**: how much the target differed from prediction
        - For terminal states: $y = r$ (no future rewards)

        **Viewed as regression:** Train $Q_\theta$ to predict target $y$ via MSE loss:
        """)
        st.latex(r"\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\bigl[(y - Q_\theta(s,a))^2\bigr]")

        # Experience Replay
        st.divider()
        st.subheader("2. Experience Replay — Why It's Non-Negotiable")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **Without experience replay** (naive online training):
            - Step $t$: observe $(s_t, a_t, r_t, s_{t+1})$, update Q immediately
            - Step $t+1$: strongly correlated with step $t$ (same region of state space)
            - Consecutive minibatches are **almost identical** — gradient descent on identical data = oscillation
            - Old transitions are **never revisited** — data used only once then discarded

            **With experience replay:**
            - All transitions $(s,a,r,s')$ stored in a ring buffer $\mathcal{D}$ of capacity $N$ (e.g. $10^6$)
            - Each training step samples a **random minibatch** from $\mathcal{D}$
            - Random sampling **breaks temporal correlations** — each minibatch is diverse
            - Each transition can be **replayed many times** — greatly improved data efficiency
            """)
        with col2:
            # Visual replay buffer
            np.random.seed(7)
            fig_rb, ax_rb = _fig(1,1,6,4)
            ax_rb.axis("off"); ax_rb.set_xlim(0,10); ax_rb.set_ylim(0,7)
            for i in range(10):
                col_buf = ALG_COL["DQN"] if i < 7 else "#1a1a2e"
                ax_rb.add_patch(FancyBboxPatch((i*0.9+0.1, 3.5), 0.8, 1.2,
                    boxstyle="round,pad=0.05", facecolor=col_buf+"55",
                    edgecolor=ALG_COL["DQN"] if i < 7 else GRID, lw=1.5))
                if i < 7:
                    ax_rb.text(i*0.9+0.5, 4.1, f"({i})", ha="center", color="white", fontsize=7)
            ax_rb.text(5, 5.2, "Replay Buffer D (capacity N)", ha="center",
                       color=ALG_COL["DQN"], fontsize=9, fontweight="bold")
            # Sample arrows
            for idx in [1, 4, 6]:
                ax_rb.annotate("", xy=(idx*0.9+0.5, 3.0), xytext=(idx*0.9+0.5, 3.5),
                    arrowprops=dict(arrowstyle="->", color="#ffa726", lw=2))
            ax_rb.add_patch(FancyBboxPatch((1.5, 1.8), 5.5, 1.0,
                boxstyle="round,pad=0.1", facecolor="#ffa72622", edgecolor="#ffa726", lw=2))
            ax_rb.text(4.25, 2.3, "Random minibatch B", ha="center",
                       color="#ffa726", fontsize=9, fontweight="bold")
            ax_rb.annotate("", xy=(4.25, 1.4), xytext=(4.25, 1.8),
                arrowprops=dict(arrowstyle="->", color="#4caf50", lw=2))
            ax_rb.add_patch(FancyBboxPatch((2.0, 0.3), 4.5, 0.9,
                boxstyle="round,pad=0.1", facecolor="#4caf5022", edgecolor="#4caf50", lw=2))
            ax_rb.text(4.25, 0.75, "Q-network update (SGD step)", ha="center",
                       color="#4caf50", fontsize=9, fontweight="bold")
            ax_rb.set_title("Experience Replay Architecture", color="white", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig_rb); plt.close()

        # Target network
        st.divider()
        st.subheader("3. Target Network — Stable Training Targets")
        st.markdown(r"""
        **The instability problem without target networks:**

        If both the prediction $Q_\theta(s,a)$ and the target $y = r + \gamma\max_{a'}Q_\theta(s',a')$
        use the **same** network $\theta$, every gradient step changes both simultaneously.
        It's like chasing a moving target — the network never converges.

        **Solution:** Maintain two networks:
        """)
        st.latex(r"\text{Online network } Q_\theta:\; \text{updated every step}")
        st.latex(r"\text{Target network } Q_{\bar\theta}:\; \text{copied from } Q_\theta \text{ every } C \text{ steps, frozen otherwise}")
        st.markdown(r"""
        Targets $y = r + \gamma\max_{a'} Q_{\bar\theta}(s',a')$ are computed with the **frozen** $\bar\theta$.
        This breaks the feedback loop: the regression target is stable for $C$ steps at a time.
        Typical $C$: 1000–10000 steps.

        **Soft updates** (used in DDPG, SAC, TD3): instead of hard copy every $C$ steps:
        """)
        st.latex(r"\bar\theta \leftarrow \tau\,\theta + (1-\tau)\,\bar\theta \qquad \tau \ll 1 \text{ (e.g. } \tau = 0.005)")

        # Run DQN
        st.divider()
        st.subheader("🎛️ Interactive: Train DQN on CartPole")
        col_c1, col_c2, col_c3 = st.columns(3)
        n_ep   = col_c1.slider("Episodes", 50, 300, 150, 25, key="dqn_ep")
        gamma  = col_c1.slider("γ (discount)", 0.9, 1.0, 0.99, 0.01, key="dqn_gm")
        lr     = col_c2.select_slider("Learning rate", [1e-4,5e-4,1e-3,5e-3,1e-2], 1e-3, key="dqn_lr")
        eps_d  = col_c2.slider("ε decay/episode", 0.90, 0.99, 0.97, 0.005, key="dqn_ed")
        t_upd  = col_c3.slider("Target update (episodes)", 5, 30, 10, 5, key="dqn_tu")
        dqn_seed = col_c3.number_input("Seed", 0, 999, 42, key="dqn_seed")

        if st.button("▶️ Train DQN", type="primary", key="btn_dqn"):
            with st.spinner("Training DQN on CartPole…"):
                r_dqn, l_dqn, td_dqn = train_dqn(n_ep, gamma, lr, 1.0, 0.05, eps_d, 64, t_upd, dqn_seed)
            st.session_state["dqn_results"] = (r_dqn, l_dqn, td_dqn)

        if "dqn_results" in st.session_state:
            r_dqn, l_dqn, td_dqn = st.session_state["dqn_results"]
            fig_d, axes_d = _fig(1,3,16,4)
            # Reward
            sr = smooth(r_dqn, 10)
            axes_d[0].plot(r_dqn, color=ALG_COL["DQN"], alpha=0.2, lw=0.8)
            axes_d[0].plot(range(len(sr)), sr, color=ALG_COL["DQN"], lw=2.5, label="DQN")
            axes_d[0].axhline(195, color="#4caf50", ls="--", lw=1.2, label="Solved (195)")
            axes_d[0].set_xlabel("Episode", color="white"); axes_d[0].set_ylabel("Episode reward", color="white")
            axes_d[0].set_title("Learning Curve", color="white", fontweight="bold")
            axes_d[0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_d[0].grid(alpha=0.12)
            # Loss
            sl = smooth(l_dqn, 50)
            axes_d[1].semilogy(sl, color="#ffa726", lw=2)
            axes_d[1].set_xlabel("Training step", color="white"); axes_d[1].set_ylabel("MSE loss (log)", color="white")
            axes_d[1].set_title("Q-Network Loss", color="white", fontweight="bold"); axes_d[1].grid(alpha=0.12)
            # TD errors
            std = smooth(td_dqn, 50)
            axes_d[2].plot(std, color="#ef5350", lw=2)
            axes_d[2].set_xlabel("Training step", color="white"); axes_d[2].set_ylabel("|TD error|", color="white")
            axes_d[2].set_title("Mean |TD Error| δ", color="white", fontweight="bold"); axes_d[2].grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_d); plt.close()
            c1,c2,c3 = st.columns(3)
            c1.metric("Best episode", f"{max(r_dqn):.0f}/200")
            c2.metric("Late-training mean (last 20)", f"{np.mean(r_dqn[-20:]):.1f}")
            c3.metric("Final mean |TD error|", f"{np.mean(td_dqn[-50:]) if td_dqn else 0:.4f}")

        with st.expander("📐 Full DQN Algorithm Pseudocode"):
            st.code(r"""
Initialise:  Q_θ (online), Q_θ̄ (target, copy of Q_θ)
             D ← empty replay buffer (capacity N)
             ε ← 1.0

For episode = 1 … M:
    s ← env.reset()
    For t = 1 … T:
        # ε-greedy action selection
        a ← random action         with prob ε
            argmax_a Q_θ(s, a)    with prob 1-ε

        # Environment step
        s', r, done ← env.step(a)
        D.push(s, a, r, s', done)
        s ← s'

        # Sample random minibatch from D
        {(sᵢ,aᵢ,rᵢ,sᵢ',doneᵢ)} ~ D   (size B)

        # Compute targets using FROZEN target network Q_θ̄
        yᵢ = rᵢ                              if doneᵢ
             rᵢ + γ max_a' Q_θ̄(sᵢ',a')     otherwise

        # Gradient descent on MSE loss
        θ ← θ - α ∇_θ Σᵢ (yᵢ - Q_θ(sᵢ,aᵢ))²

        # Periodically sync target network
        If t mod C == 0:  θ̄ ← θ

    ε ← max(ε_min, ε × decay)
""", language="text")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — DOUBLE DQN
    # ══════════════════════════════════════════════════════════════════════
    with tab_ddqn:
        _sec("🔄","Double DQN — Eliminating Overestimation Bias",
             "van Hasselt, Guez & Silver, 2016 — a one-line change that consistently improves DQN","#00897b")

        st.markdown(_card("#00897b","📉","The Overestimation Problem",
            r"""DQN's target $y = r + \gamma \max_{a'} Q_{\bar\theta}(s',a')$ has a systematic flaw:
            the <b>maximum of noisy estimates is always higher than the true maximum</b>.
            Early in training, Q-values are noisy random numbers. Taking their max
            selects whichever is the noisiest overestimate — not the truly best action.
            van Hasselt et al. showed this overestimation is <b>systematic and compounding</b>:
            overestimated Q-values propagate backward through Bellman updates, causing the agent
            to overvalue states that are actually mediocre. Double DQN fixes this with
            a single change to the target computation."""), unsafe_allow_html=True)

        st.subheader("1. The Fix — Decouple Action Selection from Value Estimation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DQN target** (coupled — uses same network for both):")
            st.latex(r"y^{\text{DQN}} = r + \gamma \max_{a'} Q_{\bar\theta}(s',a')")
            st.markdown(r"""
            Both selecting the best action AND evaluating it use $Q_{\bar\theta}$.
            Since the same noise drives both, the max picks the noisiest-high estimate
            and evaluates it optimistically — **double dipping on noise**.
            """)
        with col2:
            st.markdown("**Double DQN target** (decoupled):")
            st.latex(r"y^{\text{DDQN}} = r + \gamma Q_{\bar\theta}\!\bigl(s',\,\underbrace{\arg\max_{a'}Q_\theta(s',a')}_{\text{online net selects action}}\bigr)")
            st.markdown(r"""
            - **Online network $Q_\theta$** selects the best action $a^* = \arg\max Q_\theta(s',a')$
            - **Target network $Q_{\bar\theta}$** evaluates $Q_{\bar\theta}(s', a^*)$
            - Since online and target networks have independent noise, the overestimation bias cancels
            """)

        st.subheader("2. Why the Bias is Eliminated")
        st.latex(r"\mathbb{E}\bigl[Q_{\bar\theta}(s',\arg\max Q_\theta)\bigr] \leq \max_{a'} Q(s',a') \quad \text{(unbiased for independent estimates)}")
        st.markdown(r"""
        For any two independent unbiased estimators $Q_A, Q_B$ of the same Q-values, using $Q_A$ to
        select and $Q_B$ to evaluate gives an **unbiased estimate** of the expected value under the
        selecting policy. This is why Double DQN works — online and target networks are trained on
        different (staggered) data, making their errors approximately independent.
        """)

        # Overestimation visualisation
        st.subheader("📊 Visualising Overestimation Bias")
        n_actions = st.slider("Number of actions |A|", 2, 18, 6, key="oe_act")
        noise_std = st.slider("Estimation noise σ", 0.1, 3.0, 1.0, 0.1, key="oe_sig")
        np.random.seed(42)
        n_trials = 5000
        dqn_maxes   = np.array([np.max(np.zeros(n_actions) + np.random.randn(n_actions)*noise_std) for _ in range(n_trials)])
        ddqn_evals  = []
        for _ in range(n_trials):
            qa = np.random.randn(n_actions)*noise_std  # online (selects), true Q=0
            qb = np.random.randn(n_actions)*noise_std  # target (evaluates), true Q=0
            ddqn_evals.append(qb[np.argmax(qa)])
        ddqn_evals = np.array(ddqn_evals)

        fig_oe, axes_oe = _fig(1,2,13,4)
        axes_oe[0].hist(dqn_maxes, bins=60, color=ALG_COL["DQN"], alpha=0.7, density=True, label=f"DQN max: mean={dqn_maxes.mean():.2f}")
        axes_oe[0].hist(ddqn_evals, bins=60, color=ALG_COL["Double DQN"], alpha=0.7, density=True, label=f"DDQN eval: mean={ddqn_evals.mean():.3f}")
        axes_oe[0].axvline(0, color="white", ls="--", lw=2, label="True max Q = 0")
        axes_oe[0].set_xlabel("Estimated max Q-value", color="white")
        axes_oe[0].set_ylabel("Density", color="white")
        axes_oe[0].set_title(f"Overestimation Bias (|A|={n_actions}, σ={noise_std})", color="white", fontweight="bold")
        axes_oe[0].legend(facecolor=CARD, labelcolor="white", fontsize=8)
        axes_oe[0].grid(alpha=0.12)

        act_range = range(2, 19)
        bias_dqn  = [np.mean([np.max(np.random.randn(na)*noise_std) for _ in range(2000)]) for na in act_range]
        bias_ddqn = []
        for na in act_range:
            evals = []
            for _ in range(2000):
                qa=np.random.randn(na)*noise_std; qb=np.random.randn(na)*noise_std
                evals.append(qb[np.argmax(qa)])
            bias_ddqn.append(np.mean(evals))
        axes_oe[1].plot(act_range, bias_dqn,  color=ALG_COL["DQN"],       lw=2.5, marker="o", ms=5, label="DQN bias")
        axes_oe[1].plot(act_range, bias_ddqn, color=ALG_COL["Double DQN"], lw=2.5, marker="s", ms=5, label="DDQN bias")
        axes_oe[1].axhline(0, color="white", ls="--", lw=1.5, alpha=0.6, label="True max = 0 (unbiased)")
        axes_oe[1].set_xlabel("|A| (action space size)", color="white")
        axes_oe[1].set_ylabel("E[estimated max Q]", color="white")
        axes_oe[1].set_title("Bias vs Action Space Size", color="white", fontweight="bold")
        axes_oe[1].legend(facecolor=CARD, labelcolor="white", fontsize=8)
        axes_oe[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_oe); plt.close()
        st.caption(f"With σ={noise_std} and |A|={n_actions}: DQN overestimates Q by {dqn_maxes.mean():.3f}, DDQN by only {ddqn_evals.mean():.3f}")

        # Run comparison
        st.divider()
        st.subheader("🎛️ DQN vs Double DQN — Side-by-Side")
        n_ep2 = st.slider("Episodes", 50, 250, 150, 25, key="ddqn_ep")
        seed2 = st.number_input("Seed", 0, 999, 42, key="ddqn_sd")
        if st.button("▶️ Run Comparison", type="primary", key="btn_ddqn"):
            with st.spinner("Training DQN and Double DQN…"):
                r_dqn2,_,_ = train_dqn(n_ep2, 0.99, 0.001, 1.0, 0.05, 0.97, 64, 10, int(seed2), double=False)
                r_ddqn,_,_ = train_dqn(n_ep2, 0.99, 0.001, 1.0, 0.05, 0.97, 64, 10, int(seed2), double=True)
            st.session_state["ddqn_cmp"] = (r_dqn2, r_ddqn)
        if "ddqn_cmp" in st.session_state:
            r_d, r_dd = st.session_state["ddqn_cmp"]
            fig_cmp, ax_cmp = _fig(1,1,11,4)
            for r, label, col in [(r_d,"DQN",ALG_COL["DQN"]),(r_dd,"Double DQN",ALG_COL["Double DQN"])]:
                ax_cmp.plot(r, color=col, alpha=0.15, lw=0.7)
                sm = smooth(r, 10)
                ax_cmp.plot(range(len(sm)), sm, color=col, lw=2.5, label=f"{label} (mean late={np.mean(r[-30:]):.1f})")
            ax_cmp.axhline(195, color="white", ls="--", lw=1, alpha=0.5, label="Solved=195")
            ax_cmp.set_xlabel("Episode", color="white"); ax_cmp.set_ylabel("Episode reward", color="white")
            ax_cmp.set_title("DQN vs Double DQN on CartPole", color="white", fontweight="bold")
            ax_cmp.legend(facecolor=CARD, labelcolor="white"); ax_cmp.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_cmp); plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — DUELING DQN
    # ══════════════════════════════════════════════════════════════════════
    with tab_duel:
        _sec("🏗️","Dueling Network Architecture",
             "Wang et al. 2016 — decompose Q into state value V and action advantage A","#7c4dff")

        st.markdown(_card("#7c4dff","🧩","The key insight",
            """In many states, the <b>action doesn't matter much</b>. If you're standing in a safe open
            field in a video game, pushing left vs right makes little difference — the state just has
            high value regardless. Standard Q-networks must re-learn this separately for every action.
            Dueling networks explicitly decompose $Q(s,a) = V(s) + A(s,a)$:
            $V(s)$ = how good is this state (action-independent), $A(s,a)$ = how much better is
            action $a$ compared to the average action. The two streams share the convolutional encoder
            and are trained jointly, allowing faster generalisation across actions."""),
            unsafe_allow_html=True)

        st.subheader("1. The Dueling Decomposition")
        st.latex(r"Q(s,a;\,\theta,\alpha,\beta) = V(s;\,\theta,\beta) + \Bigl(A(s,a;\,\theta,\alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'}A(s,a';\,\theta,\alpha)\Bigr)")
        st.markdown(r"""
        **Symbol decoder:**
        - $V(s;\theta,\beta) \in \mathbb{R}$ — **state value** stream: scalar, how good is being in state $s$
        - $A(s,a;\theta,\alpha) \in \mathbb{R}^{|A|}$ — **advantage** stream: vector, how much better is each action
        - $\frac{1}{|A|}\sum_{a'} A(s,a')$ — mean advantage subtracted for **identifiability**
        - $\theta$ — shared encoder parameters (CNN feature extractor)
        - $\beta$ — value stream parameters (separate MLP head)
        - $\alpha$ — advantage stream parameters (separate MLP head)

        **Why subtract the mean advantage?**
        Without it, $V$ and $A$ are unidentifiable: $(V+c, A-c)$ gives the same $Q$ for any constant $c$.
        Subtracting the mean forces $\mathbb{E}_{a\sim\text{uniform}}[A(s,a)] = 0$,
        making $V$ and $A$ unique.
        """)

        # Architecture diagram
        st.subheader("2. Network Architecture")
        fig_arch, ax_arch = _fig(1,1,13,5)
        ax_arch.axis("off"); ax_arch.set_xlim(0,13); ax_arch.set_ylim(0,6)

        def boxt(ax, x, y, txt, col, w=2.5, h=0.8, fs=8):
            ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
                boxstyle="round,pad=0.08",facecolor=col+"33",edgecolor=col,lw=2,zorder=3))
            ax.text(x,y,txt,ha="center",va="center",color="white",fontsize=fs,fontweight="bold",zorder=4)

        boxt(ax_arch,1.2,3,"Input\ns∈ℝ⁴","#546e7a",1.8,.8)
        boxt(ax_arch,3.5,3,"Shared\nEncoder\n(MLP)","#546e7a",1.8,1.2)
        boxt(ax_arch,6.5,4.2,"Value Stream\nV(s; β)","#4caf50",2.2,1.0)
        boxt(ax_arch,6.5,1.8,"Advantage\nStream A(s,a; α)","#7c4dff",2.2,1.0)
        boxt(ax_arch,9.8,4.2,"V(s)\n∈ ℝ","#4caf50",1.5,.8)
        boxt(ax_arch,9.8,1.8,"A(s,·)\n∈ ℝ|A|","#7c4dff",1.5,.8)
        boxt(ax_arch,12.0,3,"Q(s,a)\n= V + Ã","#ffa726",1.6,.8)

        for x1,y1,x2,y2 in [(2.1,3,2.6,3),(4.4,3.4,5.4,4.2),(4.4,2.6,5.4,1.8),
                              (7.6,4.2,9.05,4.2),(7.6,1.8,9.05,1.8),
                              (10.55,4.2,11.2,3.2),(10.55,1.8,11.2,2.8)]:
            ax_arch.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="->",color="#90a4ae",lw=1.8))
        ax_arch.text(6.5,5.5,"DUELING ARCHITECTURE",ha="center",color="white",fontsize=11,fontweight="bold")
        ax_arch.text(9.8,0.7,"Q=V+(A−mean(A))",ha="center",color="#ffa726",fontsize=8)
        plt.tight_layout(); st.pyplot(fig_arch); plt.close()

        # Advantage intuition
        st.subheader("3. What V and A Learn")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **State value $V(s)$** captures "how good is this position":
            - High for states near the goal (few steps to success)
            - Low for states near failure (pole about to fall)
            - **Action-independent** — the same regardless of which action you take
            - Once learned for one action, generalises to all actions in that state
            """)
        with col2:
            st.markdown(r"""
            **Advantage $A(s,a)$** captures "how much better is this action":
            - $A(s,a) > 0$: action $a$ is better than average in state $s$
            - $A(s,a) < 0$: action $a$ is worse than average
            - $A(s,a) \approx 0$ for all $a$: actions are equally good — only $V(s)$ matters
            - This is common in early-episode "safe" states — $V$ updates efficiently
            """)

        # Visualise V and A for CartPole
        np.random.seed(42)
        theta_d = np.linspace(-0.2, 0.2, 100)
        qnet_d  = QNet(seed=5)
        q_vals  = np.array([qnet_d.forward(np.array([0,0,th,0])) for th in theta_d])
        v_vals  = q_vals.mean(axis=1)
        a_vals  = q_vals - v_vals[:,None]
        fig_va, axes_va = _fig(1,3,15,4)
        axes_va[0].plot(np.degrees(theta_d), q_vals[:,0], color="#1565c0", lw=2, label="Q(s,left)")
        axes_va[0].plot(np.degrees(theta_d), q_vals[:,1], color="#e65100", lw=2, label="Q(s,right)")
        axes_va[0].set_title("Q-values", color="white",fontweight="bold")
        axes_va[0].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_va[0].grid(alpha=0.12)
        axes_va[1].plot(np.degrees(theta_d), v_vals, color="#4caf50", lw=2.5)
        axes_va[1].set_title("State Value V(s) = mean(Q)", color="white",fontweight="bold"); axes_va[1].grid(alpha=0.12)
        axes_va[2].plot(np.degrees(theta_d), a_vals[:,0], color="#1565c0", lw=2, label="A(s,left)")
        axes_va[2].plot(np.degrees(theta_d), a_vals[:,1], color="#e65100", lw=2, label="A(s,right)")
        axes_va[2].axhline(0, color="white", lw=0.5, alpha=0.4)
        axes_va[2].set_title("Advantage A(s,a) = Q - V", color="white",fontweight="bold")
        axes_va[2].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_va[2].grid(alpha=0.12)
        for ax in np.array(axes_va).flatten():
            ax.set_xlabel("Pole angle θ (°)", color="white")
        plt.tight_layout(); st.pyplot(fig_va); plt.close()
        st.caption("For a random untrained network. After training: V should be high near θ=0 (safe) and low at edges (about to fall); A should show push-right > push-left when θ > 0.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — PRIORITIZED EXPERIENCE REPLAY
    # ══════════════════════════════════════════════════════════════════════
    with tab_per:
        _sec("🎯","Prioritized Experience Replay (PER)",
             "Schaul et al. 2016 — sample important transitions more often by their TD error","#e65100")

        st.markdown(_card("#e65100","⚡","Why uniform sampling wastes learning",
            r"""Uniform replay samples every stored transition with equal probability.
            But transitions vary enormously in their <b>informational content</b>:
            a transition where the TD error $|\delta| \approx 0$ is already well-predicted —
            replaying it again teaches the network almost nothing. A transition with
            $|\delta| = 5$ reveals a large gap between prediction and reality —
            the most valuable learning opportunity available. PER gives
            high-$|\delta|$ transitions proportionally more replay probability,
            focusing computation where it matters most."""), unsafe_allow_html=True)

        st.subheader("1. Priority-Based Sampling Probability")
        st.latex(r"p_i = \bigl(|\delta_i| + \varepsilon\bigr)^\alpha")
        st.latex(r"P(i) = \frac{p_i}{\sum_k p_k}")
        st.markdown(r"""
        **Symbol decoder:**
        - $|\delta_i|$ — absolute TD error for transition $i$: $|r+\gamma\max_{a'}Q(s',a') - Q(s,a)|$
        - $\varepsilon$ — small constant preventing zero priority (e.g. $10^{-6}$)
        - $\alpha \in [0,1]$ — controls how much priority matters: $\alpha=0$ → uniform, $\alpha=1$ → fully prioritised
        - $P(i)$ — probability of sampling transition $i$
        """)

        st.subheader("2. Importance Sampling Correction")
        st.markdown(r"""
        PER changes the sampling distribution — updates are no longer unbiased.
        **Importance sampling (IS) weights** correct for this:
        """)
        st.latex(r"w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta")
        st.markdown(r"""
        - $N$ — total number of transitions in the buffer
        - $\beta \in [0,1]$ — correction strength: $\beta=0$ → no correction (full bias), $\beta=1$ → fully corrected
        - Practice: anneal $\beta: 0.4 \to 1.0$ during training (more correction as learning stabilises)
        - Gradient update: $\theta \leftarrow \theta - \alpha\, w_i\, \delta_i\, \nabla Q_\theta(s_i,a_i)$
        """)

        # Interactive demo
        st.subheader("🎛️ Interactive: See PER Sampling Distributions")
        col_p1, col_p2, col_p3 = st.columns(3)
        n_trans  = col_p1.slider("Buffer size N", 20, 200, 50, 10, key="per_n")
        alpha_p  = col_p2.slider("α (priority exponent)", 0.0, 1.0, 0.6, 0.05, key="per_a")
        beta_p   = col_p3.slider("β (IS correction)", 0.0, 1.0, 0.4, 0.05, key="per_b")

        np.random.seed(42)
        td_errors_p = np.abs(np.random.exponential(1.0, n_trans))
        priorities  = (td_errors_p + 1e-6)**alpha_p
        probs       = priorities / priorities.sum()
        is_weights  = (1/(n_trans*probs))**beta_p
        is_weights /= is_weights.max()

        fig_per, axes_per = _fig(1,3,15,4)
        # TD errors
        axes_per[0].bar(range(n_trans), td_errors_p, color="#546e7a", alpha=0.7)
        top5 = np.argsort(td_errors_p)[-5:]
        axes_per[0].bar(top5, td_errors_p[top5], color="#ef5350", alpha=0.9, label="Top 5 |δ|")
        axes_per[0].set_title("|TD Error| per transition", color="white",fontweight="bold")
        axes_per[0].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_per[0].grid(alpha=0.12,axis="y")
        # Sampling probs
        uniform_p = np.ones(n_trans)/n_trans
        axes_per[1].bar(range(n_trans), probs, color=ALG_COL["PER"], alpha=0.8, label=f"PER (α={alpha_p})")
        axes_per[1].axhline(uniform_p[0], color="white", ls="--", lw=1.5, label="Uniform")
        axes_per[1].set_title("Sampling Probability P(i)", color="white",fontweight="bold")
        axes_per[1].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_per[1].grid(alpha=0.12,axis="y")
        # IS weights
        axes_per[2].bar(range(n_trans), is_weights, color=ALG_COL["Double DQN"], alpha=0.8)
        axes_per[2].set_title(f"IS weights wᵢ (β={beta_p}) — corrects sampling bias", color="white",fontweight="bold")
        axes_per[2].grid(alpha=0.12,axis="y")
        for ax in np.array(axes_per).flatten():
            ax.set_xlabel("Transition index", color="white")
        plt.tight_layout(); st.pyplot(fig_per); plt.close()
        gini = 1 - np.sum(probs**2) * n_trans
        st.markdown(f"""
        **Concentration stats:** Max P(i) = {probs.max():.4f} (uniform = {1/n_trans:.4f})
        | Effective buffer size = {1/np.sum(probs**2):.1f} / {n_trans}
        | Distribution uniformity = {gini:.3f} (1.0 = perfectly uniform)
        """)

        # SumTree data structure
        with st.expander("🌳 SumTree — Efficient O(log N) PER Implementation"):
            st.markdown(r"""
            Naively sampling by priority requires $O(N)$ per step — too slow for replay buffers of size $10^6$.
            A **SumTree** data structure enables $O(\log N)$ sampling:

            - Binary tree where each **leaf** = priority $p_i$ of one transition
            - Each **internal node** = sum of children's priorities
            - Root = total priority sum $\sum_k p_k$
            - **Sampling:** draw $u \sim \text{Uniform}(0, \text{root})$, traverse tree in $O(\log N)$ steps
            - **Update:** after each training step, update leaf $i$ priority in $O(\log N)$ steps
            """)
            st.code('''
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)  # complete binary tree
        self.data = [None]*capacity
        self.write = 0

    def update(self, idx, priority):
        # Update leaf priority and propagate to root in O(log N)
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx-1)//2  # parent
            self.tree[idx] += change

    def sample(self, value):
        # Find leaf with cumulative priority >= value in O(log N)
        idx = 0  # start at root
        while idx < self.capacity - 1:
            left = 2*idx + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]
''', language="python")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — C51 DISTRIBUTIONAL RL
    # ══════════════════════════════════════════════════════════════════════
    with tab_c51:
        _sec("📊","Distributional RL — C51",
             "Bellemare, Dabney & Munos 2017 — predict the full return distribution, not just its mean","#ad1457")

        st.markdown(_card("#ad1457","🎲","Why predict distributions instead of expected values?",
            r"""Standard DQN learns $\mathbb{E}[G_t | s, a]$ — the <b>expected</b> return.
            But two state-action pairs can have the same expected return with wildly different
            distributions: one always returning 50, another returning 0 or 100 with equal probability.
            <b>C51 (Categorical-51)</b> learns the <b>full probability distribution</b> of returns $Z(s,a)$,
            not just its mean. Benefits: richer information for gradient updates, intrinsic uncertainty
            quantification, and better handling of multi-modal return distributions
            (e.g. sometimes the enemy kills you, sometimes you win). The "51" refers to using
            51 discrete atoms to represent the return distribution."""), unsafe_allow_html=True)

        st.subheader("1. The Return Distribution")
        st.markdown(r"Instead of the expected value $Q(s,a) = \mathbb{E}[Z(s,a)]$, model the random variable:")
        st.latex(r"Z(s,a) \overset{\text{def}}{=} \sum_{t=0}^{\infty} \gamma^t R_{t+1} \;\Big|\; S_0=s, A_0=a")
        st.markdown(r"""
        **The distributional Bellman operator:**
        """)
        st.latex(r"\mathcal{T}Z(s,a) \overset{D}{=} R + \gamma Z(S',A')")
        st.markdown(r"""
        where $\overset{D}{=}$ means equality in distribution.

        **C51 parameterisation:** Represent $Z(s,a)$ as a categorical distribution over $N=51$ fixed atoms:
        """)
        st.latex(r"z_i = V_{\min} + i\,\frac{V_{\max}-V_{\min}}{N-1}, \quad i = 0,\ldots,N-1")
        st.latex(r"Z(s,a) = \sum_{i=0}^{N-1} p_i(s,a)\,\delta_{z_i}, \quad p_i(s,a) = \text{softmax}(f_\theta(s,a))_i")
        st.markdown(r"""
        The network outputs $N \times |A|$ logits → softmax → probability mass on each atom for each action.
        Expected Q-value (for action selection): $Q(s,a) = \sum_i z_i p_i(s,a)$
        """)

        # Interactive: show distributions
        st.subheader("🎛️ Interactive: Return Distributions for Two Actions")
        col_c1, col_c2 = st.columns(2)
        left_mu   = col_c1.slider("Left action — return mean", -10., 10., 2., 0.5, key="c51_lm")
        left_sig  = col_c1.slider("Left action — spread σ", 0.5, 5., 1.5, 0.1, key="c51_ls")
        right_mu  = col_c2.slider("Right action — return mean", -10., 10., 4., 0.5, key="c51_rm")
        right_sig = col_c2.slider("Right action — spread σ", 0.5, 5., 3., 0.1, key="c51_rs")

        N_atoms = 51
        v_min, v_max = -10., 10.
        atoms = np.linspace(v_min, v_max, N_atoms)
        def gauss_probs(mu, sig, atoms):
            p = np.exp(-0.5*((atoms-mu)/sig)**2)
            return p / p.sum()
        pl = gauss_probs(left_mu, left_sig, atoms)
        pr = gauss_probs(right_mu, right_sig, atoms)
        ql = np.sum(atoms*pl); qr = np.sum(atoms*pr)

        fig_c51, ax_c51 = _fig(1,1,11,4)
        ax_c51.bar(atoms, pl, width=0.38, color="#1565c0", alpha=0.75,
                   label=f"Left: E[Z]={ql:.2f}, σ={left_sig:.1f}")
        ax_c51.bar(atoms+0.4, pr, width=0.38, color="#e65100", alpha=0.75,
                   label=f"Right: E[Z]={qr:.2f}, σ={right_sig:.1f}")
        ax_c51.axvline(ql, color="#1565c0", ls="--", lw=2, alpha=0.8)
        ax_c51.axvline(qr, color="#e65100", ls="--", lw=2, alpha=0.8)
        ax_c51.set_xlabel("Return value z", color="white")
        ax_c51.set_ylabel("Probability p(z)", color="white")
        ax_c51.set_title(f"Return Distributions — {'Left' if ql > qr else 'Right'} action has higher expected Q",
                         color="white", fontweight="bold")
        ax_c51.legend(facecolor=CARD, labelcolor="white"); ax_c51.grid(alpha=0.12, axis="y")
        plt.tight_layout(); st.pyplot(fig_c51); plt.close()
        st.markdown(f"""
        Both actions with same expected return? DQN would be **indifferent**.
        C51 captures the full shape — useful for risk-aware or intrinsically-motivated RL.
        """)

        st.subheader("2. The C51 Training Target — Projected Distribution")
        st.markdown(r"""
        After taking action $a$ in state $s$ and observing $r, s'$, the Bellman target distribution is:
        """)
        st.latex(r"\hat{\mathcal{T}}z_j = r + \gamma z_j \quad \text{for each atom } j")
        st.markdown(r"""
        This shifted distribution must be **projected back** onto the fixed atom support $\{z_i\}$:
        """)
        st.latex(r"(\Phi\hat{\mathcal{T}}Z)_i = \sum_j \left[1-\frac{|\hat{\mathcal{T}}z_j - z_i|}{\Delta z}\right]_0^1 p_j(s',a^*)")
        st.markdown(r"""
        where $\Delta z = (V_{\max}-V_{\min})/(N-1)$ is the atom spacing.
        Loss: **KL divergence** between the projected target distribution and the predicted distribution.
        """)
        st.latex(r"\mathcal{L}(\theta) = -\sum_i (\Phi\hat{\mathcal{T}}Z)_i \log p_i(s,a;\theta)")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — RAINBOW & IQN
    # ══════════════════════════════════════════════════════════════════════
    with tab_rain:
        _sec("🌈","Rainbow & IQN — Combining Everything",
             "Hessel et al. 2018 — all improvements combined; Dabney et al. 2018 — implicit quantile networks","#f57f17")

        st.markdown(_card("#f57f17","🌈","Rainbow — 6 improvements in one agent",
            """Rainbow (Hessel et al., 2018) combines six DQN improvements and shows that together
            they are <b>better than any individual component</b>. On the Atari-57 benchmark,
            Rainbow achieves human-level performance in 7× fewer frames than DQN,
            and surpasses the best individual extension on 40 of 57 games.
            The six components work synergistically: PER identifies the most valuable transitions;
            multi-step returns provide richer targets; distributional learning provides better gradients;
            Double DQN prevents overestimation of those richer targets."""), unsafe_allow_html=True)

        st.subheader("1. The 6 Rainbow Components")
        components = [
            ("🧠","DQN","Base algorithm","Experience replay + target network","Stable online learning from pixels"),
            ("🔄","Double DQN","+Decoupled targets",r"$y = r+\gamma Q_{\bar\theta}(s',\arg\max Q_\theta)$","Removes overestimation bias"),
            ("🏗️","Dueling","+V+A streams","$Q(s,a)=V(s)+A(s,a)-\\bar{A}(s)$","Faster V learning in action-independent states"),
            ("🎯","PER","+Priority sampling","$P(i) \\propto |\\delta_i|^\\alpha$","Focus compute on surprising transitions"),
            ("🪜","Multi-step","+n-step returns","$G_t^{(n)}=\\sum_{k=0}^{n-1}\\gamma^k R_{t+k+1}+\\gamma^n Q(s_{t+n})$","Propagate reward signal faster"),
            ("📊","C51","+Distributional","$Z(s,a)\\sim$ Categorical(51)","Richer gradient signal, handles multimodal returns"),
        ]
        df_rain = pd.DataFrame({"Icon":[c[0] for c in components],
                                 "Component":[c[1] for c in components],
                                 "What it adds":[c[2] for c in components],
                                 "Key mechanism":[c[4] for c in components]})
        st.dataframe(df_rain, use_container_width=True, hide_index=True)

        # Ablation study visualisation
        st.subheader("2. Rainbow Ablation — Each Component's Contribution")
        np.random.seed(42)
        n_frames = 200
        frames = np.linspace(0, 200, n_frames)
        ablation_data = {
            "Rainbow (all)":     (np.clip(40*(1-np.exp(-frames/30))+np.random.randn(n_frames)*1.5, 0, 50), ALG_COL["Rainbow"]),
            "w/o PER":           (np.clip(35*(1-np.exp(-frames/45))+np.random.randn(n_frames)*2.0, 0, 50), "#546e7a"),
            "w/o Distributional":(np.clip(32*(1-np.exp(-frames/40))+np.random.randn(n_frames)*2.0, 0, 50), ALG_COL["C51"]),
            "w/o Multi-step":    (np.clip(30*(1-np.exp(-frames/50))+np.random.randn(n_frames)*2.0, 0, 50), ALG_COL["Double DQN"]),
            "DQN (baseline)":    (np.clip(18*(1-np.exp(-frames/80))+np.random.randn(n_frames)*2.5, 0, 50), ALG_COL["DQN"]),
        }
        fig_ab, ax_ab = _fig(1,1,11,4.5)
        for name, (scores, col) in ablation_data.items():
            sm = smooth(scores, 15)
            ax_ab.plot(frames[:len(sm)], sm, color=col, lw=2.5 if name=="Rainbow (all)" else 1.8,
                       label=name, ls="-" if "Rainbow" in name or "DQN" in name else "--")
        ax_ab.set_xlabel("Training frames (×10⁶)", color="white")
        ax_ab.set_ylabel("Median human-normalised score", color="white")
        ax_ab.set_title("Rainbow Ablation Study (Simulated)", color="white", fontweight="bold")
        ax_ab.legend(facecolor=CARD, labelcolor="white", fontsize=8, loc="upper left")
        ax_ab.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ab); plt.close()
        st.caption("Simulated to illustrate relative contributions. True Rainbow ablation: see Hessel et al. (2018), Figure 2.")

        # Multi-step returns
        st.divider()
        st.subheader("3. Multi-step Returns — Bridging TD and Monte Carlo")
        st.markdown(r"Instead of the 1-step TD target, use $n$ real rewards before bootstrapping:")
        st.latex(r"G_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k R_{t+k+1} + \gamma^n Q_{\bar\theta}(S_{t+n}, A_{t+n})")
        st.markdown(r"""
        - $n=1$: standard DQN (full bootstrap)
        - $n=\infty$: full Monte Carlo (no bootstrap, high variance)
        - Rainbow: $n=3$ (empirically best on Atari)
        - **Why it helps:** with $n=3$, rewards propagate 3× faster to early states;
          credit assignment is less reliant on accurate future value estimates
        """)

        # IQN
        st.divider()
        st.subheader("4. IQN — Implicit Quantile Networks")
        st.markdown(_card("#00838f","📐","IQN — Continuous distributional RL",
            """C51 uses 51 fixed atoms to represent the return distribution.
            IQN (Dabney et al., 2018) instead learns to map any <b>quantile level τ ∈ (0,1)</b>
            to the corresponding return quantile $F_Z^{-1}(τ)$.
            This gives infinite resolution: the network implicitly represents
            the full continuous CDF of returns, not a discretised approximation."""),
            unsafe_allow_html=True)
        st.latex(r"Z_\tau(s,a) = f_\theta(s,a;\,\phi(\tau)) \quad \tau \sim \text{Uniform}(0,1)")
        st.latex(r"\phi(\tau) = \text{ReLU}\!\left(\sum_{j=1}^{n} \cos(\pi j\tau)\,w_j + b_j\right)")
        st.markdown(r"""
        - $\tau$ — sampled quantile level; $\tau=0.1$ → 10th percentile (pessimistic), $\tau=0.9$ → 90th percentile (optimistic)
        - $\phi(\tau)$ — cosine embedding of $\tau$ into a learnable feature vector
        - Loss: **quantile Huber loss** (asymmetric to correctly train each quantile)
        """)
        st.latex(r"\mathcal{L}_\kappa(\tau) = \mathbb{E}_{\tau'}\!\left[\rho_\tau^\kappa\!\left(r+\gamma Z_{\tau'}(s',a^*) - Z_\tau(s,a)\right)\right]")
        st.markdown(r"""where $\rho_\tau^\kappa$ is the asymmetric Huber loss at quantile $\tau$.""")

        # IQN quantile demo
        st.subheader("🎛️ IQN Quantile Distribution Demo")
        iqn_mu  = st.slider("Return distribution mean", -5., 15., 8., 0.5, key="iqn_mu")
        iqn_sig = st.slider("Return distribution spread", 0.5, 6., 2.5, 0.1, key="iqn_sig")
        taus = np.linspace(0.01, 0.99, 200)
        quantiles = iqn_mu + iqn_sig*np.sqrt(2)*np.array([float(np.sign(t-0.5))*abs(2*min(t,1-t)-1)**0.5*2 for t in taus])
        from scipy.stats import norm as sp_norm
        try:
            quantiles = sp_norm.ppf(taus, loc=iqn_mu, scale=iqn_sig)
        except Exception:
            pass
        fig_iqn, axes_iqn = _fig(1,2,13,4)
        axes_iqn[0].plot(taus, quantiles, color=ALG_COL["IQN"], lw=2.5)
        axes_iqn[0].fill_between(taus[taus<0.1], quantiles[taus<0.1], quantiles.max(),
                                  alpha=0.3, color="#ef5350", label="Pessimistic τ<0.1")
        axes_iqn[0].fill_between(taus[taus>0.9], quantiles[taus>0.9], quantiles.max(),
                                  alpha=0.3, color="#4caf50", label="Optimistic τ>0.9")
        axes_iqn[0].set_xlabel("Quantile level τ", color="white")
        axes_iqn[0].set_ylabel("Return quantile F⁻¹(τ)", color="white")
        axes_iqn[0].set_title("Quantile Function (Inverse CDF)", color="white", fontweight="bold")
        axes_iqn[0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_iqn[0].grid(alpha=0.12)
        # PDF
        x_pdf = np.linspace(iqn_mu-4*iqn_sig, iqn_mu+4*iqn_sig, 300)
        try:
            y_pdf = sp_norm.pdf(x_pdf, loc=iqn_mu, scale=iqn_sig)
        except Exception:
            y_pdf = np.exp(-0.5*((x_pdf-iqn_mu)/iqn_sig)**2)/(iqn_sig*np.sqrt(2*np.pi))
        axes_iqn[1].plot(x_pdf, y_pdf, color=ALG_COL["IQN"], lw=2.5)
        axes_iqn[1].fill_between(x_pdf[x_pdf<iqn_mu-iqn_sig], y_pdf[x_pdf<iqn_mu-iqn_sig],
                                  alpha=0.3, color="#ef5350", label="Pessimistic tail")
        axes_iqn[1].fill_between(x_pdf[x_pdf>iqn_mu+iqn_sig], y_pdf[x_pdf>iqn_mu+iqn_sig],
                                  alpha=0.3, color="#4caf50", label="Optimistic tail")
        axes_iqn[1].axvline(iqn_mu, color="white", ls="--", lw=1.5, label=f"Mean = {iqn_mu:.1f}")
        axes_iqn[1].set_xlabel("Return value z", color="white"); axes_iqn[1].set_ylabel("Density", color="white")
        axes_iqn[1].set_title("Return Distribution (PDF)", color="white", fontweight="bold")
        axes_iqn[1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_iqn[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_iqn); plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 7 — DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    with tab_dash:
        _sec("📈","Algorithm Dashboard","Run all value-based RL variants and compare them on CartPole","#f57f17")

        col_d1, col_d2, col_d3 = st.columns(3)
        n_ep_d  = col_d1.slider("Episodes", 50, 250, 150, 25, key="dash_ep")
        seed_d  = col_d2.number_input("Seed", 0, 999, 42, key="dash_sd")
        run_all = col_d3.button("🚀 Run All Algorithms", type="primary", key="btn_dash")

        if run_all:
            results = {}
            with st.spinner("Training DQN…"):
                r,_,_ = train_dqn(n_ep_d,0.99,0.001,1.,0.05,0.97,64,10,int(seed_d),False,False)
                results["DQN"] = r
            with st.spinner("Training Double DQN…"):
                r,_,_ = train_dqn(n_ep_d,0.99,0.001,1.,0.05,0.97,64,10,int(seed_d),True,False)
                results["Double DQN"] = r
            with st.spinner("Training DQN + PER…"):
                r,_,_ = train_dqn(n_ep_d,0.99,0.001,1.,0.05,0.97,64,10,int(seed_d),False,True)
                results["DQN+PER"] = r
            with st.spinner("Training Double DQN + PER…"):
                r,_,_ = train_dqn(n_ep_d,0.99,0.001,1.,0.05,0.97,64,10,int(seed_d),True,True)
                results["Double+PER"] = r
            st.session_state["dash_results"] = results

        if "dash_results" in st.session_state:
            res = st.session_state["dash_results"]
            fig_d, ax_d = _fig(1,1,12,5)
            cols_d = [ALG_COL["DQN"],ALG_COL["Double DQN"],ALG_COL["PER"],"#f57f17"]
            for (name,rews),col in zip(res.items(),cols_d):
                sm = smooth(rews, 12)
                ax_d.plot(rews, color=col, alpha=0.12, lw=0.5)
                ax_d.plot(range(len(sm)), sm, color=col, lw=2.5,
                          label=f"{name} (mean-late={np.mean(rews[-30:]):.1f})")
            ax_d.axhline(195, color="white", ls="--", lw=1, alpha=0.5, label="Solved = 195")
            ax_d.set_xlabel("Episode", color="white"); ax_d.set_ylabel("Reward", color="white")
            ax_d.set_title("Value-Based RL Algorithms on CartPole", color="white", fontweight="bold")
            ax_d.legend(facecolor=CARD, labelcolor="white", fontsize=9)
            ax_d.grid(alpha=0.12); plt.tight_layout(); st.pyplot(fig_d); plt.close()

            st.subheader("📋 Performance Summary")
            rows=[]
            for name,rews in res.items():
                rows.append({"Algorithm":name,
                             "Mean reward (last 30)":f"{np.mean(rews[-30:]):.1f}",
                             "Best episode":f"{max(rews):.0f}",
                             "Episodes to 150":next((i for i,r in enumerate(smooth(rews,5)) if r>=150),"—"),
                             "Stability (σ)":f"{np.std(rews[-30:]):.1f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Summary table — all 7 algorithms
        st.divider()
        st.subheader("📊 Complete Algorithm Comparison")
        df_all = pd.DataFrame({
            "Algorithm":["DQN","Double DQN","Dueling DQN","PER","C51","Rainbow","IQN"],
            "Key innovation":["ER + target net","Decouple select/eval","V+A decomposition",
                              "Priority sampling","Full distribution","All combined","Implicit quantiles"],
            "Fixes":["Catastrophic forgetting","Overestimation bias","Action-independent V",
                     "Sample inefficiency","Mean-only targets","Cumulative DQN improvements","Discrete atom limitation"],
            "Target":["mean Q","mean Q","mean Q","mean Q","distribution Z","distribution Z","quantiles of Z"],
            "Paper year":["2015","2016","2016","2016","2017","2018","2018"],
            "Relative compute":["1×","1×","1×","1.5×","3×","5×","3×"],
        })
        st.dataframe(df_all, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 8 — STUDY PLAN
    # ══════════════════════════════════════════════════════════════════════
    with tab_plan:
        _sec("📚","Value-Based Deep RL — Study Plan",
             "A structured 4-week curriculum from DQN to Rainbow with resources and milestones","#1565c0")

        weeks = [
            ("Week 1","Foundations","#1565c0",[
                ("📄","Read Mnih et al. (2015) — DQN paper","Sections 1, 4, 5. Focus on experience replay motivation and target network argument."),
                ("💻","Implement DQN from scratch in PyTorch","CartPole first, then Atari (with gym[atari])."),
                ("🎮","Run provided CartPole simulator","Observe learning curve shape; watch loss and TD error over time."),
                ("🧮","Derive the Bellman optimality equation",r"Show that $Q^*(s,a)=r+\gamma\max_{a'} Q^*(s',a')$ arises from the value function definition."),
            ]),
            ("Week 2","Stabilisation & Efficiency","#00897b",[
                ("📄","Read van Hasselt et al. (2016) — Double DQN","Focus on the overestimation proof in Section 2."),
                ("📄","Read Schaul et al. (2016) — PER","Sections 1–3. Implement SumTree data structure."),
                ("💻","Add Double DQN to your implementation","One-line change: decouple argmax and evaluation."),
                ("💻","Add PER to your implementation","With IS weights; log the sampling distribution."),
                ("📊","Compare DQN vs DDQN vs PER","Track Q-value estimates over training — verify overestimation is reduced."),
            ]),
            ("Week 3","Architecture & Distribution","#7c4dff",[
                ("📄","Read Wang et al. (2016) — Dueling DQN","Focus on the identifiability argument in Section 4."),
                ("📄","Read Bellemare et al. (2017) — C51","Sections 1–4. Understand the projection step."),
                ("💻","Implement Dueling architecture","Add two separate MLP heads after the shared encoder."),
                ("💻","Implement C51","51-atom categorical distribution; cross-entropy loss."),
                ("📊","Visualise V and A streams","Confirm V is state-dependent and A sums near-zero."),
            ]),
            ("Week 4","Rainbow & Beyond","#f57f17",[
                ("📄","Read Hessel et al. (2018) — Rainbow","Study the ablation results in Figure 2."),
                ("📄","Read Dabney et al. (2018) — IQN","Understand the quantile regression loss."),
                ("💻","Combine all 6 Rainbow components","Integrate Double, Dueling, PER, n-step, C51 into one agent."),
                ("🎮","Benchmark on 3 Atari games","Use stable-baselines3 Rainbow baseline for reference."),
                ("📊","Ablation study","Remove one component at a time; confirm each contributes."),
            ]),
        ]

        for week_title, week_sub, color, tasks in weeks:
            st.markdown(f"""<div style="background:{color}18;border-left:4px solid {color};
                border-radius:0 10px 10px 0;padding:.7rem 1.2rem;margin:.8rem 0 .4rem">
                <b style="color:{color};font-size:1.05rem">{week_title}: {week_sub}</b>
                </div>""", unsafe_allow_html=True)
            for icon, title, desc in tasks:
                st.markdown(f"""<div style="background:#12121f;border:1px solid #2a2a3e;
                    border-radius:8px;padding:.6rem 1rem;margin:.3rem 0;display:flex;gap:.8rem">
                    <span style="font-size:1.2rem">{icon}</span>
                    <div><b style="color:white">{title}</b>
                    <br><span style="color:#9e9ebb;font-size:.87rem">{desc}</span></div>
                    </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("📖 Primary Resources")
        papers = [
            ("🧠","Mnih et al. (2015) — Human-level control through deep RL (DQN)",
             "The original DQN paper. Must-read. Focus on experience replay and target network sections.",
             "https://www.nature.com/articles/nature14236"),
            ("🔄","van Hasselt, Guez & Silver (2016) — Double DQN",
             "Two pages of math showing overestimation bias and how decoupling fixes it.",
             "https://arxiv.org/abs/1509.06461"),
            ("🏗️","Wang et al. (2016) — Dueling Network Architectures",
             "Introduces V+A decomposition. Clear identifiability argument.",
             "https://arxiv.org/abs/1511.06581"),
            ("🎯","Schaul et al. (2016) — Prioritized Experience Replay",
             "Defines SumTree, IS correction, and α/β annealing schedule.",
             "https://arxiv.org/abs/1511.05952"),
            ("📊","Bellemare, Dabney & Munos (2017) — C51",
             "Distributional RL foundations. Introduces the categorical projection.",
             "https://arxiv.org/abs/1707.06887"),
            ("🌈","Hessel et al. (2018) — Rainbow",
             "Figure 2 ablation study is essential. Shows each component's marginal contribution.",
             "https://arxiv.org/abs/1710.02298"),
            ("📐","Dabney et al. (2018) — IQN",
             "Implicit quantile regression. More flexible than C51 with comparable compute.",
             "https://arxiv.org/abs/1806.06923"),
        ]
        for icon, title, desc, url in papers:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;padding:.7rem 1.1rem;margin:.3rem 0">'
                f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                f'<br><span style="color:#9e9ebb;font-size:.87rem">{desc}</span></div>',
                unsafe_allow_html=True)
