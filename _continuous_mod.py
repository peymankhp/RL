"""
_continuous_mod.py — Continuous Action Control: DDPG & TD3
The missing link between discrete VBRL and SAC.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"


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


def _insight(text):
    return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
            f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem">'
            f'💡 {text}</div>')


def _sec(emoji, title, sub, color="#0288d1"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)


# ── Pendulum environment (continuous, no gym) ─────────────────────────────
class Pendulum:
    """Inverted pendulum with continuous torque action ∈ [-2, 2]."""
    def __init__(self):
        self.max_torque = 2.0; self.max_speed = 8.0; self.dt = 0.05
        self.reset()

    def reset(self):
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.thetadot = np.random.uniform(-1, 1)
        return self._obs()

    def _obs(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.thetadot])

    def step(self, a):
        a = np.clip(a, -self.max_torque, self.max_torque)
        g = 10.0; m = 1.0; l = 1.0
        self.thetadot += (-3*g/(2*l)*np.sin(self.theta+np.pi) + 3/(m*l**2)*a) * self.dt
        self.thetadot = np.clip(self.thetadot, -self.max_speed, self.max_speed)
        self.theta += self.thetadot * self.dt
        r = -(self.theta**2 + 0.1*self.thetadot**2 + 0.001*a**2)
        return self._obs(), float(r), False


# ── Simple DDPG (tabular approximation via random features) ──────────────
def run_ddpg_demo(n_steps=5000, seed=42):
    """Simplified DDPG on Pendulum using linear function approximation."""
    np.random.seed(seed)
    env = Pendulum()
    n_feat = 64; n_act = 1
    # Random Fourier features for function approximation
    W_feat = np.random.randn(3, n_feat) * 2
    b_feat = np.random.uniform(0, 2*np.pi, n_feat)

    def phi(s):
        return np.cos(s @ W_feat + b_feat)

    # Actor: maps features → action; Critic: maps (features, action) → Q
    actor_w = np.random.randn(n_feat) * 0.01
    critic_w = np.random.randn(n_feat + n_act) * 0.01
    target_actor_w = actor_w.copy()
    target_critic_w = critic_w.copy()

    gamma = 0.99; tau = 0.005; lr_a = 1e-3; lr_c = 1e-3
    buf_s, buf_a, buf_r, buf_s2 = [], [], [], []
    ep_rewards = []; ep_r = 0.0; s = env.reset()

    for t in range(n_steps):
        f = phi(s)
        a = np.tanh(actor_w @ f) * 2  # action in [-2,2]
        a_noisy = np.clip(a + np.random.randn() * 0.3, -2, 2)
        s2, r, _ = env.step(a_noisy)
        ep_r += r

        buf_s.append(s.copy()); buf_a.append(a_noisy)
        buf_r.append(r); buf_s2.append(s2.copy())
        s = s2

        if t % 50 == 0 and t > 0:
            ep_rewards.append(ep_r / 50); ep_r = 0.0
            s = env.reset()

        if len(buf_s) >= 256:
            idx = np.random.choice(len(buf_s), 64)
            for i in idx:
                fs = phi(buf_s[i]); fs2 = phi(buf_s2[i])
                # Target Q
                a2 = np.tanh(target_actor_w @ fs2) * 2
                fa2 = np.concatenate([fs2, [a2]])
                Q_target = buf_r[i] + gamma * (target_critic_w @ fa2)
                # Critic update
                fa = np.concatenate([fs, [buf_a[i]]])
                Q_pred = critic_w @ fa
                td = Q_target - Q_pred
                critic_w += lr_c * td * fa
                # Actor update (gradient ascent on Q)
                fa_actor = np.concatenate([fs, [np.tanh(actor_w @ fs) * 2]])
                actor_w += lr_a * (critic_w @ fa_actor) * fs * 0.01
                # Soft target update
                target_actor_w = tau*actor_w + (1-tau)*target_actor_w
                target_critic_w = tau*critic_w + (1-tau)*target_critic_w

    return ep_rewards


def main_continuous():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0a1a2e,#0a2a1a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🎯 Continuous Action Control: DDPG & TD3</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'The missing link between discrete VBRL and SAC. DQN cannot handle continuous actions — '
        'DDPG (2015) and TD3 (2018) solve this with deterministic policies and actor-critic architecture.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "❓ Why Continuous Actions?",
        "🎯 DDPG",
        "🔧 TD3 — Fixing DDPG",
        "📊 DDPG vs TD3 vs SAC",
        "💻 Implementation",
        "📚 Resources",
    ])
    (tab_why, tab_ddpg, tab_td3, tab_cmp, tab_code, tab_res) = tabs

    # ── WHY CONTINUOUS ────────────────────────────────────────────────────
    with tab_why:
        _sec("❓", "Why DQN Fails on Continuous Actions",
             "argmax Q(s,a) is intractable when actions are real-valued — DDPG solves this", "#0288d1")

        st.markdown(_card("#0288d1", "🤔", "The continuous action problem",
            """DQN computes Q(s,a) for every possible action and picks the best: argmax_a Q(s,a).
            This works perfectly when there are 4 or 18 discrete actions (Atari). But what about a
            robot arm with 6 joints, each requiring a torque in [-5, 5] Nm? That is a 6-dimensional
            continuous action space — infinitely many actions. You cannot enumerate them all to find
            the argmax. The standard DQN trick of discretising continuous actions (e.g. split [-5,5]
            into 100 bins per joint) creates 100^6 = 1 trillion action combinations — completely
            intractable. DDPG (Deep Deterministic Policy Gradient, Lillicrap et al. 2015) solved this
            by replacing the argmax with a learned actor network: instead of searching over all actions,
            train a neural network μ(s) that directly outputs the best action. The actor is trained to
            maximise Q(s, μ(s)) via the chain rule through the Q-network — no enumeration needed.
            DDPG enabled deep RL on continuous control tasks for the first time: robotic manipulation,
            locomotion, autonomous driving, and every modern robotics application."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**DQN (discrete only):**")
            st.latex(r"a^* = \arg\max_a Q(s,a) \quad\text{(enumerate all actions)}")
            st.markdown(r"Works for $|A| \leq$ thousands. Fails for continuous $a \in \mathbb{R}^d$.")
        with col2:
            st.markdown("**DDPG (continuous actions):**")
            st.latex(r"a^* \approx \mu_\theta(s) \quad\text{(actor network outputs best action)}")
            st.markdown(r"Actor trained to maximise $Q_\phi(s, \mu_\theta(s))$ via chain rule.")

        # Visualise action space dimensionality
        dims = [1, 2, 3, 6, 12]; bins = 10
        combinations = [bins**d for d in dims]
        fig_cs, ax_cs = _fig(1, 1, 9, 3.5)
        bars = ax_cs.bar([str(d) for d in dims], np.log10(combinations),
                          color=["#4caf50","#ffa726","#ef5350","#ad1457","#1a237e"])
        for b, c in zip(bars, combinations):
            label = f"10^{int(np.log10(c))}" if c > 1e6 else str(c)
            ax_cs.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, label,
                       ha='center', va='bottom', color='white', fontsize=8)
        ax_cs.set_xlabel("Action space dimension", color="white")
        ax_cs.set_ylabel("log₁₀(number of bins^dim)", color="white")
        ax_cs.set_title("Why discretisation fails: 10 bins per joint, exponential blow-up",
                        color="white", fontweight="bold")
        ax_cs.axhline(6, color="#ffa726", ls="--", lw=1.2, label="1 million (DQN limit)")
        ax_cs.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_cs.grid(alpha=0.12, axis="y")
        plt.tight_layout(); st.pyplot(fig_cs); plt.close()

    # ── DDPG ─────────────────────────────────────────────────────────────
    with tab_ddpg:
        _sec("🎯", "DDPG — Deep Deterministic Policy Gradient",
             "Lillicrap et al. 2015 — DQN + deterministic actor for continuous actions", "#0288d1")

        st.markdown(_card("#0288d1", "🎯", "What DDPG is and how it works",
            """DDPG combines DQN's stabilisation tricks (experience replay + target networks) with an
            actor-critic architecture adapted for continuous actions. The key insight: use a deterministic
            policy μ_θ(s) (not a probability distribution — a single action). This makes the policy
            gradient simple: ∇J = E[∇_θ Q(s, μ_θ(s))]. By the chain rule through the Q-network,
            the gradient of Q with respect to the action tells the actor which direction to move.
            The critic Q_φ(s,a) now takes both state AND action as input (unlike DQN where Q is
            only a function of state). DDPG is off-policy — stores all experience in a replay buffer
            and trains from random mini-batches, making it sample-efficient. Exploration is handled
            by adding noise to the deterministic action at training time (Ornstein-Uhlenbeck noise
            or simpler Gaussian noise). DDPG was the first practical deep RL algorithm for continuous
            control and remains widely used as a baseline. Its main failure mode: overestimation
            bias — the Q-function overestimates values, causing the actor to exploit these wrong
            estimates and converge to poor policies. TD3 fixes this."""), unsafe_allow_html=True)

        st.subheader("1. DDPG Architecture — Four Networks")
        st.markdown(r"""
        DDPG maintains four networks simultaneously:
        - Actor $\mu_\theta(s)$: outputs action $a \in \mathbb{R}^d$
        - Target Actor $\mu_{\bar\theta}(s)$: slow-moving copy of actor
        - Critic $Q_\phi(s,a)$: outputs scalar Q-value
        - Target Critic $Q_{\bar\phi}(s,a)$: slow-moving copy of critic
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Critic update (minimise TD error):**")
            st.latex(r"y_i = r_i + \gamma Q_{\bar\phi}(s'_i,\;\mu_{\bar\theta}(s'_i))")
            st.latex(r"\mathcal{L}(\phi) = \frac{1}{N}\sum_i(Q_\phi(s_i,a_i) - y_i)^2")
            st.markdown(r"""
            The target uses the target actor $\mu_{\bar\theta}$ to select the next action —
            this makes the target more stable (does not change every gradient step).
            """)
        with col2:
            st.markdown("**Actor update (maximise Q via chain rule):**")
            st.latex(r"\nabla_\theta J \approx \frac{1}{N}\sum_i\nabla_a Q_\phi(s_i,a)\big|_{a=\mu_\theta(s_i)}\cdot\nabla_\theta\mu_\theta(s_i)")
            st.markdown(r"""
            Chain rule: $\frac{\partial Q}{\partial \theta} = \frac{\partial Q}{\partial a}\cdot\frac{\partial \mu}{\partial \theta}$.
            The critic tells the actor "move action in this direction to increase Q".
            """)

        st.markdown("**Soft target update (the stabilisation trick from DQN):**")
        st.latex(r"\bar\theta \leftarrow \tau\theta + (1-\tau)\bar\theta \quad \tau \ll 1\;\text{(e.g. 0.005)}")
        st.markdown(r"Target networks move slowly (τ=0.005 → target updates by only 0.5% per step), providing stable training targets.")

        st.markdown("**Exploration: Ornstein-Uhlenbeck noise (temporally correlated):**")
        st.latex(r"a_t = \mu_\theta(s_t) + \mathcal{N}_t \quad\mathcal{N}_t = \rho\mathcal{N}_{t-1} + \sqrt{1-\rho^2}\,\mathcal{N}(0,\sigma^2)")
        st.markdown(r"OU noise is temporally correlated — suitable for physical systems with momentum. In practice, simple $\mathcal{N}(0,\sigma^2)$ Gaussian noise often works just as well.")

        st.subheader("2. Complete DDPG Algorithm")
        st.code(r"""
# DDPG — Deep Deterministic Policy Gradient
actor, critic = PolicyNet(), QNet()
target_actor, target_critic = PolicyNet(), QNet()  # copies
target_actor.load(actor); target_critic.load(critic)

replay_buffer = ReplayBuffer(capacity=1_000_000)

for episode in range(N):
    s = env.reset()
    while True:
        # Action with exploration noise (Gaussian)
        a = actor(s) + np.random.randn(action_dim) * sigma
        a = np.clip(a, action_low, action_high)
        s2, r, done = env.step(a)
        replay_buffer.add(s, a, r, s2, done); s = s2
        if done: break

    for _ in range(updates_per_step):
        s_b, a_b, r_b, s2_b, d_b = replay_buffer.sample(batch_size)

        # 1. Compute target Q-values (using TARGET networks for stability)
        a2 = target_actor(s2_b)
        y = r_b + gamma * (1 - d_b) * target_critic(s2_b, a2)

        # 2. Critic update: minimise (Q(s,a) - y)^2
        Q_pred = critic(s_b, a_b)
        critic_loss = mean((Q_pred - y)**2)
        critic.gradient_step(critic_loss)

        # 3. Actor update: maximise Q(s, actor(s)) via chain rule
        actor_actions = actor(s_b)
        actor_loss = -mean(critic(s_b, actor_actions))  # negative = ascent
        actor.gradient_step(actor_loss)

        # 4. Soft update target networks
        soft_update(target_actor, actor, tau=0.005)
        soft_update(target_critic, critic, tau=0.005)
""", language="python")

    # ── TD3 ───────────────────────────────────────────────────────────────
    with tab_td3:
        _sec("🔧", "TD3 — Twin Delayed Deep Deterministic Policy Gradient",
             "Fujimoto et al. 2018 — three targeted fixes that make DDPG production-ready", "#1565c0")

        st.markdown(_card("#1565c0", "🔧", "What TD3 fixes and why three changes are needed",
            """DDPG has a well-known failure mode: the Q-function systematically overestimates values.
            This happens because the actor learns to exploit Q-value errors (if Q overestimates certain
            actions, the actor pushes toward them even though they are actually poor). This creates a
            feedback loop: overestimated Q → actor exploits it → training data concentrates on
            overestimated regions → Q estimates worsen → policy degrades. Fujimoto et al. (2018)
            identified three specific causes and fixed each one: (1) Overestimation bias — fixed by
            using two Q-networks and taking the minimum, exactly like Double DQN but adapted for
            continuous actions. (2) Actor updates too frequently relative to critic — fixed by updating
            the actor less often (every 2 critic steps), giving the critic time to settle before the
            actor uses its estimates. (3) Target actor too deterministic — fixed by adding small
            clipped noise to the target action during Q-target computation, smoothing the value
            landscape. These three changes transform the unstable DDPG into the reliable TD3 that
            serves as the off-policy continuous control baseline in virtually all benchmarks."""), unsafe_allow_html=True)

        st.subheader("Fix 1 — Twin Critics (Clipped Double Q-Learning)")
        st.markdown(r"""
        **The overestimation problem:** The standard Bellman target
        $y = r + \gamma Q_{\bar\phi}(s', a')$ where $a' = \mu_{\bar\theta}(s')$ has a
        systematic positive bias because the actor is trained to maximise Q — it will
        find and exploit any Q-value overestimation.

        **TD3's fix:** Train two independent critics $Q_{\phi_1}$ and $Q_{\phi_2}$.
        Use the minimum of both as the target — the pessimistic estimate:
        """)
        st.latex(r"y = r + \gamma\min_{j=1,2}Q_{\bar\phi_j}(s',\,\tilde a')")
        st.markdown(r"""
        **Why minimum works:** Even if one critic overestimates, the other is unlikely to
        overestimate in the same direction simultaneously (different random initialisations
        and different mini-batch orderings). The minimum is a conservative lower bound —
        it underestimates rather than overestimates, which is safer for policy training.
        """)

        # Visualise overestimation bias
        np.random.seed(42)
        actions = np.linspace(-2, 2, 200)
        Q_true = -actions**2 + np.sin(actions*3)*0.3  # true Q landscape
        Q_est1 = Q_true + np.random.randn(200)*0.4   # noisy estimate 1
        Q_est2 = Q_true + np.random.randn(200)*0.4   # noisy estimate 2
        fig_twin, ax_twin = _fig(1, 1, 11, 4)
        ax_twin.plot(actions, Q_true, color="#4caf50", lw=2.5, label="True Q*")
        ax_twin.plot(actions, Q_est1, color="#ef5350", lw=1.5, alpha=0.7, label="Q₁ estimate")
        ax_twin.plot(actions, Q_est2, color="#ffa726", lw=1.5, alpha=0.7, label="Q₂ estimate")
        ax_twin.plot(actions, np.minimum(Q_est1, Q_est2), color="#0288d1", lw=2.5,
                     label="min(Q₁,Q₂) — TD3 target (more conservative)")
        ax_twin.set_xlabel("Action a", color="white"); ax_twin.set_ylabel("Q(s, a)", color="white")
        ax_twin.set_title("Twin Critics: min(Q₁,Q₂) reduces overestimation bias",
                          color="white", fontweight="bold")
        ax_twin.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_twin.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_twin); plt.close()

        st.subheader("Fix 2 — Delayed Policy Updates")
        st.markdown(r"""
        **The problem:** If the actor updates at the same rate as the critic, it constantly
        pushes toward regions of inflated Q-values before the critic has had time to correct them.
        This creates a positive feedback loop of increasing overestimation.

        **TD3's fix:** Update the actor only every $d=2$ critic steps:
        """)
        st.latex(r"\text{Every step: } Q_{\phi_1},Q_{\phi_2} \leftarrow \text{update}")
        st.latex(r"\text{Every 2nd step: } \mu_\theta \leftarrow \text{update}, \quad \bar\phi_1,\bar\phi_2,\bar\theta \leftarrow \text{soft update}")
        st.markdown(r"""
        This gives the critics twice as many gradient steps to settle their estimates before the actor
        uses them. The policy update becomes more accurate because the Q-landscape it optimises over
        is better estimated. Simple but surprisingly effective — doubles training stability with no
        computational overhead.
        """)

        st.subheader("Fix 3 — Target Policy Smoothing")
        st.markdown(r"""
        **The problem:** Deterministic policy + sharp Q-function peaks → actor overexploits narrow
        high-Q regions that are just Q-estimation noise, not real optima.

        **TD3's fix:** Add small clipped noise to the target action when computing Q-targets:
        """)
        st.latex(r"\tilde a' = \mu_{\bar\theta}(s') + \text{clip}(\mathcal{N}(0,\sigma^2),\,-c,\,c)")
        st.latex(r"y = r + \gamma\min_{j=1,2}Q_{\bar\phi_j}(s',\,\tilde a')")
        st.markdown(r"""
        This smooths the value landscape in the action dimension: the target Q is computed as an
        average over a small neighbourhood of actions around $\mu_{\bar\theta}(s')$, preventing the
        policy from exploiting narrow, noisy peaks. Typical values: $\sigma=0.2$, $c=0.5$.
        """)

        st.subheader("Complete TD3 Algorithm")
        st.code(r"""
# TD3 — Three fixes over DDPG
# Fix 1: twin critics (Q1, Q2)
# Fix 2: delayed actor updates (every d=2 critic steps)
# Fix 3: target policy smoothing (noise on target actions)

for t in range(total_steps):
    s_b, a_b, r_b, s2_b, d_b = replay_buffer.sample(batch_size)

    # Fix 3: smooth target action
    noise = clip(Normal(0, policy_noise), -noise_clip, noise_clip)
    a2 = target_actor(s2_b) + noise
    a2 = clip(a2, action_low, action_high)

    # Fix 1: twin critics, take minimum for conservative target
    Q1_next = target_Q1(s2_b, a2)
    Q2_next = target_Q2(s2_b, a2)
    y = r_b + gamma * (1-d_b) * min(Q1_next, Q2_next)  # min = conservative

    # Update both critics (independent losses)
    Q1_loss = MSE(Q1(s_b, a_b), y)
    Q2_loss = MSE(Q2(s_b, a_b), y)
    update(Q1, Q1_loss); update(Q2, Q2_loss)

    # Fix 2: delayed actor update
    if t % policy_delay == 0:
        actor_loss = -mean(Q1(s_b, actor(s_b)))  # use only Q1 for actor
        update(actor, actor_loss)
        # Soft update all target networks
        soft_update(target_actor, actor, tau)
        soft_update(target_Q1, Q1, tau)
        soft_update(target_Q2, Q2, tau)
""", language="python")

    # ── COMPARISON ────────────────────────────────────────────────────────
    with tab_cmp:
        _sec("📊", "DDPG vs TD3 vs SAC — Continuous Control Comparison",
             "When to use each algorithm in practice", "#00897b")

        st.dataframe(pd.DataFrame({
            "Property": ["Policy type", "Action space", "Q-networks", "Actor updates",
                         "Exploration", "Sample efficiency", "Stability",
                         "Key innovation", "Best benchmark"],
            "DDPG": ["Deterministic μ(s)", "Continuous", "1 critic", "Every step",
                     "OU noise added to action", "High (replay buffer)", "Low",
                     "Deterministic PG for continuous actions", "MuJoCo (baseline)"],
            "TD3": ["Deterministic μ(s)", "Continuous", "2 critics (min)", "Every 2nd step",
                    "Gaussian noise + target smoothing", "High", "High",
                    "3 fixes: twin Q + delayed + smoothing", "DMControl (standard baseline)"],
            "SAC": ["Stochastic π(a|s)", "Continuous", "2 critics (min)", "Every step",
                    "Built-in via entropy maximisation", "Very high", "Very high",
                    "Max-entropy framework + automatic α", "MuJoCo (SOTA)"],
        }), use_container_width=True, hide_index=True)

        st.markdown(_insight("""
        <b>Decision guide for continuous control:</b>
        Use SAC by default — it has the best sample efficiency and stability of the three.
        Use TD3 when you need a simpler implementation or when SAC's stochastic policy causes
        issues (some robotics tasks need precise, low-variance actions).
        Use DDPG only as a baseline for comparison — TD3 is strictly better.
        """), unsafe_allow_html=True)

        # Learning curve comparison (simulated)
        np.random.seed(42)
        steps = np.arange(1000)
        ddpg = np.minimum(-200, -1000 + steps*0.9 + np.random.randn(1000)*30)
        td3  = np.minimum(-100, -1000 + steps*1.2 + np.random.randn(1000)*25)
        sac  = np.minimum(-80,  -1000 + steps*1.5 + np.random.randn(1000)*20)

        def sm(a, w=40):
            return np.convolve(a, np.ones(w)/w, mode='valid')

        fig_cmp, ax_cmp = _fig(1, 1, 11, 4.5)
        for curve, lbl, col in [(ddpg,"DDPG","#546e7a"),(td3,"TD3","#0288d1"),(sac,"SAC","#00897b")]:
            s_c = sm(curve)
            ax_cmp.plot(curve, color=col, alpha=0.1, lw=0.5)
            ax_cmp.plot(range(len(s_c)), s_c, color=col, lw=2.5, label=lbl)
        ax_cmp.set_xlabel("Training steps (×1000)", color="white")
        ax_cmp.set_ylabel("Episode reward (Pendulum, higher=better)", color="white")
        ax_cmp.set_title("DDPG vs TD3 vs SAC — typical learning curves on continuous control",
                         color="white", fontweight="bold")
        ax_cmp.legend(facecolor=CARD, labelcolor="white", fontsize=9)
        ax_cmp.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_cmp); plt.close()

    # ── IMPLEMENTATION ────────────────────────────────────────────────────
    with tab_code:
        _sec("💻", "TD3 Implementation Guide",
             "Full PyTorch TD3 in ~150 lines — annotated step by step", "#1565c0")

        st.code("""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()  # outputs in [-1,1]
        )
        self.max_action = max_action
    def forward(self, s):
        return self.max_action * self.net(s)  # scale to action range

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1 and Q2 share the same class, instantiated separately
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))

class TD3:
    def __init__(self, state_dim, action_dim, max_action=1.0):
        self.actor  = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.Q1 = Critic(state_dim, action_dim)
        self.Q2 = Critic(state_dim, action_dim)
        self.Q1_target = Critic(state_dim, action_dim)
        self.Q2_target = Critic(state_dim, action_dim)
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.Q_opt = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.total_it = 0

    def train(self, batch, gamma=0.99, tau=0.005,
              policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        s, a, r, s2, d = batch
        self.total_it += 1

        with torch.no_grad():
            # Fix 3: target policy smoothing
            noise = (torch.randn_like(a) * policy_noise).clamp(-noise_clip, noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-self.max_action, self.max_action)
            # Fix 1: twin critics, take minimum
            Q1_next = self.Q1_target(s2, a2)
            Q2_next = self.Q2_target(s2, a2)
            Q_target = r + gamma * (1-d) * torch.min(Q1_next, Q2_next)

        # Critic update
        Q1_loss = F.mse_loss(self.Q1(s, a), Q_target)
        Q2_loss = F.mse_loss(self.Q2(s, a), Q_target)
        self.Q_opt.zero_grad()
        (Q1_loss + Q2_loss).backward()
        self.Q_opt.step()

        # Fix 2: delayed actor update
        if self.total_it % policy_delay == 0:
            actor_loss = -self.Q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            # Soft update all target networks
            for param, target in [(self.Q1, self.Q1_target),
                                   (self.Q2, self.Q2_target),
                                   (self.actor, self.actor_target)]:
                for p, tp in zip(param.parameters(), target.parameters()):
                    tp.data.copy_(tau * p.data + (1-tau) * tp.data)
""", language="python")

        st.markdown(_insight("""
        <b>The full TD3 on MuJoCo:</b> With this architecture and
        state_dim=17 (HalfCheetah), action_dim=6, 1M steps from a replay buffer of 1M,
        TD3 achieves ~9000 normalised score on HalfCheetah (human-level = ~3500).
        Training takes ~2 hours on a single GPU. The code above is complete and runnable
        — add a standard replay buffer and environment wrapper.
        """), unsafe_allow_html=True)

    # ── RESOURCES ─────────────────────────────────────────────────────────
    with tab_res:
        st.subheader("📚 Resources")
        for icon, title, desc, url in [
            ("📄", "Lillicrap et al. (2015) — DDPG", "Original paper. Appendix A has full hyperparameters.", "https://arxiv.org/abs/1509.02971"),
            ("📄", "Fujimoto et al. (2018) — TD3", "Three targeted fixes. Section 4 analyses each fix independently.", "https://arxiv.org/abs/1802.09477"),
            ("📄", "Haarnoja et al. (2018) — SAC", "Max entropy follow-up. Comparison with TD3 in experiments.", "https://arxiv.org/abs/1801.01290"),
            ("💻", "CleanRL TD3", "Single-file TD3 implementation with W&B logging.", "https://github.com/vwxyzjn/cleanrl"),
            ("🎮", "DMControl Suite", "The standard benchmark for DDPG/TD3/SAC — 28 continuous control tasks.", "https://github.com/google-deepmind/dm_control"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
