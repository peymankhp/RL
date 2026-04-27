"""
_ac_mod.py  —  Policy Gradient & Actor-Critic Methods
Covers: REINFORCE · Vanilla AC · A2C · A3C · PPO (with GAE) · TRPO overview · SAC overview
Based on Sutton & Barto Ch.13, Schulman et al. 2015/2017, Haarnoja et al. 2018
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"
ALG_COL = {
    "REINFORCE": "#7c4dff", "AC": "#0288d1", "A2C": "#00897b",
    "A3C": "#f57f17", "PPO": "#e65100", "TRPO": "#ad1457", "SAC": "#558b2f",
}

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

def _sec(emoji, title, sub, color="#7c4dff"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def smooth(a, w=10):
    return np.convolve(a, np.ones(w)/w, mode="valid") if len(a) > w else np.array(a, float)

# ── CartPole (no gym dependency) ──────────────────────────────────────────
class CartPole:
    g=9.8; mc=1.0; mp=0.1; l=0.5; dt=0.02
    def reset(self):
        self.s = np.random.uniform(-0.05, 0.05, 4)
        self.steps = 0
        return self.s.copy()
    def step(self, a):
        x,xd,th,thd = self.s
        f = 10. if a == 1 else -10.
        costh = np.cos(th); sinth = np.sin(th)
        tmp = (f + self.mp*self.l*thd**2*sinth) / (self.mc+self.mp)
        thdd = (self.g*sinth - costh*tmp) / (self.l*(4/3 - self.mp*costh**2/(self.mc+self.mp)))
        xdd  = tmp - self.mp*self.l*thdd*costh/(self.mc+self.mp)
        self.s = np.array([x+self.dt*xd, xd+self.dt*xdd,
                           th+self.dt*thd, thd+self.dt*thdd])
        self.steps += 1
        done = (abs(self.s[0]) > 2.4 or abs(self.s[2]) > 0.2095 or self.steps >= 200)
        return self.s.copy(), 1.0 if not done else 0.0, done

# ── Neural networks (numpy) ───────────────────────────────────────────────
class PolicyNet:
    def __init__(self, in_dim=4, hid=32, out_dim=2, seed=0):
        np.random.seed(seed)
        k = np.sqrt(2/in_dim)
        self.W1 = np.random.randn(in_dim, hid)*k;  self.b1 = np.zeros(hid)
        self.W2 = np.random.randn(hid, out_dim)*np.sqrt(2/hid); self.b2 = np.zeros(out_dim)

    def forward(self, x):
        x = np.array(x).flatten()
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def log_prob(self, x, a):
        return np.log(self.forward(x)[a] + 1e-8)

    def grad_log_prob(self, x, a):
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        e = np.exp(logits - logits.max()); probs = e / e.sum()
        d_logits = probs.copy(); d_logits[a] -= 1.0
        gW2 = np.outer(h, d_logits); gb2 = d_logits
        dh = (d_logits @ self.W2.T) * (h > 0)
        gW1 = np.outer(x, dh); gb1 = dh
        return gW1, gb1, gW2, gb2

    def update(self, gW1, gb1, gW2, gb2, lr):
        self.W1 += lr*gW1; self.b1 += lr*gb1
        self.W2 += lr*gW2; self.b2 += lr*gb2


class ValueNet:
    def __init__(self, in_dim=4, hid=32, seed=0):
        np.random.seed(seed+100)
        k = np.sqrt(2/in_dim)
        self.W1 = np.random.randn(in_dim, hid)*k;  self.b1 = np.zeros(hid)
        self.W2 = np.random.randn(hid, 1)*np.sqrt(2/hid); self.b2 = np.array([0.0])

    def forward(self, x):
        x = np.array(x).flatten()
        h = np.maximum(0, x @ self.W1 + self.b1)
        return float((h @ self.W2 + self.b2).item())

    def update(self, x, target, lr=0.005):
        x = np.array(x).flatten()
        h = np.maximum(0, x @ self.W1 + self.b1)
        v = float((h @ self.W2 + self.b2).item())
        err = float(v - target)
        dW2 = h[:, None]*err; db2 = np.array([err])
        dh = (err*(self.W2.flatten())) * (h > 0)
        dW1 = np.outer(x, dh); db1 = dh
        self.W1 -= lr*dW1; self.b1 -= lr*db1
        self.W2 -= lr*dW2; self.b2 -= lr*db2
        return float(err**2)


# ── REINFORCE ─────────────────────────────────────────────────────────────
def train_reinforce(n_episodes=200, lr=0.01, gamma=0.99, baseline=False, seed=42):
    np.random.seed(seed)
    env = CartPole(); pi = PolicyNet(seed=seed)
    vn = ValueNet(seed=seed) if baseline else None
    rewards = []; pg_norms = []

    for ep in range(n_episodes):
        traj = []; s = env.reset()
        while True:
            probs = pi.forward(s)
            a = np.random.choice(2, p=probs)
            ns, r, done = env.step(a)
            traj.append((s, a, r)); s = ns
            if done: break

        G = 0.0; returns = []
        for _, _, r in reversed(traj):
            G = r + gamma*G; returns.insert(0, G)

        gW1t = np.zeros_like(pi.W1); gb1t = np.zeros_like(pi.b1)
        gW2t = np.zeros_like(pi.W2); gb2t = np.zeros_like(pi.b2)
        for (s, a, _), Gt in zip(traj, returns):
            b = vn.forward(s) if baseline else 0.0
            adv = Gt - b
            if baseline: vn.update(s, Gt)
            gW1, gb1, gW2, gb2 = pi.grad_log_prob(s, a)
            gW1t += adv*gW1; gb1t += adv*gb1
            gW2t += adv*gW2; gb2t += adv*gb2
        n = len(traj)
        pi.update(gW1t/n, gb1t/n, gW2t/n, gb2t/n, lr)
        pg_norms.append(float(np.sqrt(np.sum(gW2t**2) + np.sum(gW1t**2))/n))
        rewards.append(sum(r for _, _, r in traj))

    return rewards, pg_norms


# ── Actor-Critic (1-step) ─────────────────────────────────────────────────
def train_ac(n_episodes=200, lr_pi=0.005, lr_v=0.01, gamma=0.99, seed=42):
    np.random.seed(seed)
    env = CartPole(); pi = PolicyNet(seed=seed); vn = ValueNet(seed=seed)
    rewards = []; td_errs = []

    for ep in range(n_episodes):
        s = env.reset(); ep_r = 0.0
        while True:
            probs = pi.forward(s); a = np.random.choice(2, p=probs)
            ns, r, done = env.step(a); ep_r += r
            v_s = vn.forward(s); v_ns = 0.0 if done else vn.forward(ns)
            delta = r + gamma*v_ns - v_s
            vn.update(s, r + gamma*v_ns)
            gW1, gb1, gW2, gb2 = pi.grad_log_prob(s, a)
            pi.update(delta*gW1, delta*gb1, delta*gW2, delta*gb2, lr_pi)
            td_errs.append(delta); s = ns
            if done: break
        rewards.append(ep_r)

    return rewards, td_errs


# ── A2C ───────────────────────────────────────────────────────────────────
def train_a2c(n_episodes=200, lr_pi=0.003, lr_v=0.01,
              gamma=0.99, n_steps=5, entropy_coef=0.01, seed=42):
    np.random.seed(seed)
    env = CartPole(); pi = PolicyNet(seed=seed); vn = ValueNet(seed=seed)
    ep_rewards = []; ep_r = 0.0; s = env.reset(); done = False; step_buf = []
    ep_count = 0

    for _ in range(n_episodes * 300):
        if done:
            ep_rewards.append(ep_r); ep_r = 0.0; s = env.reset()
            done = False; ep_count += 1
        if ep_count >= n_episodes:
            break

        probs = pi.forward(s); a = np.random.choice(2, p=probs)
        ns, r, done = env.step(a); ep_r += r
        step_buf.append((s, a, r, ns, done)); s = ns

        if len(step_buf) >= n_steps or done:
            R = 0.0 if done else vn.forward(step_buf[-1][3])
            gW1t = np.zeros_like(pi.W1); gb1t = np.zeros_like(pi.b1)
            gW2t = np.zeros_like(pi.W2); gb2t = np.zeros_like(pi.b2)
            for (st, at, rt, nst, dt) in reversed(step_buf):
                R = rt + gamma*R
                adv = R - vn.forward(st)
                vn.update(st, R, lr_v)
                gW1, gb1, gW2, gb2 = pi.grad_log_prob(st, at)
                gW1t += adv*gW1; gb1t += adv*gb1
                gW2t += adv*gW2; gb2t += adv*gb2
            nb = max(1, len(step_buf))
            pi.update(gW1t/nb, gb1t/nb, gW2t/nb, gb2t/nb, lr_pi)
            step_buf.clear()

    return ep_rewards


# ── PPO ───────────────────────────────────────────────────────────────────
def train_ppo(n_iterations=50, horizon=512, n_epochs=4,
              lr_pi=3e-3, lr_v=5e-3, gamma=0.99,
              lam=0.95, clip_eps=0.2, ent_coef=0.01, seed=42):
    np.random.seed(seed)
    env = CartPole(); pi = PolicyNet(seed=seed); vn = ValueNet(seed=seed)
    all_ep_rewards = []; ep_r = 0.0; s = env.reset(); done = False
    ep_rewards_buf = []

    for iteration in range(n_iterations):
        states = []; actions = []; rewards = []
        dones = []; values = []; log_probs = []

        for _ in range(horizon):
            if done:
                ep_rewards_buf.append(ep_r); ep_r = 0.0
                s = env.reset(); done = False
            probs = pi.forward(s)
            a = np.random.choice(2, p=probs)
            ns, r, done = env.step(a)
            states.append(s.copy()); actions.append(a)
            rewards.append(r); dones.append(done)
            values.append(vn.forward(s))
            log_probs.append(np.log(probs[a] + 1e-8))
            ep_r += r; s = ns

        last_v = 0.0 if done else vn.forward(s)
        values_np = np.array(values)
        rewards_np = np.array(rewards)
        dones_np   = np.array(dones, float)

        # GAE
        advantages = np.zeros(horizon); gae = 0.0
        for t in reversed(range(horizon)):
            nv = last_v if t == horizon-1 else values_np[t+1]*(1-dones_np[t])
            delta = rewards_np[t] + gamma*nv - values_np[t]
            gae   = delta + gamma*lam*(1-dones_np[t])*gae
            advantages[t] = gae
        returns_np = advantages + values_np
        advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = np.array(log_probs)
        states_np     = np.array(states)

        for _ in range(n_epochs):
            idx = np.random.permutation(horizon)
            for start in range(0, horizon, 64):
                bi = idx[start:start+64]
                gW1t = np.zeros_like(pi.W1); gb1t = np.zeros_like(pi.b1)
                gW2t = np.zeros_like(pi.W2); gb2t = np.zeros_like(pi.b2)
                for i in bi:
                    st = states_np[i]; at = int(actions[i]); adv = advantages[i]
                    cur_lp = pi.log_prob(st, at)
                    ratio  = np.exp(cur_lp - old_log_probs[i])
                    clip_r = np.clip(ratio, 1-clip_eps, 1+clip_eps)
                    scale  = (ratio if ratio*adv <= clip_r*adv else clip_r) * np.sign(adv) if abs(ratio-1) < clip_eps else 0.0
                    gW1, gb1, gW2, gb2 = pi.grad_log_prob(st, at)
                    gW1t += scale*gW1; gb1t += scale*gb1
                    gW2t += scale*gW2; gb2t += scale*gb2
                    vn.update(st, returns_np[i], lr_v)
                n = max(1, len(bi))
                pi.update(gW1t/n, gb1t/n, gW2t/n, gb2t/n, lr_pi)

        if ep_rewards_buf:
            all_ep_rewards.extend(ep_rewards_buf); ep_rewards_buf.clear()

    return all_ep_rewards


def main_ac():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a0a2e,#0a1a0a,#2a0a0a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🎭 Actor-Critic &amp; Policy Gradient</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'From REINFORCE to PPO — every policy gradient method decoded, '
        'all derivations from first principles, runnable simulations.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        '🗺️ Overview', '🎲 REINFORCE', '🎭 Actor-Critic',
        '🤝 A2C & A3C', '🔐 PPO', '🏛️ TRPO & SAC',
        '📈 Dashboard', '📚 Study Plan',
    ])
    (tab_ov, tab_rf, tab_ac_t, tab_a2c,
     tab_ppo, tab_trpo, tab_dash, tab_plan) = tabs

    # ── OVERVIEW ──────────────────────────────────────────────────────────
    with tab_ov:
        st.markdown(r"""
        <div style="background:#12121f;border-radius:12px;padding:1.4rem 1.8rem;border:1px solid #2a2a3e">
        <h3 style="color:white;margin-top:0">🏛️ The Policy Gradient Theorem — Foundation of Everything</h3>
        All actor-critic methods derive from one result. The goal is to find θ that maximises:
        </div>""", unsafe_allow_html=True)
        st.latex(r"J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\!\left[\sum_t r(s_t,a_t)\right]")
        st.markdown(r"""
        Direct differentiation is hard — the trajectory distribution $p_\theta(\tau)$ involves
        unknown environment dynamics. The **log-derivative trick** eliminates them:
        """)
        st.latex(r"\nabla_\theta J = \mathbb{E}_{\tau \sim p_\theta}\!\left[\nabla_\theta \log p_\theta(\tau)\cdot r(\tau)\right]")
        st.markdown(r"""
        Expanding the trajectory log-probability — initial state and dynamics have no $\theta$:
        """)
        st.latex(r"\nabla_\theta \log p_\theta(\tau) = \sum_{t=1}^T \nabla_\theta \log\pi_\theta(a_t|s_t)")
        st.markdown(r"This is the key: we only need to differentiate the **policy**, not the environment. Full result:")
        st.latex(r"\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{\tau}\!\left[\sum_{t}\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\sum_{t}r(s_t,a_t)\right]}")

        st.divider()
        st.markdown(r"""
        **Three variance-reducing improvements applied in sequence:**

        **1. Causality** — policy at $t$ cannot affect rewards before $t$:
        """)
        st.latex(r"\nabla_\theta J \approx \frac{1}{N}\sum_i\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\underbrace{\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})}_{\hat Q_{i,t}\text{ (reward-to-go)}}")
        st.markdown("**2. Baseline** — subtract $b(s_t)$ without changing expectation (reduces variance):")
        st.latex(r"\nabla_\theta J \approx \frac{1}{N}\sum_i\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\underbrace{(\hat Q_{i,t} - b(s_{i,t}))}_{A_{i,t}\text{ (advantage)}}")
        st.markdown("**3. Actor-Critic** — replace Monte Carlo $\\hat Q$ with learned $V(s)$ as baseline:")
        st.latex(r"A_t \approx \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)")

        st.divider()
        st.subheader("📊 Algorithm Family Comparison")
        st.dataframe(pd.DataFrame({
            "Method":    ["REINFORCE","REINFORCE+b","AC (1-step)","A2C","A3C","PPO","TRPO","SAC"],
            "Advantage": ["MC return G_t","G_t − b(s)","1-step δ","n-step return","n-step return","GAE(λ)","GAE(λ)","Soft Q − V"],
            "On/Off":    ["On","On","On","On","On","On (clipped IS)","On","Off"],
            "Stability": ["Low","Low","Medium","High","High","Very high","Very high","Very high"],
            "Complexity":["Simple","Simple","Simple","Medium","Medium","Medium","High","High"],
            "Year":      ["1992","1992","1984","2016","2016","2017","2015","2018"],
        }), use_container_width=True, hide_index=True)

    # ── REINFORCE ─────────────────────────────────────────────────────────
    with tab_rf:
        st.subheader("🎲 REINFORCE — Monte Carlo Policy Gradient (Williams 1992)")
        st.markdown(r"""
        REINFORCE collects a complete episode, computes reward-to-go returns, then updates
        the policy to make high-return actions more likely. Simple, unbiased, but high variance.
        """)
        with st.expander("📐 Full REINFORCE Derivation", expanded=True):
            st.markdown("**Objective:**")
            st.latex(r"J(\theta) = \mathbb{E}_{\tau\sim p_\theta}\!\left[\sum_t r(s_t,a_t)\right] \approx \frac{1}{N}\sum_i\sum_t r(s_{i,t},a_{i,t})")
            st.markdown("**Log-derivative trick + causality + baseline:**")
            st.latex(r"\nabla_\theta J \approx \frac{1}{N}\sum_i\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\bigl(\hat Q_{i,t} - b(s_{i,t})\bigr)")
            st.markdown(r"Where $\hat Q_{i,t} = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$ (reward-to-go, computed backward).")
            st.markdown(r"**Why baseline does not bias the gradient:**")
            st.latex(r"\mathbb{E}_{a\sim\pi}\!\bigl[\nabla_\theta\log\pi_\theta(a|s)\cdot b(s)\bigr] = b(s)\nabla_\theta\underbrace{\sum_a\pi_\theta(a|s)}_{=1} = 0")
            st.markdown("**Update rule (gradient ASCENT):**")
            st.latex(r"\theta \leftarrow \theta + \alpha\frac{1}{N}\sum_i\sum_t\nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})(\hat Q_{i,t}-b)")

        st.code(r"""
# REINFORCE Algorithm
for episode = 1 … N:
    τ = {(s₁,a₁,r₁),...,(sT,aT,rT)} ~ π_θ    # collect trajectory
    G = 0
    for t = T … 1:
        G = rₜ + γ·G                            # reward-to-go (backward)
        b = V(sₜ) if using baseline else 0
        θ ← θ + α · ∇log π_θ(aₜ|sₜ) · (G − b) # gradient ascent!
""", language="text")

        c1,c2,c3 = st.columns(3)
        n_ep_r = c1.slider("Episodes", 50, 400, 200, 25, key="rf_ep")
        lr_r   = c1.select_slider("Learning rate α", [1e-4,5e-4,1e-3,5e-3,1e-2], 1e-2, key="rf_lr")
        use_b  = c2.checkbox("Use baseline V(s)", True, key="rf_bl")
        gam_r  = c2.slider("γ", 0.9, 1.0, 0.99, 0.01, key="rf_gm")
        sd_r   = c3.number_input("Seed", 0, 999, 42, key="rf_sd")

        if st.button("▶️ Train REINFORCE", type="primary", key="btn_rf"):
            with st.spinner("Training REINFORCE…"):
                rw, pg = train_reinforce(n_ep_r, lr_r, gam_r, use_b, int(sd_r))
            st.session_state["rf_res"] = (rw, pg)

        if "rf_res" in st.session_state:
            rw, pg = st.session_state["rf_res"]
            fig_rf, axes_rf = _fig(1, 3, 17, 4)
            sm = smooth(rw, 15)
            axes_rf[0].plot(rw, color=ALG_COL["REINFORCE"], alpha=0.15, lw=0.6)
            axes_rf[0].plot(range(len(sm)), sm, color=ALG_COL["REINFORCE"], lw=2.5, label="REINFORCE")
            axes_rf[0].axhline(195, color="#4caf50", ls="--", lw=1.2, label="Solved=195")
            axes_rf[0].set_xlabel("Episode", color="white"); axes_rf[0].set_ylabel("Reward", color="white")
            axes_rf[0].set_title("Learning Curve", color="white", fontweight="bold")
            axes_rf[0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_rf[0].grid(alpha=0.12)
            var_c = [np.var(rw[max(0,i-20):i+1]) for i in range(len(rw))]
            axes_rf[1].plot(smooth(var_c, 15), color="#ffa726", lw=2)
            axes_rf[1].set_xlabel("Episode", color="white"); axes_rf[1].set_ylabel("Variance", color="white")
            axes_rf[1].set_title("Reward Variance (the core REINFORCE problem)", color="white", fontweight="bold")
            axes_rf[1].grid(alpha=0.12)
            axes_rf[2].plot(smooth(pg, 20), color="#ef5350", lw=2)
            axes_rf[2].set_xlabel("Episode", color="white"); axes_rf[2].set_ylabel("||∇J||", color="white")
            axes_rf[2].set_title("Policy Gradient Norm", color="white", fontweight="bold")
            axes_rf[2].grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_rf); plt.close()
            c1,c2,c3 = st.columns(3)
            c1.metric("Best episode", f"{max(rw):.0f}/200")
            c2.metric("Late mean (last 30)", f"{np.mean(rw[-30:]):.1f}")
            c3.metric("Final variance", f"{np.var(rw[-30:]):.1f}")

    # ── ACTOR-CRITIC ──────────────────────────────────────────────────────
    with tab_ac_t:
        st.subheader("🎭 Vanilla Actor-Critic — Online 1-Step TD")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Actor** — policy network updated by TD error:")
            st.latex(r"\theta \leftarrow \theta + \alpha_\pi\,\delta_t\,\nabla_\theta\log\pi_\theta(a_t|s_t)")
        with col2:
            st.markdown("**Critic** — value network updated by TD(0):")
            st.latex(r"\omega \leftarrow \omega - \alpha_V\,\delta_t\,\nabla_\omega V_\omega(s_t)")
        st.markdown("**TD error (1-step advantage):**")
        st.latex(r"\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)")
        st.markdown(r"""
        **Symbol decoder:**
        - $\delta_t > 0$: outcome better than expected → increase probability of $a_t$
        - $\delta_t < 0$: outcome worse than expected → decrease probability of $a_t$
        - $r_t + \gamma V(s_{t+1})$ — TD target: 1 real reward + bootstrapped future
        - $V(s_t)$ — critic's baseline: how good was $s_t$ expected to be?
        """)
        st.code(r"""
Initialise Actor π_θ, Critic V_ω
For each episode:
    s ← env.reset()
    While not done:
        a ~ π_θ(·|s)                              # Actor chooses action
        s', r, done ← env.step(a)
        δ = r + γ·V_ω(s')·(1-done) - V_ω(s)      # TD error = advantage
        ω ← ω - α_V · δ · ∇V_ω(s)               # Critic: minimise δ²
        θ ← θ + α_π · δ · ∇log π_θ(a|s)          # Actor: gradient ascent
        s ← s'
""", language="text")

        c1,c2,c3 = st.columns(3)
        n_ep_ac = c1.slider("Episodes", 50, 300, 200, 25, key="ac_ep2")
        lr_pi_ac = c1.select_slider("Actor lr", [1e-4,5e-4,1e-3,5e-3], 5e-3, key="ac_lp2")
        lr_v_ac = c2.select_slider("Critic lr", [1e-3,5e-3,1e-2,2e-2], 1e-2, key="ac_lv2")
        sd_ac = c3.number_input("Seed", 0, 999, 42, key="ac_sd2")

        if st.button("▶️ REINFORCE vs AC", type="primary", key="btn_ac2"):
            with st.spinner("Training…"):
                rw_rf2, _ = train_reinforce(n_ep_ac, 1e-2, 0.99, True, int(sd_ac))
                rw_ac2, td_ac2 = train_ac(n_ep_ac, lr_pi_ac, lr_v_ac, 0.99, int(sd_ac))
            st.session_state["ac_res2"] = (rw_rf2, rw_ac2, td_ac2)

        if "ac_res2" in st.session_state:
            rw_rf2, rw_ac2, td_ac2 = st.session_state["ac_res2"]
            fig_ac2, axes_ac2 = _fig(1, 2, 13, 4.5)
            for rw, nm, col in [(rw_rf2, "REINFORCE+baseline", ALG_COL["REINFORCE"]),
                                  (rw_ac2, "Actor-Critic (1-step)", ALG_COL["AC"])]:
                sm = smooth(rw, 12)
                axes_ac2[0].plot(rw, color=col, alpha=0.12, lw=0.5)
                axes_ac2[0].plot(range(len(sm)), sm, color=col, lw=2.5, label=nm)
            axes_ac2[0].axhline(195, color="white", ls="--", lw=1, alpha=0.5)
            axes_ac2[0].set_xlabel("Episode", color="white"); axes_ac2[0].set_ylabel("Reward", color="white")
            axes_ac2[0].set_title("REINFORCE vs Actor-Critic", color="white", fontweight="bold")
            axes_ac2[0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_ac2[0].grid(alpha=0.12)
            sm_td = smooth(td_ac2, 50)
            axes_ac2[1].plot(range(len(sm_td)), sm_td, color=ALG_COL["AC"], lw=2)
            axes_ac2[1].axhline(0, color="white", ls="--", lw=0.8, alpha=0.4)
            axes_ac2[1].set_xlabel("Step", color="white"); axes_ac2[1].set_ylabel("TD error δ", color="white")
            axes_ac2[1].set_title("TD Error δ Over Training", color="white", fontweight="bold")
            axes_ac2[1].grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_ac2); plt.close()

    # ── A2C & A3C ─────────────────────────────────────────────────────────
    with tab_a2c:
        st.subheader("🤝 A2C & A3C — Advantage Actor-Critic (Mnih et al. 2016)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**n-step return (A2C core):**")
            st.latex(r"R_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n})")
            st.latex(r"\hat A_t = R_t^{(n)} - V(s_t)")
            st.markdown(r"""
            - $n=1$: vanilla AC (high bias, low variance)
            - $n=5$: A2C default (balanced)
            - $n=\infty$: REINFORCE (zero bias, very high variance)
            """)
        with col2:
            st.markdown("**A2C full joint objective:**")
            st.latex(r"\mathcal{L} = \underbrace{-\log\pi(a|s)\hat A}_{\text{policy loss}} + c_1\underbrace{(V-R)^2}_{\text{value loss}} - c_2\underbrace{H(\pi)}_{\text{entropy}}")
            st.markdown(r"""
            - $H(\pi) = -\sum_a\pi\log\pi$ — entropy bonus prevents premature convergence
            - $c_1 \approx 0.5$, $c_2 \approx 0.01$
            """)

        c1,c2 = st.columns(2)
        n_ep_a2c = c1.slider("Episodes", 50, 300, 200, 25, key="a2c_ep2")
        n_steps_a = c1.slider("n-steps", 2, 20, 5, 1, key="a2c_ns2")
        sd_a2c = c2.number_input("Seed", 0, 999, 42, key="a2c_sd2")
        if st.button("▶️ REINFORCE vs AC vs A2C", type="primary", key="btn_a2c2"):
            with st.spinner("Training…"):
                rw_rf3, _ = train_reinforce(n_ep_a2c, 0.01, 0.99, True, int(sd_a2c))
                rw_ac3, _ = train_ac(n_ep_a2c, 0.005, 0.01, 0.99, int(sd_a2c))
                rw_a2c2 = train_a2c(n_ep_a2c, 0.003, 0.01, 0.99, n_steps_a, 0.01, int(sd_a2c))
            st.session_state["a2c_res2"] = (rw_rf3, rw_ac3, rw_a2c2)
        if "a2c_res2" in st.session_state:
            rw_rf3, rw_ac3, rw_a2c2 = st.session_state["a2c_res2"]
            fig_a2, ax_a2 = _fig(1, 1, 12, 4.5)
            for rw, nm, col in [(rw_rf3, "REINFORCE+b", ALG_COL["REINFORCE"]),
                                  (rw_ac3, "AC (1-step)", ALG_COL["AC"]),
                                  (rw_a2c2, f"A2C (n={n_steps_a})", ALG_COL["A2C"])]:
                sm = smooth(rw, 12)
                ax_a2.plot(rw, color=col, alpha=0.12, lw=0.5)
                ax_a2.plot(range(len(sm)), sm, color=col, lw=2.5,
                           label=f"{nm} (late={np.mean(rw[-30:]):.1f})")
            ax_a2.axhline(195, color="white", ls="--", lw=1, alpha=0.5, label="Solved=195")
            ax_a2.set_xlabel("Episode", color="white"); ax_a2.set_ylabel("Reward", color="white")
            ax_a2.set_title("REINFORCE vs AC vs A2C", color="white", fontweight="bold")
            ax_a2.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_a2.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_a2); plt.close()

    # ── PPO ──────────────────────────────────────────────────────────────
    with tab_ppo:
        st.subheader("🔐 PPO — Proximal Policy Optimization (Schulman et al. 2017)")
        st.markdown(r"""PPO is the dominant modern policy gradient algorithm — used for ChatGPT RLHF,
        robotics, and game-playing. Its key innovation: a **clipped surrogate** prevents catastrophic updates.""")

        st.subheader("1. GAE — Generalised Advantage Estimation")
        st.markdown("**1-step TD error:**")
        st.latex(r"\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)")
        st.markdown("**GAE = exponentially-weighted sum of TD errors:**")
        st.latex(r"\hat A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}")
        st.markdown(r"""
        - $\lambda=0$: reduces to 1-step TD error (high bias, low variance)
        - $\lambda=1$: reduces to Monte Carlo advantage (zero bias, high variance)
        - Typical: $\lambda=0.95$
        """)

        lam_v = st.slider("λ (GAE decay)", 0.0, 1.0, 0.95, 0.05, key="ppo_lam2")
        gamma_v = 0.99
        weights = [( gamma_v*lam_v)**l for l in range(20)]
        weights = [w/sum(weights) for w in weights]
        fig_gae, ax_gae = _fig(1, 1, 10, 3.5)
        ax_gae.bar(range(20), weights, color=ALG_COL["PPO"], alpha=0.85, edgecolor="white", lw=0.3)
        ax_gae.set_xlabel("l (index of future TD error)", color="white")
        ax_gae.set_ylabel(r"Weight $(\gamma\lambda)^l$", color="white")
        ax_gae.set_title(f"GAE weights for λ={lam_v:.2f} (γ=0.99) — how much future δ contributes",
                         color="white", fontweight="bold")
        ax_gae.grid(alpha=0.12, axis="y"); plt.tight_layout(); st.pyplot(fig_gae); plt.close()

        st.divider()
        st.subheader("2. Clipped Surrogate Objective")
        st.markdown("**Probability ratio:**")
        st.latex(r"r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)}")
        st.markdown("**Clipped objective — pessimistic lower bound:**")
        st.latex(r"L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat A_t,\;\text{clip}(r_t(\theta),1-\varepsilon,1+\varepsilon)\hat A_t\right)\right]")
        st.markdown(r"""
        - If $\hat A_t > 0$: clip prevents $r_t$ going above $1+\varepsilon$ → no overconfident update
        - If $\hat A_t < 0$: clip prevents $r_t$ going below $1-\varepsilon$ → no catastrophic avoidance
        - $\varepsilon \approx 0.2$: policy can change at most 20% per update
        """)

        eps_v = st.slider("ε (clip range)", 0.05, 0.5, 0.2, 0.05, key="ppo_eps2")
        ratios = np.linspace(0.4, 1.8, 300)
        fig_clip, axes_clip = _fig(1, 2, 13, 4)
        for ax, adv, title in [(axes_clip[0], 0.8, "A > 0 (good action)"),
                                (axes_clip[1], -0.8, "A < 0 (bad action)")]:
            L_unc  = ratios*adv
            L_clip = np.minimum(ratios*adv, np.clip(ratios, 1-eps_v, 1+eps_v)*adv)
            ax.plot(ratios, L_unc,  color="#42a5f5", lw=2, ls="--", label="Unclipped")
            ax.plot(ratios, L_clip, color=ALG_COL["PPO"], lw=2.5, label="Clipped")
            ax.axvline(1, color="white", lw=0.8, alpha=0.5, ls=":")
            ax.axvline(1+eps_v, color="#ffa726", lw=1.2, ls="--", alpha=0.8)
            ax.axvline(1-eps_v, color="#ffa726", lw=1.2, ls="--", alpha=0.8, label=f"±{eps_v}")
            ax.set_xlabel(r"$r_t(\theta) = \pi_\theta/\pi_{ref}$", color="white")
            ax.set_ylabel("Objective", color="white")
            ax.set_title(title, color="white", fontweight="bold")
            ax.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_clip); plt.close()

        st.divider()
        st.subheader("3. Full PPO Loss")
        st.latex(r"\mathcal{L}(\theta) = \mathbb{E}_t\!\left[L^{\text{CLIP}}_t - c_1(V_\theta(s_t)-R_t)^2 + c_2 H(\pi_\theta(s_t))\right]")
        st.markdown(r"Where $R_t = \hat A_t + V_{\text{ref}}(s_t)$ (return target), $c_1\approx0.5$, $c_2\approx0.01$.")

        c1,c2,c3 = st.columns(3)
        n_it = c1.slider("PPO iterations", 20, 100, 50, 10, key="ppo_it2")
        clip_e = c2.slider("ε (clip)", 0.05, 0.4, 0.2, 0.05, key="ppo_cli2")
        gae_l = c3.slider("λ (GAE)", 0.5, 1.0, 0.95, 0.05, key="ppo_gl2")
        sd_ppo = c3.number_input("Seed", 0, 999, 42, key="ppo_sd2")

        if st.button("▶️ Train PPO", type="primary", key="btn_ppo2"):
            with st.spinner("Training PPO…"):
                rw_ppo2 = train_ppo(n_it, 512, 4, 3e-3, 5e-3, 0.99, gae_l, clip_e, 0.01, int(sd_ppo))
            with st.spinner("AC for comparison…"):
                rw_ac_c2, _ = train_ac(min(300, len(rw_ppo2)+50), 0.005, 0.01, 0.99, int(sd_ppo))
            st.session_state["ppo_res2"] = (rw_ppo2, rw_ac_c2)

        if "ppo_res2" in st.session_state:
            rw_ppo2, rw_ac_c2 = st.session_state["ppo_res2"]
            fig_ppo2, ax_ppo2 = _fig(1, 1, 12, 4.5)
            for rw, nm, col in [(rw_ac_c2, "Actor-Critic", ALG_COL["AC"]),
                                  (rw_ppo2, "PPO", ALG_COL["PPO"])]:
                sm = smooth(rw, 12)
                ax_ppo2.plot(rw, color=col, alpha=0.12, lw=0.5)
                ax_ppo2.plot(range(len(sm)), sm, color=col, lw=2.5,
                             label=f"{nm} (late={np.mean(rw[-30:]):.1f})")
            ax_ppo2.axhline(195, color="white", ls="--", lw=1, alpha=0.5)
            ax_ppo2.set_xlabel("Episode", color="white"); ax_ppo2.set_ylabel("Reward", color="white")
            ax_ppo2.set_title("Actor-Critic vs PPO", color="white", fontweight="bold")
            ax_ppo2.legend(facecolor=CARD, labelcolor="white", fontsize=9); ax_ppo2.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_ppo2); plt.close()

    # ── TRPO & SAC ────────────────────────────────────────────────────────
    with tab_trpo:
        st.subheader("🏛️ TRPO — Trust Region Policy Optimization (Schulman 2015)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**TRPO Objective (constrained optimisation):**")
            st.latex(r"\max_\theta\; L^{\text{PG}}(\theta) \quad \text{s.t.} \quad D_{\text{KL}}(\pi_\theta\|\pi_{\text{old}}) \leq \delta")
            st.markdown(r"Solved via conjugate gradient + line search. Stronger theory than PPO but $10\times$ harder to implement.")
        with col2:
            st.markdown("**Monotonic improvement guarantee:**")
            st.latex(r"J(\pi_{\text{new}}) \geq J(\pi_{\text{old}}) - \frac{4\gamma\max|A|}{(1-\gamma)^2}D_{\text{KL}}^{\max}(\pi_{\text{new}}\|\pi_{\text{old}})")
            st.markdown("PPO approximates this guarantee via clipping — same benefit, far simpler.")

        st.divider()
        st.subheader("🌡️ SAC — Soft Actor-Critic (Haarnoja 2018)")
        st.markdown("**Maximum entropy RL objective:**")
        st.latex(r"J(\pi) = \mathbb{E}_{\tau\sim\pi}\!\left[\sum_t\gamma^t\bigl(r(s_t,a_t)+\alpha\, H(\pi(\cdot|s_t))\bigr)\right]")
        st.markdown(r"""
        - $H(\pi) = -\mathbb{E}_{a\sim\pi}[\log\pi(a|s)]$ — entropy bonus: explore more
        - $\alpha$ — temperature: balances reward vs entropy; can be learned automatically
        - **Double Q-networks** reduce overestimation:
        """)
        st.latex(r"y = r + \gamma\bigl(\min_{j}Q_{\bar\theta_j}(s',a') - \alpha\log\pi(a'|s')\bigr)")
        st.dataframe(pd.DataFrame({
            "": ["Policy type","Action space","Data source","Key innovation","Sample efficiency"],
            "PPO": ["Stochastic","Discrete+Continuous","On-policy rollout","Clipped surrogate + GAE","Medium"],
            "TRPO": ["Stochastic","Discrete+Continuous","On-policy rollout","KL trust region","Medium"],
            "SAC": ["Stochastic","Continuous only","Replay buffer","Max entropy + 2 critics","Very high"],
        }), use_container_width=True, hide_index=True)

    # ── DASHBOARD ─────────────────────────────────────────────────────────
    with tab_dash:
        st.subheader("📈 Run All Policy Gradient Methods on CartPole")
        c1, c2 = st.columns(2)
        n_ep_d = c1.slider("Episodes (MC methods)", 50, 250, 150, 25, key="acd_ep")
        sd_d = c2.number_input("Seed", 0, 999, 42, key="acd_sd")
        if st.button("🚀 Run All", type="primary", key="btn_acd"):
            results = {}
            for nm, fn, args in [
                ("REINFORCE", train_reinforce, (n_ep_d, 0.01, 0.99, False, int(sd_d))),
                ("REINFORCE+Baseline", train_reinforce, (n_ep_d, 0.01, 0.99, True, int(sd_d))),
                ("Actor-Critic", train_ac, (n_ep_d, 0.005, 0.01, 0.99, int(sd_d))),
                ("A2C (n=5)", train_a2c, (n_ep_d, 0.003, 0.01, 0.99, 5, 0.01, int(sd_d))),
            ]:
                with st.spinner(f"{nm}…"):
                    r = fn(*args)
                    results[nm] = r[0] if isinstance(r, tuple) else r
            with st.spinner("PPO…"):
                results["PPO"] = train_ppo(30, 512, 4, 3e-3, 5e-3, 0.99, 0.95, 0.2, 0.01, int(sd_d))
            st.session_state["acd_res"] = results

        if "acd_res" in st.session_state:
            res = st.session_state["acd_res"]
            fig_d, ax_d = _fig(1, 1, 13, 5)
            pal = [ALG_COL["REINFORCE"],"#b39ddb",ALG_COL["AC"],ALG_COL["A2C"],ALG_COL["PPO"]]
            for (nm, rw), col in zip(res.items(), pal):
                sm = smooth(rw, 12)
                ax_d.plot(rw, color=col, alpha=0.1, lw=0.4)
                ax_d.plot(range(len(sm)), sm, color=col, lw=2.5,
                          label=f"{nm} (late={np.mean(rw[-30:]):.1f})")
            ax_d.axhline(195, color="white", ls="--", lw=1, alpha=0.5, label="Solved=195")
            ax_d.set_xlabel("Episode", color="white"); ax_d.set_ylabel("Reward", color="white")
            ax_d.set_title("Policy Gradient Family on CartPole", color="white", fontweight="bold")
            ax_d.legend(facecolor=CARD, labelcolor="white", fontsize=8,
                        bbox_to_anchor=(1.01,1), loc="upper left")
            ax_d.grid(alpha=0.12); plt.tight_layout(); st.pyplot(fig_d); plt.close()

    # ── STUDY PLAN ────────────────────────────────────────────────────────
    with tab_plan:
        st.subheader("📚 4-Week Policy Gradient Study Plan")
        for wt, ws, col, tasks in [
            ("Week 1","Foundations","#7c4dff",[
                ("📄","Sutton & Barto Ch.13","Derive the policy gradient theorem from scratch."),
                ("💻","Implement REINFORCE","CartPole. Compute G_t backward. Add baseline."),
                ("🧮","Prove baseline does not bias gradient","Show E[∇log π · b(s)] = 0."),
            ]),
            ("Week 2","Actor-Critic","#0288d1",[
                ("📄","Sutton et al. 1999 — Policy Gradient Theorem","Proof and connection to value functions."),
                ("💻","Implement 1-step Actor-Critic","Separate π and V networks. Track TD errors."),
                ("📊","Compare 1-step vs n-step","Plot: variance decreases as n increases from 1 to ∞."),
            ]),
            ("Week 3","PPO Deep Dive","#e65100",[
                ("📄","Schulman 2015 — GAE","Derive GAE from n-step returns. Verify λ=0 → 1-step TD."),
                ("📄","Schulman 2017 — PPO","Read Section 3 (clip) and Section 5 (implementation details)."),
                ("💻","Implement PPO from scratch","K epochs, mini-batches, clip, GAE, entropy bonus."),
            ]),
            ("Week 4","Advanced","#558b2f",[
                ("📄","Haarnoja 2018 — SAC","Soft Bellman equations and temperature auto-tuning."),
                ("💻","PPO on MuJoCo","Continuous action Gaussian policy. Use stable-baselines3 as reference."),
                ("📄","RLHF with PPO — InstructGPT","Ziegler et al. 2019. Apply policy gradient to language models."),
            ]),
        ]:
            st.markdown(f'<div style="background:{col}18;border-left:4px solid {col};'
                        f'border-radius:0 10px 10px 0;padding:.6rem 1rem;margin:.7rem 0 .3rem">'
                        f'<b style="color:{col}">{wt}: {ws}</b></div>', unsafe_allow_html=True)
            for icon, title, desc in tasks:
                st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;'
                            f'border-radius:8px;padding:.5rem .9rem;margin:.25rem 0;'
                            f'display:flex;gap:.7rem"><span>{icon}</span>'
                            f'<div><b style="color:white">{title}</b>'
                            f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span>'
                            f'</div></div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📖 Primary Resources")
        for icon, title, desc, url in [
            ("📗","Sutton & Barto Ch.13 — Policy Gradient Methods","Canonical theoretical treatment. Free online.","http://incompleteideas.net/book/the-book.html"),
            ("📄","Schulman et al. (2015) — GAE","Essential for PPO. Derives GAE and bias-variance trade-off.","https://arxiv.org/abs/1506.02438"),
            ("📄","Schulman et al. (2017) — PPO","The PPO paper. Short, readable, full of implementation tips.","https://arxiv.org/abs/1707.06347"),
            ("📄","Mnih et al. (2016) — A3C","Asynchronous workers. Shows parallel exploration helps.","https://arxiv.org/abs/1602.01783"),
            ("📄","Haarnoja et al. (2018) — SAC","Max entropy RL. State-of-the-art continuous control.","https://arxiv.org/abs/1801.01290"),
            ("💻","CleanRL — PPO Implementation","150-line single-file reference implementation.","https://github.com/vwxyzjn/cleanrl"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
