"""
_ac_mod.py  —  Policy Gradient & Actor-Critic Methods
Covers: REINFORCE · Vanilla AC · A2C · A3C · PPO (with GAE) · TRPO overview · SAC overview
Based on Sutton & Barto Ch.13, Schulman et al. 2015/2017, Haarnoja et al. 2018
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import warnings
from _notes_mod import render_notes
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

def render_ac_notes(tab_title: str, tab_slug: str) -> None:
    render_notes(f"Actor-Critic & Policy Gradient - {tab_title}", tab_slug)


def render_prerequisites_html() -> None:
    html_path = Path(__file__).resolve().parent / "portal_data" / "rl_dp_mc_td_formulas.html"
    components.html(html_path.read_text(encoding="utf-8"), height=900, scrolling=True)


def render_policy_gradient_nutshell_html() -> None:
    html_path = Path(__file__).resolve().parent / "portal_data" / "policy_gradient_deep_dive.html"
    components.html(html_path.read_text(encoding="utf-8"), height=900, scrolling=True)

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
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import matplotlib.gridspec as gridspec
    import pandas as pd

    DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"
    ALG = {"REINFORCE":"#7c4dff","AC":"#0288d1","A2C":"#00897b",
           "A3C":"#f57f17","PPO":"#e65100","TRPO":"#ad1457","SAC":"#558b2f"}

    def _fig(nr=1, nc=1, w=13, h=5):
        fig, axes = plt.subplots(nr, nc, figsize=(w, h))
        fig.patch.set_facecolor(DARK)
        for ax in np.array(axes).flatten():
            ax.set_facecolor(DARK)
            ax.tick_params(colors="#9e9ebb", labelsize=8)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        return fig, axes

    def _card(color, icon, title, body):
        return (f'<div style="background:{color}18;border-left:4px solid {color};'
                f'padding:1.1rem 1.3rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
                f'<b style="font-size:1rem">{icon} {title}</b><br>'
                f'<span style="color:#b0b0cc;font-size:.93rem;line-height:1.7">{body}</span></div>')

    def _proof_box(title, content):
        return (f'<div style="background:#0a1a2e;border:1px solid #1a3a5e;border-radius:10px;'
                f'padding:1rem 1.3rem;margin:.7rem 0">'
                f'<b style="color:#42a5f5">📐 Proof: {title}</b><br>'
                f'<span style="color:#b0b0cc;font-size:.9rem;line-height:1.8">{content}</span></div>')

    def _example_box(title, content, color="#00897b"):
        return (f'<div style="background:{color}0e;border:1px solid {color}44;border-radius:10px;'
                f'padding:1rem 1.3rem;margin:.7rem 0">'
                f'<b style="color:{color}">🎯 Example: {title}</b><br>'
                f'<span style="color:#b0b0cc;font-size:.9rem;line-height:1.8">{content}</span></div>')

    def _insight(t):
        return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
                f'padding:.8rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem;'
                f'color:#b0b0cc;line-height:1.7">💡 {t}</div>')

    def _warn(t):
        return (f'<div style="background:#2a1a0a;border-left:3px solid #ffa726;'
                f'padding:.8rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem;'
                f'color:#b0b0cc;line-height:1.7">⚠️ {t}</div>')

    def smooth(a, w=12):
        return np.convolve(a, np.ones(w)/w, mode="valid") if len(a)>w else np.array(a,float)

    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a0a2e,#0a1a0a,#2a0a0a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2.1rem">🎭 Actor-Critic &amp; Policy Gradient Methods</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'A complete derivation of every policy gradient algorithm — from first principles to production code. '
        'Each method explained by the problem it solves, the mathematics that solves it (with full proofs), '
        'and practical examples showing the before/after effect of each innovation.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "Prerequisites",
        "🗺️ Overview & Why PG?",
        "🎲 REINFORCE",
        "🎭 Actor-Critic",
        "🤝 A2C & A3C",
        "🔐 PPO",
        "🏛️ TRPO & SAC",
        "📈 Dashboard",
        "Policy Gradient in a nutshell",
        "🚀 Project",
        "📚 Study Plan",
    ])
    tab_pre, tab_ov, tab_rf, tab_ac_t, tab_a2c, tab_ppo, tab_trpo, tab_dash, tab_pg_nutshell, tab_project, tab_plan = tabs

    # ══════════════════════════════════════════════════════════════════
    # TAB 0 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════
    with tab_pre:
        render_prerequisites_html()

    with tab_ov:
        st.markdown(_card("#7c4dff","🤔","Why Policy Gradient? The Fundamental Motivation",
            """Value-based methods (DQN, Rainbow) learn a Q-function Q(s,a) and derive a policy by
            argmax_a Q(s,a). This works perfectly for discrete actions — Atari has 18 buttons,
            you can compute Q for all 18 and pick the best. But consider a robot arm: action =
            [torque on joint 1, torque on joint 2, ..., torque on joint 6], each a real number
            in [-5, 5]. That is an infinite, continuous action space — argmax over all real-valued
            actions is intractable. Policy gradient methods solve this by directly parameterising
            the policy: instead of Q(s,a) → argmax → action, we learn π_θ(a|s) directly —
            a neural network that outputs action probabilities (discrete) or a Gaussian distribution
            (continuous). The network is trained by gradient ascent on expected cumulative reward.
            This approach also handles stochastic policies naturally (important for partial
            observability and multi-agent settings), and is the foundation of RLHF — every modern
            AI assistant (ChatGPT, Claude, Gemini) was aligned using PPO, a policy gradient method."""),
            unsafe_allow_html=True)

        st.divider()
        st.subheader("🏛️ The Policy Gradient Theorem — Complete Derivation from First Principles")
        st.markdown(r"""
        **The objective:** Find parameters $\theta$ of policy $\pi_\theta(a|s)$ that maximise
        total expected discounted reward. We define this objective $J(\theta)$ as:
        """)
        st.latex(r"J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)}\!\left[\sum_{t=0}^T \gamma^t r(s_t,a_t)\right]")
        st.markdown(r"""
        where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T, a_T, r_T)$ is a trajectory
        and $p_\theta(\tau)$ is the probability of that trajectory under policy $\pi_\theta$:
        """)
        st.latex(r"p_\theta(\tau) = p(s_0)\prod_{t=0}^T \pi_\theta(a_t|s_t)\,p(s_{t+1}|s_t,a_t)")
        st.markdown(r"""
        **The challenge:** Computing $\nabla_\theta J(\theta)$ requires differentiating through
        $p_\theta(\tau)$, which involves $p(s_0)$ and $p(s_{t+1}|s_t,a_t)$ — the environment
        dynamics that we do not know and cannot differentiate through.
        """)
        st.markdown(_proof_box("The Log-Derivative Trick (REINFORCE gradient)",
            """We start from the definition of J(θ) as an integral:<br>
            J(θ) = ∫ p_θ(τ) r(τ) dτ<br><br>
            Taking the gradient with respect to θ:<br>
            ∇_θ J(θ) = ∫ ∇_θ p_θ(τ) r(τ) dτ<br><br>
            Key identity: ∇_θ p_θ(τ) = p_θ(τ) · ∇_θ log p_θ(τ)<br>
            (This follows from the chain rule: d/dθ log f(θ) = f'(θ)/f(θ), so f'(θ) = f(θ)·d/dθ log f(θ))<br><br>
            Substituting:<br>
            ∇_θ J(θ) = ∫ p_θ(τ) · ∇_θ log p_θ(τ) · r(τ) dτ = E_τ[∇_θ log p_θ(τ) · r(τ)]<br><br>
            Now expand log p_θ(τ):<br>
            log p_θ(τ) = log p(s_0) + Σ_t log π_θ(a_t|s_t) + Σ_t log p(s_{t+1}|s_t,a_t)<br><br>
            Critical step: ∂/∂θ [log p(s_0)] = 0 (no θ dependence)<br>
            Critical step: ∂/∂θ [log p(s_{t+1}|s_t,a_t)] = 0 (dynamics have no θ)<br><br>
            Therefore: ∇_θ log p_θ(τ) = Σ_t ∇_θ log π_θ(a_t|s_t)<br><br>
            FINAL RESULT: ∇_θ J(θ) = E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) · Σ_t r(s_t,a_t)]"""),
            unsafe_allow_html=True)

        st.latex(r"\boxed{\nabla_\theta J(\theta) = \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot \sum_t r(s_t,a_t)\right]}")
        st.markdown(r"""
        **What this means in plain English:**
        The term $\nabla_\theta \log\pi_\theta(a_t|s_t)$ is called the **score function**.
        It is the direction in parameter space $\theta$ that makes action $a_t$ more likely
        in state $s_t$. Multiplied by the total return, this says:
        - If total return was **positive** (good episode): push $\theta$ toward making the
          actions taken more probable — reinforce them
        - If total return was **negative** (bad episode): push $\theta$ away from those actions
        - The bigger the return, the stronger the push

        This is gradient **ascent** on $J(\theta)$ — we want to maximise reward.
        """)
        st.markdown(_insight("""
        <b>The key miracle:</b> The environment dynamics p(s_{t+1}|s_t,a_t) completely vanish
        from the gradient computation. We never need to know or differentiate through the
        environment model. This is why policy gradient methods are model-free —
        we only need to be able to sample from the environment and evaluate the policy.
        """), unsafe_allow_html=True)

        # Three variance-reduction steps visualised
        st.divider()
        st.subheader("📉 Three Steps to Reduce Variance — The Complete Picture")
        st.markdown(r"""
        The raw gradient estimate has very high variance. Three successive improvements reduce it:
        """)
        steps = [
            ("Step 1: Causality", "#7c4dff",
             r"Policy at time $t$ cannot affect rewards BEFORE $t$. Including them adds pure noise.",
             r"\nabla_\theta J \approx \frac{1}{N}\sum_i\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})\underbrace{\sum_{t'=t}^T\gamma^{t'-t}r_{t'}}_{\hat Q_{i,t}\ \text{(reward-to-go)}}"),
            ("Step 2: Baseline Subtraction", "#0288d1",
             r"Subtract $b(s_t)$ without biasing the gradient — reduces variance by centring returns.",
             r"\nabla_\theta J \approx \frac{1}{N}\sum_i\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t})(\hat Q_{i,t}-b(s_{i,t}))"),
            ("Step 3: Learned Value Baseline", "#00897b",
             r"Use $V(s_t)$ as baseline — the advantage $A_t = Q - V$ tells us if action was better than average.",
             r"A_t = Q(s_t,a_t) - V(s_t) \approx \delta_t = r_t+\gamma V(s_{t+1})-V(s_t)"),
        ]
        for title, col, desc, formula in steps:
            st.markdown(f'<div style="background:{col}12;border-left:4px solid {col};'
                        f'border-radius:0 10px 10px 0;padding:.9rem 1.2rem;margin:.5rem 0">'
                        f'<b style="color:{col}">{title}</b><br>'
                        f'<span style="color:#b0b0cc;font-size:.9rem">{desc}</span></div>',
                        unsafe_allow_html=True)
            st.latex(formula)

        # Family tree chart
        st.divider()
        st.subheader("📊 Algorithm Comparison Table")
        st.dataframe(pd.DataFrame({
            "Method":["REINFORCE","REINFORCE+baseline","Actor-Critic","A2C","PPO","SAC"],
            "Advantage estimate":["MC return $G_t$","$G_t - b$","1-step TD $\\delta_t$","n-step return","GAE$(\\gamma,\\lambda)$","Soft $Q-V$"],
            "Update frequency":["Episode end","Episode end","Every step","n steps","Rollout batch","Replay buffer"],
            "On/Off-policy":["On","On","On","On","On (clipped IS)","Off"],
            "Variance":["Very high","High","Low","Medium","Low","Very low"],
            "Bias":["Zero","Zero","Some","Less","Less","Some"],
            "Year":["1992","1992","1984","2016","2017","2018"],
        }), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — REINFORCE
    # ══════════════════════════════════════════════════════════════════
        render_ac_notes("Overview", "actor_critic_policy_gradient")

    with tab_rf:
        st.subheader("🎲 REINFORCE — Monte Carlo Policy Gradient (Williams, 1992)")

        st.markdown(_card("#7c4dff","📖","What problem REINFORCE solves",
            """Before REINFORCE, there was no general method to train stochastic policies with
            function approximation. Supervised learning requires labelled correct actions —
            but in RL, we never observe the correct action, only the eventual reward.
            REINFORCE (Williams 1992) was the first practical policy gradient algorithm.
            Its insight: treat the return as a weight on the gradient of the log-probability.
            Actions that led to high returns get their probabilities increased;
            actions that led to low returns get their probabilities decreased.
            No correct action label is needed — the return signal serves as a noisy
            supervision signal. REINFORCE works for any differentiable policy (neural network,
            linear model, etc.), any action space (discrete or continuous), and requires no
            model of the environment. Its Achilles heel is high variance: the total return
            G_t fluctuates wildly between episodes due to random future events, making the
            gradient estimate noisy. This variance problem motivated every subsequent algorithm
            in this module — each one is REINFORCE with a specific variance-reduction technique added."""),
            unsafe_allow_html=True)

        st.subheader("1. Complete Algorithm Derivation")
        st.markdown(r"""
        Starting from the policy gradient theorem (derived in the Overview tab):
        """)
        st.latex(r"\nabla_\theta J(\theta) = \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot \sum_{t'} r_{t'}\right]")
        st.markdown(r"**Step 1 — Apply causality.** The policy at time $t$ can only affect rewards at time $t' \geq t$. Including earlier rewards adds noise without signal:")
        st.latex(r"\nabla_\theta J = \mathbb{E}_\tau\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t) \cdot \underbrace{\sum_{t'=t}^T \gamma^{t'-t}r_{t'}}_{G_t = \text{reward-to-go}}\right]")
        st.markdown(r"**Step 2 — Monte Carlo estimate.** Run $N$ full episodes, average the gradients:")
        st.latex(r"\nabla_\theta J \approx \frac{1}{N}\sum_{i=1}^N\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t}) \cdot G_{i,t}")
        st.markdown(r"**Step 3 — Gradient ascent update:**")
        st.latex(r"\theta \leftarrow \theta + \alpha \cdot \frac{1}{N}\sum_i\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t}) \cdot G_{i,t}")

        st.subheader("2. The Baseline — Proof That It Doesn't Bias the Gradient")
        st.markdown(r"""
        A baseline $b(s_t)$ is any function of the state (not the action). We can subtract it
        from $G_t$ without changing the expected gradient. Here is the full proof:
        """)
        st.markdown(_proof_box("Baseline Does Not Bias the Gradient",
            """We need to show: E_τ[∇_θ log π_θ(a|s) · b(s)] = 0<br><br>
            Expand the expectation over actions (the state s is fixed):<br>
            E_{a∼π}[∇_θ log π_θ(a|s) · b(s)]<br>
            = b(s) · E_{a∼π}[∇_θ log π_θ(a|s)]   (b(s) is constant w.r.t. action a)<br>
            = b(s) · Σ_a π_θ(a|s) · ∇_θ log π_θ(a|s)<br>
            = b(s) · Σ_a π_θ(a|s) · ∇_θπ_θ(a|s) / π_θ(a|s)   (definition of log derivative)<br>
            = b(s) · Σ_a ∇_θπ_θ(a|s)   (cancel π_θ(a|s))<br>
            = b(s) · ∇_θ Σ_a π_θ(a|s)   (swap sum and gradient, valid for finite actions)<br>
            = b(s) · ∇_θ 1   (probabilities always sum to 1)<br>
            = b(s) · 0 = 0 ✓<br><br>
            Since adding b(s) to the gradient formula contributes zero in expectation,
            the gradient is unchanged. But variance IS reduced because b(s) centres the
            return signal around zero rather than its mean."""), unsafe_allow_html=True)

        st.markdown(r"""
        **With optimal baseline $b^*(s_t) = \mathbb{E}[G_t | s_t]$ (average return from this state):**
        """)
        st.latex(r"\nabla_\theta J \approx \frac{1}{N}\sum_i\sum_t \nabla_\theta\log\pi_\theta(a_{i,t}|s_{i,t}) \cdot (G_{i,t} - b(s_{i,t}))")

        st.subheader("3. Why High Variance Is a Fundamental Problem")
        st.markdown(r"""
        Consider running two episodes with the same state $s$ and same action $a$:
        - **Episode 1 (lucky):** opponent makes mistakes, environment is favourable → $G_t = 180$
        - **Episode 2 (unlucky):** everything goes wrong → $G_t = 20$

        The gradient updates are:
        """)
        st.latex(r"\text{Episode 1: } \Delta\theta_1 = \alpha \cdot 180 \cdot \nabla\log\pi(a|s)")
        st.latex(r"\text{Episode 2: } \Delta\theta_2 = \alpha \cdot 20 \cdot \nabla\log\pi(a|s)")
        st.markdown(r"""
        The gradient in episode 1 is **9× larger** than episode 2 for the same $(s,a)$ pair.
        The network has no way to know if action $a$ was inherently good or just lucky.
        To get a reliable gradient estimate, we need to average over many episodes —
        typically hundreds to thousands. This is the core efficiency problem of REINFORCE.

        **Variance of the estimator:**
        """)
        st.latex(r"\text{Var}[\nabla_\theta J] = \frac{1}{N}\,\text{Var}\!\left[\sum_t \nabla\log\pi \cdot G_t\right] \propto \frac{\text{Var}[G_t]}{N}")
        st.markdown(r"To halve the standard error, we need 4× more episodes. To reduce it by 10×, we need 100× more episodes. This is why REINFORCE is slow in practice.")

        # Interactive training
        st.subheader("🎛️ Interactive Training — See the Variance Problem Live")
        c1, c2, c3 = st.columns(3)
        n_ep_r = c1.slider("Episodes", 50, 500, 300, 25, key="rf2_ep")
        lr_r   = c1.select_slider("Learning rate α", [1e-4,5e-4,1e-3,5e-3,1e-2], 1e-2, key="rf2_lr")
        use_b  = c2.checkbox("Use baseline b(s) = V(s)", True, key="rf2_bl")
        gam_r  = c2.slider("γ (discount)", 0.9, 1.0, 0.99, 0.01, key="rf2_gm")
        sd_r   = c3.number_input("Random seed", 0, 999, 42, key="rf2_sd")

        if st.button("▶️ Train REINFORCE (both with and without baseline)", type="primary", key="btn_rf2"):
            with st.spinner("Training…"):
                rw_b,   pg_b  = train_reinforce(n_ep_r, lr_r, gam_r, True,  int(sd_r))
                rw_nb,  pg_nb = train_reinforce(n_ep_r, lr_r, gam_r, False, int(sd_r))
            st.session_state["rf2_res"] = (rw_b, pg_b, rw_nb, pg_nb)

        if "rf2_res" in st.session_state:
            rw_b, pg_b, rw_nb, pg_nb = st.session_state["rf2_res"]
            fig, axes = _fig(2, 2, 16, 8)
            # Reward curves
            for rw, lbl, col, ax in [(rw_b,"With baseline",ALG["REINFORCE"],axes[0,0]),
                                      (rw_nb,"No baseline","#ef5350",axes[0,0])]:
                sm = smooth(rw, 20)
                ax.plot(rw, color=col, alpha=0.12, lw=0.5)
                ax.plot(range(len(sm)), sm, color=col, lw=2.5, label=f"{lbl} (mean={np.mean(rw[-40:]):.1f})")
            axes[0,0].axhline(195, color="#4caf50", ls="--", lw=1.2, label="Solved = 195")
            axes[0,0].set_xlabel("Episode", color="white"); axes[0,0].set_ylabel("Reward", color="white")
            axes[0,0].set_title("Learning Curves", color="white", fontweight="bold")
            axes[0,0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes[0,0].grid(alpha=0.12)
            # Variance
            w = 30
            var_b  = [np.var(rw_b [max(0,i-w):i+1]) for i in range(len(rw_b))]
            var_nb = [np.var(rw_nb[max(0,i-w):i+1]) for i in range(len(rw_nb))]
            axes[0,1].plot(smooth(var_b,15),  color=ALG["REINFORCE"], lw=2.5, label="With baseline")
            axes[0,1].plot(smooth(var_nb,15), color="#ef5350", lw=2, label="No baseline")
            axes[0,1].set_xlabel("Episode", color="white"); axes[0,1].set_ylabel("Return variance", color="white")
            axes[0,1].set_title("Return Variance (the core REINFORCE problem)", color="white", fontweight="bold")
            axes[0,1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes[0,1].grid(alpha=0.12)
            # Gradient norms
            axes[1,0].plot(smooth(pg_b,15),  color=ALG["REINFORCE"], lw=2.5, label="With baseline")
            axes[1,0].plot(smooth(pg_nb,15), color="#ef5350", lw=2, label="No baseline")
            axes[1,0].set_xlabel("Episode", color="white"); axes[1,0].set_ylabel("||∇J||", color="white")
            axes[1,0].set_title("Policy Gradient Norm Over Training", color="white", fontweight="bold")
            axes[1,0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes[1,0].grid(alpha=0.12)
            # Distribution of returns
            axes[1,1].hist(rw_b[-100:],  bins=25, color=ALG["REINFORCE"], alpha=0.6, label="With baseline (last 100)")
            axes[1,1].hist(rw_nb[-100:], bins=25, color="#ef5350", alpha=0.6, label="No baseline (last 100)")
            axes[1,1].set_xlabel("Episode reward", color="white"); axes[1,1].set_ylabel("Count", color="white")
            axes[1,1].set_title("Return Distribution (last 100 episodes)", color="white", fontweight="bold")
            axes[1,1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes[1,1].grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Best episode", f"{max(rw_b):.0f}/200")
            c2.metric("Late mean (last 40)", f"{np.mean(rw_b[-40:]):.1f}")
            c3.metric("Variance reduction", f"{np.var(rw_nb[-40:])/max(np.var(rw_b[-40:]),1):.1f}×")
            c4.metric("Baseline benefit", f"+{np.mean(rw_b[-40:])-np.mean(rw_nb[-40:]):.1f} reward")

        st.markdown(_example_box("CartPole — What the Update Actually Does",
            """In CartPole, state = [position, velocity, angle, angular_velocity]. The policy network
            outputs [p_left, p_right]. Say the agent is in state s = [0.1, 0.2, 0.05, -0.1] and
            takes action 'right' (index 1). The episode gets total return G = 150.<br><br>
            The REINFORCE update: θ ← θ + α · 150 · ∇log π(right|s)<br><br>
            ∇log π(right|s) = [0 - p_left, 1 - p_right] = [-0.4, 0.6] for the output layer.
            This gradient pushes the network to increase p_right and decrease p_left in state s.
            The magnitude 150 means this push is proportional to how good the episode was.
            Next episode, if the same action gets return G = 30, the push is 5× smaller —
            this is exactly the variance problem: the update size depends on unrelated future randomness."""),
            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — ACTOR-CRITIC
    # ══════════════════════════════════════════════════════════════════
        render_ac_notes("REINFORCE", "actor_critic_policy_gradient_reinforce")

    with tab_ac_t:
        st.subheader("🎭 Vanilla Actor-Critic — Online TD-Based Policy Gradient")

        st.markdown(_card("#0288d1","🎭","What problem Actor-Critic solves (and what REINFORCE left broken)",
            """REINFORCE has two serious problems that Actor-Critic fixes simultaneously:
            (1) High variance — total return G_t fluctuates wildly because it includes all
            future randomness. Even identical (state, action) pairs get wildly different
            gradient magnitudes.
            (2) Efficiency — REINFORCE must complete an entire episode before updating.
            For games that last 10,000 steps, you get only 1 gradient update per 10,000 steps.
            Actor-Critic introduces a second neural network — the Critic V(s) — that learns
            to estimate the expected return from each state. The Critic provides a per-step
            advantage estimate δ_t = r_t + γV(s') - V(s) (TD error) that replaces the
            noisy Monte Carlo return G_t. This has two effects: (1) Variance is dramatically
            reduced because we only look one step ahead before bootstrapping, instead of summing
            all future random rewards; (2) The agent can update after every single step —
            no need to wait for the episode to end. Actor-Critic is the ancestor of every
            modern deep RL algorithm: DQN, A3C, PPO, SAC all use the actor-critic architecture.
            The tradeoff: by bootstrapping from V(s'), we introduce some bias (V is imperfect),
            but this is a worthy trade for the massive variance reduction and online updating."""),
            unsafe_allow_html=True)

        st.subheader("1. The Two Networks — Architecture and Purpose")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background:#0288d133;border:2px solid #0288d1;border-radius:10px;
                        padding:1.1rem 1.3rem">
            <b style="color:#42a5f5;font-size:1.05rem">🎭 Actor π_θ(a|s)</b><br>
            <span style="color:#b0b0cc;font-size:.9rem;line-height:1.7">
            Input: state s<br>
            Output: action probabilities (discrete) or mean/std (continuous)<br>
            Role: Decides what action to take<br>
            Update: Gradient ascent, weighted by advantage δ<br>
            Learns: Which actions are good in which states
            </span></div>""", unsafe_allow_html=True)
            st.latex(r"\theta \leftarrow \theta + \alpha_\pi\cdot\delta_t\cdot\nabla_\theta\log\pi_\theta(a_t|s_t)")
        with col2:
            st.markdown("""
            <div style="background:#0288d133;border:2px solid #0288d1;border-radius:10px;
                        padding:1.1rem 1.3rem">
            <b style="color:#42a5f5;font-size:1.05rem">🧮 Critic V_ω(s)</b><br>
            <span style="color:#b0b0cc;font-size:.9rem;line-height:1.7">
            Input: state s<br>
            Output: scalar — expected total return from s<br>
            Role: Evaluates how good the current state is<br>
            Update: Gradient descent on TD error squared<br>
            Learns: The value function V*(s)
            </span></div>""", unsafe_allow_html=True)
            st.latex(r"\omega \leftarrow \omega - \alpha_V\cdot\delta_t\cdot\nabla_\omega V_\omega(s_t)")

        st.subheader("2. The TD Error as Advantage Estimate — Derivation")
        st.markdown(r"""
        The true advantage function measures how much better action $a$ is than the average:
        """)
        st.latex(r"A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)")
        st.markdown(r"""
        We don't know $Q^\pi$ directly, but we know the Bellman equation:
        $Q^\pi(s_t,a_t) = r_t + \gamma V^\pi(s_{t+1})$ (by definition — take action $a_t$,
        get reward $r_t$, then follow $\pi$ from $s_{t+1}$).

        Substituting:
        """)
        st.latex(r"A^\pi(s_t,a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t) = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) = \delta_t")
        st.markdown(r"""
        The TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is exactly the 1-step advantage!
        **In plain English:**
        - $V(s_t)$ = "I expected to get this much total reward from state $s_t$"
        - $r_t + \gamma V(s_{t+1})$ = "I actually got $r_t$ now, plus the expected future"
        - $\delta_t > 0$: the outcome was **better than expected** → reinforce action $a_t$
        - $\delta_t < 0$: the outcome was **worse than expected** → suppress action $a_t$
        - $\delta_t = 0$: exactly as expected → no update needed

        This is remarkably similar to the dopamine prediction error signal in neuroscience —
        dopamine neurons fire more when reward is better than expected, less when it is worse.
        """)
        st.markdown(_insight("""
        <b>Why TD reduces variance dramatically:</b> REINFORCE uses G_t = Σ_{t'=t}^T γ^{t'-t} r_{t'} —
        the sum of ALL future rewards, each random. If T=200 steps, G_t sums 200 random variables.
        Actor-Critic uses δ_t = r_t + γV(s_{t+1}) — only ONE real reward plus a deterministic
        (low-variance) prediction V(s_{t+1}). The variance reduction is enormous in practice:
        typically 10–100× fewer gradient steps needed to converge.
        """), unsafe_allow_html=True)

        st.subheader("3. The Bias-Variance Tradeoff — Why Nothing Is Free")
        st.markdown(r"""
        Bootstrapping from $V(s_{t+1})$ introduces **bias** — the critic's estimate is imperfect.
        In early training, $V_\omega(s)$ is a random initialisation and is very wrong.
        The advantage estimate $\delta_t$ is therefore wrong too, giving the actor bad feedback.
        Over time, as $V_\omega$ improves, the bias decreases. The bias-variance tradeoff:
        """)
        st.latex(r"\underbrace{G_t}_{\text{MC: zero bias, high variance}} \longleftrightarrow \underbrace{r_t + \gamma V(s_{t+1})}_{\text{1-step TD: some bias, low variance}} \longleftrightarrow \underbrace{V(s_t)}_{\text{full bootstrap: max bias, zero variance}}")
        st.markdown(r"""
        The n-step return $R_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n})$
        interpolates between these extremes. $n=1$: Actor-Critic (low variance, some bias).
        $n=\infty$: REINFORCE (zero bias, high variance). GAE (used in PPO) takes a weighted average of ALL n.
        """)

        # Training comparison
        st.subheader("🎛️ Interactive: REINFORCE vs Actor-Critic — See the Difference")
        c1,c2,c3 = st.columns(3)
        n_ep_ac = c1.slider("Episodes", 50, 300, 200, 25, key="ac2_ep")
        lr_pi   = c1.select_slider("Actor lr α_π", [1e-4,5e-4,1e-3,5e-3], 5e-3, key="ac2_lp")
        lr_v    = c2.select_slider("Critic lr α_V", [1e-3,5e-3,1e-2,2e-2], 1e-2, key="ac2_lv")
        sd_ac   = c3.number_input("Seed", 0, 999, 42, key="ac2_sd")

        if st.button("▶️ Compare REINFORCE vs Actor-Critic", type="primary", key="btn_ac2"):
            with st.spinner("Training both…"):
                rw_rf, _ = train_reinforce(n_ep_ac, 1e-2, 0.99, True, int(sd_ac))
                rw_ac, td = train_ac(n_ep_ac, lr_pi, lr_v, 0.99, int(sd_ac))
            st.session_state["ac2_res"] = (rw_rf, rw_ac, td)

        if "ac2_res" in st.session_state:
            rw_rf, rw_ac, td = st.session_state["ac2_res"]
            fig, axes = _fig(1, 3, 17, 4.5)
            for rw, nm, col in [(rw_rf,"REINFORCE+baseline",ALG["REINFORCE"]),
                                  (rw_ac,"Actor-Critic",ALG["AC"])]:
                sm = smooth(rw, 15)
                axes[0].plot(rw, color=col, alpha=0.12, lw=0.5)
                axes[0].plot(range(len(sm)), sm, color=col, lw=2.5, label=f"{nm} (mean={np.mean(rw[-30:]):.1f})")
            axes[0].axhline(195, color="white", ls="--", lw=1, alpha=0.5)
            axes[0].set_title("Learning Curves", color="white", fontweight="bold")
            axes[0].set_xlabel("Episode", color="white"); axes[0].set_ylabel("Reward", color="white")
            axes[0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes[0].grid(alpha=0.12)
            sm_td = smooth(td, 80)
            axes[1].plot(range(len(sm_td)), sm_td, color=ALG["AC"], lw=2)
            axes[1].axhline(0, color="white", ls="--", lw=0.8, alpha=0.4)
            axes[1].set_title("TD Error δ (Critic Learning Signal)", color="white", fontweight="bold")
            axes[1].set_xlabel("Step", color="white"); axes[1].set_ylabel("δ = r + γV(s') − V(s)", color="white")
            axes[1].grid(alpha=0.12)
            var_rf = [np.var(rw_rf[max(0,i-30):i+1]) for i in range(len(rw_rf))]
            var_ac = [np.var(rw_ac[max(0,i-30):i+1]) for i in range(len(rw_ac))]
            axes[2].plot(smooth(var_rf,15), color=ALG["REINFORCE"], lw=2, label="REINFORCE variance")
            axes[2].plot(smooth(var_ac,15), color=ALG["AC"], lw=2.5, label="Actor-Critic variance")
            axes[2].set_title("Variance Comparison (AC wins)", color="white", fontweight="bold")
            axes[2].set_xlabel("Episode", color="white"); axes[2].set_ylabel("Return variance", color="white")
            axes[2].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes[2].grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(_example_box("How δ Guides Learning — A Concrete Walkthrough",
            """Consider CartPole. Critic predicts V(upright_state) = 120 (expects 120 more reward).<br><br>
            <b>Case 1 — Better than expected:</b><br>
            Agent keeps pole up for 1 step: r=1, V(next_upright) = 122<br>
            δ = 1 + 0.99×122 − 120 = 1 + 120.78 − 120 = <b>+1.78</b><br>
            Actor update: increase prob of taken action by +1.78 × gradient<br>
            Critic update: V(upright) ← V(upright) − α×(−1.78) → V increases to ~121.8<br><br>
            <b>Case 2 — Worse than expected:</b><br>
            Pole falls: r=0, V(fallen) = 0<br>
            δ = 0 + 0.99×0 − 120 = <b>−120</b><br>
            Actor update: strongly decrease prob of action that led to fall<br>
            Critic update: V(upright) ← V(upright) − α×(120) → V decreases to ~108<br><br>
            The critic corrects itself AND guides the actor every single step."""),
            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — A2C & A3C
    # ══════════════════════════════════════════════════════════════════
        render_ac_notes("Actor-Critic Theory", "actor_critic_policy_gradient_actor_critic_theory")

    with tab_a2c:
        st.subheader("🤝 A2C & A3C — Advantage Actor-Critic (Mnih et al. 2016)")

        st.markdown(_card("#00897b","🤝","What problem A2C solves over vanilla Actor-Critic",
            """Vanilla Actor-Critic with 1-step TD has high bias in early training (V is wrong)
            and the actor receives correlated gradient updates (consecutive states are similar —
            seeing state s followed by s' gives almost no new information about the reward landscape).
            A2C (Advantage Actor-Critic) fixes both problems simultaneously:
            (1) Collects n steps of experience before updating — n-step returns have less bias than
            1-step TD while having less variance than full Monte Carlo.
            (2) The n-step batch naturally decorrelates updates since it covers a wider trajectory.
            (3) Adds an entropy bonus H(π) to the policy loss that explicitly prevents the policy
            from collapsing to a single deterministic action prematurely.
            (4) Normalises advantages across the batch so the actor gradient scale is stable.
            A2C also introduced the synchronous parallel workers pattern: multiple environments
            run simultaneously, collecting diverse experience in lockstep, then a single gradient
            update uses all of their data. This is more stable than the original A3C's asynchronous
            updates because all workers always use the same model version — no staleness.
            A2C demonstrated that with 16 parallel workers, learning speed improved ~16× with
            the same amount of data, making it the first practical scalable deep RL algorithm."""),
            unsafe_allow_html=True)

        st.subheader("1. n-step Returns — Deriving the Bias-Variance Tradeoff")
        st.markdown(r"""
        **Definition:** The n-step return from time $t$ is the sum of $n$ actual rewards
        plus a bootstrap estimate at step $t+n$:
        """)
        st.latex(r"R_t^{(n)} = \underbrace{r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{n-1}r_{t+n-1}}_{n\text{ real rewards}} + \underbrace{\gamma^n V_\omega(s_{t+n})}_{\text{bootstrap here}}")
        st.markdown(r"""
        **n-step advantage:**
        """)
        st.latex(r"\hat A_t^{(n)} = R_t^{(n)} - V_\omega(s_t)")
        st.markdown(r"""
        **Why does increasing $n$ reduce bias?**
        The bias comes from using $V_\omega(s_{t+n})$ — an imperfect estimate of the true
        $V^*(s_{t+n})$. The effect of this bias is discounted by $\gamma^n$. As $n$ increases,
        $\gamma^n \to 0$, so the bias contribution shrinks:
        """)
        st.latex(r"\text{Bias}(\hat A_t^{(n)}) = \gamma^n\bigl(V_\omega(s_{t+n}) - V^*(s_{t+n})\bigr)")
        st.markdown(r"""
        **Why does increasing $n$ increase variance?**
        Each additional real reward $r_{t+k}$ adds one more random variable to the sum.
        By the variance addition rule (for independent terms):
        """)
        st.latex(r"\text{Var}(R_t^{(n)}) \approx \sum_{k=0}^{n-1}\gamma^{2k}\text{Var}(r_{t+k}) + \gamma^{2n}\text{Var}(V_\omega(s_{t+n}))")
        st.markdown(r"This grows with $n$, but is bounded by the geometric series. In practice, $n=5$ works well.")

        # Bias-variance chart
        n_vals = np.arange(1, 31)
        gamma_v = 0.99; err_V = 5.0; r_var = 2.0
        bias = [gamma_v**n * err_V for n in n_vals]
        variance = [sum(gamma_v**(2*k)*r_var for k in range(n)) + gamma_v**(2*n)*0.5 for n in n_vals]
        total_err = [b+v for b,v in zip(bias, variance)]
        fig_bv, ax_bv = _fig(1, 1, 12, 4)
        ax_bv.plot(n_vals, bias, color="#ef5350", lw=2.5, label="Bias (γⁿ × V error) — decreases with n")
        ax_bv.plot(n_vals, variance, color="#42a5f5", lw=2.5, label="Variance — increases with n")
        ax_bv.plot(n_vals, total_err, color="#ffa726", lw=2.5, ls="--", label="Total error (bias + variance)")
        ax_bv.axvline(5, color="#4caf50", ls="--", lw=1.5, alpha=0.7, label="A2C default n=5")
        ax_bv.fill_between(n_vals, 0, bias, alpha=0.1, color="#ef5350")
        ax_bv.fill_between(n_vals, 0, variance, alpha=0.1, color="#42a5f5")
        ax_bv.set_xlabel("n (number of steps before bootstrap)", color="white")
        ax_bv.set_ylabel("Error magnitude", color="white")
        ax_bv.set_title("n-step Return: Bias decreases, Variance increases — A2C balances at n=5",
                         color="white", fontweight="bold")
        ax_bv.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_bv.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_bv); plt.close()

        st.subheader("2. The Entropy Bonus — Preventing Premature Policy Collapse")
        st.markdown(r"""
        Without entropy regularisation, the policy often collapses to a deterministic distribution
        early in training: one action gets probability ~1, all others ~0. This is catastrophic
        if the high-probability action is suboptimal — the agent cannot explore.

        **Shannon entropy of the policy:**
        """)
        st.latex(r"H(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s)")
        st.markdown(r"""
        - $H = \log|A|$ (maximum): uniform distribution — maximum exploration
        - $H = 0$ (minimum): deterministic policy — zero exploration

        **Proof that entropy bonus prevents collapse:**
        The entropy term adds $c_2 H(\pi)$ to the objective. Taking the gradient with respect
        to the logits $z_a$ (where $\pi(a|s) = \text{softmax}(z_a)$):
        """)
        st.latex(r"\frac{\partial H}{\partial z_a} = -\pi(a|s)(1 + \log\pi(a|s)) - \sum_{a'}\pi(a'|s)(-\pi(a|s))(1+\log\pi(a'|s))")
        st.markdown(r"""
        Simplified: $\partial H / \partial z_a = -\pi(a|s)[\log\pi(a|s) - H(\pi)]$.
        When $\pi(a|s)$ is very large (collapsed): the gradient pushes $z_a$ down,
        counteracting the collapse. The entropy gradient acts as a restoring force.

        **A2C full joint loss (3 terms):**
        """)
        st.latex(r"\mathcal{L}(\theta,\omega) = \underbrace{-\frac{1}{T}\sum_t\log\pi_\theta(a_t|s_t)\hat A_t^{(n)}}_{\text{policy loss (actor)}} + \underbrace{c_1\frac{1}{T}\sum_t\bigl(V_\omega(s_t)-R_t^{(n)}\bigr)^2}_{\text{value loss (critic)}} - \underbrace{c_2\frac{1}{T}\sum_t H(\pi_\theta(\cdot|s_t))}_{\text{entropy bonus}}")
        st.markdown(r"""
        **Symbol decoder:**
        - Policy loss: negative because we maximise $J$ (gradient ascent), but implement with a minimiser
        - Value loss: MSE between critic's prediction $V_\omega(s_t)$ and n-step return target
        - Entropy bonus: subtracted (negative sign) because we maximise it — higher entropy = better
        - $c_1 \approx 0.5$: value loss coefficient — critic learns at half rate to prevent actor domination
        - $c_2 \approx 0.01$: entropy coefficient — small but critical for exploration maintenance
        """)

        # Entropy demo
        st.markdown("**Entropy vs policy distribution:**")
        p_val = st.slider("Probability of action 0 (out of 2 actions)", 0.01, 0.99, 0.5, 0.01, key="a2c_ent_demo")
        p_arr = np.array([p_val, 1-p_val])
        H_val = -sum(p*np.log(p) for p in p_arr)
        H_max = np.log(2)
        fig_ent, axes_ent = _fig(1, 2, 12, 3.5)
        axes_ent[0].bar(["Action 0", "Action 1"], p_arr, color=[ALG["A2C"],"#546e7a"])
        axes_ent[0].set_title(f"Policy Distribution (p0={p_val:.2f})", color="white", fontweight="bold")
        axes_ent[0].set_ylabel("Probability", color="white"); axes_ent[0].grid(alpha=0.1, axis="y")
        p_range = np.linspace(0.01, 0.99, 200)
        H_curve = -p_range*np.log(p_range) - (1-p_range)*np.log(1-p_range)
        axes_ent[1].plot(p_range, H_curve, color=ALG["A2C"], lw=2.5)
        axes_ent[1].axvline(p_val, color="#ef5350", ls="--", lw=2, label=f"Current: H={H_val:.3f} nats")
        axes_ent[1].axhline(H_max, color="#4caf50", ls=":", lw=1.5, alpha=0.7, label=f"Max entropy: {H_max:.3f}")
        axes_ent[1].set_xlabel("P(action=0)", color="white"); axes_ent[1].set_ylabel("Entropy H(π)", color="white")
        axes_ent[1].set_title(f"Entropy: {H_val:.3f} nats ({100*H_val/H_max:.0f}% of max)",
                               color="white", fontweight="bold")
        axes_ent[1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_ent[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ent); plt.close()
        st.markdown(f"At p={p_val:.2f}: entropy = {H_val:.3f} nats. Maximum possible = {H_max:.3f} nats (uniform distribution).")

        # Training
        c1,c2 = st.columns(2)
        n_ep_a2c = c1.slider("Episodes", 50, 300, 200, 25, key="a2c2_ep")
        n_steps  = c1.slider("n-step rollout", 1, 20, 5, 1, key="a2c2_ns")
        ent_c    = c2.slider("Entropy coeff c₂", 0.0, 0.1, 0.01, 0.005, key="a2c2_ent")
        sd_a2c   = c2.number_input("Seed", 0, 999, 42, key="a2c2_sd")
        if st.button("▶️ Compare REINFORCE, AC, A2C", type="primary", key="btn_a2c2"):
            with st.spinner("Training all three…"):
                rw_rf2,_ = train_reinforce(n_ep_a2c, 0.01, 0.99, True, int(sd_a2c))
                rw_ac2,_ = train_ac(n_ep_a2c, 0.005, 0.01, 0.99, int(sd_a2c))
                rw_a2c2  = train_a2c(n_ep_a2c, 0.003, 0.01, 0.99, n_steps, ent_c, int(sd_a2c))
            st.session_state["a2c2_res"] = (rw_rf2, rw_ac2, rw_a2c2)
        if "a2c2_res" in st.session_state:
            rw_rf2, rw_ac2, rw_a2c2 = st.session_state["a2c2_res"]
            fig_a2, ax_a2 = _fig(1, 1, 13, 4.5)
            for rw, nm, col in [(rw_rf2,"REINFORCE+b",ALG["REINFORCE"]),
                                  (rw_ac2,"AC (1-step)",ALG["AC"]),
                                  (rw_a2c2,f"A2C (n={n_steps})",ALG["A2C"])]:
                sm = smooth(rw, 15)
                ax_a2.plot(rw, color=col, alpha=0.12, lw=0.5)
                ax_a2.plot(range(len(sm)), sm, color=col, lw=2.5, label=f"{nm} (late mean={np.mean(rw[-30:]):.1f})")
            ax_a2.axhline(195, color="white", ls="--", lw=1, alpha=0.5, label="Solved=195")
            ax_a2.set_xlabel("Episode",color="white"); ax_a2.set_ylabel("Reward",color="white")
            ax_a2.set_title(f"REINFORCE vs AC vs A2C (n={n_steps}, entropy_c={ent_c:.3f})",
                            color="white", fontweight="bold")
            ax_a2.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_a2.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_a2); plt.close()

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — PPO
    # ══════════════════════════════════════════════════════════════════
        render_ac_notes("A2C / A3C", "actor_critic_policy_gradient_a2c_a3c")

    with tab_ppo:
        st.subheader("🔐 PPO — Proximal Policy Optimization (Schulman et al. 2017)")

        st.markdown(_card("#e65100","🏆","What problem PPO solves and why it dominates modern RL",
            """The fundamental problem with vanilla policy gradient methods (including A2C) is that
            gradient steps can be catastrophically large. If a gradient step pushes the policy
            parameters θ too far, the new policy π_new can be completely different from π_old —
            and performance can collapse irreversibly. Unlike supervised learning where a bad
            gradient step can be undone by re-evaluating on the same data, RL has a catch:
            the policy determines which data you collect next. A bad policy → bad training data →
            even worse policy. This is the training instability problem.
            TRPO (2015) solved this theoretically using KL-constrained trust regions but required
            expensive second-order optimisation (computing the Fisher information matrix).
            PPO (2017) achieves the same practical stability with a brilliantly simple trick:
            clip the probability ratio r_t(θ) = π_new/π_old to stay within [1-ε, 1+ε].
            This prevents the policy from moving too far in a single update while remaining
            first-order (just SGD). The result: state-of-the-art performance with a simple
            implementation that runs on standard hardware. PPO now powers ChatGPT training (RLHF),
            Claude alignment, OpenAI Five, robotics locomotion, and is the most-used RL algorithm
            in production systems worldwide."""), unsafe_allow_html=True)

        st.subheader("1. GAE — Generalised Advantage Estimation (Schulman et al. 2015)")
        st.markdown(r"""
        PPO uses GAE instead of simple n-step returns. GAE is a key innovation that provides
        a single parameter λ to tune the entire bias-variance spectrum continuously.

        **Derivation of GAE:**
        Start with the 1-step TD error: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$

        The n-step return can be written as a sum of TD errors:
        """)
        st.latex(r"R_t^{(n)} - V(s_t) = \sum_{k=0}^{n-1}\gamma^k\delta_{t+k}")
        st.markdown(_proof_box("n-step return = sum of TD errors",
            """R_t^(n) - V(s_t)<br>
            = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n}) - V(s_t)<br>
            = [r_t + γV(s_{t+1}) - V(s_t)] + γ[r_{t+1} + γV(s_{t+2}) - V(s_{t+1})] + ... + γ^{n-1}[...]<br>
            = δ_t + γδ_{t+1} + γ²δ_{t+2} + ... + γ^{n-1}δ_{t+n-1}<br>
            = Σ_{k=0}^{n-1} γ^k δ_{t+k}  ✓"""), unsafe_allow_html=True)

        st.markdown(r"""
        **GAE takes a weighted average of ALL n-step advantages:**
        """)
        st.latex(r"\hat A_t^{\text{GAE}(\gamma,\lambda)} = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}\hat A_t^{(n)}")
        st.markdown(_proof_box("GAE simplifies to sum of decayed TD errors",
            """Expand the weighted sum: (1-λ)Σ_{n=1}^∞ λ^{n-1} Σ_{k=0}^{n-1} γ^k δ_{t+k}<br>
            Swap the order of summation (each δ_{t+k} appears in all n > k terms):<br>
            = (1-λ) Σ_{k=0}^∞ δ_{t+k} γ^k Σ_{n=k+1}^∞ λ^{n-1}<br>
            = (1-λ) Σ_{k=0}^∞ δ_{t+k} γ^k · λ^k/(1-λ)   (geometric series)<br>
            = Σ_{k=0}^∞ (γλ)^k δ_{t+k}<br><br>
            FINAL: Â_t^GAE(γ,λ) = Σ_{k=0}^∞ (γλ)^k δ_{t+k} = δ_t + γλδ_{t+1} + (γλ)²δ_{t+2} + ..."""),
            unsafe_allow_html=True)

        st.latex(r"\boxed{\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{k=0}^\infty(\gamma\lambda)^k\delta_{t+k}}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **λ=0:** Only $\delta_t$ — pure 1-step TD advantage:
            $\hat A_t = \delta_t = r_t + \gamma V(s') - V(s)$
            High bias (V must be accurate), low variance.
            """)
        with col2:
            st.markdown(r"""
            **λ=1:** Full infinite sum = MC advantage:
            $\hat A_t = G_t - V(s_t)$
            Zero bias (no bootstrapping), high variance.
            PPO default: **λ = 0.95** (effective ~20 step lookback)
            """)

        lam_demo = st.slider("λ (GAE decay) — drag to see how lookback changes", 0.0, 1.0, 0.95, 0.05, key="ppo2_lam")
        gv = 0.99; n_show = 25
        weights = [(gv*lam_demo)**k for k in range(n_show)]
        w_norm = [w/sum(weights) for w in weights]
        eff_n = 1/(1-gv*lam_demo) if lam_demo < 1 else float("inf")
        fig_gae, ax_gae = _fig(1, 1, 12, 3.5)
        bars = ax_gae.bar(range(n_show), w_norm, color=ALG["PPO"], alpha=0.85, edgecolor="white", lw=0.3)
        ax_gae.set_xlabel("Future TD error index k (δ_{t+k})", color="white")
        ax_gae.set_ylabel(r"Weight $(γλ)^k$ (normalised)", color="white")
        ax_gae.set_title(f"GAE weights for λ={lam_demo:.2f} — effective lookback ≈ {min(eff_n,100):.0f} steps",
                         color="white", fontweight="bold")
        ax_gae.grid(alpha=0.12, axis="y"); plt.tight_layout(); st.pyplot(fig_gae); plt.close()

        st.subheader("2. The Clipped Surrogate Objective — Full Derivation")
        st.markdown(r"""
        **Setting up importance sampling:**
        PPO collects data with policy $\pi_{\text{ref}}$ (the current policy), then runs $K$ gradient
        update epochs on that data. After the first epoch, $\theta$ has changed — the data distribution
        has shifted. Importance sampling corrects for this:
        """)
        st.latex(r"J(\theta) = \mathbb{E}_{(s,a)\sim\pi_{\text{ref}}}\!\left[\frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}\hat A\right] = \mathbb{E}\!\left[r_t(\theta)\hat A_t\right]")
        st.latex(r"r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} \quad\text{(probability ratio — starts at 1, drifts as θ updates)}")
        st.markdown(r"""
        The unclipped surrogate $L^{\text{PG}} = \mathbb{E}[r_t(\theta)\hat A_t]$ can encourage
        arbitrarily large $r_t$ when $\hat A_t > 0$, leading to destabilisingly large updates.
        **PPO clips** the ratio to prevent this:
        """)
        st.latex(r"L^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat A_t,\;\underbrace{\text{clip}(r_t(\theta),\,1-\varepsilon,\,1+\varepsilon)}_{\text{restricted to }[1-\varepsilon,1+\varepsilon]}\cdot\hat A_t\right)\right]")
        st.markdown(_proof_box("Why min() gives a pessimistic lower bound",
            """Case 1 — Positive advantage (A > 0): action is good, want to increase probability.<br>
            If r_t > 1+ε: unclipped = r_t·A > (1+ε)·A, clipped = (1+ε)·A<br>
            min() selects the CLIPPED (smaller) value — prevents over-reinforcing<br>
            If r_t < 1+ε: unclipped = clipped → min selects either → no clipping<br><br>
            Case 2 — Negative advantage (A < 0): action is bad, want to decrease probability.<br>
            If r_t < 1-ε: unclipped = r_t·A < (1-ε)·A (less negative), clipped = (1-ε)·A<br>
            min() selects the CLIPPED (smaller/more negative) value — prevents over-penalising<br><br>
            Result: the gradient can never benefit from moving r_t outside [1-ε, 1+ε].
            The policy is implicitly constrained to a trust region around π_ref."""), unsafe_allow_html=True)

        # Interactive clip visualisation
        eps_v = st.slider("ε (clip range)", 0.05, 0.5, 0.2, 0.05, key="ppo2_eps")
        ratios = np.linspace(0.3, 2.0, 300)
        fig_clip, axes_clip = _fig(1, 2, 14, 4.5)
        for ax, adv, title, color in [
            (axes_clip[0], 0.8, "A > 0: Good action — clip prevents over-reinforcing", "#4caf50"),
            (axes_clip[1], -0.8, "A < 0: Bad action — clip prevents over-penalising", "#ef5350")
        ]:
            L_raw  = ratios * adv
            L_clip = np.minimum(ratios*adv, np.clip(ratios, 1-eps_v, 1+eps_v)*adv)
            # Gradient of clipped objective
            grad_clip = np.where((ratios > 1-eps_v) & (ratios < 1+eps_v), adv, 0)
            ax.plot(ratios, L_raw,  color="#42a5f5", lw=2, ls="--", alpha=0.8, label="Unclipped L^PG (dangerous)")
            ax.plot(ratios, L_clip, color=color, lw=2.5, label=f"Clipped L^CLIP (ε={eps_v})")
            ax.axvline(1, color="white", lw=0.8, ls=":", alpha=0.5, label="r_t=1 (no drift)")
            ax.axvline(1+eps_v, color="#ffa726", lw=1.5, ls="--", alpha=0.8, label=f"Clip boundary ±{eps_v}")
            ax.axvline(1-eps_v, color="#ffa726", lw=1.5, ls="--", alpha=0.8)
            ax.fill_between(ratios[(ratios>1-eps_v)&(ratios<1+eps_v)],
                            L_clip[(ratios>1-eps_v)&(ratios<1+eps_v)],
                            alpha=0.15, color=color, label="Region of active gradient")
            ax.set_xlabel(r"$r_t(\theta) = \pi_\theta / \pi_\mathrm{ref}$", color="white")
            ax.set_ylabel("Objective value", color="white")
            ax.set_title(title, color="white", fontweight="bold")
            ax.legend(facecolor=CARD, labelcolor="white", fontsize=7.5); ax.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_clip); plt.close()
        st.caption(f"ε={eps_v}: Policy can move at most {eps_v*100:.0f}% away from reference before gradient is zeroed. The clipped objective is always the PESSIMISTIC (minimum) bound.")

        st.subheader("3. Complete PPO Loss — All Three Terms")
        st.latex(r"\mathcal{L}^\text{PPO}(\theta) = \mathbb{E}_t\!\left[L_t^\text{CLIP}(\theta) - c_1\underbrace{(V_\theta(s_t)-R_t)^2}_\text{value loss} + c_2\underbrace{H(\pi_\theta(\cdot|s_t))}_\text{entropy}\right]")
        st.markdown(r"""
        Where $R_t = \hat A_t^\text{GAE} + V_\text{ref}(s_t)$ is the return target.
        $c_1 \approx 0.5$, $c_2 \approx 0.01$.

        **PPO training loop:**
        1. Collect rollout of $T$ steps with $\pi_\theta$ (the reference policy)
        2. Compute GAE advantages $\hat A_t$ backward from $t=T$ to $t=0$
        3. For $K$ epochs: shuffle rollout into mini-batches, compute $L^\text{PPO}$, update $\theta$
        4. The updated $\theta$ becomes the new reference policy for the next rollout
        """)

        # PPO training
        c1,c2,c3 = st.columns(3)
        n_it   = c1.slider("Iterations", 20, 100, 50, 10, key="ppo2_it")
        clip_e = c2.slider("ε clip", 0.05, 0.4, 0.2, 0.05, key="ppo2_cli")
        gae_l  = c3.slider("λ GAE", 0.5, 1.0, 0.95, 0.05, key="ppo2_gl")
        sd_ppo = c3.number_input("Seed", 0, 999, 42, key="ppo2_sd")
        if st.button("▶️ Train PPO vs Actor-Critic", type="primary", key="btn_ppo2"):
            with st.spinner("Training PPO…"):
                rw_ppo2 = train_ppo(n_it, 512, 4, 3e-3, 5e-3, 0.99, gae_l, clip_e, 0.01, int(sd_ppo))
            with st.spinner("AC for comparison…"):
                rw_ac_c, _ = train_ac(min(300, len(rw_ppo2)+50), 0.005, 0.01, 0.99, int(sd_ppo))
            st.session_state["ppo2_res"] = (rw_ppo2, rw_ac_c)
        if "ppo2_res" in st.session_state:
            rw_ppo2, rw_ac_c = st.session_state["ppo2_res"]
            fig_ppo2, ax_ppo2 = _fig(1, 1, 13, 4.5)
            for rw, nm, col in [(rw_ac_c,"Actor-Critic (1-step)",ALG["AC"]),
                                  (rw_ppo2,"PPO (clipped + GAE)",ALG["PPO"])]:
                sm = smooth(rw, 12)
                ax_ppo2.plot(rw, color=col, alpha=0.12, lw=0.5)
                ax_ppo2.plot(range(len(sm)), sm, color=col, lw=2.5, label=f"{nm} (late={np.mean(rw[-30:]):.1f})")
            ax_ppo2.axhline(195, color="white", ls="--", lw=1, alpha=0.5, label="Solved=195")
            ax_ppo2.set_xlabel("Episode",color="white"); ax_ppo2.set_ylabel("Reward",color="white")
            ax_ppo2.set_title(f"Actor-Critic vs PPO (ε={clip_e}, λ={gae_l})", color="white", fontweight="bold")
            ax_ppo2.legend(facecolor=CARD, labelcolor="white", fontsize=9); ax_ppo2.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_ppo2); plt.close()

    # ══════════════════════════════════════════════════════════════════
    # TAB 5 — TRPO & SAC
    # ══════════════════════════════════════════════════════════════════
        render_ac_notes("PPO", "actor_critic_policy_gradient_ppo")

    with tab_trpo:
        st.subheader("🏛️ TRPO & SAC — Advanced Policy Methods")
        col_t, col_s = st.columns(2)
        with col_t:
            st.markdown("#### 🏛️ TRPO (Schulman et al. 2015)")
            st.markdown(_card("#ad1457","🏛️","What TRPO is and why it matters",
                """TRPO (Trust Region Policy Optimization) is the theoretical predecessor to PPO.
                It proved that constraining the KL divergence between old and new policies
                guarantees monotonic improvement — the new policy is provably never worse.
                TRPO solves a constrained optimisation at each step using conjugate gradient
                descent and backtracking line search to find the largest safe step within the
                trust region. While PPO has replaced TRPO in practice, understanding TRPO's
                monotonic improvement theorem explains WHY PPO's clipping works and provides
                the theoretical foundation for all trust-region methods."""), unsafe_allow_html=True)

            st.markdown("**TRPO constrained optimisation:**")
            st.latex(r"\max_\theta\;L^{\text{PG}}(\theta) \quad\text{s.t.}\quad \overline{D}_\text{KL}(\pi_\theta\|\pi_\text{old})\leq\delta")
            st.markdown(_proof_box("Monotonic Improvement Theorem",
                """Define surrogate L^π(π') = J(π) + E_{s∼d^π}[E_{a∼π'}[A^π(s,a)]]<br>
                Then: J(π') ≥ L^π(π') - C·max_s D_KL(π'||π)<br>
                where C = 4γε/(1-γ)², ε = max|A^π(s,a)|<br><br>
                If D_KL is bounded by δ, then J(π') is bounded below by L^π(π') - C·δ.<br>
                Maximising L^π(π') while keeping D_KL ≤ δ guarantees improvement in J(π').<br>
                This is the theoretical justification for trust regions — and for PPO's clip."""),
                unsafe_allow_html=True)

            st.latex(r"J(\pi_\text{new}) \geq J(\pi_\text{old}) - \frac{4\gamma\varepsilon}{(1-\gamma)^2}\overline{D}_\text{KL}(\pi_\text{new}\|\pi_\text{old})")

        with col_s:
            st.markdown("#### 🌡️ SAC (Haarnoja et al. 2018)")
            st.markdown(_card("#558b2f","🌡️","What SAC is and when to use it over PPO",
                """Soft Actor-Critic introduces the maximum entropy RL framework: simultaneously
                maximise cumulative reward AND policy entropy. This encourages exploration throughout
                training, prevents premature convergence, and produces robust policies. SAC is
                off-policy (replay buffer) and designed for continuous action spaces.
                It achieves 5–10× better sample efficiency than PPO on MuJoCo benchmarks.
                Choose SAC for continuous actions with expensive data collection (real robots).
                Choose PPO for discrete actions, fast simulation, or LLM alignment."""), unsafe_allow_html=True)

            st.markdown("**SAC maximum entropy objective:**")
            st.latex(r"J(\pi) = \mathbb{E}_\tau\!\left[\sum_t\gamma^t\bigl(r_t + \alpha H(\pi(\cdot|s_t))\bigr)\right]")
            st.markdown(r"""
            $H(\pi) = -\mathbb{E}_{a\sim\pi}[\log\pi(a|s)]$ — entropy of the policy.
            $\alpha$ — temperature (balance reward vs entropy; can be learned automatically).
            """)
            st.markdown("**SAC twin-Q Bellman target:**")
            st.latex(r"y = r + \gamma\bigl(\min_{j=1,2}Q_{\bar\theta_j}(s',a') - \alpha\log\pi(a'|s')\bigr)")
            st.markdown(r"""
            The entropy term $-\alpha\log\pi(a'|s')$ is built into the Bellman backup itself —
            the soft value function naturally encourages diverse action selection at every state.
            """)

        st.divider()
        st.subheader("📊 Method Summary — When to Use Which")
        st.dataframe(pd.DataFrame({
            "Criterion":["Action space","Data source","Sample efficiency","Stability","Key strength","Avoid when"],
            "PPO":["Discrete+Continuous","On-policy rollout","Medium","Very high",
                   "Simplicity, RLHF, fast sim","Real robots with expensive data"],
            "TRPO":["Discrete+Continuous","On-policy rollout","Medium","Guaranteed monotonic",
                    "Hard safety guarantees","Anywhere PPO is simpler"],
            "SAC":["Continuous ONLY","Off-policy replay","Very high","Very high",
                   "Real robots, best sample eff","Discrete actions, RLHF"],
        }), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 6 — DASHBOARD
    # ══════════════════════════════════════════════════════════════════
        render_ac_notes("TRPO / SAC", "actor_critic_policy_gradient_trpo_sac")

    with tab_dash:
        st.subheader("📈 Full Policy Gradient Family Benchmark — CartPole")
        st.markdown("Run all methods simultaneously and compare learning curves, convergence speed, and stability.")
        c1,c2 = st.columns(2)
        n_ep_d = c1.slider("Episodes (MC methods)", 50, 300, 150, 25, key="acd2_ep")
        sd_d   = c2.number_input("Seed", 0, 999, 42, key="acd2_sd")
        if st.button("🚀 Run All Policy Gradient Methods", type="primary", key="btn_acd2"):
            results = {}
            for nm, fn, args in [
                ("REINFORCE (no baseline)", train_reinforce, (n_ep_d,0.01,0.99,False,int(sd_d))),
                ("REINFORCE + baseline",   train_reinforce, (n_ep_d,0.01,0.99,True, int(sd_d))),
                ("Actor-Critic (1-step)",  train_ac,        (n_ep_d,0.005,0.01,0.99,int(sd_d))),
                ("A2C (n=5)",              train_a2c,       (n_ep_d,0.003,0.01,0.99,5,0.01,int(sd_d))),
            ]:
                with st.spinner(f"Training {nm}…"):
                    r = fn(*args); results[nm] = r[0] if isinstance(r,tuple) else r
            with st.spinner("PPO…"):
                results["PPO (clipped+GAE)"] = train_ppo(30,512,4,3e-3,5e-3,0.99,0.95,0.2,0.01,int(sd_d))
            st.session_state["acd2_res"] = results

        if "acd2_res" in st.session_state:
            res = st.session_state["acd2_res"]
            pal = [ALG["REINFORCE"],"#b39ddb",ALG["AC"],ALG["A2C"],ALG["PPO"]]
            fig_d, axes_d = _fig(1, 2, 16, 5)
            for (nm, rw), col in zip(res.items(), pal):
                sm = smooth(rw, 12)
                axes_d[0].plot(rw, color=col, alpha=0.1, lw=0.4)
                axes_d[0].plot(range(len(sm)), sm, color=col, lw=2.5, label=f"{nm}")
            axes_d[0].axhline(195, color="white", ls="--", lw=1, alpha=0.5, label="Solved=195")
            axes_d[0].set_xlabel("Episode",color="white"); axes_d[0].set_ylabel("Reward",color="white")
            axes_d[0].set_title("Learning Curves — Each Method's Progression",color="white",fontweight="bold")
            axes_d[0].legend(facecolor=CARD, labelcolor="white", fontsize=7.5, bbox_to_anchor=(1.01,1))
            axes_d[0].grid(alpha=0.12)
            # Bar chart summary
            names  = list(res.keys())
            late   = [np.mean(rw[-30:]) for rw in res.values()]
            colors = pal[:len(names)]
            axes_d[1].barh(names, late, color=colors, alpha=0.85)
            axes_d[1].axvline(195, color="#4caf50", ls="--", lw=1.5, label="Solved=195")
            for i, v in enumerate(late):
                axes_d[1].text(v+1, i, f"{v:.1f}", va="center", color="white", fontsize=8)
            axes_d[1].set_xlabel("Mean reward (last 30 episodes)",color="white")
            axes_d[1].set_title("Final Performance Comparison",color="white",fontweight="bold")
            axes_d[1].legend(facecolor=CARD, labelcolor="white", fontsize=8)
            axes_d[1].grid(alpha=0.12, axis="x")
            plt.tight_layout(); st.pyplot(fig_d); plt.close()

            rows = []
            for (nm, rw), col in zip(res.items(), pal):
                solved = next((i for i,r in enumerate(smooth(rw,5)) if r>=150), None)
                rows.append({"Method":nm, "Late mean":f"{np.mean(rw[-30:]):.1f}",
                             "Best episode":f"{max(rw):.0f}",
                             "Variance":f"{np.var(rw[-30:]):.1f}",
                             "Steps to 150":f"{solved}" if solved else "never"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════
    # TAB 7 — STUDY PLAN
    # ══════════════════════════════════════════════════════════════════
        render_ac_notes("Dashboard", "actor_critic_policy_gradient_dashboard")

    with tab_pg_nutshell:
        render_policy_gradient_nutshell_html()

    # ══════════════════════════════════════════════════════════════════
    # TAB — PROJECT: CleanRL PPO
    # ══════════════════════════════════════════════════════════════════
    with tab_project:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0a1a2e,#0d2137,#0a1a0e);'
            'border:1px solid #1e3a5f;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
            '<h2 style="color:white;margin:0;font-size:1.9rem">🚀 CleanRL — Build a Production PPO Agent</h2>'
            '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.7">'
            'Your capstone project for this section: implement, run, and benchmark a complete PPO agent '
            'using <b style="color:#42a5f5">CleanRL</b> — the gold-standard single-file deep RL library. '
            'Every algorithm is one self-contained Python file with zero hidden abstractions, '
            'making it the perfect codebase to learn from and extend.'
            '</p>'
            '<p style="margin-top:.8rem">'
            '<a href="https://github.com/vwxyzjn/cleanrl" target="_blank" '
            'style="background:#1e88e5;color:white;padding:.4rem 1rem;border-radius:8px;'
            'text-decoration:none;font-weight:700;font-size:.95rem">📦 github.com/vwxyzjn/cleanrl</a>'
            '</p>'
            '</div>', unsafe_allow_html=True)

        # ── What methods does this project cover? ──────────────────
        st.subheader("🎯 Methods Covered in This Project")
        cols_m = st.columns(4)
        methods_info = [
            ("#7c4dff", "🎲", "REINFORCE", "Vanilla policy gradient — the baseline you'll beat"),
            ("#0288d1", "🎭", "Actor-Critic", "TD bootstrap replaces slow Monte Carlo returns"),
            ("#e65100", "🤝", "A2C (n-step)", "Advantage estimation with n-step TD targets"),
            ("#2e7d32", "🔐", "PPO (CleanRL)", "Clipped surrogate + GAE — the full production algorithm"),
        ]
        for col, (color, icon, name, desc) in zip(cols_m, methods_info):
            col.markdown(
                f'<div style="background:{color}18;border:1px solid {color}44;border-radius:10px;'
                f'padding:1rem;text-align:center">'
                f'<div style="font-size:2rem">{icon}</div>'
                f'<b style="color:{color};font-size:1rem">{name}</b>'
                f'<p style="color:#9e9ebb;font-size:.8rem;margin-top:.4rem;line-height:1.5">{desc}</p>'
                f'</div>', unsafe_allow_html=True)

        st.divider()

        # ── Project overview ───────────────────────────────────────
        st.subheader("📋 Project Overview")
        st.markdown(
            '<div style="background:#12121f;border:1px solid #1e3a5f;border-radius:12px;'
            'padding:1.2rem 1.6rem;margin-bottom:1rem">'
            '<b style="color:#42a5f5;font-size:1.05rem">What you will build</b>'
            '<p style="color:#c5c5e0;margin-top:.5rem;line-height:1.7">'
            'You will clone CleanRL, run its PPO implementation on CartPole and LunarLander, '
            'understand every line of <code style="color:#80cbc4">ppo.py</code> (≈ 300 lines, zero magic), '
            'instrument it with Weights &amp; Biases for experiment tracking, and produce a '
            'benchmark report comparing REINFORCE, A2C, and PPO on the same environment.'
            '</p>'
            '<b style="color:#42a5f5;font-size:1.05rem">Why CleanRL?</b>'
            '<ul style="color:#c5c5e0;margin-top:.5rem;line-height:1.7">'
            '<li>Single-file algorithms — you see <i>everything</i>, no base classes hiding logic</li>'
            '<li>All 9 implementation tricks from Schulman 2017 Appendix A are included</li>'
            '<li>Built-in W&amp;B + TensorBoard tracking with one flag</li>'
            '<li>Benchmarked and cited in dozens of academic papers as a reproducible baseline</li>'
            '<li>Supports Atari, MuJoCo, continuous control, and RLHF workflows</li>'
            '</ul>'
            '</div>', unsafe_allow_html=True)

        st.divider()

        # ── Step-by-step implementation guide ─────────────────────
        st.subheader("🛠️ Step-by-Step Implementation Guide")

        steps = [
            ("1", "#1565c0", "⚙️ Environment Setup",
             "Create an isolated Python environment to avoid dependency conflicts.",
             [
                 ("bash", "# Create and activate a virtual environment\npython -m venv cleanrl-env\nsource cleanrl-env/bin/activate  # Windows: cleanrl-env\\Scripts\\activate\n\n# Verify Python version (need 3.8+)\npython --version"),
             ]),
            ("2", "#1565c0", "📦 Clone & Install CleanRL",
             "Clone the repository and install dependencies for your target environment.",
             [
                 ("bash", "# Clone the repo\ngit clone https://github.com/vwxyzjn/cleanrl.git\ncd cleanrl\n\n# Install base dependencies\npip install -r requirements/requirements.txt\n\n# For classic control (CartPole, LunarLander) — already included above\n# For Atari games:\npip install -r requirements/requirements-atari.txt\n\n# For MuJoCo (HalfCheetah, Hopper, Ant):\npip install -r requirements/requirements-mujoco.txt"),
             ]),
            ("3", "#e65100", "🔬 Read ppo.py Before Running",
             "Open the file and read every section. It is ~300 lines with no abstractions.",
             [
                 ("bash", "# Open the core PPO file\ncat cleanrl/ppo.py\n# Or in your editor:\ncode cleanrl/ppo.py"),
                 ("text", "Key sections to understand in ppo.py:\n  • Lines 1–60   : Argument parsing (all hyperparameters exposed)\n  • Lines 60–100 : Environment setup with SyncVectorEnv\n  • Lines 100–130: Actor-Critic network (shared trunk, two heads)\n  • Lines 130–180: Rollout collection loop (GAE computed inline)\n  • Lines 180–240: PPO update loop (K epochs, mini-batches, clip)\n  • Lines 240–300: Logging, W&B, video recording"),
             ]),
            ("4", "#2e7d32", "🎮 Run PPO on CartPole",
             "Your first run — watch it solve CartPole in minutes.",
             [
                 ("bash", "# Basic run (TensorBoard logging by default)\npython cleanrl/ppo.py --env-id CartPole-v1\n\n# View TensorBoard\ntensorboard --logdir runs/\n# Open http://localhost:6006 in your browser"),
             ]),
            ("5", "#2e7d32", "📊 Run with Weights & Biases Tracking",
             "Professional experiment tracking — log hyperparameters, curves, and videos automatically.",
             [
                 ("bash", "# Install W&B if not already installed\npip install wandb\nwandb login  # paste your API key from wandb.ai\n\n# Run PPO with full tracking\npython cleanrl/ppo.py \\\n    --env-id CartPole-v1 \\\n    --track \\\n    --wandb-project-name my-pg-project \\\n    --wandb-entity YOUR_USERNAME \\\n    --capture-video   # records episode videos"),
             ]),
            ("6", "#7b1fa2", "📈 Benchmark: Compare All Methods",
             "Run REINFORCE, A2C, and PPO on the same env with the same seed. Produce a comparison table.",
             [
                 ("bash", "# Run each algorithm (same seed for fair comparison)\npython cleanrl/ppo.py \\\n    --env-id LunarLander-v2 --seed 1 --track \\\n    --wandb-project-name pg-benchmark\n\n# CleanRL also has a simple REINFORCE-style baseline:\n# Look at cleanrl/ppo.py — set num_minibatches=1, update_epochs=1,\n# clip_coef=1.0 to approximate REINFORCE with baseline"),
                 ("bash", "# Run 5 seeds for statistical significance (professional practice)\nfor seed in 1 2 3 4 5; do\n    python cleanrl/ppo.py \\\n        --env-id LunarLander-v2 \\\n        --seed $seed \\\n        --track \\\n        --wandb-project-name pg-benchmark-5seeds\ndone\n# W&B will show mean ± std across seeds automatically"),
             ]),
            ("7", "#ad1457", "🧪 Key Hyperparameter Experiments",
             "Ablate each PPO component to understand its contribution.",
             [
                 ("bash", "# Ablation 1: Remove clipping (set clip_coef very high)\npython cleanrl/ppo.py --env-id CartPole-v1 --clip-coef 10.0 --track \\\n    --wandb-project-name pg-ablation\n\n# Ablation 2: Remove entropy bonus\npython cleanrl/ppo.py --env-id CartPole-v1 --ent-coef 0.0 --track \\\n    --wandb-project-name pg-ablation\n\n# Ablation 3: Single epoch (like A2C)\npython cleanrl/ppo.py --env-id CartPole-v1 --update-epochs 1 --track \\\n    --wandb-project-name pg-ablation\n\n# Ablation 4: No GAE (lambda=0, equivalent to 1-step TD)\npython cleanrl/ppo.py --env-id CartPole-v1 --gae-lambda 0.0 --track \\\n    --wandb-project-name pg-ablation"),
             ]),
            ("8", "#558b2f", "🤖 Scale to Continuous Control (MuJoCo)",
             "Graduate from CartPole to real continuous control — the regime where PPO dominates.",
             [
                 ("bash", "# Install MuJoCo dependencies first\npip install -r requirements/requirements-mujoco.txt\n\n# Run PPO on HalfCheetah (continuous actions)\npython cleanrl/ppo_continuous_action.py \\\n    --env-id HalfCheetah-v4 \\\n    --total-timesteps 2000000 \\\n    --track \\\n    --wandb-project-name mujoco-ppo\n\n# Compare with SAC (off-policy — much more sample efficient)\npython cleanrl/sac_continuous_action.py \\\n    --env-id HalfCheetah-v4 \\\n    --total-timesteps 1000000 \\\n    --track \\\n    --wandb-project-name mujoco-sac"),
             ]),
        ]

        for step_num, color, title, description, code_blocks in steps:
            st.markdown(
                f'<div style="background:{color}10;border-left:4px solid {color};'
                f'border-radius:0 10px 10px 0;padding:.8rem 1.2rem;margin:.6rem 0 .3rem">'
                f'<b style="color:{color};font-size:1.05rem">Step {step_num}: {title}</b>'
                f'<p style="color:#c5c5e0;font-size:.9rem;margin:.3rem 0 0;line-height:1.6">{description}</p>'
                f'</div>', unsafe_allow_html=True)
            for code_type, code_content in code_blocks:
                if code_type == "bash":
                    st.code(code_content, language="bash")
                else:
                    st.code(code_content, language="text")

        st.divider()

        # ── Key hyperparameters table ──────────────────────────────
        st.subheader("⚙️ CleanRL PPO: Key Hyperparameters")
        st.markdown('<p style="color:#9e9ebb;font-size:.9rem">All exposed via CLI flags — no config files needed.</p>',
                    unsafe_allow_html=True)
        import pandas as pd
        st.dataframe(pd.DataFrame({
            "Flag": ["--learning-rate", "--num-envs", "--num-steps", "--update-epochs",
                     "--clip-coef", "--ent-coef", "--gae-lambda", "--vf-coef", "--max-grad-norm"],
            "Default": ["2.5e-4", "4", "128", "4", "0.2", "0.01", "0.95", "0.5", "0.5"],
            "What it controls": [
                "Adam LR for both actor and critic",
                "Number of parallel environments",
                "Rollout length before each update",
                "Number of epochs to reuse each batch",
                "PPO clip ε — how far the policy can move",
                "Entropy bonus weight — prevents collapse",
                "GAE λ — bias-variance tradeoff (0=TD, 1=MC)",
                "Value loss weight in total loss",
                "Gradient clipping threshold",
            ],
            "Schulman checklist": ["✅ #7", "✅ #1", "✅ #2", "✅ #6", "✅ #5", "✅ #9", "✅ #4", "✅ #8", "✅ #3"],
        }), use_container_width=True, hide_index=True)

        st.divider()

        # ── Pro tips ───────────────────────────────────────────────
        st.subheader("💡 Pro Tips for Running CleanRL Professionally")
        pro_tips = [
            ("Always run ≥ 5 seeds", "A single seed result is meaningless. Use a shell loop or Python script to launch 5 seeds in parallel. W&B will compute mean ± std for you automatically."),
            ("Use EnvPool for 10× speed", "Install requirements-envpool.txt and use ppo_atari_envpool.py. EnvPool uses C++ to run environments and delivers ~3–4× more frames/second than standard gym."),
            ("Check the W&B public report", "CleanRL's benchmarks are public at wandb.ai/cleanrl/cleanrl.benchmark — compare your results against the official numbers to verify your setup is correct."),
            ("Read every argument in the code", "Run python cleanrl/ppo.py --help to see all flags. The argument parser IS the documentation. Every hyperparameter is commented with its paper reference."),
            ("Profile before optimising", "If training is slow, use torch.profiler to find bottlenecks. The rollout loop is usually the bottleneck, not the update — parallelize environments, not the model."),
            ("Use capture-video to debug", "Add --capture-video to record the agent's behaviour every N episodes. Watching the agent fail early is often faster than reading logs."),
        ]
        tip_cols = st.columns(2)
        for i, (tip_title, tip_body) in enumerate(pro_tips):
            tip_cols[i % 2].markdown(
                f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                f'padding:.8rem 1rem;margin:.3rem 0">'
                f'<b style="color:#42a5f5">💡 {tip_title}</b>'
                f'<br><span style="color:#9e9ebb;font-size:.86rem;line-height:1.6">{tip_body}</span>'
                f'</div>', unsafe_allow_html=True)

        st.divider()

        # ── Repo & resources ───────────────────────────────────────
        st.subheader("🔗 Repository & Key Resources")
        resources = [
            ("📦", "CleanRL GitHub", "github.com/vwxyzjn/cleanrl",
             "Main repo — clone this. Read ppo.py first.", "https://github.com/vwxyzjn/cleanrl"),
            ("📊", "CleanRL W&B Benchmarks", "wandb.ai/cleanrl/cleanrl.benchmark",
             "Public benchmark report — all algorithms on all envs.", "https://wandb.ai/cleanrl/cleanrl.benchmark"),
            ("📄", "CleanRL Paper (Huang et al. 2022)", "arXiv:2111.08819",
             "The paper describing the single-file design philosophy and benchmark methodology.", "https://arxiv.org/abs/2111.08819"),
            ("📄", "PPO Paper (Schulman 2017)", "arXiv:1707.06347",
             "The original PPO paper. Read Appendix A — the 9 implementation tricks CleanRL implements.", "https://arxiv.org/abs/1707.06347"),
            ("📄", "GAE Paper (Schulman 2015)", "arXiv:1506.02438",
             "Derives the Generalised Advantage Estimator used in CleanRL's PPO implementation.", "https://arxiv.org/abs/1506.02438"),
            ("🌐", "Weights & Biases", "wandb.ai",
             "Free experiment tracking. Required for --track flag. Sign up and get an API key.", "https://wandb.ai"),
        ]
        for icon, name, url_display, desc, url in resources:
            st.markdown(
                f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                f'padding:.6rem 1rem;margin:.3rem 0">'
                f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {name}</a>'
                f' <span style="color:#4a4a6a;font-size:.82rem">— {url_display}</span>'
                f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span>'
                f'</div>', unsafe_allow_html=True)

    with tab_plan:
        st.subheader("📚 4-Week Policy Gradient Mastery Plan")

        weeks = [
            ("Week 1","Foundations — From Zero to REINFORCE","#7c4dff",[
                ("📄","Sutton & Barto Ch.13","Read Sections 13.1–13.3. Derive the policy gradient theorem yourself on paper without looking at the book. This is the single most important exercise."),
                ("🧮","Prove the log-derivative trick","Show ∇f = f·∇log f. Then apply it to show ∇E[r(τ)] = E[∇log p_θ(τ)·r(τ)]."),
                ("🧮","Prove the baseline doesn't bias gradient","Show E[∇log π(a|s)·b(s)] = 0 using the identity ∇Σ_a π(a|s) = 0."),
                ("💻","Implement REINFORCE in NumPy","No PyTorch. Compute G_t backward. Apply the update. Run on CartPole. Reproduce the high-variance learning curve."),
                ("📊","Plot variance as a function of N","For N=10,50,100,500 episodes: plot std(gradient_estimate). Verify it scales as 1/√N."),
            ]),
            ("Week 2","Actor-Critic — Understanding TD","#0288d1",[
                ("📄","Sutton & Barto Ch.9 + Ch.13","Read 9.1–9.3 (function approximation) and 13.5 (actor-critic). Understand how TD(0) trains V(s)."),
                ("🧮","Derive δ_t as a 1-step advantage estimate","Show A(s,a) = Q(s,a) - V(s) = r + γV(s') - V(s) = δ_t using the Bellman equation."),
                ("💻","Implement 1-step Actor-Critic","Two separate networks. Track TD error over training. Verify δ → 0 as critic improves."),
                ("💻","Implement A2C with n-step returns","Try n=1,3,5,10,20. For each: run CartPole for 200 episodes, record late-mean reward. Plot vs n."),
                ("📊","Entropy collapse experiment","Train A2C with c₂=0 and c₂=0.01. Record the policy entropy over training. See why entropy=0 is catastrophic."),
            ]),
            ("Week 3","PPO — The Industry Standard","#e65100",[
                ("📄","Schulman et al. (2015) — GAE paper","Sections 1–4. Derive GAE = Σ(γλ)^k δ_{t+k} from the weighted n-step average. Verify λ=0 gives 1-step TD, λ=1 gives MC."),
                ("📄","Schulman et al. (2017) — PPO paper","Read all of it, especially Appendix A (implementation checklist with 9 items). Every item matters."),
                ("💻","Implement PPO from scratch","GAE backward pass, clipped objective, K=4 epochs, mini-batch SGD, value loss, entropy bonus, gradient clipping. All 9 checklist items."),
                ("📊","PPO ablation study","Remove each component one at a time. Measure the damage. Which component matters most? (Usually: clipping > GAE > entropy)"),
                ("💻","PPO on continuous control","LunarLanderContinuous-v2. Gaussian policy head. Clip actions to bounds."),
            ]),
            ("Week 4","Advanced & Real Applications","#558b2f",[
                ("📄","Schulman et al. (2015) — TRPO","Sections 3–5: the trust region bound, conjugate gradient, backtracking line search. You won't implement it but you need the theory."),
                ("📄","Haarnoja et al. (2018) — SAC","Sections 4–5: soft Bellman equations, automatic temperature tuning. Focus on how entropy enters the backup."),
                ("💻","SAC on MuJoCo HalfCheetah-v4","Implement with PyTorch. Compare sample efficiency vs PPO on the same task."),
                ("🎯","RLHF mini-project","Use TRL library. Load GPT-2. Train a reward model on sentiment. Apply PPO. Watch the model generate positive text."),
                ("📊","Full benchmark","Compare REINFORCE, AC, A2C, PPO on CartPole and LunarLander. Report mean±std over 5 seeds. This is a publishable ablation."),
            ]),
        ]

        for wt, ws, col, tasks in weeks:
            st.markdown(f'<div style="background:{col}18;border-left:4px solid {col};'
                        f'border-radius:0 10px 10px 0;padding:.7rem 1.2rem;margin:.8rem 0 .4rem">'
                        f'<b style="color:{col};font-size:1.05rem">{wt}: {ws}</b></div>',
                        unsafe_allow_html=True)
            for icon, title, desc in tasks:
                st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;'
                            f'border-radius:8px;padding:.6rem 1rem;margin:.3rem 0;'
                            f'display:flex;gap:.8rem"><span style="font-size:1.2rem">{icon}</span>'
                            f'<div><b style="color:white">{title}</b>'
                            f'<br><span style="color:#9e9ebb;font-size:.86rem;line-height:1.6">{desc}</span>'
                            f'</div></div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📖 Primary References")
        for icon, title, desc, url in [
            ("📗","Sutton & Barto Ch.13","Policy gradient theorem, REINFORCE, actor-critic. The mathematical foundation.","http://incompleteideas.net/book/the-book.html"),
            ("📄","Schulman 2015 — GAE","Derives GAE from weighted n-step returns. The maths behind PPO's advantage estimation.","https://arxiv.org/abs/1506.02438"),
            ("📄","Schulman 2017 — PPO","Short, readable, includes 9-item implementation checklist. Read Sections 3 and 5.","https://arxiv.org/abs/1707.06347"),
            ("📄","Mnih 2016 — A3C","Asynchronous advantage actor-critic. The original parallel RL paper.","https://arxiv.org/abs/1602.01783"),
            ("📄","Haarnoja 2018 — SAC","Maximum entropy RL. Soft Bellman equations and automatic temperature.","https://arxiv.org/abs/1801.01290"),
            ("🎥","CS285 Lectures 5–7","Sergey Levine's Berkeley Deep RL course. Best video explanation of policy gradients.","https://rail.eecs.berkeley.edu/deeprlcourse/"),
            ("💻","CleanRL PPO","150-line single-file PPO implementation with all 9 Schulman checklist items.","https://github.com/vwxyzjn/cleanrl"),
            ("💻","TRL — RLHF with PPO","Hugging Face library for RLHF. PPO on language models in ~20 lines.","https://github.com/huggingface/trl"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)

        render_ac_notes("Learning Plan", "actor_critic_policy_gradient_learning_plan")
