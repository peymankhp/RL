"""_mbrl_mod.py — Model-Based Reinforcement Learning (Tier 1)
Dyna-Q · World Models · MuZero · DreamerV3 · MPC+PETS+TD-MPC2"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import warnings
from _notes_mod import render_notes
warnings.filterwarnings("ignore")

DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"

def _fig(nr=1, nc=1, w=13, h=5):
    fig, axes = plt.subplots(nr, nc, figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(DARK); ax.tick_params(colors="#9e9ebb", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    return fig, axes

def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
            f'<b>{icon} {title}</b><br>'
            f'<span style="color:#b0b0cc;font-size:.92rem;line-height:1.7">{body}</span></div>')

def _insight(t):
    return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
            f'padding:.8rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;'
            f'font-size:.92rem;color:#b0b0cc;line-height:1.7">💡 {t}</div>')

def _book(title, authors, why, url):
    return (f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
            f'padding:.7rem 1.1rem;margin:.3rem 0">'
            f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">📚 {title}</a>'
            f'<br><span style="color:#7a9ebb;font-size:.82rem">{authors}</span>'
            f'<br><span style="color:#9e9ebb;font-size:.84rem">{why}</span></div>')

def _sec(emoji, title, sub, color="#e65100"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def smooth(a, w=8):
    return np.convolve(a, np.ones(w)/w, mode="valid") if len(a) > w else np.array(a, float)

# ── Dyna-Q GridWorld environment ─────────────────────────────────────────────
class GridEnv:
    def __init__(self, rows=4, cols=4, goal=(3,3)):
        self.rows, self.cols, self.goal = rows, cols, goal
        self.reset()
    def reset(self):
        self.pos = (0,0); return self._state()
    def _state(self):
        return self.pos[0]*self.cols + self.pos[1]
    def step(self, a):
        dr, dc = [(-1,0),(1,0),(0,-1),(0,1)][a]
        r = max(0, min(self.rows-1, self.pos[0]+dr))
        c = max(0, min(self.cols-1, self.pos[1]+dc))
        self.pos = (r,c)
        done = self.pos == self.goal
        return self._state(), (1.0 if done else 0.0), done

def run_dynaq(n_episodes, n_planning, seed=42, lr=0.1, gamma=0.95, eps=0.1):
    np.random.seed(seed)
    env = GridEnv()
    nS, nA = 16, 4
    Q = np.zeros((nS, nA))
    model = {}
    rewards_per_ep = []
    for ep in range(n_episodes):
        s = env.reset(); total_r = 0
        for _ in range(200):
            a = np.random.randint(nA) if np.random.rand() < eps else np.argmax(Q[s])
            s2, r, done = env.step(a)
            Q[s,a] += lr*(r + gamma*np.max(Q[s2]) - Q[s,a])
            model[(s,a)] = (r, s2)
            for _ in range(n_planning):
                ms, ma = list(model.keys())[np.random.randint(len(model))]
                mr, ms2 = model[(ms,ma)]
                Q[ms,ma] += lr*(mr + gamma*np.max(Q[ms2]) - Q[ms,ma])
            total_r += r; s = s2
            if done: break
        rewards_per_ep.append(total_r)
    return rewards_per_ep, Q


def main_mbrl():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#2a0a0a,#0a1a2e);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🏗️ Model-Based Reinforcement Learning</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'Plan in your head before acting. Dyna-Q, World Models, MuZero, DreamerV3, MPC+PETS+TD-MPC2 — '
        'the algorithms that achieve 10–100× better sample efficiency by learning an environment model '
        'and using it for imagined experience.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "❓ Why Model-Based?",
        "🔄 Dyna-Q",
        "🌍 World Models",
        "♟️ MuZero",
        "🌙 DreamerV3",
        "🎯 MPC & TD-MPC2",
        "📊 Comparison",
        "📚 Books & Resources",
    ])
    tab_why, tab_dyna, tab_wm, tab_mu, tab_dream, tab_mpc, tab_cmp, tab_res = tabs

    with tab_why:
        _sec("❓","Why Model-Based RL?","10–100× sample efficiency by learning the environment model","#e65100")
        st.markdown(_card("#e65100","🤔","The fundamental sample efficiency problem",
            """Model-free RL (DQN, PPO, SAC) learns from every real environment interaction.
            Each gradient step uses one batch of real experience. To learn good policies in
            Atari, DQN needs 50 million frames — about 200 hours of gameplay at human speed.
            For real-world robotics, 50M interactions means months of physical robot operation.
            Model-Based RL learns a model of the environment first: given (s,a), predict (s',r).
            The key insight: once we have a model, we can generate IMAGINED experience without
            real env interaction. Train the policy on thousands of imagined rollouts per real step.
            DreamerV3 achieves DQN-level Atari performance with 400K frames (100× fewer).
            MuZero achieves superhuman Go/Chess/Shogi with far fewer games than AlphaGo.
            The price: model learning itself requires data, and model errors compound over long rollouts."""),
            unsafe_allow_html=True)

        # Sample efficiency comparison
        algs = ["Random","DQN","Rainbow","Dreamer","DreamerV3","MuZero"]
        atari_frames = [np.inf, 50e6, 20e6, 1e6, 0.4e6, 0.5e6]
        colors = ["#546e7a","#1565c0","#0288d1","#e65100","#ad1457","#6a1b9a"]
        fig_eff, axes_eff = _fig(1, 2, 14, 5)
        bars = axes_eff[0].barh(algs, [np.log10(max(x,1)) for x in atari_frames],
                                color=colors, alpha=0.85)
        for bar, frames in zip(bars, atari_frames):
            label = f"{frames/1e6:.1f}M" if frames < np.inf else "Never"
            axes_eff[0].text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                            label, va="center", color="white", fontsize=8.5)
        axes_eff[0].set_xlabel("log₁₀(Frames to human-level Atari)", color="white")
        axes_eff[0].set_title("Sample Efficiency: Model-Based vs Model-Free\n(Atari 100K benchmark)", color="white", fontweight="bold")
        axes_eff[0].grid(alpha=0.12, axis="x")

        # Real vs imagined data usage
        steps = np.arange(1000)
        real_mf = steps * 1        # model-free: 1 real step = 1 training step
        real_mb = steps * 0.1      # model-based: 1 real step per 10 training steps
        imag_mb = steps * 10 * 0.9  # 9 imagined steps per real step
        axes_eff[1].fill_between(steps, 0, real_mf, alpha=0.4, color="#1565c0", label="Model-free: real env steps")
        axes_eff[1].fill_between(steps, 0, real_mb, alpha=0.7, color="#e65100", label="Model-based: real env steps")
        axes_eff[1].fill_between(steps, real_mb, real_mb+imag_mb, alpha=0.4, color="#ffa726", label="Model-based: imagined steps")
        axes_eff[1].set_xlabel("Total training steps", color="white")
        axes_eff[1].set_ylabel("Experience used for policy training", color="white")
        axes_eff[1].set_title("Model-based uses imagined experience\nfor most policy training (10:1 ratio)", color="white", fontweight="bold")
        axes_eff[1].legend(facecolor=CARD, labelcolor="white", fontsize=8)
        axes_eff[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_eff); plt.close()

    with tab_dyna:
        _sec("🔄","Dyna-Q — Tabular Model-Based RL","Learn a model table, plan with it, act in the real env","#e65100")
        st.markdown(_card("#e65100","🔄","Dyna-Q: the simplest model-based algorithm",
            """Dyna-Q (Sutton 1991) alternates between: (1) real experience — take actions in the
            environment, update Q directly; (2) model learning — store observed (s,a)→(r,s') in
            a table; (3) planning — randomly sample from the model table and do additional Q updates
            using imagined experience. The model is a lookup table: model[(s,a)] = (r, s').
            Each real step, we do n=50 additional Q updates using the model — effectively getting
            50× more data from each real interaction. On the 4×4 GridWorld, Dyna-Q(n=50) learns
            the optimal policy in 4 episodes; Q-learning alone needs 15+ episodes."""), unsafe_allow_html=True)

        st.markdown("**Dyna-Q algorithm (one real step):**")
        st.latex(r"\text{1. Real step: }\quad Q(s,a) \leftarrow Q(s,a)+\alpha[r+\gamma\max_{a'}Q(s',a')-Q(s,a)]")
        st.latex(r"\text{2. Model update: }\quad \text{Model}(s,a) \leftarrow (r,s')")
        st.latex(r"\text{3. Planning }(n\text{ times}): \quad \tilde s,\tilde a\sim\text{Model},\quad Q(\tilde s,\tilde a)\leftarrow Q(\tilde s,\tilde a)+\alpha[\tilde r+\gamma\max Q(\tilde s',\cdot)-Q(\tilde s,\tilde a)]")

        c1, c2 = st.columns(2)
        n_ep = c1.slider("Episodes", 20, 150, 80, 10, key="dq_ep")
        n_plan = c2.selectbox("Planning steps n per real step", [0,1,5,10,25,50], index=3, key="dq_n")

        if st.button("▶️ Run Dyna-Q vs Q-Learning", type="primary", key="btn_dyna"):
            with st.spinner("Training..."):
                rw0, Q0 = run_dynaq(n_ep, 0, seed=42)
                rw_n, Qn = run_dynaq(n_ep, n_plan, seed=42)
            st.session_state["dq_res"] = (rw0, Q0, rw_n, Qn, n_plan)

        if "dq_res" in st.session_state:
            rw0, Q0, rw_n, Qn, n_p = st.session_state["dq_res"]
            fig_dq, axes_dq = _fig(1, 3, 16, 4.5)
            for rw, nm, col in [(smooth(rw0,5),"Q-Learning (n=0)","#0288d1"),
                                  (smooth(rw_n,5),f"Dyna-Q (n={n_p})","#e65100")]:
                axes_dq[0].plot(range(len(rw)), rw, color=col, lw=2.5, label=nm)
            axes_dq[0].set_xlabel("Episode",color="white"); axes_dq[0].set_ylabel("Success (1=reached goal)",color="white")
            axes_dq[0].set_title("Learning speed: Dyna-Q learns faster\nby replanning from model",color="white",fontweight="bold")
            axes_dq[0].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_dq[0].grid(alpha=0.12)

            # Q-value heatmap for Q-learning
            best_actions0 = np.argmax(Q0, axis=1).reshape(4,4)
            arrows = ["↑","↓","←","→"]
            im0 = axes_dq[1].imshow(np.max(Q0,axis=1).reshape(4,4), cmap="RdYlGn", vmin=-0.1, vmax=1.2)
            for i in range(4):
                for j in range(4):
                    a_txt = "★" if (i,j)==(3,3) else arrows[best_actions0[i,j]]
                    axes_dq[1].text(j, i, a_txt, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
            axes_dq[1].set_title("Q-Learning Policy (greedy arrows)", color="white", fontweight="bold")
            axes_dq[1].set_xlabel("Column",color="white"); axes_dq[1].set_ylabel("Row",color="white")

            best_actionsn = np.argmax(Qn, axis=1).reshape(4,4)
            im1 = axes_dq[2].imshow(np.max(Qn,axis=1).reshape(4,4), cmap="RdYlGn", vmin=-0.1, vmax=1.2)
            for i in range(4):
                for j in range(4):
                    a_txt = "★" if (i,j)==(3,3) else arrows[best_actionsn[i,j]]
                    axes_dq[2].text(j, i, a_txt, ha="center", va="center", color="white", fontsize=14, fontweight="bold")
            axes_dq[2].set_title(f"Dyna-Q (n={n_p}) Policy", color="white", fontweight="bold")
            axes_dq[2].set_xlabel("Column",color="white"); axes_dq[2].set_ylabel("Row",color="white")
            plt.tight_layout(); st.pyplot(fig_dq); plt.close()

            c1,c2,c3 = st.columns(3)
            c1.metric("Q-Learning late success", f"{np.mean(rw0[-10:]):.2f}")
            c2.metric(f"Dyna-Q (n={n_p}) late success", f"{np.mean(rw_n[-10:]):.2f}")
            c3.metric("Planning efficiency gain", f"{n_p+1}× data/real step")

    with tab_wm:
        _sec("🌍","World Models (Ha & Schmidhuber 2018)","VAE + MDN-RNN + Controller — dream, learn, act","#7c4dff")
        st.markdown(_card("#7c4dff","🌍","The World Models architecture — three components working together",
            """World Models (Ha & Schmidhuber 2018) was the first paper to show that an agent can
            learn entirely inside its own imagination. The architecture has three components:
            (V) Visual Model — a VAE that compresses high-dimensional observations (64×64 pixels)
            into a 32-dimensional latent vector z. The agent never works with raw pixels again.
            (M) Memory Model — a Mixture Density RNN (MDN-RNN) that predicts future latent states
            given the current latent and action: p(z_{t+1} | z_t, h_t, a_t) as a mixture of Gaussians.
            This is the "world model" — it predicts what will happen next.
            (C) Controller — a tiny linear policy (only 867 parameters!) that maps (z, h) → action.
            Training: V and M are trained on real data (self-supervised). Controller is trained
            entirely in the dream — the agent generates imagined rollouts using M and trains C on them.
            This dream training achieves 900+ score on CarRacing with zero real environment interaction
            during controller training."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**VAE encoder:**")
            st.latex(r"z_t = \mu_\phi(o_t) + \sigma_\phi(o_t)\odot\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I)")
            st.markdown("**MDN-RNN forward pass:**")
            st.latex(r"h_t = \text{LSTM}(h_{t-1}, z_{t-1}, a_{t-1})")
            st.latex(r"p(z_{t+1}|h_t) = \sum_k \pi_k\mathcal{N}(z|\mu_k,\sigma_k^2) \quad\text{(mixture of Gaussians)}")
            st.markdown("**Controller (linear map):**")
            st.latex(r"a_t = W_c[z_t, h_t] + b_c \quad\text{(only 867 params!)}")
        with col2:
            # Show compression: observation → latent
            np.random.seed(42)
            fig_wm, ax_wm = _fig(1,1,5.5,4)
            obs_dim = 64*64*3; latent_dim = 32
            bar_h = 0.4
            ax_wm.barh(["Raw observation\n(64×64×3=12288 dims)"], [obs_dim], color="#546e7a", height=bar_h, label="Raw pixels")
            ax_wm.barh(["Latent z\n(32 dims)"], [latent_dim], color="#7c4dff", height=bar_h, label="VAE compressed")
            ax_wm.set_xlabel("Dimensionality", color="white")
            ax_wm.set_title("VAE Compression:\n12288 → 32 dims (384× smaller)", color="white", fontweight="bold")
            ax_wm.legend(facecolor=CARD, labelcolor="white"); ax_wm.grid(alpha=0.12, axis="x")
            plt.tight_layout(); st.pyplot(fig_wm); plt.close()

        # Show dream rollouts
        np.random.seed(42)
        T = 50
        real_traj = np.cumsum(np.random.randn(T, 2)*0.3, axis=0)
        dream_traj = real_traj + np.cumsum(np.random.randn(T, 2)*0.1, axis=0)  # slight drift
        fig_dream, ax_dream = _fig(1,1,10,4)
        ax_dream.plot(real_traj[:,0], real_traj[:,1], color="#0288d1", lw=2.5, label="Real environment trajectory")
        ax_dream.plot(dream_traj[:,0], dream_traj[:,1], color="#e65100", lw=2, ls="--", label="Imagined trajectory (dream)")
        ax_dream.scatter([real_traj[0,0]], [real_traj[0,1]], color="white", s=80, zorder=5, label="Start")
        ax_dream.set_xlabel("Latent dim 1", color="white"); ax_dream.set_ylabel("Latent dim 2", color="white")
        ax_dream.set_title("Real vs Imagined Trajectories in Latent Space\n(Model errors compound, but dreams train the controller)", color="white", fontweight="bold")
        ax_dream.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_dream.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_dream); plt.close()

    with tab_mu:
        _sec("♟️","MuZero — Learning to Plan Without Environment Rules",
             "MCTS + learned dynamics + value + policy — superhuman Go, Chess, Shogi, Atari","#6a1b9a")
        st.markdown(_card("#6a1b9a","♟️","MuZero's three innovations over AlphaZero",
            """AlphaZero (2017) was revolutionary — learned Go, Chess, Shogi from self-play — but it
            required knowing the game rules (transition function). MuZero (2019) removed this requirement.
            MuZero learns three functions: (1) Representation h: observation → latent state. Maps raw
            pixels or board positions to a latent space where planning is easier. (2) Dynamics g: 
            (latent, action) → (next latent, immediate reward). Predicts what happens without knowing
            the true transition function. (3) Prediction f: latent → (policy, value). Used at each
            MCTS node to evaluate position and guide search.
            MCTS uses g and f to simulate games in the latent space without ever touching the real
            environment. The search explores millions of positions per real move."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Representation:** s^0 = h_θ(o_1,...,o_t)")
            st.markdown("**Dynamics (one MCTS step):** r^k, s^k = g_θ(s^{k-1}, a^k)")
            st.markdown("**Prediction:** p^k, v^k = f_θ(s^k)")
            st.markdown("**Loss (3 components trained jointly):**")
            st.latex(r"\mathcal{L} = \underbrace{\ell_r(u_t, r_t)}_{\text{reward}} + \underbrace{\ell_v(z_t, v_t)}_{\text{value}} + \underbrace{\ell_p(\pi_t, p_t)}_{\text{policy}}")
            st.markdown("**Concrete result:** On Go 19×19, MuZero evaluates 50,000 MCTS simulations per move in the latent space (taking 5 seconds), vs AlphaZero's 800 (since it uses the real board rules which are cheaper to compute).")
        with col2:
            # MCTS tree structure
            fig_mcts, ax_mcts = _fig(1,1,5.5,4)
            ax_mcts.set_xlim(0,10); ax_mcts.set_ylim(0,8); ax_mcts.axis("off")
            # Root
            ax_mcts.add_patch(mpatches.FancyBboxPatch((4,6.5),2,1.0,boxstyle="round,pad=0.1",
                                                        facecolor="#6a1b9a",edgecolor="#b39ddb",lw=2))
            ax_mcts.text(5,7,"Root s⁰",ha="center",va="center",color="white",fontsize=8,fontweight="bold")
            ax_mcts.text(5,6.3,"v=0.65 p=←↑→",ha="center",color="#b39ddb",fontsize=7)
            # Branches — (x_pos, y_pos, action_label, value_label, edge_color)
            branches = [
                (1.5, 5.0, "a=←", "v=0.42", "#0288d1"),
                (5.0, 5.0, "a=↑", "v=0.71", "#4caf50"),
                (8.5, 5.0, "a=→", "v=0.38", "#ffa726"),
            ]
            for x2v, y2, action, value, ec in branches:
                ax_mcts.annotate("",xy=(x2v,5.8),xytext=(5,6.5),arrowprops=dict(arrowstyle="->",color=ec,lw=1.5))
                ax_mcts.add_patch(mpatches.FancyBboxPatch((x2v-1,4.5),2,0.8,boxstyle="round,pad=0.1",
                                                            facecolor=CARD,edgecolor=ec,lw=1.5))
                ax_mcts.text(x2v,4.9,f"{action} {value}",ha="center",va="center",color="white",fontsize=7)
            ax_mcts.text(5,3.5,"f_θ(s) → (policy,value)\ng_θ(s,a) → (s',r)",ha="center",color="#9e9ebb",fontsize=7.5)
            ax_mcts.text(5,2.5,"MCTS: 50,000 simulations\nin latent space",ha="center",color="#e65100",fontsize=8)
            ax_mcts.set_title("MuZero MCTS in Latent Space",color="white",fontweight="bold")
            plt.tight_layout(); st.pyplot(fig_mcts); plt.close()

        st.dataframe(pd.DataFrame({
            "Game": ["Go 19×19","Chess","Shogi","Atari (avg 57 games)"],
            "AlphaZero Elo": ["5185","3430","4940","N/A"],
            "MuZero Elo": ["5243","3468","4953","N/A"],
            "MuZero median Human norm.": ["N/A","N/A","N/A","734%"],
            "DQN median": ["N/A","N/A","N/A","100%"],
        }), use_container_width=True, hide_index=True)

    with tab_dream:
        _sec("🌙","DreamerV3 — One Algorithm for Everything",
             "Latent RSSM + symlog + fixed hyperparams — Atari, DMControl, Minecraft, robotics","#ad1457")
        st.markdown(_card("#ad1457","🌙","What makes DreamerV3 special",
            """DreamerV3 (Hafner et al. 2023) achieves something remarkable: the SAME algorithm
            with the SAME hyperparameters (no tuning) achieves strong results on:
            Atari 200K (100 games), DMControl continuous control, Crafter (open world), ProcGen,
            and Minecraft diamond collection (first RL agent to do this without demonstrations).
            The key innovations: (1) Recurrent State Space Model (RSSM) with both deterministic
            and stochastic components; (2) symlog transform for all predictions — normalises
            reward targets across environments with very different scales; (3) KL balancing to
            prevent posterior collapse; (4) Free bits regularisation for stable KL.
            DreamerV3 trains a world model from real data, then trains actor-critic entirely
            in the latent dream — no real environment steps during policy learning."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RSSM — Recurrent State Space Model:**")
            st.latex(r"h_t = f_\phi(h_{t-1},\;z_{t-1},\;a_{t-1}) \quad\text{(GRU deterministic)}")
            st.latex(r"z_t \sim q_\phi(z_t|h_t,o_t) \quad\text{(posterior: with observation)}")
            st.latex(r"\hat z_t \sim p_\phi(\hat z_t|h_t) \quad\text{(prior: for imagination)}")
            st.markdown("**Symlog transform (normalises across reward scales):**")
            st.latex(r"\text{symlog}(x) = \text{sign}(x)\cdot\log(|x|+1)")
            st.markdown("All targets (rewards, values, reconstructions) are symlog-transformed. This is why DreamerV3 works from −1 to +10,000 reward without tuning.")
        with col2:
            # Symlog comparison
            x_sym = np.linspace(-100, 100, 300)
            y_sym = np.sign(x_sym) * np.log1p(np.abs(x_sym))
            fig_sym, ax_sym = _fig(1,1,5.5,4)
            ax_sym.plot(x_sym, x_sym, color="#546e7a", lw=1.5, ls="--", alpha=0.7, label="y=x (unscaled)")
            ax_sym.plot(x_sym, y_sym, color="#ad1457", lw=2.5, label="symlog(x) = sign(x)·log(|x|+1)")
            ax_sym.set_xlabel("Raw reward value", color="white")
            ax_sym.set_ylabel("Transformed value", color="white")
            ax_sym.set_title("symlog: compresses large rewards\nwithout clipping", color="white", fontweight="bold")
            ax_sym.legend(facecolor=CARD, labelcolor="white", fontsize=8)
            ax_sym.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_sym); plt.close()

        st.markdown("**World model loss (4 terms trained jointly):**")
        st.latex(r"\mathcal{L}_\text{WM} = \underbrace{\mathbb{E}[\log p(o_t|z_t,h_t)]}_\text{reconstruction} + \underbrace{\mathbb{E}[\log p(r_t|z_t,h_t)]}_\text{reward} + \underbrace{\mathbb{E}[\log p(d_t|z_t,h_t)]}_\text{episode end} - \underbrace{\beta D_\text{KL}(q\|p)}_\text{latent regularisation}")
        st.markdown(_insight("DreamerV3 practical tip: on GPU, it runs 1000+ imagination steps per second. Real-world training uses 50% world model updates, 50% actor-critic updates in imagination. The actor never touches the real environment during training — only the replay buffer does."), unsafe_allow_html=True)

    with tab_mpc:
        _sec("🎯","MPC, PETS & TD-MPC2 — Model Predictive Control for RL",
             "Plan-as-you-go using a learned model — robust, interpretable, SOTA on DMControl","#00897b")
        st.markdown(_card("#00897b","🎯","MPC: plan at every step using your current model",
            """Model Predictive Control (MPC) takes the model-based idea to an extreme: at EVERY
            timestep, plan H steps ahead, execute only the first action, then replan.
            This is different from DreamerV3 which trains a fixed policy in imagination.
            MPC uses the model as a direct planning engine — no separate policy network needed.
            PETS (Probabilistic Ensembles with Trajectory Sampling, Chua 2018) learned the first
            practical deep MPC for continuous control: train an ensemble of 5 neural network dynamics
            models, use the Cross-Entropy Method (CEM) to optimise action sequences, account for
            model uncertainty via ensemble disagreement.
            TD-MPC2 (Hansen 2023) achieves SOTA on DMControl by combining PETS-style planning
            with a learned latent space (like DreamerV3) and temporal difference value estimation."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**MPC optimisation objective at time t:**")
            st.latex(r"a_{t:t+H}^* = \arg\max_{a_{t:t+H}}\sum_{k=0}^{H-1}\gamma^k \hat r(s_{t+k},a_{t+k})")
            st.markdown("**CEM action optimisation (PETS):**")
            st.latex(r"\mu_{k+1},\sigma_{k+1} = \text{fit\_Gaussian}(\text{top-}K\text{ action sequences})")
            st.markdown("Execute only a_t, then replan at t+1 with updated model and new observation.")
            st.markdown("**PETS ensemble uncertainty:**")
            st.latex(r"\hat\sigma^2_\text{epist}(s,a) = \frac{1}{M}\sum_m(\hat\mu_m(s,a)-\bar\mu)^2")
        with col2:
            # MPC planning horizon illustration
            np.random.seed(42); H = 15; n_traj = 8
            fig_mpc, ax_mpc = _fig(1,1,5.5,4)
            start = np.array([0.0, 0.0])
            for i in range(n_traj):
                traj = [start]
                for h in range(H):
                    step = np.random.randn(2)*0.3 + np.array([0.15, 0])
                    traj.append(traj[-1] + step)
                traj = np.array(traj)
                val = np.random.rand()
                col = plt.cm.RdYlGn(val)
                ax_mpc.plot(traj[:,0], traj[:,1], color=col, lw=1.5, alpha=0.7)
            ax_mpc.scatter([0],[0], color="white", s=100, zorder=5, label="Current state")
            ax_mpc.scatter([2.5],[0], color="#4caf50", s=100, marker="*", zorder=5, label="Goal")
            ax_mpc.set_title(f"MPC: {n_traj} candidate action seqs\n(CEM selects top-K, refit)", color="white", fontweight="bold")
            ax_mpc.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_mpc.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_mpc); plt.close()

        st.dataframe(pd.DataFrame({
            "Method": ["SAC (model-free)","PETS (ensemble MPC)","DreamerV3","TD-MPC2"],
            "DMControl score (avg)": ["850","780","910","955"],
            "Real env steps (1M)": ["1M","200K","500K","300K"],
            "Model type": ["None","Probabilistic ensemble","RSSM latent","Latent + TD value"],
            "Planning": ["None","CEM in state space","Actor in latent dream","CEM in latent space"],
        }), use_container_width=True, hide_index=True)

    with tab_cmp:
        _sec("📊","Model-Based RL Comparison","When to use each algorithm","#546e7a")
        st.dataframe(pd.DataFrame({
            "Algorithm":["Dyna-Q","World Models","MuZero","DreamerV3","PETS","TD-MPC2"],
            "Model type":["Tabular","VAE+RNN","Learned dynamics","RSSM latent","Neural ensemble","Latent+TD"],
            "Action space":["Discrete","Continuous","Both","Both","Continuous","Continuous"],
            "Key strength":["Simplest MBRL","Dream training","Superhuman games","Fixed hyperparams","Interpretable","DMControl SOTA"],
            "Limitation":["Tabular only","Requires offline M training","Expensive MCTS","Long to train","Slow at test time","Complex"],
            "Best env":["GridWorlds","CarRacing","Games (Go/Chess)","Everything","Mujoco","DMControl"],
            "Year":["1991","2018","2019","2023","2018","2023"],
        }), use_container_width=True, hide_index=True)

    with tab_res:
        _sec("📚","Books & Deep-Dive Resources","Best resources to go deep on Model-Based RL","#546e7a")
        for title, authors, why, url in [
            ("Reinforcement Learning: An Introduction — Ch.8 (Planning)",
             "Sutton & Barto (2018) — FREE at incompleteideas.net",
             "Chapter 8 covers Dyna-Q, prioritised sweeping, MCTS. The theoretical foundation for all MBRL.",
             "http://incompleteideas.net/book/the-book.html"),
            ("Dream to Control: Learning Behaviors by Latent Imagination (DreamerV1)",
             "Hafner et al. (2019) — ICLR 2020",
             "The original Dreamer paper. Cleaner than V3 for first read. Full RSSM derivation.",
             "https://arxiv.org/abs/1912.01603"),
            ("DreamerV3: Mastering Diverse Domains through World Models",
             "Hafner et al. (2023) — The SOTA world model paper",
             "All innovations: symlog, KL balancing, free bits. Required reading for modern MBRL.",
             "https://arxiv.org/abs/2301.04104"),
            ("Deep Learning for Model-Based RL (ICLR 2022 Tutorial)",
             "Hafner, Lee, Fischer, Abbeel — 3 hour video tutorial",
             "Best overview of all MBRL methods: Dyna, World Models, DreamerV3, MuZero in one place.",
             "https://sites.google.com/view/mbrl-tutorial"),
            ("TD-MPC2: Scalable Robust World Models",
             "Hansen et al. (2023) — ICLR 2024",
             "Achieves SOTA on DMControl. Very clean architecture combining MPC and latent world models.",
             "https://arxiv.org/abs/2310.16828"),
            ("PETS: Deep Reinforcement Learning in a Handful of Trials",
             "Chua et al. (2018) — NeurIPS",
             "The probabilistic ensemble MPC paper. The bridge between classical MPC and deep RL.",
             "https://arxiv.org/abs/1805.12114"),
        ]:
            st.markdown(_book(title, authors, why, url), unsafe_allow_html=True)

        st.subheader("🎓 Courses")
        for icon, title, desc, url in [
            ("🎥","CS285 Berkeley — Lecture 11: Model-Based RL",
             "Levine's lecture covering Dyna, world models, MBPO. Clear derivations.",
             "https://rail.eecs.berkeley.edu/deeprlcourse/"),
            ("💻","dreamer-pytorch (GitHub)",
             "Clean PyTorch implementation of DreamerV1 with detailed comments.",
             "https://github.com/zhaoyi11/dreamer-pytorch"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)

    mbrl_notes = [
        (tab_why, "Why Model-Based RL", "model_based_rl"),
        (tab_dyna, "Dyna-Q", "model_based_rl_dyna_q"),
        (tab_wm, "World Models", "model_based_rl_world_models"),
        (tab_mu, "MuZero", "model_based_rl_muzero"),
        (tab_dream, "Dreamer", "model_based_rl_dreamer"),
        (tab_mpc, "MPC", "model_based_rl_mpc"),
        (tab_cmp, "Comparison", "model_based_rl_comparison"),
        (tab_res, "Resources", "model_based_rl_resources"),
    ]
    for tab, note_title, note_slug in mbrl_notes:
        with tab:
            render_notes(f"Model-Based RL - {note_title}", note_slug)
