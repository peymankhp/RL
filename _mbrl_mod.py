"""
_mbrl_mod.py  — Model-Based Reinforcement Learning
Covers: Dyna-Q · World Models (Ha & Schmidhuber) · MuZero · DreamerV3
Theme: Agents that build an internal model of the world — 10–100× more sample efficient.
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"
COLORS = {"dynaq":"#e65100","world":"#7c4dff","muzero":"#1565c0","dreamer":"#00897b"}

def _fig(nrows=1, ncols=1, w=12, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(DARK); ax.tick_params(colors="#9e9ebb", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    return fig, axes

def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
            f'<b>{icon} {title}</b><br>{body}</div>')

def _insight(text):
    return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
            f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem">'
            f'💡 {text}</div>')

def _sec(emoji, title, sub, color="#e65100"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def smooth(a, w=5):
    return np.convolve(a, np.ones(w)/w, mode="valid") if len(a) > w else np.array(a, float)

# ── Simple GridWorld for Dyna-Q demo ─────────────────────────────────────
class SimpleGridWorld:
    def __init__(self, n=6):
        self.n = n
        self.obstacles = {(2,1),(2,2),(2,3),(4,2),(4,3)}
        self.goal = (n-1, n-1)
        self.reset()
    def reset(self):
        self.pos = (0, 0)
        return self._obs()
    def _obs(self):
        return self.pos[0]*self.n + self.pos[1]
    def step(self, a):
        dy, dx = [(-1,0),(1,0),(0,-1),(0,1)][a]
        ny = max(0, min(self.n-1, self.pos[0]+dy))
        nx = max(0, min(self.n-1, self.pos[1]+dx))
        if (ny,nx) not in self.obstacles:
            self.pos = (ny, nx)
        done = (self.pos == self.goal)
        r = 1.0 if done else -0.01
        return self._obs(), r, done

# ── Dyna-Q implementation ─────────────────────────────────────────────────
def run_dynaq(n_episodes=50, n_planning=10, alpha=0.1, gamma=0.99,
              eps=0.15, seed=42):
    np.random.seed(seed)
    env = SimpleGridWorld(6); n_states = 36; n_actions = 4
    Q = np.zeros((n_states, n_actions))
    model = {}  # (s,a) -> (r, s')
    rewards = []

    for ep in range(n_episodes):
        s = env.reset(); ep_r = 0; steps = 0
        while steps < 200:
            # ε-greedy action selection
            a = np.random.randint(n_actions) if np.random.rand()<eps else np.argmax(Q[s])
            s2, r, done = env.step(a)
            ep_r += r; steps += 1
            # Q-learning update on real experience
            Q[s,a] += alpha*(r + gamma*np.max(Q[s2]) - Q[s,a])
            model[(s,a)] = (r, s2)  # store in model
            # Planning: n_planning simulated updates from model
            for _ in range(n_planning):
                ks = list(model.keys())
                ps, pa = ks[np.random.randint(len(ks))]
                pr, ps2 = model[(ps,pa)]
                Q[ps,pa] += alpha*(pr + gamma*np.max(Q[ps2]) - Q[ps,pa])
            s = s2
            if done: break
        rewards.append(ep_r)
    return rewards, Q

def main_mbrl():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a0a0a,#0a0a2e,#0a1a0a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🏗️ Model-Based Reinforcement Learning</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'Agents that build an internal model of the world — 10–100× more sample efficient than model-free RL. '
        'From Dyna-Q (1990) to DreamerV3 (2023): learn an environment model, then plan inside it.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🧭 Why Model-Based?",
        "🔄 Dyna-Q",
        "🌍 World Models",
        "♟️ MuZero",
        "🌙 DreamerV3",
        "📊 Comparison",
        "📚 Resources",
    ])
    (tab_why, tab_dyna, tab_world, tab_mu, tab_dream, tab_cmp, tab_res) = tabs

    # ── WHY MODEL-BASED ───────────────────────────────────────────────────
    with tab_why:
        _sec("🧭","Why Build a World Model?",
             "The fundamental efficiency argument — imagination is cheaper than reality","#e65100")

        st.markdown(_card("#e65100","🤔","The core problem with model-free RL",
            """Model-free RL methods (DQN, PPO, SAC) learn entirely from direct environment interactions.
            Every gradient step requires real experience. To learn to play a single Atari game, DQN needs
            roughly 50 million frames — equivalent to 230 hours of non-stop gameplay at 60fps. A human
            child can learn Pong in minutes from watching a few games. The gap is that humans build a
            mental model of the game: the ball bounces off walls, the paddle blocks it, missing the ball
            costs a point. With this model, we can plan ahead and reason about consequences without
            having to physically experience every possibility. Model-based RL replicates this: learn
            a model of the environment (also called a world model, dynamics model, or transition model)
            then use it to generate imaginary rollouts for planning or additional training data.
            The payoff is dramatic: on data-limited benchmarks, model-based methods achieve the same
            performance as model-free methods with 10–100× fewer environment interactions."""),
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **What a world model learns:**
            A transition model $\hat p(s_{t+1}|s_t,a_t)$ predicts where you'll end up
            after taking action $a$ in state $s$. Often combined with:
            - $\hat r(s_t,a_t)$ — reward predictor
            - $\hat d(s_t,a_t)$ — done/terminal predictor
            - $h_t = f(h_{t-1}, s_t, a_t)$ — latent state encoder (DreamerV3)
            """)
            st.latex(r"\hat s_{t+1} = f_\phi(s_t, a_t)")
            st.latex(r"\hat r_t = g_\phi(s_t, a_t)")
        with col2:
            st.markdown(r"""
            **The planning loop:**
            Given the model, generate imagined trajectories:
            """)
            st.latex(r"\tau^{\text{imag}} = \{(s_t,a_t,\hat r_t,\hat s_{t+1})\}_{t=0}^H")
            st.markdown(r"""
            Train the policy on these imagined rollouts — no real environment needed!
            The key question: **how accurate does the model need to be?**
            Errors compound over time: a 1% step error becomes $(1.01)^{50} = 1.64\times$ bigger
            after 50 steps. This is why model-based methods are hardest in long-horizon tasks.
            """)

        st.subheader("📊 Sample Efficiency Comparison (approximate, task-dependent)")
        st.dataframe(pd.DataFrame({
            "Algorithm":["PPO","DQN","Rainbow","SAC","Dyna-Q","World Models","MuZero","DreamerV3"],
            "Type":["Model-free","Model-free","Model-free","Model-free","Model-based","Model-based","Model-based","Model-based"],
            "Real env steps to match human":["~50M frames","~50M frames","~7M frames","~3M steps","~200K steps","~100K steps","~100K steps","~200K steps"],
            "Planning ahead":["❌","❌","❌","❌","✅ Tabular","✅ Imagined","✅ MCTS","✅ Latent rollouts"],
            "Year":["2017","2015","2018","2018","1990","2018","2020","2023"],
        }), use_container_width=True, hide_index=True)
        st.caption("Numbers are illustrative — performance varies significantly by task and environment.")

    # ── DYNA-Q ────────────────────────────────────────────────────────────
    with tab_dyna:
        _sec("🔄","Dyna-Q — Planning in a Tabular Model",
             "Sutton (1990) — The first model-based RL algorithm: real experience + simulated planning","#e65100")

        st.markdown(_card("#e65100","🔄","What is Dyna-Q and why it was revolutionary",
            """Dyna-Q (Sutton 1990) is the pioneering model-based RL algorithm that introduced the
            idea of separating the learning of the environment model from using the model for planning.
            The key insight was simple but powerful: after each real environment step, instead of just
            doing one Q-learning update, do n additional Q-learning updates using simulated transitions
            drawn from the learned model. This n-step planning amplifies each real experience into n+1
            gradient steps, dramatically improving sample efficiency. The model is simply a lookup table
            that stores the observed (reward, next_state) for each (state, action) pair seen so far.
            Dyna-Q showed empirically that n=10 simulated planning steps could learn a maze task 10×
            faster than pure Q-learning with almost no computational overhead (the model lookup is O(1)).
            Although Dyna-Q only works for small tabular MDPs, the principle it introduced — interleave
            real experience with model-based planning — is the foundation of every modern model-based
            deep RL algorithm including World Models, MuZero, and DreamerV3. The architecture is
            three-component: (1) direct RL from real experience; (2) model learning from real experience;
            (3) planning/Q-learning using the model. These three loops run simultaneously."""),
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**The Dyna-Q update equations:**")
            st.latex(r"\text{Direct RL: } Q(s,a) \leftarrow Q(s,a)+\alpha\bigl[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\bigr]")
            st.latex(r"\text{Model update: } \text{Model}(s,a) \leftarrow (r, s')")
            st.markdown("**n-step planning loop:**")
            st.latex(r"\text{For } k=1\ldots n: \quad Q(s_k,a_k)\leftarrow Q(s_k,a_k)+\alpha\bigl[\hat r+\gamma\max_{a'} Q(\hat s',a')-Q(s_k,a_k)\bigr]")
            st.markdown(r"where $(s_k,a_k)\sim\text{Uniform}(\text{previously visited})$ and $(\hat r,\hat s')=\text{Model}(s_k,a_k)$")
        with col2:
            st.code(r"""
# Dyna-Q: the complete algorithm
Q = zeros(n_states, n_actions)
Model = {}  # (s,a) → (r, s')

for episode in range(N):
    s = env.reset()
    while not done:
        # 1. Take action (ε-greedy from Q)
        a = epsilon_greedy(Q[s])
        s', r, done = env.step(a)

        # 2. DIRECT Q-LEARNING (1 real step)
        Q[s,a] += α * (r + γ*max(Q[s']) - Q[s,a])

        # 3. UPDATE MODEL
        Model[(s,a)] = (r, s')

        # 4. PLANNING (n simulated steps)
        for k in range(n_planning):
            s_sim, a_sim = random(previously_seen)
            r_sim, s_next_sim = Model[(s_sim, a_sim)]
            Q[s_sim,a_sim] += α*(r_sim+γ*max(Q[s_next_sim])-Q[s_sim,a_sim])
        s = s'
""", language="python")

        st.subheader("🎛️ Interactive: How many planning steps help?")
        c1, c2, c3 = st.columns(3)
        n_ep_d = c1.slider("Episodes", 20, 100, 50, 10, key="dq_ep")
        n_plan = c2.slider("Planning steps n", 0, 50, 10, 5, key="dq_n")
        seed_d = c3.number_input("Seed", 0, 999, 42, key="dq_sd")

        if st.button("▶️ Compare n=0 (Q-Learning) vs Dyna-Q", type="primary", key="btn_dq"):
            with st.spinner("Running…"):
                r0, _ = run_dynaq(n_ep_d, 0, seed=int(seed_d))
                rn, Q = run_dynaq(n_ep_d, n_plan, seed=int(seed_d))
            st.session_state["dq_res"] = (r0, rn)

        if "dq_res" in st.session_state:
            r0, rn = st.session_state["dq_res"]
            fig_dq, ax_dq = _fig(1, 1, 10, 4)
            sm0 = smooth(r0, 7); smn = smooth(rn, 7)
            ax_dq.plot(r0, color="#546e7a", alpha=0.2, lw=0.5)
            ax_dq.plot(range(len(sm0)), sm0, color="#546e7a", lw=2, label=f"Q-Learning (n=0), late={np.mean(r0[-15:]):.2f}")
            ax_dq.plot(rn, color=COLORS["dynaq"], alpha=0.2, lw=0.5)
            ax_dq.plot(range(len(smn)), smn, color=COLORS["dynaq"], lw=2.5, label=f"Dyna-Q (n={n_plan}), late={np.mean(rn[-15:]):.2f}")
            ax_dq.set_xlabel("Episode", color="white"); ax_dq.set_ylabel("Episode reward", color="white")
            ax_dq.set_title(f"Dyna-Q vs Q-Learning on GridWorld (n={n_plan} planning steps)",
                            color="white", fontweight="bold")
            ax_dq.legend(facecolor=CARD, labelcolor="white", fontsize=9); ax_dq.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_dq); plt.close()
            st.metric("Speedup", f"{np.mean(rn[-15:])/max(abs(np.mean(r0[-15:])),0.01):.1f}×",
                      f"from {n_plan} planning steps per real step")

    # ── WORLD MODELS ──────────────────────────────────────────────────────
    with tab_world:
        _sec("🌍","World Models (Ha & Schmidhuber 2018)",
             "Dream inside a compressed latent world — train entirely without the real environment","#7c4dff")

        st.markdown(_card("#7c4dff","🌍","What are World Models and why are they powerful?",
            """World Models (Ha & Schmidhuber, 2018) introduced the idea of training an agent entirely
            inside a learned neural model of the environment — called 'dreaming'. The architecture has
            three components: (1) a Vision model (V) — a Variational Autoencoder (VAE) that compresses
            each 64×64 pixel observation into a compact latent vector z of ~32 numbers; (2) a Memory
            model (M) — an RNN (specifically an MDN-RNN: Mixture Density Network RNN) that takes z_t
            and action a_t to predict the distribution over z_{t+1} at the next step, capturing the
            temporal dynamics of the environment; (3) a Controller (C) — a tiny linear model that maps
            (z_t, h_t) directly to actions, where h_t is the RNN hidden state. The revolutionary idea:
            you can train the Controller entirely inside the model (the M component), never touching the
            real environment. The agent 'imagines' rollouts, gets predicted rewards from M, and optimises
            C using CMA-ES (a gradient-free evolutionary strategy). When this trained agent is deployed
            in the real environment, it performs well because the model is accurate enough. This showed
            that rich perceptual encoding + temporal dynamics model = enough to learn complex tasks
            entirely in imagination. The approach was later scaled and refined in Dreamer and DreamerV3."""),
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**The three-module architecture:**")
            st.latex(r"z_t = \text{VAE}_\text{encoder}(o_t) \quad \text{(Vision: compress observation)}")
            st.latex(r"h_{t+1} = \text{RNN}(h_t,\,z_t,\,a_t) \quad \text{(Memory: predict dynamics)}")
            st.latex(r"a_t = C(z_t,\,h_t) \quad \text{(Controller: linear policy)}")
        with col2:
            st.markdown("**VAE training objective** (compress + reconstruct):")
            st.latex(r"\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q(z|o)}[\log p(o|z)] + D_{\text{KL}}(q(z|o)\|p(z))")
            st.markdown(r"""
            - First term: reconstruction loss (decode z back to pixels)
            - Second term: KL divergence forcing z to stay near $\mathcal{N}(0,I)$ (regularisation)
            """)
            st.markdown("**MDN-RNN predicts z distribution:**")
            st.latex(r"p(z_{t+1}|z_t,a_t,h_t) = \sum_k\pi_k\,\mathcal{N}(\mu_k,\sigma_k^2)")

        st.markdown(r"""
        **The 'dream within a dream' trick:** Instead of collecting more real data, the controller is
        trained on sequences of latent states $(z_t, h_t)$ hallucinated by the RNN. The temperature
        parameter $\tau$ of the RNN can be increased to make training more challenging (hallucinate
        harder, more varied environments), which acts as a natural data augmentation technique that
        improves robustness of the controller.
        """)
        st.markdown(_insight("""
        <b>Why does this work?</b> The VAE forces the agent to compress the observation into its most
        important features (the latent z). The RNN captures how those features evolve over time.
        The controller, operating only on these compact representations, learns to act optimally in
        the imagined world. When deployed in the real world, the same V-M-C pipeline works because
        the model is a faithful enough representation of reality. This architecture also scales:
        just make V, M, and C bigger networks to handle more complex environments.
        """), unsafe_allow_html=True)

    # ── MUZERO ─────────────────────────────────────────────────────────────
    with tab_mu:
        _sec("♟️","MuZero (Schrittwieser et al. 2020)",
             "AlphaGo + learned dynamics — master Chess, Go, Atari with a model that only plans, never reconstructs","#1565c0")

        st.markdown(_card("#1565c0","♟️","What is MuZero and how does it differ from AlphaGo?",
            """MuZero (DeepMind, 2020) is one of the most significant breakthroughs in model-based RL.
            AlphaGo and AlphaZero used perfect known game rules for planning (Monte Carlo Tree Search).
            MuZero eliminates this requirement: it learns its own dynamics model from experience,
            without ever being given the rules of the game. Yet it achieves superhuman performance on
            Chess, Go, Shogi (board games), and 57 Atari games — all with the same algorithm and
            architecture. The critical design choice: MuZero's model does not learn to reconstruct
            pixel observations. Instead, it learns a compact latent representation of state
            (called the 'hidden state') and a dynamics model that operates entirely in this latent
            space. The model only needs to be accurate enough for planning value-relevant aspects —
            it can completely ignore irrelevant visual details. This makes it much easier to learn
            than pixel-reconstruction models like World Models. At each step, MuZero runs Monte Carlo
            Tree Search (MCTS) using the learned model to plan ahead, selecting actions based on
            value estimates and visit counts — the same as AlphaZero but with learned, not known, dynamics.
            The result: a single algorithm that works across entirely different domains (board games,
            video games, planning problems) by learning task-specific dynamics from scratch."""),
            unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**MuZero's three learned functions:**")
            st.latex(r"h_0 = f_\theta(o_{1:t}) \quad \text{(Representation: obs → latent state)}")
            st.latex(r"(r_k,h_k) = g_\theta(h_{k-1},a_k) \quad \text{(Dynamics: latent transition)}")
            st.latex(r"(p_k,v_k) = f_\theta(h_k) \quad \text{(Prediction: policy + value)}")
        with col2:
            st.markdown("**MCTS planning with learned model:**")
            st.latex(r"\text{Upper Confidence Bound: } \text{UCB}(s,a) = Q(s,a) + c_1\log\frac{N(s)}{N(s,a)+1}\frac{P(s,a)}{N(s,a)+1}")
            st.markdown(r"""
            MCTS selects actions by UCB, expands nodes using the dynamics model $g_\theta$,
            and backs up value estimates through the tree. After $\sim$800 simulations,
            the action with the most visits is selected.
            """)

        st.markdown("**MuZero training loss:**")
        st.latex(r"\mathcal{L}(\theta) = \sum_{k=0}^K\left[\ell^r(r_t^k, \hat r^k) + \ell^v(z_t^k, \hat v^k) + \ell^p(\pi_t^k, \hat p^k)\right]")
        st.markdown(r"""
        Three simultaneous supervision signals:
        - $\ell^r$: predicted reward $\hat r^k$ vs actual reward $r_t^k$ — learn to predict returns
        - $\ell^v$: predicted value $\hat v^k$ vs bootstrapped target $z_t^k$ — learn state values
        - $\ell^p$: predicted policy $\hat p^k$ vs MCTS visit counts $\pi_t^k$ — improve planning
        """)
        st.markdown(_insight("""
        <b>Why MuZero is so general:</b> By learning a dynamics model in latent space rather than
        pixel space, MuZero avoids the hard problem of generating realistic images. The model only
        needs to capture what's relevant for planning (reward-relevant features), not photorealistic
        reconstruction. This is why the same architecture works on both Chess (discrete perfect information)
        and Atari (continuous pixel input with stochastic dynamics) — the representation and dynamics
        functions adapt to each domain through training.
        """), unsafe_allow_html=True)

    # ── DREAMERV3 ─────────────────────────────────────────────────────────
    with tab_dream:
        _sec("🌙","DreamerV3 (Hafner et al. 2023)",
             "The first algorithm to train a human-level Minecraft diamond collector — all in imagination","#00897b")

        st.markdown(_card("#00897b","🌙","What is DreamerV3 and why is it a landmark result?",
            """DreamerV3 (Hafner et al., 2023) is arguably the most powerful general-purpose model-based
            RL algorithm to date. It learned to collect diamonds in Minecraft — a task requiring hundreds
            of sequential decisions, tool crafting, and resource management — from scratch, using only
            image observations and reward. No demonstrations, no hard-coded subgoals. This had never been
            achieved before by any RL algorithm. DreamerV3 trains its world model using RSSM (Recurrent
            State Space Model): a hybrid of deterministic GRU recurrence and stochastic latent variables,
            allowing it to model both deterministic dynamics (which way the ball goes given physics) and
            irreducible uncertainty (random events). Policy learning happens entirely in the imagined
            latent trajectories — the actor and critic never see real pixels. DreamerV3 introduces two
            key engineering innovations: (1) symlog predictions — using a symmetric log transform for
            all predictions normalises targets across environments with very different reward scales;
            (2) a KL-balancing trick that stabilises the world model training. These innovations make
            DreamerV3 the first single algorithm with fixed hyperparameters that works well across
            continuous control, Atari, 3D environments, and text games — a genuine general-purpose
            world model."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RSSM (Recurrent State Space Model):**")
            st.latex(r"h_t = \text{GRU}(h_{t-1},\,z_{t-1},\,a_{t-1}) \quad \text{(deterministic path)}")
            st.latex(r"z_t \sim q_\phi(z_t|h_t,o_t) \quad \text{(posterior: given real obs)}")
            st.latex(r"\hat z_t \sim p_\phi(\hat z_t|h_t) \quad \text{(prior: for imagination)}")
        with col2:
            st.markdown("**World model training — 4 losses simultaneously:**")
            st.latex(r"\mathcal{L} = \underbrace{\ell_{\text{recon}}}_{\text{decode obs}} + \underbrace{\ell_{\text{reward}}}_{\text{predict }r} + \underbrace{\ell_{\text{done}}}_{\text{predict terminal}} + \underbrace{\beta\cdot D_{\text{KL}}(q\|p)}_{\text{regularise latent}}")
            st.markdown(r"""
            The **symlog transform** $\text{sg}(x) = \text{sign}(x)\log(|x|+1)$ normalises targets:
            """)
            st.latex(r"\hat y = \text{sg}^{-1}(\text{output}) \quad \text{(works for rewards from -1000 to +1000)}")

        st.markdown("**Policy and value learned in imagination:**")
        st.latex(r"\hat\tau = \{(\hat z_t,\hat a_t,\hat r_t)\}_{t=1}^H \quad \text{(roll out H=15 steps in latent space)}")
        st.latex(r"V_\lambda(\hat z_t) = \hat r_t + \gamma(\lambda V(\hat z_{t+1})+(1-\lambda)\hat v_{t+1}) \quad \text{(TD(λ) target in imagination)}")
        st.markdown(r"""
        The actor maximises $\mathbb{E}[\sum_t V_\lambda(\hat z_t)]$ over imagined trajectories,
        the critic learns to predict $V_\lambda$. No real environment data touches the policy.
        """)

    # ── COMPARISON ────────────────────────────────────────────────────────
    with tab_cmp:
        _sec("📊","Model-Based RL Comparison","When to use each approach","#e65100")
        st.dataframe(pd.DataFrame({
            "Algorithm":["Dyna-Q","World Models","MuZero","DreamerV3"],
            "State space":["Tabular","Pixel (64×64)","Any","Any (pixel/vector)"],
            "Model type":["Lookup table","VAE + MDN-RNN","Latent dynamics","RSSM (GRU+stochastic)"],
            "Planning":["Q-learning in model","CMA-ES in imagination","MCTS in latent space","Actor-Critic in imagination"],
            "Obs reconstruction":["N/A","✅ Required","❌ Not needed","✅ With symlog"],
            "Sample efficiency":["10× vs Q-Learning","50× vs PPO (visual)","10× vs Rainbow","10–100× vs PPO"],
            "Best for":["Tabular/small MDPs","Low-dim continuous control","Board games + Atari","General: continuous + discrete"],
            "Difficulty":["Easy to implement","Moderate","Hard (MCTS required)","Moderate (with code)"],
        }), use_container_width=True, hide_index=True)

        st.markdown("""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:10px;padding:1rem 1.3rem;margin-top:1rem">
        <b style="color:white">When should you use model-based over model-free RL?</b><br><br>
        <span style="color:#b0b0cc">
        ✅ Use model-based when: (1) real environment data is expensive (robotics, clinical trials);
        (2) environment has smooth, learnable dynamics (physics simulation); (3) you need to plan ahead
        multiple steps (strategic games, tool use); (4) sample efficiency is the primary constraint.<br><br>
        ❌ Avoid model-based when: (1) environment dynamics are highly stochastic or chaotic (hard to model accurately);
        (2) model errors compound to make imagined rollouts misleading; (3) fast simulation makes
        model-free feasible; (4) action space is very large (model errors become amplified during MCTS).
        </span></div>""", unsafe_allow_html=True)

    # ── RESOURCES ─────────────────────────────────────────────────────────
    with tab_res:
        st.subheader("📚 Primary Resources")
        for icon, title, desc, url in [
            ("📄","Sutton (1990) — Dyna Architecture","Original Dyna paper. Short and foundational.","http://incompleteideas.net/papers/sutton-90.pdf"),
            ("📄","Ha & Schmidhuber (2018) — World Models","VAE + MDN-RNN + CMA-ES. Train agent in imagination.","https://worldmodels.github.io"),
            ("📄","Schrittwieser et al. (2020) — MuZero","Planning without knowing game rules. Superhuman on board games + Atari.","https://arxiv.org/abs/1911.08265"),
            ("📄","Hafner et al. (2023) — DreamerV3","Fixed hyperparameters, all domains, Minecraft diamond. State of the art.","https://arxiv.org/abs/2301.04104"),
            ("📄","Sutton & Barto Ch. 8 — Planning and Learning","Dyna framework, priority sweeping, model-based RL theory.","http://incompleteideas.net/book/the-book.html"),
            ("💻","dreamer-pytorch","Unofficial PyTorch DreamerV3 implementation. Best starting point.","https://github.com/NM512/dreamer-torch"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
