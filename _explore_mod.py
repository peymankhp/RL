"""
_explore_mod.py — Exploration in Reinforcement Learning
Covers: UCB · Thompson Sampling · ICM · RND · Curiosity-Driven Exploration
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

# ── Multi-armed bandit simulation ─────────────────────────────────────────
def run_bandit(n_arms=10, n_steps=1000, method="ucb", c=2.0, seed=42):
    np.random.seed(seed)
    true_means = np.random.randn(n_arms)
    Q = np.zeros(n_arms); N = np.zeros(n_arms) + 1e-6
    rewards_log = []; regrets_log = []
    best_arm = np.argmax(true_means)

    for t in range(1, n_steps+1):
        if method == "greedy":
            a = np.argmax(Q)
        elif method == "epsilon_greedy":
            a = np.random.randint(n_arms) if np.random.rand() < 0.1 else np.argmax(Q)
        elif method == "ucb":
            a = np.argmax(Q + c * np.sqrt(np.log(t) / N))
        elif method == "thompson":
            samples = np.random.normal(Q, 1.0/np.sqrt(N+1))
            a = np.argmax(samples)
        else:
            a = np.argmax(Q)
        r = true_means[a] + np.random.randn()
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        rewards_log.append(r)
        regrets_log.append(true_means[best_arm] - true_means[a])

    return rewards_log, regrets_log

def main_explore():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1a0a,#0a1a2a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🔍 Exploration in Reinforcement Learning</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'The fundamental unsolved problem of RL: how to explore efficiently in sparse-reward environments. '
        'From bandit algorithms (UCB, Thompson) to deep intrinsic motivation (ICM, RND, curiosity).'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🤔 Exploration vs Exploitation",
        "📊 UCB",
        "🎲 Thompson Sampling",
        "🧠 Intrinsic Motivation",
        "🔭 ICM — Curiosity",
        "🎯 RND",
        "📚 Resources",
    ])
    (tab_ee, tab_ucb, tab_ts, tab_im, tab_icm, tab_rnd, tab_res) = tabs

    with tab_ee:
        st.subheader("🤔 The Exploration-Exploitation Dilemma")
        st.markdown(_card("#ffa726","⚖️","Why exploration is the fundamental problem in RL",
            """Every RL agent faces a fundamental dilemma at every timestep: should it exploit what
            it already knows works (take the action with the highest estimated Q-value) or explore
            unfamiliar territory (try a new action that might be better or might be worse)?
            Too much exploitation: the agent gets stuck in a local optimum, never discovering
            better strategies it doesn't know about yet. Too much exploration: the agent wastes
            time trying random actions and never converges to a good policy. This dilemma becomes
            critical in sparse-reward environments: if a reward is only given after 1000 steps
            of correct actions (like collecting a diamond in Minecraft), a random explorer will
            almost never stumble upon it. Without intelligent exploration, the agent never learns.
            Simple approaches like epsilon-greedy (random action with probability epsilon) work
            for dense rewards but completely fail in sparse-reward settings. Principled exploration
            strategies — UCB for bandits, Thompson sampling for uncertainty, intrinsic motivation
            for deep RL — are the solution. Despite decades of research, exploration in large state
            spaces with sparse rewards remains largely unsolved and is an active research frontier."""), unsafe_allow_html=True)

        st.markdown(r"""
        **The exploration problem formalised (multi-armed bandit version):**
        Imagine $K$ slot machines with unknown reward means $\mu_1,\ldots,\mu_K$.
        At each step, pull one arm and observe $r \sim \mathcal{N}(\mu_a, 1)$.
        Goal: minimise cumulative **regret** — the total reward you missed by not always
        pulling the best arm:
        """)
        st.latex(r"\text{Regret}(T) = T\mu^* - \sum_{t=1}^T \mu_{a_t} \quad \text{where } \mu^* = \max_k\mu_k")
        st.markdown(r"""
        - Greedy (always exploit): $O(T)$ regret — gets stuck on a suboptimal arm forever
        - Random (always explore): $O(T)$ regret — never learns what works
        - UCB, Thompson Sampling: $O(\log T)$ regret — near-optimal theoretical guarantee
        """)

        n_arms_demo = st.slider("Number of arms (actions)", 3, 20, 10, key="ee_arms")
        n_steps_demo = st.slider("Steps", 100, 2000, 500, 100, key="ee_steps")
        if st.button("▶️ Compare Exploration Strategies", type="primary", key="btn_ee"):
            results = {}
            for method in ["greedy","epsilon_greedy","ucb","thompson"]:
                rw, reg = run_bandit(n_arms_demo, n_steps_demo, method)
                results[method] = (rw, reg)
            st.session_state["ee_res"] = results

        if "ee_res" in st.session_state:
            results = st.session_state["ee_res"]
            fig_ee, axes_ee = _fig(1, 2, 14, 4)
            labels = {"greedy":"Greedy (no exploration)","epsilon_greedy":"ε-greedy (ε=0.1)","ucb":"UCB (c=2)","thompson":"Thompson Sampling"}
            colors_m = {"greedy":"#546e7a","epsilon_greedy":"#ffa726","ucb":"#0288d1","thompson":"#4caf50"}
            for method, (rw, reg) in results.items():
                cum_reg = np.cumsum(reg)
                axes_ee[0].plot(np.convolve(rw, np.ones(30)/30, mode="valid"),
                                color=colors_m[method], lw=2, label=labels[method])
                axes_ee[1].plot(cum_reg, color=colors_m[method], lw=2, label=f"{labels[method]}: {cum_reg[-1]:.1f}")
            axes_ee[0].set_xlabel("Step", color="white"); axes_ee[0].set_ylabel("Avg reward (30-step MA)", color="white")
            axes_ee[0].set_title("Average Reward Over Time", color="white", fontweight="bold")
            axes_ee[1].set_xlabel("Step", color="white"); axes_ee[1].set_ylabel("Cumulative regret", color="white")
            axes_ee[1].set_title("Cumulative Regret (lower = better)", color="white", fontweight="bold")
            for ax in [axes_ee[0], axes_ee[1]]:
                ax.legend(facecolor=CARD, labelcolor="white", fontsize=7); ax.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_ee); plt.close()

    with tab_ucb:
        st.subheader("📊 UCB — Upper Confidence Bound (Auer et al. 2002)")
        st.markdown(_card("#0288d1","📊","What UCB does and why it achieves optimal regret",
            """Upper Confidence Bound (UCB) is the first principled exploration strategy with a
            theoretical optimality guarantee. The key idea: instead of just using the estimated
            mean reward Q(a), add an uncertainty bonus that is large for arms tried few times and
            small for arms tried many times. Specifically, UCB selects the arm with the highest
            'upper confidence bound': the mean estimate plus a confidence interval width.
            The confidence interval shrinks as we gather more data (fewer times = more uncertainty =
            wider bound). This automatically creates a natural exploration schedule: the algorithm
            explores undervisited arms (high uncertainty bonus) until the uncertainty is resolved,
            then focuses on the best arm. This is called 'optimism in the face of uncertainty':
            treat each arm as if it might be as good as its upper confidence bound, which optimistically
            assumes uncertain arms could be excellent. The confidence parameter c controls the
            exploration-exploitation balance. UCB achieves O(log T) regret — the theoretical minimum
            for this problem. In deep RL, UCB-style bonuses are used as count-based exploration
            bonuses, though counting in continuous state spaces requires approximations like
            hash functions or density models."""), unsafe_allow_html=True)

        st.markdown("**UCB action selection formula:**")
        st.latex(r"a_t = \arg\max_a \underbrace{Q_t(a)}_{\text{exploitation}} + \underbrace{c\sqrt{\frac{\ln t}{N_t(a)}}}_{\text{exploration bonus}}")
        st.markdown(r"""
        **Symbol decoder:**
        - $Q_t(a)$: sample mean reward for arm $a$ after $t$ steps — the exploitation term
        - $N_t(a)$: number of times arm $a$ has been pulled — fewer pulls = larger bonus
        - $c$: exploration constant (typical 1–2); higher = more exploration
        - $\sqrt{\ln t / N_t(a)}$: width of the confidence interval from Chernoff bounds
        - The bonus decreases as $N_t(a)$ grows and the arm becomes well-estimated
        """)
        st.markdown("**Regret guarantee:**")
        st.latex(r"\mathbb{E}[\text{Regret}(T)] \leq \sum_{a:\Delta_a>0}\frac{8c^2\ln T}{\Delta_a} + \left(1+\frac{\pi^2}{3}\right)\sum_{a:\Delta_a>0}\Delta_a")
        st.markdown(r"Where $\Delta_a = \mu^* - \mu_a$ is the gap between the optimal and arm $a$'s mean. This is $O(\log T)$.")

        c_ucb = st.slider("Exploration constant c", 0.1, 5.0, 2.0, 0.1, key="ucb_c")
        rw_ucb, reg_ucb = run_bandit(10, 500, "ucb", c=c_ucb)
        fig_ucb, ax_ucb = _fig(1, 1, 10, 3.5)
        ax_ucb.plot(np.cumsum(reg_ucb), color="#0288d1", lw=2.5, label=f"UCB (c={c_ucb:.1f})")
        ax_ucb.set_xlabel("Step", color="white"); ax_ucb.set_ylabel("Cumulative regret", color="white")
        ax_ucb.set_title(f"UCB Cumulative Regret (c={c_ucb})", color="white", fontweight="bold")
        ax_ucb.legend(facecolor=CARD, labelcolor="white"); ax_ucb.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ucb); plt.close()

    with tab_ts:
        st.subheader("🎲 Thompson Sampling — Bayesian Exploration")
        st.markdown(_card("#7c4dff","🎲","Thompson Sampling: exploration via Bayesian uncertainty",
            """Thompson Sampling is a Bayesian approach to exploration that maintains a probability
            distribution over the true mean reward of each arm and samples from these distributions
            to decide which arm to pull. The posterior naturally encodes uncertainty: arms tried few
            times have wide posteriors (high uncertainty), arms tried many times have narrow posteriors
            (low uncertainty). At each step: sample one value from each arm's posterior distribution,
            then pull the arm with the highest sample. This is called 'probability matching': the
            probability of selecting arm a equals the probability that it is actually the best arm.
            The key advantage over UCB: Thompson Sampling is naturally Bayesian and handles non-stationary
            bandits, contextual bandits, and partial information settings more gracefully. It is also
            empirically superior to UCB in many practical settings. In deep RL, Thompson Sampling
            corresponds to maintaining a distribution over neural network parameters (using Bayesian
            neural networks or ensembles) and sampling a network at each episode to use as the policy
            — the sampled network drives exploration by behaving as if its uncertain estimates are true.
            Ensemble-based Thompson Sampling is now used in practical deep RL systems."""), unsafe_allow_html=True)

        st.markdown("**Thompson Sampling update (Beta-Bernoulli conjugate model):**")
        st.latex(r"\text{Prior: } \theta_a \sim \text{Beta}(\alpha_a, \beta_a) \quad \text{(init: } \alpha=\beta=1 \text{ — uniform)}")
        st.latex(r"\text{At step } t: \;\tilde\theta_a \sim \text{Beta}(\alpha_a,\beta_a), \quad a_t = \arg\max_a\tilde\theta_a")
        st.latex(r"\text{Update after reward } r: \quad\alpha_{a_t}\mathrel{+}= r, \quad\beta_{a_t}\mathrel{+}= 1-r")
        st.markdown(r"""
        For Gaussian rewards: maintain posterior $\mathcal{N}(\mu_a, \sigma_a^2)$ and update with normal-normal conjugacy.
        Each arm's posterior concentrates around its true mean as more data arrives.
        """)

    with tab_im:
        st.subheader("🧠 Intrinsic Motivation — Exploration as Self-Reward")
        st.markdown(_card("#00897b","🧠","Why epsilon-greedy fails in sparse reward environments",
            """In deep RL with sparse rewards, simple exploration strategies like epsilon-greedy
            completely fail. Consider Montezuma's Revenge (Atari): a human player can explore and
            find rewards by understanding the game structure (get the key, open the door).
            A random explorer finds the first reward only after millions of steps by pure luck.
            The solution: give the agent intrinsic rewards for exploring novel states, regardless
            of the extrinsic (environment) reward. An agent that is intrinsically curious will
            naturally visit new states, make surprising transitions, and build a richer understanding
            of the environment — which makes it more likely to encounter the sparse extrinsic reward.
            Intrinsic motivation adds a bonus reward r_intrinsic to the agent's reward signal:
            total reward = r_extrinsic + beta * r_intrinsic. The intrinsic reward is high for
            surprising or novel transitions (the agent predicts badly) and low for familiar ones.
            Two main approaches: ICM (Intrinsic Curiosity Module) measures surprise via prediction error;
            RND (Random Network Distillation) measures novelty via prediction error of a random target."""), unsafe_allow_html=True)

        st.latex(r"r_{\text{total}} = r_{\text{extrinsic}} + \beta\cdot r_{\text{intrinsic}}")
        st.markdown(r"""
        - $r_{\text{extrinsic}}$ — the actual environment reward (sparse, might be zero for thousands of steps)
        - $r_{\text{intrinsic}}$ — self-generated bonus for novelty/surprise
        - $\beta$ — balance weight (typical: 0.01–1.0)
        - The agent maximises the total reward, so it is driven to explore even without extrinsic reward
        """)

    with tab_icm:
        st.subheader("🔭 ICM — Intrinsic Curiosity Module (Pathak et al. 2017)")
        st.markdown(_card("#e65100","🔭","ICM: curiosity as prediction error — be rewarded for surprising yourself",
            """Intrinsic Curiosity Module (ICM, Pathak et al. 2017) operationalises curiosity as the
            agent's surprise at its own transition predictions. The key insight: if the agent can already
            predict what will happen next, there is nothing to learn there. If it is surprised (prediction
            error is high), there is something worth exploring. ICM uses a forward model that predicts
            the next state's features given the current state's features and the taken action. The
            prediction error is the intrinsic reward — states that the agent cannot predict well yet.
            But there is a subtle problem: if the intrinsic reward is based on raw pixel prediction,
            the agent will become obsessed with TV-like noisy inputs that are inherently unpredictable
            but unrelated to the task (the 'noisy TV problem'). ICM solves this with an inverse model:
            alongside the forward model, train an inverse model that predicts which action was taken
            from the (state, next_state) feature pair. The representation trained by the inverse model
            is only sensitive to aspects of the state that the agent can control — ignoring uncontrollable
            noise. The forward model then operates on these controllable features, making the intrinsic
            reward meaningful rather than dominated by random environmental noise."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**ICM forward model (surprise signal):**")
            st.latex(r"\hat\phi(s_{t+1}) = f(\phi(s_t),\,a_t) \quad \text{(predict next features)}")
            st.latex(r"r_{\text{intr}}^t = \frac{\eta}{2}\|\hat\phi(s_{t+1})-\phi(s_{t+1})\|^2")
        with col2:
            st.markdown("**ICM inverse model (controllable features only):**")
            st.latex(r"\hat a_t = g(\phi(s_t),\,\phi(s_{t+1})) \quad \text{(predict action taken)}")
            st.latex(r"\mathcal{L}_{\text{inv}} = -\log p(\hat a_t = a_t)")
        st.markdown("**Combined ICM loss:**")
        st.latex(r"\mathcal{L}_{\text{ICM}} = (1-\lambda)\mathcal{L}_{\text{inv}} + \lambda\mathcal{L}_{\text{fwd}}")
        st.markdown(r"Where $\lambda\in[0,1]$ balances inverse model accuracy vs. forward model accuracy.")
        st.markdown(_insight("""
        ICM achieved superhuman exploration efficiency on Montezuma's Revenge (the hardest exploration
        Atari game) without ANY extrinsic reward — purely through curiosity. It also enabled successful
        navigation in 3D mazes where sparse rewards made standard RL fail completely.
        """), unsafe_allow_html=True)

    with tab_rnd:
        st.subheader("🎯 RND — Random Network Distillation (Burda et al. 2018)")
        st.markdown(_card("#1565c0","🎯","RND: novelty as error in distilling a random frozen network",
            """Random Network Distillation (RND, Burda et al. 2018) is a simpler and often better
            alternative to ICM for generating exploration bonuses. The idea is beautifully simple:
            initialise two networks — a fixed random target network T(s) and a predictor network
            f_theta(s). Train the predictor to match the target's output on visited states.
            The prediction error ||f_theta(s) - T(s)||^2 is large for states the predictor has
            not trained on (novel states) and small for frequently visited states. This error is
            the intrinsic reward — the agent is rewarded for visiting states where it has not had
            much training data. The key advantages over ICM: (1) no inverse model needed, so no
            risk of representation learning failing; (2) works with raw observations without needing
            a learned feature encoder; (3) the random target T is fixed (not trained), preventing
            the exploration bonus from degrading over time. RND achieved state-of-the-art performance
            on Montezuma's Revenge (400+ rooms explored vs ~20 for PPO alone) and was the first
            algorithm to reliably learn Pitfall (the hardest Atari game). Its simplicity makes it
            easy to add to any existing RL codebase — just train a predictor network alongside the
            main policy and add its error as a bonus reward."""), unsafe_allow_html=True)

        st.markdown("**RND mechanism (extremely simple):**")
        st.latex(r"r_{\text{intrinsic}}(s) = \|f_\theta(s) - T(s)\|^2")
        st.markdown(r"""
        - $T(s)$ — **frozen** random network (randomly initialised, never updated)
        - $f_\theta(s)$ — predictor network (trained to match $T$ on visited states)
        - $r_{\text{intrinsic}}$ is high for states where the predictor hasn't been trained → novel states
        - $r_{\text{intrinsic}}$ is low for frequently visited states → familiar states get less bonus
        """)
        st.code("""
# RND implementation — add these 5 lines to any RL training loop
target_net = RandomNetwork(obs_dim, hidden_dim)  # frozen
predictor_net = PredictorNetwork(obs_dim, hidden_dim)  # trained

for step in training:
    obs = env.step(action)
    with torch.no_grad():
        target_feat = target_net(obs)
    pred_feat = predictor_net(obs)

    # Intrinsic reward: prediction error
    r_int = F.mse_loss(pred_feat, target_feat, reduction='none').mean(-1)

    # Total reward: extrinsic + intrinsic
    r_total = r_extrinsic + beta * r_int

    # Update predictor (target is FROZEN)
    loss_rnd = F.mse_loss(pred_feat, target_feat.detach())
    loss_rnd.backward()
    optimizer_predictor.step()
""", language="python")

        # Visual: novelty bonus decreasing as states are visited
        np.random.seed(42)
        states_visited = np.arange(1, 1001)
        novelty_bonus = 1.0 / np.sqrt(states_visited) + np.random.randn(1000)*0.02
        fig_rnd, ax_rnd = _fig(1, 1, 10, 3.5)
        ax_rnd.plot(states_visited, np.maximum(0, novelty_bonus), color="#1565c0", lw=2)
        ax_rnd.fill_between(states_visited, 0, np.maximum(0, novelty_bonus), alpha=0.2, color="#1565c0")
        ax_rnd.set_xlabel("Times state has been visited", color="white")
        ax_rnd.set_ylabel("RND intrinsic bonus r_int", color="white")
        ax_rnd.set_title("RND bonus decreases as states become familiar", color="white", fontweight="bold")
        ax_rnd.grid(alpha=0.12); plt.tight_layout(); st.pyplot(fig_rnd); plt.close()

    with tab_res:
        st.subheader("📚 Resources")
        for icon, title, desc, url in [
            ("📄","Auer et al. (2002) — UCB","Original UCB paper with O(log T) regret guarantee.","https://link.springer.com/article/10.1023/A:1013689704352"),
            ("📄","Pathak et al. (2017) — ICM","Intrinsic Curiosity Module. Montezuma's Revenge with no extrinsic reward.","https://arxiv.org/abs/1705.05363"),
            ("📄","Burda et al. (2018) — RND","Random Network Distillation. Simple and highly effective novelty bonus.","https://arxiv.org/abs/1810.12894"),
            ("📄","Eysenbach et al. (2019) — DIAYN","Diversity is All You Need — explore by learning diverse skills.","https://arxiv.org/abs/1802.06070"),
            ("📄","Ostrovski et al. (2017) — Count-based","Neural episodic control + count-based bonuses in deep RL.","https://arxiv.org/abs/1703.01310"),
            ("💻","stable-baselines3 + intrinsic rewards","SB3 with curiosity bonus — easiest entry point.","https://github.com/AechPro/stable-baselines3-contrib"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
