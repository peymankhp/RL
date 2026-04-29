"""_advanced_mod.py — Advanced RL Specialisations (Tier 2)"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import warnings
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

def _sec(emoji, title, sub, color="#6a1b9a"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def smooth(a, w=8):
    return np.convolve(a, np.ones(w)/w, mode="valid") if len(a) > w else np.array(a, float)

def main_advanced():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a0a2e,#0a0a1a,#2a0a1a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🚀 Advanced RL Specialisations</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'Tier 2: Multi-agent coordination, long-horizon planning, safe deployment, and fast adaptation. '
        'Full derivations, worked examples with real numbers, charts, and deep-dive resources.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs(["🌐 MARL","🏗️ Hierarchical RL","🛡️ Safe RL","🧬 Meta-RL","📚 Books & Resources"])
    tab_marl, tab_hier, tab_safe, tab_meta, tab_res = tabs

    with tab_marl:
        _sec("🌐","Multi-Agent Reinforcement Learning",
             "MADDPG · QMIX · MAPPO — coordination and competition between multiple learning agents","#0288d1")

        st.markdown(_card("#0288d1","🌐","Why single-agent RL fails in multi-agent settings",
            """In single-agent RL, the environment is stationary: p(s'|s,a) is fixed.
            In multi-agent settings, every other agent updates its policy continuously.
            From agent 1's view, the environment is non-stationary — agent 2's behaviour changes every
            episode, violating the Markov property that TD and Q-learning rely on.
            The joint action space also grows exponentially: N agents × K actions each = K^N combinations.
            For N=5, K=10: 100,000 joint actions — completely intractable for tabular Q-learning.
            MARL algorithms solve this via Centralised Training with Decentralised Execution (CTDE):
            share full information during training (critic sees all agents), act using only local
            observations at deployment (actor sees only own observation).
            This means the training complexity is manageable, and the deployed agents need
            no communication infrastructure."""), unsafe_allow_html=True)

        st.subheader("1. MADDPG — Multi-Agent DDPG with Centralised Critics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Centralised critic** (training only — sees everything):")
            st.latex(r"Q_i^\pi(\mathbf{o},\mathbf{a}) \quad \mathbf{o}=(o_1,\ldots,o_N),\;\mathbf{a}=(a_1,\ldots,a_N)")
            st.markdown("**Decentralised actor** (deployment — local only):")
            st.latex(r"\pi_i(a_i|o_i) \quad\text{(acts without seeing other agents)}")
            st.markdown("**Actor update:**")
            st.latex(r"\nabla_{\theta_i}J = \mathbb{E}\!\left[\nabla_{\theta_i}\log\pi_i(a_i|o_i)\,Q_i(\mathbf{o},\mathbf{a})\right]")
            st.markdown("**Practical example:** 3 agents cooperating to push a box. Critic for agent 1 sees all 3 positions and all 3 intended forces. Actor 1 only sees its own position and pushes based on that.")
        with col2:
            np.random.seed(42)
            steps = np.arange(500)
            ind = smooth(-20 + steps*0.05 + np.random.randn(500)*4, 30)
            maddpg = smooth(-20 + steps*0.09 + np.random.randn(500)*3, 30)
            qmix_c = smooth(-20 + steps*0.12 + np.random.randn(500)*2.5, 30)
            mappo_c = smooth(-20 + steps*0.13 + np.random.randn(500)*3, 30)
            fig_ma, ax_ma = _fig(1,1,5.5,4)
            for curve, nm, col in [
                (ind,"Independent RL","#ef5350"),(maddpg,"MADDPG","#ffa726"),
                (qmix_c,"QMIX","#7c4dff"),(mappo_c,"MAPPO","#0288d1")]:
                ax_ma.plot(range(len(curve)), curve, lw=2, label=nm, color=col)
            ax_ma.set_xlabel("Episode",color="white"); ax_ma.set_ylabel("Team reward",color="white")
            ax_ma.set_title("MARL Algorithms\n(Cooperative task, 3 agents)",color="white",fontweight="bold")
            ax_ma.legend(facecolor=CARD,labelcolor="white",fontsize=7); ax_ma.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_ma); plt.close()

        st.subheader("2. QMIX — Value Decomposition for Cooperative Tasks")
        st.markdown("QMIX decomposes Q_tot(s, a) into per-agent Q_i(o_i, a_i) via a mixing network that ensures monotonicity:")
        st.latex(r"\frac{\partial Q_\text{tot}}{\partial Q_i} \geq 0 \quad\forall i \quad\text{(IGM property: global argmax = local argmaxes)}")
        st.markdown("This means each agent can independently maximise its own Q_i to get the globally optimal joint action. The mixing network weights are state-dependent: W_k = |f_k(s)| (absolute value enforces positivity).")
        st.markdown("**Concrete example:** 3 marines vs 6 enemy marines (SMAC 3m_vs_6m). Without coordination: agents all attack same enemy, leaving others unharmed. With QMIX: agents learn to focus fire — different marines attack different enemies, maximising damage per step.")
        st.dataframe(pd.DataFrame({
            "Task": ["SMAC 3m","SMAC 5m_vs_6m","SMAC MMM2"],
            "Independent PPO": ["71%","54%","42%"],
            "MADDPG": ["52%","35%","18%"],
            "QMIX": ["89%","78%","72%"],
            "MAPPO": ["91%","81%","75%"],
        }), use_container_width=True, hide_index=True)

        st.subheader("3. MAPPO — Multi-Agent PPO")
        st.markdown("The simplest and often strongest MARL baseline: standard PPO where each agent's critic sees the global state. The centralised value function provides a stable baseline that accounts for all agents' contributions.")
        st.latex(r"V_\phi(s_t) \quad\text{(centralised — sees global state, all observations)}")
        st.latex(r"\hat A_t^i = R_t^i - V_\phi(s_t) \quad\text{(advantage per agent using global baseline)}")
        st.markdown(_insight("When to use what: QMIX for discrete cooperative tasks where credit assignment is hard (StarCraft). MADDPG for continuous action cooperative/competitive tasks. MAPPO as the default — simpler, often competitive with QMIX."), unsafe_allow_html=True)

        # Action space explosion chart
        K_vals = [2,5,10,18]; N_vals = range(1,9)
        fig_as, ax_as = _fig(1,1,10,4)
        for K, col in zip(K_vals,["#4caf50","#ffa726","#ef5350","#ad1457"]):
            ax_as.semilogy([n for n in N_vals], [K**n for n in N_vals],
                          lw=2.5, marker="o", ms=5, label=f"K={K} actions/agent", color=col)
        ax_as.axhline(1e4, color="white", ls="--", lw=1, alpha=0.5, label="10K (DQN limit)")
        ax_as.set_xlabel("Number of agents N", color="white")
        ax_as.set_ylabel("Joint action space size K^N", color="white")
        ax_as.set_title("Why joint action space is intractable: K^N explosion\nValue decomposition (QMIX) avoids this", color="white", fontweight="bold")
        ax_as.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_as.grid(alpha=0.12, which="both")
        plt.tight_layout(); st.pyplot(fig_as); plt.close()

    with tab_hier:
        _sec("🏗️","Hierarchical Reinforcement Learning",
             "Options · HER · Feudal Networks — solving long-horizon tasks via temporal abstraction","#e65100")

        st.markdown(_card("#e65100","🏗️","Why flat policies fail on long-horizon tasks",
            """Consider a robot navigating: kitchen → hallway → bedroom → pick up glasses → return.
            A flat policy gets reward only on task completion — after potentially 1000+ steps.
            Probability of random completion: (1/|A|)^1000 ≈ 10^{-1301}. Never happens.
            Hierarchical RL solves this by decomposing into subgoals.
            A high-level manager sets subgoal every H=10 steps: "reach hallway."
            A low-level worker gets dense intrinsic reward for reaching that subgoal.
            Two separate, tractable problems replace one intractable one.
            The temporal abstraction is key: manager plans at coarse timescale (10s of steps),
            worker acts at fine timescale (individual actions). This is how humans solve complex tasks:
            think "go to supermarket" not "move left foot 0.5m, lift right foot..."."""), unsafe_allow_html=True)

        st.subheader("1. The Options Framework")
        st.markdown("An **option** ω = (I_ω, π_ω, β_ω) is a temporally extended action:")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"I_\omega \subseteq \mathcal{S} \quad\text{(initiation set: where option can start)}")
            st.latex(r"\pi_\omega(a|s) \quad\text{(intra-option policy: what to do while running)}")
            st.latex(r"\beta_\omega(s)\in[0,1] \quad\text{(termination probability: when to stop)}")
            st.markdown("**Option-value function:**")
            st.latex(r"Q_\Omega(s,\omega) = \sum_a\pi_\omega(a|s)\left[r(s,a)+\gamma\sum_{s'}p(s'|s,a)U(s',\omega)\right]")
            st.latex(r"U(s',\omega) = (1-\beta_\omega(s'))Q_\Omega(s',\omega)+\beta_\omega(s')V_\Omega(s')")
        with col2:
            # Flat vs hierarchical reward comparison
            np.random.seed(42); T=800
            flat = np.concatenate([np.zeros(700), np.linspace(0, 8, 100)]) + np.random.randn(T)*0.3
            hier = np.minimum(10, 0.012*np.arange(T) + np.random.randn(T)*1.2)
            fig_hc, ax_hc = _fig(1,1,5.5,4.5)
            ax_hc.plot(smooth(flat,30), color="#ef5350", lw=2.5, label="Flat PPO (sparse)")
            ax_hc.plot(smooth(hier,30), color="#4caf50", lw=2.5, label="HRL (dense subgoals)")
            ax_hc.set_xlabel("Episode",color="white"); ax_hc.set_ylabel("Episode reward",color="white")
            ax_hc.set_title("Flat vs Hierarchical:\nSparse long-horizon task",color="white",fontweight="bold")
            ax_hc.legend(facecolor=CARD,labelcolor="white",fontsize=8); ax_hc.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_hc); plt.close()

        st.subheader("2. Hindsight Experience Replay (HER)")
        st.markdown("**The key insight:** even if we failed the goal, we can relabel the trajectory as if we intended to reach the state we actually reached.")
        st.latex(r"\text{Failed: }(s_0,a_0,\ldots,s_T)\text{ toward goal }g\;\Rightarrow\;\text{Relabel: goal}=s_T\text{ (success!)}")
        st.code("""
def her_replay(trajectory, goal, replay_buffer, n_relabel=4):
    # Add original (failed) trajectory
    for t, (s, a, r, s2, done) in enumerate(trajectory):
        replay_buffer.add(s, a, r, s2, done, goal)
    # HER: pick n_relabel future states as alternative goals
    for _ in range(n_relabel):
        t_future = np.random.randint(0, len(trajectory))
        goal_her = trajectory[t_future][3][:3]  # achieved position
        for t, (s, a, _, s2, _) in enumerate(trajectory):
            # Success if within 5cm of the relabelled goal
            r_her = 1.0 if np.linalg.norm(s2[:3]-goal_her) < 0.05 else 0.0
            replay_buffer.add(s, a, r_her, s2, r_her>0, goal_her)
""", language="python")

        st.markdown("**Empirical impact:** FetchPush robotics task — SAC alone: 5% success rate. HER+SAC: 90% success. The relabelling creates ~4× more useful training data from the same interactions.")

        # Show temporal abstraction
        np.random.seed(42)
        T=80; H=10
        manager_goals = np.repeat(np.arange(T//H)*2, H)[:T]
        worker_pos = np.cumsum(np.random.choice([-0.3,0.1,0.4], T))+2
        fig_ta, ax_ta = _fig(1,1,10,3.5)
        ax_ta.step(range(T), manager_goals, color="#ffa726", lw=2.5, where="post", label=f"Manager subgoal (every H={H} steps)")
        ax_ta.plot(range(T), worker_pos, color="#0288d1", lw=1.5, alpha=0.8, label="Worker position (every step)")
        for i in range(0, T, H):
            ax_ta.axvline(i, color="#2a2a3e", lw=1, alpha=0.5)
        ax_ta.set_xlabel("Primitive step t", color="white"); ax_ta.set_ylabel("Position", color="white")
        ax_ta.set_title("Temporal Abstraction: Manager updates every H=10 steps, Worker acts every step", color="white", fontweight="bold")
        ax_ta.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_ta.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ta); plt.close()

    with tab_safe:
        _sec("🛡️","Safe Reinforcement Learning",
             "CMDP · Lagrangian PPO · CPO · CBF — constraints during training AND deployment","#ef5350")

        st.markdown(_card("#ef5350","🛡️","Two distinct safety problems in RL",
            """(1) Training-time safety: the policy must not violate constraints while exploring.
            An autonomous vehicle cannot crash 1000 times while learning. A robot arm cannot break
            itself during training. Every constraint violation has real cost.
            (2) Deployment-time safety: the final policy must satisfy constraints on all future
            inputs, including distribution shift from training. The constraint must hold at convergence
            AND generalize to unseen states.
            The fundamental tension: exploration REQUIRES trying uncertain actions, but safety REQUIRES
            avoiding potentially-unsafe actions. These goals are directly opposed.
            Solutions:
            Lagrangian PPO — soft constraint via penalty (asymptotically safe, not during training).
            CPO — trust-region update that satisfies constraint at each update step.
            CBF shielding — hard filter that mathematically guarantees safety at every step,
            at the cost of some reward efficiency."""), unsafe_allow_html=True)

        st.subheader("1. CMDP — Constrained MDP Formulation")
        st.latex(r"\max_\pi J(\pi) = \mathbb{E}_\pi\!\left[\sum_t\gamma^t r_t\right] \quad\text{s.t.}\quad J_C(\pi) = \mathbb{E}_\pi\!\left[\sum_t\gamma^t c_t\right] \leq d")
        st.markdown("**Lagrangian relaxation:**")
        st.latex(r"\mathcal{L}(\pi,\lambda) = J(\pi) - \lambda(J_C(\pi)-d)")
        st.latex(r"\lambda_{t+1} = \max(0,\;\lambda_t + \eta_\lambda(J_C(\pi_t)-d))")
        st.markdown("λ increases when constraint violated → policy penalised more. λ decreases when satisfied → policy less restricted. This creates a soft but adaptive safety enforcement.")

        # Training dynamics comparison
        np.random.seed(42); T=300
        r_unsafe = smooth(np.minimum(15, 0.06*np.arange(T)) + np.random.randn(T)*2, 20)
        c_unsafe = smooth(np.abs(np.random.randn(T)*2.5 + 2.5), 15)
        r_lag = smooth(np.minimum(10, 0.04*np.arange(T)) + np.random.randn(T)*2, 20)
        c_lag = smooth(np.maximum(0, np.abs(np.random.randn(T)*1.5) - 0.5*np.arange(T)/T), 15)
        fig_s, axes_s = _fig(2, 2, 14, 7)
        for ax, data, title, col in [
            (axes_s[0,0], r_unsafe, "Unconstrained PPO: Reward", "#4caf50"),
            (axes_s[0,1], c_unsafe, "Unconstrained PPO: Safety Cost", "#ef5350"),
            (axes_s[1,0], r_lag,   "Lagrangian PPO: Reward",  "#4caf50"),
            (axes_s[1,1], c_lag,   "Lagrangian PPO: Safety Cost","#ffa726"),
        ]:
            ax.plot(data, color=col, lw=2); ax.grid(alpha=0.12)
            ax.set_xlabel("Episode",color="white"); ax.set_title(title,color="white",fontweight="bold")
        for ax in [axes_s[0,1], axes_s[1,1]]:
            ax.axhline(1.0, color="white", ls="--", lw=1.5, label="Limit d=1.0")
            ax.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        plt.suptitle("Lagrangian PPO vs Unconstrained: reward vs safety cost tradeoff", color="white", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_s); plt.close()

        st.subheader("2. CPO — Constrained Policy Optimisation")
        st.markdown("CPO extends TRPO to also constrain the cost within each trust region update:")
        st.latex(r"\max_\theta\;g^T(\theta-\theta_k) \quad\text{s.t.}\quad \tfrac{1}{2}(\theta-\theta_k)^TF(\theta-\theta_k)\leq\delta,\;b^T(\theta-\theta_k)\leq d-J_C(\pi_k)")
        st.markdown("Each update stays within the KL trust region AND satisfies the cost constraint. This gives deployment-time guarantees if the linear approximation is accurate enough.")

        st.subheader("3. CBF — Control Barrier Functions for Training-Time Safety")
        st.latex(r"h(s)\geq 0\;\forall s\in\mathcal{C}_\text{safe} \quad\Rightarrow\quad \dot h(s,a)+\alpha h(s)\geq 0\;\text{(forward invariance)}")
        st.markdown("**Safety shield (QP at every step):**")
        st.latex(r"a^* = \arg\min_{a'}\|a'-a\|^2 \quad\text{s.t.}\quad \nabla_s h(s)^T f(s,a')+\alpha h(s)\geq 0")
        st.markdown("**Practical example:** Robot arm workspace. h(s) = d_max - ||p||. CBF condition: -ṗ·(p/||p||) + α(d_max - ||p||) ≥ 0. Automatically decelerates as arm approaches boundary. Zero constraint violations during training.")
        st.dataframe(pd.DataFrame({
            "Method":["Unconstrained PPO","Lagrangian PPO","CPO","CBF + PPO"],
            "Training violations":["~8000","~2000","~500","0 (guaranteed)"],
            "Deployment violations":["High","Low","Low","Near-zero"],
            "Final reward":["100% (baseline)","85%","88%","92%"],
            "Computational cost":["1×","1.1×","3× (TRPO)","1.05× (QP fast)"],
        }), use_container_width=True, hide_index=True)
        st.markdown(_insight("Best current practice: CBF shield (hard training-time guarantee) + CMDP/Lagrangian (asymptotic deployment guarantee). Both are needed for real-world safe deployment."), unsafe_allow_html=True)

    with tab_meta:
        _sec("🧬","Meta-Reinforcement Learning",
             "MAML · RL² · PEARL — learn to learn: adapt to new tasks in 5–10 steps","#00897b")

        st.markdown(_card("#00897b","🧬","What meta-RL is and why it's fundamentally different",
            """Standard RL learns a single optimal policy for a single fixed task.
            Meta-RL learns a meta-policy that can RAPIDLY ADAPT to NEW tasks after seeing only
            a few episodes — without retraining from scratch.
            The meta-learner sees many training tasks during meta-training and learns the ability
            to adapt quickly, not any specific task solution.
            Analogy: standard RL = learning to play one specific video game from scratch.
            Meta-RL = learning HOW to learn new video games quickly (the learning algorithm itself).
            After meta-training on 500 maze layouts, a meta-RL agent can navigate a new maze in
            3–5 episodes, compared to 200+ episodes for standard RL on the same maze.
            This is the difference that matters for robotics (new objects), medicine (new patients),
            and personalisation (new users)."""), unsafe_allow_html=True)

        st.subheader("1. MAML — Model-Agnostic Meta-Learning")
        st.markdown("**Objective:** find parameter initialisation θ such that K gradient steps on any task τ yields good performance.")
        st.latex(r"\theta^* = \arg\min_\theta \sum_{\tau\sim p(\mathcal{T})}\mathcal{L}_\tau\!\left(\theta - \alpha\nabla_\theta\mathcal{L}_\tau(\theta)\right)")
        st.markdown("**Inner loop** (task adaptation, K=1-5 gradient steps):")
        st.latex(r"\theta'_\tau = \theta - \alpha\nabla_\theta\mathcal{L}_\tau(\theta)")
        st.markdown("**Outer loop** (meta-update across all tasks):")
        st.latex(r"\theta \leftarrow \theta - \beta\nabla_\theta\sum_\tau\mathcal{L}_\tau(\theta'_\tau)")

        np.random.seed(42); K = np.arange(1, 51)
        standard = 100*(1 - np.exp(-K/20)) + np.random.randn(50)*3
        maml_curve = 100*(1 - np.exp(-K/3.5)) + np.random.randn(50)*3
        maml_curve = np.clip(maml_curve, 0, 105)
        fig_mm, axes_mm = _fig(1,2,13,4.5)
        axes_mm[0].plot(K, standard, color="#ef5350", lw=2.5, label="Standard RL (random init)")
        axes_mm[0].plot(K, maml_curve, color="#00897b", lw=2.5, label="MAML (meta-init)")
        axes_mm[0].axvline(5, color="#ffa726", ls="--", lw=1.5, alpha=0.7, label="5 gradient steps")
        axes_mm[0].fill_between(K[:5], standard[:5], maml_curve[:5], alpha=0.25, color="#00897b")
        axes_mm[0].set_xlabel("Gradient steps on new task",color="white")
        axes_mm[0].set_ylabel("Task performance (%)",color="white")
        axes_mm[0].set_title("MAML: 5 gradient steps ≈ Standard RL 30+ steps\n(Meta-init from 500 training tasks)",color="white",fontweight="bold")
        axes_mm[0].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_mm[0].grid(alpha=0.12)

        n_meta = 200; meta_scores = smooth(50 + 0.25*np.arange(n_meta) + np.random.randn(n_meta)*8, 20)
        axes_mm[1].plot(range(len(meta_scores)), meta_scores, color="#00897b", lw=2.5)
        axes_mm[1].fill_between(range(len(meta_scores)), meta_scores-5, meta_scores+5, alpha=0.15, color="#00897b")
        axes_mm[1].set_xlabel("Meta-training iteration",color="white")
        axes_mm[1].set_ylabel("5-shot adaptation score",color="white")
        axes_mm[1].set_title("MAML Outer Loop Progress:\nImproving 5-shot score across meta-training",color="white",fontweight="bold")
        axes_mm[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_mm); plt.close()

        st.subheader("2. RL² — Recurrent Meta-RL (No Gradient at Test Time)")
        st.markdown("RL² uses an LSTM whose hidden state accumulates experience within a trial. No gradient updates at test time — adaptation happens in the hidden state.")
        st.latex(r"h_t = \text{LSTM}(h_{t-1},\;[s_t,\;a_{t-1},\;r_{t-1},\;d_{t-1}])")
        st.latex(r"\pi(a_t|h_t) \quad\text{(policy conditions on full accumulated experience)}")
        st.markdown("**Practical example — new maze navigation:**")
        st.markdown("Episode 1: LSTM hidden state = random → explores. Episode 2: hidden state encodes walls discovered → improves routing. Episode 3: hidden state fully encodes maze layout → navigates near-optimally. No gradient steps. All adaptation is in h_t.")

        st.dataframe(pd.DataFrame({
            "Method":["Standard RL (scratch)","Fine-tune from checkpoint","MAML (5 steps)","RL²","PEARL"],
            "Adaptation mechanism":["Gradient descent from random","Gradient descent from checkpoint",
                                    "K gradient steps from meta-init","LSTM hidden state (no gradient)",
                                    "Posterior sampling, context encoder"],
            "Steps to 90% performance":["500+","50–100","5–10","10–20 (within trial)","5–10"],
            "Gradient at test?":["Yes","Yes","Yes (K)","No","No"],
        }), use_container_width=True, hide_index=True)

    with tab_res:
        _sec("📚","Books & Deep-Dive Resources",
             "The best books, papers, and courses for each Advanced RL specialisation","#546e7a")

        for section, items in [
            ("🌐 Multi-Agent RL", [
                ("Multi-Agent RL: Foundations and Modern Approaches",
                 "Albrecht, Christianos, Schäfer (2024) — MIT Press — FREE at marl-book.com",
                 "THE textbook for MARL. Full theory + MADDPG, QMIX, MAPPO derivations.",
                 "https://marl-book.com"),
                ("QMIX: Monotonic Value Function Factorisation",
                 "Rashid et al. (2018) — ICML — The QMIX paper",
                 "Clear derivation of the IGM property and mixing network. 4000+ citations.",
                 "https://arxiv.org/abs/1803.11605"),
                ("The StarCraft Multi-Agent Challenge (SMAC)",
                 "Samvelyan et al. (2019) — The standard MARL benchmark",
                 "Read to understand the benchmark every MARL paper uses.",
                 "https://arxiv.org/abs/1902.04043"),
            ]),
            ("🏗️ Hierarchical RL", [
                ("Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction",
                 "Sutton, Precup, Singh (1999) — Artificial Intelligence Journal",
                 "The original Options paper. All modern HRL builds on this.",
                 "https://www.sciencedirect.com/science/article/pii/S0004370299000521"),
                ("Hindsight Experience Replay",
                 "Andrychowicz et al. (2017) — NeurIPS — The HER paper",
                 "4 pages, simple idea, massive impact. Robotics manipulation went from impossible to solved.",
                 "https://arxiv.org/abs/1707.01495"),
                ("Data-Efficient Hierarchical RL (HIRO)",
                 "Nachum et al. (2018) — NeurIPS",
                 "Best practical HRL paper. Goal-conditioned with off-policy corrections.",
                 "https://arxiv.org/abs/1805.08296"),
            ]),
            ("🛡️ Safe RL", [
                ("Constrained Policy Optimisation (CPO)",
                 "Achiam et al. (2017) — ICML — The standard CMDP algorithm",
                 "Derives CPO from TRPO trust region. Appendix has full theoretical proof.",
                 "https://arxiv.org/abs/1705.10528"),
                ("Safety Gym — Benchmarks for Safe RL",
                 "Ray, Achiam, Amodei (OpenAI, 2019)",
                 "THE safe RL benchmark. Implements CPO, Lagrangian-TRPO, Lagrangian-PPO.",
                 "https://openai.com/research/safety-gym"),
                ("A Comprehensive Survey on Safe RL",
                 "García & Fernández (2015) — JMLR",
                 "Survey of all safe RL approaches. Good for theoretical foundations.",
                 "https://jmlr.org/papers/v16/garcia15a.html"),
            ]),
            ("🧬 Meta-RL", [
                ("Model-Agnostic Meta-Learning (MAML)",
                 "Finn, Abbeel, Levine (2017) — ICML — The foundational meta-learning paper",
                 "Only 9 pages. Clean bi-level optimisation derivation. Read before anything else.",
                 "https://arxiv.org/abs/1703.03400"),
                ("RL²: Fast RL via Slow RL",
                 "Duan et al. (2016) — Shows LSTM implements RL in its activations",
                 "Demonstrates that an LSTM trained with PPO implements Thompson Sampling in its hidden state.",
                 "https://arxiv.org/abs/1611.02779"),
                ("PEARL: Efficient Off-Policy Meta-RL",
                 "Rakelly et al. (2019) — ICML — Best practical meta-RL",
                 "Combines VAE task inference with SAC. State of the art on MuJoCo meta-tasks.",
                 "https://arxiv.org/abs/1903.08254"),
            ]),
        ]:
            st.subheader(section)
            for t, a, w, u in items:
                st.markdown(_book(t, a, w, u), unsafe_allow_html=True)

        st.subheader("🎓 Courses & Lectures")
        for icon, title, desc, url in [
            ("🎥","CS285 Berkeley — Lectures 16–19",
             "Levine covers Meta-RL, MARL, Safe RL, Hierarchical RL. Best video resource available.",
             "https://rail.eecs.berkeley.edu/deeprlcourse/"),
            ("🎥","Chelsea Finn — Meta-Learning Tutorial (ICLR 2020)",
             "1-hour video covering MAML, RL², PEARL with clear intuition and code.",
             "https://www.youtube.com/watch?v=0rZtSwNOTQo"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
