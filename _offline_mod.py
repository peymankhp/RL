"""
_offline_mod.py — Offline / Batch Reinforcement Learning
Covers: Why Offline RL, CQL, IQL, Decision Transformer, TD3+BC
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

def main_offline():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0a2a1a,#1a0a2e);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">📦 Offline / Batch Reinforcement Learning</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'Learn optimal policies from fixed datasets — no environment interaction during training. '
        'Essential for healthcare, robotics, and any domain where live exploration is expensive or dangerous.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "❓ Why Offline RL?",
        "🚫 CQL",
        "🎯 IQL",
        "🤖 Decision Transformer",
        "🔁 TD3+BC",
        "📊 Comparison",
    ])
    (tab_why, tab_cql, tab_iql, tab_dt, tab_td3bc, tab_cmp) = tabs

    with tab_why:
        st.subheader("❓ Why Offline RL? The Fundamental Motivation")
        st.markdown(_card("#00897b","🏥","The healthcare motivation — why online RL is often impossible",
            """In online RL, the agent explores the environment by trying actions and observing results.
            This works well for video games where the worst outcome is losing points. But consider
            deploying an RL agent to optimise sepsis treatment in an ICU: every suboptimal action
            could harm or kill a patient. You cannot let an agent explore freely in a hospital.
            However, hospitals have massive historical datasets: millions of patient trajectories
            showing what doctors did, what the outcomes were, and which treatments worked.
            Offline RL (also called batch RL) learns from exactly this kind of fixed historical
            dataset — without ever interacting with the real environment during training.
            The same principle applies to autonomous driving (crash data is expensive), industrial
            control (factory shutdowns are costly), and robotics (hardware wear and safety constraints).
            Offline RL is the bridge between the vast amounts of historical data organisations collect
            and actually deployable AI policies. The fundamental challenge is distributional shift:
            the learned policy may want to take actions that were never taken by the data-collection
            policy, making Q-value estimates unreliable in those unexplored regions."""), unsafe_allow_html=True)

        st.markdown(r"""
        **The Offline RL problem statement:**
        Given a fixed dataset $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$ collected by some
        unknown behaviour policy $\beta(a|s)$, learn the best possible policy $\pi_\theta$
        without any further environment interaction.
        """)
        st.latex(r"\mathcal{D} = \{(s_i,a_i,r_i,s'_i)\}_{i=1}^N \quad \text{collected by } \beta(a|s)")
        st.markdown(r"""
        **The core challenge — distributional shift and extrapolation error:**
        Standard Q-learning tries to maximise $Q(s,a)$ over all possible actions.
        But the dataset only covers states and actions that $\beta$ visited.
        Actions NOT in the dataset have no reliable Q-value estimates — the network
        will confidently extrapolate incorrect (often overestimated) Q-values, causing
        the policy to catastrophically select unseen, possibly dangerous actions.
        """)
        st.latex(r"\hat Q(s,a) \gg Q^*(s,a) \quad \text{for } (s,a) \notin \text{support}(\mathcal{D})")
        st.markdown(_insight("""
        The key insight of all offline RL algorithms: constrain the learned policy to stay close to
        the behaviour policy that generated the dataset. Approaches: (1) penalise Q-values for
        out-of-distribution actions (CQL); (2) learn a conservative value function (IQL);
        (3) directly constrain the policy to dataset actions (TD3+BC, BC);
        (4) reframe as sequence modelling (Decision Transformer).
        """), unsafe_allow_html=True)

        # Visualise extrapolation error
        np.random.seed(42)
        x_data = np.sort(np.random.uniform(-1, 1, 20))
        y_data = np.sin(x_data) + np.random.randn(20)*0.1
        x_full = np.linspace(-3, 3, 200)
        # Overfit polynomial
        coeffs = np.polyfit(x_data, y_data, 8)
        y_fit = np.polyval(coeffs, x_full)
        fig_ext, ax_ext = _fig(1, 1, 10, 4)
        ax_ext.scatter(x_data, y_data, color="#ffa726", s=60, zorder=5, label="Dataset (behaviour policy)")
        ax_ext.plot(x_full, np.sin(x_full), color="#4caf50", lw=2, ls="--", label="True Q* (unknown)")
        ax_ext.plot(x_full, np.clip(y_fit, -5, 5), color="#ef5350", lw=2, label="Learned Q (extrapolates badly)")
        ax_ext.axvspan(-3, -1, alpha=0.08, color="#ef5350", label="Out-of-distribution (unseen actions)")
        ax_ext.axvspan(1, 3, alpha=0.08, color="#ef5350")
        ax_ext.set_xlabel("Action a", color="white"); ax_ext.set_ylabel("Q(s, a)", color="white")
        ax_ext.set_title("Extrapolation Error: Q-function overestimates in unseen action regions",
                         color="white", fontweight="bold")
        ax_ext.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_ext.grid(alpha=0.12)
        ax_ext.set_xlim(-3, 3); ax_ext.set_ylim(-3, 3)
        plt.tight_layout(); st.pyplot(fig_ext); plt.close()

    with tab_cql:
        st.subheader("🚫 CQL — Conservative Q-Learning (Kumar et al. 2020)")
        st.markdown(_card("#1565c0","🚫","What CQL does and why conservative Q-values solve distributional shift",
            """Conservative Q-Learning adds a penalty term to the standard Q-learning objective that
            explicitly pushes down Q-values for out-of-distribution (OOD) actions and pushes up
            Q-values for actions that appear in the dataset. This produces a Q-function that is
            conservative (a lower bound) for actions the agent has not tried, making it safe for
            the policy to greedily maximise — because even if OOD actions have high true Q-values,
            CQL's conservative estimate will be lower, so the policy prefers in-distribution actions.
            The penalty works without explicitly identifying which actions are OOD: it penalises actions
            sampled from the current policy (which will include OOD actions when Q is overoptimistic),
            and rewards actions from the dataset. This self-correcting mechanism stabilises training.
            CQL achieves state-of-the-art results on the D4RL offline benchmark suite and has been
            used in real robotic manipulation tasks. The regularisation strength alpha controls the
            conservatism: higher alpha means more conservative Q-values, reducing OOD risks at the
            cost of potentially underperforming on in-distribution data."""), unsafe_allow_html=True)

        st.markdown("**CQL modified Q-learning objective:**")
        st.latex(r"\min_Q\;\underbrace{\alpha\Bigl(\mathbb{E}_{s,a\sim\pi}[Q(s,a)] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]\Bigr)}_{\text{CQL penalty}} + \underbrace{\frac{1}{2}\mathbb{E}_{(s,a,s')\sim\mathcal{D}}\!\left[(Q(s,a)-y)^2\right]}_{\text{standard Bellman error}}")
        st.markdown(r"""
        **Symbol decoder:**
        - $\mathbb{E}_{a\sim\pi}[Q(s,a)]$: soft-max over Q under current policy — includes OOD actions → pushed **DOWN**
        - $\mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]$: Q-values for actual dataset actions → pushed **UP**
        - $\alpha$ — CQL regularisation strength; typical range: 0.1 to 5.0
        - $y = r + \gamma\max_{a'}Q(s',a')$ — standard Bellman target
        - Net effect: Q is underestimated for OOD actions, approximately correct for dataset actions
        """)

        # CQL demo: conservative vs aggressive Q
        np.random.seed(1)
        data_actions = np.sort(np.random.uniform(-0.5, 0.5, 30))
        true_q = np.exp(-data_actions**2) * 2
        x_all = np.linspace(-2.5, 2.5, 300)
        naive_q = np.exp(-x_all**2)*2 + np.random.randn(300)*0.05  # accurate in data, bad outside
        naive_q = np.where(np.abs(x_all) > 0.7,
                           naive_q + 2.5*np.abs(x_all)*np.random.randn(300)*0.3, naive_q)
        cql_q = np.where(np.abs(x_all) > 0.7, 0.5, np.exp(-x_all**2)*2)

        fig_cql, ax_cql = _fig(1, 1, 11, 4)
        ax_cql.scatter(data_actions, true_q, color="#ffa726", s=50, zorder=5, label="Dataset actions")
        ax_cql.plot(x_all, np.exp(-x_all**2)*2, color="#4caf50", lw=2, ls="--", label="True Q*")
        ax_cql.plot(x_all, np.clip(naive_q, -1, 5), color="#ef5350", lw=2, label="Naive Q-learning (overestimates OOD)")
        ax_cql.plot(x_all, cql_q, color="#0288d1", lw=2.5, label="CQL (conservative on OOD)")
        ax_cql.axvspan(-2.5, -0.5, alpha=0.07, color="#ef5350"); ax_cql.axvspan(0.5, 2.5, alpha=0.07, color="#ef5350")
        ax_cql.axvspan(-0.5, 0.5, alpha=0.07, color="#4caf50")
        ax_cql.set_xlabel("Action a", color="white"); ax_cql.set_ylabel("Q(s, a)", color="white")
        ax_cql.set_title("CQL: Conservative Q-values prevent OOD action selection",
                         color="white", fontweight="bold")
        ax_cql.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_cql.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_cql); plt.close()

    with tab_iql:
        st.subheader("🎯 IQL — Implicit Q-Learning (Kostrikov et al. 2021)")
        st.markdown(_card("#7c4dff","🎯","IQL: never query Q for out-of-distribution actions",
            """Implicit Q-Learning takes a fundamentally different approach to the distributional
            shift problem: instead of penalising OOD Q-values, it avoids evaluating Q for OOD
            actions entirely. The standard Bellman backup requires computing max_{a'} Q(s', a')
            which involves querying Q for potentially unseen actions. IQL replaces this with
            the value function V(s') which only depends on the state, not the action, and is learned
            using only dataset transitions. V(s) is estimated via expectile regression — a
            generalisation of quantile regression that approximates the maximum of Q without
            explicit maximisation. A high expectile (tau=0.9) makes V approximate max Q.
            The policy is then extracted via advantage-weighted regression: the policy increases
            log-probability of dataset actions proportionally to their advantage exp(beta*A(s,a)).
            Only dataset actions ever appear in the policy loss — the policy is in-distribution
            by construction. IQL is simpler to tune than CQL, achieves comparable D4RL scores,
            and is the algorithm of choice when you want reliable offline RL without complex tuning."""), unsafe_allow_html=True)

        st.markdown("**IQL's three separate training steps:**")
        st.markdown("**Step 1 — Value function via expectile regression:**")
        st.latex(r"\mathcal{L}_V(\psi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[L_2^\tau\bigl(Q_{\bar\theta}(s,a)-V_\psi(s)\bigr)\right]")
        st.latex(r"L_2^\tau(u) = \bigl|\tau - \mathbf{1}[u<0]\bigr|\cdot u^2 \quad \tau\in(0,1)")
        st.markdown(r"""
        When $\tau=0.9$: the loss function weights positive residuals (Q > V) by 0.9 and negative
        by 0.1 — forcing V to approximate the 90th percentile of Q, approximating max Q.
        """)
        st.markdown("**Step 2 — Q-function via Bellman backup (using V, not max Q):**")
        st.latex(r"\mathcal{L}_Q(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\!\left[\bigl(r+\gamma V_\psi(s')-Q_\theta(s,a)\bigr)^2\right]")
        st.markdown("**Step 3 — Policy extraction via advantage-weighted regression:**")
        st.latex(r"\mathcal{L}_\pi(\phi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\exp\bigl(\beta(Q_{\bar\theta}(s,a)-V_\psi(s))\bigr)\cdot(-\log\pi_\phi(a|s))\right]")

    with tab_dt:
        st.subheader("🤖 Decision Transformer (Chen et al. 2021)")
        st.markdown(_card("#00897b","🤖","Offline RL as language modelling — no Bellman equations needed",
            """Decision Transformer sidesteps all Q-learning problems by reframing offline RL as
            conditional sequence modelling — exactly like GPT predicting the next token.
            The input to the transformer is a sequence of (return-to-go, state, action) triples from
            the offline dataset, and the model is trained to predict the action given this context.
            Return-to-go (RTG) R_hat_t = sum_{t'>=t} r_{t'} is the total future reward from timestep t
            onward. At test time, you set R_hat_1 to a desired high return and the model generates
            actions that achieve it — conditioning on 'I want 90 total reward from now' makes the model
            produce the actions that lead to that outcome. After each real step, R_hat decreases by the
            received reward. Decision Transformer achieves performance competitive with CQL and IQL
            while requiring no Bellman updates, no distributional shift handling, and no reward shaping.
            It naturally extends to multi-task settings (different tasks = different goal returns),
            can leverage pretrained language model weights, and handles very long-horizon tasks where
            Bellman backup errors compound badly. The main limitation: it needs a dataset where
            trajectories have informative reward structure, and it cannot improve beyond the best
            behaviour in the dataset without fine-tuning."""), unsafe_allow_html=True)

        st.markdown("**Decision Transformer input format:**")
        st.latex(r"\tau = \bigl(\hat R_1,s_1,a_1,\;\hat R_2,s_2,a_2,\;\ldots,\;\hat R_T,s_T,a_T\bigr)")
        st.markdown(r"Return-to-go: $\hat R_t = \sum_{t'=t}^T r_{t'}$ (actual future reward from $t$ in dataset)")
        st.markdown("**Supervised training objective (cross-entropy on actions):**")
        st.latex(r"\mathcal{L}_\text{DT} = \mathbb{E}_{(s_t,a_t,\hat R_t)\sim\mathcal{D}}\!\left[-\log\pi_\theta(a_t\mid\hat R_1,s_1,a_1,\ldots,\hat R_t,s_t)\right]")
        st.markdown(r"""
        **At deployment:** $\hat R_1 = $ target return. $\hat R_{t+1} = \hat R_t - r_t$.
        The model generates action $a_t = \pi_\theta(\cdot|\hat R_t, s_t, \text{context})$.
        """)
        st.markdown(_insight("""
        Decision Transformer inherits all advantages of Transformer architectures:
        attention over long contexts (handle long-horizon tasks), pretrained initialisation (faster
        learning), easy scaling (bigger model = better performance). The approach has been extended
        to Online Decision Transformer (fine-tune with online data), Multi-Game Decision Transformer
        (one model for 41 Atari games), and Gato (one model for 600+ tasks).
        """), unsafe_allow_html=True)

    with tab_td3bc:
        st.subheader("🔁 TD3+BC — Offline TD3 with Behaviour Cloning (Fujimoto & Gu 2021)")
        st.markdown(_card("#e65100","🔁","TD3+BC: the simplest offline RL algorithm that actually works well",
            """TD3+BC is the most minimal offline RL algorithm: take the standard TD3 continuous
            control algorithm and add a single behaviour cloning term to the policy update. That
            is the entire algorithm change — one additional loss term. The BC term penalises the
            policy for actions that deviate from those in the dataset, preventing it from selecting
            OOD actions. The Q-learning part pulls the policy toward high-value actions; the BC term
            keeps it close to the data distribution. A normalisation factor lambda = alpha / mean|Q|
            makes the balance between these two objectives independent of the reward scale of the
            environment — the same alpha works across environments with very different reward magnitudes.
            TD3+BC achieves competitive D4RL scores despite its simplicity and is the recommended
            starting point for practitioners new to offline RL. Its 50-line implementation is far
            easier to debug than CQL or IQL. The main limitation: behaviour cloning is only as good
            as the dataset — if the dataset contains many suboptimal trajectories, the BC term will
            pull the policy toward suboptimal behaviour. CQL and IQL handle this better by identifying
            which dataset actions are actually good."""), unsafe_allow_html=True)

        st.markdown("**TD3+BC policy update (one term added to standard TD3):**")
        st.latex(r"\pi = \arg\max_\pi\;\mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\underbrace{\lambda Q(s,\pi(s))}_{\text{standard RL}} - \underbrace{(\pi(s)-a)^2}_{\text{behaviour cloning}}\right]")
        st.markdown(r"where $\lambda = \alpha\,/\,\frac{1}{N}\sum_{(s,a)\in\mathcal{D}}|Q(s,a)|$ normalises Q to unit scale.")
        st.markdown("**Q-update is unchanged TD3 (twin delayed Q-networks):**")
        st.latex(r"y = r + \gamma\min_{j=1,2}Q_{\bar\theta_j}(s',\bar a'), \quad \bar a' = \pi_{\bar\phi}(s') + \text{clip}(\mathcal{N}(0,\sigma),-c,c)")

    with tab_cmp:
        st.subheader("📊 Offline RL Algorithm Comparison")
        st.dataframe(pd.DataFrame({
            "Algorithm": ["Behaviour Cloning","CQL","IQL","Decision Transformer","TD3+BC"],
            "Approach": ["Supervised imitation","Conservative Q-values","Expectile V + AWR","Sequence modelling","TD3 + BC penalty"],
            "Handles suboptimal data": ["❌","✅","✅","Partial","Partial"],
            "Requires Bellman backup": ["❌","✅","✅","❌","✅"],
            "OOD action protection": ["Implicit","Explicit (penalty)","Implicit (avoids OOD)","None (dataset only)","Explicit (BC term)"],
            "Hyperparameter sensitivity": ["Low","Medium","Low","Low","Low"],
            "Implementation difficulty": ["Trivial","Moderate","Moderate","High","Simple"],
            "Best for": ["Expert-only data","Mixed data, continuous","Mixed data, discrete+cont","Large datasets, multi-task","Continuous, simplicity"],
        }), use_container_width=True, hide_index=True)
