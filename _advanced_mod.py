"""
_advanced_mod.py — Tier 2 Specialisations in RL
Covers: MARL (MADDPG, QMIX, MAPPO) · Hierarchical RL · Safe RL · Meta-RL
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"

def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
            f'<b>{icon} {title}</b><br>{body}</div>')

def _insight(text):
    return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
            f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem">'
            f'💡 {text}</div>')

def _sec(emoji, title, sub, color="#7c4dff"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def main_advanced():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0a1a2e,#1a0a1e,#0a2a1a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🚀 Advanced RL Specialisations</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'Tier 2 topics: Multi-Agent RL, Hierarchical RL, Safe RL, and Meta-RL. '
        'Each domain addresses a specific failure mode of single-agent model-free RL.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🤝 Multi-Agent RL",
        "🏗️ Hierarchical RL",
        "🛡️ Safe RL",
        "🧠 Meta-RL",
        "📊 When to Use",
    ])
    (tab_marl, tab_hier, tab_safe, tab_meta, tab_when) = tabs

    # ── MARL ──────────────────────────────────────────────────────────
    with tab_marl:
        _sec("🤝","Multi-Agent Reinforcement Learning (MARL)",
             "MADDPG · QMIX · MAPPO — multiple agents cooperating or competing","#0288d1")

        st.markdown(_card("#0288d1","🤝","What is MARL and where is it essential?",
            """Multi-Agent RL studies environments with multiple agents that simultaneously take actions
            and influence each other. Applications include: traffic signal control (thousands of intersections
            as separate agents); autonomous vehicle coordination; financial markets (trading agents);
            multiplayer game-playing (OpenAI Five with 5 agents per team); drone swarms; and distributed
            sensor networks. The fundamental challenge: the environment is non-stationary from any single
            agent's perspective, because the other agents are also learning and changing their behaviour.
            This violates the Markov property that most single-agent RL theory relies on.
            Multi-agent settings fall into three categories: (1) cooperative (all agents share a reward,
            work together); (2) competitive (zero-sum, one agent's gain is another's loss); (3) mixed
            (some cooperation, some competition). The key algorithmic framework that makes cooperative MARL
            tractable is Centralised Training with Decentralised Execution (CTDE): during training,
            each agent can see all other agents' observations and actions (facilitating credit assignment);
            at execution time, each agent acts only on its own local observation (practical deployment).
            MADDPG, QMIX, and MAPPO all implement CTDE with different approaches."""), unsafe_allow_html=True)

        with st.expander("🔵 MADDPG — Multi-Agent DDPG (Lowe et al. 2017)", expanded=True):
            st.markdown(r"""
            MADDPG (Multi-Agent Deep Deterministic Policy Gradient) extends DDPG to multi-agent
            settings using the CTDE framework. Each agent $i$ has:
            - An actor $\pi_{\theta_i}(a_i|o_i)$ that conditions only on its own observation $o_i$
            - A centralised critic $Q_i(o_1,\ldots,o_n,a_1,\ldots,a_n)$ that takes ALL observations
              and actions during training

            The centralised critic solves the non-stationarity problem: from the critic's perspective,
            the environment IS stationary because it sees all agents' actions simultaneously.
            """)
            st.latex(r"y_i = r_i + \gamma Q_{\bar\theta_i}(o'_1,\ldots,o'_n,\,a'_1,\ldots,a'_n)\Big|_{a'_j=\pi_{\bar\theta_j}(o'_j)}")
            st.latex(r"\nabla_{\theta_i}J(\pi_i) = \mathbb{E}\!\left[\nabla_{\theta_i}\log\pi_i(a_i|o_i)\cdot Q_i(o_1,\ldots,o_n,a_1,\ldots,a_n)\right]")

        with st.expander("🟢 QMIX — Monotonic Value Factorisation (Rashid et al. 2018)", expanded=False):
            st.markdown(r"""
            QMIX is designed for cooperative multi-agent settings with a shared team reward.
            It learns individual Q-functions $Q_i(o_i, a_i)$ for each agent and combines them
            into a joint Q-function $Q_{tot}$ via a learned mixing network with a monotonicity constraint:
            """)
            st.latex(r"Q_{\text{tot}}(\mathbf{o},\mathbf{a}) = f_\psi(Q_1(o_1,a_1),\ldots,Q_n(o_n,a_n))")
            st.latex(r"\text{Subject to: } \frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0 \quad \forall i")
            st.markdown(r"""
            The monotonicity constraint ensures that the global argmax policy (choose the greedy action
            for each agent independently) is consistent with the global Q-function — enabling decentralised
            execution without needing to query the full joint action space. QMIX is the dominant algorithm
            for StarCraft II unit micromanagement (SMAC benchmark) with up to 27 cooperating agents.
            """)

        with st.expander("🟡 MAPPO — Multi-Agent PPO (Yu et al. 2021)", expanded=False):
            st.markdown(r"""
            MAPPO applies PPO to multi-agent cooperative tasks with a centralised value function.
            Each agent has its own PPO actor $\pi_i(a_i|o_i)$ with clipped surrogate objective.
            The value function is centralised: $V(s)$ takes the global state or all observations.
            GAE advantages are computed using this centralised value function.
            Surprisingly simple yet competitive with QMIX on most MARL benchmarks.
            """)
            st.latex(r"L^{\text{MAPPO}}_i = L^{\text{CLIP}}_i(\theta_i) + c_1(V_\omega(s) - R_t)^2 + c_2 H(\pi_i)")
            st.markdown(r"Independent PPO with shared value: $V_\omega$ takes global state $s$ (all agents' obs).")

    # ── HIERARCHICAL RL ────────────────────────────────────────────────
    with tab_hier:
        _sec("🏗️","Hierarchical Reinforcement Learning",
             "Options framework · Goal-conditioned RL · Feudal networks — temporal abstraction for long-horizon tasks","#e65100")

        st.markdown(_card("#e65100","🏗️","What is Hierarchical RL and why is it necessary?",
            """Flat RL agents (those with a single policy operating at the raw action level) struggle
            with long-horizon tasks that require hundreds or thousands of sequential decisions. Consider
            a robot cooking a meal: the raw actions are 'move joint i by 0.01 radians'. Planning a
            meal in terms of joint rotations is absurdly complex. Hierarchical RL decomposes the
            problem into levels of abstraction: a high-level manager policy decides 'go to the fridge',
            a mid-level policy decides 'extend arm toward handle', a low-level policy executes the
            joint rotations. Each level operates at a different timescale — the manager decides every
            10–20 steps, the low-level controller every step. This temporal abstraction dramatically
            reduces the effective horizon at each level, making planning tractable. The key challenges
            in HRL are: (1) how to define subgoals (handcrafted vs learned); (2) credit assignment
            across levels; (3) ensuring the high-level goals are achievable by the low level.
            Three main approaches: the Options framework (formalism), goal-conditioned RL (subgoal
            as input), and feudal networks (value decomposition across hierarchy levels)."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**The Options Framework (Sutton, Precup & Singh 1999):**")
            st.markdown(r"""
            An option $\omega = (I_\omega, \pi_\omega, \beta_\omega)$ consists of:
            - $I_\omega$ — initiation set: states where option can start
            - $\pi_\omega$ — intra-option policy: low-level actions
            - $\beta_\omega(s)$ — termination function: probability of terminating in state $s$
            """)
            st.latex(r"\pi_\Omega(o|s) = \text{policy over options (high-level manager)}")
            st.latex(r"Q_\Omega(s,o) = \sum_t\gamma^t r_t + \gamma^T V(s_T) \quad \text{(option-value)}")
        with col2:
            st.markdown("**Goal-Conditioned RL (GCRL):**")
            st.markdown(r"""
            High-level policy sets a goal $g$ (e.g. "reach position (3,2)").
            Low-level policy is conditioned on the goal: $\pi_{\text{lo}}(a|s,g)$.
            """)
            st.latex(r"\pi_{\text{hi}}(g_t|s_t) \quad \text{(select goal every k steps)}")
            st.latex(r"\pi_{\text{lo}}(a_t|s_t,g_t) \quad \text{(execute goal at every step)}")
            st.markdown(r"""
            Low-level reward: $r_{\text{lo}} = -\|s - g\|$ (distance to goal).
            Training trick: Hindsight Experience Replay (HER) relabels failed trajectories
            as successful with whatever state was actually reached as the 'goal'.
            """)

        st.markdown(_insight("""
        Feudal Networks (FuN, Vezhnevets et al. 2017) extend the two-level hierarchy by having the
        manager emit goal vectors in a learned latent space, not raw state space. The manager is
        trained with a different reward function (extrinsic rewards) from the worker (intrinsic
        reward = cosine similarity between direction to goal and actual movement). This separation
        of responsibilities allows each level to specialise independently.
        """), unsafe_allow_html=True)

    # ── SAFE RL ────────────────────────────────────────────────────────
    with tab_safe:
        _sec("🛡️","Safe Reinforcement Learning",
             "CPO · PCPO · Lagrangian methods — policies that satisfy constraints during learning","#ef5350")

        st.markdown(_card("#ef5350","🛡️","What is Safe RL and why it is critical for real-world deployment",
            """Standard RL optimises total reward without any constraint on intermediate behaviour.
            This is fine in simulation, but catastrophic in the real world: during training, the
            agent will try dangerous actions to explore, potentially causing expensive or irreversible
            harm (collision of an autonomous vehicle, failure of industrial equipment, medical error).
            Safe RL formalises the problem as a Constrained Markov Decision Process (CMDP): maximise
            the expected cumulative reward subject to constraints on other safety-relevant quantities
            (expected cost below a threshold, probability of constraint violation below some level).
            Three main approaches: (1) Constrained optimisation — directly solve the constrained problem
            using trust region methods (CPO, PCPO); (2) Lagrangian relaxation — convert the constrained
            problem to an unconstrained one by adding a penalty term with an adaptive multiplier;
            (3) Safety layers — a constraint-aware action projection that modifies any proposed action
            to satisfy constraints before execution. Safe RL is non-negotiable for robotics, autonomous
            vehicles, industrial control, and healthcare applications. As RL moves out of simulation
            and into the real world, safe exploration is arguably the most important open problem."""), unsafe_allow_html=True)

        st.markdown("**Constrained MDP formulation:**")
        st.latex(r"\max_\pi J(\pi) = \mathbb{E}_\pi\!\left[\sum_t\gamma^t r_t\right] \quad \text{s.t.} \quad J_C^i(\pi) = \mathbb{E}_\pi\!\left[\sum_t\gamma^t c_t^i\right] \leq d^i \;\forall i")
        st.markdown(r"""
        - $c_t^i$ — cost signal for constraint $i$ (e.g. 1 if the robot falls, 0 otherwise)
        - $d^i$ — constraint threshold (e.g. max 0.1 expected falls per episode)
        - Multiple constraints can be combined: collision avoidance AND energy consumption AND speed
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Lagrangian Method (simplest):**")
            st.latex(r"L(\pi,\lambda) = J(\pi) - \sum_i\lambda_i(J_C^i(\pi)-d^i)")
            st.latex(r"\lambda_i \leftarrow \lambda_i + \eta_\lambda(J_C^i(\pi)-d^i)")
            st.markdown(r"Dual ascent on $\lambda_i$: if constraint violated → increase penalty; if satisfied → decrease. Simple to implement but can oscillate.")
        with col2:
            st.markdown("**CPO — Constrained Policy Optimization (Achiam 2017):**")
            st.latex(r"\pi_{k+1} = \arg\max_\pi L_\pi(\pi) \quad \text{s.t.} \quad D_{\text{KL}}(\pi\|\pi_k)\leq\delta,\; J_{C}^i(\pi)\leq d^i")
            st.markdown(r"Extends TRPO with hard constraint enforcement. Uses Taylor series approximation of constraint and trust region for guaranteed monotonic safety improvement.")

    # ── META-RL ────────────────────────────────────────────────────────
    with tab_meta:
        _sec("🧠","Meta-RL — Learning to Learn",
             "MAML · RL² — fast adaptation to new tasks with few samples","#00897b")

        st.markdown(_card("#00897b","🧠","What is Meta-RL and why fast adaptation matters",
            """Meta-RL (learning to learn) addresses the sample inefficiency of standard RL in a
            different way from model-based methods: instead of building a world model, it trains
            the agent across many related tasks so that it can adapt to new tasks with only a handful
            of environment interactions. The motivation: a robot that has learned to pick up 1000
            different objects should be able to pick up object 1001 quickly, because the task structure
            is the same. Standard RL treats each task from scratch — it has no memory of previous tasks.
            Meta-RL explicitly optimises for fast adaptation: the meta-training objective is not to
            perform well on the training tasks, but to learn an initialisation or algorithm that allows
            fast learning on new tasks. Two main approaches: (1) MAML (gradient-based) — find an
            initialisation theta_0 such that a few gradient steps from theta_0 produce a good policy
            for any task; (2) RL^2 (recurrent) — treat the agent's entire interaction with a new task
            as a sequence, and train a recurrent network to be the learning algorithm itself.
            Meta-RL is essential for robotics (adapt to different payloads, surfaces, damages),
            game-playing (generalise to new levels/maps), and few-shot adaptation in any domain
            where massive pre-training is possible but online data is scarce."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**MAML — Model-Agnostic Meta-Learning (Finn et al. 2017):**")
            st.latex(r"\theta_0 = \arg\min_\theta\sum_{\mathcal{T}_i\sim p(\mathcal{T})}\mathcal{L}_{\mathcal{T}_i}(\theta_i')")
            st.latex(r"\theta_i' = \theta - \alpha\nabla_\theta\mathcal{L}_{\mathcal{T}_i}(\theta) \quad \text{(inner loop: k gradient steps)}")
            st.markdown(r"""
            Find $\theta_0$ (meta-parameters) such that one gradient step from $\theta_0$ on any task
            $\mathcal{T}_i$ gives a good task-specific policy $\theta_i'$.
            Training: bilevel optimisation — inner loop adapts, outer loop updates $\theta_0$.
            """)
        with col2:
            st.markdown(r"**RL² — RL as an RNN algorithm (Duan et al. 2016):**")
            st.latex(r"h_{t+1} = f_\theta(h_t,\,o_t,\,a_{t-1},\,r_{t-1},\,d_{t-1})")
            st.latex(r"\pi_\theta(a_t|h_t) \quad \text{(action from hidden state)}")
            st.markdown(r"""
            The RNN hidden state $h_t$ encodes the agent's entire history within the current task —
            the network learns to use this history to infer the task identity and adapt its policy.
            The recurrent network IS the learning algorithm: it 'learns to learn' by encoding
            task-relevant information in its hidden state.
            """)

    with tab_when:
        st.subheader("📊 When to Use Each Advanced Topic")
        st.dataframe(pd.DataFrame({
            "Topic":["MARL","Hierarchical RL","Safe RL","Meta-RL"],
            "Use when":["Multiple interacting agents","Very long horizons (>500 steps)","Real-world deployment","Many related tasks, scarce per-task data"],
            "Key challenge":["Non-stationarity","Credit assignment across levels","Exploration while satisfying constraints","Bilevel optimisation (expensive)"],
            "Representative algorithms":["MADDPG, QMIX, MAPPO","Options, GCRL+HER, FuN","CPO, Lagrangian PPO, Safety Layer","MAML, RL², PEARL, ProMP"],
            "Difficulty":["High","Very High","High","Very High"],
            "Best benchmark":["SMAC (StarCraft)","AntMaze, Fetch","Safety-Gym","Meta-World, Alchemy"],
        }), use_container_width=True, hide_index=True)
