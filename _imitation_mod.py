"""
_imitation_mod.py — Imitation Learning & Learning from Demonstrations
Covers: Behaviour Cloning · DAgger · GAIL · AIRL · Inverse RL
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from _notes_mod import render_notes
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


def _sec(emoji, title, sub, color="#7c4dff"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)


def main_imitation():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a0a2e,#0a2a1a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🎓 Imitation Learning & Learning from Demonstrations</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'When reward functions are hard to specify but expert demonstrations are available. '
        'BC, DAgger, GAIL, AIRL — the complete family from naive cloning to adversarial reward extraction.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "❓ Why Imitation Learning?",
        "📋 Behaviour Cloning",
        "🔄 DAgger",
        "⚔️ GAIL",
        "🔍 AIRL & Inverse RL",
        "📊 Method Comparison",
        "📚 Resources",
    ])
    (tab_why, tab_bc, tab_dag, tab_gail, tab_airl, tab_cmp, tab_res) = tabs

    with tab_why:
        _sec("❓", "Why Imitation Learning?",
             "Reward functions are hard to specify — expert demonstrations are often available", "#7c4dff")

        st.markdown(_card("#7c4dff", "🤔", "The reward specification problem",
            """Designing a reward function that produces desired behaviour is one of the hardest
            problems in applied RL. For a robot arm that should pick up a glass and pour water:
            how much reward for lifting? For not spilling? For speed? For gentle handling? Every
            choice changes the behaviour. Get it slightly wrong and the robot learns to cheat the
            reward (tip the glass quickly to maximise the speed bonus). Meanwhile, a human can
            demonstrate correct behaviour in seconds — pick up glass, pour slowly, set down gently.
            Imitation learning asks: can we extract a policy from these demonstrations without
            ever specifying a reward? The answer is yes — with important caveats. Behaviour Cloning
            (BC) is the simplest approach: treat demonstrations as supervised data and train a
            policy to mimic them. But BC fails at test time due to distributional shift — the agent
            visits states the expert never demonstrated, has no idea what to do, and compounds
            errors catastrophically. DAgger, GAIL, and Inverse RL all address this failure mode
            in different ways. This entire subfield is essential for real robotics, where human
            demonstration data is abundant and reward specification is difficult."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **When to use imitation learning:**
            - Expert demonstrations are available and cheap to collect
            - Reward function is hard to specify (dexterous manipulation, social navigation)
            - Want to initialise a policy before RL fine-tuning
            - Offline RL dataset contains near-optimal trajectories
            """)
        with col2:
            st.markdown(r"""
            **The imitation learning taxonomy:**
            - BC: supervised learning on demonstrations
            - DAgger: interactive BC that corrects errors
            - Inverse RL: recover reward function from demos
            - GAIL: adversarial imitation without explicit reward
            - AIRL: disentangled reward extraction for transfer
            """)

        # Compounding error visualisation
        np.random.seed(42)
        t = np.arange(50)
        expert_traj = np.sin(t*0.3)*2
        bc_policy = expert_traj + np.cumsum(np.random.randn(50)*0.15)  # compound drift
        dagger_policy = expert_traj + np.random.randn(50)*0.2  # stays close

        fig_comp, ax_comp = _fig(1, 1, 11, 4)
        ax_comp.plot(t, expert_traj, color="#4caf50", lw=2.5, label="Expert trajectory")
        ax_comp.plot(t, bc_policy, color="#ef5350", lw=2, ls="--",
                     label="BC policy (compounds errors)")
        ax_comp.plot(t, dagger_policy, color="#0288d1", lw=2,
                     label="DAgger policy (corrects errors interactively)")
        ax_comp.fill_between(t, expert_traj, bc_policy, alpha=0.1, color="#ef5350",
                             label="BC error (grows over time)")
        ax_comp.set_xlabel("Timestep", color="white")
        ax_comp.set_ylabel("State variable", color="white")
        ax_comp.set_title("Compounding Error: Why BC Fails at Test Time",
                          color="white", fontweight="bold")
        ax_comp.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_comp.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_comp); plt.close()

    with tab_bc:
        _sec("📋", "Behaviour Cloning — Supervised Imitation",
             "Treat demonstrations as supervised data — simple but breaks on distributional shift", "#0288d1")

        st.markdown(_card("#0288d1", "📋", "Behaviour Cloning: the simplest possible approach",
            """Behaviour Cloning (BC) treats imitation as pure supervised learning. Given a dataset
            of expert demonstrations D = {(s_i, a_i)} collected by an expert policy π*, train a
            policy π_θ to minimise the prediction error on the expert actions. For discrete actions,
            this is cross-entropy loss. For continuous actions, this is mean squared error.
            BC is trivially simple to implement — it is exactly the same as any classification or
            regression problem. The limitation is distributional shift: the expert data only covers
            states the expert visited. At test time, the learned policy makes small errors, visits
            states slightly off the expert's path, makes errors there, visits states even further
            off the path — and errors compound over time. The performance degrades quadratically
            with episode length T: cost ~ T². For short-horizon tasks or near-perfect imitation,
            BC works well. For long-horizon tasks or imperfect imitation, it fails. BC is the
            mandatory baseline for any imitation learning paper and the starting point for all
            offline RL approaches."""), unsafe_allow_html=True)

        st.markdown("**BC training objective:**")
        st.latex(r"\min_\theta\mathcal{L}_\text{BC}(\theta) = -\mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\log\pi_\theta(a|s)\right]")
        st.markdown(r"""
        For continuous actions (MSE loss):
        """)
        st.latex(r"\mathcal{L}_\text{BC}(\theta) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\|\mu_\theta(s)-a\|^2\right]")

        st.markdown("**The distributional shift bound (Ross & Bagnell 2010):**")
        st.latex(r"J(\pi_\text{BC}) \leq J(\pi^*) + T^2\epsilon_\text{BC}")
        st.markdown(r"""
        where $\epsilon_\text{BC}$ = expected per-step error on the training distribution and $T$ = horizon.
        This $T^2$ dependence is the fundamental limitation — errors compound.
        """)

        st.code("""
# Behaviour Cloning — complete implementation
import numpy as np
import torch, torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
    def forward(self, s): return self.net(s)

def train_bc(demonstrations, epochs=100, lr=3e-4):
    \"\"\"
    demonstrations: list of (state, action) pairs from expert
    \"\"\"
    states  = torch.FloatTensor([s for s,a in demonstrations])
    actions = torch.FloatTensor([a for s,a in demonstrations])

    policy = BCPolicy(states.shape[1], actions.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        # Standard supervised learning — predict expert action from state
        pred_actions = policy(states)
        loss = nn.MSELoss()(pred_actions, actions)  # or cross-entropy for discrete
        opt.zero_grad(); loss.backward(); opt.step()

    return policy  # deploy: a = policy(state)
""", language="python")

    with tab_dag:
        _sec("🔄", "DAgger — Dataset Aggregation",
             "Ross et al. 2011 — interactive BC that reduces error bound from T² to T", "#00897b")

        st.markdown(_card("#00897b", "🔄", "DAgger: the interactive fix for BC's compounding errors",
            """DAgger (Dataset Aggregation, Ross et al. 2011) fixes BC's distributional shift problem
            with an elegant iterative procedure: run the current policy in the environment, collect
            states the policy actually visits (including its mistakes), ask the expert to label these
            states with the correct action, and add them to the training dataset. After retraining on
            the expanded dataset, the policy has now seen corrections for the mistake states it visits.
            Next iteration: run the improved policy, collect its new mistake states, get expert labels,
            retrain. Each iteration fills in more of the state space with expert guidance, until the
            policy's visited distribution matches the expert's distribution closely enough to perform well.
            DAgger improves the performance bound from T² (BC) to T (linear in horizon) because errors
            are corrected interactively rather than compounding. The practical requirement: the expert
            must be queryable — they need to be able to label any arbitrary state with the right action.
            For a human expert, this means watching the robot in real-time and providing corrections.
            For a scripted expert (e.g. a hand-engineered controller), it means calling it on arbitrary
            states. DAgger is the foundation of many modern robot learning systems."""), unsafe_allow_html=True)

        st.markdown("**DAgger improved performance bound:**")
        st.latex(r"J(\pi_\text{DAgger}) \leq J(\pi^*) + T\epsilon_\text{DAgger}")
        st.markdown(r"Linear in $T$ vs quadratic for BC — a fundamental improvement for long-horizon tasks.")

        st.code(r"""
# DAgger Algorithm
def dagger(env, expert_policy, n_iterations=10, n_episodes_per_iter=10):
    dataset = []  # aggregated (state, expert_action) pairs

    # Step 1: Initialise with pure expert demonstrations
    for _ in range(n_episodes_per_iter):
        s = env.reset()
        while True:
            a_expert = expert_policy(s)
            dataset.append((s, a_expert))
            s, _, done, _ = env.step(a_expert)
            if done: break

    policy = train_bc(dataset)  # initial BC policy

    for iteration in range(n_iterations):
        new_data = []
        for _ in range(n_episodes_per_iter):
            s = env.reset()
            while True:
                # KEY: run CURRENT POLICY to get state distribution
                a_current = policy(s)  # may be wrong
                # KEY: ask expert what to do in this state
                a_expert = expert_policy(s)  # expert labels the state
                new_data.append((s, a_expert))  # label with EXPERT action

                s, _, done, _ = env.step(a_current)  # but EXECUTE current policy
                if done: break

        # Aggregate dataset and retrain
        dataset = dataset + new_data
        policy = train_bc(dataset)  # retrain on full aggregated dataset

    return policy
""", language="python")

        st.markdown(_insight("""
        <b>DAgger in practice:</b> For robot manipulation, the expert is often a human operator
        with a joystick who corrects the robot's mistakes in real-time. After 5–10 DAgger iterations
        of ~50 episodes each, robot policies typically achieve 80–95% of expert performance on
        tasks like pouring, stacking, and pick-and-place. Tesla's autopilot team used a DAgger-like
        system to collect intervention data from human drivers correcting autopilot mistakes.
        """), unsafe_allow_html=True)

    with tab_gail:
        _sec("⚔️", "GAIL — Generative Adversarial Imitation Learning",
             "Ho & Ermon 2016 — adversarial reward extraction without specifying a reward", "#e65100")

        st.markdown(_card("#e65100", "⚔️", "GAIL: imitation via adversarial training",
            """GAIL (Ho & Ermon 2016) is the most powerful imitation learning algorithm for complex
            tasks. Instead of directly predicting expert actions (BC) or querying the expert
            interactively (DAgger), GAIL learns a reward function implicitly through adversarial
            training — exactly like a GAN. The key insight: a policy is good if you cannot tell its
            state-action occupancy distribution apart from the expert's. Train a discriminator D
            to distinguish policy trajectories from expert trajectories. The reward signal for the
            policy is the discriminator's confusion — the more the policy looks like the expert,
            the higher its reward. The policy is trained with standard RL (TRPO or PPO) on this
            learned reward. Unlike BC, GAIL does not have a distributional shift problem because
            it trains the policy online — the discriminator sees and corrects the policy's actual
            visited distribution. Unlike Inverse RL, GAIL does not require explicitly recovering
            the reward function — it extracts the reward implicitly. GAIL achieves expert-level
            performance on complex locomotion tasks (MuJoCo HalfCheetah, Ant) with far fewer
            expert demonstrations than BC requires. The main limitation: requires online environment
            interaction (like standard RL), and adversarial training is sometimes unstable."""), unsafe_allow_html=True)

        st.markdown("**GAIL objective — occupancy measure matching via adversarial training:**")
        st.markdown(r"GAIL minimises the Jensen-Shannon divergence between the policy's and expert's state-action occupancy measures $\rho_\pi$ and $\rho_{\pi^*}$:")
        st.latex(r"\min_\pi\max_D\;\mathbb{E}_{(s,a)\sim\rho_\pi}\!\left[\log D(s,a)\right] + \mathbb{E}_{(s,a)\sim\rho_{\pi^*}}\!\left[\log(1-D(s,a))\right] - \lambda H(\pi)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Discriminator update** (classify policy vs expert):")
            st.latex(r"\mathcal{L}_D = -\mathbb{E}_{\rho_\pi}[\log D(s,a)] - \mathbb{E}_{\rho_{\pi^*}}[\log(1-D(s,a))]")
            st.markdown(r"$D(s,a) \in [0,1]$: probability that $(s,a)$ came from expert. Train to maximise this.")
        with col2:
            st.markdown("**Policy update** (fool discriminator using RL):")
            st.latex(r"r_\text{GAIL}(s,a) = -\log(1-D(s,a)) = \log D(s,a)")
            st.markdown(r"Policy trained with PPO/TRPO on reward $r_\text{GAIL}$ — high reward when discriminator is confused (thinks policy = expert).")

        st.code(r"""
# GAIL Training Loop
def train_gail(env, expert_demos, n_iterations=500):
    policy = PPOPolicy(); discriminator = Discriminator()
    opt_D = Adam(discriminator.parameters(), lr=3e-4)

    for iteration in range(n_iterations):
        # 1. Collect rollouts using current policy
        policy_trajs = collect_rollouts(env, policy, n_steps=2048)

        # 2. Update discriminator: distinguish policy from expert
        for _ in range(5):  # multiple discriminator steps per policy step
            policy_sa  = sample_state_actions(policy_trajs)
            expert_sa  = sample_state_actions(expert_demos)

            # Expert: label 1 (real), Policy: label 0 (fake)
            loss_D = -log(discriminator(expert_sa)).mean() \
                     -log(1 - discriminator(policy_sa)).mean()
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # 3. Compute GAIL reward for policy trajectories
        with torch.no_grad():
            for transition in policy_trajs:
                s, a = transition['state'], transition['action']
                # Reward = log(discriminator(s,a)): high when policy looks like expert
                transition['reward'] = torch.log(discriminator(s, a))

        # 4. Update policy with PPO on GAIL reward (+ optional entropy)
        policy.ppo_update(policy_trajs, entropy_coef=0.001)

    return policy
""", language="python")

    with tab_airl:
        _sec("🔍", "AIRL & Inverse RL — Recovering the Reward Function",
             "Fu et al. 2018 — disentangle reward from dynamics for transferable reward extraction", "#ad1457")

        st.markdown(_card("#ad1457", "🔍", "AIRL: reward extraction that transfers across environments",
            """GAIL extracts an implicit reward but it is entangled with the dynamics — the learned
            reward only works in the same environment. Adversarial Inverse Reinforcement Learning
            (AIRL, Fu et al. 2018) recovers a reward function that is explicitly disentangled from
            the environment dynamics. This means the extracted reward can be transferred to a new
            environment with different dynamics (e.g. extract the reward for 'walk forward' from
            a simulator and apply it to the real robot with different physics). The key insight:
            modify the GAIL discriminator to have a specific functional form that, at optimality,
            recovers the true reward function rather than just matching occupancy measures.
            The discriminator is structured as D(s,a,s') = exp(f(s,a,s')) / (exp(f(s,a,s')) + π(a|s))
            where f is the learned advantage function. At convergence, f recovers the true reward.
            AIRL is used when you need to extract a reward function that can be applied in a
            different environment — e.g. learning the reward for manipulation from demonstrations
            in simulation, then applying it to a real robot with different physical parameters."""), unsafe_allow_html=True)

        st.markdown("**AIRL discriminator structure:**")
        st.latex(r"D_\theta(s,a,s') = \frac{\exp(f_\theta(s,a,s'))}{\exp(f_\theta(s,a,s')) + \pi(a|s)}")
        st.markdown(r"""
        where $f_\theta(s,a,s') = g_\theta(s,a) + \gamma h_\theta(s') - h_\theta(s)$.

        At optimality, $g_\theta(s,a)$ recovers the true reward $r^*(s,a)$ and
        $h_\theta(s)$ approximates the shaping potential.

        **MaxEnt IRL (Ziebart et al. 2008) — the theoretical foundation:**
        """)
        st.latex(r"\max_r\;\mathbb{E}_{(s,a)\sim\pi^*}[r(s,a)] - \log Z(r)")
        st.markdown(r"""
        where $Z(r) = \sum_\tau \exp(\sum_t r(s_t,a_t))$ is the partition function.
        MaxEnt IRL finds the reward that makes the expert's demonstrations the most likely
        trajectories under the maximum entropy policy. This avoids the ambiguity of classical IRL
        (many rewards can explain the same demonstrations) by picking the one with maximum entropy.
        """)

        st.dataframe(pd.DataFrame({
            "Method": ["BC", "DAgger", "GAIL", "AIRL", "MaxEnt IRL"],
            "What it learns": ["Policy directly", "Policy (interactively)", "Policy (adversarially)", "Policy + Reward", "Reward function"],
            "Needs online interaction": ["❌", "✅", "✅", "✅", "❌"],
            "Reward transferable": ["N/A", "N/A", "❌ (entangled with dynamics)", "✅ (disentangled)", "✅"],
            "Sample efficiency": ["High", "High", "Low", "Low", "Medium"],
            "Practical difficulty": ["Trivial", "Easy", "Medium", "Hard", "Hard"],
            "Best for": ["Short horizons, expert demos", "Interactive robots", "Complex locomotion", "Reward transfer", "Reward analysis"],
        }), use_container_width=True, hide_index=True)

    with tab_cmp:
        _sec("📊", "Imitation Learning — Full Comparison",
             "Method selection guide and performance characteristics", "#7c4dff")

        st.markdown(r"""
        **The imitation learning decision tree:**

        1. Is the expert queryable (can answer "what to do here" for any state)?
           - Yes → DAgger (reduces compounding error from T² to T)
           - No → use fixed demonstration dataset only

        2. Do you need the reward function (for transfer to new env)?
           - Yes → AIRL or MaxEnt IRL
           - No → GAIL (simpler, usually higher performance)

        3. Is the task short-horizon (<50 steps) or very high-quality demos?
           - Yes → Behaviour Cloning (simplest, works surprisingly well)
           - No → DAgger or GAIL

        4. Is the environment interaction cheap (fast simulation)?
           - Yes → GAIL (online RL with learned reward)
           - No → BC or offline IRL
        """)

        st.markdown(_insight("""
        <b>In practice today:</b> Most robotics systems initialise with BC (cheap, fast),
        then fine-tune with RL (either standard reward or GAIL reward).
        The BC initialisation dramatically reduces the RL training time because the policy
        starts near-expert rather than random. This combination (BC pretraining + RL fine-tuning)
        is the standard pipeline in most modern robotic manipulation papers (2022–2025).
        """), unsafe_allow_html=True)

    with tab_res:
        st.subheader("📚 Key Resources")
        for icon, title, desc, url in [
            ("📄", "Ho & Ermon (2016) — GAIL", "Generative Adversarial Imitation Learning. The foundational adversarial IL paper.", "https://arxiv.org/abs/1606.03476"),
            ("📄", "Ross et al. (2011) — DAgger", "Dataset Aggregation. Reduces BC error bound from T² to T.", "https://arxiv.org/abs/1011.0686"),
            ("📄", "Fu et al. (2018) — AIRL", "Adversarial IRL with disentangled reward transfer.", "https://arxiv.org/abs/1710.11248"),
            ("📄", "Ziebart et al. (2008) — MaxEnt IRL", "Maximum Entropy IRL. Resolves IRL reward ambiguity.", "https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf"),
            ("💻", "imitation library (Python)", "Implementations of BC, DAgger, GAIL, AIRL in Python.", "https://github.com/HumanCompatibleAI/imitation"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)

    imitation_notes = [
        (tab_why, "Why Imitation Learning", "imitation_learning"),
        (tab_bc, "Behaviour Cloning", "imitation_learning_behaviour_cloning"),
        (tab_dag, "DAgger", "imitation_learning_dagger"),
        (tab_gail, "GAIL", "imitation_learning_gail"),
        (tab_airl, "AIRL", "imitation_learning_airl"),
        (tab_cmp, "Comparison", "imitation_learning_comparison"),
        (tab_res, "Resources", "imitation_learning_resources"),
    ]
    for tab, note_title, note_slug in imitation_notes:
        with tab:
            render_notes(f"Imitation Learning - {note_title}", note_slug)
