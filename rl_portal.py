"""
RL Learning Portal — Unified Entry Point
Professional educational portal with enhanced Learning Hub, Algorithm Guide, and Method Comparison.
"""

import streamlit as st

st.set_page_config(
    page_title="RL Learning Portal",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# PORTAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

body, .stApp { background: #08080f; font-family: 'Inter', sans-serif; }

.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: #111120; border-radius: 12px; padding: 5px;
    border: 1px solid #1e1e35;
}
.stTabs [data-baseweb="tab"] {
    background: #18182c; border-radius: 8px; color: #9090b8;
    padding: 8px 16px; font-weight: 600; font-size: .88rem; border: 1px solid #1e1e35;
    transition: all .2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4a148c, #0277bd);
    color: white !important; border-color: transparent;
}

div[data-testid="stButton"] > button[kind="secondary"] {
    background: #18182c; border: 1px solid #2a2a4e; color: #b0b0cc; border-radius: 8px;
}
div[data-testid="metric-container"] {
    background: #111120; border-radius: 10px; padding: 11px; border: 1px solid #1e1e35;
}
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* Module cards */
.module-card {
    background: #111120;
    border: 1px solid #1e1e35;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: border-color .2s, box-shadow .2s;
}
.module-card:hover {
    border-color: #6a4dff;
    box-shadow: 0 4px 20px rgba(106,77,255,.15);
}

/* Algorithm cards */
.alg-card {
    background: #111120;
    border: 1px solid #1e1e35;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "portal_page" not in st.session_state:
    st.session_state.portal_page = "home"


def go(page):
    st.session_state.portal_page = page
    for key in ["dp_results", "td_results", "mc_results", "results"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _stars(n, max_n=5, color="#ffa726"):
    filled = f'<span style="color:{color}">{"★" * n}</span>'
    empty  = f'<span style="color:#2a2a4a">{"☆" * (max_n - n)}</span>'
    return filled + empty


def _badge(text, color):
    return (f'<span style="background:{color}22;color:{color};border:1px solid {color}55;'
            f'border-radius:999px;padding:.18rem .65rem;font-size:.72rem;font-weight:700;'
            f'white-space:nowrap;margin-right:.3rem">{text}</span>')


def _tag_row(*tags):
    return "".join(_badge(t, c) for t, c in tags)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE DEFINITIONS — rich metadata
# ─────────────────────────────────────────────────────────────────────────────
MODULES = [
    # ── FOUNDATIONS ────────────────────────────────────────────────────────
    {
        "group": "🏗️ Foundations",
        "group_color": "#00695c",
        "page": "foundations",
        "icon": "📐",
        "title": "Math & CS Foundations",
        "subtitle": "Linear algebra, calculus, probability, information theory, Python/NumPy",
        "color": "#00695c",
        "difficulty": 2,
        "time": "3–4 weeks",
        "prereqs": "High-school math, basic Python",
        "badge": "START HERE",
        "badge_color": "#00695c",
        "description": "Every RL formula relies on one of six mathematical areas. Neural networks ARE matrix multiplications. Backpropagation IS the chain rule. π(a|s) IS a probability distribution. KL divergence appears in PPO and TRPO. Without this foundation every subsequent module feels like magic rather than engineering.",
        "algorithms": ["Gradient descent & variants", "Bayesian inference", "Monte Carlo estimation", "Information-theoretic measures"],
        "key_papers": ["3Blue1Brown — Essence of Linear Algebra", "Gilbert Strang — Linear Algebra", "DeepMind × UCL RL lectures"],
        "outcome": "Implement a 2-layer policy network forward+backward pass in pure NumPy — no PyTorch.",
    },
    {
        "group": "🏗️ Foundations",
        "group_color": "#00695c",
        "page": "prereq",
        "icon": "🧬",
        "title": "Deep Learning Prerequisites",
        "subtitle": "Autograd, optimisers, normalisation, CNNs, RNNs, Transformers, PyTorch training loop",
        "color": "#00897b",
        "difficulty": 2,
        "time": "1–2 weeks",
        "prereqs": "Math foundations",
        "badge": "Stage 0",
        "badge_color": "#00897b",
        "description": "Every deep RL algorithm is a neural network trained with a special loss function. Without understanding backpropagation, activations, normalisation, and the PyTorch training loop, the deep RL math will be opaque. This module gives you the toolkit used inside DQN, PPO, and AlphaGo.",
        "algorithms": ["MLP / CNN / RNN / LSTM", "BatchNorm & LayerNorm", "Adam, SGD, RMSProp", "PyTorch autograd"],
        "key_papers": ["Goodfellow et al. — Deep Learning (book)", "Andrej Karpathy — micrograd", "PyTorch docs"],
        "outcome": "Build a working MLP classifier in PyTorch from scratch; understand autograd.",
    },
    # ── CLASSICAL RL ───────────────────────────────────────────────────────
    {
        "group": "🎲 Classical RL",
        "group_color": "#6a1b9a",
        "page": "dp",
        "icon": "🧮",
        "title": "Dynamic Programming",
        "subtitle": "Bellman equations, policy evaluation, policy/value iteration, GPI",
        "color": "#6a1b9a",
        "difficulty": 2,
        "time": "1 week",
        "prereqs": "Math foundations",
        "badge": "Stage 1",
        "badge_color": "#6a1b9a",
        "description": "DP introduces the two most important concepts in all of RL: the Bellman equation and the policy-value relationship. Even though DP requires a full model, understanding it makes every model-free algorithm intuitive. Q-learning IS value iteration without the model.",
        "algorithms": ["Policy Evaluation", "Policy Improvement", "Policy Iteration", "Value Iteration", "Async DP"],
        "key_papers": ["Sutton & Barto Ch. 3–4", "Bellman (1957)", "Howard (1960)"],
        "outcome": "Solve 4×4 GridWorld analytically; explain why Policy Iteration converges in few steps.",
    },
    {
        "group": "🎲 Classical RL",
        "group_color": "#6a1b9a",
        "page": "mc",
        "icon": "🎲",
        "title": "Monte Carlo Methods",
        "subtitle": "Episode-based value estimation, control, off-policy importance sampling",
        "color": "#7c4dff",
        "difficulty": 3,
        "time": "1 week",
        "prereqs": "Dynamic Programming",
        "badge": "Stage 2",
        "badge_color": "#7c4dff",
        "description": "MC methods show how to learn V(s) and Q(s,a) purely from experience — no environment model needed. Off-policy MC introduces importance sampling, the mathematical foundation of PPO and experience replay.",
        "algorithms": ["First-Visit MC", "Every-Visit MC", "On-policy control", "Off-policy IS", "Weighted IS"],
        "key_papers": ["Sutton & Barto Ch. 5", "Rubinstein (1981) — Simulation"],
        "outcome": "Implement off-policy MC with IS weights; explain why the baseline does not bias the gradient.",
    },
    {
        "group": "🎲 Classical RL",
        "group_color": "#6a1b9a",
        "page": "td",
        "icon": "⚡",
        "title": "Temporal-Difference Learning",
        "subtitle": "TD(0), SARSA, Q-Learning, Expected SARSA, n-step, eligibility traces",
        "color": "#e65100",
        "difficulty": 3,
        "time": "1–2 weeks",
        "prereqs": "Monte Carlo Methods",
        "badge": "Stage 3",
        "badge_color": "#e65100",
        "description": "TD learning is the heart of modern RL. SARSA and Q-Learning are the tabular ancestors of DQN and PPO. Understanding the on-policy vs off-policy distinction here is what makes DQN vs PPO intuitive later.",
        "algorithms": ["TD(0)", "SARSA", "Q-Learning", "Expected SARSA", "Double Q-Learning", "n-step TD", "SARSA(λ)"],
        "key_papers": ["Sutton & Barto Ch. 6–7, 12", "Watkins & Dayan (1992)", "Singh et al. (1996)"],
        "outcome": "Run CliffWalking; explain WHY SARSA takes the safe path and Q-Learning takes the cliff edge.",
    },
    # ── DEEP RL ────────────────────────────────────────────────────────────
    {
        "group": "🚀 Deep RL",
        "group_color": "#1565c0",
        "page": "vbrl",
        "icon": "🎮",
        "title": "Value-Based Deep RL",
        "subtitle": "DQN → Rainbow: Atari from raw pixels with 7 progressive improvements",
        "color": "#1565c0",
        "difficulty": 3,
        "time": "2–3 weeks",
        "prereqs": "TD Learning + Deep Learning",
        "badge": "Stage 4",
        "badge_color": "#1565c0",
        "description": "DQN (2015) showed that tabular Q-Learning + CNN + experience replay + target networks = superhuman Atari performance. The Rainbow paper demonstrates 6 targeted improvements are additive and synergistic.",
        "algorithms": ["DQN", "Double DQN", "Dueling DQN", "PER", "C51", "IQN", "Rainbow"],
        "key_papers": ["Mnih et al. 2015 (DQN)", "van Hasselt et al. 2016 (DDQN)", "Hessel et al. 2018 (Rainbow)"],
        "outcome": "Implement DQN from scratch on CartPole; ablate target network and observe training collapse.",
    },
    {
        "group": "🚀 Deep RL",
        "group_color": "#1565c0",
        "page": "continuous",
        "icon": "🎯",
        "title": "Continuous Control: DDPG & TD3",
        "subtitle": "Deterministic policy gradients, twin critics, delayed updates, target smoothing",
        "color": "#0288d1",
        "difficulty": 3,
        "time": "1–2 weeks",
        "prereqs": "Value-Based Deep RL",
        "badge": "Stage 4b",
        "badge_color": "#0288d1",
        "description": "DQN cannot handle continuous actions — DDPG (2015) solved this with a deterministic actor. TD3 (2018) fixes DDPG's three failure modes with targeted surgical changes, becoming the go-to off-policy continuous baseline.",
        "algorithms": ["DDPG", "TD3 (twin Q + delayed + smoothing)"],
        "key_papers": ["Lillicrap et al. 2015 (DDPG)", "Fujimoto et al. 2018 (TD3)", "Silver et al. 2014 (DPG)"],
        "outcome": "Implement TD3 on Pendulum; verify twin critics reduce Q overestimation.",
    },
    {
        "group": "🚀 Deep RL",
        "group_color": "#1565c0",
        "page": "ac",
        "icon": "🎭",
        "title": "Actor-Critic & Policy Gradient",
        "subtitle": "REINFORCE, A2C/A3C, PPO, TRPO, SAC — the family behind ChatGPT RLHF",
        "color": "#7c4dff",
        "difficulty": 4,
        "time": "2–3 weeks",
        "prereqs": "Value-Based Deep RL",
        "badge": "Stage 5",
        "badge_color": "#7c4dff",
        "description": "Policy gradient methods directly parameterise π(a|s) with a neural network, enabling robotics, locomotion, and language model alignment. PPO is the single most deployed RL algorithm in the world today.",
        "algorithms": ["REINFORCE", "Actor-Critic", "A2C", "A3C", "PPO", "TRPO", "SAC"],
        "key_papers": ["Schulman et al. 2017 (PPO)", "Schulman et al. 2015 (TRPO)", "Haarnoja et al. 2018 (SAC)"],
        "outcome": "Implement PPO with GAE from scratch; run on a MuJoCo locomotion task.",
    },
    {
        "group": "🚀 Deep RL",
        "group_color": "#1565c0",
        "page": "imitation",
        "icon": "🎓",
        "title": "Imitation Learning",
        "subtitle": "BC, DAgger, GAIL, AIRL, Inverse RL — learn from expert demonstrations",
        "color": "#ad1457",
        "difficulty": 3,
        "time": "1–2 weeks",
        "prereqs": "Actor-Critic",
        "badge": "Stage 6",
        "badge_color": "#ad1457",
        "description": "When reward is hard to specify but expert demonstrations exist, imitation learning recovers the policy or reward function. GAIL uses adversarial training to match trajectory distributions without explicit rewards.",
        "algorithms": ["Behaviour Cloning", "DAgger", "GAIL", "AIRL", "MaxEnt IRL"],
        "key_papers": ["Ho & Ermon 2016 (GAIL)", "Fu et al. 2018 (AIRL)", "Pomerleau 1989 (ALVINN/BC)"],
        "outcome": "Train a BC baseline; show distribution shift failure; fix with DAgger.",
    },
    # ── TIER 1 ─────────────────────────────────────────────────────────────
    {
        "group": "🔬 Tier 1 — Critical Extensions",
        "group_color": "#e65100",
        "page": "mbrl",
        "icon": "🏗️",
        "title": "Model-Based RL",
        "subtitle": "Dyna-Q, World Models, MuZero, DreamerV3, TD-MPC2 — 10–100× sample efficiency",
        "color": "#e65100",
        "difficulty": 4,
        "time": "2–3 weeks",
        "prereqs": "Actor-Critic",
        "badge": "Tier 1",
        "badge_color": "#e65100",
        "description": "Model-based RL learns a world model and plans inside imagination, achieving 10–100× better sample efficiency than model-free methods. DreamerV3 (2023) trains in latent space to handle visual inputs at scale.",
        "algorithms": ["Dyna-Q", "World Models (Ha & Schmidhuber)", "MuZero", "DreamerV3", "TD-MPC2", "PETS / MPC"],
        "key_papers": ["Hafner et al. 2023 (DreamerV3)", "Schrittwieser et al. 2020 (MuZero)", "Sutton 1991 (Dyna)"],
        "outcome": "Implement Dyna-Q; show 10× fewer steps vs model-free Q-learning on GridWorld.",
    },
    {
        "group": "🔬 Tier 1 — Critical Extensions",
        "group_color": "#e65100",
        "page": "offline",
        "icon": "📦",
        "title": "Offline / Batch RL",
        "subtitle": "BC, CQL, IQL, Decision Transformer, TD3+BC — learning from fixed datasets",
        "color": "#00897b",
        "difficulty": 4,
        "time": "2 weeks",
        "prereqs": "Value-Based Deep RL + Actor-Critic",
        "badge": "Tier 1",
        "badge_color": "#00897b",
        "description": "Offline RL enables learning from logged data without environment interaction — critical for healthcare, autonomous driving, and finance. The main challenge is distributional shift: the learned policy must not visit unsupported state-action pairs.",
        "algorithms": ["Behaviour Cloning", "CQL", "IQL", "Decision Transformer", "TD3+BC", "Cal-QL"],
        "key_papers": ["Kumar et al. 2020 (CQL)", "Kostrikov et al. 2021 (IQL)", "Chen et al. 2021 (DT)"],
        "outcome": "Train CQL on a D4RL dataset; measure constraint violation vs unconstrained offline Q-learning.",
    },
    {
        "group": "🔬 Tier 1 — Critical Extensions",
        "group_color": "#e65100",
        "page": "explore",
        "icon": "🔍",
        "title": "Exploration Methods",
        "subtitle": "UCB, Thompson Sampling, ICM, RND, Count-based — solving hard exploration",
        "color": "#f57f17",
        "difficulty": 3,
        "time": "1–2 weeks",
        "prereqs": "Value-Based Deep RL",
        "badge": "Tier 1",
        "badge_color": "#f57f17",
        "description": "Exploration is the unsolved core of RL. In sparse-reward environments like Montezuma's Revenge, random exploration almost never discovers reward. Intrinsic motivation (ICM, RND) adds a curiosity bonus that drives systematic state-space coverage.",
        "algorithms": ["ε-greedy", "UCB1", "Thompson Sampling", "Count-based", "ICM", "RND", "Go-Explore"],
        "key_papers": ["Pathak et al. 2017 (ICM)", "Burda et al. 2018 (RND)", "Ecoffet et al. 2021 (Go-Explore)"],
        "outcome": "Implement RND; compare vs ε-greedy on sparse-reward Atari game.",
    },
    # ── TIER 2 ─────────────────────────────────────────────────────────────
    {
        "group": "⚡ Tier 2 — Specialisations",
        "group_color": "#6a1b9a",
        "page": "advanced",
        "icon": "🚀",
        "title": "Advanced Specialisations",
        "subtitle": "MARL, Hierarchical RL, Safe RL, Meta-RL — real-world deployment patterns",
        "color": "#6a1b9a",
        "difficulty": 5,
        "time": "4–6 weeks",
        "prereqs": "Actor-Critic + Exploration",
        "badge": "Tier 2",
        "badge_color": "#6a1b9a",
        "description": "Real-world deployments almost always need one specialisation: multi-agent coordination, temporal abstraction for long-horizon tasks, constraint satisfaction for safety, or fast adaptation to new tasks. Pick the one relevant to your domain.",
        "algorithms": ["MADDPG", "QMIX", "MAPPO", "Options", "HER", "CPO", "Lagrangian PPO", "CBF", "MAML", "RL²"],
        "key_papers": ["Lowe et al. 2017 (MADDPG)", "Rashid et al. 2018 (QMIX)", "Finn et al. 2017 (MAML)"],
        "outcome": "Run QMIX on SMAC OR implement MAML on few-shot navigation.",
    },
    {
        "group": "⚡ Tier 2 — Specialisations",
        "group_color": "#6a1b9a",
        "page": "transfer",
        "icon": "🔄",
        "title": "Transfer, Multi-Task & Modern Training",
        "subtitle": "Continual RL, PCGrad, PBT, GRPO (2025), RLVR — from DeepSeek-R1 to lifelong agents",
        "color": "#f57f17",
        "difficulty": 4,
        "time": "2–3 weeks",
        "prereqs": "Actor-Critic",
        "badge": "Tier 2+",
        "badge_color": "#f57f17",
        "description": "GRPO (2025) replaces PPO's critic with a group-relative baseline, reducing memory by 40% and powering DeepSeek-R1. RLVR eliminates the reward model entirely for verifiable tasks. EWC solves catastrophic forgetting for lifelong agents.",
        "algorithms": ["EWC", "Progressive Nets", "Experience Replay", "PCGrad", "GradNorm", "PBT", "GRPO", "RLVR"],
        "key_papers": ["Jaderberg et al. 2017 (PBT)", "DeepSeek 2025 (GRPO/R1)", "Kirkpatrick et al. 2017 (EWC)"],
        "outcome": "Implement GRPO training loop; compare to PPO on a math QA task.",
    },
    # ── TIER 3 ─────────────────────────────────────────────────────────────
    {
        "group": "🔧 Tier 3 — Engineering",
        "group_color": "#546e7a",
        "page": "engineering",
        "icon": "🔧",
        "title": "Practical RL Engineering",
        "subtitle": "Debugging, reward design, distributed RL, experiment tracking, reproducibility",
        "color": "#546e7a",
        "difficulty": 3,
        "time": "2 weeks (ongoing)",
        "prereqs": "Any Stage 4+ module",
        "badge": "Tier 3",
        "badge_color": "#546e7a",
        "description": "Most RL project failures are engineering failures, not algorithm failures. Diagnosing Q-value explosion, designing safe reward functions, scaling to 1000+ workers with IMPALA/Ape-X, and reproducing results with proper seed protocols separate practitioners from theorists.",
        "algorithms": ["6 failure mode diagnostics", "Potential-based shaping", "IMPALA", "Ape-X", "EnvPool", "Optuna + W&B"],
        "key_papers": ["Henderson et al. 2018 (reproducibility)", "Huang et al. 2022 (PPO details)", "Espeholt et al. 2018 (IMPALA)"],
        "outcome": "Set up W&B tracking; run 5-seed PPO comparison; fix one failure mode from the checklist.",
    },
    # ── TIER 4 ─────────────────────────────────────────────────────────────
    {
        "group": "🌌 Tier 4 — Frontier",
        "group_color": "#ad1457",
        "page": "frontier",
        "icon": "🔬",
        "title": "Frontier RL Research",
        "subtitle": "RLHF, DPO, Diffusion RL, Sim-to-Real, Foundation Models — cutting edge 2025",
        "color": "#ad1457",
        "difficulty": 5,
        "time": "Ongoing",
        "prereqs": "Actor-Critic + Transfer",
        "badge": "Tier 4",
        "badge_color": "#ad1457",
        "description": "RLHF and DPO power every modern AI assistant. Sim-to-real transfer is the core bottleneck for robot deployment. Diffusion models for RL represent an emerging planning paradigm. Foundation model agents challenge classic RL assumptions entirely.",
        "algorithms": ["PPO-RLHF pipeline", "DPO", "Diffuser", "Decision Diffuser", "RT-2 / Gato", "Domain Rand."],
        "key_papers": ["Ouyang et al. 2022 (InstructGPT/RLHF)", "Rafailov et al. 2023 (DPO)", "Reed et al. 2022 (Gato)"],
        "outcome": "Fine-tune a small LLM with PPO/GRPO using TRL; compare RLHF vs DPO on preference alignment.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHM DATA — comprehensive guide
# ─────────────────────────────────────────────────────────────────────────────
ALGORITHMS = [
    # ── Classical ──────────────────────────────────────────────────────────
    {
        "name": "SARSA",
        "family": "Classical TD",
        "family_color": "#e65100",
        "icon": "⚡",
        "action_space": "Discrete",
        "data_regime": "Online on-policy",
        "sample_eff": 2,
        "stability": 3,
        "complexity": 1,
        "maturity": 5,
        "tag": "Foundational",
        "tag_color": "#546e7a",
        "use_when": "Small discrete state-action spaces; safety matters during training; on-policy exploration required.",
        "avoid_when": "Continuous actions; large/visual observation spaces; need for sample efficiency.",
        "key_insight": "Learns the value of the policy <em>actually followed</em> including ε-greedy exploration — safer but suboptimal vs Q-learning.",
        "key_equation": "Q(s,a) ← Q(s,a) + α[r + γQ(s′,a′) − Q(s,a)]",
        "hyperparams": "α (lr), γ (discount), ε (exploration)",
        "best_for": "CliffWalking, tabular control, safe exploration demos.",
    },
    {
        "name": "Q-Learning",
        "family": "Classical TD",
        "family_color": "#e65100",
        "icon": "⚡",
        "action_space": "Discrete",
        "data_regime": "Online off-policy",
        "sample_eff": 2,
        "stability": 3,
        "complexity": 1,
        "maturity": 5,
        "tag": "Foundational",
        "tag_color": "#546e7a",
        "use_when": "Tabular discrete tasks; off-policy data reuse; optimal policy needed regardless of exploration.",
        "avoid_when": "Continuous actions; overestimation in noisy envs (use Double Q); large state spaces (use DQN).",
        "key_insight": "Learns the <em>greedy optimal</em> Q* regardless of the behavior policy — the ancestor of DQN.",
        "key_equation": "Q(s,a) ← Q(s,a) + α[r + γ max_a′ Q(s′,a′) − Q(s,a)]",
        "hyperparams": "α (lr), γ (discount), ε (exploration decay)",
        "best_for": "GridWorld, FrozenLake, Taxi, tabular benchmarks.",
    },
    # ── Value-Based Deep RL ────────────────────────────────────────────────
    {
        "name": "DQN",
        "family": "Value-Based Deep RL",
        "family_color": "#1565c0",
        "icon": "🎮",
        "action_space": "Discrete",
        "data_regime": "Online off-policy",
        "sample_eff": 3,
        "stability": 4,
        "complexity": 2,
        "maturity": 5,
        "tag": "Industry Standard",
        "tag_color": "#1565c0",
        "use_when": "Discrete actions with high-dimensional (pixel) observations; standard deep RL baseline.",
        "avoid_when": "Continuous actions (use TD3/SAC); very sparse reward (add RND); multi-agent.",
        "key_insight": "Q-learning + CNN + replay buffer + target network = superhuman Atari from pixels (2015 breakthrough).",
        "key_equation": "y = r + γ max_a′ Q_θ̄(s′,a′) ; L = (y − Q_θ(s,a))²",
        "hyperparams": "lr (3e-4), γ (.99), buffer size (1M), target update (10K steps), batch (32)",
        "best_for": "Atari 57, discrete game agents, visual discrete control.",
    },
    {
        "name": "Rainbow",
        "family": "Value-Based Deep RL",
        "family_color": "#1565c0",
        "icon": "🌈",
        "action_space": "Discrete",
        "data_regime": "Online off-policy",
        "sample_eff": 5,
        "stability": 5,
        "complexity": 4,
        "maturity": 5,
        "tag": "Best Discrete",
        "tag_color": "#4caf50",
        "use_when": "Discrete benchmarks where performance matters; state-of-the-art discrete RL needed.",
        "avoid_when": "Continuous actions; need simplicity; compute-constrained — Rainbow is 5× heavier than DQN.",
        "key_insight": "Combines 6 DQN improvements (DDQN + Dueling + PER + n-step + C51 + NoisyNets) — additive gains.",
        "key_equation": "All 6 components; uses distributional Bellman with categorical cross-entropy loss.",
        "hyperparams": "All DQN params + α/β (PER), n (multi-step), N_atoms (C51), V_min/V_max",
        "best_for": "Atari-57, high-performance discrete RL benchmarks.",
    },
    # ── Policy Gradient / Actor-Critic ─────────────────────────────────────
    {
        "name": "PPO",
        "family": "Actor-Critic",
        "family_color": "#7c4dff",
        "icon": "🎭",
        "action_space": "Discrete + Continuous",
        "data_regime": "Online on-policy",
        "sample_eff": 3,
        "stability": 5,
        "complexity": 2,
        "maturity": 5,
        "tag": "Default Choice",
        "tag_color": "#4caf50",
        "use_when": "<b>Default algorithm</b> for new problems. Fast simulators, RLHF, stable training, both action spaces.",
        "avoid_when": "Expensive env interactions (use SAC/TD3); strictly continuous control with heavy computation.",
        "key_insight": "Clipped surrogate objective prevents destructive policy updates — simple, stable, widely deployed.",
        "key_equation": "L = E[min(r_t A_t, clip(r_t, 1±ε) A_t)] − c₁ L_V + c₂ H(π)",
        "hyperparams": "lr (3e-4), clip ε (.2), GAE λ (.95), epochs (10), batch (64), γ (.99), ent_coef (.01)",
        "best_for": "RLHF, MuJoCo, robotics sim, games, Atari discrete, production RL.",
    },
    {
        "name": "SAC",
        "family": "Actor-Critic",
        "family_color": "#7c4dff",
        "icon": "🌡️",
        "action_space": "Continuous",
        "data_regime": "Online off-policy",
        "sample_eff": 5,
        "stability": 5,
        "complexity": 3,
        "maturity": 5,
        "tag": "Best Continuous",
        "tag_color": "#4caf50",
        "use_when": "Continuous control with expensive interactions; maximum sample efficiency; robust exploration needed.",
        "avoid_when": "Discrete actions (use PPO/DQN); very simple tasks where PPO is faster to implement.",
        "key_insight": "Maximises reward + entropy simultaneously — stochastic policy avoids local optima, auto-tuned α.",
        "key_equation": "J(π) = E[Σ γᵗ(r_t + α H(π(·|s_t)))]",
        "hyperparams": "lr (3e-4), auto α, τ (.005), buffer (1M), batch (256), hidden (256×2)",
        "best_for": "Real robot control, MuJoCo locomotion, dexterous manipulation.",
    },
    {
        "name": "TD3",
        "family": "Actor-Critic",
        "family_color": "#7c4dff",
        "icon": "🎯",
        "action_space": "Continuous",
        "data_regime": "Online off-policy",
        "sample_eff": 4,
        "stability": 5,
        "complexity": 2,
        "maturity": 5,
        "tag": "Strong Baseline",
        "tag_color": "#0288d1",
        "use_when": "Continuous control; need simpler than SAC; deterministic policy preferred for precision tasks.",
        "avoid_when": "Discrete actions; heavy exploration needed (SAC handles this better).",
        "key_insight": "Three fixes over DDPG: twin critics (min), delayed actor updates (÷2), target policy smoothing.",
        "key_equation": "y = r + γ min_j Q_j(s′, μ̄(s′) + clip(N(0,σ), −c, c))",
        "hyperparams": "lr (3e-4), policy_delay (2), policy_noise (.2), noise_clip (.5), τ (.005)",
        "best_for": "DMControl suite, continuous robotics benchmarks.",
    },
    {
        "name": "TRPO",
        "family": "Actor-Critic",
        "family_color": "#7c4dff",
        "icon": "🔐",
        "action_space": "Discrete + Continuous",
        "data_regime": "Online on-policy",
        "sample_eff": 3,
        "stability": 5,
        "complexity": 4,
        "maturity": 4,
        "tag": "Research",
        "tag_color": "#546e7a",
        "use_when": "Need principled KL trust region; theoretical guarantees; monotonic improvement required.",
        "avoid_when": "Need engineering simplicity (use PPO instead — same idea, 10× simpler).",
        "key_insight": "Constrains KL divergence between old and new policy — provably monotonic improvement.",
        "key_equation": "max E[r_t A_t] s.t. E[KL(π_old ∥ π_new)] ≤ δ",
        "hyperparams": "δ (KL constraint, .01), damping (0.1), conjugate gradient steps (10)",
        "best_for": "Policy optimisation research; safety-critical on-policy training.",
    },
    # ── Off-policy continuous ──────────────────────────────────────────────
    {
        "name": "DDPG",
        "family": "Actor-Critic",
        "family_color": "#7c4dff",
        "icon": "📉",
        "action_space": "Continuous",
        "data_regime": "Online off-policy",
        "sample_eff": 3,
        "stability": 2,
        "complexity": 2,
        "maturity": 4,
        "tag": "Baseline Only",
        "tag_color": "#ef5350",
        "use_when": "Baseline comparison or teaching DDPG concepts. <b>Use TD3 in practice.</b>",
        "avoid_when": "Production use — TD3 strictly dominates via its three fixes.",
        "key_insight": "First practical deep RL for continuous actions; deterministic actor + off-policy replay.",
        "key_equation": "∇J ≈ (1/N) Σ ∇_a Q|_{a=μ(s)} · ∇_θ μ_θ(s)",
        "hyperparams": "lr_actor (1e-4), lr_critic (1e-3), τ (.001), σ (OU noise .2)",
        "best_for": "Teaching actor-critic concepts; DDPG ablation studies.",
    },
    # ── Model-Based ───────────────────────────────────────────────────────
    {
        "name": "DreamerV3",
        "family": "Model-Based RL",
        "family_color": "#e65100",
        "icon": "🌌",
        "action_space": "Discrete + Continuous",
        "data_regime": "Online off-policy (+ imagination)",
        "sample_eff": 5,
        "stability": 4,
        "complexity": 5,
        "maturity": 4,
        "tag": "Sample Champion",
        "tag_color": "#4caf50",
        "use_when": "Physical robots or expensive simulators; pixel observations; sample efficiency is critical.",
        "avoid_when": "Simple environments; need interpretability; model errors are catastrophic.",
        "key_insight": "Learns a latent world model; trains policy entirely in imagination — 10–100× fewer real env steps.",
        "key_equation": "L = L_recon + L_KL + L_reward + L_continue",
        "hyperparams": "RSSM latent dim (1024), imagine horizon (15), lr (1e-4), batch (16 seq of 64)",
        "best_for": "Atari, DMControl visual, real robot manipulation, DeepMind lab.",
    },
    # ── Offline RL ────────────────────────────────────────────────────────
    {
        "name": "CQL",
        "family": "Offline RL",
        "family_color": "#00897b",
        "icon": "📦",
        "action_space": "Discrete + Continuous",
        "data_regime": "Offline only",
        "sample_eff": 5,
        "stability": 4,
        "complexity": 3,
        "maturity": 5,
        "tag": "Offline Standard",
        "tag_color": "#00897b",
        "use_when": "Fixed dataset (healthcare, robotics logs, autonomous driving); no environment access.",
        "avoid_when": "Can interact with environment — online RL gives better performance.",
        "key_insight": "Penalises Q-values for OOD actions to prevent overestimation without env feedback.",
        "key_equation": "L_CQL = E[log Σ_a exp(Q(s,a)) − Q(s,a_data)] + ½ (Q − y_Bellman)²",
        "hyperparams": "α (CQL weight, 1–10), lr (3e-4), dataset coverage threshold",
        "best_for": "D4RL benchmark, logged robotics data, healthcare decision support.",
    },
    {
        "name": "Decision Transformer",
        "family": "Offline RL",
        "family_color": "#00897b",
        "icon": "🤖",
        "action_space": "Discrete + Continuous",
        "data_regime": "Offline sequence modelling",
        "sample_eff": 5,
        "stability": 5,
        "complexity": 3,
        "maturity": 4,
        "tag": "Sequence Model",
        "tag_color": "#00897b",
        "use_when": "Large diverse offline datasets; multi-task offline learning; GPT-style architecture.",
        "avoid_when": "Small datasets; dynamic stitching of sub-optimal trajectories needed (use IQL/CQL).",
        "key_insight": "Frames RL as sequence modelling with return-conditioned GPT — no Bellman backups needed.",
        "key_equation": "π(a|s, g, h) via autoregressive transformer on (R̂,s,a) tokens",
        "hyperparams": "Context length K (20), n_layers (3–6), n_heads (8), lr (1e-4)",
        "best_for": "D4RL, Atari offline, multi-game DT, robotic manipulation logs.",
    },
    # ── Exploration ───────────────────────────────────────────────────────
    {
        "name": "RND",
        "family": "Exploration",
        "family_color": "#f57f17",
        "icon": "🔍",
        "action_space": "Discrete + Continuous",
        "data_regime": "Online add-on",
        "sample_eff": 4,
        "stability": 4,
        "complexity": 2,
        "maturity": 4,
        "tag": "Exploration Bonus",
        "tag_color": "#f57f17",
        "use_when": "Sparse rewards; hard exploration; add on top of any base RL algorithm.",
        "avoid_when": "Dense reward environments where exploration is not the bottleneck.",
        "key_insight": "Prediction error of a random target network = novelty signal; decays as states are revisited.",
        "key_equation": "r_int = ‖f_θ(o) − T(o)‖² (T fixed random, f_θ trained to match)",
        "hyperparams": "int_coef (.5–1.0), normalise running mean/std of intrinsic reward",
        "best_for": "Montezuma's Revenge, sparse Atari, hard exploration robotics.",
    },
    # ── Multi-Agent ───────────────────────────────────────────────────────
    {
        "name": "QMIX",
        "family": "Multi-Agent RL",
        "family_color": "#0288d1",
        "icon": "🌐",
        "action_space": "Discrete",
        "data_regime": "Online off-policy (CTDE)",
        "sample_eff": 4,
        "stability": 4,
        "complexity": 3,
        "maturity": 5,
        "tag": "MARL Standard",
        "tag_color": "#0288d1",
        "use_when": "Cooperative multi-agent with discrete actions; centralised training + decentralised execution.",
        "avoid_when": "Continuous multi-agent (use MADDPG); competitive (use self-play PPO); single agent.",
        "key_insight": "Monotonic mixing network decomposes Q_tot into per-agent Q_i — global optimal = local optima.",
        "key_equation": "∂Q_tot/∂Q_i ≥ 0 ∀i (IGM property ensures decentralised execution is globally optimal)",
        "hyperparams": "mixing_embed_dim (32), lr (5e-4), buffer (5K eps), ε-anneal (50K steps)",
        "best_for": "StarCraft Multi-Agent Challenge (SMAC), cooperative grid games.",
    },
    {
        "name": "MAPPO",
        "family": "Multi-Agent RL",
        "family_color": "#0288d1",
        "icon": "🌐",
        "action_space": "Discrete + Continuous",
        "data_regime": "Online on-policy (CTDE)",
        "sample_eff": 3,
        "stability": 5,
        "complexity": 2,
        "maturity": 4,
        "tag": "MARL Baseline",
        "tag_color": "#0288d1",
        "use_when": "Cooperative multi-agent with continuous actions; simpler than QMIX; competitive with QMIX on many tasks.",
        "avoid_when": "Discrete tasks with complex credit assignment (QMIX better); competitive.",
        "key_insight": "Standard PPO with centralised critic V(s) — surprisingly competitive with specialised MARL methods.",
        "key_equation": "Standard PPO objective; critic sees global state s (all agents' observations)",
        "hyperparams": "Same as PPO; data_chunk_length (10), share_policy (True/False)",
        "best_for": "SMAC, MPE, Hanabi, multi-robot coordination.",
    },
    # ── Safe RL ───────────────────────────────────────────────────────────
    {
        "name": "CPO / Lagrangian PPO",
        "family": "Safe RL",
        "family_color": "#c62828",
        "icon": "🛡️",
        "action_space": "Discrete + Continuous",
        "data_regime": "Online on-policy",
        "sample_eff": 3,
        "stability": 4,
        "complexity": 3,
        "maturity": 4,
        "tag": "Safety-Critical",
        "tag_color": "#ef5350",
        "use_when": "Physical robot deployment; constraint violations have real costs; safety budgets must be met.",
        "avoid_when": "Constraints cannot be measured reliably; toy environments without real risk.",
        "key_insight": "CMDP: maximise reward subject to cost constraint J_C(π) ≤ d; Lagrangian λ adapts penalty automatically.",
        "key_equation": "max J(π) s.t. J_C(π) ≤ d ; λ_{t+1} = max(0, λ_t + η(J_C − d))",
        "hyperparams": "λ_init (1.0), cost_limit d, η_λ (lr for Lagrange multiplier)",
        "best_for": "Safety Gym, constrained locomotion, autonomous vehicle training.",
    },
    # ── Meta-RL ───────────────────────────────────────────────────────────
    {
        "name": "MAML",
        "family": "Meta-RL",
        "family_color": "#6a1b9a",
        "icon": "🧬",
        "action_space": "Discrete + Continuous",
        "data_regime": "Online (meta-training + few-shot)",
        "sample_eff": 4,
        "stability": 3,
        "complexity": 4,
        "maturity": 4,
        "tag": "Few-Shot Adapt.",
        "tag_color": "#6a1b9a",
        "use_when": "Many related tasks available for meta-training; rapid adaptation (5–10 gradient steps) needed at test time.",
        "avoid_when": "Single-task training; cannot sample from task distribution; compute-constrained.",
        "key_insight": "Finds parameter init θ* such that K gradient steps on any task τ yields good performance.",
        "key_equation": "θ* = argmin_θ Σ_τ L_τ(θ − α∇L_τ(θ))",
        "hyperparams": "α (inner lr, .1), β (outer lr, 3e-4), K (inner steps, 1–5), N (tasks per meta-batch)",
        "best_for": "Few-shot robot navigation, personalised recommendation, clinical adaptation.",
    },
    # ── LLM / RLHF ───────────────────────────────────────────────────────
    {
        "name": "PPO-RLHF",
        "family": "Human Feedback",
        "family_color": "#ad1457",
        "icon": "💬",
        "action_space": "Token sequences (LLM)",
        "data_regime": "Online on-policy + reward model",
        "sample_eff": 3,
        "stability": 4,
        "complexity": 4,
        "maturity": 5,
        "tag": "LLM Alignment",
        "tag_color": "#ad1457",
        "use_when": "Aligning LLMs to human preferences via learned reward model; complex non-verifiable tasks.",
        "avoid_when": "Tasks with verifiable answers (use RLVR/GRPO); simple preference data (use DPO).",
        "key_insight": "SFT → reward model (Bradley-Terry) → PPO with KL penalty vs reference model.",
        "key_equation": "r_total = r_φ(x,y) − β KL(π_θ ∥ π_ref)",
        "hyperparams": "β (KL coeff, .1–.5), rm_lr (1e-5), ppo_lr (1e-6), clip_ratio (.2)",
        "best_for": "ChatGPT, Claude, Gemini alignment; instruction following; RLHF pipelines.",
    },
    {
        "name": "GRPO",
        "family": "Human Feedback",
        "family_color": "#ad1457",
        "icon": "⚡",
        "action_space": "Token sequences (LLM)",
        "data_regime": "Online on-policy (no critic)",
        "sample_eff": 4,
        "stability": 5,
        "complexity": 2,
        "maturity": 4,
        "tag": "2025 Default",
        "tag_color": "#4caf50",
        "use_when": "LLM RL training; want to eliminate critic (40% memory saving); verifiable or preference rewards.",
        "avoid_when": "Classic RL environments where value function improves credit assignment significantly.",
        "key_insight": "Group-relative advantage A_i = (r_i − mean(r)) / std(r) — no value network needed.",
        "key_equation": "A_i = (r_i − μ_r)/σ_r ; clip ratio same as PPO",
        "hyperparams": "G (group size, 8), β (KL, .01), clip ε (.2), lr (5e-7)",
        "best_for": "DeepSeek-R1, Qwen-2.5, math/code LLM training, open RLHF frameworks.",
    },
    {
        "name": "DPO",
        "family": "Human Feedback",
        "family_color": "#ad1457",
        "icon": "🎯",
        "action_space": "Token sequences (LLM)",
        "data_regime": "Offline preference pairs",
        "sample_eff": 5,
        "stability": 5,
        "complexity": 1,
        "maturity": 4,
        "tag": "Offline Alignment",
        "tag_color": "#ad1457",
        "use_when": "Preference pairs available; want supervised-style training; no reward model; simpler pipeline.",
        "avoid_when": "Complex tasks needing online exploration; only verifiable rewards available (use RLVR).",
        "key_insight": "Reparameterises RLHF reward in terms of policy ratio — trains purely supervised on preference pairs.",
        "key_equation": "L_DPO = −E[log σ(β log π_θ(y_w)/π_ref(y_w) − β log π_θ(y_l)/π_ref(y_l))]",
        "hyperparams": "β (.1–.5), lr (1e-6 to 5e-6), batch (16–128)",
        "best_for": "Preference fine-tuning, chat model alignment, constitutional AI.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# RENDER: LEARNING HUB
# ─────────────────────────────────────────────────────────────────────────────
def render_learning_hub(button_prefix):

    # ── Header ──────────────────────────────────────────────────────────────
    st.html("""
    <div style="background:linear-gradient(135deg,#0d1a2e,#0a2a1a,#1a0d2e);
                border:1px solid #1e2e4e;border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem">
        <h2 style="color:white;margin:0;font-size:1.7rem;font-weight:800">📚 RL Curriculum — Complete Module Guide</h2>
        <p style="color:#8090b0;margin:.5rem 0 0;font-size:.95rem">
            16 modules · 4 tiers · 60+ algorithms · Structured from first principles to frontier research.
            Each card shows prerequisites, key algorithms, estimated time, and key papers.
        </p>
        <div style="display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.9rem">
            <span style="background:#4caf5022;color:#4caf50;border:1px solid #4caf5044;border-radius:6px;padding:.2rem .7rem;font-size:.78rem;font-weight:600">★ Difficulty 1–2 · Accessible</span>
            <span style="background:#ffa72622;color:#ffa726;border:1px solid #ffa72644;border-radius:6px;padding:.2rem .7rem;font-size:.78rem;font-weight:600">★★★ Difficulty 3 · Intermediate</span>
            <span style="background:#ef535022;color:#ef5350;border:1px solid #ef535044;border-radius:6px;padding:.2rem .7rem;font-size:.78rem;font-weight:600">★★★★★ Difficulty 5 · Advanced</span>
        </div>
    </div>
    """)

    # ── Group modules ────────────────────────────────────────────────────────
    groups = {}
    for m in MODULES:
        g = m["group"]
        groups.setdefault(g, {"color": m["group_color"], "modules": []})
        groups[g]["modules"].append(m)

    for group_name, group_data in groups.items():
        gc = group_data["color"]
        mods = group_data["modules"]
        st.markdown(f"""
        <div style="background:{gc}14;border-left:4px solid {gc};border-radius:0 10px 10px 0;
                    padding:.7rem 1.2rem;margin:1.4rem 0 .8rem;display:flex;align-items:center;gap:.8rem">
            <h3 style="color:white;margin:0;font-size:1.15rem;font-weight:700">{group_name}</h3>
            <span style="color:{gc};font-size:.8rem;font-weight:600">{len(mods)} module{"s" if len(mods)!=1 else ""}</span>
        </div>
        """, unsafe_allow_html=True)

        for m in mods:
            col_card, col_btn = st.columns([5, 1])
            with col_card:
                diff_stars = _stars(m["difficulty"], color="#ffa726" if m["difficulty"] <= 3 else "#ef5350")
                alg_list = " · ".join(m["algorithms"][:5]) + (" ···" if len(m["algorithms"]) > 5 else "")
                papers_list = " · ".join(m["key_papers"][:2])
                tags_html = _tag_row(
                    (m["badge"], m["badge_color"]),
                    (m["action_space"] if "action_space" in m else "Both", "#546e7a"),
                    (m["time"], "#0288d1"),
                )
                st.markdown(f"""
                <div style="background:#0e0e1e;border:1px solid {m['color']}33;border-radius:14px;
                            padding:1.3rem 1.5rem;margin-bottom:.5rem;
                            border-left:4px solid {m['color']};
                            transition:all .2s">
                    <div style="display:flex;align-items:flex-start;gap:1rem;margin-bottom:.8rem">
                        <span style="font-size:2rem;line-height:1">{m['icon']}</span>
                        <div style="flex:1">
                            <div style="display:flex;align-items:center;gap:.8rem;flex-wrap:wrap;margin-bottom:.3rem">
                                <b style="color:white;font-size:1.05rem;font-weight:700">{m['title']}</b>
                                {tags_html}
                            </div>
                            <p style="color:#7080a0;font-size:.84rem;margin:0 0 .3rem">{m['subtitle']}</p>
                            <div style="display:flex;align-items:center;gap:.5rem">
                                <span style="font-size:.82rem">{diff_stars}</span>
                                <span style="color:#5060a0;font-size:.78rem">Difficulty {m['difficulty']}/5</span>
                            </div>
                        </div>
                    </div>
                    <p style="color:#9090b8;font-size:.87rem;line-height:1.65;margin:0 0 .9rem">{m['description']}</p>
                    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.8rem">
                        <div style="background:#0a0a18;border-radius:8px;padding:.7rem .9rem;border:1px solid #1a1a30">
                            <div style="color:{m['color']};font-size:.72rem;font-weight:700;margin-bottom:.3rem">⚙️ KEY ALGORITHMS</div>
                            <div style="color:#a0a8c8;font-size:.8rem;line-height:1.5">{alg_list}</div>
                        </div>
                        <div style="background:#0a0a18;border-radius:8px;padding:.7rem .9rem;border:1px solid #1a1a30">
                            <div style="color:#4caf50;font-size:.72rem;font-weight:700;margin-bottom:.3rem">📚 KEY PAPERS</div>
                            <div style="color:#a0a8c8;font-size:.8rem;line-height:1.5">{papers_list}</div>
                        </div>
                        <div style="background:#0a0a18;border-radius:8px;padding:.7rem .9rem;border:1px solid #1a1a30">
                            <div style="color:#ffa726;font-size:.72rem;font-weight:700;margin-bottom:.3rem">🔑 PREREQUISITE</div>
                            <div style="color:#a0a8c8;font-size:.8rem;line-height:1.5">{m['prereqs']}</div>
                        </div>
                    </div>
                    <div style="background:#0a1a0a;border:1px solid #1a3a1a;border-radius:8px;
                                padding:.6rem .9rem;margin-top:.7rem">
                        <span style="color:#4caf50;font-size:.72rem;font-weight:700">✅ MILESTONE: </span>
                        <span style="color:#8aa890;font-size:.81rem">{m['outcome']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_btn:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button(f"{m['icon']} Open", key=f"{button_prefix}_{m['page']}", use_container_width=True):
                    go(m["page"])


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: ALGORITHM SELECTION GUIDE
# ─────────────────────────────────────────────────────────────────────────────
def render_algorithm_selection_guide():
    import pandas as pd

    st.html("""
    <div style="background:linear-gradient(135deg,#0d1428,#1a0d28);
                border:1px solid #1e1e3e;border-radius:16px;padding:1.5rem 2rem;margin:1.5rem 0 1rem">
        <h2 style="color:white;margin:0;font-size:1.6rem;font-weight:800">🧭 Algorithm Selection Guide</h2>
        <p style="color:#8090b0;margin:.5rem 0 0;font-size:.92rem">
            20 algorithms · 8 families · Rated on 4 dimensions · Decision criteria + key equations + hyperparameters.
        </p>
    </div>
    """)

    # ── Family filter ────────────────────────────────────────────────────────
    families = sorted(set(a["family"] for a in ALGORITHMS))
    fam_colors = {a["family"]: a["family_color"] for a in ALGORITHMS}

    selected_fam = st.selectbox(
        "Filter by algorithm family",
        ["All Families"] + families,
        key="alg_guide_fam_filter"
    )
    show_algs = ALGORITHMS if selected_fam == "All Families" else [a for a in ALGORITHMS if a["family"] == selected_fam]

    # ── Ratings legend ───────────────────────────────────────────────────────
    st.html("""
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-bottom:1rem;
                background:#0a0a18;border-radius:10px;padding:.8rem 1.2rem;border:1px solid #1a1a2e">
        <div><span style="color:#ffa726;font-weight:700;font-size:.82rem">⚡ SAMPLE EFF.</span>
             <span style="color:#606080;font-size:.78rem"> — how many env steps to converge</span></div>
        <div><span style="color:#4caf50;font-weight:700;font-size:.82rem">🏋️ STABILITY</span>
             <span style="color:#606080;font-size:.78rem"> — training robustness across seeds</span></div>
        <div><span style="color:#90caf9;font-weight:700;font-size:.82rem">🔧 COMPLEXITY</span>
             <span style="color:#606080;font-size:.78rem"> — implementation difficulty</span></div>
        <div><span style="color:#ce93d8;font-weight:700;font-size:.82rem">🎓 MATURITY</span>
             <span style="color:#606080;font-size:.78rem"> — production readiness</span></div>
        <div><span style="color:#546e7a;font-size:.78rem">★★★★★ = excellent, ★☆☆☆☆ = poor</span></div>
    </div>
    """)

    # ── Group by family ──────────────────────────────────────────────────────
    shown_families = {}
    for a in show_algs:
        shown_families.setdefault(a["family"], []).append(a)

    for fam, algs in shown_families.items():
        fc = fam_colors[fam]
        st.html(f"""
        <div style="background:{fc}12;border-left:3px solid {fc};border-radius:0 8px 8px 0;
                    padding:.55rem 1rem;margin:1.2rem 0 .6rem;display:flex;gap:.8rem;align-items:center">
            <b style="color:{fc};font-size:1rem">{fam}</b>
            <span style="color:#404060;font-size:.8rem">— {len(algs)} algorithm{"s" if len(algs)!=1 else ""}</span>
        </div>
        """)

        for a in algs:
            fc2 = a["family_color"]
            # Build rating bars
            def rating_bar(n, color):
                filled = f'<span style="color:{color}">{"■" * n}</span>'
                empty  = f'<span style="color:#1a1a2e">{"■" * (5 - n)}</span>'
                return filled + empty

            st.html(f"""
            <div style="background:#0c0c1c;border:1px solid {fc2}2a;border-radius:12px;
                        padding:1.1rem 1.3rem;margin-bottom:.75rem;border-left:3px solid {fc2}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;
                            flex-wrap:wrap;gap:.5rem;margin-bottom:.8rem">
                    <div style="display:flex;align-items:center;gap:.7rem">
                        <span style="font-size:1.4rem">{a['icon']}</span>
                        <div>
                            <b style="color:white;font-size:1rem">{a['name']}</b>
                            <div style="margin-top:.15rem">
                                {_badge(a['tag'], a['tag_color'])}
                                {_badge(a['action_space'], '#2a3a5e')}
                                {_badge(a['data_regime'], '#1a2a3e')}
                            </div>
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(4,auto);gap:.5rem 1.2rem;
                                background:#08081a;border-radius:8px;padding:.5rem .9rem;font-size:.78rem">
                        <div><span style="color:#ffa726">⚡ Smpl Eff</span><br>{rating_bar(a['sample_eff'], '#ffa726')}</div>
                        <div><span style="color:#4caf50">🏋️ Stable</span><br>{rating_bar(a['stability'], '#4caf50')}</div>
                        <div><span style="color:#90caf9">🔧 Cmplx</span><br>{rating_bar(a['complexity'], '#ef5350')}</div>
                        <div><span style="color:#ce93d8">🎓 Mature</span><br>{rating_bar(a['maturity'], '#ce93d8')}</div>
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:.7rem">
                    <div style="background:#0a1a0a;border-radius:8px;padding:.65rem .85rem;border:1px solid #1a3a1a">
                        <div style="color:#4caf50;font-size:.7rem;font-weight:700;margin-bottom:.25rem">✅ USE WHEN</div>
                        <div style="color:#90b090;font-size:.82rem;line-height:1.5">{a['use_when']}</div>
                    </div>
                    <div style="background:#1a0a0a;border-radius:8px;padding:.65rem .85rem;border:1px solid #3a1a1a">
                        <div style="color:#ef5350;font-size:.7rem;font-weight:700;margin-bottom:.25rem">❌ AVOID WHEN</div>
                        <div style="color:#b08080;font-size:.82rem;line-height:1.5">{a['avoid_when']}</div>
                    </div>
                </div>

                <div style="display:grid;grid-template-columns:2fr 1fr 1fr;gap:.7rem">
                    <div style="background:#0a0d1a;border-radius:8px;padding:.65rem .85rem;border:1px solid #1a1e3a">
                        <div style="color:#90caf9;font-size:.7rem;font-weight:700;margin-bottom:.25rem">💡 KEY INSIGHT</div>
                        <div style="color:#8090b8;font-size:.82rem;line-height:1.5">{a['key_insight']}</div>
                    </div>
                    <div style="background:#0a0a0a;border-radius:8px;padding:.65rem .85rem;border:1px solid #1a1a1a;
                                font-family:monospace">
                        <div style="color:{fc2};font-size:.7rem;font-weight:700;margin-bottom:.25rem">📐 EQUATION</div>
                        <div style="color:#7080a8;font-size:.75rem;line-height:1.55;word-break:break-all">{a['key_equation']}</div>
                    </div>
                    <div style="background:#0a0a18;border-radius:8px;padding:.65rem .85rem;border:1px solid #1a1a30">
                        <div style="color:#ffa726;font-size:.7rem;font-weight:700;margin-bottom:.25rem">⚙️ HYPERPARAMS</div>
                        <div style="color:#8080a8;font-size:.75rem;line-height:1.55">{a['hyperparams']}</div>
                    </div>
                </div>
            </div>
            """)

    # ── Quick summary table ──────────────────────────────────────────────────
    st.markdown("### 📊 Algorithm Properties at a Glance")
    rows = []
    for a in ALGORITHMS:
        def r2s(n): return "★"*n + "☆"*(5-n)
        rows.append({
            "Algorithm": f"{a['icon']} {a['name']}",
            "Family": a["family"],
            "Action Space": a["action_space"],
            "Data Regime": a["data_regime"],
            "Sample Eff.": r2s(a["sample_eff"]),
            "Stability": r2s(a["stability"]),
            "Complexity": r2s(a["complexity"]),
            "Tag": a["tag"],
        })
    st.dataframe(
        __import__("pandas").DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: QUICK DECISION TREE
# ─────────────────────────────────────────────────────────────────────────────
def render_decision_tree():
    st.html("""
    <div style="background:linear-gradient(135deg,#0d1a0d,#1a0d1a);
                border:1px solid #1e3e1e;border-radius:16px;padding:1.5rem 2rem;margin:1.5rem 0 1rem">
        <h2 style="color:white;margin:0;font-size:1.6rem;font-weight:800">🎯 Algorithm Selection Decision Tree</h2>
        <p style="color:#8090b0;margin:.5rem 0 0;font-size:.92rem">
            Answer 7 questions in order to find the right algorithm family for your problem.
            Each question eliminates families until you have a concrete recommendation.
        </p>
    </div>
    """)

    questions = [
        {
            "q": "Q1 — Do you have complete access to environment dynamics p(s′,r|s,a)?",
            "icon": "🧮",
            "yes_label": "YES → Model-Based (Known Dynamics)",
            "yes_color": "#6a1b9a",
            "yes_body": "Use <b>Dynamic Programming</b> (Policy/Value Iteration). Gives exact optimal policy. Applies when you have the full MDP: board games with known rules, traffic simulators, known physics.",
            "yes_algs": ["Policy Iteration", "Value Iteration", "Modified Policy Iteration"],
            "no_label": "NO → Learn from experience (most real problems)",
            "no_color": "#546e7a",
            "no_body": "You must interact with the environment. Proceed to Q2.",
        },
        {
            "q": "Q2 — Do you have a fixed dataset and CANNOT interact with the environment?",
            "icon": "📦",
            "yes_label": "YES → Offline RL",
            "yes_color": "#00897b",
            "yes_body": "Use <b>CQL</b> (general offline, strong constraint). <b>IQL</b> (avoids OOD Q queries). <b>Decision Transformer</b> (large diverse datasets). <b>BC</b> as fast baseline. <b>RLVR</b> if task has verifiable ground truth.",
            "yes_algs": ["Behaviour Cloning", "CQL", "IQL", "Decision Transformer", "TD3+BC"],
            "no_label": "NO → Online RL (can interact with environment)",
            "no_color": "#546e7a",
            "no_body": "You can collect new data during training. Proceed to Q3.",
        },
        {
            "q": "Q3 — Do you have expert demonstrations and a hard-to-specify reward?",
            "icon": "🎓",
            "yes_label": "YES → Imitation Learning",
            "yes_color": "#ad1457",
            "yes_body": "<b>BC</b> first (fast, 10 min to train). <b>DAgger</b> if expert can be queried on new states. <b>GAIL</b> for adversarial distribution matching without a reward. <b>AIRL</b> if you need a transferable reward.",
            "yes_algs": ["Behaviour Cloning", "DAgger", "GAIL", "AIRL", "MaxEnt IRL"],
            "no_label": "NO → Standard RL with reward signal",
            "no_color": "#546e7a",
            "no_body": "A reward function is available. Proceed to Q4.",
        },
        {
            "q": "Q4 — Are you training or aligning a large language model?",
            "icon": "💬",
            "yes_label": "YES → LLM RL (RLHF / GRPO / DPO)",
            "yes_color": "#e65100",
            "yes_body": "<b>GRPO</b> (2025 default — no critic, 40% less memory, DeepSeek-R1). <b>RLVR</b> if answers are verifiable (math, code — no reward model needed). <b>PPO-RLHF</b> with reward model for complex preferences. <b>DPO</b> if only preference pairs available (no online rollout).",
            "yes_algs": ["GRPO", "RLVR", "PPO-RLHF", "DPO"],
            "no_label": "NO → Standard control/game RL",
            "no_color": "#546e7a",
            "no_body": "Proceed to Q5 for action space decision.",
        },
        {
            "q": "Q5 — Is your action space discrete (finite choices) or continuous (real-valued)?",
            "icon": "🎮",
            "yes_label": "DISCRETE → Value-Based or PPO",
            "yes_color": "#1565c0",
            "yes_body": "<b>DQN/Rainbow</b> for high-dimensional pixel observations. <b>PPO</b> as stable on-policy alternative (also works for discrete). For multi-agent discrete: <b>QMIX</b> (cooperative) or <b>MAPPO</b> (competitive/mixed). For very sparse rewards: add <b>RND</b> exploration bonus.",
            "yes_algs": ["DQN", "Double DQN", "Rainbow", "PPO (discrete)", "QMIX"],
            "no_label": "CONTINUOUS → Actor-Critic",
            "no_color": "#7c4dff",
            "no_body": "<b>SAC</b> (best sample efficiency + max-entropy exploration). <b>TD3</b> (simpler, deterministic). <b>PPO</b> (stable on-policy). Never use vanilla DDPG in production — TD3 is strictly better. Proceed to Q6.",
        },
        {
            "q": "Q6 — Is environment interaction expensive? (physical robot, slow sim, clinical trial)",
            "icon": "💸",
            "yes_label": "YES → Sample-Efficient Off-Policy or Model-Based",
            "yes_color": "#00838f",
            "yes_body": "<b>SAC + replay buffer</b> (reuses every transition many times). <b>DreamerV3</b> for maximum efficiency — learns world model, trains policy in imagination (10–100× fewer real steps). <b>Pre-train offline with CQL/IQL</b> then fine-tune online with <b>Cal-QL</b>.",
            "yes_algs": ["SAC", "TD3", "DreamerV3", "TD-MPC2", "CQL → Cal-QL"],
            "no_label": "NO → On-policy PPO is fine",
            "no_color": "#f57f17",
            "no_body": "Fast simulators make on-policy competitive. <b>PPO</b> is the safe default. <b>A2C</b> for parallel workers. Only switch to SAC if training > 100M steps.",
        },
        {
            "q": "Q7 — Does your problem involve multiple agents OR very long sequential subgoals?",
            "icon": "🚀",
            "yes_label": "YES → Advanced Specialisations",
            "yes_color": "#6a1b9a",
            "yes_body": "<b>Multi-agent (cooperative):</b> QMIX (discrete), MADDPG (continuous), MAPPO (both). <b>Multi-agent (competitive):</b> Self-play PPO. <b>Long-horizon/subgoals:</b> Goal-conditioned RL + HER. <b>Safety constraints:</b> CPO or Lagrangian PPO + CBF shielding.",
            "yes_algs": ["QMIX", "MADDPG", "MAPPO", "HER", "CPO", "Lagrangian PPO"],
            "no_label": "✅ Your answer is from Q5/Q6",
            "no_color": "#4caf50",
            "no_body": "The algorithm from Q5–Q6 is your answer. Standard single-agent RL applies.",
        },
    ]

    for i, node in enumerate(questions):
        st.markdown(f"""
        <div style="background:#0c0c1c;border:1px solid #1e1e3e;border-radius:14px;
                    padding:1.2rem 1.5rem;margin-bottom:.7rem;
                    border-left:4px solid #2a2a5e">
            <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.9rem">
                <span style="background:#1a1a3e;border:1px solid #2a2a5e;border-radius:8px;
                             padding:.3rem .8rem;color:#8090c8;font-size:.78rem;font-weight:700">
                    QUESTION {i+1} of {len(questions)}
                </span>
                <span style="font-size:1.3rem">{node['icon']}</span>
                <b style="color:white;font-size:.98rem">{node['q']}</b>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem">
                <div style="background:{node['yes_color']}12;border:1px solid {node['yes_color']}30;
                            border-radius:10px;padding:.9rem 1.1rem">
                    <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem">
                        <span style="background:{node['yes_color']}22;color:{node['yes_color']};
                                     border-radius:6px;padding:.18rem .6rem;font-size:.75rem;font-weight:700">
                            {node['yes_label']}
                        </span>
                    </div>
                    <div style="color:#9098b8;font-size:.83rem;line-height:1.6;margin-bottom:.6rem">{node['yes_body']}</div>
                    <div style="display:flex;flex-wrap:wrap;gap:.3rem">
                        {"".join(_badge(a, node['yes_color']) for a in node['yes_algs'])}
                    </div>
                </div>
                <div style="background:#1a1a2a;border:1px solid {node['no_color']}22;
                            border-radius:10px;padding:.9rem 1.1rem">
                    <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem">
                        <span style="background:{node['no_color']}22;color:{node['no_color']};
                                     border-radius:6px;padding:.18rem .6rem;font-size:.75rem;font-weight:700">
                            {node['no_label']}
                        </span>
                    </div>
                    <div style="color:#6070a0;font-size:.83rem;line-height:1.6">{node['no_body']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Use-case quick reference ─────────────────────────────────────────────
    st.markdown("### ⚡ Use-Case Quick Reference")
    cases = [
        ("🕹️", "Atari / Discrete Games",     "DQN → Double DQN → Rainbow",              "#1565c0"),
        ("🤖", "Robotics (Simulation)",       "PPO (stable) or SAC (efficient)",          "#7c4dff"),
        ("🤖", "Robotics (Real Hardware)",    "SAC + replay buffer → DreamerV3",          "#00897b"),
        ("💬", "LLM Alignment (2025)",        "GRPO + RLVR (no critic, no reward model)", "#ff7043"),
        ("💬", "LLM Alignment (Classic)",     "PPO + reward model (InstructGPT style)",   "#e65100"),
        ("🏃", "Continuous Locomotion",       "SAC → TD3",                                "#00838f"),
        ("🏭", "Offline Dataset Only",        "CQL → IQL → Decision Transformer",         "#546e7a"),
        ("📸", "Pixel Observations",          "CNN encoder + DQN or PPO",                 "#ad1457"),
        ("♟️", "Board Games / Perfect Info",  "MCTS + value network (AlphaZero/MuZero)",  "#6a1b9a"),
        ("🎓", "Have Expert Demonstrations",  "BC → DAgger → GAIL → AIRL",               "#ad1457"),
        ("🌍", "Sample Efficiency Critical",  "DreamerV3 world model (10–100× less data)","#e65100"),
        ("🌐", "Multi-Agent Cooperative",     "MAPPO (any) or QMIX (discrete)",           "#0288d1"),
        ("🌐", "Multi-Agent Competitive",     "Self-play PPO → Nash DQN",                 "#0288d1"),
        ("🛡️", "Safety-Critical Deployment", "Lagrangian PPO + CBF shielding + CPO",     "#ef5350"),
        ("🔁", "Continual / Lifelong",        "EWC + experience replay",                  "#f57f17"),
        ("🎯", "Multi-Task Learning",         "MT-SAC + PCGrad / GradNorm",               "#7c4dff"),
        ("🧬", "Fast Adaptation (few-shot)",  "MAML → PEARL → RL²",                      "#6a1b9a"),
        ("📈", "Hard Exploration (sparse)",   "Base algo + RND / ICM / Go-Explore",       "#f57f17"),
    ]
    cols = st.columns(2)
    for i, (emoji, case, alg, color) in enumerate(cases):
        with cols[i % 2]:
            st.html(f"""
            <div style="background:#0c0c1c;border:1px solid #1a1a2e;border-radius:10px;
                        padding:.65rem 1rem;margin-bottom:.4rem;
                        display:flex;align-items:center;gap:.9rem">
                <span style="font-size:1.25rem;min-width:1.5rem">{emoji}</span>
                <div style="flex:1;min-width:0">
                    <div style="color:white;font-size:.84rem;font-weight:600;white-space:nowrap;
                                overflow:hidden;text-overflow:ellipsis">{case}</div>
                    <div style="color:{color};font-size:.78rem;font-weight:500;margin-top:.1rem">{alg}</div>
                </div>
            </div>
            """)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: METHOD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def render_method_comparison():
    import pandas as pd

    st.html("""
    <div style="background:linear-gradient(135deg,#0a0d1a,#1a0a0d);
                border:1px solid #1e1e3e;border-radius:16px;padding:1.5rem 2rem;margin:1.5rem 0 1rem">
        <h2 style="color:white;margin:0;font-size:1.6rem;font-weight:800">⚖️ Method Family Comparison</h2>
        <p style="color:#8090b0;margin:.5rem 0 0;font-size:.92rem">
            A structured comparison of 5 fundamental axes across all major RL algorithm families.
        </p>
    </div>
    """)

    # ── Axis 1: Dynamics Knowledge ───────────────────────────────────────────
    st.markdown("#### Axis 1 — Does the agent need to know / learn environment dynamics?")
    cards_1 = [
        ("#6a1b9a", "🧮 Model-Based (Known)",
         "Full dynamics p(s′,r|s,a) available at training time.",
         ["✅ Zero model error", "✅ Exact optimal policy", "✅ No env interaction needed"],
         ["❌ Rarely available in practice", "❌ Scales poorly to complex MDPs"],
         "Policy Iteration · Value Iteration · Exact DP"),
        ("#e65100", "🏗️ Model-Based (Learned)",
         "World model is learned from env interactions, then used for planning.",
         ["✅ 10–100× fewer real env steps", "✅ Generalises via imagination", "✅ Works on pixels (DreamerV3)"],
         ["❌ Model errors compound", "❌ High implementation complexity"],
         "Dyna-Q · DreamerV3 · MuZero · TD-MPC2 · PETS"),
        ("#1565c0", "🎮 Model-Free Online",
         "Learns directly from env interactions with no model.",
         ["✅ No model errors", "✅ Works in any environment", "✅ Simple to implement"],
         ["❌ Needs many env steps", "❌ Cannot plan ahead"],
         "DQN · PPO · SAC · TD3 · A2C"),
        ("#00897b", "📦 Offline (No Env Access)",
         "Fixed historical dataset; zero online interaction.",
         ["✅ Healthcare/finance safe", "✅ Uses existing logs", "✅ No exploration risk"],
         ["❌ Bounded by data coverage", "❌ Distributional shift challenge"],
         "BC · CQL · IQL · Decision Transformer · TD3+BC"),
    ]
    cols = st.columns(4)
    for col, (color, title, desc, pros, cons, algs) in zip(cols, cards_1):
        with col:
            pros_html = "".join(f'<div style="color:#80c080;font-size:.75rem;line-height:1.5">{p}</div>' for p in pros)
            cons_html = "".join(f'<div style="color:#c08080;font-size:.75rem;line-height:1.5">{c}</div>' for c in cons)
            st.html(f"""
            <div style="background:{color}10;border:1px solid {color}35;border-radius:12px;padding:.9rem;
                        height:100%;border-top:3px solid {color}">
                <b style="color:{color};font-size:.88rem">{title}</b>
                <p style="color:#6070a0;font-size:.78rem;margin:.4rem 0 .6rem;line-height:1.4">{desc}</p>
                {pros_html}
                <div style="height:.5rem"></div>
                {cons_html}
                <div style="margin-top:.6rem;background:#08081a;border-radius:6px;padding:.4rem .6rem">
                    <div style="color:{color};font-size:.68rem;font-weight:700;margin-bottom:.15rem">ALGORITHMS</div>
                    <div style="color:#5060a0;font-size:.72rem;line-height:1.4">{algs}</div>
                </div>
            </div>
            """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Axis 2: Update Timing ───────────────────────────────────────────────
    st.markdown("#### Axis 2 — When does learning happen?")
    timing = [
        ("#7c4dff", "🎲 Monte Carlo", "After full episode", "True G_t", "Zero", "High", "Episodic only. No model needed. Simple but slow.", "MC Control · Every-Visit MC"),
        ("#e65100", "⚡ TD (1-step)", "Every step", "r + γV(s′)", "Low", "Low", "Online learning. Handles continuing tasks. Backbone of DQN/PPO.", "SARSA · Q-Learning · TD(0)"),
        ("#0288d1", "🪜 n-step TD", "Every n steps", "Σᵢ γⁱrᵢ + γⁿV(sₙ)", "Medium", "Medium", "Bridges TD and MC. Faster credit propagation. Used in Rainbow.", "n-step SARSA · Rainbow n-step"),
        ("#1565c0", "🎮 Batch (replay)", "Mini-batch from buffer", "Bellman target", "Low", "Low", "GPU-efficient. Off-policy reuse. DQN/SAC paradigm.", "DQN · SAC · TD3 · DDPG"),
        ("#4caf50", "🎭 Rollout (on-policy)", "After rollout collection", "GAE advantage", "Low", "Low", "Stable on-policy. PPO/TRPO paradigm.", "PPO · TRPO · A2C · GRPO"),
        ("#ad1457", "📦 Never (offline)", "Never (fixed data)", "Behaviour cloning / Q-constraint", "Zero", "Zero", "No env interaction. Offline RL. CQL/DT paradigm.", "BC · CQL · IQL · DT"),
    ]
    st.html("""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-bottom:1rem">
    """ + "".join(f"""
        <div style="background:#0c0c1c;border:1px solid {c}2a;border-radius:10px;padding:.8rem;border-left:3px solid {c}">
            <b style="color:{c};font-size:.85rem">{icon} {name}</b>
            <div style="color:#505080;font-size:.73rem;margin:.3rem 0"><b style="color:#6070a0">Updates:</b> {when}</div>
            <div style="color:#505080;font-size:.73rem"><b style="color:#6070a0">Target:</b> {target}</div>
            <div style="color:#505080;font-size:.73rem"><b style="color:#4caf50">Bias:</b> {bias} &nbsp;
                <b style="color:#ffa726">Var:</b> {var}</div>
            <div style="color:#6070a0;font-size:.77rem;margin-top:.4rem;line-height:1.4">{note}</div>
            <div style="color:{c};font-size:.7rem;margin-top:.35rem">{algs}</div>
        </div>
    """ for c, icon, name, when, target, bias, var, note, algs in [
        (t[0], t[1].split()[0], " ".join(t[1].split()[1:]), t[2], t[3], t[4], t[5], t[6], t[7]) for t in timing
    ]) + "</div>")

    # ── Axis 3: On/Off/Offline ──────────────────────────────────────────────
    st.markdown("#### Axis 3 — Data regime: On-Policy vs Off-Policy vs Offline")
    cols3 = st.columns(3)
    regime_cards = [
        ("#f57f17", "🔵 On-Policy",
         "Uses ONLY data from current policy. Discards old data after each update.",
         [("Stability", "⭐⭐⭐⭐⭐"), ("Sample Efficiency", "⭐⭐⭐"), ("Memory", "⭐⭐⭐⭐⭐")],
         "SARSA · MC · A2C · PPO · TRPO · GRPO",
         "PPO is simplest safe default. Data collection + training are tightly coupled."),
        ("#00838f", "🟠 Off-Policy",
         "Replay buffer holds data from ANY past policy. Can reuse old transitions many times.",
         [("Stability", "⭐⭐⭐⭐"), ("Sample Efficiency", "⭐⭐⭐⭐⭐"), ("Memory", "⭐⭐⭐")],
         "Q-Learning · DQN · SAC · TD3 · DDPG",
         "SAC is the go-to for continuous control. DQN for discrete. More complex IS corrections."),
        ("#ad1457", "📦 Offline-Only",
         "Static historical dataset. Zero environment interaction during training.",
         [("Stability", "⭐⭐⭐⭐"), ("Coverage Limit", "⭐⭐⭐⭐⭐"), ("Deployment Safety", "⭐⭐⭐⭐⭐")],
         "BC · CQL · IQL · Decision Transformer · TD3+BC",
         "Use for healthcare, finance, robotics logs. Main challenge: distributional shift."),
    ]
    for col, (c, icon_name, desc, metrics, algs, note) in zip(cols3, regime_cards):
        with col:
            metric_html = "".join(
                f'<div style="display:flex;justify-content:space-between;font-size:.76rem;'
                f'border-bottom:1px solid #1a1a2e;padding:.2rem 0">'
                f'<span style="color:#6070a0">{m}</span>'
                f'<span style="color:{c}">{v}</span></div>'
                for m, v in metrics
            )
            st.html(f"""
            <div style="background:{c}0d;border:1px solid {c}30;border-radius:12px;padding:1rem;
                        height:100%;border-top:3px solid {c}">
                <b style="color:{c}">{icon_name}</b>
                <p style="color:#6070a0;font-size:.8rem;margin:.4rem 0 .6rem;line-height:1.4">{desc}</p>
                {metric_html}
                <div style="margin-top:.6rem;background:#08081a;border-radius:6px;padding:.4rem .6rem">
                    <div style="color:{c};font-size:.68rem;font-weight:700">ALGORITHMS</div>
                    <div style="color:#5060a0;font-size:.73rem">{algs}</div>
                </div>
                <div style="color:#404060;font-size:.75rem;margin-top:.5rem;line-height:1.4;font-style:italic">{note}</div>
            </div>
            """)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Master comparison table ──────────────────────────────────────────────
    st.markdown("#### Full Algorithm Matrix (24 algorithms · 10 dimensions)")
    rows = [
        ["Policy Iteration","DP","Any","Model-Based","Zero","Zero","N/A (exact)","★★★★★","Known MDPs, grid games"],
        ["SARSA","Classical TD","Discrete","On-Policy","Low","Low","★★☆☆☆","★★★☆☆","Safe tabular control"],
        ["Q-Learning","Classical TD","Discrete","Off-Policy","Low","Low","★★☆☆☆","★★★☆☆","Tabular optimal control"],
        ["DQN","Value-Based","Discrete","Off-Policy","Low","Low","★★★☆☆","★★★★☆","Atari, visual discrete"],
        ["Double DQN","Value-Based","Discrete","Off-Policy","Low","Low","★★★☆☆","★★★★☆","Atari, reduce overestimation"],
        ["Dueling DQN","Value-Based","Discrete","Off-Policy","Low","Low","★★★☆☆","★★★★☆","Atari, many-action discrete"],
        ["PER","Value-Based (add-on)","Any","Off-Policy","Low","Low","★★★★☆","★★★★☆","DQN improvement, sparse events"],
        ["Rainbow","Value-Based","Discrete","Off-Policy","Low","Low","★★★★★","★★★★★","Best discrete RL benchmark"],
        ["REINFORCE","Policy Grad.","Both","On-Policy","Zero","Very High","★★☆☆☆","★★☆☆☆","Teaching, toy episodic tasks"],
        ["A2C / A3C","Actor-Critic","Both","On-Policy","Low","Medium","★★★☆☆","★★★☆☆","Parallel envs, fast iteration"],
        ["PPO","Actor-Critic","Both","On-Policy","Low","Low","★★★☆☆","★★★★★","Default: RLHF, robotics, games"],
        ["TRPO","Actor-Critic","Both","On-Policy","Low","Low","★★★☆☆","★★★★★","Research, monotonic improvement"],
        ["DDPG","Actor-Critic","Continuous","Off-Policy","Low","Low","★★★☆☆","★★★☆☆","Baseline only (use TD3 instead)"],
        ["TD3","Actor-Critic","Continuous","Off-Policy","Low","Low","★★★★☆","★★★★★","Continuous control baseline"],
        ["SAC","Actor-Critic","Continuous","Off-Policy","Low","Low","★★★★★","★★★★★","Real robots, continuous SoTA"],
        ["Dyna-Q","Model-Based","Discrete","Online+Imagine","Low","Low","★★★★☆","★★★☆☆","Sample-efficient tabular RL"],
        ["DreamerV3","Model-Based","Both","Online+Imagine","Low","Low","★★★★★","★★★★☆","Visual, physical robot, 10–100× eff."],
        ["BC","Offline","Both","Offline","—","—","★★☆☆☆","★★★★☆","Fast imitation baseline"],
        ["CQL","Offline","Both","Offline","—","—","★★★★☆","★★★★☆","D4RL, no-env offline RL"],
        ["IQL","Offline","Both","Offline","—","—","★★★★☆","★★★★☆","Offline fine-tune, in-sample Q"],
        ["Decision Transformer","Offline Seq","Both","Offline","—","—","★★★☆☆","★★★★☆","Multi-task large offline datasets"],
        ["PPO-RLHF","LLM RL","Tokens","On-Policy","Low","Low","★★★☆☆","★★★★★","InstructGPT, Claude, LLM align."],
        ["GRPO","LLM RL","Tokens","On-Policy","Low","Low","★★★★☆","★★★★★","DeepSeek-R1, 2025 default LLM RL"],
        ["DPO","LLM Pref.","Tokens","Offline","—","—","★★★★★","★★★★★","Pref. fine-tuning without RL loop"],
    ]
    df = pd.DataFrame(rows, columns=[
        "Algorithm","Family","Action Space","Data Regime",
        "Bias","Variance","Sample Eff.","Stability","Best For"
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_home():
    st.markdown("""
    <div style="background:linear-gradient(140deg,#08081a 0%,#12083a 35%,#080d1a 65%,#080d10 100%);
                border:1px solid #1a1a3e;border-radius:20px;padding:2.5rem 3rem 2rem;margin-bottom:1.5rem">
        <div style="display:flex;align-items:center;gap:1.3rem;margin-bottom:.9rem">
            <span style="font-size:3.2rem">🧠</span>
            <div>
                <h1 style="color:white;margin:0;font-size:2.5rem;font-weight:800;letter-spacing:-.5px;
                            background:linear-gradient(135deg,#ffffff,#9090e0);-webkit-background-clip:text;
                            -webkit-text-fill-color:transparent">
                    Deep RL Learning Portal</h1>
                <p style="color:#7080a8;margin:.35rem 0 0;font-size:1rem">
                    A structured, formula-rich roadmap from zero to professional Deep RL practitioner
                </p>
            </div>
        </div>
        <div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-top:1rem">
            <span style="background:#ffffff0a;border:1px solid #ffffff18;color:#9090c0;border-radius:20px;padding:.28rem .9rem;font-size:.84rem">📚 60+ algorithms</span>
            <span style="background:#ffffff0a;border:1px solid #ffffff18;color:#9090c0;border-radius:20px;padding:.28rem .9rem;font-size:.84rem">📐 130+ interactive charts</span>
            <span style="background:#ffffff0a;border:1px solid #ffffff18;color:#9090c0;border-radius:20px;padding:.28rem .9rem;font-size:.84rem">🧪 5 live environments</span>
            <span style="background:#ffffff0a;border:1px solid #ffffff18;color:#9090c0;border-radius:20px;padding:.28rem .9rem;font-size:.84rem">📖 16 modules · 4 tiers</span>
            <span style="background:#4caf5018;border:1px solid #4caf5035;color:#4caf50;border-radius:20px;padding:.28rem .9rem;font-size:.84rem">✨ GRPO & RLVR — 2025</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_catalog, tab_hub, tab_study, tab_discussion = st.tabs([
        "📚 Module Catalog",
        "📚 Learning Hub",
        "📖 Study Material",
        "💬 Discussion Board",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — MODULE CATALOG
    # ════════════════════════════════════════════════════════════════════
    with tab_catalog:
        render_learning_hub("catalog")

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — LEARNING HUB
    # ════════════════════════════════════════════════════════════════════
    if False:
        st.markdown("""
        <h2 style="color:white;font-size:1.5rem;margin-bottom:.3rem;font-weight:800">
            🗺️ Your Path to Professional Deep RL</h2>
        <p style="color:#7080a0;margin-bottom:1.5rem;font-size:.92rem">
        Follow this sequence. Each stage builds directly on the previous one. Click any stage to open that module.
        </p>
        """, unsafe_allow_html=True)

        stages = [
            {
                "step": "STAGE −1", "icon": "📐", "title": "Math & CS Foundations",
                "subtitle": "Linear algebra, calculus, probability, info theory, Python/NumPy",
                "color": "#00695c", "badge": "START HERE", "badge_color": "#00695c", "duration": "3–4 weeks",
                "why": "Every RL formula relies on one of six mathematical areas. Without this base, every module will feel like magic rather than engineering. Neural networks ARE matrix multiplications. Backprop IS the chain rule. π(a|s) IS a probability distribution.",
                "covers": ["Linear algebra: vectors, matrices, Jacobian", "Calculus: gradients, chain rule, gradient descent", "Probability: distributions, expectations, Bayes", "Information theory: entropy, KL divergence", "Python & NumPy: vectorised ops", "NN math: forward pass, backprop from scratch"],
                "milestone": "Implement 2-layer policy network forward+backward pass in pure NumPy.", "btn_page": "foundations",
            },
            {
                "step": "STAGE 0", "icon": "🧬", "title": "Deep Learning Prerequisites",
                "subtitle": "Autograd, optimisers, normalisation, CNNs, RNNs, PyTorch training loop",
                "color": "#00897b", "badge": "No prior RL needed", "badge_color": "#00897b", "duration": "1–2 weeks",
                "why": "Every deep RL algorithm is a neural network trained with a special loss function. Without understanding backpropagation, activations, normalisation, and the PyTorch training loop, all the deep RL math will be opaque.",
                "covers": ["RNNs & LSTMs (memory for POMDPs)", "BatchNorm & LayerNorm (stable training)", "SGD · Adam · RMSProp optimisers", "PyTorch autograd & full training loop", "CNN architecture for visual RL (DQN)"],
                "milestone": "Build a working MLP classifier in PyTorch from scratch.", "btn_page": "prereq",
            },
            {
                "step": "STAGE 1–3", "icon": "🎲", "title": "Classical RL: DP → MC → TD",
                "subtitle": "Bellman equations, Monte Carlo, SARSA, Q-Learning, n-step, eligibility traces",
                "color": "#6a1b9a", "badge": "Theory foundation", "badge_color": "#6a1b9a", "duration": "3–4 weeks",
                "why": "DP, MC, and TD learning form the mathematical backbone of all modern RL. Q-learning IS value iteration without the model. TD(λ) IS a generalisation of both TD and MC. Understanding these makes every deep RL algorithm intuitive.",
                "covers": ["DP: Policy/Value Iteration, GPI", "MC: First-visit, off-policy IS, weighted IS", "TD: SARSA, Q-learning, Expected SARSA", "Double Q-Learning, n-step TD", "SARSA(λ) & eligibility traces"],
                "milestone": "Run CliffWalking; explain WHY SARSA takes the safe path and Q-learning takes the cliff edge.", "btn_page": "td",
            },
            {
                "step": "STAGE 4–5", "icon": "🎮", "title": "Deep RL: Value-Based + Actor-Critic",
                "subtitle": "DQN → Rainbow, PPO, SAC, TD3 — the core modern RL algorithms",
                "color": "#1565c0", "badge": "Core Deep RL", "badge_color": "#1565c0", "duration": "4–6 weeks",
                "why": "DQN proved that Q-learning + CNN + experience replay = superhuman Atari performance. PPO is the most deployed RL algorithm in the world (RLHF, robotics, games). SAC achieves state-of-the-art sample efficiency on continuous tasks.",
                "covers": ["DQN, Double DQN, Dueling, PER, C51, Rainbow", "REINFORCE, Actor-Critic, A2C, A3C", "PPO (RLHF standard), TRPO (trust region)", "SAC (max entropy, best continuous)", "TD3, DDPG (continuous control)"],
                "milestone": "Implement PPO from scratch; run on MuJoCo; ablate the clipping and observe instability.", "btn_page": "ac",
            },
            {
                "step": "TIER 1", "icon": "🏗️", "title": "Critical Extensions: Model-Based · Offline · Exploration",
                "subtitle": "10–100× sample efficiency, fixed-dataset RL, hard exploration solved",
                "color": "#e65100", "badge": "High Impact", "badge_color": "#e65100", "duration": "3–5 weeks",
                "why": "Model-based RL (DreamerV3) achieves 10–100× better sample efficiency. Offline RL (CQL, IQL) enables learning from fixed datasets without env interaction — critical for healthcare and robotics. Exploration (RND, ICM) solves the fundamental unsolved problem of sparse reward environments.",
                "covers": ["Dyna-Q, World Models, MuZero, DreamerV3", "BC, CQL, IQL, Decision Transformer", "UCB, Thompson Sampling, ICM, RND", "Go-Explore for extreme hard exploration"],
                "milestone": "Train CQL on D4RL; implement RND on Montezuma's Revenge; implement Dyna-Q.", "btn_page": "mbrl",
            },
            {
                "step": "TIER 2", "icon": "🚀", "title": "Specialisations: MARL · HRL · Safe · Meta · Transfer",
                "subtitle": "Multi-agent coordination, long-horizon planning, safe deployment, rapid adaptation",
                "color": "#6a1b9a", "badge": "Deployment", "badge_color": "#6a1b9a", "duration": "4–6 weeks",
                "why": "Real-world deployments almost always need a specialisation: MARL for multi-agent systems, Hierarchical RL for long-horizon tasks, Safe RL for physical deployment, Meta-RL for fast adaptation, GRPO/RLVR for LLM training (2025).",
                "covers": ["MADDPG, QMIX, MAPPO (multi-agent)", "Options, HER, goal-conditioned RL (hierarchical)", "CPO, Lagrangian PPO, CBF (safe RL)", "MAML, RL², PEARL (meta-RL)", "GRPO, RLVR, EWC, PBT (transfer/modern training)"],
                "milestone": "Run QMIX on SMAC OR implement GRPO training loop for a math LLM task.", "btn_page": "advanced",
            },
            {
                "step": "TIER 3", "icon": "🔧", "title": "Engineering: Debug · Reward Design · Distributed · Tracking",
                "subtitle": "The practical skills that separate successful practitioners from theorists",
                "color": "#546e7a", "badge": "Production RL", "badge_color": "#546e7a", "duration": "2 weeks+",
                "why": "Most RL project failures are engineering failures. Diagnosing Q-value explosion, designing reward functions without Goodhart's Law violations, scaling to 1000+ workers with IMPALA, and proper seed protocols are what make deployed systems work.",
                "covers": ["6 RL failure mode diagnostics + fixes", "Potential-based reward shaping theorem", "RLHF reward modelling pipeline", "IMPALA, Ape-X, EnvPool distributed RL", "W&B, Optuna, reproducibility checklist"],
                "milestone": "Set up W&B; run 5-seed PPO comparison; fix one failure mode from the checklist.", "btn_page": "engineering",
            },
            {
                "step": "TIER 4", "icon": "🔬", "title": "Frontier: RLHF · Diffusion RL · Foundation Models · Theory",
                "subtitle": "Research-level topics — the cutting edge of 2025 RL",
                "color": "#ad1457", "badge": "Research level", "badge_color": "#ad1457", "duration": "Ongoing",
                "why": "RLHF and DPO power every modern AI assistant. Sim-to-real transfer is the core bottleneck for robot deployment. Diffusion models for RL represent an emerging planning paradigm. Foundation model agents challenge classic RL assumptions.",
                "covers": ["RLHF: SFT → reward model → PPO pipeline", "DPO: direct preference optimisation", "Domain randomisation + system identification", "Diffuser, Decision Diffuser (diffusion RL)", "PAC-MDP, regret bounds, convergence theory"],
                "milestone": "Fine-tune a small LLM with GRPO using TRL; compare RLHF vs DPO on a preference task.", "btn_page": "frontier",
            },
        ]

        for i, s in enumerate(stages):
            col_l, col_r = st.columns([5, 1])
            with col_l:
                covers_html = "".join(
                    f'<span style="color:#6070a0;font-size:.8rem">▸ {c}</span><br>'
                    for c in s["covers"]
                )
                st.markdown(f"""
                <div style="background:#0c0c1c;border:1px solid {s['color']}30;border-radius:14px;
                            padding:1.3rem 1.6rem;margin-bottom:.7rem;border-left:5px solid {s['color']}">
                    <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.65rem;flex-wrap:wrap">
                        <span style="background:{s['color']}22;color:{s['color']};border-radius:7px;
                                     padding:.2rem .65rem;font-size:.73rem;font-weight:700">{s['step']}</span>
                        <span style="font-size:1.5rem">{s['icon']}</span>
                        <div>
                            <b style="color:white;font-size:1rem;font-weight:700">{s['title']}</b>
                            <br><span style="color:#5060a0;font-size:.83rem">{s['subtitle']}</span>
                        </div>
                        <span style="margin-left:auto;background:{s['badge_color']}18;color:{s['badge_color']};
                                     border:1px solid {s['badge_color']}35;border-radius:20px;
                                     padding:.18rem .75rem;font-size:.72rem;white-space:nowrap">{s['badge']}</span>
                    </div>
                    <p style="color:#7080a0;font-size:.87rem;margin:.4rem 0 .8rem;line-height:1.65">{s['why']}</p>
                    <div style="display:flex;gap:1.5rem;flex-wrap:wrap">
                        <div style="flex:1;min-width:180px">{covers_html}</div>
                        <div style="flex:1;min-width:180px;border-left:1px solid #1a1a2e;padding-left:1.2rem">
                            <div style="color:#ffa726;font-size:.73rem;font-weight:700;margin-bottom:.3rem">⏱ TIME</div>
                            <div style="color:#7080a0;font-size:.86rem;margin-bottom:.6rem">{s['duration']}</div>
                            <div style="background:#0a1a0a;border-radius:7px;padding:.55rem .75rem;border:1px solid #1a2a1a">
                                <div style="color:#4caf50;font-size:.7rem;font-weight:700;margin-bottom:.2rem">✅ MILESTONE</div>
                                <div style="color:#6a8a6a;font-size:.79rem;line-height:1.5">{s['milestone']}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r:
                st.markdown("<br><br>", unsafe_allow_html=True)
                key_safe = s['btn_page'].replace("-","_")
                if st.button(f"Open {s['icon']}", key=f"road_{key_safe}", use_container_width=True):
                    go(s["btn_page"])
            if i < len(stages) - 1:
                st.markdown('<div style="text-align:center;color:#1a1a3a;font-size:1.2rem;margin:.1rem 0">▼ ▼ ▼</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — LEARNING HUB
    # ════════════════════════════════════════════════════════════════════
    with tab_hub:
        hub_tabs = st.tabs([
            "🧭 Algorithm Guide",
            "🎯 Decision Tree",
            "⚖️ Method Comparison",
        ])
        with hub_tabs[0]:
            render_algorithm_selection_guide()
        with hub_tabs[1]:
            render_decision_tree()
        with hub_tabs[2]:
            render_method_comparison()

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — STUDY MATERIAL
    # ════════════════════════════════════════════════════════════════════
    with tab_study:
        mod = load_study_material()
        mod.main_study_material()

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — DISCUSSION BOARD
    # ════════════════════════════════════════════════════════════════════
    with tab_discussion:
        mod = load_discussion_board()
        mod.main_discussion_board()


# ─────────────────────────────────────────────────────────────────────────────
# BACK BUTTON
# ─────────────────────────────────────────────────────────────────────────────
def show_back_bar(module_name, module_color, module_icon):
    col_back, col_title, _ = st.columns([1, 5, 2])
    with col_back:
        if st.button("← Home", key="back_home", use_container_width=True):
            go("home")
    with col_title:
        st.markdown(
            f'<div style="padding:.35rem 0;color:{module_color};font-weight:700;font-size:1.05rem">'
            f'{module_icon} RL Portal › {module_name}</div>',
            unsafe_allow_html=True)
    st.markdown('<hr style="margin:.4rem 0 1rem;border-color:#111120">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LAZY MODULE LOADERS
# ─────────────────────────────────────────────────────────────────────────────
def _load_mod(name, path):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

@st.cache_resource(show_spinner=False)
def load_ac():           return _load_mod("ac_mod", "_ac_mod.py")
@st.cache_resource(show_spinner=False)
def load_vbrl():         return _load_mod("vbrl_mod", "_vbrl_mod.py")
@st.cache_resource(show_spinner=False)
def load_prereq():       return _load_mod("prereq_mod", "_prereq_mod.py")
@st.cache_resource(show_spinner=False)
def load_dp():           return _load_mod("dp_mod", "_dp_mod.py")
@st.cache_resource(show_spinner=False)
def load_mc():           return _load_mod("mc_mod", "_mc_mod.py")
@st.cache_resource(show_spinner=False)
def load_foundations():  return _load_mod("foundations_mod", "_foundations_mod.py")
@st.cache_resource(show_spinner=False)
def load_continuous():   return _load_mod("continuous_mod", "_continuous_mod.py")
@st.cache_resource(show_spinner=False)
def load_imitation():    return _load_mod("imitation_mod", "_imitation_mod.py")
@st.cache_resource(show_spinner=False)
def load_transfer():     return _load_mod("transfer_mod", "_transfer_mod.py")
@st.cache_resource(show_spinner=False)
def load_td():           return _load_mod("td_mod", "_td_mod.py")
@st.cache_resource(show_spinner=False)
def load_mbrl():         return _load_mod("mbrl_mod", "_mbrl_mod.py")
@st.cache_resource(show_spinner=False)
def load_offline():      return _load_mod("offline_mod", "_offline_mod.py")
@st.cache_resource(show_spinner=False)
def load_explore():      return _load_mod("explore_mod", "_explore_mod.py")
@st.cache_resource(show_spinner=False)
def load_advanced():     return _load_mod("advanced_mod", "_advanced_mod.py")
@st.cache_resource(show_spinner=False)
def load_engineering():  return _load_mod("engineering_mod", "_engineering_mod.py")
@st.cache_resource(show_spinner=False)
def load_frontier():     return _load_mod("frontier_mod", "_frontier_mod.py")
@st.cache_resource(show_spinner=False)
def load_study_material():   return _load_mod("study_material_mod", "_study_material_mod.py")
@st.cache_resource(show_spinner=False)
def load_discussion_board(): return _load_mod("discussion_mod", "_discussion_mod.py")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.portal_page

_routes = {
    "home":        (None, None, None, None),
    "foundations": ("Math & CS Foundations", "#80cbc4", "📐", load_foundations),
    "continuous":  ("Continuous Control: DDPG & TD3", "#81d4fa", "🎯", load_continuous),
    "imitation":   ("Imitation Learning", "#f48fb1", "🎓", load_imitation),
    "transfer":    ("Transfer, Multi-Task & Modern Training", "#ffcc80", "🔄", load_transfer),
    "ac":          ("Actor-Critic & Policy Gradient", "#ce93d8", "🎭", load_ac),
    "vbrl":        ("Value-Based Deep RL", "#90caf9", "🎮", load_vbrl),
    "prereq":      ("Deep Learning Prerequisites", "#80cbc4", "🧬", load_prereq),
    "dp":          ("Dynamic Programming", "#ce93d8", "🧮", load_dp),
    "mc":          ("Monte Carlo Methods", "#b39ddb", "🎲", load_mc),
    "td":          ("Temporal-Difference Learning", "#ffb74d", "⚡", load_td),
    "mbrl":        ("Model-Based RL", "#ff7043", "🏗️", load_mbrl),
    "offline":     ("Offline / Batch RL", "#80cbc4", "📦", load_offline),
    "explore":     ("Exploration Methods", "#ffd54f", "🔍", load_explore),
    "advanced":    ("Advanced Specialisations", "#b39ddb", "🚀", load_advanced),
    "engineering": ("Practical RL Engineering", "#90a4ae", "🔧", load_engineering),
    "frontier":    ("Frontier RL Research", "#f48fb1", "🔬", load_frontier),
}

_main_fns = {
    "foundations": "main_foundations",
    "continuous":  "main_continuous",
    "imitation":   "main_imitation",
    "transfer":    "main_transfer",
    "ac":          "main_ac",
    "vbrl":        "main_vbrl",
    "prereq":      "main_prereq",
    "dp":          "main_dp",
    "mc":          "main_mc",
    "td":          "main_td",
    "mbrl":        "main_mbrl",
    "offline":     "main_offline",
    "explore":     "main_explore",
    "advanced":    "main_advanced",
    "engineering": "main_engineering",
    "frontier":    "main_frontier",
}

if page == "home":
    show_home()
elif page in _routes and page != "home":
    name, color, icon, loader = _routes[page]
    show_back_bar(name, color, icon)
    mod = loader()
    fn  = _main_fns.get(page, "main")
    getattr(mod, fn)()
else:
    go("home")
