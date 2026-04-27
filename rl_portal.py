"""
RL Learning Portal — Unified Entry Point
Combines DP, MC, and TD explorers into a single professional educational portal.
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
/* ── Global dark theme ── */
body, .stApp { background: #0a0a14; }

.stTabs [data-baseweb="tab-list"] {
    gap:6px; background:#12121f; border-radius:10px; padding:4px;
}
.stTabs [data-baseweb="tab"] {
    background:#1e1e2e; border-radius:8px; color:#b0b0cc;
    padding:7px 13px; font-weight:600; font-size:.87rem;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#4a148c,#0277bd);
    color:white !important;
}

/* ── Method cards ── */
.method-card {
    background: #12121f;
    border: 1px solid #2a2a3e;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin: .4rem 0;
    transition: border-color .25s, transform .15s;
    cursor: pointer;
}
.method-card:hover {
    border-color: #7c4dff;
    transform: translateY(-2px);
}
.method-card-dp  { border-left: 5px solid #6a1b9a; }
.method-card-mc  { border-left: 5px solid #7c4dff; }
.method-card-td  { border-left: 5px solid #e65100; }
.method-card-pre { border-left: 5px solid #00897b; }

/* ── Back button ── */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #1e1e2e;
    border: 1px solid #2a2a3e;
    color: #b0b0cc;
    border-radius: 8px;
}

/* ── Progress / metric ── */
div[data-testid="metric-container"] {
    background:#1e1e2e; border-radius:10px;
    padding:11px; border:1px solid #2d2d44;
}

/* ── Hide default Streamlit menu on landing ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "portal_page" not in st.session_state:
    st.session_state.portal_page = "home"


def go(page):
    st.session_state.portal_page = page
    # Reset sub-explorer results when switching methods
    for key in ["dp_results", "td_results", "mc_results", "results"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_home():
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(140deg,#0d0d2b 0%,#1a0a3e 40%,#0a1a3e 70%,#0d1a14 100%);
                border:1px solid #2a2a4e;border-radius:20px;padding:2.5rem 3rem 2rem;margin-bottom:1.5rem">
        <div style="display:flex;align-items:center;gap:1.2rem;margin-bottom:.8rem">
            <span style="font-size:3rem">🧠</span>
            <div>
                <h1 style="color:white;margin:0;font-size:2.4rem;font-weight:800;letter-spacing:-0.5px">
                    Deep RL Learning Portal</h1>
                <p style="color:#9e9ebb;margin:.3rem 0 0;font-size:1rem">
                    A structured, formula-rich roadmap from zero to professional Deep RL practitioner
                </p>
            </div>
        </div>
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem">
            <span style="background:#ffffff11;border:1px solid #ffffff22;color:#b0bec5;border-radius:20px;padding:.3rem .9rem;font-size:.85rem">📚 31+ algorithms</span>
            <span style="background:#ffffff11;border:1px solid #ffffff22;color:#b0bec5;border-radius:20px;padding:.3rem .9rem;font-size:.85rem">📐 110+ interactive charts</span>
            <span style="background:#ffffff11;border:1px solid #ffffff22;color:#b0bec5;border-radius:20px;padding:.3rem .9rem;font-size:.85rem">🧪 5 live environments</span>
            <span style="background:#ffffff11;border:1px solid #ffffff22;color:#b0bec5;border-radius:20px;padding:.3rem .9rem;font-size:.85rem">🎯 Based on Sutton & Barto</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_road, tab_compare, tab_when, tab_all = st.tabs([
        "🗺️ Learning Roadmap",
        "⚖️ Method Comparison",
        "🎯 When to Use Which",
        "📦 All Modules",
    ])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — ROADMAP
    # ════════════════════════════════════════════════════════════════════════
    with tab_road:
        st.markdown("""
        <h2 style="color:white;font-size:1.5rem;margin-bottom:.3rem">🗺️ Your Path to Professional Deep RL</h2>
        <p style="color:#9e9ebb;margin-bottom:1.5rem">
        Follow this sequence. Each stage builds directly on the previous one.
        Skipping steps is possible but will leave gaps that surface later.
        </p>
        """, unsafe_allow_html=True)

        # ── Stage cards ───────────────────────────────────────────────────
        stages = [
            {
                "step": "STAGE 0",
                "icon": "🧬",
                "title": "Deep Learning Prerequisites",
                "subtitle": "Neural network foundations — essential for everything that follows",
                "color": "#00897b",
                "badge": "START HERE — No prior RL needed",
                "badge_color": "#00897b",
                "duration": "~1–2 weeks",
                "why": "Every deep RL algorithm is a neural network trained with a special loss function. Without understanding backpropagation, activations, normalization, and the PyTorch training loop, all the deep RL math will be opaque. This module gives you the exact toolkit used inside DQN, PPO, and AlphaGo.",
                "covers": ["RNNs & LSTMs (memory for POMDPs)", "Batch Norm & Layer Norm (stable training)", "SGD · Adam · RMSProp (optimisers)", "PyTorch autograd & full training loop", "CNN architecture for visual RL (DQN)"],
                "milestone": "Build a working MLP classifier in PyTorch from scratch",
                "btn_key": "rm_pre",
                "btn_page": "prereq",
            },
            {
                "step": "STAGE 1",
                "icon": "🧮",
                "title": "Dynamic Programming",
                "subtitle": "The mathematical bedrock — how optimal policies are computed when everything is known",
                "color": "#6a1b9a",
                "badge": "Theory foundation",
                "badge_color": "#6a1b9a",
                "duration": "~1 week",
                "why": "DP introduces the two most important concepts in all of RL: the Bellman equation and the policy-value relationship. Even though DP requires a full model (rarely available in practice), understanding it makes every model-free algorithm click. Q-learning IS value iteration without the model.",
                "covers": ["Bellman expectation & optimality equations", "Policy Evaluation → exact V(s)", "Policy Improvement → greedy upgrade", "Policy Iteration vs Value Iteration", "GPI — the framework behind ALL RL"],
                "milestone": "Solve 4×4 GridWorld analytically; explain why Policy Iteration converges in 3 steps",
                "btn_key": "rm_dp",
                "btn_page": "dp",
            },
            {
                "step": "STAGE 2",
                "icon": "🎲",
                "title": "Monte Carlo Methods",
                "subtitle": "Model-free learning from complete episodes — no dynamics needed",
                "color": "#7c4dff",
                "badge": "Model-free begins here",
                "badge_color": "#7c4dff",
                "duration": "~1 week",
                "why": "MC methods show how to learn V(s) and Q(s,a) purely from experience — no environment model needed. This is the bridge from DP theory to practical RL. Off-policy MC introduces importance sampling, which is the mathematical foundation of PPO and experience replay.",
                "covers": ["First-Visit vs Every-Visit MC", "On-policy vs off-policy control", "Importance sampling (IS) — foundation of PPO", "Weighted IS — lower variance off-policy", "Incremental & per-decision IS variants"],
                "milestone": "Implement off-policy MC with IS weights; explain why the baseline does not bias the gradient",
                "btn_key": "rm_mc",
                "btn_page": "mc",
            },
            {
                "step": "STAGE 3",
                "icon": "⚡",
                "title": "Temporal-Difference Learning",
                "subtitle": "Online, step-by-step learning — the direct ancestor of DQN",
                "color": "#e65100",
                "badge": "Core RL engine",
                "badge_color": "#e65100",
                "duration": "~1–2 weeks",
                "why": "TD learning is the heart of modern RL. SARSA and Q-Learning are the tabular versions of the algorithms that power DQN and PPO. Understanding the on-policy vs off-policy distinction here (SARSA learns the safe path, Q-Learning learns the optimal path) is what makes DQN vs PPO intuitive later.",
                "covers": ["TD(0) prediction — bootstrapping explained", "SARSA (on-policy) vs Q-Learning (off-policy)", "Expected SARSA & Double Q-Learning", "n-step TD — bridges TD and MC", "SARSA(λ) & eligibility traces"],
                "milestone": "Run CliffWalking; explain WHY SARSA takes the safe path and Q-Learning takes the cliff edge",
                "btn_key": "rm_td",
                "btn_page": "td",
            },
            {
                "step": "STAGE 4",
                "icon": "🎮",
                "title": "Value-Based Deep RL",
                "subtitle": "Q-Learning + neural networks = DQN → Rainbow: Atari from raw pixels",
                "color": "#1565c0",
                "badge": "Deep RL begins here",
                "badge_color": "#1565c0",
                "duration": "~2–3 weeks",
                "why": "DQN (2015) proved that tabular Q-Learning + a CNN + two engineering tricks (experience replay, target network) = superhuman Atari play. The subsequent Rainbow paper shows that 6 targeted improvements are additive. This is the most important architecture family for discrete-action deep RL.",
                "covers": ["DQN: experience replay + target networks", "Double DQN: eliminate overestimation bias", "Dueling DQN: V(s) + A(s,a) decomposition", "PER: prioritised experience replay", "C51 & IQN: distributional return modelling", "Rainbow: all 6 improvements combined"],
                "milestone": "Implement DQN from scratch on CartPole; ablate target network and see training collapse",
                "btn_key": "rm_vbrl",
                "btn_page": "vbrl",
            },
            {
                "step": "STAGE 5",
                "icon": "🎭",
                "title": "Actor-Critic & Policy Gradient",
                "subtitle": "Directly optimise π(a|s) — the algorithm family behind ChatGPT RLHF",
                "color": "#7c4dff",
                "badge": "Modern deep RL",
                "badge_color": "#7c4dff",
                "duration": "~2–3 weeks",
                "why": "Value-based methods cannot handle continuous action spaces (infinite argmax). Policy gradient methods directly parameterise π(a|s) with a neural network, enabling robotics, locomotion, and language model alignment (RLHF). PPO is the single most deployed RL algorithm in the world today.",
                "covers": ["REINFORCE: log-derivative trick from first principles", "Actor-Critic: 1-step TD advantage", "A2C/A3C: n-step + async parallel workers", "PPO: clipped surrogate + GAE advantage", "TRPO: trust region with KL constraint", "SAC: maximum entropy for continuous control"],
                "milestone": "Implement PPO from scratch; run it on a MuJoCo locomotion task; understand why it outperforms vanilla AC",
                "btn_key": "rm_ac",
                "btn_page": "ac",
            },
        ]

        for i, s in enumerate(stages):
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.markdown(f"""
                <div style="background:#12121f;border:1px solid {s['color']}44;border-radius:14px;
                            padding:1.4rem 1.6rem;margin-bottom:.8rem;
                            border-left:5px solid {s['color']}">
                    <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.7rem">
                        <span style="background:{s['color']}33;color:{s['color']};
                                     border-radius:8px;padding:.25rem .7rem;
                                     font-size:.75rem;font-weight:700;letter-spacing:.5px">
                            {s['step']}
                        </span>
                        <span style="font-size:1.6rem">{s['icon']}</span>
                        <div>
                            <b style="color:white;font-size:1.05rem">{s['title']}</b>
                            <br><span style="color:#9e9ebb;font-size:.85rem">{s['subtitle']}</span>
                        </div>
                        <span style="margin-left:auto;background:{s['badge_color']}22;
                                     color:{s['badge_color']};border:1px solid {s['badge_color']}44;
                                     border-radius:20px;padding:.2rem .8rem;
                                     font-size:.75rem;white-space:nowrap">
                            {s['badge']}
                        </span>
                    </div>
                    <p style="color:#b0b0cc;font-size:.9rem;margin:.5rem 0;line-height:1.6">
                        {s['why']}
                    </p>
                    <div style="display:flex;gap:1.5rem;margin-top:.8rem;flex-wrap:wrap">
                        <div style="flex:1;min-width:200px">
                            <b style="color:{s['color']};font-size:.8rem">COVERS</b><br>
                            {''.join(f"<span style='color:#9e9ebb;font-size:.83rem'>▸ {c}</span><br>" for c in s['covers'])}
                        </div>
                        <div style="flex:1;min-width:200px;border-left:1px solid #2a2a3e;padding-left:1.2rem">
                            <b style="color:#ffa726;font-size:.8rem">⏱ ESTIMATED TIME</b><br>
                            <span style="color:#b0b0cc;font-size:.9rem">{s['duration']}</span><br><br>
                            <b style="color:#4caf50;font-size:.8rem">✅ MILESTONE</b><br>
                            <span style="color:#b0b0cc;font-size:.83rem;line-height:1.5">{s['milestone']}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_r:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"Open {s['icon']}", use_container_width=True, key=s['btn_key']):
                    go(s['btn_page'])

            if i < len(stages) - 1:
                st.markdown(f"""
                <div style="text-align:center;color:#2a2a4e;font-size:1.3rem;
                             margin:-.3rem 0 .2rem;letter-spacing:2px">▼ ▼ ▼</div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — METHOD COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    with tab_compare:
        st.markdown("""
        <h2 style="color:white;font-size:1.5rem;margin-bottom:.3rem">⚖️ RL Method Family Comparison</h2>
        <p style="color:#9e9ebb;margin-bottom:1.2rem">
        Understanding the differences between method families is as important as knowing the algorithms themselves.
        </p>
        """, unsafe_allow_html=True)

        # Core axis 1: Model-based vs Model-free
        st.markdown("""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem">
        <h3 style="color:white;font-size:1.1rem;margin-top:0">🔑 Axis 1: Does the agent know the environment dynamics?</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.8rem">
        <div style="background:#1a0a2e;border-radius:8px;padding:1rem;border-left:3px solid #6a1b9a">
            <b style="color:#ce93d8">🧮 Model-Based (DP)</b><br>
            <span style="color:#9e9ebb;font-size:.88rem">Agent has full knowledge of p(s′,r|s,a)</span><br><br>
            <b style="color:#4caf50;font-size:.82rem">✅ Advantages</b><br>
            <span style="color:#b0b0cc;font-size:.83rem">• Exact optimal policy guaranteed<br>• No sampling noise<br>• No training needed</span><br><br>
            <b style="color:#ef5350;font-size:.82rem">❌ Limitations</b><br>
            <span style="color:#b0b0cc;font-size:.83rem">• Full model rarely available in reality<br>• Scales badly (curse of dimensionality)<br>• Cannot handle unknown environments</span><br><br>
            <b style="color:#ce93d8;font-size:.82rem">📦 Algorithms:</b>
            <span style="color:#9e9ebb;font-size:.83rem">Policy Iteration, Value Iteration, Async DP</span>
        </div>
        <div style="background:#0a1a2e;border-radius:8px;padding:1rem;border-left:3px solid #1565c0">
            <b style="color:#90caf9">🎲🎮🎭 Model-Free (MC, TD, Deep RL)</b><br>
            <span style="color:#9e9ebb;font-size:.88rem">Agent learns purely from experience</span><br><br>
            <b style="color:#4caf50;font-size:.82rem">✅ Advantages</b><br>
            <span style="color:#b0b0cc;font-size:.83rem">• Works in unknown environments<br>• Scales to high-dimensional spaces<br>• Powers all modern AI (Atari, Go, ChatGPT)</span><br><br>
            <b style="color:#ef5350;font-size:.82rem">❌ Limitations</b><br>
            <span style="color:#b0b0cc;font-size:.83rem">• Needs many environment interactions<br>• Noisy gradient estimates<br>• Harder to train stably</span><br><br>
            <b style="color:#90caf9;font-size:.82rem">📦 Algorithms:</b>
            <span style="color:#9e9ebb;font-size:.83rem">MC, SARSA, Q-Learning, DQN, PPO, SAC</span>
        </div>
        </div></div>
        """, unsafe_allow_html=True)

        # Axis 2: When does the update happen?
        st.markdown("""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem">
        <h3 style="color:white;font-size:1.1rem;margin-top:0">🔑 Axis 2: When does the agent update its estimate?</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:.8rem">
        <div style="background:#0d1a0d;border-radius:8px;padding:1rem;border-left:3px solid #7c4dff">
            <b style="color:#b39ddb">🎲 MC — Episode-end</b><br>
            <span style="color:#9e9ebb;font-size:.82rem">Wait for episode to finish, then update using real return G</span><br><br>
            <span style="color:#4caf50;font-size:.82rem">✅ Zero bias — uses actual reward</span><br>
            <span style="color:#ef5350;font-size:.82rem">❌ High variance — G fluctuates<br>❌ Episodic tasks only</span>
        </div>
        <div style="background:#1a0d0a;border-radius:8px;padding:1rem;border-left:3px solid #e65100">
            <b style="color:#ffb74d">⚡ TD — Every step</b><br>
            <span style="color:#9e9ebb;font-size:.82rem">Update after each step using r + γV(s′) as target</span><br><br>
            <span style="color:#4caf50;font-size:.82rem">✅ Low variance — 1-step look<br>✅ Works online & continuously</span><br>
            <span style="color:#ef5350;font-size:.82rem">❌ Biased — bootstraps from estimates</span>
        </div>
        <div style="background:#0a0d1a;border-radius:8px;padding:1rem;border-left:3px solid #1565c0">
            <b style="color:#90caf9">🎮 Deep RL — Batched</b><br>
            <span style="color:#9e9ebb;font-size:.82rem">Collect rollout/replay batch, update in mini-batches</span><br><br>
            <span style="color:#4caf50;font-size:.82rem">✅ GPU-efficient batching<br>✅ Stable with replay buffer</span><br>
            <span style="color:#ef5350;font-size:.82rem">❌ More complex infrastructure<br>❌ Off-policy corrections needed</span>
        </div>
        </div></div>
        """, unsafe_allow_html=True)

        # Axis 3: On-policy vs Off-policy
        st.markdown("""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem">
        <h3 style="color:white;font-size:1.1rem;margin-top:0">🔑 Axis 3: Does the agent learn from its own behaviour or from stored data?</h3>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:.8rem">
        <div style="background:#1a1a0a;border-radius:8px;padding:1rem;border-left:3px solid #f57f17">
            <b style="color:#ffa726">🔵 On-Policy</b><br>
            <span style="color:#9e9ebb;font-size:.88rem">Learn only from the current policy's own experience</span><br><br>
            <span style="color:#b0b0cc;font-size:.83rem">Data collected by π<sub>θ</sub> is used to improve π<sub>θ</sub>. Must discard old data after update.</span><br><br>
            <b style="color:#4caf50;font-size:.82rem">✅ More stable, easier to tune</b><br>
            <b style="color:#ef5350;font-size:.82rem">❌ Sample inefficient — old data wasted</b><br><br>
            <b style="color:#ffa726;font-size:.82rem">📦 Examples:</b>
            <span style="color:#9e9ebb;font-size:.83rem">SARSA, MC Control, A2C, PPO, TRPO</span>
        </div>
        <div style="background:#0a1a1a;border-radius:8px;padding:1rem;border-left:3px solid #00838f">
            <b style="color:#80deea">🟠 Off-Policy</b><br>
            <span style="color:#9e9ebb;font-size:.88rem">Learn from data generated by any policy (even old ones)</span><br><br>
            <span style="color:#b0b0cc;font-size:.83rem">Data stored in a replay buffer. Current policy π<sub>θ</sub> can learn from transitions collected by π<sub>old</sub>.</span><br><br>
            <b style="color:#4caf50;font-size:.82rem">✅ Sample efficient — reuse data<br>✅ Can learn from human demos</b><br>
            <b style="color:#ef5350;font-size:.82rem">❌ Needs IS correction (or ignores it)</b><br><br>
            <b style="color:#80deea;font-size:.82rem">📦 Examples:</b>
            <span style="color:#9e9ebb;font-size:.83rem">Q-Learning, DQN, Double DQN, SAC, TD3</span>
        </div>
        </div></div>
        """, unsafe_allow_html=True)

        # Full property matrix
        st.markdown("### 📊 Full Algorithm Property Matrix")
        st.markdown("""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:12px;padding:1rem 1.5rem">
        <table style="width:100%;border-collapse:collapse;font-size:.83rem;color:#b0b0cc">
        <tr style="border-bottom:2px solid #2a2a3e">
          <th style="text-align:left;padding:.5rem .4rem;color:white">Algorithm</th>
          <th style="padding:.5rem .4rem;color:white">Model?</th>
          <th style="padding:.5rem .4rem;color:white">On/Off</th>
          <th style="padding:.5rem .4rem;color:white">Update</th>
          <th style="padding:.5rem .4rem;color:white">Action space</th>
          <th style="padding:.5rem .4rem;color:white">Bias</th>
          <th style="padding:.5rem .4rem;color:white">Variance</th>
          <th style="padding:.5rem .4rem;color:white">Sample eff.</th>
          <th style="padding:.5rem .4rem;color:white">Stability</th>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;background:#0d0d1a">
          <td style="padding:.4rem;color:#ce93d8;font-weight:700">Policy Iteration (DP)</td>
          <td style="text-align:center;padding:.4rem">✅ Full</td><td style="text-align:center">On</td>
          <td style="text-align:center">Each sweep</td><td style="text-align:center">Any</td>
          <td style="text-align:center;color:#4caf50">Zero</td><td style="text-align:center;color:#4caf50">Zero</td>
          <td style="text-align:center">N/A</td><td style="text-align:center;color:#4caf50">★★★★★</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.4rem;color:#b39ddb;font-weight:700">Monte Carlo</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">Both</td>
          <td style="text-align:center">Episode end</td><td style="text-align:center">Discrete</td>
          <td style="text-align:center;color:#4caf50">Zero</td><td style="text-align:center;color:#ef5350">High</td>
          <td style="text-align:center;color:#ef5350">Low</td><td style="text-align:center;color:#ffa726">★★★☆☆</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;background:#0d0d1a">
          <td style="padding:.4rem;color:#ffb74d;font-weight:700">SARSA</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">On</td>
          <td style="text-align:center">Every step</td><td style="text-align:center">Discrete</td>
          <td style="text-align:center;color:#ffa726">Some</td><td style="text-align:center;color:#4caf50">Low</td>
          <td style="text-align:center;color:#ffa726">Medium</td><td style="text-align:center;color:#ffa726">★★★☆☆</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.4rem;color:#ffb74d;font-weight:700">Q-Learning</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">Off</td>
          <td style="text-align:center">Every step</td><td style="text-align:center">Discrete</td>
          <td style="text-align:center;color:#ffa726">Some</td><td style="text-align:center;color:#4caf50">Low</td>
          <td style="text-align:center;color:#ffa726">Medium</td><td style="text-align:center;color:#ffa726">★★★☆☆</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;background:#0d0d1a">
          <td style="padding:.4rem;color:#90caf9;font-weight:700">DQN</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">Off</td>
          <td style="text-align:center">Mini-batch</td><td style="text-align:center">Discrete</td>
          <td style="text-align:center;color:#ffa726">Some</td><td style="text-align:center;color:#4caf50">Low</td>
          <td style="text-align:center;color:#4caf50">High</td><td style="text-align:center;color:#ffa726">★★★★☆</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.4rem;color:#90caf9;font-weight:700">Rainbow</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">Off</td>
          <td style="text-align:center">Mini-batch</td><td style="text-align:center">Discrete</td>
          <td style="text-align:center;color:#ffa726">Some</td><td style="text-align:center;color:#4caf50">Low</td>
          <td style="text-align:center;color:#4caf50">Very high</td><td style="text-align:center;color:#4caf50">★★★★★</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;background:#0d0d1a">
          <td style="padding:.4rem;color:#ce93d8;font-weight:700">REINFORCE</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">On</td>
          <td style="text-align:center">Episode end</td><td style="text-align:center">Both</td>
          <td style="text-align:center;color:#4caf50">Zero</td><td style="text-align:center;color:#ef5350">Very high</td>
          <td style="text-align:center;color:#ef5350">Low</td><td style="text-align:center;color:#ef5350">★★☆☆☆</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.4rem;color:#ce93d8;font-weight:700">A2C / A3C</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">On</td>
          <td style="text-align:center">n-step rollout</td><td style="text-align:center">Both</td>
          <td style="text-align:center;color:#ffa726">Some</td><td style="text-align:center;color:#ffa726">Medium</td>
          <td style="text-align:center;color:#ffa726">Medium</td><td style="text-align:center;color:#ffa726">★★★☆☆</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e;background:#0d0d1a">
          <td style="padding:.4rem;color:#ce93d8;font-weight:700">PPO</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">On (clipped)</td>
          <td style="text-align:center">Rollout batch</td><td style="text-align:center">Both</td>
          <td style="text-align:center;color:#ffa726">Some</td><td style="text-align:center;color:#4caf50">Low</td>
          <td style="text-align:center;color:#ffa726">Medium</td><td style="text-align:center;color:#4caf50">★★★★★</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.4rem;color:#a5d6a7;font-weight:700">SAC</td>
          <td style="text-align:center;padding:.4rem">❌</td><td style="text-align:center">Off</td>
          <td style="text-align:center">Mini-batch</td><td style="text-align:center">Continuous</td>
          <td style="text-align:center;color:#ffa726">Some</td><td style="text-align:center;color:#4caf50">Low</td>
          <td style="text-align:center;color:#4caf50">Very high</td><td style="text-align:center;color:#4caf50">★★★★★</td>
        </tr>
        </table></div>
        """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — WHEN TO USE WHICH
    # ════════════════════════════════════════════════════════════════════════
    with tab_when:
        st.markdown("""
        <h2 style="color:white;font-size:1.5rem;margin-bottom:.3rem">🎯 Decision Guide — What Algorithm for Your Problem?</h2>
        <p style="color:#9e9ebb;margin-bottom:1.2rem">
        Work through these questions to find the right algorithm family for your RL problem.
        </p>
        """, unsafe_allow_html=True)

        # Decision tree as visual cards
        decision_tree = [
            {
                "q": "Q1: Do you have complete knowledge of the environment dynamics p(s′,r|s,a)?",
                "yes": ("✅ YES → Use Dynamic Programming", "#6a1b9a",
                        "You know the full MDP. Policy Iteration or Value Iteration will give the exact optimal policy with no sampling noise. This is rare in practice — only robotics simulators or board games with known rules qualify."),
                "no": ("❌ NO → Continue to Q2", "#546e7a",
                       "Most real problems. You must learn from experience."),
            },
            {
                "q": "Q2: Can you run complete episodes? (Does the environment terminate?)",
                "yes": ("✅ YES and you want simplicity → Monte Carlo", "#7c4dff",
                        "MC methods use true returns G_t — zero bias, no bootstrapping errors. Good for short episodic tasks where you can afford to wait. Works well for board games, simple navigation."),
                "no": ("❌ NO (or episodes are very long) → TD Learning / Deep RL", "#e65100",
                       "TD methods update every step — essential for continuing tasks (traffic control, server management) or long-horizon tasks where waiting for episode end is too slow."),
            },
            {
                "q": "Q3: Is the action space discrete (finite choices) or continuous (real-valued)?",
                "yes": ("🎮 DISCRETE → Value-Based Methods (DQN family)", "#1565c0",
                        "argmax over Q(s,a) works for discrete actions. Use DQN for pixel observations, Double DQN to fix overestimation, Rainbow to combine all improvements. Perfect for Atari, card games, combinatorial tasks."),
                "no": ("🤖 CONTINUOUS → Policy Gradient Methods (PPO / SAC)", "#7c4dff",
                       "argmax over continuous Q is intractable. Policy gradient directly parameterises π(a|s) as a Gaussian. PPO for on-policy (simpler, more stable); SAC for off-policy (better sample efficiency, max-entropy exploration)."),
            },
            {
                "q": "Q4: Is data collection expensive? (Real robots, simulations with long wall time?)",
                "yes": ("💸 YES → Off-policy + Replay Buffer (DQN / SAC / TD3)", "#00838f",
                        "Replay buffers allow every transition to be reused many times. Off-policy methods (DQN, SAC) can train for thousands of gradient steps on the same collected data. Critical for real-robot RL where each episode costs money."),
                "no": ("💰 NO → On-policy (PPO / A2C / TRPO)", "#f57f17",
                       "If data is cheap (fast simulators), on-policy methods are simpler to tune and more stable. PPO is the default choice. A2C/A3C add parallelism for even faster data collection."),
            },
            {
                "q": "Q5: Do you need a well-calibrated uncertainty estimate over returns?",
                "yes": ("📊 YES → Distributional RL (C51 / IQN)", "#ad1457",
                        "Distributional methods model the full return distribution Z(s,a) rather than just E[G]. This gives richer gradient signal, natural risk-awareness, and often better performance. Use when outcome variance matters (finance, safety-critical systems)."),
                "no": ("📉 NO → Standard Q or Policy Gradient", "#546e7a",
                       "Standard expected-value methods are simpler and often sufficient for most tasks."),
            },
        ]

        for i, node in enumerate(decision_tree):
            st.markdown(f"""
            <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:12px;
                        padding:1.1rem 1.4rem;margin-bottom:.7rem">
                <b style="color:#ffa726;font-size:.82rem">QUESTION {i+1}</b>
                <p style="color:white;font-size:.98rem;font-weight:600;margin:.3rem 0 .8rem">{node['q']}</p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem">
                    <div style="background:{node['yes'][1]}18;border-left:3px solid {node['yes'][1]};
                                border-radius:0 8px 8px 0;padding:.8rem 1rem">
                        <b style="color:{node['yes'][1]};font-size:.85rem">{node['yes'][0]}</b><br>
                        <span style="color:#b0b0cc;font-size:.83rem;line-height:1.5">{node['yes'][2]}</span>
                    </div>
                    <div style="background:{node['no'][1]}18;border-left:3px solid {node['no'][1]};
                                border-radius:0 8px 8px 0;padding:.8rem 1rem">
                        <b style="color:{node['no'][1]};font-size:.85rem">{node['no'][0]}</b><br>
                        <span style="color:#b0b0cc;font-size:.83rem;line-height:1.5">{node['no'][2]}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Quick reference
        st.markdown("### ⚡ Quick Reference by Use Case")
        cases = [
            ("🕹️ Atari / discrete games",       "DQN → Double DQN → Rainbow",         "#1565c0"),
            ("🤖 Robotics / locomotion",           "PPO (simulated) → SAC (real robot)",  "#7c4dff"),
            ("💬 LLM alignment (RLHF)",           "PPO with reward model",               "#e65100"),
            ("♟️ Perfect-info board games",       "MCTS + DP or AlphaZero pattern",      "#6a1b9a"),
            ("🎲 Tabular / small state spaces",   "Q-Learning or SARSA",                 "#f57f17"),
            ("🌡️ Continuous control (MuJoCo)",   "SAC or TD3",                          "#00838f"),
            ("🏭 Offline data (no new envs)",     "CQL / IQL (offline RL)",              "#546e7a"),
            ("👁️ Visual observations (pixels)",  "CNN encoder + DQN or PPO",            "#ad1457"),
            ("📅 Long-horizon / sparse reward",   "PPO + curiosity / intrinsic rewards", "#558b2f"),
            ("🌐 Multi-agent cooperation",         "MAPPO / QMIX / MADDPG",              "#0288d1"),
        ]
        cols = st.columns(2)
        for i, (case, alg, col) in enumerate(cases):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;
                            padding:.7rem 1rem;margin-bottom:.5rem;display:flex;align-items:center;gap:.8rem">
                    <span style="font-size:1.3rem">{case.split()[0]}</span>
                    <div>
                        <b style="color:white;font-size:.88rem">{' '.join(case.split()[1:])}</b><br>
                        <span style="color:{col};font-size:.82rem;font-weight:600">{alg}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — ALL MODULES
    # ════════════════════════════════════════════════════════════════════════
    with tab_all:
        st.markdown("""
        <h2 style="color:white;font-size:1.5rem;margin-bottom:.3rem">📦 All Modules</h2>
        <p style="color:#9e9ebb;margin-bottom:1.2rem">Click any module to jump directly to it.</p>
        """, unsafe_allow_html=True)

        modules = [
            ("prereq", "🧬", "Deep Learning Prerequisites", "#00897b", "#80cbc4",
             "RNNs & LSTMs · Normalisation · Optimisers · Autograd · PyTorch Loop · Transformers",
             "9 topics · 37 formulas · 28-term glossary"),
            ("dp",     "🧮", "Dynamic Programming",         "#6a1b9a", "#ce93d8",
             "Policy Evaluation · Policy Improvement · Policy Iteration · Value Iteration · Async DP · GPI",
             "6 algorithms · 4×4 GridWorld · S&B §4"),
            ("mc",     "🎲", "Monte Carlo Methods",         "#7c4dff", "#b39ddb",
             "First-Visit · Every-Visit · On-policy Control · Off-policy IS · Weighted IS · Incremental MC",
             "9 algorithms · 5×5 GridWorld · S&B §5"),
            ("td",     "⚡", "Temporal-Difference Learning","#e65100", "#ffb74d",
             "TD(0) · SARSA · Q-Learning · Expected SARSA · Double Q · n-step TD · SARSA(λ)",
             "7 algorithms · CliffWalking 4×12 · S&B §6–7, §12"),
            ("vbrl",   "🎮", "Value-Based Deep RL",         "#1565c0", "#90caf9",
             "DQN · Double DQN · Dueling DQN · PER · C51 · Rainbow · IQN",
             "7 algorithms · CartPole · Mnih 2015 → Hessel 2018"),
            ("ac",     "🎭", "Actor-Critic & Policy Gradient","#7c4dff","#ce93d8",
             "REINFORCE · Actor-Critic · A2C · A3C · PPO · TRPO · SAC",
             "7 algorithms · CartPole · Schulman 2017"),
        ]

        for page_key, icon, title, color, text_col, topics, meta in modules:
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"""
                <div style="background:#12121f;border:1px solid {color}44;border-radius:12px;
                            padding:1rem 1.4rem;margin-bottom:.6rem;border-left:4px solid {color}">
                    <div style="display:flex;align-items:center;gap:.8rem">
                        <span style="font-size:1.8rem">{icon}</span>
                        <div>
                            <b style="color:white;font-size:1rem">{title}</b><br>
                            <span style="color:{text_col};font-size:.82rem">{topics}</span><br>
                            <span style="color:#546e7a;font-size:.78rem">{meta}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"Open {icon}", key=f"all_{page_key}", use_container_width=True):
                    go(page_key)

        # S&B reference
        st.markdown("""
        <div style="background:#0d1a0d;border:1px solid #1b5e20;border-radius:10px;
                    padding:.9rem 1.3rem;margin-top:1rem;display:flex;align-items:center;gap:1rem">
            <span style="font-size:1.8rem">📖</span>
            <div>
                <b style="color:#a5d6a7">Sutton & Barto — "Reinforcement Learning: An Introduction" (2nd ed.)</b><br>
                <span style="color:#9e9ebb;font-size:.88rem">
                    DP: Ch.3–4 · MC: Ch.5 · TD: Ch.6–7,12 · Policy Gradient: Ch.13 |
                    All tabular examples match textbook figures exactly.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BACK BUTTON (shown in sub-modules)
# ─────────────────────────────────────────────────────────────────────────────
def show_back_bar(module_name, module_color, module_icon):
    col_back, col_title, col_help = st.columns([1, 5, 2])
    with col_back:
        if st.button("← Home", key="back_home", use_container_width=True):
            go("home")
    with col_title:
        st.markdown(
            f'<div style="padding:.35rem 0; color:{module_color}; font-weight:700; font-size:1.05rem">'
            f'{module_icon} RL Portal › {module_name}</div>',
            unsafe_allow_html=True)
    with col_help:
        st.markdown(
            '<div style="padding:.35rem 0; text-align:right; color:#555577; font-size:.82rem">'
            'Use sidebar to configure &amp; run</div>',
            unsafe_allow_html=True)
    st.markdown('<hr style="margin:.4rem 0 1rem; border-color:#1a1a2e">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SUB-MODULES LAZILY
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ac():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("ac_mod", "_ac_mod.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["ac_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner=False)
def load_vbrl():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("vbrl_mod", "_vbrl_mod.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["vbrl_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner=False)
def load_prereq():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("prereq_mod", "_prereq_mod.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["prereq_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner=False)
def load_dp():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("dp_mod", "_dp_mod.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["dp_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner=False)
def load_mc():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("mc_mod", "_mc_mod.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["mc_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner=False)
def load_td():
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("td_mod", "_td_mod.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["td_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
page = st.session_state.portal_page

if page == "home":
    show_home()

elif page == "ac":
    show_back_bar("Actor-Critic & Policy Gradient", "#ce93d8", "🎭")
    mod = load_ac()
    mod.main_ac()

elif page == "vbrl":
    show_back_bar("Value-Based Deep RL", "#90caf9", "🎮")
    mod = load_vbrl()
    mod.main_vbrl()

elif page == "prereq":
    show_back_bar("Deep Learning Prerequisites", "#80cbc4", "🧬")
    mod = load_prereq()
    mod.main_prereq()

elif page == "dp":
    show_back_bar("Dynamic Programming", "#ce93d8", "🧮")
    mod = load_dp()
    mod.main_dp()

elif page == "mc":
    show_back_bar("Monte Carlo Methods", "#b39ddb", "🎲")
    mod = load_mc()
    mod.main_mc()

elif page == "td":
    show_back_bar("Temporal-Difference Learning", "#ffb74d", "⚡")
    mod = load_td()
    mod.main_td()

else:
    go("home")
