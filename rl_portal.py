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
    # Hero banner
    st.markdown("""
    <div style="
        background: linear-gradient(140deg, #0d0d2b 0%, #1a0a3e 35%, #0a1a3e 65%, #0d1a14 100%);
        border: 1px solid #2a2a4e;
        border-radius: 20px;
        padding: 3rem 3.5rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <div style="font-size:3.5rem; margin-bottom:.5rem">🧠</div>
        <h1 style="color:white; margin:0; font-size:2.8rem; font-weight:800; letter-spacing:-0.5px">
            Reinforcement Learning Portal
        </h1>
        <p style="color:#9e9ebb; margin-top:.8rem; font-size:1.15rem; max-width:700px; margin-left:auto; margin-right:auto">
            An interactive, formula-rich learning environment covering the three pillars of
            classical RL — Dynamic Programming, Monte Carlo, and Temporal-Difference methods.
        </p>
        <div style="display:flex; gap:1.5rem; justify-content:center; margin-top:1.5rem; flex-wrap:wrap">
            <span style="background:#00897b22; border:1px solid #00897b55; color:#80cbc4;
                         border-radius:20px; padding:.35rem 1rem; font-size:.9rem">
                🧬 Deep Learning Prerequisites
            </span>
            <span style="background:#6a1b9a22; border:1px solid #6a1b9a55; color:#ce93d8;
                         border-radius:20px; padding:.35rem 1rem; font-size:.9rem">
                🧮 Dynamic Programming
            </span>
            <span style="background:#7c4dff22; border:1px solid #7c4dff55; color:#b39ddb;
                         border-radius:20px; padding:.35rem 1rem; font-size:.9rem">
                🎲 Monte Carlo Methods
            </span>
            <span style="background:#e6510022; border:1px solid #e6510055; color:#ffb74d;
                         border-radius:20px; padding:.35rem 1rem; font-size:.9rem">
                ⚡ Temporal-Difference Learning
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick facts strip
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Methods covered", "17+3", help="DP=6, MC=9, TD=7 + Prerequisites (RNN, Norm, PyTorch)")
    c2.metric("Interactive charts", "70+", help="Heatmaps, convergence plots, policy arrows, learning curves, LSTM gates, normalization demos")
    c3.metric("Theory expanders", "30+", help="Each with full formulas, symbol decoders, worked examples")
    c4.metric("Environments", "3+", help="4×4 GridWorld (DP), 5×5 GridWorld (MC), CliffWalking (TD), 2D Classification (Prereq)")

    st.markdown("---")

    # ── Method selection ─────────────────────────────────────────────────────
    st.markdown("""
    <h2 style="color:white; font-size:1.6rem; margin-bottom:.3rem">
        📚 Choose your learning module
    </h2>
    <p style="color:#9e9ebb; margin-bottom:1.5rem; font-size:.97rem">
        <b>Recommended path:</b> Start with Prerequisites to build the neural network foundations,
        then explore DP for the mathematical theory, MC for model-free sampling, and TD for online step-by-step learning.
    </p>
    """, unsafe_allow_html=True)

    # ── Prereq Card (full width) ──────────────────────────────────────────
    st.markdown("""
    <div class="method-card method-card-pre">
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap">
            <div style="font-size:2.2rem">🧬</div>
            <div>
                <h3 style="color:white;margin:0;font-size:1.2rem">Deep Learning Prerequisites
                    <span style="background:#00897b33;color:#80cbc4;border-radius:10px;
                                 padding:.15rem .6rem;font-size:.75rem;margin-left:.5rem">
                        START HERE
                    </span>
                </h3>
                <p style="color:#80cbc4;font-size:.82rem;margin:.2rem 0 0;font-weight:600">
                    RNNs & LSTMs · Batch & Layer Normalization · PyTorch Full Training Loop
                </p>
            </div>
        </div>
        <p style="color:#b0b0cc;font-size:.92rem;margin:.8rem 0">
            Before diving into DP/MC/TD, every RL practitioner needs a solid understanding of
            the neural network building blocks used in modern deep RL systems.
            This module covers sequential memory (RNNs, LSTMs), training stabilisation
            (Batch & Layer Normalization), and the complete PyTorch training loop that powers
            DQN, PPO, AlphaGo, and every other deep RL algorithm.
            Each topic includes interactive visualisations, full mathematical derivations,
            and working simulations you can run directly in the browser.
        </p>
        <div style="display:flex; gap:.5rem; flex-wrap:wrap">
            <span style="background:#00897b33;color:#80cbc4;border-radius:10px;padding:.2rem .6rem;font-size:.8rem">🔁 RNN unrolling demo</span>
            <span style="background:#00897b33;color:#80cbc4;border-radius:10px;padding:.2rem .6rem;font-size:.8rem">🧩 LSTM gate simulator</span>
            <span style="background:#00897b33;color:#80cbc4;border-radius:10px;padding:.2rem .6rem;font-size:.8rem">📊 Normalisation visualiser</span>
            <span style="background:#00897b33;color:#80cbc4;border-radius:10px;padding:.2rem .6rem;font-size:.8rem">🎮 Live training simulation</span>
            <span style="background:#00897b33;color:#80cbc4;border-radius:10px;padding:.2rem .6rem;font-size:.8rem">🗺️ Full concept map</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Start Prerequisites Module", use_container_width=True, key="btn_pre",
                 type="primary"):
        go("prereq")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # ── DP Card ──────────────────────────────────────────────────────────────
    with col1:
        st.markdown("""
        <div class="method-card method-card-dp">
            <div style="font-size:2.2rem">🧮</div>
            <h3 style="color:white;margin:.4rem 0 .3rem;font-size:1.25rem">Dynamic Programming</h3>
            <p style="color:#ce93d8;font-size:.82rem;margin:0 0 .8rem;font-weight:600">
                Requires full environment model
            </p>
            <p style="color:#b0b0cc;font-size:.92rem;margin:0 0 1rem">
                The mathematical foundation of RL. Learn how to compute exact optimal
                policies when you <em>know everything</em> about the environment.
            </p>
            <div style="font-size:.83rem;color:#9e9ebb">
                <b style="color:#ce93d8">6 methods covered:</b><br>
                Policy Evaluation · Policy Improvement · Policy Iteration ·
                Value Iteration · Async DP · GPI Framework
            </div>
            <div style="margin-top:.9rem; font-size:.83rem; color:#9e9ebb">
                <b style="color:#ce93d8">Environment:</b> 4×4 GridWorld (S&B §4.1)<br>
                <b style="color:#ce93d8">Key insight:</b> Bellman equations + model → exact optimality
            </div>
            <div style="margin-top:.9rem; display:flex; gap:.5rem; flex-wrap:wrap">
                <span style="background:#6a1b9a33;color:#ce93d8;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">Synchronous sweeps</span>
                <span style="background:#6a1b9a33;color:#ce93d8;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">Bellman equations</span>
                <span style="background:#6a1b9a33;color:#ce93d8;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">GPI framework</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start DP Explorer", use_container_width=True, key="btn_dp"):
            go("dp")

    # ── MC Card ──────────────────────────────────────────────────────────────
    with col2:
        st.markdown("""
        <div class="method-card method-card-mc">
            <div style="font-size:2.2rem">🎲</div>
            <h3 style="color:white;margin:.4rem 0 .3rem;font-size:1.25rem">Monte Carlo Methods</h3>
            <p style="color:#b39ddb;font-size:.82rem;margin:0 0 .8rem;font-weight:600">
                Model-free · Episode-based learning
            </p>
            <p style="color:#b0b0cc;font-size:.92rem;margin:0 0 1rem">
                Learn from complete episodes without any model of the world.
                Understand how averaging real outcomes leads to optimal policies.
            </p>
            <div style="font-size:.83rem;color:#9e9ebb">
                <b style="color:#b39ddb">9 methods covered:</b><br>
                First-Visit MC · Every-Visit MC · On-policy Control ·
                Off-policy IS · Incremental MC · Per-Decision IS ·
                Discounting-Aware IS · Off-policy Control · Weighted IS
            </div>
            <div style="margin-top:.9rem; font-size:.83rem; color:#9e9ebb">
                <b style="color:#b39ddb">Environment:</b> 5×5 GridWorld (custom)<br>
                <b style="color:#b39ddb">Key insight:</b> True returns → unbiased estimates
            </div>
            <div style="margin-top:.9rem; display:flex; gap:.5rem; flex-wrap:wrap">
                <span style="background:#7c4dff33;color:#b39ddb;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">Importance sampling</span>
                <span style="background:#7c4dff33;color:#b39ddb;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">Return G</span>
                <span style="background:#7c4dff33;color:#b39ddb;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">ε-greedy GPI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start MC Explorer", use_container_width=True, key="btn_mc"):
            go("mc")

    # ── TD Card ──────────────────────────────────────────────────────────────
    with col3:
        st.markdown("""
        <div class="method-card method-card-td">
            <div style="font-size:2.2rem">⚡</div>
            <h3 style="color:white;margin:.4rem 0 .3rem;font-size:1.25rem">Temporal-Difference</h3>
            <p style="color:#ffb74d;font-size:.82rem;margin:0 0 .8rem;font-weight:600">
                Model-free · Online step-by-step learning
            </p>
            <p style="color:#b0b0cc;font-size:.92rem;margin:0 0 1rem">
                The best of both worlds: no model needed, yet learns after every
                single step. The foundation of modern deep RL (DQN, PPO, A3C).
            </p>
            <div style="font-size:.83rem;color:#9e9ebb">
                <b style="color:#ffb74d">7 methods covered:</b><br>
                TD(0) · SARSA · Q-Learning · Expected SARSA ·
                Double Q-Learning · n-step TD · SARSA(λ) / TD(λ)
            </div>
            <div style="margin-top:.9rem; font-size:.83rem; color:#9e9ebb">
                <b style="color:#ffb74d">Environment:</b> CliffWalking 4×12 (S&B §6.5)<br>
                <b style="color:#ffb74d">Key insight:</b> TD error δ + bootstrapping
            </div>
            <div style="margin-top:.9rem; display:flex; gap:.5rem; flex-wrap:wrap">
                <span style="background:#e6510033;color:#ffb74d;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">TD error δ</span>
                <span style="background:#e6510033;color:#ffb74d;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">Bootstrapping</span>
                <span style="background:#e6510033;color:#ffb74d;border-radius:10px;
                             padding:.2rem .6rem;font-size:.77rem">Eligibility traces</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start TD Explorer", use_container_width=True, key="btn_td"):
            go("td")

    # ── Learning path recommendation ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <h3 style="color:white;margin-bottom:.5rem">🗺️ Recommended Learning Path</h3>
    """, unsafe_allow_html=True)

    path_col1, path_col2 = st.columns([1.3, 0.7])
    with path_col1:
        st.markdown("""
        <div style="background:#12121f; border:1px solid #2a2a3e; border-radius:12px; padding:1.2rem 1.5rem">
        <table style="width:100%; border-collapse:collapse; color:#b0b0cc; font-size:.9rem">
        <tr style="border-bottom:1px solid #2a2a3e">
          <th style="text-align:left; padding:.5rem .3rem; color:#9e9ebb; font-weight:600">Step</th>
          <th style="text-align:left; padding:.5rem .3rem; color:#9e9ebb; font-weight:600">Module</th>
          <th style="text-align:left; padding:.5rem .3rem; color:#9e9ebb; font-weight:600">What you'll learn</th>
          <th style="text-align:left; padding:.5rem .3rem; color:#9e9ebb; font-weight:600">Prereq.</th>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.5rem .3rem">0️⃣</td>
          <td style="padding:.5rem .3rem; color:#80cbc4; font-weight:600">🧬 Prerequisites</td>
          <td style="padding:.5rem .3rem">RNNs, LSTMs, Batch/Layer Norm, PyTorch training loop</td>
          <td style="padding:.5rem .3rem; color:#4caf50">None</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.5rem .3rem">1️⃣</td>
          <td style="padding:.5rem .3rem; color:#ce93d8; font-weight:600">🧮 DP</td>
          <td style="padding:.5rem .3rem">Bellman equations, exact optimality, value functions, GPI</td>
          <td style="padding:.5rem .3rem; color:#4caf50">None</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.5rem .3rem">2️⃣</td>
          <td style="padding:.5rem .3rem; color:#b39ddb; font-weight:600">🎲 MC</td>
          <td style="padding:.5rem .3rem">Model-free learning, returns, importance sampling</td>
          <td style="padding:.5rem .3rem; color:#ff9800">DP basics</td>
        </tr>
        <tr>
          <td style="padding:.5rem .3rem">3️⃣</td>
          <td style="padding:.5rem .3rem; color:#ffb74d; font-weight:600">⚡ TD</td>
          <td style="padding:.5rem .3rem">Online learning, TD error, SARSA vs Q-Learning, traces</td>
          <td style="padding:.5rem .3rem; color:#ff9800">MC basics</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    with path_col2:
        st.markdown("""
        <div style="background:#12121f; border:1px solid #2a2a3e; border-radius:12px; padding:1.2rem 1.5rem">
        <h4 style="color:white;margin:0 0 .8rem;font-size:1rem">📊 Method Comparison</h4>
        <table style="width:100%; border-collapse:collapse; color:#b0b0cc; font-size:.84rem">
        <tr style="border-bottom:1px solid #2a2a3e">
          <th style="text-align:left;padding:.3rem;color:#9e9ebb">Property</th>
          <th style="text-align:center;padding:.3rem;color:#ce93d8">DP</th>
          <th style="text-align:center;padding:.3rem;color:#b39ddb">MC</th>
          <th style="text-align:center;padding:.3rem;color:#ffb74d">TD</th>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.3rem">Needs model</td>
          <td style="text-align:center;padding:.3rem">✅</td>
          <td style="text-align:center;padding:.3rem">❌</td>
          <td style="text-align:center;padding:.3rem">❌</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.3rem">Updates when</td>
          <td style="text-align:center;padding:.3rem">Each sweep</td>
          <td style="text-align:center;padding:.3rem">Episode end</td>
          <td style="text-align:center;padding:.3rem">Each step</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.3rem">Bias</td>
          <td style="text-align:center;padding:.3rem">Zero</td>
          <td style="text-align:center;padding:.3rem">Zero</td>
          <td style="text-align:center;padding:.3rem">Some</td>
        </tr>
        <tr style="border-bottom:1px solid #1a1a2e">
          <td style="padding:.3rem">Variance</td>
          <td style="text-align:center;padding:.3rem">Zero</td>
          <td style="text-align:center;padding:.3rem">High</td>
          <td style="text-align:center;padding:.3rem">Low</td>
        </tr>
        <tr>
          <td style="padding:.3rem">Continuing tasks</td>
          <td style="text-align:center;padding:.3rem">✅</td>
          <td style="text-align:center;padding:.3rem">❌</td>
          <td style="text-align:center;padding:.3rem">✅</td>
        </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Sutton & Barto reference ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="background:#0d1a0d; border:1px solid #1b5e20; border-radius:10px;
                padding:.9rem 1.3rem; display:flex; align-items:center; gap:1rem">
        <span style="font-size:1.8rem">📖</span>
        <div>
            <b style="color:#a5d6a7">Based on Sutton & Barto — "Reinforcement Learning: An Introduction" (2nd ed.)</b><br>
            <span style="color:#9e9ebb; font-size:.9rem">
                DP → Chapters 3–4 · Monte Carlo → Chapter 5 · TD → Chapters 6–7, 12 |
                All examples match textbook figures exactly.
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
