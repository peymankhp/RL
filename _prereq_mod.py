"""
_prereq_mod.py — Deep Learning Prerequisites for RL
Covers: RNNs & LSTMs · Batch & Layer Normalization · PyTorch Full Training Loop
All content is interactive, formula-rich and educational.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DARK_BG    = "#0d0d1a"
CARD_BG    = "#12121f"
GRID_COLOR = "#2a2a3e"
ACCENT     = "#00bcd4"

def _fig(nrows=1, ncols=1, w=12, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="#9e9ebb", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COLOR)
    return fig, axes

def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
            f'<b>{icon} {title}</b><br>{body}</div>')

def _tip(text):
    return (f'<div style="background:#1a2a1a;border-left:3px solid #4caf50;'
            f'padding:.6rem 1rem;border-radius:0 6px 6px 0;margin:.5rem 0;font-size:.92rem">{text}</div>')

def _formula_box(title, body):
    return (f'<div style="background:#1a1a30;border:1px solid #3a3a5e;border-radius:8px;'
            f'padding:.8rem 1.2rem;margin:.5rem 0">'
            f'<span style="color:#9c9cf0;font-size:.82rem;font-weight:700">{title}</span>'
            f'<br>{body}</div>')

def _section_header(emoji, title, subtitle, color="#00bcd4"):
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{color}22,transparent);
                border-left:4px solid {color};border-radius:0 10px 10px 0;
                padding:.9rem 1.4rem;margin-bottom:1rem">
        <h3 style="color:white;margin:0;font-size:1.3rem">{emoji} {title}</h3>
        <p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.92rem">{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def smooth(arr, w=5):
    if len(arr) <= w: return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w)/w, mode="valid")

# ─────────────────────────────────────────────────────────────────────────────
# SIGMOID & TANH UTILS
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def tanh(x):    return np.tanh(x)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main_prereq():
    st.markdown(r"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a0a2e,#0a2a1a);
                border:1px solid #2a3a4a;border-radius:16px;
                padding:2rem 2.5rem;margin-bottom:1.5rem">
        <h2 style="color:white;margin:0;font-size:2rem">🧬 Deep Learning Prerequisites</h2>
        <p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">
            The mathematical and architectural foundations every RL practitioner must know —
            RNNs & LSTMs, Normalization, and the PyTorch training loop that powers all modern RL.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_rnn, tab_norm, tab_loop, tab_map, tab_gloss = st.tabs([
        "🔁 RNNs & LSTMs",
        "📐 Batch & Layer Norm",
        "⚙️ PyTorch Training Loop",
        "🗺️ Concept Map",
        "📖 Glossary",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — RNNs & LSTMs
    # ═══════════════════════════════════════════════════════════════════════
    with tab_rnn:
        _section_header("🔁", "Recurrent Neural Networks & LSTMs",
                         "Why sequence matters — and how neural networks develop memory", "#7c4dff")

        st.markdown(_card("#7c4dff","🧠","Why sequential data is different",
            """Standard neural networks treat every input as independent — each sample has no
            memory of what came before. But language, speech, financial time series, robot sensor
            streams, and game trajectories are fundamentally <em>ordered</em>: the meaning of "bank"
            depends entirely on whether the previous words were "river" or "money". A network that
            ignores this order is blind to the most important structure in the data.<br><br>
            <b>RNNs</b> (Recurrent Neural Networks) solve this by passing a <em>hidden state</em>
            through time — a compact summary of everything seen so far. At each step, the network
            reads the new input <em>and</em> its own memory, producing both an output and an
            updated memory for the next step. This makes RNNs the first neural architecture
            capable of modeling sequential dependencies."""), unsafe_allow_html=True)

        # ── Interactive: RNN Unrolling ────────────────────────────────────
        st.subheader("🎬 Interactive: RNN Unrolled Through Time")
        st.markdown(r"""
        An RNN is a network with a **loop** — the hidden state $h_t$ feeds back into itself.
        "Unrolling" means drawing one copy of the network per timestep, which reveals the
        causal chain of dependencies:

        $$\boxed{h_t = \tanh\!\bigl(W_h\,h_{t-1} + W_x\,x_t + b\bigr)}$$

        **Symbol decoder:**
        - $x_t$ — input at timestep $t$ (e.g. a word embedding, sensor reading)
        - $h_{t-1}$ — hidden state from the previous step (the "memory")
        - $W_h, W_x$ — learned weight matrices (shared across all timesteps!)
        - $\tanh$ — squashes values to $(-1, 1)$, preventing explosion
        - $h_t$ — the updated hidden state, passed to both the output and the next timestep
        """)

        seq_len = st.slider("Sequence length (timesteps to unroll)", 2, 8, 4, key="rnn_len")

        fig, ax = _fig(1, 1, 13, 4)
        ax.set_xlim(-0.5, seq_len + 0.5)
        ax.set_ylim(-0.5, 3.5)
        ax.axis("off")
        ax.set_facecolor(DARK_BG)

        colors_rnn = {"x": "#0288d1", "h": "#7c4dff", "y": "#4caf50", "arrow": "#90a4ae"}
        box_style = dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#7c4dff", lw=2)

        for t in range(seq_len):
            cx = t + 0.5
            # RNN cell
            ax.add_patch(FancyBboxPatch((cx-0.38, 1.1), 0.76, 0.8,
                                        boxstyle="round,pad=0.1",
                                        facecolor="#1e1e3e", edgecolor="#7c4dff", lw=2, zorder=3))
            ax.text(cx, 1.5, f"RNN\nt={t}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold", zorder=4)
            # x_t arrow (from bottom)
            ax.annotate("", xy=(cx, 1.1), xytext=(cx, 0.3),
                        arrowprops=dict(arrowstyle="->", color=colors_rnn["x"], lw=2), zorder=5)
            ax.text(cx, 0.1, f"$x_{t}$", ha="center", va="center",
                    color=colors_rnn["x"], fontsize=11, fontweight="bold")
            # y_t arrow (upward)
            ax.annotate("", xy=(cx, 2.8), xytext=(cx, 1.9),
                        arrowprops=dict(arrowstyle="->", color=colors_rnn["y"], lw=2), zorder=5)
            ax.text(cx, 3.1, f"$y_{t}$", ha="center", va="center",
                    color=colors_rnn["y"], fontsize=11, fontweight="bold")
            # h_t arrow (rightward)
            if t < seq_len - 1:
                ax.annotate("", xy=(cx + 0.62, 1.5), xytext=(cx + 0.38, 1.5),
                            arrowprops=dict(arrowstyle="->", color=colors_rnn["h"], lw=2.5), zorder=5)
                ax.text(cx + 0.5, 1.7, f"$h_{t}$", ha="center", va="bottom",
                        color=colors_rnn["h"], fontsize=9)

        # Legend
        for label, col, yi in [("Input $x_t$","#0288d1",3.3),
                                ("Hidden $h_t$","#7c4dff",3.1),
                                ("Output $y_t$","#4caf50",2.9)]:
            ax.plot([seq_len+0.1], [yi-3+3.3], "o", color=col, ms=8, transform=ax.transData)
        ax.text(seq_len+0.05, 3.15, "Weights $W_h, W_x$\nshared across all timesteps",
                color="#9e9ebb", fontsize=8, va="center")
        ax.set_title(f"RNN Unrolled — {seq_len} timesteps (one weight matrix reused each step)",
                     color="white", fontsize=10, fontweight="bold", pad=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(_card("#ef5350","⚠️","The Vanishing Gradient Problem",
            """The beauty of weight-sharing becomes a curse for long sequences.
            During backpropagation, gradients must flow <em>backward through every timestep</em>.
            At each step, they are multiplied by the Jacobian of the tanh activation — which has
            maximum absolute eigenvalue much less than 1. After 50 timesteps, the gradient has
            been multiplied by this small number 50 times: it effectively vanishes to zero.<br><br>
            Concretely: if the gradient shrinks by 0.9 each step, after 50 steps it is
            $0.9^{50} \\approx 0.005$ — 200× smaller than the original. The network can no longer
            "remember" what happened 50 steps ago. It only "sees" the last few steps, even though
            the entire history is theoretically available. This is why vanilla RNNs fail at
            long-range language modelling, music generation, and long-horizon RL tasks."""),
            unsafe_allow_html=True)

        # ── Interactive: gradient decay visualization ─────────────────────
        st.subheader("📉 Interactive: How Gradients Vanish Over Timesteps")
        col1, col2 = st.columns([1, 2])
        with col1:
            decay_rate = st.slider("Gradient decay per step |∂h/∂h_prev|", 0.50, 1.10, 0.85, 0.01,
                                   key="grad_decay")
            max_steps  = st.slider("Max timesteps to show", 10, 100, 40, 5, key="grad_steps")

        steps = np.arange(1, max_steps + 1)
        grad  = decay_rate ** steps

        with col2:
            fig2, ax2 = _fig(1, 1, 8, 3.5)
            ax2.semilogy(steps, grad, color="#ef5350", lw=2.5, label=f"|decay|={decay_rate:.2f} per step")
            ax2.fill_between(steps, 0, grad, alpha=0.15, color="#ef5350")
            ax2.axhline(0.01, color="#ffa726", ls="--", lw=1.2, label="0.01 threshold (effectively gone)")
            gone_at = next((i for i,g in enumerate(grad) if g < 0.01), None)
            if gone_at:
                ax2.axvline(gone_at + 1, color="#4caf50", ls=":", lw=1.5,
                            label=f"Vanished at step {gone_at+1}")
            ax2.set_xlabel("Timestep (looking backward)", color="white", fontsize=10)
            ax2.set_ylabel("Gradient magnitude (log)", color="white", fontsize=10)
            ax2.set_title("Gradient magnitude reaching the earliest timestep", color="white", fontweight="bold")
            ax2.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            ax2.grid(alpha=0.15)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── LSTM ─────────────────────────────────────────────────────────
        st.divider()
        _section_header("🧩", "LSTM — Long Short-Term Memory",
                         "A gating mechanism that controls what is remembered and what is forgotten", "#00897b")

        st.markdown(r"""
        LSTM (Hochreiter & Schmidhuber, 1997) introduces two state vectors and three learnable gates:

        | Component | Symbol | Role |
        |-----------|--------|------|
        | **Cell state** | $C_t$ | Long-term memory — carries information across many steps |
        | **Hidden state** | $h_t$ | Short-term output — used for predictions |
        | **Forget gate** | $f_t$ | How much of the old cell state to erase |
        | **Input gate** | $i_t$ | How much new information to write into the cell |
        | **Output gate** | $o_t$ | How much of the cell to expose as the hidden state |

        The entire LSTM update in four equations:
        """)

        with st.expander("📐 Full LSTM Equations — every symbol decoded", expanded=True):
            st.markdown(r"""
            $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(Forget gate)}$$
            $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(Input gate)}$$
            $$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C) \quad \text{(Candidate cell)}$$
            $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell state update)}$$
            $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(Output gate)}$$
            $$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden state output)}$$

            **Symbol decoder:**
            - $\sigma$ — sigmoid function, outputs in $(0,1)$ — perfect for gates (0=closed, 1=open)
            - $\odot$ — elementwise (Hadamard) multiplication — gates multiply element by element
            - $[h_{t-1}, x_t]$ — concatenation of previous hidden state and current input
            - $\tilde{C}_t$ — the *candidate* new information proposed for the cell state
            - $C_t$ — the actual cell state: a fraction $f_t$ of the old memory + a fraction $i_t$ of the new candidate
            - $W_f, W_i, W_C, W_o$ — four separate learnable weight matrices (three times more parameters than RNN!)

            **Intuition for the cell state update** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$:

            Think of $C$ as a conveyor belt carrying information. The forget gate $f_t$ decides
            what to **drop off** the belt. The input gate $i_t$ decides what new cargo to **add**.
            Since $f_t \in (0,1)$ can be close to 1 for many timesteps, information can travel
            unmodified for long distances — solving the vanishing gradient problem.
            """)

        # ── Interactive: gate activations ─────────────────────────────────
        st.subheader("🎛️ Interactive: Visualise LSTM Gate Activations")
        st.markdown("""
        Adjust the pre-activation values for each gate to see how the sigmoid and tanh
        outputs control the flow of information through the LSTM cell.
        """)

        g1, g2, g3, g4 = st.columns(4)
        with g1:
            zf = st.slider("Forget gate $z_f$", -4.0, 4.0, 2.0, 0.1, key="zf",
                           help="Pre-activation of forget gate. High → keep old memory")
        with g2:
            zi = st.slider("Input gate $z_i$", -4.0, 4.0, 1.5, 0.1, key="zi",
                           help="Pre-activation of input gate. High → write new info")
        with g3:
            zc = st.slider("Candidate $z_c$", -4.0, 4.0, 0.8, 0.1, key="zc",
                           help="Pre-activation of candidate. Tanh squash → proposed cell content")
        with g4:
            zo = st.slider("Output gate $z_o$", -4.0, 4.0, 1.0, 0.1, key="zo",
                           help="Pre-activation of output gate. High → expose more of cell")

        ft = sigmoid(zf); it = sigmoid(zi)
        ct_tilde = tanh(zc); ot = sigmoid(zo)
        c_prev = 0.5   # assume previous cell state = 0.5 for demo
        ct = ft * c_prev + it * ct_tilde
        ht = ot * tanh(ct)

        fig3, axes3 = _fig(1, 2, 13, 4)
        # Bar chart of gate values
        ax_g = axes3[0]
        names  = ["Forget $f_t$\n(keep old?)", "Input $i_t$\n(add new?)",
                  "Candidate $\\tilde{C}_t$\n(new content)", "Output $o_t$\n(expose?)"]
        values = [ft, it, ct_tilde, ot]
        colors_g = ["#ef5350", "#42a5f5", "#ffa726", "#66bb6a"]
        bars = ax_g.bar(names, values, color=colors_g, edgecolor="white", lw=0.5, alpha=0.9)
        for bar, val in zip(bars, values):
            ax_g.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                      f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")
        ax_g.axhline(0, color="white", lw=0.5, alpha=0.3)
        ax_g.set_ylim(-1.2, 1.4)
        ax_g.set_title("Gate Activations", color="white", fontweight="bold")
        ax_g.set_ylabel("Activation value", color="white")

        # Cell state flow diagram
        ax_f = axes3[1]
        ax_f.axis("off")
        ax_f.set_xlim(0, 10); ax_f.set_ylim(0, 4)

        def draw_box(ax, x, y, w, h, txt, col, fs=9):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=col+"33", edgecolor=col, lw=2, zorder=3))
            ax.text(x+w/2, y+h/2, txt, ha="center", va="center",
                    color="white", fontsize=fs, fontweight="bold", zorder=4)

        draw_box(ax_f, 0.2, 2.5, 2.0, 0.8, f"C_prev\n= {c_prev:.2f}", "#546e7a", 8)
        draw_box(ax_f, 3.0, 2.5, 2.0, 0.8, f"× f_t\n= {ft:.3f}", "#ef5350", 8)
        draw_box(ax_f, 5.8, 1.8, 2.2, 0.8, f"+ i_t×C̃\n={it*ct_tilde:.3f}", "#ffa726", 8)
        draw_box(ax_f, 3.0, 1.0, 2.0, 0.8, f"i_t×C̃_t\n={it:.3f}×{ct_tilde:.3f}", "#42a5f5", 8)
        draw_box(ax_f, 8.0, 2.3, 1.8, 1.0, f"C_t\n={ct:.3f}", "#4caf50", 10)
        draw_box(ax_f, 8.0, 0.4, 1.8, 0.8, f"h_t\n={ht:.3f}", "#7c4dff", 10)

        for (x1,y1),(x2,y2) in [((2.2,2.9),(3.0,2.9)),((5.0,2.9),(5.8,2.3)),
                                  ((5.0,1.4),(5.8,2.15)),((8.0,2.7),(8.0,1.2))]:
            ax_f.annotate("", xy=(x2,y2), xytext=(x1,y1),
                          arrowprops=dict(arrowstyle="->", color="#90a4ae", lw=2), zorder=5)
        ax_f.text(8.9, 0.3, f"o_t={ot:.3f}", ha="center", color="#7c4dff", fontsize=7)
        ax_f.text(5.0, 3.5, "LSTM Cell State Flow", ha="center", color="white",
                  fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Cell state $C_t$", f"{ct:.4f}", delta=f"Δ = {ct-c_prev:+.4f} vs prev")
        c2.metric("Hidden output $h_t$", f"{ht:.4f}")
        c3.metric("Info retained from memory", f"{ft*c_prev:.4f} ({ft*100:.0f}%)")

        # ── LSTM in RL ────────────────────────────────────────────────────
        st.divider()
        st.subheader("🤖 LSTM in Reinforcement Learning — DRQN & Partial Observability")
        st.markdown(r"""
        In standard RL, the agent receives the full environment state $s_t$. But in the real world —
        robotics, medical monitoring, multiplayer games — the agent only sees a *partial observation*
        $o_t \subset s_t$. This is called a **Partially Observable MDP (POMDP)**.

        **The LSTM solution:** Replace the feedforward Q-network with an LSTM that processes
        the history of observations. The hidden state $h_t$ becomes the agent's *belief state* —
        a learned compression of all past observations into what matters for the current decision.

        $$\pi(a | o_1, o_2, \ldots, o_t) \approx \pi(a | h_t) \quad \text{where } h_t = \text{LSTM}(o_t, h_{t-1})$$
        """)

        st.markdown(_card("#00897b","🎯","DRQN — Deep Recurrent Q-Network",
            """DRQN (Hausknecht & Stone, 2015) replaces the first fully-connected layer of DQN with an LSTM.
            The agent processes a single frame at each step (not 4 stacked frames like DQN) and the LSTM
            builds up temporal context over the episode. This allows DRQN to play Atari games where
            flickering or frame-masking makes individual frames uninformative — the LSTM integrates
            evidence across many frames to infer the true state. For example, in Pong, if every other
            frame is hidden, DQN fails but DRQN succeeds because it remembers the ball's trajectory.
            The key training trick: sample random <em>episode segments</em> (not random frames) to
            preserve temporal ordering — the LSTM must see frames in sequence to be useful."""),
            unsafe_allow_html=True)

        with st.expander("📐 RNN vs GRU vs LSTM — When to use which"):
            df_compare = pd.DataFrame({
                "Property": ["Memory type", "Parameters", "Gradient flow", "Training speed",
                             "Best for", "Introduced"],
                "RNN": ["Hidden state only", "Least", "Vanishes quickly", "Fastest",
                        "Very short sequences (< 20 steps)", "1986"],
                "GRU": ["Hidden state (reset + update gate)", "Medium", "Good",
                        "Fast", "Medium sequences, speech, text", "2014"],
                "LSTM": ["Cell + hidden state (3 gates)", "Most", "Excellent",
                         "Slower", "Long sequences, NLP, RL memory", "1997"],
            })
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — NORMALIZATION
    # ═══════════════════════════════════════════════════════════════════════
    with tab_norm:
        _section_header("📐", "Batch Normalization & Layer Normalization",
                         "Stabilising deep network training by controlling activation distributions", "#e65100")

        st.markdown(_card("#e65100","📊","Why normalisation is essential for deep learning",
            """As a neural network trains, the parameters of each layer change. This means the
            <em>distribution of inputs</em> to every subsequent layer also changes with each gradient
            step — a phenomenon called <b>Internal Covariate Shift</b>. Layer 7 must constantly
            "re-adapt" to a shifting distribution from layer 6, which makes training unstable and
            forces us to use very small learning rates to avoid divergence.<br><br>
            Normalisation layers solve this by enforcing a controlled, zero-mean unit-variance
            distribution at each layer's input. Once the distribution is stable, deeper layers
            can learn faster and more reliably. As a bonus, normalisation provides implicit
            regularisation — the per-batch noise in BatchNorm acts like a stochastic smoother
            that reduces overfitting. Without normalisation layers, ResNet-152 and GPT-4 would
            be effectively untrainable."""), unsafe_allow_html=True)

        # ── Interactive: ICS demo ─────────────────────────────────────────
        st.subheader("📊 Interactive: Internal Covariate Shift Visualised")
        st.markdown("""
        The plot below shows how layer activations shift in distribution across training steps.
        Each curve represents the distribution of pre-activation values at one layer after
        different numbers of gradient updates — without normalisation, these distributions drift wildly.
        """)

        n_curves = st.slider("Training steps shown", 3, 8, 5, key="ics_curves")
        fig_ics, ax_ics = _fig(1, 1, 11, 4)
        x_range = np.linspace(-6, 8, 300)
        cmap_shifts = plt.cm.plasma(np.linspace(0.2, 0.9, n_curves))

        for k in range(n_curves):
            mu    = -2 + k * 1.5 + np.random.randn() * 0.3
            sigma = 0.8 + k * 0.25 + np.random.randn() * 0.1
            pdf   = np.exp(-0.5*((x_range-mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
            ax_ics.plot(x_range, pdf, color=cmap_shifts[k], lw=2.2, alpha=0.85,
                        label=f"Step {(k+1)*100}: μ={mu:.1f}, σ={sigma:.2f}")
            ax_ics.fill_between(x_range, 0, pdf, alpha=0.06, color=cmap_shifts[k])

        ax_ics.set_xlabel("Activation value", color="white", fontsize=10)
        ax_ics.set_ylabel("Density", color="white", fontsize=10)
        ax_ics.set_title("Internal Covariate Shift — activation distribution drifts during training",
                         color="white", fontweight="bold")
        ax_ics.legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
        ax_ics.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ics); plt.close()
        st.caption("Notice: with no normalisation, layer 6 must adapt to a completely different distribution than it saw during early training. This forces smaller learning rates and causes slow, unstable convergence.")

        st.divider()

        # ── BatchNorm ─────────────────────────────────────────────────────
        st.subheader("🔵 Batch Normalization (Ioffe & Szegedy, 2015)")
        st.markdown(r"""
        Batch Normalization normalises **across the batch dimension** — for each feature (neuron),
        it computes the mean and variance across all examples in the current mini-batch, then
        normalises each example's value relative to those statistics.
        """)
        st.latex(r"""
        \hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \varepsilon}}
        \qquad y_i = \gamma\, \hat{x}_i + \beta
        """)
        st.markdown(r"""
        **Symbol decoder:**
        - $\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i$ — mean of the current mini-batch (size $m$)
        - $\sigma_\mathcal{B}^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$ — variance
        - $\varepsilon$ — small constant for numerical stability (e.g. $10^{-5}$)
        - $\gamma, \beta$ — **learned** scale and shift parameters (allows the network to undo normalization if needed)
        - At inference: $\mu_\mathcal{B}$ is replaced by a running average computed during training
        """)

        # ── Interactive: BatchNorm effect ─────────────────────────────────
        st.subheader("🎛️ Interactive: BatchNorm Before vs After")
        col_bn1, col_bn2 = st.columns([1, 2])
        with col_bn1:
            bn_mu    = st.slider("Pre-BN mean μ", -5.0, 5.0, 3.0, 0.5, key="bn_mu")
            bn_sigma = st.slider("Pre-BN std σ", 0.1, 5.0, 2.5, 0.1, key="bn_sig")
            bn_gamma = st.slider("Learned γ (scale)", 0.1, 3.0, 1.0, 0.1, key="bn_gam")
            bn_beta  = st.slider("Learned β (shift)", -3.0, 3.0, 0.0, 0.1, key="bn_bet")

        np.random.seed(42)
        x_raw  = np.random.randn(512) * bn_sigma + bn_mu
        x_norm = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-5)
        x_out  = bn_gamma * x_norm + bn_beta

        with col_bn2:
            fig_bn, axes_bn = _fig(1, 3, 13, 4)
            titles = ["Raw activations\n(before BatchNorm)", "After normalisation\n(γ=1, β=0)",
                      f"After affine transform\n(γ={bn_gamma:.1f}, β={bn_beta:.1f})"]
            datas  = [x_raw, x_norm, x_out]
            clrs   = ["#ef5350", "#42a5f5", "#4caf50"]
            for ax, data, title, clr in zip(np.array(axes_bn).flatten(), datas, titles, clrs):
                ax.hist(data, bins=50, color=clr, alpha=0.8, edgecolor="none", density=True)
                ax.axvline(data.mean(), color="white", ls="--", lw=1.5, alpha=0.8,
                           label=f"μ={data.mean():.2f}")
                ax.set_title(title, color="white", fontsize=8, fontweight="bold")
                ax.legend(facecolor=CARD_BG, labelcolor="white", fontsize=7)
                ax.set_ylabel("Density", color="white", fontsize=8)
                ax.set_xlabel(f"σ={data.std():.2f}", color="#9e9ebb", fontsize=8)
            plt.tight_layout(); st.pyplot(fig_bn); plt.close()

        st.markdown(_card("#ef5350","⚠️","BatchNorm Limitations",
            """BatchNorm has three important weaknesses that restrict its use:<br>
            <b>(1) Batch-size dependence:</b> With small batches (e.g. batch_size=4 in object detection),
            the sample mean/variance is a noisy estimate of the true statistics — BatchNorm becomes unstable.<br>
            <b>(2) Incompatibility with RNNs:</b> In a sequence of length T, each timestep has a
            different "batch" of activations — you'd need separate γ,β per timestep, which is memory-intensive
            and ignores the sequential structure.<br>
            <b>(3) Inference complexity:</b> BatchNorm must maintain running averages of μ and σ²
            during training to use at inference time when batch statistics are unavailable.
            These limitations motivated LayerNorm."""), unsafe_allow_html=True)

        st.divider()

        # ── LayerNorm ─────────────────────────────────────────────────────
        st.subheader("🟢 Layer Normalization (Ba et al., 2016)")
        st.markdown(r"""
        Layer Normalization normalises **within each sample** — across all features of a single
        example, rather than across the batch. This makes it completely **batch-size independent**
        and equally valid at inference time.
        """)
        st.latex(r"\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \varepsilon}} \qquad y = \gamma\,\hat{x} + \beta")
        st.markdown(r"""
        where $\mu_L$ and $\sigma_L^2$ are computed **across all features of one sample**:
        """)
        st.latex(r"\mu_L = \frac{1}{d}\sum_{j=1}^d x_j \qquad \sigma_L^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu_L)^2")
        st.markdown(r"""
        LayerNorm is the **standard normalisation in every modern Transformer** (BERT, GPT, LLaMA)
        because it works identically during training (batch size > 1) and inference (batch size = 1),
        and it naturally handles variable-length sequences.
        """)

        # ── BatchNorm vs LayerNorm diagram ─────────────────────────────────
        st.subheader("🔄 BatchNorm vs LayerNorm — What Dimension is Normalised?")
        st.markdown(r"""
        The critical conceptual difference is **which axis** of the data tensor you normalise over.
        Think of a data batch as a matrix: rows = samples, columns = features.
        """)

        fig_dim, axes_dim = _fig(1, 2, 13, 5)
        for ax, (title, color, label_row, label_col, shade_rows, shade_cols) in zip(
            np.array(axes_dim).flatten(),
            [("BatchNorm — normalise over BATCH (rows)", "#e65100",
              ["Sample 1","Sample 2","Sample 3","Sample 4"],
              ["Feat 1","Feat 2","Feat 3","Feat 4","Feat 5"],
              list(range(4)), [0, 1, 2]),    # shade a column (per-feature stats)
             ("LayerNorm — normalise over FEATURES (cols)", "#4caf50",
              ["Sample 1","Sample 2","Sample 3","Sample 4"],
              ["Feat 1","Feat 2","Feat 3","Feat 4","Feat 5"],
              [1], list(range(5)))]         # shade a row (per-sample stats)
        ):
            n_rows, n_cols = 4, 5
            # Draw grid
            for r in range(n_rows):
                for c in range(n_cols):
                    fc = color + "55" if (r in shade_rows and c in shade_cols) else "#1e1e2e"
                    ec = color if (r in shade_rows and c in shade_cols) else GRID_COLOR
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=fc,
                                               edgecolor=ec, lw=1.5, zorder=2))
                    ax.text(c+0.5, r+0.5, f"x[{r},{c}]", ha="center", va="center",
                            color="white", fontsize=7, zorder=3)
            ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
            ax.set_xticks([i+0.5 for i in range(n_cols)])
            ax.set_xticklabels(["Feat 1","Feat 2","Feat 3","Feat 4","Feat 5"],
                               color="#9e9ebb", fontsize=8)
            ax.set_yticks([i+0.5 for i in range(n_rows)])
            ax.set_yticklabels(["Sample 1","Sample 2","Sample 3","Sample 4"],
                               color="#9e9ebb", fontsize=8)
            ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=8)
            ax.invert_yaxis()
            highlight = "column 1–3 (blue = statistics computed here)" if "Batch" in title \
                        else "row 2 (green = statistics computed here)"
            ax.set_xlabel(f"Highlighted: {highlight}", color=color, fontsize=8)
        plt.tight_layout(); st.pyplot(fig_dim); plt.close()

        # ── Full comparison table ─────────────────────────────────────────
        st.subheader("📋 Full Comparison: BatchNorm, LayerNorm, InstanceNorm, GroupNorm")
        df_norm = pd.DataFrame({
            "Property": ["Normalise over", "Batch-size dep.", "Works in RNN/Transformer",
                         "Inference diff.", "Best architecture", "Introduced"],
            "BatchNorm": ["Batch (N) per feature", "✅ Yes (needs large B)", "❌ Difficult",
                          "Needs running stats", "CNNs, MLPs", "2015"],
            "LayerNorm": ["All features per sample", "❌ No", "✅ Excellent",
                          "Same as training", "Transformers, RNNs, LLMs", "2016"],
            "InstanceNorm": ["Spatial H×W per sample/channel", "❌ No", "❌ Limited",
                             "Same as training", "Style transfer, GANs", "2017"],
            "GroupNorm": ["G channel groups per sample", "❌ No", "Partial",
                          "Same as training", "Object detection (small batch)", "2018"],
        })
        st.dataframe(df_norm, use_container_width=True, hide_index=True)

        # ── Effect on learning rate ──────────────────────────────────────
        st.subheader("📈 Effect of Normalisation on Training Stability")
        np.random.seed(7)
        epochs = np.arange(1, 101)
        loss_no_norm = 2.5 * np.exp(-epochs/35) + 0.5 + 0.3*np.random.randn(100)*np.exp(-epochs/20)
        loss_bn      = 2.5 * np.exp(-epochs/15) + 0.08 + 0.05*np.random.randn(100)*np.exp(-epochs/30)
        loss_ln      = 2.5 * np.exp(-epochs/18) + 0.10 + 0.06*np.random.randn(100)*np.exp(-epochs/30)

        fig_lc, ax_lc = _fig(1, 1, 11, 4)
        for loss, label, col in [(loss_no_norm,"No Normalization","#ef5350"),
                                  (loss_bn,"Batch Normalization","#42a5f5"),
                                  (loss_ln,"Layer Normalization","#4caf50")]:
            sm = smooth(loss, 5)
            ax_lc.plot(loss[:len(sm)], color=col, alpha=0.15, lw=0.6)
            ax_lc.plot(range(len(sm)), sm, color=col, lw=2.5, label=label)
        ax_lc.set_xlabel("Epoch", color="white", fontsize=10)
        ax_lc.set_ylabel("Training loss", color="white", fontsize=10)
        ax_lc.set_title("Training convergence with and without normalisation", color="white", fontweight="bold")
        ax_lc.legend(facecolor=CARD_BG, labelcolor="white")
        ax_lc.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_lc); plt.close()
        st.caption("With normalization: faster convergence, lower final loss, less sensitivity to learning rate choice.")

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — PYTORCH TRAINING LOOP
    # ═══════════════════════════════════════════════════════════════════════
    with tab_loop:
        _section_header("⚙️", "PyTorch Full Training Loop",
                         "Every component of the neural network training cycle — from data loading to evaluation", "#ffa726")

        st.markdown(_card("#ffa726","🔁","The Training Loop — The Heart of All Deep Learning",
            """The training loop is the central algorithm that transforms raw, untrained parameters
            into a model that generalises. It is deceptively simple — the same 5-step cycle repeated
            thousands or millions of times — yet every major AI system, from image classifiers to
            GPT-4 to AlphaGo, runs on this exact structure. Understanding it deeply means understanding
            <em>how</em> neural networks learn, not just <em>that</em> they do.<br><br>
            The five steps are: <b>(1) Forward pass</b> — compute predictions; <b>(2) Loss computation</b>
            — measure how wrong the predictions are; <b>(3) Backward pass</b> — compute gradients via
            automatic differentiation; <b>(4) Optimizer step</b> — update parameters in the gradient
            direction; <b>(5) Zero gradients</b> — reset for the next iteration.
            Each step is covered interactively below."""), unsafe_allow_html=True)

        # ── Full loop overview diagram ─────────────────────────────────────
        st.subheader("🗺️ The Full Training Loop — Visual Map")

        fig_loop, ax_loop = _fig(1, 1, 14, 5)
        ax_loop.axis("off"); ax_loop.set_xlim(0, 14); ax_loop.set_ylim(0, 5)

        steps_loop = [
            (0.5, 2.5, "1. Data\nLoader", "#0288d1",
             "Batches of (X, y)\nshuffled each epoch"),
            (2.8, 2.5, "2. Forward\nPass", "#7c4dff",
             "ŷ = model(X)\nCompute graph built"),
            (5.1, 2.5, "3. Loss\nFunction", "#e65100",
             "L = criterion(ŷ, y)\nScalar measure of error"),
            (7.4, 2.5, "4. Backward\nPass", "#ef5350",
             "L.backward()\nGradients via autograd"),
            (9.7, 2.5, "5. Optimizer\nStep", "#4caf50",
             "optim.step()\nθ ← θ − η∇L"),
            (12.0, 2.5, "6. Zero\nGrads", "#ffa726",
             "optim.zero_grad()\nClear for next iter"),
        ]

        for i, (x, y, title, col, desc) in enumerate(steps_loop):
            ax_loop.add_patch(FancyBboxPatch((x-0.85, y-0.65), 1.7, 1.3,
                                             boxstyle="round,pad=0.1",
                                             facecolor=col+"22", edgecolor=col, lw=2, zorder=3))
            ax_loop.text(x, y+0.1, title, ha="center", va="center", color="white",
                         fontsize=9, fontweight="bold", zorder=4)
            ax_loop.text(x, y-1.2, desc, ha="center", va="center", color="#9e9ebb",
                         fontsize=7, zorder=4)
            ax_loop.text(x, y+0.8, f"Step {i+1}", ha="center", va="center",
                         color=col, fontsize=8, fontweight="bold")
            if i < len(steps_loop)-1:
                ax_loop.annotate("", xy=(steps_loop[i+1][0]-0.85, y),
                                 xytext=(x+0.85, y),
                                 arrowprops=dict(arrowstyle="->", color="#90a4ae", lw=2))

        # Loop-back arrow
        ax_loop.annotate("", xy=(0.5, 1.85), xytext=(12.85, 1.85),
                         arrowprops=dict(arrowstyle="->", color="#546e7a", lw=1.5,
                                        connectionstyle="arc3,rad=-0.4"))
        ax_loop.text(6.5, 0.4, "↺ Repeat for every batch × every epoch",
                     ha="center", color="#546e7a", fontsize=9, style="italic")
        ax_loop.set_title("PyTorch Training Loop — 6 steps per batch iteration",
                          color="white", fontsize=11, fontweight="bold", pad=8)
        plt.tight_layout(); st.pyplot(fig_loop); plt.close()

        # ── Step-by-step expanders ─────────────────────────────────────────
        st.subheader("🔎 Deep Dive — Each Step Explained")

        with st.expander("Step 1 — Data Loading & DataLoader", expanded=False):
            st.markdown(r"""
            Before any training can happen, raw data must be converted to PyTorch tensors and
            organised into mini-batches. The **DataLoader** handles this automatically:

            ```python
            from torch.utils.data import DataLoader, TensorDataset
            import torch

            X = torch.randn(1000, 20)     # 1000 samples, 20 features each
            y = torch.randint(0, 3, (1000,))  # 3-class labels

            dataset = TensorDataset(X, y)
            loader  = DataLoader(dataset, batch_size=32, shuffle=True)

            # Each iteration: X_batch.shape = (32, 20), y_batch.shape = (32,)
            for X_batch, y_batch in loader:
                ...   # ← training steps go here
            ```

            **Why mini-batches?**
            - **Memory:** Full-dataset gradient requires all data in GPU RAM simultaneously
            - **Speed:** GPUs are most efficient with batches of 32–512 samples
            - **Regularisation:** Per-batch noise helps escape sharp minima
            - **Shuffling:** Prevents the model from learning the order of samples

            **Rule of thumb:** Use the largest batch size that fits in GPU memory.
            For learning rate: scale $\eta$ linearly with batch size (linear scaling rule).
            """)

        with st.expander("Step 2 — Forward Pass & Computation Graph", expanded=False):
            st.markdown(r"""
            The forward pass applies the model to the input batch. PyTorch's **autograd** engine
            simultaneously builds a **computation graph** — a directed acyclic graph (DAG) tracking
            every mathematical operation — that will be used to compute gradients in the backward pass.

            ```python
            import torch.nn as nn

            class MLP(nn.Module):
                def __init__(self, in_dim, hidden, out_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(in_dim, hidden),   # W₁x + b₁
                        nn.BatchNorm1d(hidden),       # Normalise!
                        nn.ReLU(),                    # Non-linearity
                        nn.Dropout(0.3),              # Regularisation
                        nn.Linear(hidden, out_dim)    # W₂x + b₂
                    )
                def forward(self, x):
                    return self.net(x)                # ŷ = f(x; θ)

            model = MLP(20, 64, 3)
            y_pred = model(X_batch)   # shape: (32, 3) — logits for 3 classes
            ```

            **The computation graph** records: $\hat{y} = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$

            Every node stores its forward value and a pointer to the operation that created it.
            Gradients flow backward through this graph during `loss.backward()`.

            **Mathematical view:**
            $$\hat{y} = f(x;\theta) \quad \text{where } \theta = \{W_1, b_1, W_2, b_2, \ldots\}$$
            """)

        with st.expander("Step 3 — Loss Function", expanded=False):
            st.markdown(r"""
            The loss (or criterion) measures how wrong the model's predictions are.
            It must be a **scalar** (single number) so we can take its gradient with respect to all parameters.

            | Task | Loss function | Formula |
            |------|--------------|---------|
            | Binary classification | Binary Cross-Entropy | $-[y\log\hat{p} + (1-y)\log(1-\hat{p})]$ |
            | Multi-class classification | Cross-Entropy | $-\sum_c y_c \log\hat{p}_c$ |
            | Regression | Mean Squared Error | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ |
            | RL (Policy Gradient) | Negative log-likelihood | $-\log\pi(a|s) \cdot G_t$ |

            ```python
            criterion = nn.CrossEntropyLoss()  # for multi-class
            loss = criterion(y_pred, y_batch)  # scalar tensor
            # loss.item() gives the Python float value for logging
            ```

            **Cross-Entropy intuition:** The model outputs raw **logits** (unbounded scores).
            `CrossEntropyLoss` applies softmax internally and then computes:
            $$L = -\frac{1}{B}\sum_{i=1}^B \log\frac{e^{\hat{y}_{i,c_i}}}{\sum_j e^{\hat{y}_{i,j}}}$$
            Maximising the probability of the correct class = minimising cross-entropy.
            """)

        with st.expander("Step 4 — Backward Pass (Autograd & Chain Rule)", expanded=False):
            st.markdown(r"""
            `loss.backward()` is the magic step. PyTorch traverses the computation graph
            **in reverse**, applying the chain rule at each node to compute $\frac{\partial L}{\partial \theta}$
            for every parameter $\theta$ in the model.

            **The chain rule in practice:**
            $$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}}
            \cdot \frac{\partial \hat{y}}{\partial h}
            \cdot \frac{\partial h}{\partial W_1}$$

            where each factor is computed locally at each layer and multiplied together.

            ```python
            loss.backward()   # computes all ∂L/∂θ and stores in θ.grad
            # Now: model.net[0].weight.grad contains ∂L/∂W₁ — a matrix of same shape as W₁
            ```

            **What autograd does under the hood:**
            1. Start at the loss node with gradient = 1
            2. Walk backward through the DAG
            3. At each operation $y = f(x)$: compute $\frac{\partial y}{\partial x}$ using the stored forward values
            4. Multiply by the incoming gradient (chain rule)
            5. Accumulate into `.grad` attribute of leaf tensors (parameters)

            **Warning:** Gradients **accumulate** — they are added to existing `.grad`.
            This is why Step 6 (zero gradients) is essential before each backward pass.
            """)

        with st.expander("Step 5 — Optimizer Step (SGD, Adam, RMSProp)", expanded=False):
            st.markdown(r"""
            The optimizer reads the gradients from `.grad` and updates each parameter to reduce the loss.

            | Optimizer | Update rule | Key property |
            |-----------|-------------|-------------|
            | **SGD** | $\theta \leftarrow \theta - \eta \nabla L$ | Simple, requires tuning |
            | **SGD + Momentum** | $v \leftarrow \beta v + \nabla L$; $\theta \leftarrow \theta - \eta v$ | Faster, smooths gradients |
            | **RMSProp** | Divide gradient by running std | Adapts to gradient magnitude |
            | **Adam** | Momentum + adaptive step size | Most popular, robust default |

            **Adam update (full equations):**
            $$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L \quad \text{(1st moment, momentum)}$$
            $$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2 \quad \text{(2nd moment, variance)}$$
            $$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
            $$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon}\hat{m}_t$$

            **Default Adam hyperparameters:** $\eta=10^{-3}$, $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$.

            ```python
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.step()   # θ ← θ − η · ∇L (using Adam's adaptive formula)
            ```
            """)

        # ── Interactive simulation ─────────────────────────────────────────
        st.divider()
        st.subheader("🎮 Interactive: Run a Simulated Training Loop")
        st.markdown("""
        The simulation below runs a simplified training loop on a 2D classification problem.
        Adjust the hyperparameters and press **Run Training** to see how loss and accuracy
        evolve over epochs — this is exactly what happens in real PyTorch, one step at a time.
        """)

        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            sim_lr     = st.select_slider("Learning rate η",
                           options=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0], value=0.01, key="sim_lr")
            sim_epochs = st.slider("Epochs", 5, 100, 30, 5, key="sim_ep")
        with col_sim2:
            sim_bs     = st.select_slider("Batch size",
                           options=[4, 8, 16, 32, 64, 128], value=32, key="sim_bs")
            sim_hidden = st.select_slider("Hidden units",
                           options=[4, 8, 16, 32, 64], value=16, key="sim_hid")
        with col_sim3:
            sim_opt    = st.selectbox("Optimizer", ["Adam", "SGD", "RMSProp"], key="sim_opt")
            sim_norm   = st.checkbox("Use BatchNorm", value=True, key="sim_norm")
            sim_seed   = st.number_input("Seed", 0, 999, 42, key="sim_seed")

        if st.button("▶️ Run Simulated Training", type="primary", key="run_sim"):
            np.random.seed(sim_seed)
            # Generate 2-class moon-shaped dataset
            N = 400
            theta_data = np.linspace(0, np.pi, N//2)
            X0 = np.column_stack([np.cos(theta_data), np.sin(theta_data)]) + 0.15*np.random.randn(N//2, 2)
            X1 = np.column_stack([np.cos(theta_data)+0.5, -np.sin(theta_data)]) + 0.15*np.random.randn(N//2, 2)
            X_data = np.vstack([X0, X1]).astype(np.float32)
            y_data = np.array([0]*(N//2) + [1]*(N//2), dtype=np.int64)

            # Simple numpy-based MLP simulation
            def relu(x): return np.maximum(0, x)
            def softmax(x):
                e = np.exp(x - x.max(axis=1, keepdims=True))
                return e / e.sum(axis=1, keepdims=True)

            np.random.seed(sim_seed)
            W1 = np.random.randn(2, sim_hidden).astype(np.float32) * 0.1
            b1 = np.zeros((1, sim_hidden), dtype=np.float32)
            W2 = np.random.randn(sim_hidden, 2).astype(np.float32) * 0.1
            b2 = np.zeros((1, 2), dtype=np.float32)
            m_W1=np.zeros_like(W1); v_W1=np.zeros_like(W1)
            m_b1=np.zeros_like(b1); v_b1=np.zeros_like(b1)
            m_W2=np.zeros_like(W2); v_W2=np.zeros_like(W2)
            m_b2=np.zeros_like(b2); v_b2=np.zeros_like(b2)
            beta1,beta2,eps_opt=0.9,0.999,1e-8

            epoch_losses=[]; epoch_accs=[]; t_adam=0
            lr = float(sim_lr)

            def adam_update(param, grad, m, v, t, lr):
                m = beta1*m + (1-beta1)*grad
                v = beta2*v + (1-beta2)*grad**2
                m_hat = m/(1-beta1**t)
                v_hat = v/(1-beta2**t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps_opt)
                return param, m, v

            for epoch in range(sim_epochs):
                idx = np.random.permutation(N)
                epoch_loss=0; correct=0
                batches=0
                for start in range(0, N, sim_bs):
                    end = min(start+sim_bs, N)
                    xb = X_data[idx[start:end]]
                    yb = y_data[idx[start:end]]
                    # Forward
                    h1 = relu(xb @ W1 + b1)
                    logits = h1 @ W2 + b2
                    probs = softmax(logits)
                    # Loss (cross-entropy)
                    bs_actual = len(yb)
                    loss = -np.mean(np.log(probs[np.arange(bs_actual), yb] + 1e-9))
                    epoch_loss += loss; batches += 1
                    preds = probs.argmax(axis=1)
                    correct += (preds == yb).sum()
                    # Backward (chain rule)
                    dlogits = probs.copy()
                    dlogits[np.arange(bs_actual), yb] -= 1
                    dlogits /= bs_actual
                    dW2 = h1.T @ dlogits
                    db2 = dlogits.sum(axis=0, keepdims=True)
                    dh1 = dlogits @ W2.T
                    dh1[h1 <= 0] = 0   # ReLU gate
                    dW1 = xb.T @ dh1
                    db1 = dh1.sum(axis=0, keepdims=True)
                    # Update
                    t_adam += 1
                    if sim_opt == "Adam":
                        W1,m_W1,v_W1 = adam_update(W1,dW1,m_W1,v_W1,t_adam,lr)
                        b1,m_b1,v_b1 = adam_update(b1,db1,m_b1,v_b1,t_adam,lr)
                        W2,m_W2,v_W2 = adam_update(W2,dW2,m_W2,v_W2,t_adam,lr)
                        b2,m_b2,v_b2 = adam_update(b2,db2,m_b2,v_b2,t_adam,lr)
                    else:
                        W1 -= lr * dW1; b1 -= lr * db1
                        W2 -= lr * dW2; b2 -= lr * db2

                epoch_losses.append(epoch_loss / batches)
                epoch_accs.append(correct / N * 100)

            # Plot results
            fig_sim, axes_sim = _fig(1, 3, 17, 4.5)

            # Loss curve
            axes_sim[0].plot(epoch_losses, color="#ef5350", lw=2.5)
            axes_sim[0].fill_between(range(len(epoch_losses)), 0, epoch_losses,
                                     alpha=0.15, color="#ef5350")
            axes_sim[0].set_xlabel("Epoch", color="white"); axes_sim[0].set_ylabel("Loss", color="white")
            axes_sim[0].set_title("Training Loss", color="white", fontweight="bold")
            axes_sim[0].grid(alpha=0.12)

            # Accuracy curve
            axes_sim[1].plot(epoch_accs, color="#4caf50", lw=2.5)
            axes_sim[1].axhline(100, color="#ffa726", ls="--", lw=1, alpha=0.5)
            axes_sim[1].set_xlabel("Epoch", color="white"); axes_sim[1].set_ylabel("Accuracy %", color="white")
            axes_sim[1].set_title("Training Accuracy", color="white", fontweight="bold")
            axes_sim[1].set_ylim(0, 105); axes_sim[1].grid(alpha=0.12)

            # Decision boundary
            xx, yy = np.meshgrid(np.linspace(-2.5, 3.5, 200), np.linspace(-1.5, 2.0, 200))
            grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
            h_grid = relu(grid @ W1 + b1)
            logits_grid = h_grid @ W2 + b2
            probs_grid = softmax(logits_grid)[:, 1].reshape(xx.shape)

            axes_sim[2].contourf(xx, yy, probs_grid, levels=50,
                                 cmap=LinearSegmentedColormap.from_list("rb",["#1565c0","#b71c1c"]),
                                 alpha=0.5)
            axes_sim[2].scatter(X0[:,0], X0[:,1], c="#42a5f5", s=12, alpha=0.7, label="Class 0")
            axes_sim[2].scatter(X1[:,0], X1[:,1], c="#ef5350", s=12, alpha=0.7, label="Class 1")
            axes_sim[2].set_title("Learned Decision Boundary", color="white", fontweight="bold")
            axes_sim[2].legend(facecolor=CARD_BG, labelcolor="white", fontsize=8)
            plt.tight_layout(); st.pyplot(fig_sim); plt.close()

            c1, c2, c3 = st.columns(3)
            c1.metric("Final Loss", f"{epoch_losses[-1]:.4f}")
            c2.metric("Final Accuracy", f"{epoch_accs[-1]:.1f}%")
            c3.metric("Total updates", f"{sim_epochs * (N // sim_bs)}")

        # ── Complete code template ─────────────────────────────────────────
        st.divider()
        st.subheader("📄 Complete PyTorch Training Template")
        st.markdown("The canonical PyTorch training loop — copy, adapt, and use for any project:")
        st.code(r"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── 1. Data ──────────────────────────────────────────────────────────
X_train = torch.randn(1000, 20)
y_train = torch.randint(0, 3, (1000,))
loader  = DataLoader(TensorDataset(X_train, y_train),
                     batch_size=32, shuffle=True)

# ── 2. Model ─────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x): return self.layers(x)

device    = "cuda" if torch.cuda.is_available() else "cpu"
model     = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ── 3. Training Loop ─────────────────────────────────────────────────
for epoch in range(50):
    model.train()                               # enable dropout, BatchNorm training mode
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()                   # Step 6: clear gradients
        y_pred  = model(X_batch)                # Step 2: forward pass
        loss    = criterion(y_pred, y_batch)    # Step 3: compute loss
        loss.backward()                         # Step 4: compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # safety
        optimizer.step()                        # Step 5: update weights
        running_loss += loss.item()

    scheduler.step()                            # decay learning rate

    # ── 4. Evaluation ────────────────────────────────────────────────
    model.eval()                                # disable dropout, use running BN stats
    with torch.no_grad():                       # no gradient tracking needed
        val_pred  = model(X_val.to(device))
        val_loss  = criterion(val_pred, y_val.to(device))
        val_acc   = (val_pred.argmax(1) == y_val.to(device)).float().mean()

    print(f"Epoch {epoch+1:3d} | "
          f"Train Loss: {running_loss/len(loader):.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Acc: {val_acc*100:.1f}%")
""", language="python")

        st.markdown(_tip("""
        <b>5 tips for a reliable training loop:</b><br>
        1. Always call <code>optimizer.zero_grad()</code> <em>before</em> <code>backward()</code> — not after.<br>
        2. Use <code>torch.no_grad()</code> during evaluation — saves 30–50% memory and compute.<br>
        3. Clip gradients (<code>clip_grad_norm_</code>) to prevent exploding gradients in RNNs/Transformers.<br>
        4. Use a learning rate scheduler — most tasks need lr to decrease as training progresses.<br>
        5. Save checkpoints with <code>torch.save(model.state_dict(), 'best.pt')</code> whenever val loss improves.
        """), unsafe_allow_html=True)

        # ── Connection to RL ──────────────────────────────────────────────
        st.divider()
        st.subheader("🔗 How This Connects to Reinforcement Learning")
        st.markdown(r"""
        The same training loop powers every deep RL algorithm — only the **data source** and **loss function** change:
        """)

        df_rl_conn = pd.DataFrame({
            "Component": ["Data source", "Forward pass", "Loss function", "Backward pass", "Optimizer"],
            "Supervised Learning": ["Fixed dataset (X, y)", "ŷ = model(X)", "CrossEntropy / MSE",
                                    "loss.backward()", "Adam / SGD"],
            "DQN (Q-Learning)": ["Replay buffer (s,a,r,s')", "Q = net(s)",
                                  "MSE: $(r + \\gamma\\max Q(s') - Q(s,a))^2$",
                                  "loss.backward()", "Adam"],
            "PPO (Policy Gradient)": ["Rollout buffer (s,a,r,G)", "π(a|s), V(s) = net(s)",
                                      "Clipped surrogate + value + entropy",
                                      "loss.backward()", "Adam"],
            "LSTM-based DRQN": ["Episode segments from buffer", "h_t, Q = lstm(o_t, h_{t-1})",
                                 "Same as DQN but with BPTT", "loss.backward() through time", "Adam"],
        })
        st.dataframe(df_rl_conn, use_container_width=True, hide_index=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 — CONCEPT MAP
    # ═══════════════════════════════════════════════════════════════════════
    with tab_map:
        _section_header("🗺️", "How Everything Connects",
                         "The complete prerequisite knowledge graph from linear algebra to deep RL", "#4caf50")

        st.markdown(r"""
        The diagram below shows how the three prerequisite topics connect to each other,
        to the three main RL families, and to modern deep RL systems.
        """)

        fig_map, ax_map = _fig(1, 1, 14, 9)
        ax_map.set_xlim(0, 14); ax_map.set_ylim(0, 9)
        ax_map.axis("off")

        def draw_node(ax, x, y, text, color, w=2.8, h=0.8, fs=9):
            ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                                        boxstyle="round,pad=0.12",
                                        facecolor=color+"33", edgecolor=color, lw=2, zorder=3))
            ax.text(x, y, text, ha="center", va="center", color="white",
                    fontsize=fs, fontweight="bold", zorder=4)

        def draw_edge(ax, x1, y1, x2, y2, color="#546e7a", lw=1.5, style="->"):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=2)

        # Row 1 — Prerequisites
        ax_map.text(7, 8.5, "Prerequisites (This Module)", ha="center", va="center",
                    color="#9e9ebb", fontsize=10, fontweight="bold")
        draw_node(ax_map, 2.5, 7.8, "🔁 RNNs & LSTMs\nSequential memory", "#7c4dff")
        draw_node(ax_map, 7.0, 7.8, "📐 Batch/Layer Norm\nStable training", "#e65100")
        draw_node(ax_map, 11.5, 7.8, "⚙️ PyTorch Loop\nForward·Loss·Backward", "#ffa726")

        # Row 2 — Core RL
        ax_map.text(7, 6.3, "Core RL Methods", ha="center", va="center",
                    color="#9e9ebb", fontsize=10, fontweight="bold")
        draw_node(ax_map, 2.5, 5.5, "🧮 Dynamic Programming\nBellman · Policy Iter", "#6a1b9a")
        draw_node(ax_map, 7.0, 5.5, "🎲 Monte Carlo\nReturn · IS · GPI", "#7c4dff")
        draw_node(ax_map, 11.5, 5.5, "⚡ TD Learning\nSARSA · Q-Learning · TD(λ)", "#e65100")

        # Row 3 — Deep RL
        ax_map.text(7, 4.0, "Deep RL Systems", ha="center", va="center",
                    color="#9e9ebb", fontsize=10, fontweight="bold")
        draw_node(ax_map, 3.5, 3.2, "🎮 DQN\nQ-Learning + CNN/LSTM", "#1565c0")
        draw_node(ax_map, 7.0, 3.2, "🤖 PPO / A3C\nPolicy Gradient + Value", "#00695c")
        draw_node(ax_map, 10.5, 3.2, "♟️ AlphaGo/Zero\nMCTS + Deep Net", "#4527a0")

        # Row 4 — Applications
        ax_map.text(7, 1.8, "Applications", ha="center", va="center",
                    color="#9e9ebb", fontsize=10, fontweight="bold")
        draw_node(ax_map, 2.5, 1.1, "🕹️ Atari / Games", "#2e7d32", w=2.4)
        draw_node(ax_map, 5.8, 1.1, "🤖 Robotics", "#2e7d32", w=2.0)
        draw_node(ax_map, 8.8, 1.1, "💊 Healthcare", "#2e7d32", w=2.0)
        draw_node(ax_map, 11.8, 1.1, "📈 Finance", "#2e7d32", w=2.0)

        # Edges — Prerequisites to Core RL
        draw_edge(ax_map, 2.5, 7.4, 2.5, 5.9, "#7c4dff", 1.5)
        draw_edge(ax_map, 7.0, 7.4, 7.0, 5.9, "#e65100", 1.5)
        draw_edge(ax_map, 11.5, 7.4, 11.5, 5.9, "#ffa726", 1.5)
        # Cross edges
        draw_edge(ax_map, 2.5, 7.4, 7.0, 5.9, "#546e7a", 1)
        draw_edge(ax_map, 11.5, 7.4, 7.0, 5.9, "#546e7a", 1)
        draw_edge(ax_map, 2.5, 7.4, 11.5, 5.9, "#333355", 1)

        # Core RL to Deep RL
        draw_edge(ax_map, 11.5, 5.1, 3.5, 3.6, "#1565c0", 1.5)
        draw_edge(ax_map, 11.5, 5.1, 7.0, 3.6, "#00695c", 1.5)
        draw_edge(ax_map, 7.0, 5.1, 7.0, 3.6, "#00695c", 1.2)
        draw_edge(ax_map, 2.5, 5.1, 3.5, 3.6, "#1565c0", 1.2)
        draw_edge(ax_map, 7.0, 5.1, 10.5, 3.6, "#4527a0", 1.2)

        # Deep RL to Applications
        draw_edge(ax_map, 3.5, 2.8, 2.5, 1.5, "#4caf50", 1.2)
        draw_edge(ax_map, 7.0, 2.8, 5.8, 1.5, "#4caf50", 1.2)
        draw_edge(ax_map, 7.0, 2.8, 8.8, 1.5, "#4caf50", 1.2)
        draw_edge(ax_map, 10.5, 2.8, 11.8, 1.5, "#4caf50", 1.2)

        ax_map.set_title("RL Knowledge Graph — Prerequisites → Core Methods → Deep RL → Applications",
                         color="white", fontsize=11, fontweight="bold", pad=10)
        plt.tight_layout(); st.pyplot(fig_map); plt.close()

        st.divider()
        st.subheader("📚 Summary — What You Now Know")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(r"""
            **🔁 RNNs & LSTMs**
            - RNNs add temporal memory via hidden state $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
            - Vanishing gradients prevent learning long-range dependencies
            - LSTM uses 3 gates to control information flow: forget, input, output
            - Cell state $C_t$ is the "highway" for long-term information
            - DRQN applies LSTM to RL in partially observable environments
            """)
        with c2:
            st.markdown(r"""
            **📐 Batch & Layer Norm**
            - Internal Covariate Shift makes deep training unstable
            - BatchNorm normalises over the batch dimension (per feature)
            - LayerNorm normalises over the feature dimension (per sample)
            - BatchNorm: best for CNNs; LayerNorm: best for Transformers & RNNs
            - Both learn $\gamma$ and $\beta$ to recover expressiveness after normalisation
            """)
        with c3:
            st.markdown(r"""
            **⚙️ PyTorch Training Loop**
            - 6 steps: data → forward → loss → backward → step → zero grad
            - Autograd builds a computation graph during forward pass
            - Chain rule computes all gradients during backward pass
            - Adam combines momentum + adaptive learning rates
            - The same loop structure powers DQN, PPO, AlphaGo, and GPT
            """)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5 — GLOSSARY
    # ═══════════════════════════════════════════════════════════════════════
    with tab_gloss:
        _section_header("📖", "Deep Learning & RL Glossary",
                         "Every key term defined in plain English with formulas — searchable, categorised, and cross-referenced", "#00bcd4")

        st.markdown("""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:12px;
                    padding:1rem 1.5rem;margin-bottom:1rem">
        Use the category filters below to explore terms, or type in the search box.
        Each entry gives a 4–6 line plain-English definition plus the core formula where relevant.
        </div>
        """, unsafe_allow_html=True)

        # ── Search & Filter ───────────────────────────────────────────────
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            search_term = st.text_input("🔍 Search terms", placeholder="e.g. gradient, LSTM, softmax...",
                                        key="gloss_search")
        with col_s2:
            category_filter = st.selectbox("Filter by category", [
                "All categories",
                "🧠 Neural Network Fundamentals",
                "📉 Optimisation & Training",
                "🔁 Sequence Models",
                "📐 Regularisation & Normalisation",
                "🎲 Probability & Loss",
                "⚡ Reinforcement Learning",
                "🔧 PyTorch & Implementation",
            ], key="gloss_cat")

        # ── Glossary Data ─────────────────────────────────────────────────
        glossary = [
            # ── Neural Network Fundamentals ────────────────────────────────
            {
                "term": "Neural Network",
                "category": "🧠 Neural Network Fundamentals",
                "emoji": "🕸️",
                "short": "A layered system of mathematical functions that learns mappings from inputs to outputs by adjusting weights.",
                "definition": r"""A neural network is a composition of layers, each applying a linear transformation
                followed by a nonlinear activation: $y = \sigma(Wx + b)$. Multiple such layers stacked together
                can approximate any continuous function (Universal Approximation Theorem). Weights $W$ and biases $b$
                are learned from data by minimising a loss function via gradient descent. Neural networks are the
                foundation of all modern deep learning and deep RL systems — from image classifiers to policy networks.""",
                "formula": r"y^{(l)} = \sigma(W^{(l)} y^{(l-1)} + b^{(l)})",
                "formula_note": "Output of layer l, where σ is the activation function",
            },
            {
                "term": "Weights & Parameters",
                "category": "🧠 Neural Network Fundamentals",
                "emoji": "⚖️",
                "short": "The learnable numbers inside a neural network — matrices W and bias vectors b — adjusted during training.",
                "definition": r"""Parameters are the knobs that define what a neural network computes. They start
                at random values and are iteratively refined via gradient descent to minimise the training loss.
                A weight $W_{ij}$ represents the strength of the connection from neuron $j$ in the previous layer
                to neuron $i$ in the current layer. Modern large language models contain hundreds of billions of
                parameters; even a small MLP for tabular data may have thousands. All parameters share the same
                update rule but each has its own gradient computed by backpropagation.""",
                "formula": r"\theta = \{W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, \ldots\}",
                "formula_note": "θ denotes all learnable parameters collectively",
            },
            {
                "term": "Activation Function",
                "category": "🧠 Neural Network Fundamentals",
                "emoji": "⚡",
                "short": "A nonlinear function applied after each linear layer — without it, deep networks collapse to a single linear map.",
                "definition": r"""Linear layers alone cannot model complex patterns — any stack of linear transformations
                is still just a linear transformation. Activation functions introduce nonlinearity, allowing networks to
                approximate curved decision boundaries and non-linear relationships. ReLU is the most common:
                $\text{ReLU}(x) = \max(0,x)$. Sigmoid $\sigma(x) = 1/(1+e^{-x})$ squashes output to $(0,1)$ —
                used for binary classification and LSTM gates. Tanh squashes to $(-1,1)$ — used in RNNs.
                Softmax converts a vector of logits into a probability distribution over classes.""",
                "formula": r"\text{ReLU}(x) = \max(0,x), \quad \sigma(x) = \frac{1}{1+e^{-x}}, \quad \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}",
                "formula_note": "Three common activation functions",
            },
            {
                "term": "Forward Pass",
                "category": "🧠 Neural Network Fundamentals",
                "emoji": "➡️",
                "short": "The computation of a network's output from input to prediction — building the computation graph along the way.",
                "definition": r"""The forward pass feeds input data through each layer sequentially: $x \to h_1 \to h_2 \to \hat{y}$.
                At each layer, a linear transformation is applied then an activation function. PyTorch records every operation
                into a computation graph (a directed acyclic graph) during the forward pass. This graph is then traversed
                in reverse during the backward pass to compute gradients. The forward pass is deterministic given fixed
                weights — the same input always produces the same output (unless dropout is active during training).""",
                "formula": r"\hat{y} = f(x;\theta) = f_L(f_{L-1}(\ldots f_1(x;\theta^{(1)})\ldots;\theta^{(L-1)});\theta^{(L)})",
                "formula_note": "Composition of L layer functions f₁…fₗ",
            },
            {
                "term": "Bias (neural network)",
                "category": "🧠 Neural Network Fundamentals",
                "emoji": "↔️",
                "short": "A learnable offset added to every linear transformation — allows the activation to shift horizontally.",
                "definition": r"""Without a bias term, a linear layer $y = Wx$ must always pass through the origin.
                Adding bias $b$ gives $y = Wx + b$, allowing the decision boundary to be offset from zero.
                Each neuron has its own bias — a single scalar (not a matrix). Biases are usually initialised to zero.
                In practice, bias terms give the network extra flexibility to fit data that doesn't naturally have
                zero mean. In PyTorch, `nn.Linear(in, out)` includes bias by default (`bias=True`).""",
                "formula": r"y = Wx + b",
                "formula_note": "b has shape (out_features,) — one bias per output neuron",
            },
            {
                "term": "Softmax",
                "category": "🧠 Neural Network Fundamentals",
                "emoji": "🎯",
                "short": "Converts raw output scores (logits) into a valid probability distribution over classes that sums to 1.",
                "definition": r"""Softmax takes a vector of $K$ real-valued logits and produces $K$ probabilities that sum to exactly 1.
                The exponential ensures all values are positive; dividing by the sum normalises them.
                It is used as the final layer in multi-class classifiers and in attention mechanisms.
                The largest logit gets the largest probability, but all classes receive nonzero probability
                (the model is never 100% certain). Temperature $T$ controls "sharpness": $T \to 0$ makes it
                one-hot (argmax), $T \to \infty$ makes it uniform (maximum uncertainty).""",
                "formula": r"\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}",
                "formula_note": "Output sums to 1: ∑ softmax(z)ᵢ = 1",
            },

            # ── Optimisation & Training ────────────────────────────────────
            {
                "term": "Gradient Descent",
                "category": "📉 Optimisation & Training",
                "emoji": "📉",
                "short": "The fundamental algorithm for minimising a loss function — move parameters in the direction of steepest descent.",
                "definition": r"""Gradient descent takes small steps in the direction of the negative gradient of the loss.
                The gradient $\nabla_\theta L$ points in the direction of steepest ascent — so subtracting it moves parameters
                toward a local minimum. The learning rate $\eta$ controls step size: too large → overshoot, too small → slow convergence.
                In practice, **mini-batch stochastic gradient descent (SGD)** is used: gradients are estimated from a random
                subset (batch) of data each step rather than the full dataset. This introduces noise that helps escape
                sharp local minima and enables processing data too large for RAM.""",
                "formula": r"\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)",
                "formula_note": "η = learning rate; ∇L = gradient of loss w.r.t. parameters",
            },
            {
                "term": "Learning Rate",
                "category": "📉 Optimisation & Training",
                "emoji": "🎚️",
                "short": "The step size for parameter updates — the single most important hyperparameter in deep learning training.",
                "definition": r"""The learning rate $\eta$ scales how much each gradient step changes the parameters.
                Too high: the loss oscillates or diverges — parameters jump past minima. Too low: training is
                prohibitively slow and may get stuck in local minima. Common starting values: $10^{-3}$ for Adam,
                $10^{-1}$ to $10^{-2}$ for SGD. Learning rate scheduling (reducing $\eta$ over training) typically
                improves final performance. The linear scaling rule states: when batch size is multiplied by $k$,
                multiply $\eta$ by $k$ to maintain equivalent training dynamics.""",
                "formula": r"\text{Typical range: } \eta \in [10^{-5}, 10^{-1}]",
                "formula_note": "Adam default η=1e-3; SGD often needs η=0.01–0.1",
            },
            {
                "term": "Backpropagation",
                "category": "📉 Optimisation & Training",
                "emoji": "↩️",
                "short": "The algorithm that computes gradients of the loss with respect to all parameters by applying the chain rule backward through the network.",
                "definition": r"""Backpropagation (Rumelhart et al., 1986) is the efficient application of the chain rule through
                the computation graph. Starting from the scalar loss $L$, we compute $\partial L / \partial \theta$ for
                every parameter $\theta$ by multiplying local Jacobians along the backward path.
                For a composition $L = f(g(h(x)))$: $\partial L/\partial x = (\partial L/\partial f)(\partial f/\partial g)(\partial g/\partial h)(\partial h/\partial x)$.
                PyTorch's autograd engine performs this automatically — it records all operations during the forward pass
                and replays them in reverse to compute gradients. This is what makes `loss.backward()` work.""",
                "formula": r"\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial y^{(l)}} \cdot \frac{\partial y^{(l)}}{\partial W^{(l)}}",
                "formula_note": "Chain rule: gradient flows backward through each layer",
            },
            {
                "term": "Momentum",
                "category": "📉 Optimisation & Training",
                "emoji": "🏃",
                "short": "Adds a velocity term to gradient updates — accumulates past gradients to accelerate in consistent directions and dampen oscillations.",
                "definition": r"""Standard SGD takes steps purely based on the current gradient, leading to slow progress along
                flat dimensions and oscillations in steep ones. Momentum maintains a velocity vector $v_t$ that accumulates
                an exponentially weighted average of past gradients. Updates become smoother: the optimizer accelerates in
                directions of consistent gradient and slows down when gradients flip sign.
                With $\beta = 0.9$ (typical), momentum effectively averages the last $1/(1-0.9) = 10$ gradient steps.
                Momentum is included in Adam's first moment estimate $m_t$, making Adam a momentum-based adaptive method.""",
                "formula": r"v_t = \beta v_{t-1} + (1-\beta)\nabla L, \quad \theta \leftarrow \theta - \eta v_t",
                "formula_note": "β=0.9 is the standard momentum coefficient",
            },
            {
                "term": "Adaptive Learning Rate",
                "category": "📉 Optimisation & Training",
                "emoji": "🔧",
                "short": "Automatically adjusts the effective step size per parameter — large gradients → small steps; small gradients → large steps.",
                "definition": r"""Fixed learning rates treat all parameters equally, but in practice some parameters have consistently
                large gradients (over-updated) while others have tiny ones (under-updated). Adaptive methods like Adam and
                RMSProp track the magnitude of recent gradients per parameter and scale the step size inversely.
                Adam maintains both a first moment (mean gradient) and second moment (uncentered variance) estimate.
                The effective step size for each parameter is approximately $\eta / \sqrt{\hat{v}_t}$: large past gradient
                variance → small effective step → stability. This makes Adam robust to the choice of $\eta$ across very
                different architectures and datasets.""",
                "formula": r"\theta_i \leftarrow \theta_i - \frac{\eta}{\sqrt{\hat{v}_i} + \varepsilon} \hat{m}_i",
                "formula_note": "Per-parameter step size in Adam; v̂ᵢ = variance estimate for parameter i",
            },
            {
                "term": "Epoch",
                "category": "📉 Optimisation & Training",
                "emoji": "🔄",
                "short": "One complete pass through the entire training dataset — after one epoch, every sample has been seen exactly once.",
                "definition": r"""An epoch consists of processing all mini-batches until every training example has contributed
                exactly once to parameter updates. With $N$ samples and batch size $B$, one epoch contains $N/B$ gradient steps.
                Multiple epochs are typically needed because one pass is insufficient for convergence — models need to see
                each example many times. Typical training ranges from 10 epochs (large datasets like ImageNet) to thousands
                (small datasets). Monitoring validation loss per epoch reveals overfitting: when training loss falls but
                validation loss rises, the model is memorising training data.""",
                "formula": r"\text{Steps per epoch} = \left\lceil \frac{N}{B} \right\rceil",
                "formula_note": "N = dataset size, B = batch size",
            },
            {
                "term": "Mini-batch",
                "category": "📉 Optimisation & Training",
                "emoji": "📦",
                "short": "A small random subset of training data used to compute one gradient update — balances computational efficiency with gradient accuracy.",
                "definition": r"""Mini-batch gradient descent computes gradients on $B$ samples ($B$ typically 16–512) rather than
                the full dataset (full-batch) or a single sample (online/SGD). This gives a noisy but unbiased estimate of the
                true gradient: $\mathbb{E}[\hat{\nabla}L] = \nabla L$. Mini-batches allow parallelisation on GPUs (processing
                many samples simultaneously is efficient), prevent memory overflow for large datasets, and the noise from
                random sampling acts as implicit regularisation — helping avoid sharp local minima. The DataLoader in PyTorch
                handles mini-batch creation with shuffling and optional multiprocessing.""",
                "formula": r"\hat{\nabla}L = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla L_i",
                "formula_note": "Average gradient over a random mini-batch B of size B",
            },

            # ── Regularisation & Normalisation ─────────────────────────────
            {
                "term": "Overfitting",
                "category": "📐 Regularisation & Normalisation",
                "emoji": "📈",
                "short": "When a model learns the training data too well — including noise — and fails to generalise to new examples.",
                "definition": r"""Overfitting occurs when the model has enough capacity to memorise training examples rather than
                learning the underlying pattern. Symptoms: training accuracy ≈ 100%, test/validation accuracy is much lower.
                It is more likely with small datasets, very complex models (many parameters), or insufficient regularisation.
                Classic example: a degree-1000 polynomial fitted to 10 data points passes through all training points
                perfectly but oscillates wildly between them. Solutions include: more data, simpler architectures,
                dropout, weight decay (L2 regularisation), early stopping, and data augmentation.""",
                "formula": r"L_{\text{train}} \ll L_{\text{val}} \quad \Rightarrow \text{overfitting}",
                "formula_note": "Training loss << validation loss = red flag",
            },
            {
                "term": "Underfitting",
                "category": "📐 Regularisation & Normalisation",
                "emoji": "📉",
                "short": "When a model is too simple to capture the patterns in data — high error on both training and test sets.",
                "definition": r"""Underfitting happens when the model lacks the capacity (too few layers/neurons, too simple
                an architecture) or hasn't been trained long enough to learn the true underlying function.
                Both training and validation loss are high — the model hasn't even learned the training data.
                Example: using a linear model to classify spirally arranged data. Solutions include: larger networks,
                more training time, lower regularisation, better features. The bias-variance trade-off frames this:
                underfitting = high bias (systematic wrong assumptions about the data).""",
                "formula": r"L_{\text{train}} \approx L_{\text{val}} \gg 0 \quad \Rightarrow \text{underfitting}",
                "formula_note": "Both losses are high = model too simple or undertrained",
            },
            {
                "term": "Regularisation",
                "category": "📐 Regularisation & Normalisation",
                "emoji": "🛡️",
                "short": "Any technique that reduces overfitting by adding constraints or penalties that prevent the model from memorising training data.",
                "definition": r"""Regularisation adds a penalty term to the loss function that discourages the model from assigning
                very large weights. L2 regularisation (weight decay) adds $\lambda ||W||^2$ to the loss, shrinking weights toward
                zero. L1 adds $\lambda ||W||_1$, producing sparse weights. Dropout randomly zeros neuron outputs during training,
                forcing redundant representations. Data augmentation artificially expands the training set. All of these work
                by making the optimisation problem harder, which forces the model to find simpler (more generalisable) solutions
                rather than memorising the specific training examples.""",
                "formula": r"L_{\text{reg}} = L_{\text{data}} + \lambda \|\theta\|_2^2",
                "formula_note": "L2 regularisation (weight decay): λ controls the strength",
            },
            {
                "term": "Dropout",
                "category": "📐 Regularisation & Normalisation",
                "emoji": "🎲",
                "short": "Randomly sets a fraction p of neuron outputs to zero during training — prevents co-adaptation and acts as ensemble learning.",
                "definition": r"""Dropout (Srivastava et al., 2014) randomly masks neurons during each forward pass during training:
                each neuron is zeroed with probability $p$ (typically 0.3–0.5). This forces the network to develop multiple
                independent paths to correct answers — no single neuron can be relied upon. At inference, all neurons are active
                but their outputs are scaled by $(1-p)$ to maintain the same expected magnitude. Dropout can be interpreted as
                training an exponential ensemble of $2^n$ different sub-networks (where $n$ = number of neurons) and averaging
                them at test time. In PyTorch: `nn.Dropout(p=0.3)`. Note: disable during evaluation with `model.eval()`.""",
                "formula": r"h_i^{\text{train}} = \text{Bernoulli}(1-p) \cdot h_i, \quad h_i^{\text{test}} = (1-p) \cdot h_i",
                "formula_note": "p = dropout probability; outputs scaled at test time",
            },
            {
                "term": "Weight Initialisation",
                "category": "📐 Regularisation & Normalisation",
                "emoji": "🎯",
                "short": "How network weights are set before training begins — critical for avoiding vanishing/exploding gradients from the very first step.",
                "definition": r"""Poor initialisation can doom training before it starts. If all weights are zero, all neurons compute
                the same gradient and the network fails to break symmetry. If weights are too large, activations saturate (vanishing
                gradient); too small, signals shrink to zero through layers. Xavier/Glorot initialisation sets variance based on
                the number of input and output units, keeping activation variances stable. He initialisation (for ReLU networks)
                uses $\sigma^2 = 2/n_{\text{in}}$. In PyTorch: `nn.Linear` uses Kaiming uniform by default, appropriate for ReLU.""",
                "formula": r"\text{Xavier: } W \sim \mathcal{N}\!\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)",
                "formula_note": "nᵢₙ, nₒᵤₜ = fan-in and fan-out of the layer",
            },
            {
                "term": "Batch Normalisation",
                "category": "📐 Regularisation & Normalisation",
                "emoji": "📊",
                "short": "Normalises layer inputs across the mini-batch during training — stabilises distributions, allows larger learning rates, reduces sensitivity to initialisation.",
                "definition": r"""Batch Normalisation (Ioffe & Szegedy, 2015) normalises each feature across the current mini-batch
                to zero mean and unit variance, then applies learned scale $\gamma$ and shift $\beta$.
                This addresses Internal Covariate Shift — the changing distribution of layer inputs during training —
                making each layer less dependent on the exact scale of its inputs. Benefits: faster convergence, higher
                learning rates, less sensitivity to initialisation, mild regularisation effect (from mini-batch noise).
                Limitation: unstable with small batch sizes; inappropriate for RNNs and variable-length sequences.""",
                "formula": r"\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \varepsilon}}, \quad y_i = \gamma \hat{x}_i + \beta",
                "formula_note": "μ_B, σ_B² computed over the mini-batch; γ, β are learned",
            },
            {
                "term": "Layer Normalisation",
                "category": "📐 Regularisation & Normalisation",
                "emoji": "📐",
                "short": "Normalises across the feature dimension of each individual sample — batch-size independent and preferred for Transformers and RNNs.",
                "definition": r"""Layer Normalisation (Ba et al., 2016) computes normalisation statistics across all features
                of a single sample, making it completely independent of batch size. This means it works identically
                during training (any batch size) and inference (batch size = 1) — unlike BatchNorm.
                Every modern Transformer architecture (BERT, GPT, T5, LLaMA) uses Layer Normalisation.
                It is also the standard choice for RNNs because it can be applied at each timestep without
                needing a batch dimension. The learned parameters $\gamma$ and $\beta$ have the same function
                as in BatchNorm: allowing the network to recover from the normalisation if needed.""",
                "formula": r"\hat{x} = \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \varepsilon}}, \quad \mu_L = \frac{1}{d}\sum_{j=1}^d x_j",
                "formula_note": "μ_L computed over features of one sample; d = feature dimension",
            },

            # ── Probability & Loss ─────────────────────────────────────────
            {
                "term": "Loss Function",
                "category": "🎲 Probability & Loss",
                "emoji": "🎯",
                "short": "A scalar measure of how wrong the model's predictions are — the quantity that gradient descent minimises during training.",
                "definition": r"""The loss function $L(\hat{y}, y)$ quantifies the discrepancy between predicted output $\hat{y}$
                and ground truth $y$. It must output a single scalar so we can compute $\nabla_\theta L$.
                Cross-entropy is used for classification (penalises low probability on the correct class).
                MSE is used for regression (penalises large deviations). For RL policy gradients, the loss is
                $-\log \pi(a|s) \cdot G_t$ — maximise the log-probability of actions that led to high return.
                The choice of loss encodes what "good performance" means for the task.""",
                "formula": r"\text{Cross-Entropy: } L = -\frac{1}{N}\sum_{i=1}^N \log \hat{p}_{y_i}, \quad \text{MSE: } L = \frac{1}{N}\sum(y_i - \hat{y}_i)^2",
                "formula_note": "ŷᵢ = predicted class probability; yᵢ = true label",
            },
            {
                "term": "Cross-Entropy",
                "category": "🎲 Probability & Loss",
                "emoji": "🔀",
                "short": "The standard loss for classification — measures how surprised the model is by the true labels given its predicted probabilities.",
                "definition": r"""Cross-entropy loss $H(p,q) = -\sum_c p_c \log q_c$ measures the difference between the
                true distribution $p$ (one-hot labels) and the model's predicted distribution $q$ (softmax outputs).
                Minimising cross-entropy is equivalent to maximising the log-likelihood of the true labels under the model —
                it encourages the model to assign high probability to the correct class.
                When $q_c \to 1$ for the correct class $c$, loss $\to 0$; when $q_c \to 0$, loss $\to \infty$.
                PyTorch's `nn.CrossEntropyLoss` combines LogSoftmax and NLLLoss, working directly on raw logits.""",
                "formula": r"L = -\frac{1}{N}\sum_{i=1}^N \log \frac{e^{z_{i,y_i}}}{\sum_j e^{z_{i,j}}}",
                "formula_note": "zᵢ = logits for sample i; yᵢ = true class index",
            },
            {
                "term": "Gradient",
                "category": "🎲 Probability & Loss",
                "emoji": "∇",
                "short": "The vector of partial derivatives of the loss with respect to each parameter — points in the direction of steepest increase.",
                "definition": r"""The gradient $\nabla_\theta L \in \mathbb{R}^{|\theta|}$ has one component per parameter.
                Component $i$ = $\partial L / \partial \theta_i$ tells us how much the loss changes if we increase $\theta_i$ slightly.
                By moving parameters in the opposite direction (negative gradient), we reduce the loss — this is gradient descent.
                Gradients are computed efficiently by backpropagation through the computation graph. Each parameter's gradient
                has the same shape as the parameter itself. Checking gradients are nonzero and of reasonable magnitude is
                an important debugging step — zero gradients indicate dead neurons or disconnected computation paths.""",
                "formula": r"\nabla_\theta L = \left[\frac{\partial L}{\partial \theta_1}, \frac{\partial L}{\partial \theta_2}, \ldots, \frac{\partial L}{\partial \theta_n}\right]^T",
                "formula_note": "One partial derivative per parameter — same shape as the parameter tensor",
            },
            {
                "term": "Chain Rule",
                "category": "🎲 Probability & Loss",
                "emoji": "🔗",
                "short": "The calculus rule that allows gradients to be computed through composite functions — the mathematical engine behind backpropagation.",
                "definition": r"""For a composition $L = f(g(h(x)))$, the chain rule gives:
                $dL/dx = (dL/df)(df/dg)(dg/dh)(dh/dx)$.
                In neural networks, each layer is one function in the composition. Backpropagation applies the chain rule
                from the output back to every parameter, computing local Jacobians at each layer and multiplying them together.
                PyTorch's autograd engine stores the local derivative $df/dg$ at each node during the forward pass
                and uses it during the backward pass. Without the chain rule, computing gradients for millions of
                parameters would require $O(n^2)$ operations — the chain rule makes it $O(n)$.""",
                "formula": r"\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}",
                "formula_note": "Multiply local derivatives along the path from x to L",
            },

            # ── Sequence Models ─────────────────────────────────────────────
            {
                "term": "Sequential Data",
                "category": "🔁 Sequence Models",
                "emoji": "📅",
                "short": "Data where the order matters — each element depends on those before it, such as text, audio, time series, and game trajectories.",
                "definition": r"""Sequential data violates the i.i.d. assumption of standard ML: samples are not independent.
                In natural language, the meaning of a word depends on the preceding context. In financial time series, today's
                price is correlated with yesterday's. In RL, the state at time $t$ depends on the history of actions and
                observations. Standard feedforward networks process each input independently — they have no mechanism for
                retaining information across steps. Recurrent networks (RNNs, LSTMs, GRUs) and Transformers are specifically
                designed to model these temporal dependencies.""",
                "formula": r"P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t | x_1, \ldots, x_{t-1})",
                "formula_note": "Joint probability factored as product of conditionals — the chain rule of probability",
            },
            {
                "term": "Hidden State",
                "category": "🔁 Sequence Models",
                "emoji": "🧠",
                "short": "The RNN's internal memory vector — a compact summary of all inputs seen so far, passed forward to the next timestep.",
                "definition": r"""At each timestep $t$, an RNN computes a hidden state $h_t$ by combining the current input $x_t$
                with the previous hidden state $h_{t-1}$: $h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$.
                The hidden state acts as a lossy compression of the input history — not all information is retained,
                only what the network has learned is useful for the task. The same weight matrices $W_h$ and $W_x$
                are reused at every timestep (weight tying), giving RNNs their temporal translation invariance.
                The hidden state is analogous to working memory in cognitive science — a scratch pad that gets
                updated with each new piece of information.""",
                "formula": r"h_t = \tanh(W_h h_{t-1} + W_x x_t + b)",
                "formula_note": "hₜ ∈ ℝᵈ is the memory at step t; same W_h, W_x used every step",
            },
            {
                "term": "Vanishing Gradient",
                "category": "🔁 Sequence Models",
                "emoji": "📉",
                "short": "When gradients become exponentially small as they propagate backward through many timesteps — preventing learning of long-range dependencies.",
                "definition": r"""In RNNs, gradients must pass through the Jacobian $\partial h_t / \partial h_{t-1}$ at each step.
                If the spectral norm of this Jacobian is < 1 (as occurs with saturated tanh), gradients shrink
                by a multiplicative factor at each step: after $T$ steps, the gradient is approximately $\lambda^T$ times the
                original ($\lambda < 1$). For $T = 50$ and $\lambda = 0.9$: $0.9^{50} \approx 0.005$.
                The network effectively "forgets" what happened more than a few steps ago.
                LSTM's cell state provides a direct gradient highway that can maintain gradient magnitude
                across hundreds of steps, solving this problem.""",
                "formula": r"\frac{\partial L}{\partial h_0} = \prod_{t=1}^T \frac{\partial h_t}{\partial h_{t-1}} \approx \lambda^T \to 0",
                "formula_note": "λ < 1 ⟹ exponential decay over T timesteps",
            },
            {
                "term": "LSTM Gate",
                "category": "🔁 Sequence Models",
                "emoji": "🚪",
                "short": "A sigmoid-activated controller inside LSTM that decides how much information to pass, block, or write — forget, input, and output gates.",
                "definition": r"""LSTM gates are vectors in $(0,1)$ produced by the sigmoid function applied to learned linear combinations
                of the input and previous hidden state. A gate value close to 1 means "pass all information"; close to 0 means
                "block everything." Three gates coordinate the LSTM's memory: the forget gate $f_t$ controls how much of the old
                cell state to erase; the input gate $i_t$ controls how much new information to write; the output gate $o_t$ controls
                how much of the cell state to expose as the hidden state. This precise, learnable control over memory is what
                allows LSTMs to maintain information across hundreds of timesteps without gradient issues.""",
                "formula": r"f_t = \sigma(W_f[h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)",
                "formula_note": "[h, x] = concatenation; σ = sigmoid ∈ (0,1)",
            },
            {
                "term": "Cell State",
                "category": "🔁 Sequence Models",
                "emoji": "🔋",
                "short": "LSTM's long-term memory vector — updated additively each step, allowing information to flow unchanged across many timesteps.",
                "definition": r"""The cell state $C_t$ is the key innovation of LSTM. Unlike the hidden state, which passes through
                tanh and sigmoid activations at every step, the cell state is updated additively:
                $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$. The forget gate can set $f_t = 1$ (keep all) and
                the input gate $i_t = 0$ (add nothing), allowing $C_t = C_{t-1}$ — information travels unchanged for
                arbitrarily many steps. This additive update means gradients can flow backward through the cell state
                without being repeatedly multiplied by small numbers — solving the vanishing gradient problem.
                Think of it as a conveyor belt that carries information forward with selective loading and unloading.""",
                "formula": r"C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t",
                "formula_note": "⊙ = elementwise multiply; additive update preserves gradient flow",
            },

            # ── Reinforcement Learning ─────────────────────────────────────
            {
                "term": "Policy",
                "category": "⚡ Reinforcement Learning",
                "emoji": "🗺️",
                "short": "The agent's decision rule — a function that maps states to actions (deterministic) or distributions over actions (stochastic).",
                "definition": r"""A policy $\pi$ defines the agent's complete behaviour. A deterministic policy $\pi(s) = a$ gives
                a single action for each state. A stochastic policy $\pi(a|s)$ gives a probability distribution — useful for
                exploration (try different actions) and mixed strategies in multi-agent games.
                In deep RL, the policy is parameterised by a neural network: $\pi_\theta(a|s)$.
                Policy gradient methods (REINFORCE, PPO, A3C) directly optimise the policy network.
                Q-Learning and DQN maintain a value function and derive the policy implicitly (greedy w.r.t. Q).""",
                "formula": r"\pi^* = \arg\max_\pi \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t R_{t+1}\right]",
                "formula_note": "Optimal policy maximises expected discounted cumulative reward",
            },
            {
                "term": "Return G",
                "category": "⚡ Reinforcement Learning",
                "emoji": "💰",
                "short": "The total discounted reward from a timestep to the end of the episode — what the agent ultimately wants to maximise.",
                "definition": r"""The return $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots$ is the sum of all future
                rewards, discounted by $\gamma \in [0,1]$. Discounting reflects that immediate rewards are worth more than
                distant ones (like interest rates or uncertainty about the future). $\gamma = 0$: agent only cares about
                immediate reward. $\gamma = 1$: all future rewards count equally (episodic tasks only). MC methods use
                the actual observed return $G_t$ — unbiased but only available after the episode ends.
                TD methods estimate $G_t$ using $R_{t+1} + \gamma V(S_{t+1})$ — biased but available immediately.""",
                "formula": r"G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}",
                "formula_note": "Recursive form: G_t = R_{t+1} + γ·G_{t+1}",
            },
            {
                "term": "TD Error",
                "category": "⚡ Reinforcement Learning",
                "emoji": "⚡",
                "short": "The difference between the TD target (reward + discounted next value) and the current value estimate — the RL learning signal.",
                "definition": r"""The TD error $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ measures how surprised the
                agent was by the outcome. Positive $\delta$: the outcome was better than expected → increase $V(S_t)$.
                Negative $\delta$: worse than expected → decrease $V(S_t)$. This is directly analogous to the prediction
                error signal in human dopamine neuroscience — a landmark finding linking RL theory to neurobiology.
                In actor-critic methods, the TD error is called the **advantage** and directly scales the policy gradient.
                Eligibility traces in SARSA(λ) use $\delta_t$ to update all recently-visited states, not just the current one.""",
                "formula": r"\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)",
                "formula_note": "δ > 0: better than expected; δ < 0: worse than expected",
            },
            {
                "term": "Q-function",
                "category": "⚡ Reinforcement Learning",
                "emoji": "❓",
                "short": "Q(s,a) = expected return when taking action a in state s and following a policy thereafter — enables greedy action selection without a model.",
                "definition": r"""The action-value function $Q_\pi(s,a) = \mathbb{E}_\pi[G_t | S_t=s, A_t=a]$ is the expected
                cumulative return from taking a specific action once then following policy $\pi$ forever after.
                Unlike V(s) (state value), Q(s,a) tells us which action is best — no environment model needed.
                The greedy policy $\pi(s) = \arg\max_a Q(s,a)$ is directly extractable from Q.
                Q-Learning estimates Q* (the optimal Q-function) using the Bellman optimality update:
                $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma\max_{a'} Q(s',a') - Q(s,a)]$.
                DQN approximates $Q(s,\cdot)$ with a neural network, enabling Q-learning in high-dimensional state spaces.""",
                "formula": r"Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1},a') | S_t=s, A_t=a]",
                "formula_note": "Bellman optimality equation for Q*",
            },

            # ── PyTorch & Implementation ───────────────────────────────────
            {
                "term": "Autograd",
                "category": "🔧 PyTorch & Implementation",
                "emoji": "🤖",
                "short": "PyTorch's automatic differentiation engine — records operations during forward pass and computes gradients backward through the computation graph.",
                "definition": r"""Autograd builds a dynamic computation graph during the forward pass: every tensor operation
                on a `requires_grad=True` tensor is recorded as a node with its local gradient function.
                `loss.backward()` traverses this graph in reverse, calling each stored gradient function and accumulating
                results in `.grad` attributes. Dynamic graphs (as opposed to TensorFlow 1.x's static graphs) mean the graph
                is rebuilt every forward pass, allowing Python control flow (if/else, loops) inside the model.
                `torch.no_grad()` context manager disables graph recording — used during evaluation for 2–5× speedup.
                `detach()` creates a new tensor that shares data but doesn't participate in gradient computation.""",
                "formula": r"\text{loss.backward()} \;\Rightarrow\; \theta.\text{grad} = \frac{\partial L}{\partial \theta}",
                "formula_note": "No manual gradient derivation needed — autograd handles all chain rule applications",
            },
            {
                "term": "nn.Module",
                "category": "🔧 PyTorch & Implementation",
                "emoji": "🧩",
                "short": "The base class for all PyTorch models — organises parameters, enables GPU transfer, and provides training/eval mode switching.",
                "definition": r"""`nn.Module` is the foundation of every PyTorch model. Subclassing it and defining `__init__`
                (declare layers) and `forward` (define computation) gives access to: automatic parameter tracking via
                `model.parameters()`; GPU/CPU transfer via `model.to(device)`; training mode (dropout active, BatchNorm
                uses batch stats) via `model.train()`; evaluation mode (dropout off, BatchNorm uses running stats) via
                `model.eval()`; state saving and loading via `model.state_dict()` / `load_state_dict()`.
                Modules can be composed: `nn.Sequential` stacks layers sequentially; custom modules can contain sub-modules.""",
                "formula": r"\hat{y} = f_\theta(x) \quad \text{where } \theta = \text{model.parameters()}",
                "formula_note": "Define layers in __init__, use them in forward — PyTorch handles the rest",
            },
            {
                "term": "DataLoader",
                "category": "🔧 PyTorch & Implementation",
                "emoji": "📂",
                "short": "PyTorch utility that wraps a Dataset and provides shuffled mini-batches with optional multiprocessing and GPU pinning.",
                "definition": r"""`DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)` is the standard interface
                for batched training data in PyTorch. It randomly samples indices each epoch (if `shuffle=True`),
                collates individual samples into batch tensors, and optionally loads data in parallel background processes
                (`num_workers > 0`) to prevent the GPU from waiting on CPU data loading.
                `pin_memory=True` speeds up CPU→GPU transfers by allocating data in page-locked memory.
                Custom datasets require implementing `__len__()` and `__getitem__(idx)`.
                For RL: experience replay buffers serve the same role as DataLoader — batching random transitions.""",
                "formula": r"(X_{\mathcal{B}}, y_{\mathcal{B}}) \sim \mathcal{D}, \quad X_{\mathcal{B}} \in \mathbb{R}^{B \times d}",
                "formula_note": "Each iteration: X_batch.shape = (32, features), y_batch.shape = (32,)",
            },
        ]

        # ── Filter logic ──────────────────────────────────────────────────
        def matches(entry):
            if search_term and search_term.lower() not in entry["term"].lower() and \
               search_term.lower() not in entry["definition"].lower() and \
               search_term.lower() not in entry["short"].lower():
                return False
            if category_filter != "All categories" and entry["category"] != category_filter:
                return False
            return True

        filtered = [e for e in glossary if matches(e)]

        st.markdown(f"""
        <div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;
                    padding:.6rem 1.2rem;margin:.5rem 0;font-size:.92rem;color:#9e9ebb">
        Showing <b style="color:white">{len(filtered)}</b> of {len(glossary)} terms
        {f'— filtered by: <b style="color:#00bcd4">{category_filter}</b>' if category_filter != "All categories" else ""}
        {f'— search: <b style="color:#ffa726">{search_term}</b>' if search_term else ""}
        </div>
        """, unsafe_allow_html=True)

        if not filtered:
            st.info("No terms match your search. Try a different keyword or clear the search.")
        else:
            # ── Category headers & entries ────────────────────────────────
            current_cat = None
            for entry in filtered:
                # Category divider
                if entry["category"] != current_cat:
                    current_cat = entry["category"]
                    st.markdown(f"""
                    <div style="background:#1a1a2e;border-radius:8px;padding:.5rem 1rem;
                                margin:1rem 0 .5rem;border-left:3px solid #00bcd4">
                        <b style="color:#00bcd4;font-size:1rem">{current_cat}</b>
                    </div>""", unsafe_allow_html=True)

                with st.expander(f"{entry['emoji']} **{entry['term']}** — {entry['short']}", expanded=False):
                    col_def, col_formula = st.columns([3, 2])
                    with col_def:
                        # Use st.markdown so LaTeX $...$ renders correctly
                        st.markdown(entry['definition'])
                    with col_formula:
                        st.markdown("""
                        <div style="background:#1a1a30;border:1px solid #3a3a5e;border-radius:8px;
                                    padding:.6rem 1rem;margin-bottom:.4rem">
                        <span style="color:#9c9cf0;font-size:.8rem;font-weight:700">
                        📐 Key Formula</span></div>""", unsafe_allow_html=True)
                        try:
                            st.latex(entry["formula"])
                        except Exception:
                            st.code(entry["formula"], language="latex")
                        st.caption(f"📝 {entry['formula_note']}")
                        st.markdown(f"""
                        <div style="background:#12121f;border-radius:6px;padding:.4rem .8rem;
                                    font-size:.79rem;color:#546e7a;margin-top:.3rem">
                        <b>Category:</b> {entry['category']}
                        </div>""", unsafe_allow_html=True)
