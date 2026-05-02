"""
_foundations_mod.py — Mathematical & Programming Foundations (Complete Edition)
10 tabs: Why | Linear Algebra | Calculus | Optimisation | Probability
         Statistics | Information Theory | Python & NumPy | Neural Net Math | Self-Assessment
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
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    return fig, axes

def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
            f'<b>{icon} {title}</b><br>'
            f'<span style="color:#b0b0cc;font-size:.93rem;line-height:1.7">{body}</span></div>')

def _proof(title, body):
    return (f'<div style="background:#0a1520;border:1px solid #1a3555;border-radius:10px;'
            f'padding:1rem 1.3rem;margin:.8rem 0">'
            f'<b style="color:#42a5f5">Proof: {title}</b><br>'
            f'<span style="color:#b0b0cc;font-size:.9rem;line-height:1.8">{body}</span></div>')

def _insight(text):
    return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
            f'padding:.8rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;'
            f'font-size:.93rem;color:#b0b0cc;line-height:1.7">💡 {text}</div>')

def _sec(emoji, title, sub, color="#00695c"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)



def main_foundations():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0a2a0a,#0a1a2e,#1a0a2e);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">📐 Mathematical &amp; Programming Foundations</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'Stage -1: Six mathematical areas and two programming skills before Deep Learning. '
        'Every concept derived from scratch, directly mapped to RL formulas, with interactive charts.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🗺️ Why & What",
        "🔢 Linear Algebra",
        "📈 Calculus",
        "🎯 Optimisation",
        "🎲 Probability",
        "📊 Statistics",
        "📡 Information Theory",
        "🐍 Python & NumPy",
        "🧠 Neural Network Math",
        "✅ Self-Assessment",
    ])
    (tab_why, tab_la, tab_calc, tab_opt, tab_prob,
     tab_stat, tab_info, tab_py, tab_nn, tab_test) = tabs

    # ── WHY ───────────────────────────────────────────────────────────────
    with tab_why:
        _sec("🗺️", "Why These Six Areas?",
             "Every RL formula maps to one of these mathematical foundations", "#00695c")
        st.markdown(_card("#00695c", "🎯", "Direct map from math to RL algorithms",
            """<b>Linear algebra:</b> neural network = matrix multiplications.
            Q(s,a) = W₂·relu(W₁·s+b₁)+b₂ is entirely linear algebra.<br>
            <b>Calculus:</b> policy gradient ∇J(θ) = vector of partial derivatives;
            backpropagation = chain rule applied recursively through every layer.<br>
            <b>Optimisation:</b> gradient ascent θ←θ+α∇J; Adam/RMSProp update rules;
            PPO trust region; TRPO KL constraint; CMA-ES in World Models.<br>
            <b>Probability:</b> π(a|s) IS a probability distribution;
            V(s) = E[G_t|s_t=s] is an expectation; PSRL uses Bayesian posteriors.<br>
            <b>Statistics:</b> MC gradient estimation; 5-seed evaluation protocols;
            bias-variance tradeoff in n-step returns and GAE.<br>
            <b>Information theory:</b> H(π) entropy in SAC; D_KL(π||π_old) in PPO/TRPO;
            cross-entropy = −logπ(a|s) policy loss; RLHF reward model loss.<br>
            You need working fluency — ability to compute, implement, and intuit each concept."""),
            unsafe_allow_html=True)

        st.dataframe(pd.DataFrame({
            "Foundation": ["Linear Algebra", "Calculus", "Optimisation", "Probability",
                           "Statistics", "Information Theory", "Python & NumPy", "Neural Net Math"],
            "Key RL formulas": [
                "Q(s,a)=W₂relu(W₁s+b₁)+b₂; Fisher matrix F in TRPO; SVD for PCA",
                "∇_θJ(θ): policy gradient; chain rule in backprop; Bellman contraction",
                "θ←θ+α∇J; Adam; PPO clip; TRPO trust region; CMA-ES",
                "π(a|s) = P(a|s); V(s) = E[G_t|s_t=s]; PSRL Bayesian posterior",
                "MC gradient variance 1/N; GAE bias-variance; 5-seed protocols",
                "H(π) in SAC/A2C; D_KL in PPO/TRPO; −logπ(a|s) policy loss",
                "All portal simulations; PyTorch≈NumPy+autograd; env batching",
                "Forward pass; backprop; He/Xavier init; loss functions; activations",
            ],
            "Time needed": ["2 wks", "2 wks", "1 wk", "2 wks", "1 wk", "3 days", "1 wk", "1 wk"],
            "Best resource": [
                "3Blue1Brown LA (YouTube)", "3Blue1Brown Calculus (YouTube)",
                "Goodfellow DL Ch.4 (free)", "Brownlee Prob for ML",
                "Khan Academy Statistics", "Cover & Thomas Ch.1-2",
                "numpy.org quickstart", "Karpathy micrograd (GitHub)"],
        }), use_container_width=True, hide_index=True)

    # ── LINEAR ALGEBRA ────────────────────────────────────────────────────
    with tab_la:
        _sec("🔢", "Linear Algebra — The Language of Neural Networks",
             "Vectors, matrices, dot products, eigenvalues — inside every neural network and RL formula", "#6a1b9a")

        st.markdown(_card("#6a1b9a", "🔢", "Why this is prerequisite #1",
            """A neural network layer is: h = relu(W @ x + b). Every symbol here is linear algebra.
            W is a matrix (n_out×n_in), x is a vector (n_in,), @ is matrix-vector multiplication,
            relu is element-wise, b is a bias vector. If you cannot compute W@x by hand, you cannot
            understand what the network does. The policy gradient ∇_θJ lives in parameter space
            (it is a vector). The Fisher information matrix F = E[∇logπ ∇logπ^T] appears in TRPO.
            Eigenvalues determine gradient descent convergence speed. SVD underlies PCA."""), unsafe_allow_html=True)

        st.subheader("1. Vectors — States, Actions, Parameters")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CartPole state vector:**")
            st.latex(r"s = [x,\;\dot{x},\;\theta,\;\dot{\theta}]^\top \in \mathbb{R}^4")
            st.markdown("**Dot product (the most-used operation):**")
            st.latex(r"\mathbf{u}\cdot\mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta_{uv}")
            st.markdown("Linear value function: Q(s) = w^T φ(s) — a dot product of weights and features.")
            st.markdown("**Vector norm (gradient clipping uses this):**")
            st.latex(r"\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}")
        with col2:
            u = np.array([2.0, 1.0]); v = np.array([1.0, 2.0])
            fig_dot, ax_dot = _fig(1, 1, 5.5, 4)
            ax_dot.quiver(0, 0, u[0], u[1], angles="xy", scale_units="xy", scale=1,
                          color="#7c4dff", width=0.04, label=f"u={u}")
            ax_dot.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                          color="#ffa726", width=0.04, label=f"v={v}")
            proj = np.dot(u, v) / np.dot(v, v) * v
            ax_dot.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1,
                          color="#4caf50", width=0.02, alpha=0.7,
                          label=f"projection, u·v={int(np.dot(u,v))}")
            ax_dot.set_xlim(-0.5, 3.5); ax_dot.set_ylim(-0.5, 3.0)
            ax_dot.set_title(f"Dot product = {int(np.dot(u,v))}", color="white", fontweight="bold")
            ax_dot.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_dot.grid(alpha=0.15); ax_dot.axhline(0, color="#2a2a3e"); ax_dot.axvline(0, color="#2a2a3e")
            plt.tight_layout(); st.pyplot(fig_dot); plt.close()

        st.subheader("2. Matrices — Every Neural Network Layer Is One")
        st.markdown("**Matrix-vector multiply = the core operation of every layer:**")
        st.latex(r"\mathbf{h} = W\mathbf{x} + \mathbf{b} \quad W\in\mathbb{R}^{m\times n},\;\mathbf{x}\in\mathbb{R}^n \;\Rightarrow\; \mathbf{h}\in\mathbb{R}^m")
        st.markdown("**Backpropagation uses transpose:** if h = W @ x, then dL/dx = W^T @ dL/dh")
        st.markdown("Each row of W is a feature detector — fires when input aligns with it.")

        c1, c2 = st.columns(2)
        a11 = c1.slider("W[0,0]", -3., 3., 2., 0.1, key="la_s11")
        a12 = c1.slider("W[0,1]", -3., 3., 1., 0.1, key="la_s12")
        a21 = c2.slider("W[1,0]", -3., 3., 0., 0.1, key="la_s21")
        a22 = c2.slider("W[1,1]", -3., 3., 1.5, 0.1, key="la_s22")
        Wm = np.array([[a11, a12], [a21, a22]])
        sq = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
        tr = Wm @ sq
        fig_m, ax_m = _fig(1, 1, 9, 4)
        ax_m.plot(sq[0], sq[1], color="#546e7a", lw=2, ls="--", label="Original unit square")
        ax_m.fill(sq[0], sq[1], alpha=0.15, color="#546e7a")
        ax_m.plot(tr[0], tr[1], color="#7c4dff", lw=2.5, label="After W multiplication")
        ax_m.fill(tr[0], tr[1], alpha=0.2, color="#7c4dff")
        ax_m.set_xlim(-4, 4); ax_m.set_ylim(-4, 4)
        ax_m.axhline(0, color="#2a2a3e"); ax_m.axvline(0, color="#2a2a3e")
        det = np.linalg.det(Wm)
        eigs = np.linalg.eigvals(Wm).real
        ax_m.set_title(f"det(W)={det:.2f} | eigenvalues ≈ {eigs.round(2)}", color="white", fontweight="bold")
        ax_m.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_m.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_m); plt.close()
        if abs(det) > 1.5:
            st.markdown("⚠️ **|det|>1:** space expands — gradient explosion risk in deep networks.")
        elif abs(det) < 0.3:
            st.markdown("⚠️ **|det|<1:** space contracts — gradient vanishing risk.")
        else:
            st.markdown("✅ **det ≈ 1:** volume roughly preserved — healthy for training.")

        st.subheader("3. The Jacobian — Derivative of a Vector w.r.t. Another")
        st.markdown("When input and output are both vectors, the derivative is a matrix:")
        st.latex(r"J_{ij} = \frac{\partial f_i}{\partial x_j} \quad\Rightarrow\quad J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} \in \mathbb{R}^{m\times n}")
        st.markdown("For linear layer h=Wx: Jacobian = W. For ReLU: diagonal matrix with 1[z>0]. Backprop computes J^T @ upstream_gradient efficiently without forming J.")

        st.subheader("4. Implementation")
        st.code("""
import numpy as np
s  = np.array([0.02, -0.01, 0.04, 0.03])     # CartPole state, shape (4,)
W1 = np.random.randn(32, 4) * np.sqrt(2/4)   # He init, shape (32, 4)
b1 = np.zeros(32)
h1 = W1 @ s + b1    # Matrix-vector multiply: (32,4)@(4,) = (32,) — one layer
h1 = np.maximum(0, h1)  # ReLU activation element-wise
# Backprop uses transpose:
d_h1 = np.random.randn(32)  # upstream gradient from next layer
d_s  = W1.T @ d_h1   # shape (4,) — gradient to previous layer
d_W1 = np.outer(d_h1, s)   # shape (32,4) — gradient w.r.t. weights
""", language="python")

    # ── CALCULUS ─────────────────────────────────────────────────────────
    with tab_calc:
        _sec("📈", "Calculus — Derivatives Power Every RL Update",
             "Gradients, chain rule, backpropagation — the mathematics of all policy learning", "#e65100")

        st.markdown(_card("#e65100", "📈", "Why calculus is unavoidable in RL",
            """Every RL parameter update is: θ ← θ + α∇J(θ). The gradient ∇J is a vector of partial
            derivatives — how does total reward change if we nudge each weight by a tiny amount?
            For a million-parameter network, computing this efficiently requires the chain rule
            applied backward through the computation graph — this IS backpropagation.
            The Bellman operator is a contraction (a calculus result proving TD convergence).
            GAE is a geometric series (calculus/series). PPO convergence uses Lipschitz gradient bounds.
            Understanding derivatives makes all of this intuitive rather than magical."""), unsafe_allow_html=True)

        st.subheader("1. Derivatives — From the Limit Definition")
        st.markdown("**The fundamental definition** (origin of all derivatives):")
        st.latex(r"f'(x) = \frac{df}{dx} = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}")
        st.markdown("**Every derivative you use in RL daily:**")
        col1, col2 = st.columns(2)
        with col1:
            st.latex(r"\frac{d}{dx}x^n = nx^{n-1}")
            st.latex(r"\frac{d}{dx}e^x = e^x")
            st.latex(r"\frac{d}{dx}\ln x = \frac{1}{x}")
            st.latex(r"\frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x))")
        with col2:
            st.latex(r"\frac{d}{dx}\max(0,x) = \mathbf{1}[x>0] \quad\text{(ReLU)}")
            st.latex(r"\frac{d}{dx}\tanh(x) = 1-\tanh^2(x)")
            st.latex(r"\frac{d}{dx}\|x\|^2 = 2x")
            st.latex(r"\frac{d}{dx}\log p(x) = \frac{p'(x)}{p(x)} \quad\text{(score fn.)}")

        fn_choice = st.selectbox("Visualise function and its derivative",
                                  ["x²", "relu", "sigmoid", "tanh", "ln(x)"], key="calc_fn")
        xr = np.linspace(-3, 3, 300)
        fn_map = {
            "x²":     (xr**2, 2*xr, "2x"),
            "relu":   (np.maximum(0, xr), (xr > 0).astype(float), "1[x>0]"),
            "sigmoid":(1/(1+np.exp(-xr)),
                       (1/(1+np.exp(-xr)))*(1-1/(1+np.exp(-xr))), "σ(1−σ)"),
            "tanh":   (np.tanh(xr), 1-np.tanh(xr)**2, "1−tanh²"),
            "ln(x)":  (np.where(xr>0.01, np.log(np.maximum(xr,0.01)), np.nan),
                       np.where(xr>0.01, 1/np.maximum(xr,0.01), np.nan), "1/x"),
        }
        fy, dfy, dn = fn_map[fn_choice]
        fig_c, ax_c = _fig(1, 1, 10, 4)
        ax_c.plot(xr, fy, color="#7c4dff", lw=2.5, label=f"f(x) = {fn_choice}")
        ax_c.plot(xr, dfy, color="#ffa726", lw=2, ls="--", label=f"f'(x) = {dn}")
        ax_c.axhline(0, color="#2a2a3e"); ax_c.axvline(0, color="#2a2a3e")
        ax_c.set_ylim(-2.5, 2.5); ax_c.set_xlabel("x", color="white"); ax_c.set_ylabel("y", color="white")
        ax_c.set_title(f"{fn_choice} and its derivative", color="white", fontweight="bold")
        ax_c.legend(facecolor=CARD, labelcolor="white", fontsize=9); ax_c.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_c); plt.close()

        st.subheader("2. Partial Derivatives and Gradients")
        st.markdown("For a function of many parameters θ = [θ₁, θ₂, ..., θₙ], the **gradient** collects all partial derivatives:")
        st.latex(r"\nabla_\theta\mathcal{L} = \left[\frac{\partial\mathcal{L}}{\partial\theta_1},\;\frac{\partial\mathcal{L}}{\partial\theta_2},\;\ldots,\;\frac{\partial\mathcal{L}}{\partial\theta_n}\right]^\top")
        st.markdown("The gradient points toward the **steepest increase** of L.")
        st.markdown("**Gradient descent** (minimise loss): θ ← θ − α∇L | **Gradient ascent** (maximise reward J): θ ← θ + α∇J")

        x_g, y_g = np.meshgrid(np.linspace(-3,3,25), np.linspace(-3,3,25))
        z_g = x_g**2 + 2*y_g**2
        fig_g, ax_g = _fig(1, 1, 10, 4.5)
        ax_g.contourf(x_g, y_g, z_g, levels=20, cmap="RdYlGn_r", alpha=0.7)
        sk = 2; gx, gy = 2*x_g, 4*y_g
        ax_g.quiver(x_g[::sk,::sk], y_g[::sk,::sk],
                    -gx[::sk,::sk]/5, -gy[::sk,::sk]/5,
                    color="white", alpha=0.7, scale=20, width=0.004)
        ax_g.set_xlabel("θ₁", color="white"); ax_g.set_ylabel("θ₂", color="white")
        ax_g.set_title("Gradient descent on L = θ₁² + 2θ₂² — arrows show descent direction",
                       color="white", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_g); plt.close()

        st.subheader("3. Chain Rule — The Heart of Backpropagation")
        st.markdown("For composed functions f(g(x)): the derivative is the product of derivatives at each step.")
        st.latex(r"\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{dh}\cdot\frac{dh}{dx} \quad\text{(chain rule — applied recursively through all layers)}")
        st.markdown(_proof("Backprop through one ReLU layer: h = relu(Wx + b)",
            """Given dL/dh from the layer above (upstream gradient):<br>
            Step 1 — Through ReLU: dL/dz = dL/dh * (z > 0)  [element-wise: zero for negatives]<br>
            Step 2 — Through bias: dL/db = dL/dz  [identity: ∂z/∂b = 1]<br>
            Step 3 — Through W: dL/dW = outer(x, dL/dz)  [∂(Wx)/∂W: outer product]<br>
            Step 4 — Through x: dL/dx = W.T @ dL/dz  [∂(Wx)/∂x = W^T]<br><br>
            This is the COMPLETE backprop algorithm for one linear+ReLU layer.
            Stack it recursively for n layers and you have full backpropagation."""), unsafe_allow_html=True)

        st.code("""
# Manual backprop — what PyTorch autograd does internally
z1 = W1 @ s + b1; h1 = np.maximum(0, z1)       # forward
z2 = W2 @ h1 + b2; e = np.exp(z2-z2.max()); p = e/e.sum()  # softmax

# Backward (chain rule right to left):
d_z2 = p.copy(); d_z2[action] -= 1; d_z2 *= -advantage   # CE grad × return
d_W2 = np.outer(h1, d_z2); d_b2 = d_z2                   # through W2
d_h1 = W2.T @ d_z2; d_z1 = d_h1 * (z1 > 0)              # through ReLU
d_W1 = np.outer(s, d_z1); d_b1 = d_z1                     # through W1
""", language="python")

    # ── OPTIMISATION ──────────────────────────────────────────────────────
    with tab_opt:
        _sec("🎯", "Optimisation — Finding the Best Policy Parameters",
             "Gradient descent, Adam, learning rates, convergence — all RL updates are optimisation", "#f57f17")

        st.markdown(_card("#f57f17", "🎯", "Why optimisation theory matters",
            """RL training IS optimisation. Actor update = gradient ascent on J(θ). Critic update =
            gradient descent on TD loss. PPO = constrained optimisation (clip enforces trust region).
            TRPO = KL-constrained trust region with natural gradient.
            Understanding why gradient descent works, what makes it fail (saddle points, high curvature),
            and how Adam/RMSProp address these is essential for debugging RL training failures.
            The learning rate is the single most impactful hyperparameter in RL — too small = slow,
            too large = diverges, optimal = the sweet spot that changes during training."""), unsafe_allow_html=True)

        st.subheader("1. Gradient Descent — The Fundamental Algorithm")
        st.latex(r"\theta_{t+1} = \theta_t - \alpha\nabla_\theta\mathcal{L}(\theta_t)")
        st.markdown("**Convergence guarantee** (convex, L-smooth functions): step size α ≤ 1/L ensures each step decreases loss:")
        st.latex(r"\mathcal{L}(\theta_{t+1}) \leq \mathcal{L}(\theta_t) - \frac{\alpha}{2}\|\nabla\mathcal{L}(\theta_t)\|^2")

        def gd_path(lr, n=50):
            x, y = 3.0, 2.5; path = [(x, y)]
            for _ in range(n): x -= lr*2*x; y -= lr*10*y; path.append((x, y))
            return np.array(path)
        xm, ym = np.meshgrid(np.linspace(-4,4,60), np.linspace(-3,3,60))
        zm = xm**2 + 5*ym**2
        fig_o, axes_o = _fig(1, 3, 15, 4)
        configs = [(0.05, "lr=0.05 — too small, slow"), (0.18, "lr=0.18 — optimal"), (0.22, "lr=0.22 — oscillates")]
        for ax, (lr, title) in zip(axes_o, configs):
            p = gd_path(lr)
            ax.contourf(xm, ym, zm, levels=15, cmap="RdYlGn_r", alpha=0.7)
            ax.plot(p[:,0], p[:,1], "w-", lw=1.5, alpha=0.8)
            ax.plot(p[:,0], p[:,1], "wo", ms=2, alpha=0.7)
            ax.plot(p[0,0], p[0,1], "rs", ms=8); ax.plot(0, 0, "g*", ms=10)
            ax.set_title(title, color="white", fontsize=9, fontweight="bold")
            ax.set_xlim(-4, 4); ax.set_ylim(-3, 3)
        plt.tight_layout(); st.pyplot(fig_o); plt.close()

        st.subheader("2. Adam — The Default Optimiser for RL")
        st.markdown("Adam combines momentum (gradient history) with adaptive per-parameter learning rates:")
        st.latex(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t \quad\text{(1st moment: gradient mean)}")
        st.latex(r"v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2 \quad\text{(2nd moment: gradient variance)}")
        st.latex(r"\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t} \quad\text{(bias correction for early steps)}")
        st.latex(r"\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat v_t}+\varepsilon}\hat m_t")
        st.markdown("Parameters with large gradients → smaller effective lr. Parameters with small gradients → larger effective lr. Default: β₁=0.9, β₂=0.999, ε=1e-8, lr=3e-4.")

        st.dataframe(pd.DataFrame({
            "Optimiser": ["SGD", "Momentum", "RMSProp", "Adam", "AdamW"],
            "Update rule": ["θ -= α·g", "v=βv-αg; θ+=v", "s=βs+(1-β)g²; θ-=α·g/√s",
                            "θ -= α·m̂/√v̂", "Adam + weight decay λ·θ"],
            "Key advantage": ["Simple", "Builds velocity in consistent directions",
                              "Adapts lr per parameter", "Momentum + adaptive lr (robust)",
                              "Better generalisation for LLMs"],
            "Used in RL for": ["Rarely", "Classic policy gradient", "A3C (original paper)",
                               "DQN, PPO, SAC, DDPG, TD3 (default)", "LLM RLHF fine-tuning"],
        }), use_container_width=True, hide_index=True)

    # ── PROBABILITY ───────────────────────────────────────────────────────
    with tab_prob:
        _sec("🎲", "Probability — The Language of Stochastic Policies",
             "Distributions, expectations, Bayes — RL is fundamentally probabilistic at every level", "#7c4dff")

        st.markdown(_card("#7c4dff", "🎲", "Why probability is unavoidable in RL",
            """The policy π(a|s) IS a probability distribution — for each state, it assigns a probability
            to each possible action. The value function V(s) = E[G_t | s_t=s] is an expectation over
            future returns. The policy gradient theorem derives E_τ[Σ∇logπ·r] — an expectation over
            trajectories. Monte Carlo methods estimate expectations by sampling. PSRL maintains a
            Bayesian posterior over MDP dynamics. Thompson sampling draws from a posterior distribution.
            The proof that the baseline doesn't bias the gradient requires E[∇logπ] = 0.
            You cannot read a single RL equation without encountering probability."""), unsafe_allow_html=True)

        st.subheader("1. Kolmogorov Axioms — Foundation of All Probability")
        st.latex(r"P(A) \geq 0 \quad \forall A \qquad P(\Omega)=1 \qquad P(A\cup B)=P(A)+P(B)\;\text{if }A\cap B=\emptyset")
        st.markdown("**Conditional probability** — the most important concept for RL:")
        st.latex(r"P(A|B) = \frac{P(A\cap B)}{P(B)} \quad\Rightarrow\quad \pi(a|s) = P(\text{action}=a\,|\,\text{state}=s)")
        st.markdown("The policy is a conditional probability distribution: given the state, what is the probability of each action?")

        st.subheader("2. Expectation and Variance")
        st.latex(r"\mathbb{E}[X] = \sum_x x\,P(X=x) \quad\text{(discrete)} \qquad \mathbb{E}[aX+bY] = a\mathbb{E}[X]+b\mathbb{E}[Y]\;\text{(linearity)}")
        st.latex(r"\text{Var}[X] = \mathbb{E}[(X-\mathbb{E}[X])^2] = \mathbb{E}[X^2]-(\mathbb{E}[X])^2")
        st.markdown("**In RL:** J(θ) = E_τ[Σr_t] is an expectation. Monte Carlo: run N episodes, average total rewards. Variance of returns → variance of gradient estimate → affects convergence speed.")

        st.subheader("3. Key Distributions in RL")
        st.dataframe(pd.DataFrame({
            "Distribution": ["Categorical", "Gaussian N(μ,σ²)", "Beta(α,β)", "Dirichlet(α)"],
            "Formula": ["P(X=k)=p_k, Σp_k=1", "p(x)∝exp(-(x-μ)²/2σ²)",
                        "p(x)∝x^(α-1)(1-x)^(β-1)", "p(p)∝Πp_k^(αk-1)"],
            "RL application": [
                "Discrete action policy — softmax(logits)",
                "Continuous action policy in PPO/SAC: a ~ N(μ(s), σ²(s))",
                "Thompson sampling prior for Bernoulli bandit arms",
                "PSRL posterior over categorical transition probabilities",
            ],
            "Key trick": [
                "Differentiable via Gumbel-softmax reparameterisation",
                "Reparameterisation: a = μ + σε, ε~N(0,1) — gradient flows through",
                "Conjugate to Binomial: update α+=r, β+=1-r after each step",
                "Conjugate to Categorical: α_k += N(s,a,s') counts",
            ],
        }), use_container_width=True, hide_index=True)

        st.subheader("4. Bayes' Theorem — Used in PSRL and Bayesian RL")
        st.latex(r"P(\theta|\mathcal{D}) = \frac{P(\mathcal{D}|\theta)\cdot P(\theta)}{P(\mathcal{D})} \propto P(\mathcal{D}|\theta)\cdot P(\theta)")
        st.markdown("PSRL: maintain a posterior P(MDP | experience). Each episode, **sample one MDP** from the posterior and act optimally in it. This is Thompson Sampling extended to full MDPs — it automatically balances exploration and exploitation.")

    # ── STATISTICS ────────────────────────────────────────────────────────
    with tab_stat:
        _sec("📊", "Statistics — Evaluating and Comparing RL Algorithms",
             "Variance, bias-variance, confidence intervals — critical for reproducible RL research", "#0288d1")

        st.markdown(_card("#0288d1", "📊", "Why statistics is essential for RL practitioners",
            """Henderson et al. (2018) showed PPO on HalfCheetah varied from 2000 to 6000 reward
            across 5 seeds — a 3× range. Without statistics, you cannot tell if algorithm A beats B
            or just got luckier random seeds. Variance of the gradient estimator directly determines
            how many environment steps you need to converge. The bias-variance tradeoff is the
            fundamental lens for understanding n-step returns, GAE, and the difference between
            Monte Carlo and TD methods. Confidence intervals tell you how many seeds to run.
            Every serious RL practitioner and researcher needs working statistics."""), unsafe_allow_html=True)

        st.subheader("1. Standard Error — The Core of MC Gradient Estimation")
        st.latex(r"\text{SE}[\hat\mu_N] = \frac{\sigma}{\sqrt{N}} \quad\text{(how accurate is averaging N samples?)}")
        st.markdown("To **halve** the standard error: need **4× more** samples. To reduce it **10×**: need **100× more** samples. This is why high-variance REINFORCE is so slow — you pay a quadratic cost to reduce gradient noise.")

        np.random.seed(42)
        true_mean = 150.0; ns = [5, 10, 25, 50, 100, 250, 500]
        stds = [np.std([np.random.normal(true_mean, 40, n).mean() for _ in range(200)]) for n in ns]
        fig_v, axes_v = _fig(1, 2, 13, 4)
        axes_v[0].plot(ns, stds, color="#ef5350", lw=2.5, marker="o", ms=7, label="Empirical std")
        axes_v[0].plot(ns, [40/np.sqrt(n) for n in ns], color="#4caf50", lw=2, ls="--", label="40/√N (theory)")
        axes_v[0].set_xlabel("N (episodes averaged)", color="white")
        axes_v[0].set_ylabel("Std of gradient estimate", color="white")
        axes_v[0].set_title("SE ∝ 1/√N — the fundamental cost of MC gradient estimation",
                            color="white", fontweight="bold")
        axes_v[0].legend(facecolor=CARD, labelcolor="white"); axes_v[0].grid(alpha=0.12)
        for n, col in [(5, "#ef5350"), (50, "#ffa726"), (500, "#4caf50")]:
            axes_v[1].hist([np.random.normal(true_mean, 40, n).mean() for _ in range(200)],
                           bins=25, alpha=0.6, color=col, label=f"N={n}")
        axes_v[1].axvline(true_mean, color="white", ls="--", lw=2, label="True mean=150")
        axes_v[1].set_xlabel("Estimated return", color="white"); axes_v[1].set_ylabel("Count", color="white")
        axes_v[1].set_title("Distribution of estimates: more samples → tighter", color="white", fontweight="bold")
        axes_v[1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_v[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_v); plt.close()

        st.subheader("2. Bias-Variance Tradeoff in RL Advantage Estimators")
        st.latex(r"\text{MSE}[\hat\theta] = \text{Bias}[\hat\theta]^2 + \text{Var}[\hat\theta]")
        st.markdown("In RL, this tradeoff appears directly in how we estimate the advantage A(s,a) = Q(s,a) - V(s):")
        st.dataframe(pd.DataFrame({
            "Estimator": ["MC return G_t", "1-step TD δ_t", "n-step return", "GAE(γ,λ)"],
            "Bias": ["Zero", "High (V is imperfect)", "γⁿ × V_error (shrinks with n)", "Tunable via λ∈[0,1]"],
            "Variance": ["Very high (sums all future random rewards)", "Low (only 1 random reward)", "Grows with n", "Tunable via λ"],
            "Formula": ["Σ γ^k r_{t+k}", "r_t+γV(s')−V(s)", "Σ_{k<n} γ^k r + γⁿV(s_{t+n})", "Σ_k (γλ)^k δ_{t+k}"],
            "Equivalent λ": ["λ=1", "λ=0", "Fixed n", "λ∈[0,1]"],
        }), use_container_width=True, hide_index=True)

        st.subheader("3. Reproducibility — The RL Research Crisis and the Solution")
        st.code("""
# The correct evaluation protocol (required for publication-quality results):
seeds = [0, 1, 2, 3, 4]  # minimum 5 seeds; 10 preferred
final_rewards = [train_and_evaluate(seed=s, steps=1_000_000) for s in seeds]

mean   = np.mean(final_rewards)
std    = np.std(final_rewards)
stderr = std / np.sqrt(len(seeds))  # standard error of the mean

print(f"Result: {mean:.0f} ± {stderr:.0f} (n={len(seeds)} seeds)")
# NEVER report: "our method gets 6000 reward" from a single lucky seed
# Always report: mean ± stderr over at least 5 seeds
""", language="python")

    # ── INFORMATION THEORY ────────────────────────────────────────────────
    with tab_info:
        _sec("📡", "Information Theory — Entropy, KL Divergence, Cross-Entropy",
             "H(π) in SAC, D_KL in PPO/TRPO, −logπ as policy loss — in every modern RL algorithm", "#ad1457")

        st.markdown(_card("#ad1457", "📡", "Three quantities that appear directly in RL formulas",
            """(1) <b>Entropy H(π)</b>: SAC objective = E[r + αH(π)]; A2C/A3C entropy bonus −c₂H(π);
            PPO entropy term prevents premature policy collapse.
            (2) <b>KL divergence D_KL(π||π_old)</b>: TRPO constraint; approximated by clip in PPO;
            penalty β·D_KL(π||π_ref) in RLHF prevents reward hacking.
            (3) <b>Cross-entropy H(P,Q)</b>: the policy loss −logπ(a|s) in REINFORCE and PPO is
            cross-entropy between the one-hot distribution and the policy; RLHF reward model
            training uses cross-entropy on preference pairs."""), unsafe_allow_html=True)

        st.subheader("1. Entropy — Measuring Policy Diversity and Uncertainty")
        st.latex(r"H(P) = -\sum_k P(k)\log P(k) = \mathbb{E}_{x\sim P}[-\log P(x)]")
        st.markdown("**Derivation from optimal coding:** The optimal code length for event k with probability P(k) is −log₂P(k) bits. Entropy is the **expected optimal code length** — the minimum average bits needed to encode a random draw.")
        st.markdown("Key properties: H ≥ 0 always. H = 0 iff deterministic (one event has prob=1). H = log|A| iff uniform (maximum diversity).")
        st.markdown("**SAC objective** explicitly maximises entropy:")
        st.latex(r"J^{\text{SAC}}(\pi) = \mathbb{E}_\tau\!\left[\sum_t\gamma^t\bigl(r_t + \alpha H(\pi(\cdot|s_t))\bigr)\right]")

        p_v = st.slider("P(action=0)", 0.01, 0.99, 0.5, 0.01, key="info_p_v")
        p2 = np.array([p_v, 1-p_v])
        H_v = -sum(p * np.log(p) for p in p2 if p > 0)
        H_max = np.log(2)
        xr = np.linspace(0.01, 0.99, 200)
        Hc = -xr*np.log(xr) - (1-xr)*np.log(1-xr)
        fig_e, axes_e = _fig(1, 2, 12, 3.5)
        axes_e[0].bar(["Action 0", "Action 1"], p2, color=["#ad1457", "#546e7a"])
        axes_e[0].set_title(f"H={H_v:.3f} nats ({100*H_v/H_max:.0f}% of max)",
                            color="white", fontweight="bold")
        axes_e[0].set_ylabel("Probability", color="white"); axes_e[0].grid(alpha=0.1, axis="y")
        axes_e[1].plot(xr, Hc, color="#ad1457", lw=2.5)
        axes_e[1].axvline(p_v, color="#ef5350", ls="--", lw=2, label=f"Current H={H_v:.3f}")
        axes_e[1].axhline(H_max, color="#4caf50", ls=":", lw=1.5, label=f"Max H={H_max:.3f}")
        axes_e[1].set_xlabel("P(action=0)", color="white"); axes_e[1].set_ylabel("H(π)", color="white")
        axes_e[1].set_title("Entropy — maximum at uniform distribution", color="white", fontweight="bold")
        axes_e[1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_e[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_e); plt.close()

        st.subheader("2. KL Divergence — How Far the Policy Has Moved")
        st.latex(r"D_\text{KL}(P\|Q) = \sum_k P(k)\log\frac{P(k)}{Q(k)} = \mathbb{E}_{x\sim P}\!\left[\log\frac{P(x)}{Q(x)}\right]")
        st.markdown(_proof("KL divergence is always ≥ 0 (Gibbs' inequality)",
            """By Jensen's inequality applied to the concave function log:<br>
            D_KL(P||Q) = -E_{x~P}[log(Q(x)/P(x))]<br>
            ≥ -log(E_{x~P}[Q(x)/P(x)])  — Jensen's: -log(E) ≤ E[-log] for concave log<br>
            = -log(∑_x P(x)·Q(x)/P(x)) = -log(∑_x Q(x)) = -log(1) = 0 ✓<br>
            Equality holds if and only if P = Q everywhere."""), unsafe_allow_html=True)
        st.markdown("**PPO:** clip r_t ∈ [1-ε, 1+ε] approximately enforces D_KL(π_new||π_old) ≤ ε²/2 per update. **RLHF:** β·D_KL(π||π_ref) prevents the fine-tuned LLM from drifting into reward-hacking territory.")

        mu2 = st.slider("New policy mean μ₂", -2.0, 2.0, 0.5, 0.1, key="kl_mu2_v")
        xkl = np.linspace(-5, 5, 300); sig = 1.0
        p1 = np.exp(-xkl**2 / (2*sig**2)) / (sig*np.sqrt(2*np.pi))
        p2kl = np.exp(-(xkl-mu2)**2 / (2*sig**2)) / (sig*np.sqrt(2*np.pi))
        kl_val = (mu2**2) / (2*sig**2)
        fig_kl, ax_kl = _fig(1, 1, 10, 3.5)
        ax_kl.plot(xkl, p1, color="#0288d1", lw=2.5, label="π_old: N(0,1)")
        ax_kl.plot(xkl, p2kl, color="#e65100", lw=2.5, label=f"π_new: N({mu2:.1f},1)")
        ax_kl.fill_between(xkl, 0, np.minimum(p1, p2kl), alpha=0.2, color="#4caf50", label="Overlap region")
        status_kl = "✅ safe" if kl_val < 0.015 else "⚠️ borderline" if kl_val < 0.05 else "❌ too large"
        ax_kl.set_title(f"KL(π_new||π_old) = {kl_val:.3f} — {status_kl} (PPO enforces KL ≤ ε²/2 ≈ 0.02)",
                        color="white", fontweight="bold")
        ax_kl.set_xlabel("Action value", color="white"); ax_kl.set_ylabel("Density", color="white")
        ax_kl.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_kl.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_kl); plt.close()

        st.subheader("3. Cross-Entropy — The Policy Training Loss")
        st.latex(r"H(P,Q) = -\sum_k P(k)\log Q(k) = H(P) + D_\text{KL}(P\|Q)")
        st.markdown("**The policy loss is cross-entropy:**")
        st.latex(r"H(\delta_a, \pi_\theta) = -\sum_k \delta(k=a)\log\pi_\theta(k|s) = -\log\pi_\theta(a|s)")
        st.markdown("The one-hot δ(k=a) collapses the sum to one term. Minimising cross-entropy = maximising log-probability of taken action. Weighted by advantage → REINFORCE/PPO gradient update.")

    # ── PYTHON & NUMPY ────────────────────────────────────────────────────
    with tab_py:
        _sec("🐍", "Python & NumPy — The Implementation Layer",
             "Vectorised ops, broadcasting, all essential RL patterns — used throughout the portal", "#0288d1")

        st.subheader("1. Array Fundamentals")
        st.code("""
import numpy as np

s   = np.array([0.02, -0.01, 0.04, 0.03])    # 1D vector, shape (4,)
W   = np.random.randn(32, 4) * np.sqrt(2/4)  # 2D matrix, shape (32,4)
Q   = np.zeros((100, 4))                       # Q-table, shape (100,4)
eye = np.eye(4)                                # 4×4 identity matrix

# Attributes
print(W.shape)   # (32, 4)  — dimensions
print(W.dtype)   # float64
print(W.ndim)    # 2  — number of dimensions

# Indexing
W[0]        # first row, shape (4,)
W[:, 0]     # first column, shape (32,)
W[0:5, 2]   # rows 0-4 of column 2, shape (5,)

# Reshape (critical for batching)
s.reshape(1, -1)   # (4,) → (1,4): add batch dimension
W.reshape(-1)      # (32,4) → (128,): flatten all
""", language="python")

        st.subheader("2. Vectorised Operations — Why Python Loops Are Avoided")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Slow — Python loop:**")
            st.code("for i in range(1000):\n    Q[i] += alpha*(r[i]+gamma*Qn[i]-Q[i])\n# ~1ms/element = 1 second total", language="python")
        with col2:
            st.markdown("**Fast — vectorised (1000× faster):**")
            st.code("Q += alpha * (r + gamma*Qn - Q)\n# NumPy C loop: ~1 microsecond", language="python")

        st.subheader("3. Broadcasting Rules")
        st.code("""
# Broadcasting: dimensions aligned right-to-left, missing dims broadcast

states = np.random.randn(64, 4)    # batch of 64 states
W      = np.random.randn(32, 4)    # weight matrix
h = states @ W.T + np.zeros(32)   # (64,4)@(4,32)+(32,)=(64,32) — bias broadcasts

# Advantage normalisation (used in PPO)
adv = np.random.randn(512)
adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
""", language="python")

        st.subheader("4. Essential RL Patterns")
        st.code("""
# 1. Softmax (numerical stability)
def softmax(x): e = np.exp(x - x.max()); return e/e.sum()

# 2. Epsilon-greedy action selection
def eps_greedy(Q, eps=0.1):
    if np.random.rand() < eps: return np.random.randint(len(Q))
    return np.argmax(Q)

# 3. Discounted returns (backward pass)
def discounted_returns(rewards, gamma=0.99):
    G = 0.0; R = []
    for r in reversed(rewards): G = r + gamma*G; R.insert(0, G)
    return np.array(R)

# 4. Replay buffer (DQN/SAC/TD3)
class ReplayBuffer:
    def __init__(self, cap, obs_dim):
        self.s  = np.zeros((cap, obs_dim)); self.a  = np.zeros(cap)
        self.r  = np.zeros(cap);           self.s2 = np.zeros((cap, obs_dim))
        self.d  = np.zeros(cap, dtype=bool); self.ptr = self.size = 0; self.cap = cap
    def add(self, s, a, r, s2, d):
        i = self.ptr; self.s[i]=s; self.a[i]=a; self.r[i]=r; self.s2[i]=s2; self.d[i]=d
        self.ptr = (self.ptr+1)%self.cap; self.size = min(self.size+1, self.cap)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]

# 5. GAE — Generalised Advantage Estimation (PPO)
def compute_gae(rewards, values, dones, last_val, gamma=0.99, lam=0.95):
    T = len(rewards); A = np.zeros(T); gae = 0.0
    for t in reversed(range(T)):
        next_v = last_val if t==T-1 else values[t+1]*(1-dones[t])
        delta = rewards[t] + gamma*next_v - values[t]
        gae = delta + gamma*lam*(1-dones[t])*gae; A[t] = gae
    return A, A + values   # advantages, returns
""", language="python")

    # ── NEURAL NETWORK MATH ──────────────────────────────────────────────
    with tab_nn:
        _sec("🧠", "Neural Network Math — Implement Backprop Once, Use Forever",
             "Activations, forward pass, manual backprop, weight init — understand before using autograd", "#558b2f")

        st.markdown(_card("#558b2f", "🧠", "Why implement backprop manually before using PyTorch",
            """PyTorch computes gradients automatically via autograd. This is convenient but hides the
            mechanics. If you have never implemented backpropagation manually, you will struggle to:
            debug gradient issues (exploding, vanishing, wrong shapes), understand why architectural
            choices (residual connections, layer norm) help, implement custom loss functions for new
            RL algorithms, and read theoretical RL papers that give gradient expressions explicitly.
            Implement it once in NumPy (this section does it), then use PyTorch for everything else.
            The effort is ~2 hours and the understanding is permanent."""), unsafe_allow_html=True)

        st.subheader("1. Activation Functions — Why Non-Linearity Is Essential")
        st.markdown("Without activations: W₃(W₂(W₁x)) = (W₃W₂W₁)x = Wx — just one big linear map. Non-linearity makes networks universal approximators.")
        xact = np.linspace(-3, 3, 200)
        acts = [
            ("ReLU",    np.maximum(0, xact), (xact>0).astype(float), "#7c4dff",
             "Default hidden layers. No vanishing gradient for x>0. Use He init."),
            ("Sigmoid", 1/(1+np.exp(-xact)),
             (1/(1+np.exp(-xact)))*(1-1/(1+np.exp(-xact))), "#0288d1",
             "Output layer for probs. LSTM gates. Reward model outputs."),
            ("Tanh",    np.tanh(xact), 1-np.tanh(xact)**2, "#00897b",
             "Output in [-1,1]. Better than sigmoid for hidden layers."),
            ("GELU",    xact*0.5*(1+np.tanh(np.sqrt(2/np.pi)*(xact+0.044715*xact**3))),
             np.ones_like(xact)*0.5, "#e65100",
             "Transformers, GPT, BERT. Smooth ReLU variant."),
        ]
        fig_a, axes_a = _fig(1, 4, 16, 3.5)
        for ax, (nm, f, df, col, usage) in zip(axes_a, acts):
            ax.plot(xact, f, color=col, lw=2.5, label="f(x)")
            ax.plot(xact, df, color=col, lw=1.5, ls="--", alpha=0.7, label="f'(x)")
            ax.set_title(nm, color="white", fontsize=9, fontweight="bold")
            ax.axhline(0, color="#2a2a3e"); ax.axvline(0, color="#2a2a3e"); ax.grid(alpha=0.1)
            ax.legend(facecolor=CARD, labelcolor="white", fontsize=6)
        plt.tight_layout(); st.pyplot(fig_a); plt.close()

        st.subheader("2. Complete Forward + Backward Pass Implementation")
        st.code("""
class TwoLayerPolicy:
    \"\"\"2-layer policy for CartPole: 4 inputs → 32 hidden → 2 action probs\"\"\"
    def __init__(self, seed=0):
        np.random.seed(seed)
        # He initialisation: variance = 2/fan_in for ReLU networks
        self.W1 = np.random.randn(4, 32) * np.sqrt(2/4)   # (in, hid)
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 2) * np.sqrt(2/32)  # (hid, out)
        self.b2 = np.zeros(2)
        self.cache = {}  # store intermediates for backward pass

    def forward(self, x):
        z1 = x @ self.W1 + self.b1      # linear: (4,)@(4,32)=(32,)
        h1 = np.maximum(0, z1)           # ReLU: zero out negatives
        z2 = h1 @ self.W2 + self.b2     # linear: (32,)@(32,2)=(2,)
        e  = np.exp(z2 - z2.max())      # stability: subtract max
        p  = e / e.sum()                 # softmax: action probabilities
        self.cache = dict(x=x, z1=z1, h1=h1, z2=z2, p=p)
        return p

    def backward(self, action, advantage):
        \"\"\"Chain rule applied backward. Returns gradients for all weights.\"\"\"
        x, z1, h1, z2, p = (self.cache[k] for k in ['x','z1','h1','z2','p'])

        # Output layer: d(cross-entropy)/d(logits) for softmax
        d_z2 = p.copy(); d_z2[action] -= 1.0    # gradient of CE loss
        d_z2 *= -advantage                        # REINFORCE: scale by return

        # Backward through W2: z2 = h1 @ W2 + b2
        d_W2 = np.outer(h1, d_z2)   # (32,2) gradient for W2
        d_b2 = d_z2                   # (2,) gradient for b2
        d_h1 = self.W2 @ d_z2        # (32,) — chain through W2

        # Backward through ReLU: h1 = max(0, z1)
        d_z1 = d_h1 * (z1 > 0)      # (32,) — zero gradient for negatives

        # Backward through W1: z1 = x @ W1 + b1
        d_W1 = np.outer(x, d_z1)    # (4,32) gradient for W1
        d_b1 = d_z1                   # (32,) gradient for b1

        return d_W1, d_b1, d_W2, d_b2

    def update(self, grads, lr=0.01):
        d_W1, d_b1, d_W2, d_b2 = grads
        # Gradient ascent on J (loss was negated in backward, so this is +∇J)
        self.W1 -= lr * d_W1; self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2; self.b2 -= lr * d_b2
""", language="python")

        st.subheader("3. Weight Initialisation — He vs Xavier")
        st.markdown("**He initialisation** (designed for ReLU networks):")
        st.latex(r"W \sim \mathcal{N}\!\left(0,\;\frac{2}{n_\text{in}}\right)")
        st.markdown("Why 2/n_in? ReLU zeros ~half of neurons. Multiply initial variance by 2 to compensate and keep activation variance constant through layers. **Xavier** (for tanh/sigmoid): 2/(n_in+n_out).")

        np.random.seed(42); n = 100; nl = 10
        fig_ini, axes_ini = _fig(1, 2, 12, 4)
        for ax, scale, title, col in [
            (axes_ini[0], 1.0, "Bad init (scale=1): activations die or explode", "#ef5350"),
            (axes_ini[1], np.sqrt(2/n), "He init (√2/n): stable activations through 10 layers", "#4caf50")
        ]:
            xv = np.random.randn(1000, n)
            for _ in range(nl):
                xv = np.maximum(0, xv @ (np.random.randn(n, n) * scale))
            ax.hist(xv.flatten(), bins=50, color=col, alpha=0.7)
            ax.set_xlabel("Activation value", color="white")
            ax.set_ylabel("Count", color="white")
            ax.set_title(title, color="white", fontweight="bold", fontsize=9)
            ax.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ini); plt.close()

    # ── SELF-ASSESSMENT ────────────────────────────────────────────────────
    with tab_test:
        _sec("✅", "Self-Assessment — Are You Ready for Stage 0?",
             "Answer all questions confidently before starting Deep Learning Prerequisites", "#ffa726")
        st.markdown("Try each question without looking. Click to reveal. If you cannot answer confidently, go back to that tab.")

        questions = [
            ("🔢 Linear Algebra", [
                ("What shape is the output of (32,4) @ (4,)?",
                 "(32,). Matrix (m,n) @ vector (n,) → vector (m,). Each of the 32 rows dot-products with the 4-element input to give one scalar output."),
                ("Why does backprop use W.T @ gradient?",
                 "If h=W@x, then ∂L/∂x = W^T @ ∂L/∂h. This comes from the Jacobian of the linear map being W itself, and backprop uses the transpose Jacobian."),
                ("What does det(W) < 0.01 mean for a neural network layer?",
                 "The matrix is near-singular — it collapses almost all input information to a near-zero dimensional subspace. Gradients vanish. This is the vanishing gradient problem in matrix form."),
            ]),
            ("📈 Calculus", [
                ("d/dx relu(x) = ?",
                 "1 if x>0, else 0 (Heaviside step function). In NumPy: (x > 0).astype(float). Used in every backprop through a ReLU layer."),
                ("If L = (y_hat - y)², what is ∂L/∂y_hat?",
                 "2(y_hat - y). Chain rule: d/dy_hat [(y_hat-y)²] = 2(y_hat-y)·1. This is the gradient of MSE loss."),
                ("In RL, why do we write θ ← θ + α∇J (not θ - α∇J)?",
                 "We MAXIMISE reward J(θ), so we do gradient ASCENT — move parameters in the direction of steepest increase (+). Loss minimisation uses descent (−)."),
            ]),
            ("🎲 Probability & Statistics", [
                ("What is E[G_t | s_t = s] in RL?",
                 "The value function V(s) — the expected total discounted future reward starting from state s following policy π. This is what the Critic estimates."),
                ("Why does baseline b(s) not bias the policy gradient?",
                 "E_{a~π}[∇logπ(a|s)·b(s)] = b(s)·∇∑_a π(a|s) = b(s)·∇1 = 0. The gradient of a sum that always equals 1 is zero."),
                ("5 seeds give rewards [150,200,180,120,160]. What do you report?",
                 "Mean±stderr: mean=162, std=29.3, stderr=29.3/√5=13.1. Report: 162±13 (n=5 seeds). Never report just the best single seed."),
            ]),
            ("📡 Information Theory", [
                ("Entropy of uniform 4-action policy?",
                 "H = -4×(0.25×log(0.25)) = log(4) ≈ 1.386 nats. This is the maximum possible entropy for 4 actions."),
                ("PPO shows KL(π_new||π_old) = 0.15. Acceptable?",
                 "No. PPO clip ε=0.2 approximately enforces KL ≤ ε²/2 ≈ 0.02. KL=0.15 is 7.5× too large — reduce learning rate or stop early."),
                ("Why is -log π(a|s) called cross-entropy loss?",
                 "H(one_hot(a), π) = -∑_k δ(k=a)·logπ(k|s) = -logπ(a|s). The one-hot collapses the sum to a single term."),
            ]),
        ]

        for subject, qs in questions:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:10px;'
                        f'padding:.8rem 1.2rem;margin:.5rem 0">'
                        f'<b style="color:white">{subject}</b></div>', unsafe_allow_html=True)
            for q, ans in qs:
                with st.expander(f"❓ {q}"):
                    st.markdown(f'<div style="background:#0a2a0a;border-left:3px solid #4caf50;'
                                f'padding:.7rem 1rem;border-radius:0 8px 8px 0">'
                                f'<b style="color:#4caf50">✅ Answer:</b> '
                                f'<span style="color:#b0b0cc;line-height:1.7">{ans}</span></div>',
                                unsafe_allow_html=True)

        st.divider()
        st.subheader("📚 The 6 Best Resources")
        for icon, title, desc, url in [
            ("🎥","3Blue1Brown — Essence of Linear Algebra","15 visual videos. Build geometric intuition before formulas. Essential.","https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"),
            ("🎥","3Blue1Brown — Essence of Calculus","12 videos. Chain rule, gradients, the fundamental theorem.","https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr"),
            ("💻","Andrej Karpathy — micrograd","170 lines of Python implementing autograd from scratch. Read every line, then reimplement without looking.","https://github.com/karpathy/micrograd"),
            ("📖","Michael Nielsen — Neural Networks and Deep Learning","Free online book. Chapter 2 derives backpropagation completely from scratch.","http://neuralnetworksanddeeplearning.com"),
            ("📖","Mathematics for Machine Learning — Deisenroth et al.","Free PDF. All 6 topics with ML context throughout. Graduate-level but accessible.","https://mml-book.github.io"),
            ("💻","NumPy Official Quickstart","Master arrays, broadcasting, vectorisation. Takes 2 hours.","https://numpy.org/doc/stable/user/quickstart.html"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)

    foundation_notes = [
        (tab_why, "Why & What", "foundations"),
        (tab_la, "Linear Algebra", "foundations_linear_algebra"),
        (tab_calc, "Calculus", "foundations_calculus"),
        (tab_opt, "Optimisation", "foundations_optimisation"),
        (tab_prob, "Probability", "foundations_probability"),
        (tab_stat, "Statistics", "foundations_statistics"),
        (tab_info, "Information Theory", "foundations_information_theory"),
        (tab_py, "Python & NumPy", "foundations_python_numpy"),
        (tab_nn, "Neural Network Math", "foundations_neural_network_math"),
        (tab_test, "Self-Assessment", "foundations_self_assessment"),
    ]
    for tab, note_title, note_slug in foundation_notes:
        with tab:
            render_notes(f"Math & CS Foundations - {note_title}", note_slug)
