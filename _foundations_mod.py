"""
_foundations_mod.py — Mathematical & Programming Foundations
Everything you need before starting Deep Learning Prerequisites.
Tabs: Linear Algebra | Calculus & Optimisation | Probability & Statistics
      Information Theory | Python & NumPy | Neural Network Math
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


def _sec(emoji, title, sub, color="#00897b"):
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
        '<h2 style="color:white;margin:0;font-size:2rem">📐 Mathematical & Programming Foundations</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'Stage -1: Everything you need before starting Deep Learning. '
        'Linear algebra, calculus, probability, information theory, Python, and neural network math — '
        'each derived from scratch with practical RL examples.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🗺️ What You Need & Why",
        "🔢 Linear Algebra",
        "📈 Calculus & Optimisation",
        "🎲 Probability & Statistics",
        "📡 Information Theory",
        "🐍 Python & NumPy",
        "🧠 Neural Network Math",
        "✅ Self-Assessment",
    ])
    (tab_ov, tab_la, tab_calc, tab_prob,
     tab_info, tab_py, tab_nn, tab_test) = tabs

    # ── OVERVIEW ─────────────────────────────────────────────────────────
    with tab_ov:
        _sec("🗺️", "What You Need Before Deep Learning",
             "Honest prerequisites — what to study and why each topic appears in RL", "#00897b")

        st.markdown(_card("#00897b", "📋", "The complete prerequisite map",
            """Every formula in deep RL requires some combination of these six foundations.
            Linear algebra: the neural network is a sequence of matrix multiplications.
            Calculus: backpropagation is the chain rule applied recursively.
            Probability: the policy π(a|s) is a probability distribution; we optimise expectations.
            Information theory: KL divergence appears in PPO and TRPO; entropy bonus appears in SAC and A2C.
            Python/NumPy: all implementations in this portal use these; GPU RL uses the same operations.
            Neural network math: you need to derive backpropagation once from scratch before using PyTorch autograd.
            None of these require deep expertise before starting. You need working knowledge —
            able to apply the concept, not prove every theorem. This module builds exactly that."""),
            unsafe_allow_html=True)

        st.dataframe(pd.DataFrame({
            "Topic": ["Linear Algebra", "Calculus & Optimisation", "Probability & Statistics",
                      "Information Theory", "Python & NumPy", "Neural Network Math"],
            "Why it matters in RL": [
                "Policy network = matrix multiplications; value function = linear combination of features",
                "Gradient ascent on J(θ); chain rule gives backpropagation; SGD updates all weights",
                "Policy π(a|s) is a distribution; objectives are expectations E[...]; Bayes used in PSRL",
                "Entropy bonus in SAC/A2C; KL divergence in PPO/TRPO; cross-entropy as policy loss",
                "NumPy used throughout portal; PyTorch = NumPy with autograd; vectorised env batching",
                "Backprop gives ∇J; activations define expressivity; loss functions define what to optimise",
            ],
            "Time to sufficient level": [
                "2–3 weeks", "2–3 weeks", "2–3 weeks", "3–5 days",
                "1–2 weeks", "1 week",
            ],
            "Starting resource": [
                "3Blue1Brown Essence of Linear Algebra (YouTube)",
                "3Blue1Brown Essence of Calculus (YouTube)",
                "Probability for Machine Learning — Jason Brownlee (free)",
                "Cover & Thomas Elements of Information Theory Ch.1–2",
                "NumPy quickstart tutorial (numpy.org)",
                "Andrej Karpathy micrograd (GitHub)",
            ],
        }), use_container_width=True, hide_index=True)

    # ── LINEAR ALGEBRA ────────────────────────────────────────────────────
    with tab_la:
        _sec("🔢", "Linear Algebra — The Language of Neural Networks",
             "Vectors, matrices, dot products, eigenvalues, SVD — all through the lens of RL", "#6a1b9a")

        st.markdown(_card("#6a1b9a", "🔢", "Why linear algebra is the first prerequisite",
            """A neural network is literally a sequence of matrix multiplications followed by non-linear
            functions. When you write Q(s,a) = W2·relu(W1·s + b1) + b2, every operation is linear
            algebra: W1 and W2 are matrices, s is a vector, relu is element-wise, b1 and b2 are vectors.
            Understanding what a matrix multiplication does geometrically — it rotates and scales
            a vector — is what makes neural networks intuitive rather than magical.
            The policy gradient ∇J is a vector in parameter space pointing toward higher reward.
            The Fisher information matrix F appears in TRPO as a metric on the space of policies.
            Eigenvalues determine how fast gradient descent converges. SVD underpins representation
            learning. You do not need to prove theorems — you need to compute fluently."""), unsafe_allow_html=True)

        st.subheader("1. Vectors — The Fundamental Object")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered list of $n$ real numbers.
            In RL: the state $s \in \mathbb{R}^n$ (CartPole has $n=4$: position, velocity, angle, angular velocity).
            """)
            st.latex(r"\mathbf{s} = \begin{bmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{bmatrix} \in \mathbb{R}^4")
            st.markdown(r"""
            **Dot product** (projects one vector onto another):
            """)
            st.latex(r"\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta")
            st.markdown(r"In RL: the linear layer $\mathbf{w}^\top\mathbf{s}$ is a dot product — it measures how much the state aligns with the weight direction.")
        with col2:
            # Visualise dot product
            np.random.seed(42)
            fig_dot, ax_dot = _fig(1, 1, 5.5, 4)
            u = np.array([2.0, 1.0]); v = np.array([1.0, 2.0])
            ax_dot.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
                          color="#7c4dff", width=0.03, label=f"u = {u}")
            ax_dot.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                          color="#ffa726", width=0.03, label=f"v = {v}")
            proj = np.dot(u, v) / np.dot(v, v) * v
            ax_dot.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1,
                          color="#4caf50", width=0.02, alpha=0.7, label=f"proj u→v")
            ax_dot.set_xlim(-0.5, 3.5); ax_dot.set_ylim(-0.5, 3.0)
            ax_dot.set_xlabel("x", color="white"); ax_dot.set_ylabel("y", color="white")
            ax_dot.set_title(f"u·v = {np.dot(u,v):.1f}", color="white", fontweight="bold")
            ax_dot.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_dot.grid(alpha=0.15); ax_dot.axhline(0, color="#2a2a3e"); ax_dot.axvline(0, color="#2a2a3e")
            plt.tight_layout(); st.pyplot(fig_dot); plt.close()

        st.subheader("2. Matrices — Transformations of Space")
        st.markdown(r"""
        A matrix $W \in \mathbb{R}^{m \times n}$ transforms an $n$-dimensional vector into an $m$-dimensional vector.
        **Matrix-vector multiplication** is the core operation of every neural network layer:
        """)
        st.latex(r"\mathbf{h} = W\mathbf{x} + \mathbf{b} \quad W\in\mathbb{R}^{m\times n},\;\mathbf{x}\in\mathbb{R}^n,\;\mathbf{h}\in\mathbb{R}^m")
        st.markdown(r"""
        **Practical example:** First layer of a CartPole policy network:
        - $\mathbf{s} \in \mathbb{R}^4$ (4 state features)
        - $W_1 \in \mathbb{R}^{32 \times 4}$: 32 neurons, each with 4 weights (one per input feature)
        - $\mathbf{h}_1 = W_1\mathbf{s} + \mathbf{b}_1 \in \mathbb{R}^{32}$: 32 hidden activations

        **Why the transformation matters geometrically:** $W$ rotates and stretches the input space.
        Each row of $W$ is a "feature detector" — it fires (large output) when the input aligns with it.
        """)
        st.code("""
import numpy as np

# CartPole state: [position, velocity, angle, angular_velocity]
s = np.array([0.02, -0.01, 0.04, 0.03])   # shape (4,)

# First hidden layer: 32 neurons × 4 inputs
W1 = np.random.randn(32, 4) * np.sqrt(2/4)  # He initialisation
b1 = np.zeros(32)

# Matrix-vector product: the core of every neural network layer
h1 = W1 @ s + b1   # shape (32,) — 32 hidden activations
h1 = np.maximum(0, h1)  # ReLU activation: max(0, x) element-wise
print(h1.shape)  # (32,)
""", language="python")

        st.subheader("3. Key Matrix Operations for Deep RL")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Transpose** — flip rows and columns:")
            st.latex(r"(W^\top)_{ij} = W_{ji} \quad W\in\mathbb{R}^{m\times n} \Rightarrow W^\top\in\mathbb{R}^{n\times m}")
            st.markdown(r"Used in backpropagation: $\nabla_\mathbf{x}\mathcal{L} = W^\top\nabla_\mathbf{h}\mathcal{L}$")
            st.markdown("**Matrix multiplication** — compose transformations:")
            st.latex(r"(AB)_{ij} = \sum_k A_{ik}B_{kj} \quad A\in\mathbb{R}^{m\times k},\;B\in\mathbb{R}^{k\times n}")
            st.markdown(r"The full forward pass: $Q(s,a) = W_2\,\text{relu}(W_1 s + b_1) + b_2$")
        with col2:
            st.markdown("**Jacobian** — derivative of vector w.r.t. vector:")
            st.latex(r"J = \frac{\partial\mathbf{f}}{\partial\mathbf{x}} \in \mathbb{R}^{m\times n} \quad J_{ij} = \frac{\partial f_i}{\partial x_j}")
            st.markdown(r"Backpropagation is computing Jacobian-vector products efficiently (the chain rule).")
            st.markdown("**Eigenvalues** — how much each direction is stretched:")
            st.latex(r"W\mathbf{v} = \lambda\mathbf{v} \quad \text{(eigenvector }\mathbf{v}\text{, eigenvalue }\lambda\text{)}")
            st.markdown(r"Largest eigenvalue of $W^\top W$ = spectral norm. Gradient explosion occurs when this is very large.")

        st.subheader("4. Interactive: Matrix Multiplication")
        st.markdown("Try it — enter a 2×2 matrix and see what it does to the unit square:")
        c1, c2 = st.columns(2)
        a11 = c1.slider("W[0,0]", -3.0, 3.0, 2.0, 0.1, key="la_a11")
        a12 = c1.slider("W[0,1]", -3.0, 3.0, 1.0, 0.1, key="la_a12")
        a21 = c2.slider("W[1,0]", -3.0, 3.0, 0.0, 0.1, key="la_a21")
        a22 = c2.slider("W[1,1]", -3.0, 3.0, 1.5, 0.1, key="la_a22")
        W = np.array([[a11, a12], [a21, a22]])
        square = np.array([[0,1,1,0,0],[0,0,1,1,0]], dtype=float)
        transformed = W @ square
        fig_mat, ax_mat = _fig(1, 1, 9, 4)
        ax_mat.plot(square[0], square[1], color="#546e7a", lw=2, ls="--", label="Original unit square")
        ax_mat.fill(square[0], square[1], alpha=0.15, color="#546e7a")
        ax_mat.plot(transformed[0], transformed[1], color="#7c4dff", lw=2.5, label="After W multiplication")
        ax_mat.fill(transformed[0], transformed[1], alpha=0.2, color="#7c4dff")
        ax_mat.set_xlim(-4, 4); ax_mat.set_ylim(-4, 4)
        ax_mat.axhline(0, color="#2a2a3e", lw=0.8); ax_mat.axvline(0, color="#2a2a3e", lw=0.8)
        ax_mat.set_title(f"det(W)={np.linalg.det(W):.2f} (area scaling factor)", color="white", fontweight="bold")
        ax_mat.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_mat.grid(alpha=0.12); plt.tight_layout(); st.pyplot(fig_mat); plt.close()

    # ── CALCULUS ─────────────────────────────────────────────────────────
    with tab_calc:
        _sec("📈", "Calculus & Optimisation — Derivatives Power Everything",
             "Gradients, chain rule, gradient descent — the engine of all policy learning", "#e65100")

        st.markdown(_card("#e65100", "📈", "Why calculus is the engine of deep RL",
            """Every policy gradient update is gradient ascent. Every value function update is gradient
            descent. Backpropagation — the algorithm that makes training neural networks possible —
            is nothing more than the chain rule applied recursively. When you see θ ← θ + α∇J(θ),
            the ∇J(θ) is a vector of partial derivatives: how much does the total reward change if
            you nudge each weight by a tiny amount? Computing this efficiently for a million-parameter
            network requires the chain rule applied through dozens of layer operations simultaneously.
            If you do not understand derivatives, gradient descent is magic. Once you do, it becomes
            straightforward: always move in the direction that increases (ascent) or decreases (descent)
            your objective function, scaled by the learning rate α."""), unsafe_allow_html=True)

        st.subheader("1. Derivatives — Instantaneous Rate of Change")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"**Definition from first principles (limit definition):**")
            st.latex(r"\frac{df}{dx} = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h}")
            st.markdown(r"**Interpretation:** slope of $f$ at point $x$ — how fast $f$ changes per unit increase in $x$.")
            st.markdown(r"**Key derivatives you must know:**")
            st.latex(r"\frac{d}{dx}x^n = nx^{n-1}, \quad \frac{d}{dx}e^x = e^x, \quad \frac{d}{dx}\ln x = \frac{1}{x}")
            st.latex(r"\frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x)) \quad\text{(sigmoid — used in reward models)}")
        with col2:
            x = np.linspace(-3, 3, 200)
            fig_deriv, ax_deriv = _fig(1, 1, 5.5, 4)
            ax_deriv.plot(x, x**2, color="#7c4dff", lw=2.5, label="f(x) = x²")
            ax_deriv.plot(x, 2*x, color="#ffa726", lw=2, label="f\'(x) = 2x")
            x0 = 1.5; tangent = x**2 + 2*x0*(x - x0)  # wait, tangent at x0=1.5
            # slope at x0=1.5 is 2*1.5=3, tangent: y - 1.5²= 3(x-1.5) → y = 3x - 4.5+2.25 = 3x-2.25
            tang = 3*x - 2.25
            ax_deriv.plot(x, tang, color="#4caf50", lw=1.5, ls="--", alpha=0.7, label=f"tangent at x={x0}")
            ax_deriv.scatter([x0], [x0**2], color="#ef5350", s=80, zorder=5)
            ax_deriv.set_ylim(-4, 6); ax_deriv.set_xlabel("x", color="white"); ax_deriv.set_ylabel("y", color="white")
            ax_deriv.set_title("Derivative = slope of tangent line", color="white", fontweight="bold")
            ax_deriv.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_deriv.grid(alpha=0.12); ax_deriv.axhline(0, color="#2a2a3e"); ax_deriv.axvline(0, color="#2a2a3e")
            plt.tight_layout(); st.pyplot(fig_deriv); plt.close()

        st.subheader("2. Partial Derivatives & Gradients — Multi-Variable Functions")
        st.markdown(r"""
        A neural network has millions of parameters. The loss function $\mathcal{L}(\theta_1,\theta_2,\ldots,\theta_n)$
        depends on all of them simultaneously. The **partial derivative** $\frac{\partial\mathcal{L}}{\partial\theta_i}$
        measures how $\mathcal{L}$ changes when we nudge only $\theta_i$, holding all others fixed.
        The **gradient** collects all partial derivatives into a vector:
        """)
        st.latex(r"\nabla_\theta\mathcal{L} = \left[\frac{\partial\mathcal{L}}{\partial\theta_1},\;\frac{\partial\mathcal{L}}{\partial\theta_2},\;\ldots,\;\frac{\partial\mathcal{L}}{\partial\theta_n}\right]^\top")
        st.markdown(r"""
        **The gradient always points toward the steepest increase of $\mathcal{L}$.**
        For gradient descent (minimise loss): $\theta \leftarrow \theta - \alpha\nabla_\theta\mathcal{L}$
        For gradient ascent (maximise reward): $\theta \leftarrow \theta + \alpha\nabla_\theta J$
        """)

        # Gradient visualisation
        x_g, y_g = np.meshgrid(np.linspace(-3,3,30), np.linspace(-3,3,30))
        z_g = x_g**2 + 2*y_g**2  # simple bowl
        gx, gy = 2*x_g, 4*y_g    # gradient
        fig_grad, ax_grad = _fig(1, 1, 9, 4.5)
        ax_grad.contourf(x_g, y_g, z_g, levels=20, cmap="RdYlGn_r", alpha=0.7)
        skip = 3
        ax_grad.quiver(x_g[::skip,::skip], y_g[::skip,::skip],
                       -gx[::skip,::skip], -gy[::skip,::skip],
                       color="white", alpha=0.8, scale=80, width=0.003)
        ax_grad.set_xlabel("θ₁", color="white"); ax_grad.set_ylabel("θ₂", color="white")
        ax_grad.set_title("Gradient descent: arrows point toward minimum (loss = x²+2y²)",
                          color="white", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_grad); plt.close()

        st.subheader("3. Chain Rule — The Heart of Backpropagation")
        st.markdown(r"""
        **The chain rule** computes derivatives of composed functions. If $\mathcal{L} = f(g(x))$:
        """)
        st.latex(r"\frac{d\mathcal{L}}{dx} = \frac{d\mathcal{L}}{dg}\cdot\frac{dg}{dx}")
        st.markdown(r"""
        For a 2-layer network $\mathcal{L}(\theta) = \text{loss}(W_2\cdot\text{relu}(W_1\cdot s))$:
        """)
        st.latex(r"\frac{\partial\mathcal{L}}{\partial W_1} = \frac{\partial\mathcal{L}}{\partial\mathbf{h}_2}\cdot\frac{\partial\mathbf{h}_2}{\partial\mathbf{h}_1}\cdot\frac{\partial\mathbf{h}_1}{\partial W_1}")
        st.markdown(r"""
        Backpropagation applies the chain rule from the output (loss) backward through every layer.
        **The key insight:** each layer's gradient depends only on its input and the gradient from the next layer.
        This is why gradients are computed backward (output → input) — each layer passes its gradient
        to the previous layer.

        **Worked example:** Policy network forward and backward pass:
        """)
        st.code("""
# Forward pass
s   = state          # shape (4,)
h1  = W1 @ s + b1   # shape (32,) — linear
h1r = np.maximum(0, h1)  # relu activation
logits = W2 @ h1r + b2  # shape (2,) — action logits
probs = softmax(logits)  # shape (2,) — action probs

# Loss: negative log prob of taken action a
loss = -np.log(probs[a]) * G_t  # REINFORCE loss

# Backward pass (chain rule applied in reverse)
d_logits = probs.copy(); d_logits[a] -= 1  # ∂loss/∂logits (softmax-CE gradient)
d_W2 = np.outer(h1r, d_logits)             # ∂loss/∂W2
d_h1r = W2.T @ d_logits                    # ∂loss/∂h1r
d_h1 = d_h1r * (h1 > 0)                    # ∂relu: 1 if positive, 0 if negative
d_W1 = np.outer(s, d_h1)                   # ∂loss/∂W1

# Update (gradient ascent for RL)
W1 += alpha * (-d_W1) * G_t  # negative because chain gives -∇J
""", language="python")

        st.subheader("4. Gradient Descent Variants")
        st.dataframe(pd.DataFrame({
            "Algorithm": ["SGD", "SGD + Momentum", "RMSProp", "Adam"],
            "Update rule": [
                "θ ← θ - α·∇L",
                "v ← βv - α·∇L; θ ← θ + v",
                "s ← βs + (1-β)(∇L)²; θ ← θ - α·∇L/√(s+ε)",
                "m ← β₁m+(1-β₁)∇L; v ← β₂v+(1-β₂)(∇L)²; θ ← θ - α·m̂/√(v̂+ε)",
            ],
            "Key property": [
                "Simple but sensitive to α; oscillates in ravines",
                "Builds up velocity in consistent directions; dampens oscillations",
                "Adapts learning rate per parameter; good for sparse gradients",
                "Combines momentum + RMSProp; default for most RL; robust α choice",
            ],
            "Used in RL": ["Rarely", "Sometimes with DDPG", "RMSProp in A3C", "Adam in DQN, PPO, SAC"],
        }), use_container_width=True, hide_index=True)

    # ── PROBABILITY ───────────────────────────────────────────────────────
    with tab_prob:
        _sec("🎲", "Probability & Statistics — RL Is a Probabilistic Framework",
             "Distributions, expectations, Bayes, conditional probability — foundations of stochastic policies", "#7c4dff")

        st.markdown(_card("#7c4dff", "🎲", "Why probability is unavoidable in RL",
            """The policy π(a|s) IS a probability distribution over actions given state. The reward signal
            is often stochastic. The environment transition p(s'|s,a) is a conditional probability.
            The value function V(s) = E[G_t|s_t=s] is an expectation. The policy gradient theorem
            is derived by differentiating through an expectation. Monte Carlo methods estimate
            expectations via sampling. Bayesian approaches maintain probability distributions over
            unknown quantities (PSRL maintains a posterior over MDP dynamics). You cannot read a single
            RL paper without encountering expectations, conditional probabilities, and distributions.
            This section builds the probabilistic intuition from first principles."""), unsafe_allow_html=True)

        st.subheader("1. Probability Fundamentals")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(r"""
            **Probability axioms (Kolmogorov):**
            """)
            st.latex(r"P(A) \geq 0 \quad \forall A")
            st.latex(r"P(\Omega) = 1 \quad\text{(total probability = 1)}")
            st.latex(r"P(A\cup B) = P(A)+P(B) \quad\text{if }A\cap B=\emptyset")
            st.markdown(r"**Conditional probability:**")
            st.latex(r"P(A|B) = \frac{P(A\cap B)}{P(B)} \quad\text{(prob of A given B occurred)}")
            st.markdown(r"**In RL:** $\pi(a|s) = P(\text{action}=a \mid \text{state}=s)$")
        with col2:
            st.markdown(r"**Bayes' theorem** (used in model-based RL):")
            st.latex(r"P(\theta|D) = \frac{P(D|\theta)\cdot P(\theta)}{P(D)}")
            st.markdown(r"""
            - $P(\theta)$ — prior: belief about model parameters before data
            - $P(D|\theta)$ — likelihood: probability of observed data given model
            - $P(\theta|D)$ — posterior: updated belief after seeing data
            - PSRL maintains posterior $P(\text{MDP}|\text{experience})$
            """)

        st.subheader("2. Expectation — The Most Important Operation in RL")
        st.markdown(r"""
        The expected value of a random variable $X$ weighted by its probability:
        """)
        st.latex(r"\mathbb{E}[X] = \sum_x x\cdot P(X=x) \quad\text{(discrete)}")
        st.latex(r"\mathbb{E}[X] = \int x\cdot p(x)\,dx \quad\text{(continuous)}")
        st.markdown(r"""
        **Properties (used constantly in RL derivations):**
        """)
        st.latex(r"\mathbb{E}[aX+bY] = a\mathbb{E}[X]+b\mathbb{E}[Y] \quad\text{(linearity)}")
        st.latex(r"\mathbb{E}[g(X)] = \sum_x g(x)P(X=x) \quad\text{(law of the unconscious statistician)}")
        st.markdown(r"""
        **In RL:** The objective is $J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}[\sum_t r_t]$ — the
        expected total reward. Monte Carlo estimation: run $N$ episodes, average the total rewards.
        Each episode is one sample from the distribution of trajectories.
        """)
        st.code("""
# Monte Carlo estimation of E[total_reward] — what REINFORCE does
N = 1000  # number of episodes
total_rewards = []
for _ in range(N):
    trajectory = collect_episode(env, policy)
    total_rewards.append(sum(r for _,_,r in trajectory))

E_reward = np.mean(total_rewards)  # estimate of J(θ)
std_err = np.std(total_rewards) / np.sqrt(N)  # ± this much uncertainty
""", language="python")

        st.subheader("3. Key Probability Distributions in RL")
        st.dataframe(pd.DataFrame({
            "Distribution": ["Categorical / Multinomial", "Gaussian / Normal", "Beta", "Dirichlet"],
            "Formula": [
                "P(X=k) = p_k, Σp_k = 1",
                "p(x) = exp(-(x-μ)²/2σ²) / (σ√2π)",
                "p(x) = x^(α-1)(1-x)^(β-1) / B(α,β)",
                "p(p) ∝ Πp_k^(αk-1)",
            ],
            "Where used in RL": [
                "Discrete action policy: π(a|s) = softmax(logits)",
                "Continuous action policy in PPO/SAC: a ~ N(μ(s), σ²(s))",
                "Thompson Sampling prior for bandit arms",
                "Bayesian prior over categorical distributions (PSRL)",
            ],
            "Key property": [
                "Sums to 1; differentiable via softmax parameterisation",
                "Conjugate prior for normal likelihood; reparameterisable",
                "Conjugate to Binomial; models probabilities in [0,1]",
                "Conjugate to Categorical; models distribution over distributions",
            ],
        }), use_container_width=True, hide_index=True)

        st.subheader("4. Variance — Why REINFORCE is Hard")
        st.markdown(r"""
        **Variance** measures how spread out a distribution is:
        """)
        st.latex(r"\text{Var}[X] = \mathbb{E}[(X-\mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2")
        st.markdown(r"""
        **Why high variance is a problem for RL:** The REINFORCE gradient estimator:
        """)
        st.latex(r"\hat g = \frac{1}{N}\sum_i G_i\cdot\nabla_\theta\log\pi_\theta(a_i|s_i)")
        st.markdown(r"""
        has variance proportional to $\text{Var}[G_t]$ — how much episode returns fluctuate.
        High variance → noisy gradient estimate → need more samples to converge.
        **Baseline trick reduces variance:** $\text{Var}[G_t - b] = \text{Var}[G_t] - 2\text{Cov}[G_t,b] + \text{Var}[b]$.
        With $b = \mathbb{E}[G_t]$, this reduces variance significantly.
        """)

    # ── INFORMATION THEORY ────────────────────────────────────────────────
    with tab_info:
        _sec("📡", "Information Theory — Directly Used in RL Algorithms",
             "Entropy · KL divergence · Cross-entropy — the language of PPO, SAC, and reward models", "#ad1457")

        st.markdown(_card("#ad1457", "📡", "Why information theory appears in RL formulas",
            """Three specific information-theoretic quantities appear directly in RL algorithm formulas:
            (1) Entropy H(π) — the SAC objective maximises reward plus policy entropy; the A2C/A3C
            entropy bonus prevents premature policy collapse; PPO includes an entropy bonus term.
            (2) KL divergence D_KL(π_new||π_old) — the constraint in TRPO; approximated by the
            clip in PPO; the penalty in RLHF/DPO.
            (3) Cross-entropy — the policy loss in REINFORCE and PPO is the cross-entropy between
            the desired distribution and the policy distribution. Understanding these quantities
            makes algorithm formulas readable rather than opaque."""), unsafe_allow_html=True)

        st.subheader("1. Entropy — Measuring Uncertainty / Diversity")
        st.markdown(r"""
        **Definition:** Entropy measures how uncertain or spread-out a distribution is.
        For a discrete distribution $P$ over $K$ outcomes:
        """)
        st.latex(r"H(P) = -\sum_{k=1}^K P(k)\log P(k) \quad\text{(units: bits if log₂, nats if ln)}")
        st.markdown(r"""
        **Where it comes from:** The optimal code length for outcome $k$ is $-\log P(k)$.
        Entropy is the expected code length — the minimum average bits to encode a random outcome.

        **Key properties:**
        - $H(P) \geq 0$ always
        - $H(P) = 0$ if one outcome has probability 1 (deterministic — no uncertainty)
        - $H(P) = \log K$ if all outcomes are equally likely (maximum entropy — maximum uncertainty)
        - In RL: $H(\pi(\cdot|s)) = -\sum_a \pi(a|s)\log\pi(a|s)$ — how random the policy is

        **Why we maximise entropy in SAC:** Higher entropy → more exploratory policy →
        less likely to get stuck in local optima. The SAC objective $r + \alpha H(\pi)$ explicitly
        rewards the policy for being uncertain (keeping options open).
        """)
        # Entropy visualisation
        p = np.linspace(0.001, 0.999, 200)
        H = -(p*np.log(p) + (1-p)*np.log(1-p))
        fig_ent, ax_ent = _fig(1, 1, 9, 3.5)
        ax_ent.plot(p, H, color="#ad1457", lw=2.5)
        ax_ent.axvline(0.5, color="#4caf50", ls="--", lw=1.5, label="Max entropy at p=0.5")
        ax_ent.fill_between(p, 0, H, alpha=0.15, color="#ad1457")
        ax_ent.set_xlabel("P(action=0)", color="white"); ax_ent.set_ylabel("H(π) [nats]", color="white")
        ax_ent.set_title("Binary policy entropy — maximised when actions are equally likely",
                         color="white", fontweight="bold")
        ax_ent.legend(facecolor=CARD, labelcolor="white"); ax_ent.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ent); plt.close()

        st.subheader("2. KL Divergence — Distance Between Distributions")
        st.markdown(r"""
        **Definition:** KL divergence measures how much distribution $Q$ differs from distribution $P$:
        """)
        st.latex(r"D_\text{KL}(P\|Q) = \sum_k P(k)\log\frac{P(k)}{Q(k)} = \mathbb{E}_{x\sim P}\!\left[\log\frac{P(x)}{Q(x)}\right]")
        st.markdown(r"""
        **Key properties:**
        - $D_\text{KL}(P\|Q) \geq 0$ always (Gibbs' inequality)
        - $D_\text{KL}(P\|Q) = 0$ if and only if $P = Q$
        - **Not symmetric:** $D_\text{KL}(P\|Q) \neq D_\text{KL}(Q\|P)$ in general

        **In RL algorithms:**
        - TRPO constraint: $D_\text{KL}(\pi_\text{new}\|\pi_\text{old}) \leq \delta$ — trust region
        - PPO: $D_\text{KL}$ implicitly bounded by the clip $\varepsilon$
        - RLHF: $\beta D_\text{KL}(\pi_\theta\|\pi_\text{ref})$ — prevents reward hacking
        - SAC: $D_\text{KL}(\pi\|e^{Q/Z})$ — SAC policy update minimises this KL
        """)

        st.subheader("3. Cross-Entropy — The Policy Training Loss")
        st.markdown(r"""
        **Definition:** Cross-entropy between target distribution $P$ and predicted distribution $Q$:
        """)
        st.latex(r"H(P, Q) = -\sum_k P(k)\log Q(k) = H(P) + D_\text{KL}(P\|Q)")
        st.markdown(r"""
        **In RL — the policy gradient loss is cross-entropy:**
        The REINFORCE/PPO loss $-\log\pi_\theta(a|s)$ is the cross-entropy between the
        one-hot distribution over action $a$ and the policy $\pi_\theta$.
        Minimising cross-entropy is equivalent to maximising log-probability of the taken action.

        **Practical example:** Policy predicts [0.3, 0.5, 0.2] for 3 actions. Agent took action 1 (index).
        """)
        st.code("""
probs = np.array([0.3, 0.5, 0.2])  # policy predictions
a = 1  # taken action (one-hot: [0,1,0])

# Cross-entropy loss (policy gradient uses this with reward weighting)
loss = -np.log(probs[a])   # = -log(0.5) = 0.693
# This is the "how surprised were we by the taken action" metric
# Gradient descent on this loss increases probs[a]
""", language="python")

    # ── PYTHON & NUMPY ────────────────────────────────────────────────────
    with tab_py:
        _sec("🐍", "Python & NumPy — The Implementation Layer",
             "Vectorised operations, broadcasting, arrays — everything the portal simulations use", "#0288d1")

        st.markdown(_card("#0288d1", "🐍", "Why NumPy mastery matters for RL implementation",
            """All the algorithms in this portal are implemented in NumPy — no PyTorch, no TensorFlow.
            This is intentional: if you can implement REINFORCE or Q-learning in NumPy, you understand
            the algorithm. PyTorch autograd and GPU acceleration are engineering tools built on top of
            the same mathematical operations. Understanding NumPy broadcasting, vectorised operations,
            and array indexing makes you able to: read other people's RL implementations, debug gradient
            issues at the numerical level, and implement new ideas before using a deep learning framework.
            PyTorch tensors behave almost identically to NumPy arrays — the API is nearly identical."""), unsafe_allow_html=True)

        st.subheader("1. Arrays — The Core Data Structure")
        st.code("""
import numpy as np

# Creating arrays
s = np.array([0.02, -0.01, 0.04, 0.03])  # 1D: CartPole state, shape (4,)
W = np.random.randn(32, 4)               # 2D: weight matrix, shape (32, 4)
Q = np.zeros((100, 4))                   # 2D: Q-table, shape (100, 4)
replay = np.zeros((10000, 4))            # 2D: replay buffer states

# Key attributes
print(s.shape)   # (4,)  — dimensions
print(W.dtype)   # float64
print(s.ndim)    # 1  — number of dimensions

# Reshape (critical for batching)
s_batch = s.reshape(1, -1)  # (4,) → (1, 4): add batch dimension
s_flat = W.reshape(-1)       # (32, 4) → (128,): flatten
""", language="python")

        st.subheader("2. Vectorised Operations — Why Loops Are Avoided")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Slow (explicit loop):**")
            st.code("""
# Q-value update for N=1000 states
results = []
for i in range(1000):
    results.append(Q[i] + alpha * (r[i] + gamma*Q_next[i] - Q[i]))
# ~1ms per loop = 1 second total
""", language="python")
        with col2:
            st.markdown("**Fast (vectorised):**")
            st.code("""
# Same operation, vectorised — 1000x faster
Q = Q + alpha * (r + gamma*Q_next - Q)
# All 1000 states updated in one GPU operation
# Runs in microseconds
""", language="python")

        st.subheader("3. Broadcasting — Operating on Different-Shaped Arrays")
        st.code("""
# Broadcasting rules: dimensions are aligned right-to-left
# Missing dimensions are expanded (broadcast) to match

states = np.random.randn(64, 4)   # batch of 64 states, shape (64, 4)
W      = np.random.randn(32, 4)   # weight matrix, shape (32, 4)

# Matrix multiplication: (64,4) @ (4,32) = (64,32)
h = states @ W.T + np.zeros(32)   # biases broadcast over batch

# Advantage normalisation (used in PPO)
advantages = np.random.randn(512)  # shape (512,)
advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
# mean() and std() collapse to scalars, which broadcast over all 512 elements
""", language="python")

        st.subheader("4. Common RL NumPy Patterns")
        st.code("""
import numpy as np

# 1. Softmax — converts logits to probabilities
def softmax(x):
    e = np.exp(x - x.max())  # subtract max for numerical stability
    return e / e.sum()

# 2. Sample from categorical distribution
probs = softmax(np.array([1.0, 2.0, 0.5]))
action = np.random.choice(3, p=probs)

# 3. Compute discounted returns (reward-to-go) — used in REINFORCE
def discounted_returns(rewards, gamma=0.99):
    G = 0.0; returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns)

# 4. Epsilon-greedy action selection
def eps_greedy(Q_values, eps=0.1):
    if np.random.rand() < eps:
        return np.random.randint(len(Q_values))
    return np.argmax(Q_values)

# 5. Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.buf = np.zeros((capacity, obs_dim*2 + 3))  # s,a,r,s',done
        self.ptr = self.size = 0; self.cap = capacity
    def add(self, s, a, r, s2, done):
        self.buf[self.ptr] = np.concatenate([s,[a,r],s2,[done]])
        self.ptr = (self.ptr+1) % self.cap; self.size = min(self.size+1, self.cap)
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.buf[idx]
""", language="python")

    # ── NEURAL NETWORK MATH ──────────────────────────────────────────────
    with tab_nn:
        _sec("🧠", "Neural Network Math — From Equations to Code",
             "Activations, forward pass, backpropagation from scratch — understand before using PyTorch", "#558b2f")

        st.markdown(_card("#558b2f", "🧠", "Why implement backprop once before using autograd",
            """PyTorch and TensorFlow compute gradients automatically via autograd. This is incredibly
            convenient but hides what is actually happening. If you have never implemented backpropagation
            manually, you will struggle to: debug gradient issues (exploding, vanishing, wrong shapes),
            understand why certain architectural choices (residual connections, layer norm) help,
            implement custom loss functions for new RL algorithms, and read theoretical RL papers
            that give gradient expressions explicitly. Implement it once in NumPy (this section does it
            for a 2-layer policy network), then use PyTorch for everything else. The effort is
            approximately 2 hours and the payoff is permanent clarity."""), unsafe_allow_html=True)

        st.subheader("1. Activation Functions — Why Non-Linearity Matters")
        st.markdown(r"""
        Without activation functions, a neural network is just one big matrix multiplication —
        it can only learn linear functions. Activation functions introduce non-linearity:
        """)
        x_act = np.linspace(-4, 4, 200)
        fig_act, axes_act = _fig(1, 4, 16, 3.5)
        acts = [
            ("ReLU", np.maximum(0, x_act), "max(0,x)", "#7c4dff", "Most common hidden layers; no vanishing gradient for x>0"),
            ("Sigmoid", 1/(1+np.exp(-x_act)), "1/(1+e^-x)", "#0288d1", "Output layer for probabilities; gates in LSTMs"),
            ("Tanh", np.tanh(x_act), "tanh(x)", "#00897b", "Output in [-1,1]; better than sigmoid for hidden layers"),
            ("GELU", x_act * (1 + np.vectorize(lambda x: __import__('math').erf(x/np.sqrt(2)))(x_act))/2,
             "x·Φ(x)", "#e65100", "Used in Transformers (BERT, GPT); smooth ReLU variant"),
        ]
        for ax, (name, vals, form, col, desc) in zip(axes_act, acts):
            ax.plot(x_act, vals, color=col, lw=2.5)
            ax.set_title(f"{name}\n{form}", color="white", fontsize=8, fontweight="bold")
            ax.axhline(0, color="#2a2a3e"); ax.axvline(0, color="#2a2a3e")
            ax.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_act); plt.close()

        st.subheader("2. Forward Pass — Computing the Network Output")
        st.code("""
import numpy as np

class TwoLayerPolicy:
    \"\"\"2-layer policy network for CartPole (4 inputs → 2 action probs)\"\"\"
    def __init__(self, in_dim=4, hid=32, out_dim=2, seed=0):
        np.random.seed(seed)
        # He initialisation: variance = 2/fan_in (good for ReLU)
        self.W1 = np.random.randn(in_dim, hid) * np.sqrt(2/in_dim)
        self.b1 = np.zeros(hid)
        self.W2 = np.random.randn(hid, out_dim) * np.sqrt(2/hid)
        self.b2 = np.zeros(out_dim)
        # Cache for backward pass
        self.cache = {}

    def forward(self, x):
        # Layer 1: linear + ReLU
        z1 = x @ self.W1 + self.b1  # shape (hid,)
        h1 = np.maximum(0, z1)       # ReLU: element-wise max(0, z1)

        # Layer 2: linear + softmax
        z2 = h1 @ self.W2 + self.b2  # shape (out_dim,)
        e = np.exp(z2 - z2.max())    # numerical stability
        probs = e / e.sum()           # softmax probabilities

        # Cache intermediates for backprop
        self.cache = {'x': x, 'z1': z1, 'h1': h1, 'z2': z2, 'probs': probs}
        return probs
""", language="python")

        st.subheader("3. Backpropagation — Chain Rule Applied")
        st.code("""
    def backward(self, action, advantage):
        \"\"\"
        Compute gradients via backpropagation.
        Loss = -log(probs[action]) * advantage  (REINFORCE policy gradient)
        Goal: compute d_Loss/d_W1, d_Loss/d_b1, d_Loss/d_W2, d_Loss/d_b2
        \"\"\"
        x, z1, h1, z2, probs = [self.cache[k] for k in ['x','z1','h1','z2','probs']]

        # --- Backward through softmax + log + advantage weighting ---
        # d(Loss)/d(z2): gradient of -log(probs[a])*adv w.r.t. z2
        # Softmax-CE gradient: probs - one_hot(action), scaled by -advantage
        d_z2 = probs.copy()
        d_z2[action] -= 1.0      # ∂(-log p_a)/∂logits = p - 1_{a}
        d_z2 *= -advantage        # policy gradient: maximise → negate for descent

        # --- Backward through Layer 2: z2 = h1 @ W2 + b2 ---
        d_W2 = np.outer(h1, d_z2)   # ∂z2/∂W2 = h1 (outer product)
        d_b2 = d_z2                  # ∂z2/∂b2 = 1
        d_h1 = self.W2 @ d_z2       # ∂z2/∂h1 = W2^T @ d_z2

        # --- Backward through ReLU: h1 = max(0, z1) ---
        d_z1 = d_h1 * (z1 > 0)      # ∂relu/∂z1 = 1 if z1>0 else 0

        # --- Backward through Layer 1: z1 = x @ W1 + b1 ---
        d_W1 = np.outer(x, d_z1)    # ∂z1/∂W1 = x
        d_b1 = d_z1                  # ∂z1/∂b1 = 1

        return d_W1, d_b1, d_W2, d_b2

    def update(self, grads, lr=0.01):
        d_W1, d_b1, d_W2, d_b2 = grads
        # Gradient DESCENT on the loss = gradient ASCENT on J(θ)
        self.W1 -= lr * d_W1   # (loss was negated above, so this is +∇J)
        self.b1 -= lr * d_b1
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
""", language="python")

        st.subheader("4. Weight Initialisation — Why It Matters")
        st.markdown(r"""
        Poor initialisation causes gradients to vanish (all activations → 0) or explode (all → ∞).
        **He initialisation** (for ReLU networks):
        """)
        st.latex(r"W \sim \mathcal{N}\!\left(0,\;\frac{2}{n_\text{in}}\right) \quad\text{(fan-in: number of inputs to each neuron)}")
        st.markdown(r"""
        **Why 2/n_in?** With ReLU, approximately half the neurons are zeroed out (negative inputs).
        So the effective variance is halved. Multiplying by 2 compensates, keeping the variance of
        activations roughly constant through layers. Without this, deep networks fail to train.

        **Xavier initialisation** (for tanh/sigmoid):
        """)
        st.latex(r"W \sim \mathcal{N}\!\left(0,\;\frac{2}{n_\text{in}+n_\text{out}}\right) \quad\text{(harmonic mean of fan-in and fan-out)}")

    # ── SELF-ASSESSMENT ───────────────────────────────────────────────────
    with tab_test:
        _sec("✅", "Self-Assessment — Are You Ready?",
             "Check your understanding before moving to Deep Learning Prerequisites", "#ffa726")

        st.markdown("Answer these questions mentally. If you can answer all of them confidently, proceed to Stage 0.")

        questions = [
            ("🔢 Linear Algebra", [
                ("What is the result of a 3×4 matrix multiplied by a 4×1 vector? What shape?", "3×1 vector. Dimensions: (rows of A) × (cols of B). Each element of result = dot product of one row of matrix with the vector."),
                ("Why can't we matrix-multiply a (4,3) and a (4,3) matrix directly?", "Inner dimensions must match: (4,3)@(4,3) fails because 3≠4. We'd need (4,3)@(3,4) = (4,4), or transpose one: (4,3)@(3,4)."),
                ("What does the transpose of a matrix do?", "Swaps rows and columns. (m,n) → (n,m). Used in backprop: gradient of Wx is W^T times output gradient."),
            ]),
            ("📈 Calculus", [
                ("What is the derivative of sigmoid σ(x) = 1/(1+e^(-x))?", "σ'(x) = σ(x)(1-σ(x)). Derivation: apply quotient rule to 1/(1+e^(-x))."),
                ("If L = (y - ŷ)², what is ∂L/∂ŷ?", "∂L/∂ŷ = -2(y-ŷ) = 2(ŷ-y). Chain rule: ∂L/∂ŷ = 2(ŷ-y)·1."),
                ("In RL, why do we do gradient ASCENT not descent?", "We maximise reward J(θ), not minimise. So θ ← θ + α∇J. Loss functions are minimised (descent), reward functions are maximised (ascent)."),
            ]),
            ("🎲 Probability", [
                ("What is E[G_t] if G_t is uniformly distributed between 0 and 100?", "50. For Uniform[a,b]: E[X] = (a+b)/2 = (0+100)/2 = 50."),
                ("Why does the baseline b(s) not change the policy gradient expectation?", "E[∇log π · b(s)] = b(s)·∇E[π] = b(s)·∇1 = 0, because probabilities always sum to 1."),
                ("What is conditional probability P(a|s) and why is it the right model for policies?", "P(a|s) = probability of action a given state s. Policy is conditional because the right action depends on the current state — π(a|s) gives a distribution over actions for each specific state."),
            ]),
            ("📡 Information Theory", [
                ("What is the entropy of a fair coin? Of a biased coin with P(H)=0.99?", "Fair: H = -0.5·log(0.5) - 0.5·log(0.5) = log(2) ≈ 0.693 nats. Biased: H = -0.99·log(0.99) - 0.01·log(0.01) ≈ 0.056 nats."),
                ("Why is D_KL(P||Q) ≥ 0?", "Jensen's inequality: E_P[log(Q/P)] ≤ log(E_P[Q/P]) = log(1) = 0. So -E_P[log(Q/P)] ≥ 0, i.e., D_KL(P||Q) ≥ 0."),
                ("If PPO clips the probability ratio r_t(θ) to [1-ε, 1+ε], what KL divergence does this roughly enforce?", "KL ≤ ε approximately, since for small ε: D_KL ≈ (ratio-1)²/2 ≈ ε²/2. PPO's clip is a first-order approximation of TRPO's KL constraint."),
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
                                f'<b style="color:#4caf50">Answer:</b> '
                                f'<span style="color:#b0b0cc">{ans}</span></div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📚 Best Resources to Study These Foundations")
        resources = [
            ("🎥", "3Blue1Brown — Essence of Linear Algebra", "15 videos, each 5–20 min. Visual-first — builds geometric intuition before formulas.", "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"),
            ("🎥", "3Blue1Brown — Essence of Calculus", "12 videos. Chain rule, gradients, and the fundamental theorem. Required watching.", "https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr"),
            ("📄", "Michael Nielsen — Neural Networks and Deep Learning", "Free online book. Chapter 2 derives backpropagation from first principles in full.", "http://neuralnetworksanddeeplearning.com"),
            ("💻", "Andrej Karpathy — micrograd", "170 lines of Python implementing autograd from scratch. Read it, then reimplement it.", "https://github.com/karpathy/micrograd"),
            ("💻", "NumPy Official Tutorial", "numpy.org/doc/stable/user/quickstart.html — arrays, broadcasting, vectorisation.", "https://numpy.org/doc/stable/user/quickstart.html"),
            ("📄", "Mathematics for Machine Learning — Deisenroth et al.", "Free PDF. Covers all six topics in depth with ML context throughout.", "https://mml-book.github.io"),
            ("📄", "Dive into Deep Learning — Chapters 1–4", "Interactive Jupyter notebooks. Linear algebra, autograd, linear networks, MLPs.", "https://d2l.ai"),
        ]
        for icon, title, desc, url in resources:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
