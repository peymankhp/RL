"""
_frontier_mod.py — Frontier RL Research (Tier 4 — Full Edition 2025)
12 tabs: Overview | RLHF at Scale | World Models+RL | Exploration (Large)
         Safe RL (Formal) | Foundation Models | Offline→Online RL
         LLM+RL Basics | Sim-to-Real | Diffusion RL | RL Theory | Roadmap
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


def _warn(text):
    return (f'<div style="background:#2a1a0a;border-left:3px solid #ffa726;'
            f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem">'
            f'⚠️ {text}</div>')


def _sec(emoji, title, sub, color="#ad1457"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)


def smooth(a, w=8):
    return np.convolve(a, np.ones(w) / w, mode="valid") if len(a) > w else np.array(a, float)


def main_frontier():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#2a0a1a,#0a0a2a,#1a0a2a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🔬 Frontier RL Research (2025)</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'Six open problems defining the next decade of RL — each covered with full mathematical '
        'derivations from first principles, practical examples, and the latest research directions.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🗺️ Overview",
        "🧠 RLHF at Scale",
        "🌍 World Models + RL",
        "🔍 Exploration (Large Spaces)",
        "🛡️ Safe RL (Formal Guarantees)",
        "🌐 Foundation Models for RL",
        "📐 Offline → Online RL",
        "💬 LLM + RL Basics",
        "🤖 Sim-to-Real",
        "🌊 Diffusion RL",
        "📏 RL Theory",
        "🗺️ Roadmap",
    ])
    (tab_ov, tab_rlhf, tab_wm, tab_exp,
     tab_safe, tab_fm, tab_o2o, tab_llm,
     tab_s2r, tab_diff, tab_theory, tab_road) = tabs

    # ── OVERVIEW ─────────────────────────────────────────────────────────
    with tab_ov:
        _sec("🗺️", "Six Open Problems at the Frontier",
             "What is still unsolved after mastering all the core RL algorithms", "#ad1457")
        st.markdown(_card("#ad1457", "🔬", "Why these six?",
            """After mastering DQN, PPO, SAC, CQL and the rest, a practitioner can solve most
            benchmark tasks. But five failure modes appear immediately in real-world deployment:
            (1) RLHF breaks when the AI exceeds the evaluator's capability — how do you align a
            system smarter than you? (2) World models work in simulation but compound errors destroy
            contact-rich robotics — why, and what is missing? (3) UCB achieves O(log T) regret in
            discrete bandits but no algorithm achieves this in continuous spaces — a fundamental
            open problem. (4) Safe RL guarantees constraint satisfaction at convergence but not
            during training — catastrophic for real deployment. (5) Every RL system is task-specific;
            true foundation models for RL remain unsolved. (6) Offline-to-online fine-tuning degrades
            performance due to Q-function miscalibration — no principled solution exists yet.
            These six are the bottlenecks preventing RL from being deployed widely. Each tab covers
            one problem with full mathematical treatment and concrete practical examples."""),
            unsafe_allow_html=True)

        st.dataframe(pd.DataFrame({
            "Problem": ["RLHF at Scale", "World Models + RL",
                        "Exploration (Large Spaces)", "Safe RL (Formal Guarantees)",
                        "Foundation Models for RL", "Offline → Online RL"],
            "Status": ["Active", "Active", "Open (unsolved)", "Active", "Emerging", "Active"],
            "Why hard": [
                "Humans cannot evaluate superhuman outputs",
                "1% per-step errors compound to 16% over 15 steps",
                "Lower bound: T^(2/3) regret in continuous MDPs",
                "Asymptotic guarantees fail during training itself",
                "In-context RL still weaker than fine-tuning",
                "Q-function miscalibration breaks conservative policies",
            ],
            "Best current approach": [
                "Constitutional AI + Debate + Scalable Oversight",
                "Latent RSSM + uncertainty estimation + short horizons",
                "PSRL + neural density models + ensemble disagreement",
                "CBF shielding + CPO + Lagrangian primal-dual",
                "Gato, RT-2, Algorithm Distillation, DPT",
                "Cal-QL + IQL→online + mixed replay buffer",
            ],
        }), use_container_width=True, hide_index=True)

    # ── RLHF AT SCALE ────────────────────────────────────────────────────
    with tab_rlhf:
        _sec("🧠", "RLHF at Scale — Aligning Superhuman AI",
             "Scalable oversight, debate, Constitutional AI — when the AI surpasses its evaluators", "#e65100")

        st.markdown(_card("#e65100", "🧠", "The evaluator collapse problem",
            """Standard RLHF asks humans to compare two AI responses and label which is better.
            This works for tasks humans understand (write a poem, summarise a document). But when
            the AI produces a novel mathematical proof, a security exploit analysis, or a long-horizon
            strategic plan that the human evaluator cannot actually verify — the human says 'response A
            looks more confident and well-written' without knowing if the math is correct. The reward
            model learns to predict which responses look good to non-experts, not which responses are
            actually correct. The resulting AI is optimised to appear correct rather than be correct.
            This is evaluator collapse — the central challenge of aligning AI systems that exceed human
            performance in specific domains. Three approaches are actively researched: Constitutional AI
            (use AI to evaluate against written principles), Debate (two AI systems argue and humans
            judge the argument, which is easier than judging the original question), and Scalable
            Oversight (humans use AI assistance to extend their evaluation capacity). All are partial
            solutions; the problem remains fundamentally open for true superhuman capability levels."""),
            unsafe_allow_html=True)

        st.subheader("1. Standard RLHF — The Baseline")
        st.markdown(r"""
        The Bradley-Terry model: probability that response $A$ is preferred over $B$
        is a sigmoid of their reward difference. **Derivation from first principles:**
        Assume preferences come from a latent utility model $P(A\succ B) = \sigma(r(A)-r(B))$.
        The log-likelihood of observed labels $(y_w \succ y_l)$ maximised over $r_\phi$:
        """)
        st.latex(r"\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)\right]")
        st.markdown(r"The optimal RLHF policy under KL constraint can be shown to satisfy:")
        st.latex(r"\pi^*(y|x) \propto \pi_\text{ref}(y|x)\cdot\exp\!\left(\frac{r(x,y)}{\beta}\right)")
        st.markdown(r"PPO finds this policy by gradient ascent on:")
        st.latex(r"J(\theta) = \mathbb{E}_{y\sim\pi_\theta}\!\left[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]")
        st.markdown(_warn("""
        When r_φ learns to predict 'looks correct' rather than 'is correct', PPO produces
        confidently wrong, authoritative-sounding outputs. This is reward hacking at scale.
        """), unsafe_allow_html=True)

        st.subheader("2. Constitutional AI — AI Evaluates Against Written Principles")
        st.markdown(r"""
        **The key insight (Bai et al. 2022):** Instead of human labels for every preference pair,
        use the AI to critique its own outputs against a written constitution of principles,
        then generate AI preference labels (RLAIF = RL from AI Feedback).

        **Stage 1 — Supervised critique and revision:**
        For each harmful or flawed response $y$ and principle $p$:
        """)
        st.latex(r"\text{critique} = \pi_\text{SL}(x, y, p) \quad\text{(AI critiques using principle }p\text{)}")
        st.latex(r"y' = \pi_\text{SL}(x, y, \text{critique}) \quad\text{(AI revises based on critique)}")
        st.markdown(r"Repeat for all principles. Fine-tune on revised outputs (SL-CAI stage).")
        st.markdown(r"**Stage 2 — AI feedback for reward model (RLAIF):**")
        st.latex(r"P(y_w \succ y_l | x, p) = \sigma\!\bigl(\text{score}_\pi(x,y_w,p) - \text{score}_\pi(x,y_l,p)\bigr)")
        st.markdown(r"""
        AI generates millions of preference pairs using the constitution — no human annotation per pair.
        Only the constitution itself requires human authorship.

        **Practical example:** Principle: "prefer responses that acknowledge uncertainty rather than
        state incorrect facts confidently". The model generates response A ("The answer is definitely X")
        and response B ("I believe X, but this is uncertain because..."). It applies the principle
        and labels B as preferred. This preference label trains the reward model.
        CAI reduced Anthropic's per-pair human annotation requirement by ~10× while maintaining alignment.
        """)

        st.subheader("3. Debate — Argument Verification Is Easier Than Answer Verification")
        st.markdown(r"""
        **The core idea (Irving et al. 2018):** Two AI debaters argue opposite positions.
        A human judges the debate. The Nash equilibrium of this game is honest debate —
        because any false claim can be exposed by the opponent, and the human can verify
        the exposure even if they cannot verify the original claim.

        **Formal setup:** Debaters $D_1$, $D_2$ make argument sequences $a_1, a_2 \in \mathcal{A}^*$.
        Human judge: $J(q, a_1, a_2) \in \{+1, -1\}$. Optimal strategy:
        """)
        st.latex(r"D_1^* = \arg\max_{D_1}\min_{D_2}\,\mathbb{E}[J(q, D_1(q), D_2(q))]")
        st.markdown(r"""
        **Why honest debate is the Nash equilibrium:** If $D_1$ makes a false claim,
        $D_2$ can expose it. The human sees the exposure and can verify "yes, that claim was wrong"
        even without solving the original problem. An honest $D_2$ will always expose false claims.
        Therefore $D_1$'s best strategy against an honest $D_2$ is to make only true claims.

        **Current limitation:** The equilibrium holds only if both debaters are capable enough
        to expose all deceptions. Against a much stronger system, a weaker debater may not
        catch subtle lies. This remains an active research question.
        """)

        st.subheader("4. Scalable Oversight — AI-Assisted Human Evaluation")
        st.markdown(r"""
        **Iterated Amplification (Christiano et al. 2018):** Use a weaker assistant model
        to help humans evaluate a stronger model's outputs. This bootstraps oversight:
        """)
        st.latex(r"\text{Oversight}(H, x) = H\bigl(x, \text{AssistantModel}(x, y_1, y_2)\bigr)")
        st.markdown(r"""
        At each capability level $n$: use level-$(n-1)$ model to assist evaluating level-$n$ model.
        This creates a hierarchy where oversight capacity grows with AI capability.

        **Practical example:** To verify a 100-page mathematical proof:
        - Human alone: too hard to verify all steps
        - With AI assistant that can spot potential errors and highlight suspicious steps:
          human can focus evaluation effort on flagged regions → verification becomes tractable
        """)

        # Visualise oversight scaling
        np.random.seed(42)
        cap = np.linspace(0, 10, 100)
        fig_so, ax_so = _fig(1, 1, 11, 4)
        ax_so.plot(cap, np.minimum(3, 1.5 + 0.2*cap), color="#ffa726", lw=2.5,
                   label="Human alone (caps ~3)")
        ax_so.plot(cap, np.minimum(7, 0.5 + 0.65*cap), color="#0288d1", lw=2.5,
                   label="Human + AI assist (scalable oversight)")
        ax_so.plot(cap, np.minimum(9, 1.0 + 0.8*cap), color="#4caf50", lw=2.5,
                   label="CAI + Debate (further extension)")
        ax_so.axvline(3, color="#ffa726", ls=":", lw=1.5, alpha=0.7)
        ax_so.text(3.2, 0.5, "Human\ncap", color="#ffa726", fontsize=7)
        ax_so.set_xlabel("AI capability level", color="white")
        ax_so.set_ylabel("Max evaluable capability", color="white")
        ax_so.set_title("Scalable Oversight: Extending Human Evaluation Capacity",
                        color="white", fontweight="bold")
        ax_so.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_so.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_so); plt.close()

    # ── WORLD MODELS + RL ────────────────────────────────────────────────
    with tab_wm:
        _sec("🌍", "World Models + RL — Scaling to Real Robotics",
             "DreamerV3 works in games — why not real robots? The compounding error problem", "#00897b")

        st.markdown(_card("#00897b", "🌍", "Why world models have not conquered real robotics",
            """DreamerV3 solves Minecraft, 150+ Atari games, and continuous control with fixed
            hyperparameters. But translating this to unstructured real robotics remains unsolved.
            The core problem is compounding model error: a world model trained on real robot data
            has prediction error ε per step. Over a 15-step imagination horizon (DreamerV3 default),
            error compounds to (1+ε)^15. For real robotics with contact dynamics (grasping, manipulation),
            even 1% per-step errors in contact force prediction cause the agent to plan actions that
            fail catastrophically in reality. Additionally, real sensors have noise, latency, and
            occlusion that simulation never models accurately. Three questions define this frontier:
            (1) How to build world models accurate enough for contact-rich robotics?
            (2) How to detect when the model is wrong and fall back to model-free behaviour?
            (3) How to transfer world model knowledge from simulation to real hardware?"""),
            unsafe_allow_html=True)

        st.subheader("1. RSSM — The DreamerV3 Architecture")
        st.markdown(r"""
        **Deriving the Recurrent State Space Model from the latent variable objective:**
        We want a latent state $z_t$ that captures all information needed to predict future
        observations and rewards. The variational evidence lower bound (ELBO) for the
        sequence model gives us the objective:
        """)
        st.latex(r"\text{ELBO} = \underbrace{\mathbb{E}_q[\log p(o_t|z_t,h_t)]}_\text{reconstruction} - \underbrace{D_\text{KL}(q(z_t|h_t,o_t)\,\|\,p(z_t|h_t))}_\text{regularisation}")
        st.markdown(r"""
        The RSSM splits the latent state into deterministic and stochastic components:
        """)
        st.latex(r"h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1}) \quad\text{(GRU — handles long-range memory)}")
        st.latex(r"z_t \sim q_\phi(z_t|h_t,o_t) \quad\text{(posterior: use real obs during training)}")
        st.latex(r"\hat{z}_t \sim p_\phi(\hat{z}_t|h_t) \quad\text{(prior: use for imagination — no obs needed)}")
        st.markdown(r"""
        **Why the split?** Deterministic $h_t$ handles long-range temporal dependencies efficiently.
        Stochastic $z_t$ models irreducible uncertainty (random environment events).
        During training: posterior $q_\phi$ gets the benefit of seeing real observations.
        During imagination: prior $p_\phi$ generates rollouts without any real data needed.

        **Full DreamerV3 world model loss:**
        """)
        st.latex(r"\mathcal{L}_\text{WM} = \mathbb{E}\!\left[\log p(o_t|h_t,z_t) + \log p(r_t|h_t,z_t) + \log p(d_t|h_t,z_t) - \beta D_\text{KL}(q\|p)\right]")
        st.markdown(r"""
        **Symlog transform (DreamerV3 innovation):** All predictions use $\text{symlog}(x)=\text{sign}(x)\log(|x|+1)$.
        This normalises targets across environments with wildly different reward scales (−1000 to +1000)
        without manual reward clipping — the first algorithm to use truly fixed hyperparameters across
        all domains.
        """)

        st.subheader("2. The Compounding Error Problem")
        np.random.seed(42)
        H = np.arange(1, 51)
        fig_comp, ax_comp = _fig(1, 1, 11, 4)
        for eps, col, lbl in [(0.001, "#4caf50", "ε=0.1%/step (sim)"),
                               (0.01,  "#ffa726", "ε=1%/step (real robot)"),
                               (0.05,  "#ef5350", "ε=5%/step (hard contacts)")]:
            ax_comp.plot(H, [(1+eps)**h - 1 for h in H], color=col, lw=2.5, label=lbl)
        ax_comp.axvline(15, color="white", ls="--", lw=1.2, alpha=0.6, label="DreamerV3 H=15")
        ax_comp.axhline(0.2, color="#b0b0cc", ls=":", lw=1, alpha=0.5)
        ax_comp.text(48, 0.21, "20% error", color="#b0b0cc", fontsize=7, ha="right")
        ax_comp.set_xlabel("Imagination steps H", color="white")
        ax_comp.set_ylabel("Cumulative error (1+ε)^H − 1", color="white")
        ax_comp.set_title("Compounding Error: Why Short Horizons Matter for Real Robotics",
                          color="white", fontweight="bold")
        ax_comp.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_comp.grid(alpha=0.12); ax_comp.set_ylim(0, 1.5)
        plt.tight_layout(); st.pyplot(fig_comp); plt.close()

        st.subheader("3. Uncertainty-Aware Planning")
        st.markdown(r"""
        **Epistemic uncertainty estimation with ensembles:**
        Train $M$ world models $\{f_{\phi_m}\}_{m=1}^M$ with different initialisations.
        Epistemic uncertainty at $(s,a)$:
        """)
        st.latex(r"\sigma^2_\text{epist}(s,a) = \frac{1}{M}\sum_{m=1}^M(\hat\mu_m(s,a)-\bar\mu(s,a))^2")
        st.markdown(r"""
        **Decision rule:** If $\sigma^2_\text{epist}(s,a) > \tau$ (threshold), do NOT use the world
        model — take a real environment step instead. This prevents the agent from planning
        into regions where the model is unreliable. Practical implementation: DreamerV3 with
        an ensemble of 5 world models; threshold tuned to 95th percentile of training uncertainty.
        """)
        st.markdown(_insight("""
        <b>The key open question:</b> What is the minimum world model accuracy required for
        contact-rich manipulation? Current estimates suggest ε < 0.05% per step for reliable grasping —
        far below what current neural world models achieve on real data. Closing this gap requires
        either better physics-informed architectures or much larger training datasets from real robots.
        """), unsafe_allow_html=True)

    # ── EXPLORATION IN LARGE SPACES ──────────────────────────────────────
    with tab_exp:
        _sec("🔍", "Exploration in Large State Spaces",
             "No algorithm achieves O(log T) regret in continuous high-dimensional MDPs — here is why", "#7c4dff")

        st.markdown(_card("#7c4dff", "🔍", "Why the bandit solution does not scale",
            """UCB achieves O(log T) regret in K-armed bandits by maintaining per-arm confidence bounds.
            In tabular MDPs, UCRL2 achieves polynomial sample complexity by counting visits N(s,a).
            Both fail in continuous spaces: you cannot maintain a confidence bound per point in R^n
            (uncountably many states) or count visits to a continuous state (each visited exactly once).
            Neural network approximations of confidence bounds introduce errors that destroy guarantees.
            No algorithm has achieved O(log T) regret in continuous MDPs. This is not a gap in known
            techniques — there are information-theoretic lower bounds showing O(log T) may be impossible
            without strong structural assumptions. The T^(2/3) lower bound shows that continuous
            exploration is fundamentally harder than discrete. In practice: every deep RL system uses
            heuristic exploration (ε-greedy, entropy bonus, count approximations) with no guarantees.
            Montezuma's Revenge requires millions of steps before seeing a single reward. The frontier
            explores PSRL (posterior sampling), neural density models, and uncertainty-based exploration."""),
            unsafe_allow_html=True)

        st.subheader("1. Information-Theoretic Lower Bound")
        st.markdown(r"""
        **Constructing the hard instance:** Consider a 1D continuous MDP on $[0,1]$. Reward is 1 at
        an unknown point $x^*\sim\text{Uniform}[0,1]$, zero elsewhere.
        Any algorithm must essentially search the interval, yielding:
        """)
        st.latex(r"\mathbb{E}[\text{Regret}(T)] \geq c\,T^{2/3}")
        st.markdown(r"""
        **Derivation sketch:** Let $N(t)$ = distinct regions visited by step $t$.
        Cost of exploration: $\sim N(t)$ steps with zero reward.
        Probability of not finding $x^*$: $\propto 1 - N(t)/T$.
        Expected loss from missed reward: $\sim T \cdot (1-N(t)/T)$.
        Minimise total loss: $N(t) + T(1-N(t)/T)$.
        Optimal $N(t) \sim T^{1/3}$ gives total regret $\sim T^{2/3}$.
        """)
        st.markdown(_warn("""
        For T = 10 million steps: T^(2/3) ≈ 46,416 times the single-step regret.
        Compare to discrete UCB's log(T) ≈ 16. Continuous exploration is fundamentally ~3000× harder.
        This explains why exploration remains the primary bottleneck in sparse-reward deep RL.
        """), unsafe_allow_html=True)

        st.subheader("2. Posterior Sampling for RL (PSRL)")
        st.markdown(r"""
        **PSRL (Osband et al. 2013):** Maintain a Bayesian posterior over MDP dynamics.
        Each episode: sample one MDP from the posterior, act optimally in it.

        **Why it works:** With probability equal to the true MDP's posterior mass, you sample
        the correct MDP and exploit. With the remaining probability, you sample uncertain MDPs
        and explore regions that reduce posterior uncertainty. This is Thompson Sampling extended
        to full MDPs — automatically balancing exploration and exploitation.

        **Regret bound under linear function approximation:**
        """)
        st.latex(r"\mathbb{E}[\text{Regret}(T)] \leq \tilde O(d\sqrt{T}) \quad\text{(linear MDPs, dimension }d\text{)}")
        st.markdown(r"""
        Better than the worst-case $T^{2/3}$, worse than discrete $O(\log T)$.

        **Practical neural implementation (RLSVI / ensemble exploration):**
        """)
        st.latex(r"\sigma^2_\text{epistemic}(s,a) = \frac{1}{M}\sum_{m=1}^M(Q_{\theta_m}(s,a)-\bar{Q}(s,a))^2")
        st.markdown(r"""
        Each episode: sample one Q-network from the ensemble. Act greedily under it.
        High-uncertainty states have high disagreement $\sigma^2$ → sampling different networks
        leads to naturally diverse exploration across workers.
        """)

        st.subheader("3. Neural Density Models for Count-Based Bonuses")
        st.markdown(r"""
        **Count-based exploration in continuous spaces:**
        Replace the discrete count $n(s)$ with a neural density model $\hat p_\phi(s)$.
        The intrinsic bonus:
        """)
        st.latex(r"r_\text{bonus}(s) = \beta\cdot(-\log\hat p_\phi(s)) \quad\text{(high bonus for low-density states)}")
        st.markdown(r"""
        As the agent visits state $s$ more, $\hat p_\phi(s)$ increases → bonus decreases.
        Novel states have low density → high bonus. This mimics count-based exploration in continuous spaces.

        **Practical options for $\hat p_\phi$:**
        - RND (Random Network Distillation): bonus = $\|f_\theta(s) - T(s)\|^2$ where $T$ is frozen random
        - Pseudo-counts (Bellemare 2016): density model $\hat\rho$ gives pseudo-count $\hat n(s) = \hat\rho(s)/(1-\hat\rho(s))$
        - Flow-based density: exact log-likelihood from normalising flows
        """)

        # Regret scaling visualisation
        T = np.logspace(3, 7, 100)
        fig_reg, ax_reg = _fig(1, 1, 11, 4)
        ax_reg.loglog(T, np.log(T), color="#4caf50", lw=2.5, label=r"$O(\log T)$ — discrete UCB (optimal)")
        ax_reg.loglog(T, T**0.5 / 100, color="#ffa726", lw=2.5, label=r"$O(\sqrt{T})$ — PSRL in linear MDPs")
        ax_reg.loglog(T, T**(2/3) / 200, color="#ef5350", lw=2.5, label=r"$O(T^{2/3})$ — continuous lower bound")
        ax_reg.loglog(T, T*0.01, color="#546e7a", lw=2, ls="--", label="$O(T)$ — random exploration")
        ax_reg.set_xlabel("Steps T", color="white")
        ax_reg.set_ylabel("Regret", color="white")
        ax_reg.set_title("Exploration Regret Scaling: Discrete vs Continuous State Spaces",
                         color="white", fontweight="bold")
        ax_reg.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_reg.grid(alpha=0.12, which="both")
        plt.tight_layout(); st.pyplot(fig_reg); plt.close()

    # ── SAFE RL FORMAL ───────────────────────────────────────────────────
    with tab_safe:
        _sec("🛡️", "Safe RL with Formal Guarantees",
             "CBF shielding, CPO, Lagrangian — constraint satisfaction during training, not just at convergence", "#ef5350")

        st.markdown(_card("#ef5350", "🛡️", "Why asymptotic safety is not enough",
            """Standard safe RL methods (CPO, Lagrangian PPO) prove constraint satisfaction at convergence.
            But during training, the policy violates constraints thousands of times as it explores.
            For a surgical robot, autonomous vehicle, or industrial controller, even a single training-time
            violation can be catastrophic. We need algorithms that provably never violate constraints
            during training — not just eventually. The mathematical challenge: exploration requires trying
            uncertain actions, but safety requires avoiding potentially unsafe actions. These goals are
            in direct tension. Current approaches: (1) Control Barrier Functions (CBFs) — a safety filter
            that mathematically guarantees the agent stays in a safe set at every step; (2) Conservative
            exploration — only explore actions provably safe under model uncertainty; (3) Shielding —
            override any unsafe proposed action; (4) Safety via Lyapunov functions. Each makes different
            assumptions about the constraint structure."""), unsafe_allow_html=True)

        st.subheader("1. CMDP — The Formal Problem Setup")
        st.latex(r"\max_\pi J(\pi)=\mathbb{E}_\pi\!\left[\sum_{t}\gamma^t r_t\right] \quad\text{s.t.}\quad J_{C_i}(\pi)=\mathbb{E}_\pi\!\left[\sum_t\gamma^t c_t^i\right]\leq d^i \;\forall i")
        st.markdown(r"""
        **Lagrangian relaxation** (standard approach, asymptotic guarantee only):
        """)
        st.latex(r"\mathcal{L}(\pi,\lambda)=J(\pi)-\sum_i\lambda_i(J_{C_i}(\pi)-d^i)")
        st.latex(r"\lambda_i\leftarrow\max(0,\;\lambda_i+\eta_\lambda(J_{C_i}(\pi)-d^i))")
        st.markdown(_warn("""
        The Lagrangian only guarantees constraint satisfaction AT convergence.
        During training, λ_i is adjusting — the policy can violate constraints for 10,000+ steps.
        """), unsafe_allow_html=True)

        st.subheader("2. Control Barrier Functions — Hard Safety at Every Step")
        st.markdown(r"""
        **CBF definition:** Function $h: S \to \mathbb{R}$ defines safe set $\mathcal{C}=\{s:h(s)\geq 0\}$.
        The CBF condition ensures forward invariance (safe set is invariant under the system dynamics):
        """)
        st.latex(r"\dot h(s,a) + \alpha h(s) \geq 0 \quad \forall (s,a) \text{ where } s\in\mathcal{C}")
        st.markdown(r"""
        **How to use as a safety filter:**
        1. RL policy proposes action $a=\pi_\theta(s)$
        2. Check CBF condition: $\nabla_s h(s)^T f(s,a)+\alpha h(s)\geq 0$?
        3. If yes: execute $a$. If no: project to nearest safe action:
        """)
        st.latex(r"a^*=\arg\min_{a'}\|a'-a\|^2 \quad\text{s.t.}\quad \nabla_s h(s)^Tf(s,a')+\alpha h(s)\geq 0")
        st.markdown(r"""
        This QP is solved in microseconds. The RL policy learns from filtered actions,
        eventually learning to propose safe actions itself. **Guaranteed no violation at any training step.**

        **Example — robot workspace:**
        $h(s) = d_\text{max} - \|p\|$, safe set $= \|p\|\leq d_\text{max}$.
        CBF condition: $-\dot p\cdot\frac{p}{\|p\|}+\alpha(d_\text{max}-\|p\|)\geq 0$.
        Forces deceleration as arm approaches workspace boundary — mathematically guaranteed.
        """)

        st.subheader("3. CPO — Trust Region with Constraint Satisfaction")
        st.markdown(r"""
        CPO (Achiam et al. 2017) extends TRPO with constraint linearisation:
        """)
        st.latex(r"\max_\theta\;g^T(\theta-\theta_k) \quad\text{s.t.}\quad \tfrac{1}{2}(\theta-\theta_k)^TF(\theta-\theta_k)\leq\delta,\quad b_i^T(\theta-\theta_k)\leq d^i-J_{C_i}(\pi_k)")
        st.markdown(r"""
        $g=\nabla_\theta J$ (reward gradient), $F$ = Fisher information, $b_i=\nabla_\theta J_{C_i}$.
        The update lies in the intersection of the trust region sphere and constraint halfspaces.
        **Limitation:** First-order linearisation errors mean true constraints can still be violated
        by $O(\delta^2)$ — training-time safety still not fully guaranteed.
        """)

        st.markdown(_insight("""
        <b>Best current practice:</b> CBF as hard filter (guaranteed per-step safety) + CMDP/Lagrangian
        for policy learning (eventually constraint-satisfying). The CBF handles the hard guarantee;
        the RL method handles reward optimisation within the safe region.
        Key papers: Safety-Gym (OpenAI 2019), SauteRL (Sootla 2022), RCPO (Tessler 2019).
        """), unsafe_allow_html=True)

    # ── FOUNDATION MODELS FOR RL ─────────────────────────────────────────
    with tab_fm:
        _sec("🌐", "Foundation Models for RL",
             "Gato · RT-2 · Algorithm Distillation — one model, any task, no retraining", "#0288d1")

        st.markdown(_card("#0288d1", "🌐", "What foundation models for RL means",
            """In NLP, a single GPT-4 model can write code, translate, do arithmetic, and summarise —
            adapting to each task via prompts, not retraining. The analogous goal for RL: one model that
            controls a robot arm, plays Atari, navigates mazes, and plays chess — adapting via in-context
            demonstrations rather than gradient updates. This would eliminate the need to train separate
            policies for every task. Three existing systems show partial versions: Gato (DeepMind 2022)
            — a transformer trained on 600+ tasks with tokenised actions; RT-2 (Google 2023) — robotic
            control using a VLM backbone that transfers internet knowledge to novel objects; Algorithm
            Distillation (Laskin 2023) and DPT (Lee 2024) — transformers that implement RL in their
            forward pass using only in-context experience. All have impressive demos but fail to fully
            match fine-tuned single-task models — the gap is closing rapidly with scale."""), unsafe_allow_html=True)

        st.subheader("1. Gato — One Transformer for 600+ Tasks")
        st.markdown(r"""
        **The Gato tokenisation scheme (Reed et al. 2022):**
        All modalities serialised to a flat token sequence:
        """)
        st.latex(r"\tau = (o_1^\text{token}, a_1^\text{token}, r_1^\text{token},\; o_2^\text{token}, \ldots)")
        st.markdown(r"""
        - Images: ViT patch embeddings projected to token dimension
        - Continuous actions: discretised to 1024 bins, one token per dimension
        - Text: standard BPE tokens
        - Trained with next-token prediction loss on all modalities:
        """)
        st.latex(r"\mathcal{L}_\text{Gato} = -\sum_t\log p_\theta(x_{t+1}|x_1,\ldots,x_t)")
        st.markdown(r"""
        **Result:** 1.2B parameters, 50–80% of single-task specialist performance across 600+ tasks.
        **What it cannot do:** True in-context adaptation — Gato requires full retraining for new tasks.
        """)

        st.subheader("2. RT-2 — Internet Knowledge for Robot Control")
        st.latex(r"\pi_\text{RT-2}(a_t|o_{1:t}, \text{instruction}) = p_\theta(a_t^\text{text}|o_{1:t}, \text{instruction})")
        st.markdown(r"""
        **Why internet pretraining helps:**
        RT-2 follows "pick up the soda can next to the apple" zero-shot.
        The VLM backbone provides object recognition from internet data; robot fine-tuning teaches control.
        **Zero-shot novel object generalisation:** 62% success on never-seen objects (vs 32% for RT-1).
        """)

        st.subheader("3. In-Context RL — Algorithm Distillation")
        st.markdown(r"""
        **The true foundation model goal:** A transformer that implements RL in its forward pass —
        no gradient updates at deployment, just a context of experience from the target task.

        **Algorithm Distillation (Laskin et al. 2023):**
        Train a transformer on sequences of entire RL learning histories from many tasks.
        The model must implement credit assignment and policy improvement in its attention mechanism.
        """)
        st.latex(r"\tau_\text{context} = (s_1,a_1,r_1,\ldots,s_t,a_t,r_t) \quad\text{(history from target task)}")
        st.latex(r"\pi_\text{AD}(a_{t+1}|s_{t+1},\tau_\text{context}) \quad\text{(predicts expert action after seeing }\tau\text{)}")
        st.markdown(r"""
        **DPT (Decision-Pretrained Transformer, Lee et al. 2024):** Proves that with sufficient
        training data, the transformer converges to Bayesian-optimal in-context RL.

        **Current limitation:** Works for simple tasks (bandits, short-horizon navigation).
        Scaling to complex long-horizon robotics: open problem.
        """)
        st.markdown(_insight("""
        <b>Practical demo:</b> Show Gato 10 demonstrations of a new block type during inference
        (no gradient updates). Performance on that block type improves by ~28% over zero-shot —
        this is in-context learning in RL. Not yet competitive with fine-tuning, but the gap
        is closing with scale (10× more parameters → 2× smaller in-context gap empirically).
        """), unsafe_allow_html=True)

    # ── OFFLINE TO ONLINE ────────────────────────────────────────────────
    with tab_o2o:
        _sec("📐", "Offline → Online RL — Fine-Tuning Without Forgetting",
             "Cal-QL · IQL→online · mixed replay — closing the distributional shift gap", "#558b2f")

        st.markdown(_card("#558b2f", "📐", "The offline-to-online problem",
            """Offline RL (CQL, IQL) learns strong initial policies from historical data. Online
            fine-tuning should improve performance further — but naively resuming online training
            from an offline-trained policy causes catastrophic performance degradation. The offline
            Q-function was trained to be conservative (underestimate OOD actions). When online
            training starts, this conservatism conflicts with learning from new data: the Q-function
            underestimates good new experiences, so the policy does not learn to exploit them.
            Removing the conservatism causes Q-value overestimation for new actions, leading to
            unstable updates. This is the offline-to-online gap. Solving it matters enormously:
            in real robotics, healthcare, and autonomous driving, you always have historical data
            before online collection. An efficient pipeline could reduce required online interactions
            by 10× by starting from a good offline policy rather than random initialisation."""),
            unsafe_allow_html=True)

        st.subheader("1. Why Naive Fine-Tuning Fails — Q-Function Miscalibration")
        st.markdown(r"""
        **Two-sided problem:**
        CQL trains a conservative Q-function that is a lower bound for in-distribution actions:
        """)
        st.latex(r"Q_\text{offline}(s,a_\text{new}) \ll Q^*(s,a_\text{new}) \quad\text{(underestimates new online experiences)}")
        st.markdown(r"""
        Policy ignores good new online actions → does not improve. But remove CQL conservatism:
        """)
        st.latex(r"Q_\text{no-reg}(s,a_\text{online}) \gg Q^*(s,a_\text{online}) \quad\text{(overestimates without regularisation)}")
        st.markdown(r"Policy exploits overestimates → takes bad actions → performance collapses.")

        # Simulate the problem
        np.random.seed(42)
        steps = np.arange(300)
        off_phase = 100
        off_perf = 40 + 35*(1 - np.exp(-steps[:off_phase]/30)) + np.random.randn(off_phase)*2
        naive = np.concatenate([off_perf,
            np.maximum(10, 75 - 30*np.exp(-(steps[off_phase:]-off_phase)/80) + np.random.randn(200)*3)])
        calql_curve = np.concatenate([off_perf,
            75 + 15*(1-np.exp(-(steps[off_phase:]-off_phase)/50)) + np.random.randn(200)*2])
        iql_curve = np.concatenate([off_perf-3,
            73 + 14*(1-np.exp(-(steps[off_phase:]-off_phase)/60)) + np.random.randn(200)*2.5])

        fig_o2o, ax_o2o = _fig(1, 1, 12, 4.5)
        ax_o2o.axvspan(0, off_phase, alpha=0.07, color="#546e7a")
        ax_o2o.axvspan(off_phase, 300, alpha=0.07, color="#00897b")
        ax_o2o.axvline(off_phase, color="white", ls="--", lw=1.5, alpha=0.5)
        ax_o2o.text(off_phase/2, 12, "Offline phase", color="#9e9ebb", ha="center", fontsize=8)
        ax_o2o.text(off_phase+50, 12, "Online fine-tuning", color="#9e9ebb", ha="center", fontsize=8)
        ax_o2o.plot(smooth(naive, 10), color="#ef5350", lw=2.5, label="Naive fine-tuning (catastrophic drop)")
        ax_o2o.plot(smooth(calql_curve, 10), color="#4caf50", lw=2.5, label="Cal-QL (smooth improvement)")
        ax_o2o.plot(smooth(iql_curve, 10), color="#ffa726", lw=2, label="IQL → online (moderate improvement)")
        ax_o2o.set_xlabel("Training steps", color="white")
        ax_o2o.set_ylabel("Normalised performance", color="white")
        ax_o2o.set_title("Offline→Online RL: Q-Function Miscalibration Causes Naive Fine-Tuning to Fail",
                         color="white", fontweight="bold")
        ax_o2o.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_o2o.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_o2o); plt.close()

        st.subheader("2. Cal-QL — Calibrated Conservative Q-Learning")
        st.markdown(r"""
        **Cal-QL (Nakamoto et al. 2023)** adds a calibration term to CQL that corrects underestimates
        for new online transitions without introducing overestimation for offline data:
        """)
        st.latex(r"\mathcal{L}_\text{Cal-QL}(Q) = \mathcal{L}_\text{CQL}(Q) + \alpha\,\mathbb{E}_{(s,a)\sim\mathcal{D}_\text{online}}\!\left[\max(0,\; Q^\text{target}(s,a)-Q(s,a))\right]")
        st.markdown(r"""
        **Symbol decoder:**
        - $\mathcal{L}_\text{CQL}$: conservative offline loss — pushes down OOD Q-values
        - Calibration term: if Q underestimates a new online transition → push it up
        - $\max(0,\cdot)$: only correct underestimates — do not create overestimation
        - $Q^\text{target}$: Bellman target from online experience (unregularised)

        **Why this solves the problem:** CQL conservatism preserved for offline data (no regression);
        new online data gets properly calibrated Q-values (no underestimation); smooth transition.
        """)

        st.subheader("3. IQL → Online — Mixed Replay Buffer")
        st.markdown(r"""
        IQL's design naturally extends to online settings. Because IQL uses expectile regression
        to avoid evaluating Q for OOD actions, the Q-function stays well-calibrated when new
        online transitions arrive. The key: keep offline data in the replay buffer during online training.
        """)
        st.latex(r"\mathcal{D}_\text{train}(t) = \mathcal{D}_\text{offline} \cup \mathcal{D}_\text{online}(t)")
        st.markdown(r"""
        **Annealing the offline ratio:** Start with 100% offline sampling. Gradually increase
        online sampling fraction as $|\mathcal{D}_\text{online}|$ grows:
        """)
        st.latex(r"p_\text{online}(t) = \frac{|\mathcal{D}_\text{online}(t)|}{|\mathcal{D}_\text{offline}|+|\mathcal{D}_\text{online}(t)|}")
        st.markdown(_insight("""
        <b>D4RL Hopper benchmark:</b> Random policy: ~10. Offline IQL: ~87. Naive online fine-tune: drops to ~65.
        IQL→online with mixed buffer: ~91. Cal-QL: ~93. Pure online SAC: ~90 (5× more online steps).
        The offline→online pipeline gets SAC-level performance with 5× fewer expensive real interactions.
        """), unsafe_allow_html=True)

    # ── LLM + RL BASICS ──────────────────────────────────────────────────
    with tab_llm:
        _sec("💬", "LLM + RL: RLHF and DPO Basics",
             "PPO on language models · DPO · The technical pipeline behind ChatGPT and Claude", "#e65100")

        st.markdown(_card("#e65100", "💬", "RLHF: the technique behind every modern aligned LLM",
            """RLHF applies PPO to fine-tune a pretrained language model using a reward signal from
            human preference comparisons. The language model IS the policy; tokens ARE actions; full
            responses ARE trajectories. InstructGPT (2022) was the first large-scale RLHF system;
            ChatGPT, Claude, Gemini all use variants. The KL penalty prevents the policy from
            collapsing to reward-hacking outputs (very short responses, repetition of trigger words)
            that score high on the reward model but are useless. DPO (2023) showed that the RLHF
            objective has a closed-form solution — you can directly train on preference data without
            a reward model or PPO, achieving competitive results with far simpler code."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RLHF Reward Model (Bradley-Terry):**")
            st.latex(r"P(y_w\succ y_l) = \sigma(r_\phi(y_w)-r_\phi(y_l))")
            st.latex(r"\mathcal{L}_\text{RM} = -\mathbb{E}\!\left[\log\sigma(r_\phi(y_w)-r_\phi(y_l))\right]")
            st.markdown("**PPO RLHF objective:**")
            st.latex(r"J(\theta) = \mathbb{E}\!\left[r_\phi(y) - \beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]")
        with col2:
            st.markdown("**DPO — closed-form RLHF:**")
            st.markdown(r"From optimal RLHF: $\pi^*(y|x)\propto\pi_\text{ref}(y|x)\exp(r/\beta)$. Substitute into Bradley-Terry:")
            st.latex(r"\mathcal{L}_\text{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}\right)\right]")
            st.markdown("No reward model, no PPO — one supervised loss on preference pairs.")

    # ── SIM-TO-REAL ──────────────────────────────────────────────────────
    with tab_s2r:
        _sec("🤖", "Sim-to-Real Transfer",
             "Domain randomisation · System ID · RMA — closing the reality gap", "#00897b")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Domain Randomisation:**")
            st.latex(r"\xi\sim p(\xi), \quad \pi^*=\arg\max_\pi\mathbb{E}_\xi[J(\pi;\xi)]")
            st.markdown(r"Randomise mass, friction, damping, sensor noise. Real robot ≈ one sample from $p(\xi)$.")
        with col2:
            st.markdown("**RMA — Rapid Motor Adaptation:**")
            st.latex(r"\pi(a_t|s_t,z_t),\quad z_t=\text{Enc}(\tau_{t-H:t})")
            st.markdown(r"Phase 1: train with privileged $z^*$. Phase 2: learn to estimate $z^*$ from observable history. Phase 3: deploy on real robot.")
        st.markdown(_insight("""
        OpenAI Dexterous Hand, ETH ANYmal, and Boston Dynamics Spot all combine domain randomisation
        + system identification + adaptive policies. The combination is more effective than any
        single approach alone — broad randomisation provides robustness, system ID provides accuracy,
        adaptation handles the residual gap.
        """), unsafe_allow_html=True)

    # ── DIFFUSION RL ─────────────────────────────────────────────────────
    with tab_diff:
        _sec("🌊", "Diffusion Models for RL",
             "Diffuser · Decision Diffuser — trajectories as denoised samples", "#7c4dff")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Forward noising (destroys trajectory):**")
            st.latex(r"q(\tau^k|\tau^0)=\mathcal{N}(\sqrt{\bar\alpha_k}\tau^0,(1-\bar\alpha_k)I)")
            st.markdown("**Reverse denoising (planning):**")
            st.latex(r"p_\theta(\tau^{k-1}|\tau^k)=\mathcal{N}(\mu_\theta(\tau^k,k),\Sigma_k)")
        with col2:
            st.markdown("**Return-conditioned planning (Diffuser):**")
            st.latex(r"\tau\sim p_\theta(\tau^0|\hat R)\propto p_\theta(\tau^0)\cdot p(\hat R|\tau^0)")
            st.markdown(r"Classifier guidance steers denoising: $\nabla_{\tau^k}\log p(\hat R|\tau^k)$.")
        st.markdown(_insight("""
        Key advantage over Decision Transformer: diffusion models capture multimodal distributions.
        If 3 different paths achieve a goal, diffusion represents all 3 simultaneously; autoregressive
        models average between them. Critical for manipulation where many action sequences achieve the same goal.
        """), unsafe_allow_html=True)

    # ── RL THEORY ────────────────────────────────────────────────────────
    with tab_theory:
        _sec("📏", "RL Theory — Mathematical Foundations",
             "PAC-MDP · Regret bounds · Policy gradient convergence · Key theorems", "#6a1b9a")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PAC-MDP sample complexity:**")
            st.latex(r"O\!\left(\frac{|S||A|}{\varepsilon^2(1-\gamma)^3}\log\frac{1}{\delta}\right)")
            st.markdown("**Bellman optimality:**")
            st.latex(r"V^*(s)=\max_a[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V^*(s')]")
            st.markdown("**Regret lower bound (Lai & Robbins):**")
            st.latex(r"\mathbb{E}[\text{Regret}]\geq\sum_{a:\Delta_a>0}\frac{\ln T}{\text{KL}(\mu_a\|\mu^*)}")
        with col2:
            st.markdown("**Policy gradient convergence:**")
            st.latex(r"\frac{1}{T}\sum_{t}\|\nabla J(\theta_t)\|^2\leq O(1/\sqrt{T})")
            st.markdown("**TRPO improvement guarantee:**")
            st.latex(r"J(\pi')\geq J(\pi)-\frac{4\gamma\epsilon}{(1-\gamma)^2}\overline{D}_\text{KL}(\pi'\|\pi)")
            st.markdown("**CQL lower bound:**")
            st.latex(r"\mathbb{E}_\beta[Q^\text{CQL}]\leq\mathbb{E}_\beta[Q^\pi]")

    # ── ROADMAP ──────────────────────────────────────────────────────────
    with tab_road:
        st.subheader("🗺️ Frontier RL Research Roadmap (2025)")
        problems = [
            ("🧠","RLHF at Scale","Active","#e65100",
             "CAI + Debate + Scalable Oversight",
             "Evaluator collapse for superhuman systems",
             "Bai et al. CAI 2022; Irving Debate 2018; Christiano Amplification 2018"),
            ("🌍","World Models + RL","Active","#00897b",
             "Ensemble uncertainty + short horizons + contact-aware models",
             "1% per-step error → 16% cumulative over H=15 steps",
             "Hafner DreamerV3 2023; Osband RLSVI; ensemble uncertainty methods"),
            ("🔍","Exploration (Large Spaces)","Open","#7c4dff",
             "PSRL + neural density models + ensemble disagreement",
             "T^(2/3) lower bound — no O(log T) in continuous MDPs",
             "Osband PSRL 2013; Ostrovski count-based 2017; Burda RND 2018"),
            ("🛡️","Safe RL (Formal)","Active","#ef5350",
             "CBF shielding + CPO + Lagrangian primal-dual",
             "Training-time violations; hard safety during exploration",
             "Achiam CPO 2017; Safety-Gym; SauteRL 2022; CBF-RL survey"),
            ("🌐","Foundation Models for RL","Emerging","#0288d1",
             "Gato + RT-2 + Algorithm Distillation + DPT",
             "In-context RL weaker than fine-tuning; long-horizon scaling",
             "Reed Gato 2022; Brohan RT-2 2023; Laskin AD 2023; Lee DPT 2024"),
            ("📐","Offline → Online RL","Active","#558b2f",
             "Cal-QL + IQL→online + mixed replay buffer + annealing",
             "Q-function miscalibration; forgetting offline knowledge",
             "Nakamoto Cal-QL 2023; Lee IQL 2021; balanced replay research"),
        ]
        for icon, title, status, color, approach, challenge, papers in problems:
            sc = {"Active":"#0288d1","Open":"#ef5350","Emerging":"#00897b"}[status]
            with st.expander(f"{icon} {title} — {status}", expanded=False):
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f'<div style="background:{color}18;border-left:3px solid {color};'
                                f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                                f'<b style="color:{color}">🎯 Challenge:</b><br>'
                                f'<span style="color:#b0b0cc;font-size:.9rem">{challenge}</span></div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div style="background:#0a2a0a;border-left:3px solid #4caf50;'
                                f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                                f'<b style="color:#4caf50">✅ Best approach:</b><br>'
                                f'<span style="color:#b0b0cc;font-size:.9rem">{approach}</span></div>',
                                unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;'
                                f'border-radius:8px;padding:.7rem 1rem">'
                                f'<b style="color:{sc}">Status: {status}</b><br><br>'
                                f'<b style="color:#9e9ebb;font-size:.82rem">Key papers:</b><br>'
                                f'<span style="color:#b0b0cc;font-size:.82rem">{papers}</span></div>',
                                unsafe_allow_html=True)
        st.divider()
        st.subheader("📚 Resources")
        for icon, title, desc, url in [
            ("🎥","CS285 Berkeley (Levine)","Lectures 18–22 cover frontier topics: scalable oversight, world models, safety.","https://rail.eecs.berkeley.edu/deeprlcourse/"),
            ("📄","Anthropic Research","Constitutional AI, RLHF at scale, interpretability research.","https://www.anthropic.com/research"),
            ("📄","DeepMind RL Papers","DreamerV3, Gato, safety research — frontier applications.","https://www.deepmind.com/research"),
            ("💻","TRL — Transformer RL Library","RLHF + DPO for LLMs. Best entry point for LLM alignment work.","https://github.com/huggingface/trl"),
            ("💻","Safety-Gym (OpenAI)","Standard benchmark for safe RL — CPO, PPO-Lagrangian, TRPO-Lagrangian.","https://github.com/openai/safety-gym"),
            ("📰","Yannic Kilcher — RL Papers","YouTube explanations of cutting-edge RL papers — accessible and deep.","https://www.youtube.com/@YannicKilcher"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
