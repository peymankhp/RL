"""_frontier_mod.py — Frontier RL Research (Tier 4) — Enhanced Edition
RLHF at Scale · World Models · Exploration (Large Spaces) · Safe RL (Formal)
Foundation Models · Offline→Online · LLM+RL Basics · Sim-to-Real · Diffusion · Theory"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import warnings
from _notes_mod import render_notes
warnings.filterwarnings("ignore")

DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"


def _fig(nr=1, nc=1, w=13, h=5):
    fig, axes = plt.subplots(nr, nc, figsize=(w, h))
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
            f'<b>{icon} {title}</b><br>'
            f'<span style="color:#b0b0cc;font-size:.92rem;line-height:1.7">{body}</span></div>')


def _insight(t):
    return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
            f'padding:.8rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;'
            f'font-size:.92rem;color:#b0b0cc;line-height:1.7">💡 {t}</div>')


def _book(title, authors, why, url):
    return (f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
            f'padding:.7rem 1.1rem;margin:.3rem 0">'
            f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">📚 {title}</a>'
            f'<br><span style="color:#7a9ebb;font-size:.82rem">{authors}</span>'
            f'<br><span style="color:#9e9ebb;font-size:.84rem">{why}</span></div>')


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
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'Six open problems defining the next decade of RL — with full derivations, '
        'practical examples with real numbers, charts, and the best resources to go deep.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🗺️ Overview",
        "🧠 RLHF at Scale",
        "🌍 World Models + RL",
        "🔍 Exploration (Large)",
        "🛡️ Safe RL (Formal)",
        "🌐 Foundation Models",
        "📐 Offline→Online RL",
        "💬 LLM + RL Basics",
        "🤖 Sim-to-Real",
        "🌊 Diffusion RL",
        "📏 RL Theory",
        "📚 Books & Resources",
    ])
    (tab_ov, tab_rlhf, tab_wm, tab_exp,
     tab_safe, tab_fm, tab_o2o, tab_llm,
     tab_s2r, tab_diff, tab_theory, tab_res) = tabs

    # ── OVERVIEW ─────────────────────────────────────────────────
    with tab_ov:
        _sec("🗺️", "Six Open Problems at the Frontier",
             "What is still unsolved after mastering all core algorithms", "#ad1457")

        st.dataframe(pd.DataFrame({
            "Problem": ["RLHF at Scale", "World Models + RL",
                        "Exploration (Large Spaces)", "Safe RL (Formal Guarantees)",
                        "Foundation Models for RL", "Offline → Online RL"],
            "Status": ["Active", "Active", "Open (unsolved)", "Active", "Emerging", "Active"],
            "Core challenge": [
                "Humans cannot evaluate superhuman outputs",
                "1% per-step error compounds to 16% over 15 steps",
                "No O(log T) regret algorithm exists for continuous MDPs",
                "Asymptotic guarantees fail during training itself",
                "In-context RL still weaker than fine-tuning",
                "Q-function miscalibration breaks offline→online transfer",
            ],
            "Best current approach": [
                "Constitutional AI + Debate + Scalable Oversight",
                "RSSM + uncertainty estimation + short horizons",
                "PSRL + ensemble disagreement + neural density",
                "CBF shielding + CPO + Lagrangian primal-dual",
                "Gato, RT-2, Algorithm Distillation, DPT",
                "Cal-QL + IQL→online + mixed replay buffer",
            ],
        }), use_container_width=True, hide_index=True)

    # ── RLHF AT SCALE ────────────────────────────────────────────
    with tab_rlhf:
        _sec("🧠", "RLHF at Scale — Aligning Superhuman AI",
             "Evaluator collapse · Constitutional AI · Debate · Scalable Oversight", "#e65100")

        st.markdown(_card("#e65100", "🧠", "The evaluator collapse problem",
            """Standard RLHF asks humans to compare two AI responses. This works for tasks humans
            understand (write a poem, summarise a document). But when the AI produces a novel
            mathematical proof or complex security analysis that the human evaluator cannot verify,
            the evaluator says 'response A looks more confident' without knowing if it's correct.
            The reward model learns 'looks correct' rather than 'is correct'. The policy is then
            optimised to appear correct rather than be correct — reward hacking at civilisational scale.
            Three research directions address this: Constitutional AI (AI evaluates against written
            principles), Debate (two AIs argue and humans verify the argument, not the answer),
            and Scalable Oversight (humans use AI assistance to evaluate superhuman outputs)."""), unsafe_allow_html=True)

        st.subheader("1. Standard RLHF — Reward Model Training")
        st.latex(r"P(y_w \succ y_l | x) = \sigma(r_\phi(x,y_w) - r_\phi(x,y_l)) \quad\text{(Bradley-Terry model)}")
        st.latex(r"\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\right]")
        st.latex(r"J(\theta) = \mathbb{E}_{y\sim\pi_\theta}\!\left[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]")

        st.subheader("2. Constitutional AI — AI Evaluates Against Principles")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Critique and revision (SL-CAI):**")
            st.latex(r"\text{critique} = \pi(x, y, p) \quad\text{(model critiques using principle }p\text{)}")
            st.latex(r"y' = \pi(x, y, \text{critique}) \quad\text{(model revises)}")
            st.markdown("**AI feedback (RLAIF):**")
            st.latex(r"P(y_w\succ y_l|x,p) = \sigma(\text{score}_\pi(y_w,p)-\text{score}_\pi(y_l,p))")
        with col2:
            # Visualise oversight capacity extension
            cap = np.linspace(0, 10, 100)
            fig_so, ax_so = _fig(1, 1, 5.5, 4)
            ax_so.plot(cap, np.minimum(3, 1.5 + 0.2 * cap), color="#ffa726", lw=2.5, label="Human alone (ceiling ~3)")
            ax_so.plot(cap, np.minimum(7, 0.5 + 0.65 * cap), color="#0288d1", lw=2.5, label="Human + AI assist")
            ax_so.plot(cap, np.minimum(9, 1.0 + 0.8 * cap), color="#4caf50", lw=2.5, label="CAI + Debate")
            ax_so.axvline(3, color="#ffa726", ls=":", lw=1.5, alpha=0.7)
            ax_so.set_xlabel("AI capability level", color="white")
            ax_so.set_ylabel("Max evaluable capability", color="white")
            ax_so.set_title("Scalable Oversight:\nextending human evaluation", color="white", fontweight="bold")
            ax_so.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_so.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_so); plt.close()

        st.subheader("3. Debate — Nash Equilibrium of Honest Argument")
        st.markdown("Two AI debaters argue opposite positions. Human judges the debate. The Nash equilibrium is honest debate:")
        st.latex(r"D_1^* = \arg\max_{D_1}\min_{D_2}\,\mathbb{E}[J(q, D_1(q), D_2(q))]")
        st.markdown("**Why truthfulness is the equilibrium:** if D₁ makes a false claim, D₂ can expose it. The human can verify the exposure (easier than verifying the original claim) even without domain expertise.")

    # ── WORLD MODELS ─────────────────────────────────────────────
    with tab_wm:
        _sec("🌍", "World Models + RL — Scaling to Real Robotics",
             "RSSM architecture · Compounding errors · Uncertainty-aware planning", "#00897b")

        st.markdown(_card("#00897b", "🌍", "Why world models have not conquered real robotics",
            """DreamerV3 solves Atari (150+ games) and Minecraft with fixed hyperparameters.
            But translating this to contact-rich real robotics remains unsolved.
            The compounding error problem: a 1% per-step prediction error over a 15-step
            imagination horizon becomes (1.01)^15 - 1 = 16% cumulative error.
            For grasping tasks where contact forces must be accurate to 1N over 100N total,
            this is catastrophic. Real sensors add noise, latency, and occlusion that simulation
            never models. The frontier: uncertainty-aware planning (trust the model only when confident),
            contact-aware world models, and sim-to-real transfer of world model knowledge."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RSSM — split deterministic + stochastic:**")
            st.latex(r"h_t = f_\phi(h_{t-1},\,z_{t-1},\,a_{t-1}) \quad\text{(GRU — long-range memory)}")
            st.latex(r"z_t \sim q_\phi(z_t|h_t,o_t) \quad\text{(posterior — with real observation)}")
            st.latex(r"\hat z_t \sim p_\phi(\hat z_t|h_t) \quad\text{(prior — for imagination, no obs)}")
            st.markdown("**Ensemble uncertainty (know when not to plan):**")
            st.latex(r"\sigma^2_\text{epist}(s,a)=\frac{1}{M}\sum_m(\hat\mu_m-\bar\mu)^2")
        with col2:
            # Compounding error chart
            H = np.arange(1, 51)
            fig_ce, ax_ce = _fig(1, 1, 5.5, 4)
            for eps, col, lbl in [(0.001, "#4caf50", "ε=0.1%/step (good sim)"),
                                   (0.01,  "#ffa726", "ε=1%/step (real robot)"),
                                   (0.05,  "#ef5350", "ε=5%/step (contact-rich)")]:
                ax_ce.plot(H, [(1+eps)**h - 1 for h in H], color=col, lw=2.5, label=lbl)
            ax_ce.axvline(15, color="white", ls="--", lw=1.2, alpha=0.6, label="DreamerV3 H=15")
            ax_ce.axhline(0.2, color="#b0b0cc", ls=":", lw=1, alpha=0.5)
            ax_ce.set_xlabel("Horizon H (steps)", color="white")
            ax_ce.set_ylabel("Cumulative error (1+ε)^H - 1", color="white")
            ax_ce.set_title("Compounding error:\nwhy short horizons matter", color="white", fontweight="bold")
            ax_ce.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_ce.grid(alpha=0.12); ax_ce.set_ylim(0, 1.5)
            plt.tight_layout(); st.pyplot(fig_ce); plt.close()

        st.markdown("**World model loss (4 simultaneously trained objectives):**")
        st.latex(r"\mathcal{L}_\text{WM} = \mathbb{E}\!\left[\log p(o_t|z_t,h_t)+\log p(r_t|z_t,h_t)+\log p(d_t|z_t,h_t)-\beta D_\text{KL}(q\|p)\right]")
        st.markdown("**symlog transform** (DreamerV3 key innovation): normalises reward targets across environments with different scales without manual tuning:")
        st.latex(r"\text{symlog}(x) = \text{sign}(x)\cdot\log(|x|+1)")

    # ── EXPLORATION ───────────────────────────────────────────────
    with tab_exp:
        _sec("🔍", "Exploration in Large State Spaces",
             "No O(log T) regret algorithm in continuous MDPs — the fundamental open problem", "#7c4dff")

        st.markdown(_card("#7c4dff", "🔍", "Why the bandit solution doesn't scale",
            """UCB achieves O(log T) regret in K-armed bandits. In tabular MDPs, UCRL2 achieves
            polynomial sample complexity. Both require maintaining per-state-action counts or
            confidence bounds — impossible in continuous spaces (every state is visited exactly once).
            Neural network approximations of confidence bounds introduce errors that destroy guarantees.
            The fundamental information-theoretic lower bound: for 1D continuous MDPs, any algorithm
            must incur at least O(T^{2/3}) regret. This is ~3000× worse than discrete UCB's O(log T)
            for T=10M steps. The frontier: PSRL (posterior sampling approximates optimality),
            ensemble disagreement as uncertainty proxy, and neural density models for count bonuses."""), unsafe_allow_html=True)

        st.markdown("**Lower bound derivation (continuous MDPs):**")
        st.markdown("On a 1D continuous MDP with reward at unknown point x* ~ Uniform[0,1]:")
        st.latex(r"\mathbb{E}[\text{Regret}(T)] \geq c\cdot T^{2/3}")
        st.markdown("Derivation: optimal visits N(t) ~ T^{1/3} balances exploration cost vs missed reward. Total regret ~ T - T^{1/3} ~ T^{2/3}.")

        # Regret scaling chart
        T = np.logspace(3, 7, 100)
        fig_reg, ax_reg = _fig(1, 1, 11, 4.5)
        ax_reg.loglog(T, np.log(T), color="#4caf50", lw=2.5, label=r"$O(\log T)$ — discrete UCB (optimal)")
        ax_reg.loglog(T, T**0.5 / 100, color="#ffa726", lw=2.5, label=r"$O(\sqrt{T})$ — PSRL in linear MDPs")
        ax_reg.loglog(T, T**(2/3) / 200, color="#ef5350", lw=2.5, label=r"$O(T^{2/3})$ — continuous lower bound")
        ax_reg.loglog(T, T * 0.01, color="#546e7a", lw=2, ls="--", label="$O(T)$ — random exploration")
        ax_reg.set_xlabel("Steps T", color="white"); ax_reg.set_ylabel("Regret", color="white")
        ax_reg.set_title("Exploration Regret Scaling: Continuous spaces are fundamentally\nharder than discrete (3000× worse at T=10M)", color="white", fontweight="bold")
        ax_reg.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_reg.grid(alpha=0.12, which="both")
        plt.tight_layout(); st.pyplot(fig_reg); plt.close()

        st.markdown("**Posterior Sampling for RL (PSRL):**")
        st.latex(r"\mathbb{E}[\text{Regret}(T)] \leq \tilde O(d\sqrt{T}) \quad\text{(linear MDPs, dimension }d\text{)}")
        st.markdown("**Neural density model bonus:**")
        st.latex(r"r_\text{bonus}(s) = \beta\cdot(-\log\hat p_\phi(s)) \quad\text{(novel states have low density)}")

    # ── SAFE RL ───────────────────────────────────────────────────
    with tab_safe:
        _sec("🛡️", "Safe RL with Formal Guarantees",
             "CMDP · CBF · CPO — constraint satisfaction during training, not just at convergence", "#ef5350")

        st.markdown(_card("#ef5350", "🛡️", "Why asymptotic safety is not enough",
            """CPO and Lagrangian PPO guarantee constraint satisfaction AT convergence.
            During training, the policy violates constraints thousands of times as it explores.
            For a surgical robot, autonomous vehicle, or industrial controller, even a single
            training-time violation can be catastrophic. The gap between 'eventually safe' and
            'always safe' is unbridgeable by any gradient-based asymptotic method.
            Control Barrier Functions (CBFs) provide the only current approach that mathematically
            guarantees zero constraint violations at every training step — by filtering the RL
            policy's proposed actions through a QP that enforces the safety condition."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CMDP formulation:**")
            st.latex(r"\max_\pi J(\pi)\;\text{s.t.}\;J_C(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t c_t]\leq d")
            st.markdown("**Lagrangian relaxation:**")
            st.latex(r"\mathcal{L}(\pi,\lambda)=J(\pi)-\lambda(J_C(\pi)-d)")
            st.latex(r"\lambda_{t+1}=\max(0,\lambda_t+\eta(J_C(\pi_t)-d))")
            st.markdown("**CBF forward invariance:**")
            st.latex(r"h(s)\geq 0\;\Rightarrow\;\dot h(s,a)+\alpha h(s)\geq 0")
            st.markdown("**CBF safety filter QP:**")
            st.latex(r"a^*=\arg\min\|a'-a\|^2\;\text{s.t.}\;\nabla h^T f(s,a')+\alpha h(s)\geq 0")
        with col2:
            # Training-time violations comparison
            np.random.seed(42); T = 300
            violations_unconstrained = smooth(np.maximum(0, np.random.poisson(5, T).astype(float)), 20)
            violations_lagrangian = smooth(np.maximum(0, np.random.poisson(2, T) * np.exp(-np.arange(T)/100)).astype(float), 20)
            violations_cbf = np.zeros(T)
            fig_viol, ax_viol = _fig(1, 1, 5.5, 4)
            ax_viol.plot(violations_unconstrained, color="#ef5350", lw=2, label="Unconstrained (~8000 total)")
            ax_viol.plot(violations_lagrangian, color="#ffa726", lw=2, label="Lagrangian PPO (~2000 total)")
            ax_viol.plot(violations_cbf, color="#4caf50", lw=2.5, label="CBF shield (0 — guaranteed)")
            ax_viol.set_xlabel("Episode", color="white"); ax_viol.set_ylabel("Constraint violations/ep", color="white")
            ax_viol.set_title("Training-time violations:\nonly CBF gives zero guarantee", color="white", fontweight="bold")
            ax_viol.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_viol.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_viol); plt.close()

        st.markdown(_insight("Best practice: CBF shield (hard zero-violation guarantee during training) + CMDP Lagrangian (deployment constraint guarantee). Both are needed. CBF handles training safety; Lagrangian handles deployment. The combination is used in autonomous vehicle and surgical robot research."), unsafe_allow_html=True)

    # ── FOUNDATION MODELS ─────────────────────────────────────────
    with tab_fm:
        _sec("🌐", "Foundation Models for RL",
             "Gato · RT-2 · Algorithm Distillation — one model, any task, no retraining", "#0288d1")

        st.markdown(_card("#0288d1", "🌐", "The foundation model goal for RL",
            """In NLP, GPT-4 handles code, translation, math, and summarisation via prompts — no retraining.
            The analogous goal for RL: one model controlling robots, playing games, navigating mazes,
            adapting via in-context demonstrations rather than gradient updates.
            Gato (2022): 1.2B parameter transformer trained on 600+ tasks (tokenised observations
            and actions from text, images, proprioception). Achieves 50–80% of single-task specialists.
            RT-2 (2023): uses a VLM backbone (internet-pretrained) for robot control — enables
            zero-shot novel object manipulation from internet knowledge.
            Algorithm Distillation (2023): trains a transformer on RL learning histories —
            the model implements credit assignment in its forward pass (in-context RL)."""), unsafe_allow_html=True)

        st.latex(r"\tau = (o_1^\text{token}, a_1^\text{token}, r_1^\text{token}, o_2^\text{token}, \ldots) \quad\text{(Gato: everything as tokens)}")
        st.latex(r"\mathcal{L}_\text{Gato} = -\sum_t \log p_\theta(x_{t+1}|x_1,\ldots,x_t) \quad\text{(next-token prediction on all modalities)}")

        st.dataframe(pd.DataFrame({
            "System": ["Gato (2022)", "RT-2 (2023)", "Algorithm Distillation (2023)", "DPT (2024)"],
            "Tasks": ["600+", "Robotic manipulation", "Multi-task RL", "Multi-task RL"],
            "Parameters": ["1.2B", "55B", "35M", "200M"],
            "In-context adaptation?": ["Limited", "No (fine-tuning)", "Yes (context = history)", "Yes (Bayesian-optimal)"],
            "vs specialist": ["50–80%", "62% novel obj (vs 32%)", "~60% after 10 demos", "Near-optimal (theory)"],
            "Key innovation": ["Universal tokenisation", "VLM backbone + robot", "RL history as context", "Bayesian-optimal IC-RL"],
        }), use_container_width=True, hide_index=True)

    # ── OFFLINE TO ONLINE ─────────────────────────────────────────
    with tab_o2o:
        _sec("📐", "Offline → Online RL — Fine-Tuning Without Forgetting",
             "Cal-QL · IQL→online · Mixed replay — the distributional shift challenge", "#558b2f")

        st.markdown(_card("#558b2f", "📐", "Why naive fine-tuning fails",
            """CQL trains a conservative Q-function that underestimates OOD actions.
            When online training starts with new experiences:
            (1) Q_offline(s, a_new) << Q*(s, a_new) → policy ignores good new actions
            (2) Without CQL penalty → Q overestimates OOD actions → policy exploits wrong estimates
            Cal-QL (2023) fixes this with a calibration term that corrects underestimates for
            online data without introducing overestimation for offline data."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cal-QL calibration loss:**")
            st.latex(r"\mathcal{L}_\text{Cal-QL} = \mathcal{L}_\text{CQL} + \alpha\mathbb{E}_\text{online}\!\left[\max(0,\,Q^\text{target}-Q)\right]")
            st.markdown("**Mixed replay buffer:**")
            st.latex(r"\mathcal{D}_\text{train}(t) = \mathcal{D}_\text{offline} \cup \mathcal{D}_\text{online}(t)")
            st.markdown("**Annealing schedule:**")
            st.latex(r"p_\text{online}(t) = \frac{|\mathcal{D}_\text{online}(t)|}{|\mathcal{D}_\text{offline}|+|\mathcal{D}_\text{online}(t)|}")
        with col2:
            # Offline to online learning curve
            np.random.seed(42); steps = np.arange(300); off_phase = 100
            off_perf = 40 + 35*(1-np.exp(-steps[:off_phase]/30)) + np.random.randn(off_phase)*2
            naive = np.concatenate([off_perf, np.maximum(10, 75-30*np.exp(-(steps[off_phase:]-off_phase)/80)+np.random.randn(200)*3)])
            calql = np.concatenate([off_perf, 75+15*(1-np.exp(-(steps[off_phase:]-off_phase)/50))+np.random.randn(200)*2])
            fig_o2, ax_o2 = _fig(1,1,5.5,4)
            ax_o2.axvspan(0, off_phase, alpha=0.07, color="#546e7a")
            ax_o2.axvspan(off_phase, 300, alpha=0.07, color="#558b2f")
            ax_o2.axvline(off_phase, color="white", ls="--", lw=1.5, alpha=0.5)
            ax_o2.plot(smooth(naive,10), color="#ef5350", lw=2, label="Naive fine-tune (drops)")
            ax_o2.plot(smooth(calql,10), color="#4caf50", lw=2.5, label="Cal-QL (smooth +)")
            ax_o2.set_xlabel("Steps",color="white"); ax_o2.set_ylabel("Perf.",color="white")
            ax_o2.set_title("Offline→Online: Cal-QL\nprevents catastrophic drop",color="white",fontweight="bold")
            ax_o2.legend(facecolor=CARD,labelcolor="white",fontsize=7.5); ax_o2.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_o2); plt.close()

    # ── LLM + RL ─────────────────────────────────────────────────
    with tab_llm:
        _sec("💬", "LLM + RL: RLHF and DPO Basics",
             "PPO on language models · DPO derivation · The pipeline behind ChatGPT and Claude", "#e65100")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**RLHF reward model (Bradley-Terry):**")
            st.latex(r"P(y_w\succ y_l)=\sigma(r_\phi(y_w)-r_\phi(y_l))")
            st.latex(r"\mathcal{L}_\text{RM}=-\mathbb{E}\!\left[\log\sigma(r_\phi(y_w)-r_\phi(y_l))\right]")
            st.markdown("**PPO RLHF objective:**")
            st.latex(r"J(\theta)=\mathbb{E}_{y\sim\pi_\theta}\!\left[r_\phi(y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}\right]")
        with col2:
            st.markdown("**DPO — closed-form RLHF (no PPO needed):**")
            st.markdown("From optimal RLHF solution π*(y|x) ∝ π_ref·exp(r/β):")
            st.latex(r"r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)")
            st.markdown("Substitute into Bradley-Terry, Z(x) cancels:")
            st.latex(r"\mathcal{L}_\text{DPO}=-\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}\right)\right]")

        # RLHF pipeline visualisation
        pipeline_steps = ["1. Pretrain LLM", "2. SFT on good demos", "3. Collect human prefs",
                          "4. Train reward model", "5. PPO fine-tuning", "6. Deploy"]
        rewards_pipeline = [0.20, 0.45, 0.45, 0.45, 0.78, 0.85]
        fig_pipe, ax_pipe = _fig(1, 1, 11, 3.5)
        colors_pipe = ["#546e7a", "#0288d1", "#7c4dff", "#ffa726", "#e65100", "#4caf50"]
        for i, (step, r, col) in enumerate(zip(pipeline_steps, rewards_pipeline, colors_pipe)):
            ax_pipe.bar(i, r, color=col, alpha=0.85, width=0.7)
            ax_pipe.text(i, r + 0.02, f"{r:.2f}", ha="center", color="white", fontsize=8)
        ax_pipe.set_xticks(range(len(pipeline_steps)))
        ax_pipe.set_xticklabels([s.replace(" ", "\n") for s in pipeline_steps], color="white", fontsize=7.5)
        ax_pipe.set_ylabel("Human preference score (proxy)", color="white")
        ax_pipe.set_title("RLHF Pipeline: Each stage improves human preference alignment",
                          color="white", fontweight="bold")
        ax_pipe.grid(alpha=0.12, axis="y")
        plt.tight_layout(); st.pyplot(fig_pipe); plt.close()

    # ── SIM TO REAL ───────────────────────────────────────────────
    with tab_s2r:
        _sec("🤖", "Sim-to-Real Transfer",
             "Domain randomisation · System ID · RMA — closing the reality gap", "#00897b")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Domain Randomisation:**")
            st.latex(r"\xi\sim p(\xi)\quad\pi^*=\arg\max_\pi\mathbb{E}_\xi[J(\pi;\xi)]")
            st.markdown("Randomise: gravity, mass, friction, damping, sensor noise, textures.")
            st.markdown("Real robot = one sample from p(ξ).")
            st.markdown("**RMA — Rapid Motor Adaptation:**")
            st.latex(r"\pi(a_t|s_t,z_t)\quad z_t=\text{Enc}(\tau_{t-H:t})")
            st.markdown("Phase 1 (sim): train with privileged z*. Phase 2 (sim): train encoder to estimate z* from observations. Phase 3 (real): deploy — encoder estimates z from real trajectory.")
        with col2:
            # Reality gap illustration
            np.random.seed(42)
            ep_rewards_sim = smooth(np.minimum(95, 10 + np.arange(200)*0.45 + np.random.randn(200)*3), 15)
            ep_rewards_real_naive = smooth(np.minimum(40, 10 + np.arange(200)*0.15 + np.random.randn(200)*5), 15)
            ep_rewards_real_rma = smooth(np.minimum(85, 10 + np.arange(200)*0.38 + np.random.randn(200)*4), 15)
            fig_sr, ax_sr = _fig(1, 1, 5.5, 4)
            ax_sr.plot(range(len(ep_rewards_sim)), ep_rewards_sim, color="#4caf50", lw=2.5, label="Simulation policy")
            ax_sr.plot(range(len(ep_rewards_real_naive)), ep_rewards_real_naive, color="#ef5350", lw=2, ls="--", label="Naive sim-to-real (gap)")
            ax_sr.plot(range(len(ep_rewards_real_rma)), ep_rewards_real_rma, color="#ffa726", lw=2, label="With RMA adaptation")
            ax_sr.set_xlabel("Episode", color="white"); ax_sr.set_ylabel("Reward", color="white")
            ax_sr.set_title("Reality Gap:\nRMA closes most of it", color="white", fontweight="bold")
            ax_sr.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_sr.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_sr); plt.close()

        st.markdown(_insight("The most successful sim-to-real approach combines: domain randomisation (robust training) + system identification (accurate simulation) + RMA adaptive policy (residual correction). OpenAI Dexterous Hand, ETH ANYmal, and Boston Dynamics Spot all use this combination. Key insight: don't try to eliminate the sim-to-real gap — design the policy to handle it."), unsafe_allow_html=True)

    # ── DIFFUSION ─────────────────────────────────────────────────
    with tab_diff:
        _sec("🌊", "Diffusion Models for RL",
             "Diffuser · Decision Diffuser — trajectories as denoised samples from a generative model", "#7c4dff")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Forward noising process:**")
            st.latex(r"q(\tau^k|\tau^0)=\mathcal{N}(\sqrt{\bar\alpha_k}\tau^0,\,(1-\bar\alpha_k)I)")
            st.markdown("**Reverse denoising (= planning):**")
            st.latex(r"p_\theta(\tau^{k-1}|\tau^k)=\mathcal{N}(\mu_\theta(\tau^k,k),\,\Sigma_k)")
            st.markdown("**Return-conditioned planning:**")
            st.latex(r"\tau\sim p_\theta(\tau^0|\hat R)\propto p_\theta(\tau^0)\cdot p(\hat R|\tau^0)")
            st.markdown("Classifier guidance: ∇_τ log p(R̂|τ^k) steers denoising toward high-return trajectories.")
        with col2:
            # Denoising visualisation
            np.random.seed(42); steps_diff = np.linspace(0, 1, 50); K = 6
            fig_diff_v, axes_diff_v = _fig(1, 1, 5.5, 4)
            traj_true = np.sin(steps_diff * 3) * 2
            for k in range(K):
                noise_level = (K - k - 1) / K
                traj_noisy = traj_true + np.random.randn(50) * noise_level * 2
                axes_diff_v.plot(steps_diff, traj_noisy, color=plt.cm.RdYlGn(k / K), lw=1.5, alpha=0.7)
            axes_diff_v.plot(steps_diff, traj_true, color="#4caf50", lw=3, label="Denoised trajectory (plan)")
            axes_diff_v.set_xlabel("Time step", color="white"); axes_diff_v.set_ylabel("State", color="white")
            axes_diff_v.set_title("Diffusion Planning:\niteratively denoised trajectory", color="white", fontweight="bold")
            axes_diff_v.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            axes_diff_v.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_diff_v); plt.close()

        st.markdown("**Key advantage over Decision Transformer:** diffusion models capture multimodal distributions. If 3 different paths achieve the same goal, diffusion represents all 3. Autoregressive models average between modes — giving trajectories that achieve none of them.")
        st.markdown("**Current limitation:** 100–1000 denoising steps required per plan → slow for real-time replanning. DDIM and consistency models reduce this to 1–5 steps.")

    # ── RL THEORY ─────────────────────────────────────────────────
    with tab_theory:
        _sec("📏", "RL Theory — Mathematical Foundations",
             "PAC-MDP · Regret bounds · Policy gradient convergence — why algorithms work", "#6a1b9a")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**PAC-MDP sample complexity (Rmax):**")
            st.latex(r"O\!\left(\frac{|S||A|}{\varepsilon^2(1-\gamma)^3}\log\frac{1}{\delta}\right)")
            st.markdown("**Bellman optimality equations:**")
            st.latex(r"V^*(s)=\max_a[r(s,a)+\gamma\sum_{s'}p(s'|s,a)V^*(s')]")
            st.markdown("**Regret lower bound (Lai & Robbins 1985):**")
            st.latex(r"\mathbb{E}[\text{Regret}]\geq\sum_{a:\Delta_a>0}\frac{\ln T}{\text{KL}(\mu_a\|\mu^*)}")
            st.markdown("No algorithm can beat O(log T) in K-armed bandits. UCB and Thompson Sampling achieve this.")
        with col2:
            st.markdown("**Policy gradient convergence (α=O(1/√T)):**")
            st.latex(r"\frac{1}{T}\sum_t\|\nabla J(\theta_t)\|^2\leq O(1/\sqrt{T})")
            st.markdown("**TRPO monotonic improvement:**")
            st.latex(r"J(\pi')\geq J(\pi)-\frac{4\gamma\varepsilon}{(1-\gamma)^2}\overline{D}_\text{KL}(\pi'\|\pi)")
            st.markdown("**CQL conservative lower bound:**")
            st.latex(r"\mathbb{E}_\beta[Q^\text{CQL}(s,a)]\leq\mathbb{E}_\beta[Q^\pi(s,a)]")

        # Convergence rate comparison chart
        T = np.arange(1, 1001)
        fig_conv, ax_conv = _fig(1, 1, 11, 4)
        ax_conv.semilogy(T, 1/T, color="#4caf50", lw=2.5, label=r"$O(1/T)$ — full-batch GD (convex)")
        ax_conv.semilogy(T, 1/np.sqrt(T), color="#ffa726", lw=2.5, label=r"$O(1/\sqrt{T})$ — SGD policy gradient")
        ax_conv.semilogy(T, np.log(T)/T, color="#0288d1", lw=2, label=r"$O(\log T / T)$ — variance-reduced PG")
        ax_conv.set_xlabel("Training iterations T", color="white")
        ax_conv.set_ylabel("||∇J||² (gradient norm squared)", color="white")
        ax_conv.set_title("Policy Gradient Convergence Rates: SGD achieves O(1/√T)\n(why large batches help — reduces variance)", color="white", fontweight="bold")
        ax_conv.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_conv.grid(alpha=0.12, which="both")
        plt.tight_layout(); st.pyplot(fig_conv); plt.close()

    # ── BOOKS ─────────────────────────────────────────────────────
    with tab_res:
        _sec("📚", "Books & Deep-Dive Resources",
             "The definitive reading list for every Frontier RL topic", "#546e7a")

        sections = [
            ("🧠 RLHF at Scale", [
                ("Training Language Models to Follow Instructions with Human Feedback (InstructGPT)",
                 "Ouyang et al. (2022) — OpenAI — The paper that launched RLHF at scale",
                 "The original 175B model aligned with RLHF. Full training details, ablations, reward model architecture.",
                 "https://arxiv.org/abs/2203.02155"),
                ("Constitutional AI: Harmlessness from AI Feedback",
                 "Bai et al. (2022) — Anthropic — RLAIF without human annotators",
                 "Derives Constitutional AI from first principles. Section 3 has the full RLAIF pipeline.",
                 "https://arxiv.org/abs/2212.08073"),
                ("Debate: AI Safety via Debate (Irving et al. 2018)",
                 "Irving, Christiano, Amodei — The debate paper",
                 "Proves Nash equilibrium of debate is honest argument. The theoretical foundation for scalable oversight.",
                 "https://arxiv.org/abs/1805.00899"),
            ]),
            ("🌍 World Models + RL", [
                ("DreamerV3: Mastering Diverse Domains through World Models",
                 "Hafner et al. (2023) — SOTA world model",
                 "All innovations: RSSM, symlog, KL balancing, free bits. Required reading for modern MBRL.",
                 "https://arxiv.org/abs/2301.04104"),
                ("World Models (Ha & Schmidhuber 2018)",
                 "Ha & Schmidhuber — First dream-based RL",
                 "The original world models paper. VAE + MDN-RNN + Controller. Elegant and readable.",
                 "https://arxiv.org/abs/1803.10122"),
            ]),
            ("🔍 Exploration · 🛡️ Safe RL", [
                ("Constrained Policy Optimisation (CPO)",
                 "Achiam et al. (2017) — ICML — The standard CMDP algorithm",
                 "Full derivation from TRPO trust region + constraint. Appendix has convergence proof.",
                 "https://arxiv.org/abs/1705.10528"),
                ("Safety Gym: Benchmarks for Safe RL",
                 "Ray, Achiam, Amodei (OpenAI, 2019)",
                 "THE safe RL benchmark. Implements CPO, TRPO-Lagrangian, PPO-Lagrangian. Essential.",
                 "https://openai.com/research/safety-gym"),
                ("First Return, then Explore (Go-Explore)",
                 "Ecoffet et al. (2021) — Nature",
                 "State-of-the-art on hard exploration. 400K+ on Montezuma's Revenge. Archive-based approach.",
                 "https://arxiv.org/abs/2004.12919"),
            ]),
            ("🌐 Foundation Models · 📐 Offline→Online", [
                ("Gato: A Generalist Agent (Reed et al. 2022)",
                 "Reed et al. (2022) — DeepMind — Multi-task transformer for 600+ tasks",
                 "Universal tokenisation of all modalities. Architecture and training details for the 1.2B model.",
                 "https://arxiv.org/abs/2205.06175"),
                ("Algorithm Distillation: Learning to Learn from Demonstrations",
                 "Laskin et al. (2023) — ICLR — In-context RL",
                 "Trains transformer on RL learning histories. Implements credit assignment in forward pass.",
                 "https://arxiv.org/abs/2210.14215"),
                ("Cal-QL: Calibrated Offline RL for Online Fine-Tuning",
                 "Nakamoto et al. (2023) — NeurIPS",
                 "The best offline→online RL method. Calibration term derivation is very clean.",
                 "https://arxiv.org/abs/2303.05479"),
            ]),
            ("📚 Textbooks & Courses", [
                ("Reinforcement Learning: An Introduction (2nd ed.)",
                 "Sutton & Barto (2018) — FREE at incompleteideas.net",
                 "The textbook. Chapters 13–17 cover policy gradients, eligibility traces, and planning.",
                 "http://incompleteideas.net/book/the-book.html"),
                ("Deep Reinforcement Learning (Plaat 2022)",
                 "Plaat — Springer — Graduate textbook covering deep RL comprehensively",
                 "Covers DQN, policy gradients, model-based RL, multi-agent. Good companion to Sutton&Barto.",
                 "https://link.springer.com/book/10.1007/978-3-031-18138-4"),
                ("CS285 Berkeley Deep RL (Levine) — Lectures 18–22",
                 "Levine — Best video course for frontier topics",
                 "RLHF, meta-RL, MARL, safe RL, exploration — all covered at graduate level.",
                 "https://rail.eecs.berkeley.edu/deeprlcourse/"),
            ]),
        ]

        for section, items in sections:
            st.subheader(section)
            for title, authors, why, url in items:
                st.markdown(_book(title, authors, why, url), unsafe_allow_html=True)

    frontier_notes = [
        (tab_ov, "Overview", "frontier_rl_research"),
        (tab_rlhf, "RLHF at Scale", "frontier_rl_research_rlhf_at_scale"),
        (tab_wm, "World Models", "frontier_rl_research_world_models"),
        (tab_exp, "Exploration", "frontier_rl_research_exploration"),
        (tab_safe, "Safe RL", "frontier_rl_research_safe_rl"),
        (tab_fm, "Foundation Models", "frontier_rl_research_foundation_models"),
        (tab_o2o, "Offline to Online", "frontier_rl_research_offline_to_online"),
        (tab_llm, "LLM + RL", "frontier_rl_research_llm_rl"),
        (tab_s2r, "Sim-to-Real", "frontier_rl_research_sim_to_real"),
        (tab_diff, "Diffusion", "frontier_rl_research_diffusion"),
        (tab_theory, "Theory", "frontier_rl_research_theory"),
        (tab_res, "Resources", "frontier_rl_research_resources"),
    ]
    for tab, note_title, note_slug in frontier_notes:
        with tab:
            render_notes(f"Frontier RL Research - {note_title}", note_slug)
