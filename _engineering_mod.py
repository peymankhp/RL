"""_engineering_mod.py — Practical RL Engineering (Tier 3)
Debugging · Reward Design · Distributed RL · Experiment Tracking — with charts, examples, books"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import warnings
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


def _sec(emoji, title, sub, color="#546e7a"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)


def smooth(a, w=8):
    return np.convolve(a, np.ones(w) / w, mode="valid") if len(a) > w else np.array(a, float)


def main_engineering():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1a0e,#0e0e1a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🔧 Practical RL Engineering</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'Tier 3: The engineering knowledge that separates practitioners from theorists. '
        'Debugging training failures, designing robust rewards, scaling to 1000+ workers, '
        'and reproducing results — skills most papers never teach but every practitioner needs.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🐛 RL Debugging",
        "🎯 Reward Design",
        "⚡ Distributed RL",
        "📊 Experiment Tracking",
        "📚 Books & Resources",
    ])
    tab_debug, tab_rew, tab_dist, tab_track, tab_res = tabs

    # ══════════════════════════════════════════════════════════════
    # DEBUGGING
    # ══════════════════════════════════════════════════════════════
    with tab_debug:
        _sec("🐛", "Diagnosing RL Training Failures",
             "The 6 most common failure modes — symptoms, root causes, and fixes", "#ef5350")

        st.markdown(_card("#ef5350", "🐛", "Why RL debugging is uniquely hard",
            """Debugging RL is fundamentally different from debugging supervised learning.
            In supervised learning, high validation loss immediately signals a problem.
            In RL: (1) The reward curve can be flat for millions of steps then suddenly rise —
            was it learning slowly or broken? (2) The policy can perform well during training
            but fail catastrophically during evaluation. (3) The agent may find reward-hacking
            solutions (high reward, wrong behaviour). (4) Gradients can explode silently,
            corrupting all future learning. (5) Entropy collapse leaves the agent unable to
            explore. (6) The deadly triad (bootstrapping + function approximation + off-policy)
            causes Q-value divergence without obvious signals.
            The only defence: log EVERYTHING from step 1 — reward, entropy, Q-values,
            gradient norms, TD error, explained variance. Most RL bugs are invisible without
            comprehensive diagnostic logging."""), unsafe_allow_html=True)

        # Simulate 6 failure modes
        np.random.seed(42)
        T = 500
        t = np.arange(T)
        fig_fail, axes_fail = _fig(2, 3, 17, 8)
        failure_data = [
            (axes_fail[0, 0], "Q-value Explosion",
             np.minimum(1e6, np.exp(t / 130)) + np.random.randn(T) * 100,
             "#ef5350", "Q grows exponentially → NaN"),
            (axes_fail[0, 1], "Entropy Collapse",
             np.maximum(0.01, 2.0 * np.exp(-t / 80) + np.random.randn(T) * 0.04),
             "#ffa726", "Policy becomes deterministic → stuck"),
            (axes_fail[0, 2], "Good Training (Reference)",
             np.minimum(195, t * 0.45 + np.random.randn(T) * 8),
             "#4caf50", "Reward rises monotonically → correct"),
            (axes_fail[1, 0], "TD Loss Explosion",
             np.minimum(1e4, np.exp(t / 200)) + np.random.randn(T) * 10,
             "#ad1457", "Deadly triad: bootstrap+approx+off-policy"),
            (axes_fail[1, 1], "Gradient Explosion",
             np.where(t < 250, np.abs(np.random.randn(T) * 0.3 + 0.5),
                      np.minimum(50, np.exp((t - 250) / 70))),
             "#7c4dff", "Missing gradient clipping → divergence"),
            (axes_fail[1, 2], "Reward Hacking",
             np.minimum(180, t * 0.38) + np.random.randn(T) * 3,
             "#0288d1", "Train reward up, eval task metric flat"),
        ]
        for ax, title, data, col, desc in failure_data:
            ax.plot(smooth(data, 12), color=col, lw=2)
            ax.set_title(f"{title}\n{desc}", color="white", fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Step", color="white", fontsize=7)
            ax.grid(alpha=0.1)
        plt.suptitle("Common RL Training Failure Signatures — Learn to Recognise Each",
                     color="white", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_fail)
        plt.close()

        st.subheader("🩺 Failure Mode Diagnosis Guide")
        failure_modes = [
            {
                "name": "Q-value Explosion",
                "color": "#ef5350",
                "symptom": "Q-values grow to 1e6+, rewards collapse, agent takes terminal actions",
                "cause": "Missing gradient clipping, lr too high, target network not updated, overestimation bias",
                "fix": "Clip gradients to norm 0.5–1.0; reduce lr by 10×; check target network update freq; use Double DQN",
                "metric": "Log max(|Q-values|) per batch — alert if >1000",
                "code": "# Fix: gradient clipping in PyTorch\ntorch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
            },
            {
                "name": "Policy Entropy Collapse",
                "color": "#ffa726",
                "symptom": "H(π)→0, agent always picks same action, reward plateaus or crashes",
                "cause": "lr too high early, entropy coefficient too low, sparse reward with no exploration",
                "fix": "Increase entropy coeff c2 (0.01→0.05); reduce lr; add learning rate warmup schedule",
                "metric": "Monitor H(π) per episode — alert if < 0.1 for discrete or < 1.0 for continuous",
                "code": "# PPO entropy bonus\nentropy_loss = -ent_coef * entropy.mean()\ntotal_loss = policy_loss + value_loss + entropy_loss",
            },
            {
                "name": "Reward Hacking (Goodhart's Law)",
                "color": "#7c4dff",
                "symptom": "Training reward ↑ but evaluation task metric flat or ↓",
                "cause": "Reward function has unintended optima; agent found shortcut",
                "fix": "Decouple train reward from eval metric; adversarial evaluation; RLHF reward modelling",
                "metric": "Always log separate eval metric distinct from training reward",
                "code": "# Always track both:\nwandb.log({'train_reward': r, 'eval_task_metric': success_rate})",
            },
            {
                "name": "TD Loss Explosion (Deadly Triad)",
                "color": "#ad1457",
                "symptom": "TD loss explodes, Q-values diverge, training becomes unstable",
                "cause": "Bootstrapping + function approximation + off-policy = unstable combination",
                "fix": "Target networks (τ=0.005 soft update); reduce γ; n-step returns ≤5; prioritise replay",
                "metric": "Monitor TD loss per batch — should plateau or decrease, never explode",
                "code": "# Soft target network update\nfor p, tp in zip(online.parameters(), target.parameters()):\n    tp.data.copy_(tau*p.data + (1-tau)*tp.data)",
            },
            {
                "name": "Gradient Explosion/Vanishing",
                "color": "#6a1b9a",
                "symptom": "Loss NaN, sudden performance drop, weights become very large or very small",
                "cause": "Deep networks, high lr, poor initialisation, missing normalisation",
                "fix": "Gradient clipping; He initialisation; layer norm; reduce lr; check for NaN after each update",
                "metric": "Log gradient norm — should be stable around 0.1–2.0, not growing",
                "code": "# Monitor gradient norm\nfor name, param in model.named_parameters():\n    if param.grad is not None:\n        wandb.log({f'grad/{name}': param.grad.norm().item()})",
            },
            {
                "name": "Catastrophic Forgetting",
                "color": "#0288d1",
                "symptom": "Performance on early part of training suddenly drops when distribution shifts",
                "cause": "Replay buffer too small, correlated sampling, fast policy updates",
                "fix": "Larger replay buffer (1M+), uniform random sampling, slower lr for critic",
                "metric": "Evaluate on held-out rollouts from 100K steps ago periodically",
                "code": "# Large replay buffer\nbuffer = ReplayBuffer(capacity=1_000_000)\n# Sample uniformly — no temporal correlation\nbatch = buffer.sample(n=256)  # random indices",
            },
        ]

        for fm in failure_modes:
            with st.expander(f"🔴 {fm['name']}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div style="background:{fm["color"]}18;border-left:3px solid {fm["color"]};'
                                f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                                f'<b style="color:{fm["color"]}">🩺 Symptom:</b><br>'
                                f'<span style="color:#b0b0cc;font-size:.9rem">{fm["symptom"]}</span></div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div style="background:#2a1a0a;border-left:3px solid #ffa726;'
                                f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                                f'<b style="color:#ffa726">🔍 Root cause:</b><br>'
                                f'<span style="color:#b0b0cc;font-size:.9rem">{fm["cause"]}</span></div>',
                                unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div style="background:#0a2a0a;border-left:3px solid #4caf50;'
                                f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                                f'<b style="color:#4caf50">🔧 Fix:</b><br>'
                                f'<span style="color:#b0b0cc;font-size:.9rem">{fm["fix"]}</span></div>',
                                unsafe_allow_html=True)
                    st.markdown(f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
                                f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                                f'<b style="color:#0288d1">📊 Monitor:</b><br>'
                                f'<span style="color:#b0b0cc;font-size:.9rem">{fm["metric"]}</span></div>',
                                unsafe_allow_html=True)
                st.code(fm["code"], language="python")

        st.subheader("📋 Essential Metrics to Log from Day 1")
        st.dataframe(pd.DataFrame({
            "Metric": ["Episode reward mean ± std", "Policy entropy H(π)",
                       "Value loss / TD loss", "Max |Q-value|",
                       "Gradient norm", "KL divergence (PPO)",
                       "Explained variance of V", "Episode length"],
            "Why it matters": [
                "Primary task performance — the thing you actually care about",
                "How exploratory the policy is — collapse = stuck forever",
                "How well critic fits targets — explosion = divergence",
                "Detect Q-explosion before it crashes training",
                "Detect exploding/vanishing gradients immediately",
                "How far PPO policy moved (>0.05 = too large)",
                "How well V(s) explains actual returns (< 0 = useless)",
                "Decreasing to 1 step = dying early = something wrong",
            ],
            "Warning threshold": [
                "Flat for >500K steps",
                "< 0.1 (discrete) / < 1.0 (continuous)",
                "Increasing trend over time",
                "> 1000",
                "> 2.0 per update",
                "> 0.1 per update",
                "< 0.0 (V gives no signal)",
                "Monotonically decreasing to 1",
            ],
        }), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # REWARD DESIGN
    # ══════════════════════════════════════════════════════════════
    with tab_rew:
        _sec("🎯", "Reward Design — The Most Underrated Skill in Applied RL",
             "Potential-based shaping · RLHF pipeline · Goodhart's Law — a bad reward is the #1 failure mode", "#ffa726")

        st.markdown(_card("#ffa726", "🎯", "Why reward design is harder than the algorithm",
            """In academic RL benchmarks, reward functions are carefully crafted by domain experts.
            In real applications, specifying what you actually want an agent to optimise is
            extraordinarily difficult. Goodhart's Law: 'When a measure becomes a target, it ceases
            to be a good measure.' Famous examples: boat racing agent learned to spin in circles
            collecting pickup items rather than finishing the race (high score, wrong behaviour);
            robot learned to flip over to avoid falling rather than walk upright; LLM learned to
            write persuasive-sounding but factually wrong content because human raters couldn't
            verify facts. Every real-world RL deployment starts with reward design, and most early
            failures are reward design failures, not algorithm failures.
            The solutions: (1) potential-based reward shaping for dense feedback; (2) reward modelling
            from demonstrations (inverse RL); (3) RLHF for complex human preferences."""), unsafe_allow_html=True)

        st.subheader("1. Potential-Based Reward Shaping — Safe Dense Rewards")
        st.markdown("**The core theorem** (Ng et al. 1999): you can add any potential-based shaping bonus without changing the optimal policy:")
        st.latex(r"r'(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s) \quad\text{(shaping bonus)}")
        st.markdown("**Proof that optimal policy is unchanged:**")
        st.latex(r"Q^*(s,a;\,r') = Q^*(s,a;\,r) + \Phi(s) \quad\text{(value functions differ only by the potential)}")
        st.markdown("Since the policy π*(s) = argmax_a Q*(s,a) — and the argmax is unchanged when we add a constant Φ(s) to all actions — the optimal policy is identical.")
        st.markdown("**Common choices for Φ(s):**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distance to goal (navigation):**")
            st.latex(r"\Phi(s) = -\|s_\text{pos} - s_\text{goal}\|_2")
            st.markdown("Agent gets bonus for moving closer to goal. Provides dense gradient signal without changing what the optimal goal-reaching policy is.")
            st.markdown("**State value function (theory):**")
            st.latex(r"\Phi(s) = V^*(s)")
            st.markdown("Using the optimal value as potential makes the shaped reward identical to the TD error δ_t. This is why TD methods work — they implicitly use this potential.")
        with col2:
            # Visualise potential-based shaping
            x = np.linspace(0, 10, 100)
            goal = 8.0
            phi_dist = -(np.abs(x - goal))
            r_sparse = np.where(np.abs(x - goal) < 0.5, 1.0, 0.0)
            # Shaping bonus: γΦ(s') - Φ(s) — compute at same points using diff
            shaping = 0.99 * np.roll(phi_dist, -1) - phi_dist
            shaping[-1] = 0  # boundary: no next state at end
            r_shaped = r_sparse + shaping
            fig_pot, ax_pot = _fig(1, 1, 5.5, 4)
            ax_pot.plot(x, r_sparse, color="#546e7a", lw=2, ls="--", label="Sparse reward (only at goal)")
            ax_pot.plot(x, r_shaped / 5 + 0.1, color="#ffa726", lw=2.5, label="Shaped reward (potential-based)")
            ax_pot.axvline(goal, color="#4caf50", ls="--", lw=1.5, label=f"Goal={goal}")
            ax_pot.set_xlabel("Agent position", color="white")
            ax_pot.set_ylabel("Reward signal", color="white")
            ax_pot.set_title("Potential shaping:\ndense signal, same optimal policy", color="white", fontweight="bold")
            ax_pot.legend(facecolor=CARD, labelcolor="white", fontsize=7)
            ax_pot.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_pot); plt.close()

        st.subheader("2. Common Reward Design Mistakes with Examples")
        mistakes = [
            ("Sparse reward on long-horizon tasks",
             "Robot never receives reward. Zero gradient signal. Nothing is learned.",
             "Add distance shaping Φ(s) = -||pos - goal||. Or use HER to relabel failed trajectories."),
            ("Reward scale mismatch",
             "+1000 for task completion, −0.01 per step. Agent ignores tiny penalties.",
             "Normalise all reward components to similar magnitude (±1). Check: does every term matter?"),
            ("Negative reward only",
             "Agent learns to end episodes immediately to minimise total negative reward.",
             "Include survival bonus (+0.1/step) or shift reward scale to have positive components."),
            ("Terminal reward only",
             "No gradient from intermediate steps. Value function cannot propagate signal backwards.",
             "Add intermediate milestones. Reward sub-goal completion at each stage."),
            ("Reward based on position not velocity",
             "Agent oscillates at goal boundary to maximise position-based reward.",
             "Reward change in position (progress toward goal) rather than absolute position."),
            ("Reward hacking via unintended actions",
             "Robot learns to flip over to avoid falling penalty rather than walk. Boat spins for pickups.",
             "Adversarial evaluation: actively search for ways the agent might cheat. Add multiple metrics."),
        ]
        for mistake, consequence, fix in mistakes:
            st.markdown(f'<div style="background:#2a1a0a;border-left:3px solid #ffa726;'
                        f'padding:.6rem 1rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                        f'<b style="color:#ffa726">❌ {mistake}</b><br>'
                        f'<span style="color:#ef9a9a;font-size:.85rem">→ {consequence}</span><br>'
                        f'<span style="color:#a5d6a7;font-size:.85rem">✅ Fix: {fix}</span></div>',
                        unsafe_allow_html=True)

        st.subheader("3. RLHF Reward Model — Learning Human Preferences")
        st.markdown("When the reward is too complex to specify analytically, learn it from human comparisons:")
        st.latex(r"P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \quad\text{(Bradley-Terry model)}")
        st.latex(r"\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))\right]")

        # RLHF pipeline diagram
        np.random.seed(42); T = 300
        sft_phase = np.minimum(0.6, 0.003 * np.arange(T)) + np.random.randn(T) * 0.02
        rlhf_phase = np.minimum(0.95, 0.003 * np.arange(T) + 0.6 * (np.arange(T) > 100) * 0.002) + np.random.randn(T) * 0.02
        fig_rlhf, ax_rlhf = _fig(1, 1, 11, 4)
        ax_rlhf.plot(smooth(sft_phase, 20), color="#546e7a", lw=2.5, label="SFT only (no RL)")
        ax_rlhf.plot(smooth(rlhf_phase, 20), color="#ffa726", lw=2.5, label="SFT → RLHF with learned reward")
        ax_rlhf.axvline(100, color="#ffa726", ls="--", lw=1.5, alpha=0.7, label="RLHF training starts")
        ax_rlhf.set_xlabel("Training iteration", color="white")
        ax_rlhf.set_ylabel("Human preference score (proxy)", color="white")
        ax_rlhf.set_title("RLHF Pipeline: SFT baseline then PPO on learned reward model",
                          color="white", fontweight="bold")
        ax_rlhf.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_rlhf.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_rlhf); plt.close()

    # ══════════════════════════════════════════════════════════════
    # DISTRIBUTED RL
    # ══════════════════════════════════════════════════════════════
    with tab_dist:
        _sec("⚡", "Distributed Reinforcement Learning",
             "IMPALA · Ape-X · R2D2 · EnvPool — scale to 1000+ parallel workers", "#0288d1")

        st.markdown(_card("#0288d1", "⚡", "Why distributed RL is necessary at scale",
            """Single-machine RL is limited by data collection speed and GPU utilisation.
            A single worker on Atari gets ~1,000 frames/second. Getting to 50M frames (DQN training)
            takes 14 hours of pure data collection time. For complex environments like StarCraft II
            or physics simulation, data collection is 10–100× slower.
            Distributed RL separates data collection from model training:
            hundreds of 'actor' processes run environment simulations in parallel on CPUs,
            sending experience to a central 'learner' GPU that updates the model continuously.
            Actors periodically download the latest weights. This achieves 100,000+ frames/second
            while keeping the GPU fully utilised.
            The key engineering challenge: actors run old policy while learner updates — creating
            an off-policy gap. IMPALA handles this with V-trace corrections. Ape-X uses PER.
            R2D2 adds recurrence for memory. EnvPool gives 50,000 steps/second on a single machine."""),
            unsafe_allow_html=True)

        st.subheader("1. IMPALA — V-trace Off-Policy Correction")
        st.markdown("IMPALA (Espeholt et al. 2018) corrects for the staleness gap between actor policy μ (old) and learner policy π (current):")
        st.latex(r"v_s = V(x_s) + \sum_{t=s}^{s+n-1}\gamma^{t-s}\left(\prod_{i=s}^{t-1}c_i\right)\delta_t v")
        st.latex(r"\delta_t v = \rho_t(r_t+\gamma v_{s+1}-V(x_t)) \quad\rho_t=\min\!\left(\bar\rho,\frac{\pi(a_t|x_t)}{\mu(a_t|x_t)}\right),\;c_t=\min\!\left(\bar c,\frac{\pi}{\mu}\right)")
        st.markdown("**Symbol decoder:** ρ_t = importance sampling ratio clipped to ρ̄ (prevents variance explosion). c_t = trace coefficient clipped to c̄. The clipping introduces bias but keeps gradients stable. Typical: ρ̄=1, c̄=1 (only one-step correction).")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**IMPALA throughput (500 CPUs):**")
            st.markdown("- Single-machine DQN: ~1,000 fps")
            st.markdown("- IMPALA (500 actors): ~250,000 fps")
            st.markdown("- **250× speedup from parallelism**")
            st.markdown("**Ape-X (distributed PER):**")
            st.markdown("Multiple actors → central prioritised replay buffer → single learner. No V-trace needed (purely off-policy). 50× throughput of single DQN.")
        with col2:
            # Throughput comparison
            systems = ["Single DQN", "A3C (16 workers)", "Ape-X (64)", "IMPALA (500)", "SEED RL (2000)"]
            fps = [1_000, 10_000, 50_000, 250_000, 2_500_000]
            colors_bar = ["#546e7a", "#0288d1", "#7c4dff", "#e65100", "#4caf50"]
            fig_dist, ax_dist = _fig(1, 1, 5.5, 4)
            ax_dist.barh(systems, np.log10(fps), color=colors_bar, alpha=0.85)
            for i, (s, f) in enumerate(zip(systems, fps)):
                ax_dist.text(np.log10(f) + 0.05, i, f"{f:,} fps", va="center", color="white", fontsize=7.5)
            ax_dist.set_xlabel("log₁₀(Frames per second)", color="white")
            ax_dist.set_title("Distributed RL throughput comparison", color="white", fontweight="bold")
            ax_dist.grid(alpha=0.12, axis="x")
            plt.tight_layout(); st.pyplot(fig_dist); plt.close()

        st.subheader("2. EnvPool — 50,000 Steps/Second Without Distributed Infrastructure")
        st.code("""
# EnvPool: run 1024 envs in parallel with C++ backend
import envpool
import numpy as np

# Create 1024 parallel Atari environments
envs = envpool.make("Atari-v5", env_type="gym",
                    num_envs=1024, episodic_life=True)
# Steps all 1024 envs simultaneously in C++ — ~50,000 steps/second
obs, rew, done, info = envs.step(actions)  # actions.shape = (1024,)

# Training loop for PPO with vectorised envs
obs = envs.reset()  # shape: (1024, 4, 84, 84) for Atari
for _ in range(n_steps):
    actions = policy(obs)  # batch inference: (1024,)
    obs, rew, done, info = envs.step(actions)
    # 1024 experiences collected per call → ~50K fps on one machine
""", language="python")

        st.markdown(_insight("For most research projects: EnvPool on a single machine (50K fps) is sufficient. 50M Atari frames takes only ~17 minutes. Only use true distributed RL (IMPALA/Ape-X) when you need millions of steps per hour for very long training runs or have access to a compute cluster."), unsafe_allow_html=True)

        # Show speedup from parallelism
        n_workers = [1, 4, 16, 64, 256, 1024]
        speedup_linear = n_workers
        speedup_actual = [1, 3.5, 12, 45, 170, 600]  # overhead from communication
        fig_sp, ax_sp = _fig(1, 1, 10, 4)
        ax_sp.loglog(n_workers, speedup_linear, color="#546e7a", lw=2, ls="--", label="Linear speedup (ideal)")
        ax_sp.loglog(n_workers, speedup_actual, color="#0288d1", lw=2.5, marker="o", ms=7, label="Actual speedup (communication overhead)")
        ax_sp.set_xlabel("Number of parallel workers", color="white")
        ax_sp.set_ylabel("Speedup vs single worker", color="white")
        ax_sp.set_title("Distributed RL Scalability: diminishing returns from communication overhead",
                       color="white", fontweight="bold")
        ax_sp.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_sp.grid(alpha=0.12, which="both")
        plt.tight_layout(); st.pyplot(fig_sp); plt.close()

    # ══════════════════════════════════════════════════════════════
    # EXPERIMENT TRACKING
    # ══════════════════════════════════════════════════════════════
    with tab_track:
        _sec("📊", "Experiment Tracking & Reproducibility",
             "W&B · MLflow · Seeds · Evaluation protocol · Optuna — most RL results aren't reproducible without this", "#00897b")

        st.markdown(_card("#00897b", "📊", "Why reproducibility is a crisis in RL research",
            """Henderson et al. (2018) showed that PPO on HalfCheetah varied from 2000 to 6000 total
            reward across 5 random seeds — a 3× difference. This is not noise; it is structural
            variance in RL training. An algorithm appearing to outperform a baseline by 20% might
            simply be luckier with its seed. The RL community has gradually adopted better practices:
            (1) Always report mean ± stderr over at least 5 seeds; (2) Use a separate evaluation
            policy (greedy, no ε) on 100+ episodes; (3) Log the full hyperparameter config;
            (4) Save model checkpoints with random state; (5) Report environment steps not wall-clock.
            Experiment tracking tools like W&B and MLflow make this easier by automatically logging
            everything and enabling cross-run comparison."""), unsafe_allow_html=True)

        st.subheader("1. Essential Reproducibility Checklist")
        checks = [
            ("🌱", "Set ALL random seeds", "Set numpy, torch, random, and environment seeds. Log the seed used so any run can be reproduced.",
             "import torch, numpy as np, random\ntorch.manual_seed(seed); np.random.seed(seed)\nrandom.seed(seed); env.seed(seed)"),
            ("📊", "Separate evaluation protocol", "Use greedy policy (no ε, no stochastic sampling) on 100 episodes. Log separately from training.",
             "def evaluate(policy, env, n_episodes=100):\n    rewards = []\n    for _ in range(n_episodes):\n        s = env.reset(); G = 0; done = False\n        while not done:\n            a = policy.act_greedy(s)  # no epsilon!\n            s, r, done, _ = env.step(a); G += r\n        rewards.append(G)\n    return np.mean(rewards), np.std(rewards)"),
            ("💾", "Log full hyperparameter config", "Every run logs its complete config automatically. Never lose the settings that produced a good run.",
             "import wandb\nwandb.init(project='rl-experiment', config=vars(args))\n# All args automatically logged — fully reproducible"),
            ("🔢", "Run ≥5 seeds, report mean±stderr", "Report mean±stderr, not mean±std. Include n in the report.",
             "results = [train(seed=s) for s in range(5)]\nprint(f'{np.mean(results):.0f} ± {np.std(results)/np.sqrt(5):.0f} (n=5)')"),
            ("📏", "Report env steps not wall-clock time", "Wall-clock time varies by hardware. Environment steps are comparable across papers and hardware.",
             "wandb.log({'reward': ep_reward, 'env_steps': total_steps})  # not time!"),
        ]
        for icon, title, desc, code in checks:
            with st.expander(f"{icon} {title}"):
                st.markdown(f'<span style="color:#b0b0cc;font-size:.9rem">{desc}</span>', unsafe_allow_html=True)
                st.code(code, language="python")

        st.subheader("2. Hyperparameter Search with Optuna")
        st.code("""
import optuna
from stable_baselines3 import PPO

def objective(trial):
    # Optuna samples hyperparameters from specified distributions
    lr           = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    ent_coef     = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    gae_lambda   = trial.suggest_float("gae_lambda", 0.8, 0.99)
    n_steps      = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    n_epochs     = trial.suggest_int("n_epochs", 3, 10)
    clip_range   = trial.suggest_float("clip_range", 0.1, 0.4)

    model = PPO("MlpPolicy", "HalfCheetah-v4",
                learning_rate=lr, ent_coef=ent_coef,
                gae_lambda=gae_lambda, n_steps=n_steps,
                n_epochs=n_epochs, clip_range=clip_range)
    model.learn(500_000)

    # Evaluate over 20 episodes with greedy policy
    rewards = [evaluate_greedy(model) for _ in range(20)]
    return np.mean(rewards)  # Optuna maximises this

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)
print("Best hyperparameters:", study.best_params)
# Access full results: study.trials_dataframe()
""", language="python")

        # Show typical Optuna hyperparameter importance
        hparams = ["learning_rate", "n_steps", "gae_lambda", "ent_coef", "n_epochs", "clip_range"]
        importance = [0.38, 0.22, 0.18, 0.12, 0.06, 0.04]
        fig_opt, ax_opt = _fig(1, 1, 10, 4)
        bars_opt = ax_opt.barh(hparams, importance,
                               color=["#e65100", "#0288d1", "#7c4dff", "#00897b", "#546e7a", "#2a2a3e"],
                               alpha=0.85)
        for bar, imp in zip(bars_opt, importance):
            ax_opt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                       f"{imp:.0%}", va="center", color="white", fontsize=9)
        ax_opt.set_xlabel("Relative importance", color="white")
        ax_opt.set_title("PPO Hyperparameter Importance (Optuna analysis, HalfCheetah-v4)\n"
                         "Learning rate matters most — spend search budget there first",
                         color="white", fontweight="bold")
        ax_opt.grid(alpha=0.12, axis="x")
        plt.tight_layout(); st.pyplot(fig_opt); plt.close()

        st.markdown(_insight("Use Optuna with W&B together: each Optuna trial logs to W&B automatically, giving a full hyperparameter importance analysis and parallel coordinate plot. This combination finds optimal PPO hyperparameters for a new environment in 50 trials (~3 GPU hours), saving days of manual tuning."), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # BOOKS
    # ══════════════════════════════════════════════════════════════
    with tab_res:
        _sec("📚", "Books & Deep-Dive Resources",
             "Best books, papers, and tools for practical RL engineering", "#546e7a")

        st.subheader("🐛 RL Debugging")
        for item in [
            _book("Deep Reinforcement Learning That Matters",
                  "Henderson et al. (2018) — NeurIPS Reproducibility Workshop",
                  "The paper that exposed the reproducibility crisis. PPO results varying 3× across seeds. Required reading.",
                  "https://arxiv.org/abs/1709.06560"),
            _book("Implementation Matters in Deep RL (PPO implementation details)",
                  "Huang et al. (2022) — ICLR Blogpost — 13 implementation tricks for PPO",
                  "Shows that 13 implementation details (gradient clipping, value normalisation, etc.) account for PPO's performance. Read the GitHub too.",
                  "https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/"),
        ]:
            st.markdown(item, unsafe_allow_html=True)

        st.subheader("🎯 Reward Design")
        for item in [
            _book("Reward Misspecification and Specification Gaming",
                  "Krakovna et al. (2020) — DeepMind Blog",
                  "Comprehensive list of reward hacking examples from real RL experiments. Sobering reading.",
                  "https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/"),
            _book("Reward Shaping: When Can It Harm? (Ng et al. 1999)",
                  "Ng, Harada, Russell (1999) — ICML — The potential-based shaping theorem",
                  "Proves that ONLY potential-based shaping preserves optimal policy. All other shaping changes optimality.",
                  "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf"),
        ]:
            st.markdown(item, unsafe_allow_html=True)

        st.subheader("⚡ Distributed RL")
        for item in [
            _book("IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures",
                  "Espeholt et al. (2018) — ICML — The V-trace distributed RL algorithm",
                  "V-trace derivation, architecture, and scalability results. 250,000 fps on 500 CPUs.",
                  "https://arxiv.org/abs/1802.01561"),
            _book("Ape-X: Distributed Prioritised Experience Replay",
                  "Horgan et al. (2018) — ICLR — Distributed PER with many actors",
                  "Simple and effective: many actors → central prioritised replay → one learner. 50× speedup.",
                  "https://arxiv.org/abs/1803.00933"),
        ]:
            st.markdown(item, unsafe_allow_html=True)

        st.subheader("📊 Experiment Tracking Tools")
        for icon, title, desc, url in [
            ("💻", "Weights & Biases (W&B)",
             "The most widely used RL experiment tracker. Auto-logs metrics, configs, system stats. Free for individuals.",
             "https://wandb.ai"),
            ("💻", "Optuna — Hyperparameter Optimisation",
             "TPE sampler + pruning + W&B integration. Best open-source HPO tool for RL.",
             "https://optuna.org"),
            ("💻", "CleanRL — Clean RL Implementations",
             "Single-file, highly readable PPO, DQN, SAC, TD3 implementations with W&B logging. Best starting point.",
             "https://github.com/vwxyzjn/cleanrl"),
            ("📄", "Reporting RL Results — NeurIPS Checklist",
             "The official NeurIPS reproducibility checklist for RL experiments. Use as a template for all papers.",
             "https://reproducibility-challenge.github.io/neurips2023/"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)
