"""
_engineering_mod.py — Practical RL Engineering (Tier 3)
Covers: RL Debugging · Reward Design · Distributed RL · Experiment Tracking
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
        ax.set_facecolor(DARK); ax.tick_params(colors="#9e9ebb", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    return fig, axes

def _card(color, icon, title, body):
    return (f'<div style="background:{color}18;border-left:4px solid {color};'
            f'padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin-bottom:.9rem">'
            f'<b>{icon} {title}</b><br>{body}</div>')

def _warn(text):
    return (f'<div style="background:#2a1a0a;border-left:3px solid #ffa726;'
            f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem">'
            f'⚠️ {text}</div>')

def _insight(text):
    return (f'<div style="background:#0a1a2a;border-left:3px solid #0288d1;'
            f'padding:.7rem 1rem;border-radius:0 8px 8px 0;margin:.6rem 0;font-size:.93rem">'
            f'💡 {text}</div>')

def _sec(emoji, title, sub, color="#546e7a"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def main_engineering():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1a0e,#0e0e1a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🔧 Practical RL Engineering</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'Tier 3: The engineering knowledge that separates practitioners from theorists. '
        'Debugging failures, designing rewards, scaling to 1000+ workers, and reproducing results — '
        'the skills that most papers never teach but every practitioner needs.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🐛 RL Debugging",
        "🎯 Reward Design",
        "⚡ Distributed RL",
        "📊 Experiment Tracking",
    ])
    (tab_debug, tab_rew, tab_dist, tab_track) = tabs

    # ── DEBUGGING ─────────────────────────────────────────────────────────
    with tab_debug:
        _sec("🐛","Diagnosing RL Training Failures",
             "The 6 most common failure modes — each with symptoms, causes, and fixes","#ef5350")

        st.markdown(_card("#ef5350","🐛","Why RL debugging is uniquely hard",
            """Debugging RL is fundamentally different from debugging supervised learning. In supervised
            learning, a high validation loss tells you clearly that something is wrong. In RL, the reward
            curve can be flat for 10 million steps, then suddenly start increasing — was it learning
            slowly, or did it just break and get lucky? The policy can appear to perform well during
            training but catastrophically fail at evaluation. The agent might find a reward-hacking
            solution that gets high reward without actually solving the task. Gradient norms can explode
            silently, corrupting all future learning. The value function can become wildly overoptimistic
            (Q-value explosion), causing the policy to take actions that lead to instant termination.
            Entropy can collapse to near-zero, leaving the agent unable to explore and stuck permanently
            in a suboptimal behaviour. Unlike supervised learning where the data is fixed and bugs are
            reproducible, RL bugs are often stochastic and disappear when you change the random seed.
            The only defence is systematic monitoring of key diagnostic metrics throughout training —
            ideally from the very first run. This section covers the 6 most common failure modes,
            their signatures in your training logs, and what to do when you see them."""), unsafe_allow_html=True)

        failure_modes = [
            {
                "name": "Q-value explosion",
                "color": "#ef5350",
                "symptom": "Q-values grow to 1e6+, rewards collapse, policy takes terminal actions",
                "cause": "Missing gradient clipping, bad learning rate, target network not updated",
                "fix": "Clip gradients to 0.5–1.0, reduce lr, check target network update frequency",
                "metric": "Log max(|Q-values|) per batch; alert if >100",
                "viz": "Q-value divergence: exponential growth then NaN"
            },
            {
                "name": "Policy collapse (entropy collapse)",
                "color": "#ffa726",
                "symptom": "Policy entropy → 0, agent always picks same action, reward drops or stagnates",
                "cause": "Learning rate too high, entropy coefficient too low, reward signal too sparse",
                "fix": "Increase entropy coefficient c2, reduce lr, add learning rate schedule",
                "metric": "Monitor H(π) = -Σ π log π; should stay above 0.1 for discrete, 1.0 for continuous",
                "viz": "Sharp entropy drop followed by reward plateau"
            },
            {
                "name": "Reward hacking",
                "color": "#7c4dff",
                "symptom": "Reward increases but agent solves the wrong problem (Goodhart's Law)",
                "cause": "Reward function has unintended optima; agent is smarter than the reward designer",
                "fix": "Evaluate with multiple metrics; use adversarial evaluation; RLHF reward modelling",
                "metric": "Decouple training reward from evaluation metric; monitor both",
                "viz": "Training reward up, evaluation metric flat or down"
            },
            {
                "name": "Catastrophic forgetting",
                "color": "#0288d1",
                "symptom": "Agent learns skill A, then learns skill B, then forgets A",
                "cause": "Distribution shift in replay buffer, too small buffer, correlated samples",
                "fix": "Larger replay buffer, PER, population-based training, EWC regularisation",
                "metric": "Evaluate on held-out tasks from earlier in training periodically",
                "viz": "Reward on task A drops when training shifts to task B"
            },
            {
                "name": "Deadly triad divergence",
                "color": "#ad1457",
                "symptom": "TD loss explodes, Q-values diverge, training becomes unstable",
                "cause": "Function approximation + bootstrapping + off-policy = unstable combination",
                "fix": "Target networks (slow updates), reduce discount γ, use n-step returns conservatively",
                "metric": "Monitor TD loss per batch; should decrease or plateau, never explode",
                "viz": "TD loss explodes exponentially"
            },
            {
                "name": "Gradient starvation (sparse reward)",
                "color": "#558b2f",
                "symptom": "Policy barely changes over millions of steps; reward stays at zero",
                "cause": "Reward never received; gradient signal is zero; agent cannot learn",
                "fix": "Reward shaping, HER, intrinsic motivation (RND/ICM), curriculum learning",
                "metric": "Track % of episodes with any positive reward; if <1%, reshape reward",
                "viz": "Reward flat at 0 for 10M+ steps"
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

        st.divider()
        st.subheader("📊 Essential Metrics to Log from Day 1")
        st.dataframe(pd.DataFrame({
            "Metric": ["Episode reward (mean ± std)", "Policy entropy H(π)",
                       "Value loss / TD loss", "Max |Q-value|",
                       "Policy gradient norm", "KL divergence (old vs new)",
                       "Explained variance of value fn", "Episode length"],
            "What it tells you": [
                "Primary task performance signal",
                "How exploratory the policy is (collapse → stuck)",
                "How well the critic is fitting targets",
                "Detect Q-value explosion early",
                "Detect exploding/vanishing gradients",
                "How far PPO policy moves per update (>0.05 → too large)",
                "How well V(s) explains actual returns (>0.5 is good)",
                "Whether agent is dying early or getting stuck",
            ],
            "Warning level": [
                "Flat for >1M steps",
                "< 0.1 (discrete) or < 1.0 (continuous)",
                "Increasing over time",
                "> 1000",
                "> 1.0",
                "> 0.1 per update",
                "< 0 (value fn is useless)",
                "Steadily decreasing to 1 step",
            ],
        }), use_container_width=True, hide_index=True)

        # Simulate different failure modes
        np.random.seed(42)
        t = np.arange(500)
        fig_debug, axes = _fig(2, 3, 16, 7)
        plots = [
            (axes[0,0], "Q-value explosion", np.exp(t/200)*np.random.randn(500)*0.5+np.exp(t/200), "#ef5350"),
            (axes[0,1], "Entropy collapse", np.maximum(0.05, 2.0*np.exp(-t/150)+np.random.randn(500)*0.05), "#ffa726"),
            (axes[0,2], "Good training (reward)", np.where(t<100,0,np.minimum(200,(t-100)*0.8))+np.random.randn(500)*5, "#4caf50"),
            (axes[1,0], "TD loss exploding", np.exp(t/300)+np.random.randn(500)*0.1, "#ad1457"),
            (axes[1,1], "Gradient norm", np.where(t<200, np.random.randn(500)*0.3+0.5, np.exp((t-200)/100)), "#0288d1"),
            (axes[1,2], "Reward hacking", np.minimum(180, t*0.4)+np.random.randn(500)*3, "#7c4dff"),
        ]
        for ax, title, data, col in plots:
            ax.plot(t, data, color=col, lw=1.5)
            ax.set_title(title, color="white", fontsize=9, fontweight="bold")
            ax.set_xlabel("Step", color="white", fontsize=7)
            ax.grid(alpha=0.1)
        axes[0,2].axhline(195, color="white", ls="--", lw=0.8, alpha=0.5)
        axes[1,2].set_title("Reward hacking\n(task metric ≠ reward)", color="#7c4dff", fontsize=9, fontweight="bold")
        plt.suptitle("Common RL Training Failure Signatures", color="white", fontsize=11, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_debug); plt.close()

    # ── REWARD DESIGN ─────────────────────────────────────────────────────
    with tab_rew:
        _sec("🎯","Reward Design — The Most Underrated RL Skill",
             "Potential-based shaping · RLHF · Preference learning — a bad reward is the #1 failure mode","#ffa726")

        st.markdown(_card("#ffa726","🎯","Why reward design is the hardest part of applied RL",
            """In academic RL benchmarks, reward functions are carefully crafted by domain experts.
            In the real world, specifying what you actually want an agent to optimise is extraordinarily
            difficult. Goodhart's Law states: 'When a measure becomes a target, it ceases to be a good
            measure.' In RL, this means: when an agent optimises your reward function, it will find
            every way to get high reward that you did not intend. Famous examples: boat racing agent
            learned to spin in circles collecting point pickups, never finishing the race; robot hand
            learned to flip over to avoid falling rather than walk. Every real-world RL deployment
            starts with reward design, and most early failures are reward design failures, not algorithm
            failures. The solutions: (1) potential-based reward shaping for dense feedback without
            changing the optimal policy; (2) reward modelling from human demonstrations; (3) RLHF
            (Reinforcement Learning from Human Feedback) for complex preferences that are hard to
            specify analytically. This module covers all three approaches with the necessary theory."""), unsafe_allow_html=True)

        st.subheader("1. Reward Shaping — Add Dense Signals Without Changing Optimal Policy")
        st.markdown(r"""
        Sparse rewards (e.g. +1 only at goal) make learning slow. Adding intermediate rewards
        (e.g. +0.1 for getting closer) can speed up learning — but the wrong shaping changes
        which policy is optimal (the agent may prefer the dense shaped reward over the true goal).
        **Potential-based shaping** is the only safe way to add intermediate rewards:
        """)
        st.latex(r"r'(s,a,s') = r(s,a,s') + \gamma\Phi(s') - \Phi(s) \quad \text{(shaping bonus)}")
        st.markdown(r"""
        **Theorem (Ng et al. 1999):** The optimal policy for $r'$ is the same as for $r$ for any
        potential function $\Phi: \mathcal{S} \to \mathbb{R}$. Common choices:
        - $\Phi(s) = -\|s - s_{\text{goal}}\|$ (distance to goal — reward for getting closer)
        - $\Phi(s) = V^*(s)$ (optimal value — equivalent to DIAYN)
        - $\Phi(s) = Q^*(s, \pi(s))$ (Q-value under current policy)
        """)
        st.markdown(_insight("""
        The potential-based shaping trick lets you add as many intermediate rewards as you want
        without any risk of changing which policy is optimal at the end. This is mathematically
        proven — no need to worry about the shaped reward dominating the true objective.
        In practice, distance-to-goal shaping reduces the training time by 5–20× for navigation tasks.
        """), unsafe_allow_html=True)

        st.divider()
        st.subheader("2. RLHF — Reinforcement Learning from Human Feedback")
        st.markdown(r"""
        When the reward is too complex to specify analytically (e.g. 'write a helpful, harmless,
        honest response'), train a **reward model** from human preferences:
        """)
        st.latex(r"P(\text{response A} \succ \text{response B}) = \sigma(r_\phi(A) - r_\phi(B))")
        st.markdown(r"""
        **The RLHF pipeline:**
        1. **Collect preferences**: show pairs of responses to humans, get labels (A better or B better)
        2. **Train reward model**: $r_\phi$ trained to predict human preferences via Bradley-Terry model
        3. **Fine-tune with PPO**: train language policy $\pi_\theta$ to maximise $r_\phi$ while staying close to the reference model:
        """)
        st.latex(r"J(\theta) = \mathbb{E}_{(x,y)\sim\pi_\theta}\!\left[r_\phi(x,y) - \beta\,D_{\text{KL}}(\pi_\theta(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))\right]")
        st.markdown(r"""
        - $\beta$ controls the KL penalty — prevents the policy from deviating too far from the reference (reward hacking the reward model)
        - This pipeline powers ChatGPT, Claude, Gemini, and every other modern RLHF-trained LLM
        """)

        st.divider()
        st.subheader("3. Common Reward Design Mistakes")
        mistakes = [
            ("Sparse reward without HER/shaping", "Agent never sees positive reward → zero gradient → no learning"),
            ("Scale too extreme (±1000)", "Dominates all other learning signals; hard to tune learning rates"),
            ("Reward clipping too aggressive", "Agent cannot distinguish good from great; plateaus early"),
            ("Terminal reward only", "All intermediate transitions receive zero gradient; bad for value learning"),
            ("Negative-only reward", "Agent learns to end episodes quickly to minimise total negative reward"),
            ("Reward based on position (not velocity)", "Agent can exploit by oscillating near goal boundary"),
        ]
        for mistake, consequence in mistakes:
            st.markdown(f'<div style="background:#2a1a0a;border-left:3px solid #ffa726;'
                        f'padding:.5rem .9rem;border-radius:0 8px 8px 0;margin:.3rem 0">'
                        f'<b style="color:#ffa726;font-size:.87rem">❌ {mistake}</b><br>'
                        f'<span style="color:#b0b0cc;font-size:.83rem">{consequence}</span></div>',
                        unsafe_allow_html=True)

    # ── DISTRIBUTED RL ────────────────────────────────────────────────────
    with tab_dist:
        _sec("⚡","Distributed Reinforcement Learning",
             "IMPALA · Ape-X · R2D2 · EnvPool — scale to 1000+ parallel workers","#0288d1")

        st.markdown(_card("#0288d1","⚡","Why distributed RL is necessary at scale",
            """Single-machine RL is limited by how fast you can interact with environments and how
            quickly the GPU can train. For Atari, a single worker gets ~1000 frames/second.
            Achieving human-level performance on Atari requires ~50 million frames — that is 14 hours
            of wall-clock time just for data collection. For more complex environments like StarCraft II
            or robotics simulations, data collection is even slower. Distributed RL solves this by
            separating data collection from model training: hundreds of 'actor' processes run
            environment simulations in parallel on CPUs, sending experience to a central 'learner'
            GPU that updates the model continuously. The actors periodically download the latest model
            weights. This achieves data collection rates of 100,000+ frames/second while keeping
            the GPU fully utilised. The key engineering challenge: actors run the old policy while
            the learner has updated the model, creating an off-policy gap. IMPALA handles this with
            V-trace corrections; Ape-X uses PER replay buffer; R2D2 adds recurrence for memory.
            EnvPool (non-distributed but parallelised) uses C++ environments with Python interface
            to get 50,000+ steps/second on a single machine without any distributed infrastructure."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**IMPALA — V-trace off-policy correction:**")
            st.latex(r"\delta_t v_s = \rho_t(r_t + \gamma v_{s+1} - V(x_t))")
            st.latex(r"\rho_t = \min\!\left(\bar\rho,\,\frac{\pi(a_t|x_t)}{\mu(a_t|x_t)}\right), \quad c_t=\min\!\left(\bar c,\,\frac{\pi}{\mu}\right)")
            st.markdown(r"""
            $\mu$ is the actor's (old) policy; $\pi$ is the learner's (current) policy.
            V-trace truncates importance sampling ratios to $\bar\rho$ and $\bar c$ to prevent
            variance explosion. The truncation introduces bias but keeps gradients stable.
            IMPALA achieves 250,000 frames/second on 500 CPUs.
            """)
        with col2:
            st.markdown("**Ape-X — Prioritised distributed replay:**")
            st.markdown(r"""
            Multiple actors each with own local environment → experiences sent to central
            **prioritised replay buffer** → single learner samples from buffer.
            Actors compute initial TD priorities; learner refines them.
            Unlike IMPALA, Ape-X is purely off-policy (no V-trace needed).
            DQN-based but achieves 50× the throughput of a single DQN.
            """)
            st.markdown("**R2D2 — Recurrent Ape-X:**")
            st.markdown(r"""
            Adds LSTM to Ape-X to handle partially observable environments.
            Stores hidden states in replay buffer to restore recurrent context.
            State-of-the-art on procedurally generated environments where memory is needed.
            """)

        st.markdown("**EnvPool — vectorised environments without distributed infra:**")
        st.code("""
# EnvPool: run 1024 envs in parallel with C++ backend
import envpool

envs = envpool.make("Atari-v5", env_type="gym",
                    num_envs=1024, episodic_life=True)
# Steps all 1024 envs simultaneously — ~50,000 steps/second on 1 machine
obs, rew, done, info = envs.step(actions)  # actions.shape = (1024,)
""", language="python")

        st.markdown(_insight("""
        For most research and small-scale applications, EnvPool on a single machine is sufficient —
        50,000 steps/second means you can collect 50M Atari frames in ~17 minutes.
        Only use true distributed RL (IMPALA/Ape-X) when you need millions of steps per hour
        for very long training runs or when you have access to a compute cluster.
        """), unsafe_allow_html=True)

    # ── EXPERIMENT TRACKING ───────────────────────────────────────────────
    with tab_track:
        _sec("📊","Experiment Tracking & Reproducibility",
             "W&B · MLflow · Seeds · Evaluation protocol · Optuna — most RL results are not reproducible without this","#00897b")

        st.markdown(_card("#00897b","📊","Why reproducibility is a crisis in RL research",
            """A 2018 paper by Henderson et al. ('Deep Reinforcement Learning That Matters') showed
            that many published RL results cannot be reproduced even with the same code and same
            hyperparameters, just different random seeds. PPO on HalfCheetah varied from 2000 to 6000
            total reward across 5 seeds — a 3× difference. This is not noise; it is structural
            variance in RL training. An algorithm that appears to outperform a baseline by 20% might
            just be luckier with its seed. The RL community has gradually adopted better practices:
            always report mean ± std over at least 5 seeds; always report the number of environment
            steps not wall-clock time; always use a separate evaluation policy (no ε-greedy) on 100+
            episodes; always log the full hyperparameter configuration; always save model checkpoints.
            Experiment tracking tools like Weights & Biases (W&B) and MLflow make this easier:
            they automatically log training curves, hyperparameters, system metrics, and code versions,
            enabling you to compare runs, share results, and reproduce any experiment from its logged
            configuration. Hyperparameter search tools like Optuna or Ray Tune automate the tedious
            process of finding the right learning rate, entropy coefficient, and GAE lambda —
            which can change optimal performance by 5–10× on a new environment."""), unsafe_allow_html=True)

        st.subheader("Essential Reproducibility Checklist")
        checks = [
            ("🌱","Set and log random seeds","Set numpy, torch, random, and env seeds. Log the seed used.","env.seed(42); torch.manual_seed(42); np.random.seed(42)"),
            ("📊","Evaluate with a separate evaluation policy","At eval time, use greedy policy (no ε-greedy, no stochastic sampling). Run 100 episodes.","eval_reward = mean([eval_episode() for _ in range(100)])"),
            ("📁","Log hyperparameters to W&B","Every run logs its full config automatically for comparison.","wandb.init(config=args); wandb.log({'reward': ep_reward})"),
            ("💾","Save model checkpoints","Save every N steps. Include the random state so training can resume exactly.","torch.save({'model': model.state_dict(), 'rng': torch.get_rng_state()}, 'ckpt.pt')"),
            ("🔢","Run ≥5 seeds","Report mean ± std (or IQM for robustness). Never report a single seed result.","results = [train(seed=s) for s in range(5)]"),
            ("📏","Report steps not episodes","Episodes have variable length; steps are comparable across algorithms.","wandb.log({'reward': r, 'step': total_steps})"),
        ]
        for icon, title, desc, code in checks:
            with st.expander(f"{icon} {title}"):
                st.markdown(f'<span style="color:#b0b0cc;font-size:.9rem">{desc}</span>', unsafe_allow_html=True)
                st.code(code, language="python")

        st.divider()
        st.subheader("Hyperparameter Search with Optuna")
        st.code("""
import optuna
from stable_baselines3 import PPO

def objective(trial):
    # Optuna samples hyperparameters from specified distributions
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])

    model = PPO("MlpPolicy", "CartPole-v1",
                learning_rate=lr,
                ent_coef=ent_coef,
                gae_lambda=gae_lambda,
                n_steps=n_steps)
    model.learn(50_000)

    # Evaluate on 20 episodes
    rewards = [evaluate(model) for _ in range(20)]
    return np.mean(rewards)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Run 50 hyperparameter configurations
print(study.best_params)
""", language="python")

        st.markdown(_insight("""
        Use Optuna with W&B together: each Optuna trial automatically logs to W&B, giving you
        a full hyperparameter importance analysis and parallel coordinate plot showing which
        parameters matter most. This combination can find the optimal PPO hyperparameters for
        a new environment in 50 trials (~2 hours on a GPU), saving days of manual tuning.
        """), unsafe_allow_html=True)
