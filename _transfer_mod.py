"""
_transfer_mod.py — Transfer, Multi-Task, Continual RL & Modern Training Methods
Covers: Continual RL · Multi-Task RL · Population-Based Training · GRPO · RLVR
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


def _sec(emoji, title, sub, color="#f57f17"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)


def main_transfer():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1a0a,#0a1a2e);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🔄 Transfer, Multi-Task & Modern Training Methods</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem">'
        'Continual RL, Multi-Task RL, Population-Based Training, GRPO, and RLVR — '
        'the methods that take RL from single-task research to real-world deployment and 2025 LLM training.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🔁 Continual RL",
        "🎯 Multi-Task RL",
        "🌍 Population-Based Training",
        "🧠 GRPO (2025)",
        "✅ RLVR & DeepSeek-R1",
        "📊 Overview",
    ])
    (tab_cont, tab_mt, tab_pbt, tab_grpo, tab_rlvr, tab_ov) = tabs

    # ── CONTINUAL RL ─────────────────────────────────────────────────────
    with tab_cont:
        _sec("🔁", "Continual Reinforcement Learning",
             "Learning new tasks without forgetting old ones — catastrophic interference and its solutions", "#f57f17")

        st.markdown(_card("#f57f17", "🔁", "Catastrophic interference — the core problem",
            """Standard neural networks suffer from catastrophic forgetting: when trained sequentially
            on task B after task A, the network overwrites the weights that encode task A's knowledge
            to fit task B. This happens because SGD optimises for the current task only — it has no
            memory of previous tasks. For deployed robots, autonomous agents, or any system that
            encounters new tasks over its lifetime, this is a critical failure. An autonomous vehicle
            that learns to navigate a new city should not forget how to navigate its home city.
            Continual RL addresses this with three main approaches: (1) Regularisation-based —
            add penalties that prevent important weights (for old tasks) from changing too much;
            (2) Architecture-based — allocate dedicated parameters for each task; (3) Replay-based
            — store examples from old tasks and replay them during new task training. Each approach
            makes different assumptions about what information is available about past tasks.
            Continual RL is distinct from multi-task RL (which trains on all tasks simultaneously)
            and meta-RL (which trains for fast adaptation). Continual RL trains sequentially —
            you see task 1, then task 2, then task 3, and must retain all previously learned knowledge."""),
            unsafe_allow_html=True)

        st.subheader("1. Elastic Weight Consolidation (EWC) — Regularisation-Based")
        st.markdown(r"""
        **EWC (Kirkpatrick et al. 2017)** adds a regularisation term that penalises changes to
        weights that were important for previous tasks. Importance is measured by the Fisher
        information matrix — the curvature of the loss surface for previous tasks.

        **Deriving the EWC penalty from a Bayesian perspective:**
        After training on task A, the posterior over parameters is:
        """)
        st.latex(r"\log p(\theta|\mathcal{D}_A) = \log p(\mathcal{D}_A|\theta) + \log p(\theta) - \log p(\mathcal{D}_A)")
        st.markdown(r"""
        When training on task B, we want to maximise $\log p(\theta|\mathcal{D}_A, \mathcal{D}_B)$.
        Using the Laplace approximation around $\theta_A^*$ (optimal for task A):
        """)
        st.latex(r"\log p(\theta|\mathcal{D}_A) \approx \log p(\theta_A^*|\mathcal{D}_A) - \frac{1}{2}\sum_i F_i(\theta_i - \theta_{A,i}^*)^2")
        st.markdown(r"""
        This gives the **EWC loss**:
        """)
        st.latex(r"\mathcal{L}_B(\theta) + \underbrace{\frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_{A,i}^*)^2}_{\text{EWC penalty}}")
        st.markdown(r"""
        **Symbol decoder:**
        - $F_i$ — Fisher information for parameter $\theta_i$: how sensitive task A's loss is to $\theta_i$
        - $\theta_{A,i}^*$ — optimal parameter value for task A (saved after task A training)
        - $\lambda$ — EWC regularisation strength (typically 1–10,000)
        - High $F_i$: parameter is important for task A → penalise changing it → protect task A knowledge
        - Low $F_i$: parameter is unimportant for task A → free to change for task B
        """)

        st.markdown("**Computing the Fisher information matrix:**")
        st.latex(r"F_i = \mathbb{E}_{(s,a)\sim\rho_{\pi_A}}\!\left[\left(\frac{\partial\log\pi_A(a|s)}{\partial\theta_i}\right)^2\right]")
        st.markdown(r"Approximated by averaging squared gradients over samples from the old task's data.")

        st.code("""
# EWC Implementation
class EWC:
    def __init__(self, model, old_task_data, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc
        # Save optimal parameters for old task
        self.theta_star = {n: p.clone() for n,p in model.named_parameters()}
        # Compute Fisher information (diagonal approximation)
        self.fisher = self._compute_fisher(old_task_data)

    def _compute_fisher(self, data):
        fisher = {n: torch.zeros_like(p) for n,p in self.model.named_parameters()}
        for s, a in data:
            log_prob = self.model.log_prob(s, a)
            self.model.zero_grad()
            log_prob.backward()
            for n, p in self.model.named_parameters():
                fisher[n] += p.grad.data ** 2 / len(data)  # squared gradients
        return fisher

    def ewc_loss(self):
        loss = 0
        for n, p in self.model.named_parameters():
            # Penalise parameters that drifted from old task optimum, weighted by importance
            loss += (self.fisher[n] * (p - self.theta_star[n]) ** 2).sum()
        return self.lambda_ewc / 2 * loss

# Training on new task B
for batch in task_B_data:
    task_loss = compute_rl_loss(model, batch)      # current task loss
    ewc_penalty = ewc.ewc_loss()                    # protection for old task
    total_loss = task_loss + ewc_penalty            # combined
    total_loss.backward(); optimizer.step()
""", language="python")

        st.subheader("2. Progressive Networks — Architecture-Based")
        st.markdown(r"""
        **Progressive Networks (Rusu et al. 2016)** take a different approach: never modify old
        task columns. Each new task gets a fresh neural network column, and lateral connections
        allow new columns to access features from all previous columns.
        """)
        st.latex(r"h_k^{(i)} = f\!\left(W_k^{(i)}h_{k-1}^{(i)} + \sum_{j<i}U_{k}^{(i:j)}h_{k-1}^{(j)}\right)")
        st.markdown(r"""
        - $h_k^{(i)}$ — hidden layer $k$ of task $i$ column
        - $W_k^{(i)}$ — weights of column $i$ at layer $k$ (trained for task $i$)
        - $U_k^{(i:j)}$ — lateral connections from column $j$ to column $i$ (adapter weights)
        - Task A column frozen, task B column learns freely, task C column accesses both A and B features

        **Advantage:** Zero forgetting — old columns never modified.
        **Disadvantage:** Network size grows linearly with number of tasks.
        """)

        st.subheader("3. Experience Replay — Memory-Based")
        st.markdown(r"""
        The simplest continual learning method: store a small buffer of examples from old tasks
        and replay them during new task training. Also called Episodic Memory.
        """)
        st.latex(r"\mathcal{L} = \mathcal{L}_B(\theta) + \sum_{k<B}\lambda_k\mathcal{L}_k(\theta;\mathcal{M}_k)")
        st.markdown(r"""
        - $\mathcal{M}_k$ — replay buffer for task $k$ (typically 1000–10,000 transitions)
        - $\mathcal{L}_k$ — loss on task $k$ using the replay buffer
        - **DER (Dark Experience Replay, 2020):** replay both stored transitions AND the model's predictions
          at those states from when they were stored (matching "dark knowledge" prevents drift)
        """)

        # Visualise catastrophic forgetting
        np.random.seed(42)
        tasks = ["Task A (training)", "Task B (training)", "Task C (training)"]
        perf_forgetting = np.array([[0.9, 0.3, 0.2], [0.9, 0.85, 0.3], [0.9, 0.85, 0.88]])
        perf_ewc = np.array([[0.9, 0.75, 0.65], [0.9, 0.88, 0.70], [0.9, 0.88, 0.85]])

        fig_cl, axes_cl = _fig(1, 2, 13, 4)
        for ax, perf, title in [(axes_cl[0], perf_forgetting, "Without EWC (catastrophic forgetting)"),
                                  (axes_cl[1], perf_ewc, "With EWC (knowledge preserved)")]:
            im = ax.imshow(perf, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(["Task A\nperf", "Task B\nperf", "Task C\nperf"], color="white", fontsize=8)
            ax.set_yticklabels(tasks, color="white", fontsize=8)
            ax.set_title(title, color="white", fontweight="bold", fontsize=9)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{perf[i,j]:.2f}", ha="center", va="center",
                            color="black", fontsize=9, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_cl); plt.close()
        st.caption("Each row = after training that many tasks. Columns = performance on each task. EWC prevents forgetting old tasks.")

    # ── MULTI-TASK RL ────────────────────────────────────────────────────
    with tab_mt:
        _sec("🎯", "Multi-Task Reinforcement Learning",
             "One policy for many tasks simultaneously — gradient conflict, task balancing, shared representations", "#0288d1")

        st.markdown(_card("#0288d1", "🎯", "Multi-task RL vs meta-RL vs continual RL",
            """Multi-task RL trains a single policy to perform well on multiple tasks simultaneously —
            all tasks are presented during training in a shared replay buffer or curriculum.
            This is different from: meta-RL (trains for fast adaptation to new tasks with gradient
            updates at test time) and continual RL (trains sequentially on tasks, one at a time).
            Multi-task RL assumes all tasks are known during training. The practical motivation:
            a single robot arm policy should handle pick-and-place, pouring, stacking, and wiping
            without switching between separate policies. The key challenges: (1) gradient conflict —
            improvements on task A may harm task B if their optimal policies require different
            representations; (2) task balancing — tasks with very different reward scales dominate
            gradients; (3) negative transfer — learning task B actively degrades task A performance
            beyond what training on A alone would achieve. Solutions: PCGrad (project conflicting
            gradients onto orthogonal subspace), GradNorm (balance gradient magnitudes adaptively),
            and careful architecture design (shared trunk + task-specific heads)."""), unsafe_allow_html=True)

        st.subheader("1. The Gradient Conflict Problem")
        st.markdown(r"""
        When training on tasks A and B simultaneously, the combined gradient is:
        """)
        st.latex(r"\mathbf{g} = \mathbf{g}_A + \mathbf{g}_B \quad\text{where } \mathbf{g}_A = \nabla_\theta\mathcal{L}_A,\; \mathbf{g}_B = \nabla_\theta\mathcal{L}_B")
        st.markdown(r"""
        **Conflict condition:** $\cos(\mathbf{g}_A, \mathbf{g}_B) < 0$ — the gradients point in opposing directions.
        Updating with the combined gradient harms at least one task.

        **PCGrad (Yu et al. 2020) — projecting conflicting gradients:**
        If tasks A and B conflict ($\mathbf{g}_A \cdot \mathbf{g}_B < 0$):
        """)
        st.latex(r"\mathbf{g}_A' = \mathbf{g}_A - \frac{\mathbf{g}_A\cdot\mathbf{g}_B}{\|\mathbf{g}_B\|^2}\mathbf{g}_B \quad\text{(remove component of }g_A\text{ that conflicts with }g_B\text{)}")
        st.markdown(r"The projected gradient $\mathbf{g}_A'$ is orthogonal to $\mathbf{g}_B$ — it improves task A without hurting task B.")

        st.code("""
# PCGrad: Project Conflicting Gradients
def pcgrad_update(losses, model, optimizer):
    \"\"\"losses: list of per-task losses\"\"\"
    grads = []
    for loss in losses:
        model.zero_grad()
        loss.backward(retain_graph=True)
        grads.append({n: p.grad.clone() for n,p in model.named_parameters()})

    # Project each task's gradient away from conflicting tasks
    projected = [g.copy() for g in grads]
    for i in range(len(grads)):
        for j in range(len(grads)):
            if i == j: continue
            # Check for conflict
            dot = sum((grads[i][n] * grads[j][n]).sum() for n in grads[i])
            if dot < 0:  # conflict!
                # Project g_i: remove component that conflicts with g_j
                norm_j_sq = sum((grads[j][n]**2).sum() for n in grads[j])
                for n in projected[i]:
                    projected[i][n] -= (dot / norm_j_sq) * grads[j][n]

    # Apply sum of projected gradients
    model.zero_grad()
    for n, p in model.named_parameters():
        p.grad = sum(g[n] for g in projected)
    optimizer.step()
""", language="python")

        st.subheader("2. GradNorm — Adaptive Task Balancing")
        st.markdown(r"""
        Different tasks have different loss scales and learning rates. Without balancing,
        tasks with large losses dominate gradients. GradNorm (Chen et al. 2018) adaptively
        reweights task losses to equalise gradient magnitudes:
        """)
        st.latex(r"\mathcal{L} = \sum_i w_i\mathcal{L}_i \quad\text{with adaptive weights }w_i")
        st.latex(r"\mathcal{L}_\text{GradNorm} = \sum_i\!\left\|\|\nabla_W(w_i\mathcal{L}_i)\| - \bar G\cdot\tilde\mathcal{L}_i^\alpha\right\|_1")
        st.markdown(r"""
        where $\bar G$ = mean gradient norm across tasks, $\tilde\mathcal{L}_i = \mathcal{L}_i/\mathcal{L}_i^{(0)}$ = relative
        loss (how much loss has changed since training start), and $\alpha > 0$ controls the
        pace of task rebalancing (typical: 1.5). Tasks that are falling behind (high relative loss)
        get higher weight.
        """)

        st.subheader("3. Multi-Task Architecture Design")
        st.markdown(r"""
        The standard architecture for multi-task RL:
        - **Shared trunk** $\phi(s)$: encodes state into task-agnostic representation
        - **Task-specific heads** $\pi_i(a|\phi(s))$ or $Q_i(\phi(s),a)$: one per task

        Task identity can be provided as: one-hot vector concatenated to state, task embedding,
        or learned context encoder (for zero-shot generalisation to new tasks).
        """)
        st.latex(r"\phi = f_\text{shared}(s) \quad\text{(shared encoder, trained for all tasks)}")
        st.latex(r"\pi_i(a|s) = g_i(\phi(s)) \quad\text{(task-specific head for task }i\text{)}")
        st.markdown(_insight("""
        <b>MT-SAC on Meta-World benchmark:</b> Training one SAC policy with task embeddings on
        50 manipulation tasks achieves 80%+ success across all tasks. Individual per-task SAC
        policies achieve 95%+ — so multi-task incurs a ~15% performance cost but eliminates
        the need for 50 separate policies. The trade-off is worth it for deployment simplicity.
        """), unsafe_allow_html=True)

    # ── PBT ──────────────────────────────────────────────────────────────
    with tab_pbt:
        _sec("🌍", "Population-Based Training (PBT)",
             "Jaderberg et al. 2017 — adaptive hyperparameters during training via evolutionary selection", "#00897b")

        st.markdown(_card("#00897b", "🌍", "Why PBT outperforms grid search and Bayesian optimisation",
            """Grid search and Bayesian optimisation (Optuna) find static hyperparameters before training.
            But optimal hyperparameters are not static — the ideal learning rate for early training
            (when gradients are large) is different from the ideal learning rate for fine-tuning
            (when you need small precise steps). Population-Based Training (PBT, Jaderberg et al. 2017)
            solves this by maintaining a population of agents with different hyperparameters and
            adapting them during training itself. Every N steps, the bottom 20% of performers
            copy the weights and hyperparameters of the top 20%, then randomly perturb the
            hyperparameters (mutation). The top performers' hyperparameters are exploited; the
            perturbation allows exploration of nearby configurations. PBT effectively performs
            a continuous evolutionary search over the hyperparameter schedule. Empirically, PBT
            consistently outperforms manual tuning by 15–40% on complex RL tasks. It was used
            to train Capture the Flag (DeepMind 2019), Starcraft II agents, and is part of the
            standard training pipeline at DeepMind. The key advantage over static search: PBT
            discovers hyperparameter schedules (learning rate annealing, entropy decay) that no
            static search can find because they are time-varying."""), unsafe_allow_html=True)

        st.subheader("The PBT Algorithm")
        st.code(r"""
# Population-Based Training
def pbt(env, population_size=20, n_steps=1_000_000, exploit_interval=50_000):
    # Initialise population with random hyperparameters
    population = []
    for _ in range(population_size):
        agent = PPOAgent(env)
        agent.hyperparams = {
            'lr':          10**np.random.uniform(-4, -2),    # log-uniform
            'entropy_coef': np.random.uniform(0.001, 0.1),
            'clip_eps':     np.random.uniform(0.1, 0.4),
            'gamma':        np.random.uniform(0.95, 0.999),
        }
        population.append(agent)

    for step in range(0, n_steps, exploit_interval):
        # Step 1: TRAIN each agent in parallel for exploit_interval steps
        for agent in population:
            agent.train(exploit_interval)

        # Step 2: EXPLOIT — bottom 25% copy from top 25%
        scores = [(agent.eval_score(), agent) for agent in population]
        scores.sort(reverse=True)
        top_agents    = [a for _, a in scores[:population_size//4]]
        bottom_agents = [a for _, a in scores[-population_size//4:]]

        for bad_agent in bottom_agents:
            good_agent = random.choice(top_agents)
            bad_agent.load_weights(good_agent)        # copy weights
            bad_agent.hyperparams = good_agent.hyperparams.copy()  # copy hyperparams

            # Step 3: EXPLORE — perturb hyperparameters
            for key in bad_agent.hyperparams:
                if np.random.rand() < 0.5:
                    bad_agent.hyperparams[key] *= np.random.choice([0.8, 1.2])

    best_agent = max(population, key=lambda a: a.eval_score())
    return best_agent
""", language="python")

        # PBT vs grid search visualisation
        np.random.seed(42)
        t = np.arange(200)
        grid_best = np.minimum(0.85, 0.1 + t/200 * 0.75 + np.random.randn(200)*0.02)
        pbt_curve  = np.minimum(0.95, 0.1 + t/200 * 0.85 + np.random.randn(200)*0.015 + t/200**2 * 10)

        fig_pbt, ax_pbt = _fig(1, 1, 10, 4)
        ax_pbt.plot(t, grid_best, color="#546e7a", lw=2.5, label="Best grid search (fixed hyperparams)")
        ax_pbt.plot(t, pbt_curve, color="#00897b", lw=2.5, label="PBT (adaptive hyperparams)")
        ax_pbt.fill_between(t, grid_best, pbt_curve, alpha=0.15, color="#00897b",
                            label="PBT improvement")
        for exploit_t in range(0, 200, 25):
            ax_pbt.axvline(exploit_t, color="#ffa726", lw=0.8, alpha=0.3)
        ax_pbt.text(12, 0.92, "Exploit\nintervals", color="#ffa726", fontsize=7)
        ax_pbt.set_xlabel("Training steps (×1000)", color="white")
        ax_pbt.set_ylabel("Eval performance", color="white")
        ax_pbt.set_title("PBT vs Grid Search: Adaptive hyperparameters outperform static ones",
                         color="white", fontweight="bold")
        ax_pbt.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_pbt.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_pbt); plt.close()

    # ── GRPO ─────────────────────────────────────────────────────────────
    with tab_grpo:
        _sec("🧠", "GRPO — Group Relative Policy Optimization (2025)",
             "DeepSeek's PPO replacement — no critic needed, dominant LLM RL algorithm of 2025", "#e65100")

        st.markdown(_card("#e65100", "🧠", "What GRPO is and why it replaced PPO for LLM training",
            """Group Relative Policy Optimization (GRPO) was introduced in DeepSeek's technical
            report (2025) and has rapidly become the dominant algorithm for training LLMs with RL.
            It replaces PPO's value network (critic) with a much simpler group-relative baseline:
            for each prompt x, sample G responses from the current policy, compute the reward for
            each response, and use the within-group mean reward as the baseline. The advantage for
            each response is simply how much better it is than the group average — no separate
            value network required. This eliminates a major source of instability in PPO-based RLHF
            (value network training), reduces GPU memory by 40–50% (no separate critic network),
            and removes the critic learning rate as a hyperparameter. GRPO is used in DeepSeek-R1,
            Qwen-2.5, and dozens of open-source RLHF implementations released in 2025. The algorithm
            is strikingly simple: its main loop is 20 lines of code, yet it achieves competitive or
            superior performance to PPO on mathematical reasoning and coding benchmarks."""), unsafe_allow_html=True)

        st.subheader("GRPO Algorithm — Derivation")
        st.markdown(r"""
        **Standard PPO advantage** requires a value network $V_\phi(s)$ as baseline:
        """)
        st.latex(r"A_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \quad\text{(needs separate critic training)}")
        st.markdown(r"""
        **GRPO group-relative advantage** — no value network:

        For prompt $x$, sample $G$ responses $\{y_1, y_2, \ldots, y_G\}$ from $\pi_\theta$,
        compute reward $r_i$ for each response, then:
        """)
        st.latex(r"A_i = \frac{r_i - \text{mean}(\{r_1,\ldots,r_G\})}{\text{std}(\{r_1,\ldots,r_G\})+\varepsilon}")
        st.markdown(r"""
        This is a **z-score normalisation within the group**: response $i$ is good if it scored
        higher than average among the group, bad if lower. No value network, no bootstrapping.

        **GRPO objective (clipped IS, same as PPO):**
        """)
        st.latex(r"\mathcal{L}_\text{GRPO}(\theta) = \mathbb{E}_x\!\left[\frac{1}{G}\sum_{i=1}^G\min\!\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_\text{old}}(y_i|x)}A_i,\;\text{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_\text{old}}},1\pm\varepsilon\right)\!A_i\right) - \beta D_\text{KL}(\pi_\theta\|\pi_\text{ref})\right]")

        st.code(r"""
# GRPO — Full implementation (the 2025 LLM RL algorithm)
# G: group size, eps: clip range, beta: KL penalty weight
def grpo_step(model, ref_model, reward_fn, prompts, G=8, eps=0.2, beta=0.01):
    all_losses = []
    for x in prompts:
        # 1. Sample G responses from current policy
        responses = [model.generate(x) for _ in range(G)]

        # 2. Compute rewards for each response
        rewards = [reward_fn(x, y) for y in responses]
        rewards = torch.tensor(rewards)

        # 3. Group-relative advantage (z-score normalisation)
        A = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 4. PPO clipped objective over the group
        for y_i, a_i in zip(responses, A):
            log_prob_new = model.log_prob(x, y_i)
            log_prob_old = log_prob_new.detach()
            ratio = (log_prob_new - log_prob_old).exp()
            clipped = ratio.clamp(1-eps, 1+eps)
            policy_loss = -torch.min(ratio * a_i, clipped * a_i).mean()
            kl = (model.log_prob(x,y_i) - ref_model.log_prob(x,y_i)).mean()
            all_losses.append(policy_loss + beta * kl)

    loss = torch.stack(all_losses).mean()
    loss.backward(); optimizer.step()
""", language="python")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**GRPO vs PPO comparison:**")
            st.dataframe(pd.DataFrame({
                "Property": ["Baseline", "Value network", "Memory", "Hyperparameters", "Stability"],
                "PPO": ["V(s) from critic", "Required (50% more params)", "Full (actor+critic)", "lr_π + lr_V + clip", "Medium"],
                "GRPO": ["Group mean reward", "None needed", "Actor only", "clip + β + G", "High"],
            }), use_container_width=True, hide_index=True)
        with col2:
            st.markdown(_insight("""
            <b>Why G matters:</b> G=8 is the DeepSeek-R1 default.
            Larger G → lower variance advantage estimates → more stable training.
            Smaller G → fewer samples needed per prompt → faster training.
            For coding and math tasks, G=4–8 works well with verifiable rewards.
            For preference-based tasks, G=2 (just prefer A over B) is the DPO case.
            """), unsafe_allow_html=True)

    # ── RLVR ─────────────────────────────────────────────────────────────
    with tab_rlvr:
        _sec("✅", "RLVR & DeepSeek-R1 — RL with Verifiable Rewards",
             "The paradigm that trained DeepSeek-R1 and opened-source RLHF — no reward model needed", "#558b2f")

        st.markdown(_card("#558b2f", "✅", "RLVR: eliminating the reward model entirely",
            """Standard RLHF requires training a reward model r_φ from human preferences —
            an expensive, error-prone step that can be gamed (reward hacking). RLVR (RL with
            Verifiable Rewards) eliminates the reward model entirely for tasks where correctness
            can be verified objectively: mathematical proofs (check if the final answer matches),
            code generation (run the test suite), formal logic (automated theorem prover).
            For these tasks, the reward is binary (correct/incorrect) and perfectly reliable —
            no human labelling needed, no reward model to train. DeepSeek-R1 (January 2025)
            used RLVR with GRPO on math and code tasks, achieving performance comparable to
            OpenAI o1 while being fully open-source. The key insight: for domains with ground-truth
            verifiers, the verifier IS the reward model — and it is perfect, scalable, and free.
            This represents a fundamental shift in how we think about RL for reasoning: instead of
            learning what humans prefer, we learn what is objectively correct by verification."""), unsafe_allow_html=True)

        st.subheader("RLVR Training Pipeline")
        for step, color, title, desc in [
            ("Step 1", "#546e7a", "Start from a strong pretrained LLM (e.g. DeepSeek-V3-Base)",
             "Pretrain on large text corpus. The base model has strong language understanding but no RL-optimised reasoning. No SFT stage needed for RLVR — go directly to RL."),
            ("Step 2", "#00897b", "Define a verifiable reward function",
             "For math: reward = 1 if final answer matches ground truth, 0 otherwise. For code: reward = fraction of test cases passed. For proofs: reward = theorem prover verification. The verifier must be fast (milliseconds) to allow thousands of RL steps."),
            ("Step 3", "#e65100", "Apply GRPO with the verifiable reward",
             "Sample G responses per math problem using GRPO. Compute binary rewards. Apply group-relative normalisation. Update policy with clipped objective. No reward model, no critic."),
            ("Step 4", "#0288d1", "Emergence of chain-of-thought reasoning",
             "Remarkably, without any explicit supervision for reasoning steps, the model spontaneously develops extended chain-of-thought: 'Let me think step by step...' — because longer, more careful reasoning leads to correct answers and higher reward."),
        ]:
            st.markdown(f'<div style="background:{color}18;border-left:4px solid {color};'
                        f'border-radius:0 10px 10px 0;padding:.8rem 1.1rem;margin:.4rem 0">'
                        f'<b style="color:{color}">{step}: {title}</b><br>'
                        f'<span style="color:#b0b0cc;font-size:.9rem">{desc}</span></div>',
                        unsafe_allow_html=True)

        st.subheader("The RLVR Reward Function")
        st.code(r"""
# RLVR reward functions for different verifiable domains

# Math (most common)
def math_reward(problem, response):
    predicted_answer = extract_boxed_answer(response)  # parse LaTeX \boxed{}
    correct_answer = problem['answer']
    return 1.0 if predicted_answer == correct_answer else 0.0

# Code generation
def code_reward(problem, response):
    code = extract_code_block(response)  # parse ```python...``` block
    test_results = run_tests(code, problem['test_cases'])
    return sum(test_results) / len(test_results)  # fraction passing

# Format reward (encourage chain-of-thought structure)
def format_reward(response):
    has_thinking = '<think>' in response and '</think>' in response
    has_answer   = '<answer>' in response and '</answer>' in response
    return 0.5 * (has_thinking + has_answer)  # small bonus for correct format

# Combined reward used in DeepSeek-R1
def combined_reward(problem, response):
    return math_reward(problem, response) + 0.1 * format_reward(response)
""", language="python")

        st.subheader("RLVR Results — Why It Matters")
        st.dataframe(pd.DataFrame({
            "Model": ["GPT-4o", "Claude-3.5-Sonnet", "DeepSeek-R1 (RLVR+GRPO)", "OpenAI o1"],
            "AIME 2024 (math)": ["9.3%", "16.0%", "79.8%", "74.4%"],
            "MATH-500": ["76.6%", "71.1%", "97.3%", "96.4%"],
            "Codeforces percentile": ["23%", "18%", "96%", "89%"],
            "Training method": ["SFT+RLHF", "SFT+RLHF", "RLVR+GRPO", "Unknown (RL)"],
            "Reward model": ["Yes", "Yes", "No (verifier)", "Unknown"],
        }), use_container_width=True, hide_index=True)

        st.markdown(_insight("""
        <b>The key insight from DeepSeek-R1:</b> RL with verifiable rewards on math and code
        produces stronger reasoning than RLHF with human preference labels — because human
        preferences are noisy approximations of correctness, while verifiers give perfect signal.
        This suggests that for any domain with objective correctness, RLVR should be preferred
        over RLHF. The open challenge: extending RLVR to domains without ground-truth verifiers
        (creative writing, strategic advice, scientific hypotheses).
        """), unsafe_allow_html=True)

    # ── OVERVIEW ─────────────────────────────────────────────────────────
    with tab_ov:
        st.subheader("📊 Summary: Transfer & Modern Training Methods")
        st.dataframe(pd.DataFrame({
            "Method": ["Continual RL (EWC)", "Continual RL (Replay)", "Progressive Nets",
                       "Multi-Task RL (PCGrad)", "PBT", "GRPO", "RLVR"],
            "What problem it solves": [
                "Catastrophic forgetting of old tasks",
                "Catastrophic forgetting via memory",
                "Zero forgetting (separate columns)",
                "Gradient conflict between tasks",
                "Static hyperparameters during training",
                "PPO critic is expensive and unstable",
                "Reward models are noisy and hackable",
            ],
            "Key equation / trick": [
                "L += λ/2 Σ F_i(θ_i - θ*_i)²",
                "L = L_new + Σ L_old(replay buffer)",
                "Frozen old columns + lateral connections",
                "Project g_A onto orthogonal of g_B",
                "Bottom 25% copy top 25% + mutate",
                "A_i = (r_i - mean(r)) / std(r)",
                "reward = verifier(response) ∈ {0,1}",
            ],
            "Used by": [
                "Sequential robot learning",
                "Any continual system",
                "DeepMind progressive nets",
                "Meta-World manipulation",
                "DeepMind Capture the Flag",
                "DeepSeek-R1, Qwen-2.5",
                "DeepSeek-R1, OpenAI o1",
            ],
            "Year": ["2017", "2018", "2016", "2020", "2017", "2025", "2024–2025"],
        }), use_container_width=True, hide_index=True)
