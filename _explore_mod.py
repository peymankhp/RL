"""_explore_mod.py — Exploration Methods (Tier 1)
UCB · Thompson Sampling · ICM · RND · Count-based — with charts, examples, books"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from _notes_mod import render_notes
warnings.filterwarnings("ignore")

DARK, CARD, GRID = "#0d0d1a", "#12121f", "#2a2a3e"

def _fig(nr=1, nc=1, w=13, h=5):
    fig, axes = plt.subplots(nr, nc, figsize=(w, h))
    fig.patch.set_facecolor(DARK)
    for ax in np.array(axes).flatten():
        ax.set_facecolor(DARK); ax.tick_params(colors="#9e9ebb", labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
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

def _sec(emoji, title, sub, color="#f57f17"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def smooth(a, w=8):
    return np.convolve(a, np.ones(w)/w, mode="valid") if len(a) > w else np.array(a, float)

# ── Multi-armed bandit simulation ─────────────────────────────────────────────
def run_bandit(n_arms, n_steps, strategy, seed=42):
    np.random.seed(seed)
    true_means = np.random.randn(n_arms) + 1
    best_arm = np.argmax(true_means)
    counts = np.zeros(n_arms); totals = np.zeros(n_arms)
    alpha_ts = np.ones(n_arms); beta_ts = np.ones(n_arms)
    rewards = []; regrets = []
    for t in range(1, n_steps+1):
        if strategy == "greedy":
            a = np.argmax(totals/(counts+1e-8)) if t > n_arms else t-1
        elif strategy == "eps_greedy":
            a = np.random.randint(n_arms) if np.random.rand() < 0.1 else np.argmax(totals/(counts+1e-8))
        elif strategy == "ucb":
            Q = totals/(counts+1e-8) + np.sqrt(2*np.log(t)/(counts+1e-8))
            a = np.argmax(Q)
        elif strategy == "thompson":
            samples = np.random.beta(alpha_ts, beta_ts)
            a = np.argmax(samples)
        r = np.random.randn() + true_means[a]
        counts[a] += 1; totals[a] += r
        if strategy == "thompson":
            r_bin = 1 if r > true_means.mean() else 0
            alpha_ts[a] += r_bin; beta_ts[a] += 1-r_bin
        rewards.append(r)
        regrets.append(true_means[best_arm] - true_means[a])
    return np.array(rewards), np.cumsum(regrets)


def main_explore():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#1a1a0a,#0a1a0a,#0a0a1a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">🔍 Exploration Methods</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'The explore-exploit dilemma solved: UCB, Thompson Sampling, curiosity-driven exploration, '
        'and intrinsic motivation. From multi-armed bandits to Montezuma\'s Revenge — with '
        'interactive simulations, derivations, and deep-dive resources.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "⚖️ Explore-Exploit",
        "📊 UCB",
        "🎲 Thompson Sampling",
        "🔢 Count-Based",
        "🧠 ICM (Curiosity)",
        "🎯 RND",
        "📚 Books & Resources",
    ])
    tab_ee, tab_ucb, tab_ts, tab_cb, tab_icm, tab_rnd, tab_res = tabs

    with tab_ee:
        _sec("⚖️","The Explore-Exploit Dilemma","Maximise cumulative reward vs gather information — a fundamental tension","#f57f17")
        st.markdown(_card("#f57f17","⚖️","Why exploration is the hardest part of RL",
            """The explore-exploit dilemma: should you take the action that currently looks best
            (exploit), or try a different action that might be better but has more uncertainty (explore)?
            Always exploiting: you quickly converge to a suboptimal action — the first action that
            seemed good gets locked in. You never discover that a different action is much better.
            Always exploring: you never stop gathering information and never commit to the best action.
            The optimal strategy is to balance these over time, gradually shifting from exploration
            (early, when uncertainty is high) to exploitation (later, when the best action is identified).
            In RL: the same dilemma applies to policy improvement. During early training, try diverse
            actions to discover which ones lead to reward. Later, exploit the learned policy.
            In sparse-reward environments (Montezuma's Revenge, robotic manipulation), the agent
            receives reward almost never — random exploration almost never discovers the reward.
            This is the hard exploration problem that motivates intrinsic motivation methods."""), unsafe_allow_html=True)

        # The 3 cases
        np.random.seed(42); T = 500; K = 5
        true_means = np.array([1.0, 1.5, 0.8, 2.0, 0.5])
        best = np.max(true_means)
        fig_ee, axes_ee = _fig(1, 3, 16, 4)
        for ax, strategy, color, title in [
            (axes_ee[0], "greedy", "#ef5350", "Greedy (always exploit)\n→ stuck on suboptimal arm"),
            (axes_ee[1], "eps_greedy", "#ffa726", "ε-greedy (ε=0.1)\n→ balanced but always 10% random"),
            (axes_ee[2], "ucb", "#4caf50", "UCB (explore uncertainty)\n→ converges to best arm"),
        ]:
            rw, reg = run_bandit(K, T, strategy)
            ax.plot(smooth(rw, 30), color=color, lw=2.5, label=f"Avg reward (smth)")
            ax.axhline(best, color="white", ls="--", lw=1.2, alpha=0.5, label=f"Best arm={best:.1f}")
            ax.set_xlabel("Step", color="white"); ax.set_ylabel("Reward", color="white")
            ax.set_title(title, color="white", fontweight="bold", fontsize=9)
            ax.legend(facecolor=CARD, labelcolor="white", fontsize=7.5); ax.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ee); plt.close()

        st.markdown("**Regret** = opportunity cost of not always playing the best arm:")
        st.latex(r"\text{Regret}(T) = T\mu^* - \sum_{t=1}^T \mu_{a_t} = \sum_{t=1}^T(\mu^*-\mu_{a_t})")
        st.markdown("**Lower bound (Lai & Robbins 1985):** no algorithm can achieve better than O(log T) regret. UCB and Thompson Sampling are **asymptotically optimal** — they achieve this bound.")

    with tab_ucb:
        _sec("📊","UCB — Upper Confidence Bound","Optimism in the face of uncertainty: pick the action with the highest upper confidence bound","#f57f17")
        st.markdown(_card("#f57f17","📊","UCB: be optimistic about uncertain actions",
            """UCB (Upper Confidence Bound) formalises the intuition: uncertain actions deserve more
            exploration because their TRUE value might be higher than estimated. UCB adds a
            confidence bonus to each arm's estimated value — the bonus is large for rarely-tried arms
            and shrinks as the arm is tried more. The agent always picks the arm with the highest
            UCB value, automatically balancing exploration (high bonus for uncertain arms) and
            exploitation (high estimated value for well-explored arms).
            The log(t) term ensures the exploration bonus decreases over time — as total steps grow,
            we explore less and exploit more. The UCB1 algorithm achieves the optimal O(log T) regret."""), unsafe_allow_html=True)

        st.markdown("**UCB1 action selection:**")
        st.latex(r"a_t = \arg\max_a\underbrace{\hat\mu_a(t)}_\text{exploit: estimated mean} + \underbrace{c\sqrt{\frac{\ln t}{N_a(t)}}}_\text{explore: confidence bonus}")
        st.markdown("Where N_a(t) = number of times arm a was tried, t = total steps, c = exploration constant (typically √2).")
        st.markdown("**Regret bound (UCB1):**")
        st.latex(r"\mathbb{E}[\text{Regret}(T)] \leq \sum_{a:\Delta_a>0}\left(\frac{8\ln T}{\Delta_a} + \left(1+\frac{\pi^2}{3}\right)\Delta_a\right)")
        st.markdown("Where Δ_a = μ* − μ_a is the suboptimality gap of arm a. Arms with larger gaps get less exploration (their suboptimality is discovered faster).")

        c1, c2 = st.columns(2)
        n_arms_u = c1.slider("Number of arms K", 3, 20, 10, 1, key="ucb_k")
        n_steps_u = c1.slider("Steps T", 200, 2000, 1000, 100, key="ucb_t")
        c_ucb = c2.slider("Exploration constant c", 0.1, 3.0, 1.41, 0.1, key="ucb_c")
        seed_u = c2.number_input("Seed", 0, 999, 42, key="ucb_seed")

        if st.button("▶️ Run All Strategies", type="primary", key="btn_ucb"):
            np.random.seed(int(seed_u))
            results = {}
            for strat in ["greedy","eps_greedy","ucb","thompson"]:
                rw, reg = run_bandit(n_arms_u, n_steps_u, strat, int(seed_u))
                results[strat] = (rw, reg)
            st.session_state["ucb_res"] = results

        if "ucb_res" in st.session_state:
            res = st.session_state["ucb_res"]
            fig_ucb, axes_ucb = _fig(1, 2, 14, 4.5)
            colors = {"greedy":"#ef5350","eps_greedy":"#ffa726","ucb":"#4caf50","thompson":"#0288d1"}
            labels = {"greedy":"Greedy","eps_greedy":"ε-greedy (ε=0.1)","ucb":"UCB1","thompson":"Thompson"}
            for strat, (rw, reg) in res.items():
                sm = smooth(rw, 30)
                axes_ucb[0].plot(range(len(sm)), sm, color=colors[strat], lw=2.5, label=labels[strat])
                axes_ucb[1].plot(reg, color=colors[strat], lw=2.5, label=f"{labels[strat]} (final={reg[-1]:.0f})")
            axes_ucb[0].set_xlabel("Step",color="white"); axes_ucb[0].set_ylabel("Average reward",color="white")
            axes_ucb[0].set_title("Reward per step (smoothed)", color="white",fontweight="bold")
            axes_ucb[0].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_ucb[0].grid(alpha=0.12)
            axes_ucb[1].set_xlabel("Step",color="white"); axes_ucb[1].set_ylabel("Cumulative regret",color="white")
            axes_ucb[1].set_title("Cumulative regret: O(log T) optimal for UCB+Thompson", color="white",fontweight="bold")
            axes_ucb[1].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_ucb[1].grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_ucb); plt.close()

            # Arm visit counts
            np.random.seed(42)
            final_counts = {s: np.zeros(n_arms_u) for s in res}
            true_means_here = np.random.randn(n_arms_u) + 1
            best_arm = np.argmax(true_means_here)
            for strat, (rw, reg) in res.items():
                _, arm_visits = run_bandit(n_arms_u, n_steps_u, strat, 42)
                # reconstruct counts approximately
                pass
            c1,c2,c3,c4 = st.columns(4)
            for col_ui, (strat, (rw,reg)) in zip([c1,c2,c3,c4], res.items()):
                col_ui.metric(labels[strat], f"Regret={reg[-1]:.0f}", f"Avg r={np.mean(rw[-100:]):.2f}")

    with tab_ts:
        _sec("🎲","Thompson Sampling — Bayesian Exploration","Sample from posterior beliefs — provably optimal, naturally adapts to uncertainty","#0288d1")
        st.markdown(_card("#0288d1","🎲","Thompson Sampling: Bayesian approach to exploration",
            """Thompson Sampling (TS) maintains a posterior distribution over each arm's mean reward.
            At each step: sample one value from each arm's posterior distribution, and pick the
            arm with the highest sample. This is pure Bayesian decision theory — the action selection
            probability equals the probability that each arm is truly optimal given observed data.
            Thompson Sampling is elegant because: (1) it's optimal — achieves the same O(log T) regret
            as UCB asymptotically; (2) it naturally handles heterogeneous uncertainty — arms with wide
            posteriors (high uncertainty) get explored more; (3) for Bernoulli bandits, the Beta
            distribution is the conjugate prior, giving closed-form Bayesian updates.
            TS is the algorithm used in Netflix's A/B testing, Google's ad systems, and clinical
            trial design — anywhere efficient Bayesian exploration matters."""), unsafe_allow_html=True)

        st.markdown("**Thompson Sampling for Bernoulli bandits:**")
        st.latex(r"\theta_a \sim \text{Beta}(\alpha_a, \beta_a) \quad a_t = \arg\max_a\theta_a")
        st.markdown("**Conjugate Bayesian update:**")
        st.latex(r"\text{If reward=1: }\alpha_a \leftarrow \alpha_a+1 \quad\text{If reward=0: }\beta_a \leftarrow \beta_a+1")

        # Show posterior evolution
        np.random.seed(42)
        true_probs = [0.3, 0.5, 0.7]  # 3 arms
        alpha_arr = [[1,1,1], [3,2,1], [8,5,2], [15,8,4], [25,12,7]]
        beta_arr  = [[1,1,1], [7,4,1], [12,7,2], [15,12,4], [25,18,7]]
        x_beta = np.linspace(0, 1, 200)
        fig_ts, axes_ts = _fig(1, len(alpha_arr), 16, 3.5)
        colors_arm = ["#ef5350","#ffa726","#4caf50"]
        n_obs = [0, 10, 20, 35, 50]
        for ax, al, be, n in zip(axes_ts, alpha_arr, beta_arr, n_obs):
            for i, (a, b, col) in enumerate(zip(al, be, colors_arm)):
                from scipy import stats as sp_stats
                y = sp_stats.beta.pdf(x_beta, a, b)
                ax.plot(x_beta, y, color=col, lw=2, label=f"Arm {i+1} (p={true_probs[i]})")
                ax.axvline(true_probs[i], color=col, lw=1, ls="--", alpha=0.5)
            ax.set_title(f"After {n} obs", color="white", fontsize=8, fontweight="bold")
            ax.set_xlabel("p (success probability)", color="white")
            ax.grid(alpha=0.12); ax.set_ylim(0, None)
        axes_ts[0].legend(facecolor=CARD, labelcolor="white", fontsize=7)
        plt.suptitle("Thompson Sampling: Posterior Beta distributions narrow as evidence accumulates",
                    color="white", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_ts); plt.close()
        try:
            from scipy import stats as sp_stats
        except ImportError:
            st.info("scipy not available — skipping posterior plot details")

        st.markdown(_insight("Thompson Sampling in practice: Netflix estimates click probability for movie thumbnails using TS with Beta(α,β) posteriors. Each user impression updates α or β based on whether the user clicked. TS automatically shows the most promising thumbnail more often while still gathering data on alternatives."), unsafe_allow_html=True)

    with tab_cb:
        _sec("🔢","Count-Based Exploration","Visit counts as exploration bonus — from tabular to neural density models","#7c4dff")
        st.markdown(_card("#7c4dff","🔢","Count-based: explore states you\'ve visited least",
            """In tabular RL (small state space), maintaining a visit count N(s) for each state
            and adding exploration bonus r_bonus = β/√N(s) is the optimal exploration strategy
            (RMAX algorithm). States visited rarely get high bonus; frequently visited states
            get low bonus. This mimics UCB for state-action pairs.
            In deep RL with continuous state spaces, exact counts are impossible — each state
            is visited exactly once. Solution: learn a density model p̂(s) that approximates the
            state visitation frequency, then use -log p̂(s) as the bonus (novel states have low density).
            Pseudo-count (Bellemare 2016): derive a count-like quantity from the density model.
            Practical simplification (RND, Burda 2018): instead of a full density model, train
            a fixed random network and a predictor network to match it — the prediction error
            is high for unseen states, providing a simple novelty signal."""), unsafe_allow_html=True)

        st.markdown("**Count-based bonus (tabular):**")
        st.latex(r"r_\text{total}(s,a) = r(s,a) + \beta/\sqrt{N(s,a)} \quad\text{(RMAX: optimism in face of uncertainty)}")
        st.markdown("**Pseudo-count (continuous states — Bellemare 2016):**")
        st.latex(r"\hat n(s) = \frac{\hat\rho_n(s)}{1-\hat\rho_n(s)} \quad\text{where } \hat\rho_n(s) = \text{density model probability}")
        st.latex(r"r_\text{bonus}(s) = \beta/\sqrt{\hat n(s)+0.01}")

        # Count-based bonus visualisation
        np.random.seed(42)
        x_states = np.linspace(0, 10, 100)
        visit_counts = np.maximum(0, np.random.poisson(3, 100) * np.exp(-0.3*(x_states-5)**2))
        bonus = 0.5 / (np.sqrt(visit_counts + 0.1))
        fig_cb, axes_cb = _fig(1, 2, 13, 4)
        axes_cb[0].bar(x_states, visit_counts, width=0.09, color="#0288d1", alpha=0.8)
        axes_cb[0].set_xlabel("State (1D)", color="white"); axes_cb[0].set_ylabel("Visit count N(s)", color="white")
        axes_cb[0].set_title("Visit counts: most states near centre", color="white", fontweight="bold")
        axes_cb[0].grid(alpha=0.12, axis="y")
        axes_cb[1].plot(x_states, bonus, color="#f57f17", lw=2.5)
        axes_cb[1].bar(x_states, bonus, width=0.09, color="#f57f17", alpha=0.3)
        axes_cb[1].set_xlabel("State (1D)", color="white"); axes_cb[1].set_ylabel("Exploration bonus β/√N(s)", color="white")
        axes_cb[1].set_title("Count-based bonus: high for rarely-visited states", color="white", fontweight="bold")
        axes_cb[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_cb); plt.close()

    with tab_icm:
        _sec("🧠","ICM — Intrinsic Curiosity Module","Pathak et al. 2017 — reward = prediction error of your own actions","#ad1457")
        st.markdown(_card("#ad1457","🧠","ICM: be curious about what your actions do",
            """ICM (Pathak et al. 2017) generates intrinsic reward from the agent\'s inability to
            predict the consequences of its own actions. The key insight: if you cannot predict
            what happens when you take action a in state s, that state-action pair is worth exploring.
            The ICM module has two components: (1) Forward model: given (φ(s_t), a_t), predict φ(s_{t+1}).
            The prediction error is the intrinsic reward — high when the dynamics are novel or complex.
            (2) Inverse model: given (φ(s_t), φ(s_{t+1})), predict which action was taken.
            This ensures the feature representation φ only captures aspects of the state that are
            AFFECTED by the agent\'s actions (action-relevant features). Static parts of the environment
            (background, irrelevant noise) don\'t affect the action and thus don\'t get learned into φ.
            This is the solution to the noisy-TV problem: a pure prediction error bonus would reward
            watching a TV (high entropy = high prediction error). ICM\'s inverse model learns that
            TV pixels aren\'t action-relevant and ignores them."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Feature embedding (action-relevant):**")
            st.latex(r"\phi(s) = \text{Encoder}(s) \quad\text{(removes action-irrelevant features)}")
            st.markdown("**Forward model** (predicts next feature):")
            st.latex(r"\hat\phi_{t+1} = f(\phi(s_t), a_t) \quad\text{(learned dynamics)}")
            st.markdown("**Intrinsic reward** (curiosity):")
            st.latex(r"r_t^\text{int} = \frac{\eta}{2}\|\hat\phi_{t+1}-\phi(s_{t+1})\|^2")
            st.markdown("**Inverse model** (learns what actions do):")
            st.latex(r"\hat a_t = g(\phi(s_t),\phi(s_{t+1})) \quad\text{(predicts taken action)}")
            st.markdown("**Total reward:**")
            st.latex(r"r_t = (1-\beta)r_t^\text{ext} + \beta r_t^\text{int}")
        with col2:
            # Show ICM reward signal over training
            np.random.seed(42); T = 300
            ext_reward = np.concatenate([np.zeros(200), smooth(np.arange(100)*0.08 + np.random.randn(100), 15)])
            int_reward = smooth(2*np.exp(-np.arange(T)/100) + 0.3 + np.random.randn(T)*0.3, 15)
            fig_icm, ax_icm = _fig(1,1,5.5,4)
            ax_icm.plot(smooth(ext_reward,20), color="#ffa726", lw=2, label="Extrinsic reward (sparse)")
            ax_icm.plot(int_reward, color="#ad1457", lw=2, label="Intrinsic reward (ICM curiosity)")
            ax_icm.set_xlabel("Episode",color="white"); ax_icm.set_ylabel("Reward",color="white")
            ax_icm.set_title("ICM: Intrinsic reward drives early\nexploration before reward is found",color="white",fontweight="bold")
            ax_icm.legend(facecolor=CARD,labelcolor="white",fontsize=7); ax_icm.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_icm); plt.close()

        st.markdown("**Empirical results (Pathak et al. 2017):**")
        st.dataframe(pd.DataFrame({
            "Game": ["VizDoom (no reward)","Super Mario Bros (no reward)","Montezuma's Revenge"],
            "A3C (extrinsic only)": ["No progress","No progress","~300 score"],
            "ICM (intrinsic only)": ["Explores full map","Completes levels 1-3","~2500 score"],
            "ICM + extrinsic": ["Solves in half steps","Fastest completion","~3000 score"],
        }), use_container_width=True, hide_index=True)

    with tab_rnd:
        _sec("🎯","RND — Random Network Distillation","Burda et al. 2018 — simplest scalable exploration bonus for deep RL","#f57f17")
        st.markdown(_card("#f57f17","🎯","RND: novelty from prediction error of a random network",
            """RND (Burda et al. 2018) simplifies ICM to its essence: instead of learning forward
            dynamics, use a fixed random network T(s) and a predictor network f_θ(s) trained to
            match it. The prediction error ||f_θ(s) - T(s)||² is the intrinsic bonus.
            Novel states: f_θ hasn\'t been trained on them → high prediction error → high bonus.
            Familiar states: f_θ has been trained many times → low error → low bonus.
            As training progresses and states are revisited, the predictor improves at matching
            the target network, naturally reducing the bonus for known states.
            RND is simpler than ICM (no inverse model, no action-relevant encoding) and scales
            well to large observation spaces. It solved Montezuma\'s Revenge with superhuman
            performance in 2018 — the hardest Atari exploration challenge at the time."""), unsafe_allow_html=True)

        st.markdown("**RND architecture:**")
        st.latex(r"r_t^\text{int} = \|f_\theta(o_t) - T(o_t)\|^2 \quad\text{(T is fixed random, }f_\theta\text{ is trained to match T)}")
        st.markdown("**Predictor training loss:**")
        st.latex(r"\mathcal{L}_\text{RND}(\theta) = \mathbb{E}_{o_t\sim\text{rollout}}\!\left[\|f_\theta(o_t) - T(o_t)\|^2\right]")
        st.markdown("**Why this works:** T(o) is a fixed deterministic function of the observation. For a familiar state, f_θ has seen many examples and converges to T. For a novel state, f_θ has no training signal and produces random output far from T.")

        # Show RND intrinsic reward evolution
        np.random.seed(42); T_steps = 500
        # Novel states: high error early, decreases as visited more
        visits = np.random.exponential(5, T_steps)  # irregular visit times
        rnd_bonus_new = 2.0 * np.exp(-np.arange(T_steps)/100)  # decreasing for repeatedly-seen
        rnd_bonus_novel = 1.8 + 0.3*np.sin(np.arange(T_steps)*0.1) + np.random.randn(T_steps)*0.2  # novel states always high

        fig_rnd, axes_rnd = _fig(1, 2, 13, 4.5)
        axes_rnd[0].plot(smooth(rnd_bonus_new, 30), color="#ffa726", lw=2.5, label="Familiar state (decreasing bonus)")
        axes_rnd[0].plot(smooth(rnd_bonus_novel, 15), color="#f57f17", lw=2, ls="--", label="Novel states (persistently high)")
        axes_rnd[0].set_xlabel("Training step",color="white"); axes_rnd[0].set_ylabel("RND intrinsic bonus",color="white")
        axes_rnd[0].set_title("RND Bonus: decreases for revisited states\n(predictor improves with practice)", color="white",fontweight="bold")
        axes_rnd[0].legend(facecolor=CARD,labelcolor="white",fontsize=8); axes_rnd[0].grid(alpha=0.12)

        # Montezuma performance comparison
        algos = ["DQN","A3C","ICM","RND","Go-Explore"]
        scores = [0, 69, 2500, 17000, 43000]
        cols_bar = ["#546e7a","#0288d1","#ad1457","#f57f17","#4caf50"]
        axes_rnd[1].barh(algos, scores, color=cols_bar, alpha=0.85)
        axes_rnd[1].axvline(17500, color="white", ls="--", lw=1.5, alpha=0.5, label="Human=17500")
        for i, s in enumerate(scores):
            if s > 0:
                axes_rnd[1].text(s+200, i, str(s), va="center", color="white", fontsize=8.5)
        axes_rnd[1].set_xlabel("Montezuma\'s Revenge score", color="white")
        axes_rnd[1].set_title("Exploration Methods on Hard-Exploration Task\n(Montezuma\'s Revenge)", color="white",fontweight="bold")
        axes_rnd[1].grid(alpha=0.12, axis="x"); axes_rnd[1].legend(facecolor=CARD,labelcolor="white",fontsize=8)
        plt.tight_layout(); st.pyplot(fig_rnd); plt.close()

        st.markdown("**Practical tip:** RND should be normalised — maintain running mean and std of the intrinsic reward and normalise to unit variance. This prevents intrinsic reward from dominating extrinsic reward as the predictor improves over training.")

    with tab_res:
        _sec("📚","Books & Deep-Dive Resources","The best exploration RL papers, books, and courses","#546e7a")
        for title, authors, why, url in [
            ("Reinforcement Learning: An Introduction — Ch.2 (Multi-armed Bandits)",
             "Sutton & Barto (2018) — FREE at incompleteideas.net",
             "Chapter 2 derives UCB, ε-greedy, gradient bandits with full proofs. The essential foundation.",
             "http://incompleteideas.net/book/the-book.html"),
            ("Regret Analysis of Stochastic and Non-Stochastic Multi-Armed Bandit Problems",
             "Bubeck & Cesa-Bianchi (2012) — Foundations and Trends in ML",
             "The definitive theoretical treatment of bandit algorithms. UCB, Thompson Sampling, lower bounds.",
             "https://arxiv.org/abs/1204.5721"),
            ("Curiosity-Driven Exploration by Self-Supervised Prediction (ICM)",
             "Pathak et al. (2017) — ICML",
             "The ICM paper. Motivates inverse model for action-relevant features. Includes VizDoom results.",
             "https://arxiv.org/abs/1705.05363"),
            ("Exploration by Random Network Distillation (RND)",
             "Burda et al. (2018) — ICLR 2019",
             "Simplest scalable exploration bonus. Superhuman Montezuma's Revenge. Very readable.",
             "https://arxiv.org/abs/1810.12894"),
            ("Count-Based Exploration with Neural Density Models",
             "Ostrovski et al. (2017) — ICML — Pseudo-counts for deep RL",
             "PixelCNN-based density model for count-based bonuses in Atari. Bridges tabular and deep RL.",
             "https://arxiv.org/abs/1703.01310"),
            ("First Return, then Explore (Go-Explore)",
             "Ecoffet et al. (2021) — Nature",
             "State-of-the-art hard exploration. Archive-based approach. Reaches 400K+ on Montezuma's Revenge.",
             "https://arxiv.org/abs/2004.12919"),
        ]:
            st.markdown(_book(title, authors, why, url), unsafe_allow_html=True)

        st.subheader("🎓 Courses & Implementations")
        for icon, title, desc, url in [
            ("🎥","CS285 Lecture 15: Exploration",
             "Levine covers UCB, Thompson Sampling, count-based, ICM, and RND in one lecture.",
             "https://rail.eecs.berkeley.edu/deeprlcourse/"),
            ("💻","stable-baselines3 + custom reward wrappers",
             "Add intrinsic reward wrappers to any SB3 algorithm. See the documentation examples.",
             "https://stable-baselines3.readthedocs.io"),
        ]:
            st.markdown(f'<div style="background:#12121f;border:1px solid #2a2a3e;border-radius:8px;'
                        f'padding:.6rem 1rem;margin:.3rem 0">'
                        f'<a href="{url}" target="_blank" style="color:#42a5f5;font-weight:700">{icon} {title}</a>'
                        f'<br><span style="color:#9e9ebb;font-size:.86rem">{desc}</span></div>',
                        unsafe_allow_html=True)

    explore_notes = [
        (tab_ee, "Exploration vs Exploitation", "exploration_methods"),
        (tab_ucb, "UCB", "exploration_methods_ucb"),
        (tab_ts, "Thompson Sampling", "exploration_methods_thompson_sampling"),
        (tab_cb, "Count-Based Exploration", "exploration_methods_count_based_exploration"),
        (tab_icm, "ICM", "exploration_methods_icm"),
        (tab_rnd, "RND", "exploration_methods_rnd"),
        (tab_res, "Resources", "exploration_methods_resources"),
    ]
    for tab, note_title, note_slug in explore_notes:
        with tab:
            render_notes(f"Exploration Methods - {note_title}", note_slug)
