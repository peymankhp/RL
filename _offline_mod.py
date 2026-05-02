"""_offline_mod.py — Offline / Batch Reinforcement Learning (Tier 1)
BC · CQL · IQL · Decision Transformer · TD3+BC"""
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

def _sec(emoji, title, sub, color="#00897b"):
    st.markdown(
        f'<div style="background:linear-gradient(90deg,{color}22,transparent);'
        f'border-left:4px solid {color};border-radius:0 10px 10px 0;'
        f'padding:.9rem 1.4rem;margin-bottom:1rem">'
        f'<h3 style="color:white;margin:0;font-size:1.25rem">{emoji} {title}</h3>'
        f'<p style="color:#9e9ebb;margin:.3rem 0 0;font-size:.9rem">{sub}</p></div>',
        unsafe_allow_html=True)

def smooth(a, w=8):
    return np.convolve(a, np.ones(w)/w, mode="valid") if len(a) > w else np.array(a, float)


def main_offline():
    st.markdown(
        '<div style="background:linear-gradient(135deg,#0a2a1a,#0a0a2a);'
        'border:1px solid #2a2a4a;border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem">'
        '<h2 style="color:white;margin:0;font-size:2rem">📦 Offline / Batch Reinforcement Learning</h2>'
        '<p style="color:#9e9ebb;margin-top:.6rem;font-size:1rem;line-height:1.6">'
        'Learn optimal policies from historical data — no environment interaction during training. '
        'BC, CQL, IQL, Decision Transformer, TD3+BC — with the extrapolation error problem explained '
        'from first principles, worked examples, charts, and deep-dive resources.'
        '</p></div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "❓ Why Offline RL?",
        "📋 Behaviour Cloning",
        "🔒 CQL",
        "📐 IQL",
        "🤖 Decision Transformer",
        "🎯 TD3+BC",
        "📊 Benchmark",
        "📚 Books & Resources",
    ])
    tab_why, tab_bc, tab_cql, tab_iql, tab_dt, tab_td3bc, tab_cmp, tab_res = tabs

    with tab_why:
        _sec("❓","Why Offline RL?","Learn from historical data without environment interaction","#00897b")
        st.markdown(_card("#00897b","💊","When offline RL is the only option",
            """Online RL requires interacting with the environment during training. This is fine for
            games and fast simulators. It is impossible for many real-world applications:
            Healthcare: you cannot test experimental treatment policies on patients while training.
            Autonomous driving: you cannot cause 10,000 accidents while learning to drive safely.
            Industrial control: you cannot damage a chemical plant while exploring suboptimal actions.
            Finance: you cannot lose millions of dollars during policy training.
            In all these domains, we have large datasets of historical interactions (EHR records,
            dashcam footage, sensor logs, trading histories) but cannot generate new data cheaply.
            Offline RL asks: can we learn a policy that is BETTER than the historical behaviour
            using only this fixed dataset? The challenge: distributional shift. The learned policy
            may take actions far outside the training distribution — the Q-function extrapolates
            wildly, leading to catastrophically overconfident and wrong estimates."""), unsafe_allow_html=True)

        # Visualise the extrapolation error problem
        np.random.seed(42)
        actions = np.linspace(-3, 3, 200)
        # Dataset actions concentrate around -1 to 1
        dataset_mask = (actions > -1.2) & (actions < 1.2)
        Q_true = -(actions**2) + 2  # true Q: parabola, max at 0
        # Q estimate: correct in-distribution, wild extrapolation out-of-distribution
        Q_est_ood = Q_true.copy()
        Q_est_ood[~dataset_mask] = Q_true[~dataset_mask] + np.abs(actions[~dataset_mask])**2 * 1.5
        Q_est_cql = np.where(dataset_mask, Q_true, Q_true * 0.6)  # conservative: underestimates OOD

        fig_ext, ax_ext = _fig(1, 1, 11, 4.5)
        ax_ext.plot(actions, Q_true, color="#4caf50", lw=2.5, ls="--", label="True Q*(s,a)")
        ax_ext.plot(actions, Q_est_ood, color="#ef5350", lw=2.5, label="Standard Q-learning (wild OOD extrapolation)")
        ax_ext.plot(actions, Q_est_cql, color="#0288d1", lw=2.5, label="CQL (conservative: stays below true Q)")
        ax_ext.fill_between(actions, -4, 4, where=~dataset_mask, alpha=0.1, color="#ffa726", label="Out-of-distribution (OOD) region")
        ax_ext.fill_between(actions, -4, 4, where=dataset_mask, alpha=0.07, color="#4caf50", label="Dataset coverage")
        ax_ext.axhline(0, color="#2a2a3e", lw=0.8); ax_ext.set_ylim(-4, 6)
        ax_ext.set_xlabel("Action value", color="white"); ax_ext.set_ylabel("Q-value estimate", color="white")
        ax_ext.set_title("The Extrapolation Error Problem: Standard Q-learning overestimates OOD actions\nCQL stays conservative — safe to optimise against", color="white", fontweight="bold")
        ax_ext.legend(facecolor=CARD, labelcolor="white", fontsize=8)
        ax_ext.grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_ext); plt.close()

        st.markdown(_insight("Why extrapolation is catastrophic: if Q(s, a_ood) is overestimated, the policy will choose a_ood as its best action. But a_ood was never in the dataset — the Q estimate is wrong. The policy tries an action the model has no data for, gets terrible reward, and the offline training loop cannot correct this because there's no new environment interaction."), unsafe_allow_html=True)

    with tab_bc:
        _sec("📋","Behaviour Cloning","The simplest offline approach: clone the expert","#546e7a")
        st.markdown(_card("#546e7a","📋","Behaviour Cloning: supervised learning on demonstrations",
            """Behaviour Cloning (BC) is the simplest offline RL approach: treat it as supervised learning.
            Given dataset D = {(s_i, a_i)}, train a policy π_θ to minimise prediction error on expert actions.
            For discrete actions: cross-entropy loss. For continuous: mean squared error.
            This completely avoids the extrapolation problem — we never need Q-values.
            The limitation: distributional shift at test time. The BC policy makes small errors,
            visits states slightly off the training distribution, makes larger errors there,
            and errors compound quadratically with episode length T: cost ~ T².
            BC works well when: (1) the task is short-horizon (T < 50 steps), (2) the expert
            demonstrations are dense and high quality, (3) combined with DAgger for interactive
            correction. BC is the mandatory baseline — if your algorithm cannot beat BC, reconsider."""), unsafe_allow_html=True)

        st.markdown("**BC training objective:**")
        st.latex(r"\min_\theta \mathcal{L}_\text{BC}(\theta) = -\mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\log\pi_\theta(a|s)\right] \quad\text{(cross-entropy for discrete)}")
        st.latex(r"\mathcal{L}_\text{BC}(\theta) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\|\mu_\theta(s)-a\|^2\right] \quad\text{(MSE for continuous)}")
        st.markdown("**Error bound (Ross & Bagnell 2010) — why BC fails on long horizons:**")
        st.latex(r"J(\pi_\text{BC}) \leq J(\pi^*) - T^2\varepsilon_\text{BC} \quad\text{(quadratic in episode length T!)}")

        # Compounding error visualisation
        T = np.arange(1, 101)
        eps_bc = 0.01  # per-step error
        fig_bc, axes_bc = _fig(1, 2, 13, 4)
        for eps, col, lbl in [(0.005,"#4caf50","ε=0.5%"),(0.01,"#ffa726","ε=1%"),(0.02,"#ef5350","ε=2%")]:
            axes_bc[0].plot(T, T**2 * eps, color=col, lw=2.5, label=f"BC error T²·{lbl}")
        axes_bc[0].set_xlabel("Episode length T", color="white"); axes_bc[0].set_ylabel("Cumulative error", color="white")
        axes_bc[0].set_title("BC Compounding Error: T² growth\n(bad for long-horizon tasks)", color="white", fontweight="bold")
        axes_bc[0].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_bc[0].grid(alpha=0.12)

        # BC vs BC+IQL
        np.random.seed(42); episodes = np.arange(200)
        bc_perf = np.concatenate([np.minimum(0.6, 0.004*episodes[:100]) + np.random.randn(100)*0.02,
                                   np.ones(100)*0.58 + np.random.randn(100)*0.02])
        iql_perf = np.minimum(0.92, 0.007*episodes + np.random.randn(200)*0.025)
        axes_bc[1].plot(smooth(bc_perf,15), color="#546e7a", lw=2.5, label="BC (plateaus at 60%)")
        axes_bc[1].plot(smooth(iql_perf,15), color="#00897b", lw=2.5, label="IQL (reaches 90%+)")
        axes_bc[1].set_xlabel("Training iterations", color="white"); axes_bc[1].set_ylabel("Normalised score", color="white")
        axes_bc[1].set_title("BC vs Offline RL (IQL)\nD4RL HalfCheetah-medium", color="white", fontweight="bold")
        axes_bc[1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_bc[1].grid(alpha=0.12)
        plt.tight_layout(); st.pyplot(fig_bc); plt.close()

    with tab_cql:
        _sec("🔒","CQL — Conservative Q-Learning","Kumar et al. 2020 — the most widely used offline RL algorithm","#0288d1")
        st.markdown(_card("#0288d1","🔒","CQL: solve extrapolation by penalising OOD Q-values",
            """CQL (Kumar et al. 2020) fixes the extrapolation problem directly: add a regularisation term
            to the Q-learning objective that penalises high Q-values for out-of-distribution actions
            and rewards high Q-values for in-distribution actions (from the dataset).
            The result: the learned Q-function is a lower bound on the true Q-value — conservative.
            When the policy maximises this conservative Q, it won't be fooled by inflated OOD estimates.
            CQL is the most widely used offline RL algorithm: used in robotics (D4RL benchmark),
            healthcare policy learning, and as the offline pre-training stage before online fine-tuning
            (Cal-QL). On the D4RL benchmark, CQL consistently outperforms BC by 15–40% across tasks."""), unsafe_allow_html=True)

        st.markdown("**CQL objective — standard Q-learning + conservative regularisation:**")
        st.latex(r"\min_\phi \underbrace{\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\!\left[(Q_\phi(s,a) - y)^2\right]}_\text{Bellman error} + \underbrace{\alpha\!\left[\mathbb{E}_{s\sim\mathcal{D},a\sim\hat\mu}\![Q_\phi(s,a)] - \mathbb{E}_{(s,a)\sim\mathcal{D}}\![Q_\phi(s,a)]\right]}_\text{CQL penalty}")
        st.markdown(r"""
        **Reading the CQL penalty:**
        - First term $\mathbb{E}_{a\sim\hat\mu}[Q]$: expected Q over a broad distribution $\hat\mu$ covering OOD actions — this is **pushed DOWN**
        - Second term $\mathbb{E}_\mathcal{D}[Q]$: expected Q on dataset actions — this is **pushed UP**
        - α controls the conservatism strength (typically 1–10)

        **Theorem (Kumar et al.):** CQL Q-function is a lower bound on the true Q^π:
        """)
        st.latex(r"\mathbb{E}_\mathcal{D}[Q^\text{CQL}(s,a)] \leq \mathbb{E}_\mathcal{D}[Q^\pi(s,a)] \quad\text{(for the dataset policy π_β)}")
        st.markdown("Maximising a lower bound prevents the policy from exploiting wrong estimates.")

        # CQL vs standard Q-learning on D4RL
        tasks = ["HC-medium","HC-med-rep","Walker-med","Ant-med","Hopper-med"]
        bc_scores = [42, 36, 75, 35, 52]
        cql_scores = [74, 45, 83, 48, 86]
        iql_scores = [71, 47, 87, 48, 91]
        x = np.arange(len(tasks))
        fig_cql, ax_cql = _fig(1, 1, 11, 4.5)
        w = 0.26
        ax_cql.bar(x-w, bc_scores, w, color="#546e7a", alpha=0.85, label="BC")
        ax_cql.bar(x, cql_scores, w, color="#0288d1", alpha=0.85, label="CQL")
        ax_cql.bar(x+w, iql_scores, w, color="#00897b", alpha=0.85, label="IQL")
        ax_cql.set_xticks(x); ax_cql.set_xticklabels(tasks, rotation=15, color="white", fontsize=8)
        ax_cql.set_ylabel("D4RL Normalised Score (100=expert)", color="white")
        ax_cql.set_title("BC vs CQL vs IQL on D4RL locomotion benchmark", color="white", fontweight="bold")
        ax_cql.legend(facecolor=CARD, labelcolor="white"); ax_cql.grid(alpha=0.12, axis="y")
        ax_cql.axhline(100, color="#ffa726", ls="--", lw=1, alpha=0.5, label="Expert level")
        plt.tight_layout(); st.pyplot(fig_cql); plt.close()

    with tab_iql:
        _sec("📐","IQL — Implicit Q-Learning","Kostrikov et al. 2021 — no OOD action evaluation, expectile regression","#7c4dff")
        st.markdown(_card("#7c4dff","📐","IQL: avoid OOD actions entirely via expectile regression",
            """IQL (Kostrikov et al. 2021) takes a completely different approach to the extrapolation
            problem: never evaluate the Q-function on OOD actions at all. Instead, use expectile
            regression to implicitly fit Q*(s,a) = max_a Q(s,a) without ever taking the argmax.
            The key idea: replace the standard Bellman target y = r + γ max_a Q(s',a) with
            an expectile regression that fits the upper quantile of Q(s', a') over dataset actions.
            The policy is then extracted via advantage-weighted regression: weight each action
            by exp(β·A(s,a)) — high advantage → higher weight.
            No OOD action is ever evaluated during training. The extrapolation problem doesn't arise.
            IQL achieves state-of-the-art on D4RL and transfers naturally to online fine-tuning."""), unsafe_allow_html=True)

        st.markdown("**Expectile regression** (replaces max operator):")
        st.latex(r"\mathcal{L}_V(V_\psi) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\mathcal{L}_2^\tau(Q_{\hat\phi}(s,a)-V_\psi(s))\right]")
        st.latex(r"\mathcal{L}_2^\tau(u) = |\tau - \mathbf{1}[u<0]|\cdot u^2 \quad\text{(\tau=0.9 fits 90th percentile)}")
        st.markdown("High τ (e.g. 0.9) → V(s) estimates the maximum Q-value over dataset actions — approximating the true value function without any OOD queries.")
        st.markdown("**Advantage-weighted policy extraction:**")
        st.latex(r"\mathcal{L}_\pi(\pi_\theta) = \mathbb{E}_{(s,a)\sim\mathcal{D}}\!\left[\exp\bigl(\beta(Q_{\hat\phi}(s,a)-V_\psi(s))\bigr)\cdot(-\log\pi_\theta(a|s))\right]")
        st.markdown("Actions with positive advantage (Q > V) get high weight → policy imitates high-value actions more strongly. β controls how aggressively we exploit high-advantage actions (typically 3–10).")

        # Expectile regression illustration
        np.random.seed(42)
        a_vals = np.linspace(-2, 2, 200)
        Q_samples = -(a_vals**2) + np.random.randn(200)*0.5 + 1
        fig_iq, axes_iq = _fig(1, 2, 13, 4)
        axes_iq[0].scatter(a_vals, Q_samples, color="#546e7a", s=8, alpha=0.5, label="Q(s,a) samples")
        for tau, col, lbl in [(0.5,"#0288d1","τ=0.5 (median)"),(0.7,"#ffa726","τ=0.7"),(0.9,"#7c4dff","τ=0.9 (IQL default)")]:
            v = np.percentile(Q_samples, tau*100)
            axes_iq[0].axhline(v, color=col, lw=2, label=f"V(s) estimate {lbl}")
        axes_iq[0].set_xlabel("Action a", color="white"); axes_iq[0].set_ylabel("Q(s,a)", color="white")
        axes_iq[0].set_title("IQL Expectile Regression:\nτ=0.9 estimates max Q without OOD queries", color="white", fontweight="bold")
        axes_iq[0].legend(facecolor=CARD, labelcolor="white", fontsize=7); axes_iq[0].grid(alpha=0.12)
        # Advantage weighting
        A_vals = np.linspace(-3, 3, 200); beta = 5
        weights = np.exp(np.clip(beta*A_vals, -5, 5))
        axes_iq[1].plot(A_vals, weights, color="#7c4dff", lw=2.5)
        axes_iq[1].axvline(0, color="white", ls="--", lw=1, alpha=0.5, label="A=0: average action")
        axes_iq[1].set_xlabel("Advantage A(s,a) = Q(s,a) - V(s)", color="white")
        axes_iq[1].set_ylabel(f"Weight exp(β·A), β={beta}", color="white")
        axes_iq[1].set_title("IQL Policy Extraction: advantage-weighted\n(high advantage = high imitation weight)", color="white", fontweight="bold")
        axes_iq[1].legend(facecolor=CARD, labelcolor="white", fontsize=8); axes_iq[1].grid(alpha=0.12)
        axes_iq[1].set_ylim(0, 20)
        plt.tight_layout(); st.pyplot(fig_iq); plt.close()

    with tab_dt:
        _sec("🤖","Decision Transformer","Chen et al. 2021 — offline RL as sequence modelling with GPT","#ffa726")
        st.markdown(_card("#ffa726","🤖","Decision Transformer: reframe RL as language modelling",
            """Decision Transformer (DT, Chen et al. 2021) reframes offline RL completely:
            instead of learning Q-functions, treat the trajectory as a sequence and use a
            GPT-style transformer to predict actions. The input sequence conditions on
            return-to-go (how much reward remains), observation, and action triples.
            At test time, you specify the desired return-to-go — the model generates actions
            that achieve that return. This completely avoids the extrapolation problem:
            there are no Q-values and no Bellman backups.
            DT is particularly powerful for: (1) datasets with diverse quality levels —
            DT can be conditioned on different target returns to get different policies;
            (2) multi-task datasets — one DT model can handle many tasks with task tokens;
            (3) very long sequences where temporal credit assignment is hard."""), unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input sequence for DT:**")
            st.latex(r"\tau = (\hat R_1,s_1,a_1,\;\hat R_2,s_2,a_2,\;\ldots,\;\hat R_T,s_T,a_T)")
            st.markdown("Where:")
            st.latex(r"\hat R_t = \sum_{t'=t}^T r_{t'} \quad\text{(return-to-go from step t)}")
            st.markdown("**Cross-entropy/MSE training loss:**")
            st.latex(r"\mathcal{L}_\text{DT} = -\sum_t\log\pi_\theta(a_t|\hat R_t,s_t,\hat R_{t-1},\ldots)")
            st.markdown("**Inference:** set R̂_1 = desired return (e.g. R̂_1 = 90 for high performance), predict a_1, take a_1, compute r_1, update R̂_2 = R̂_1 − r_1, predict a_2, etc.")
        with col2:
            # Decision Transformer return conditioning
            np.random.seed(42); T_seq = 30
            rtg_high = np.maximum(0, 90 - 3*np.arange(T_seq) + np.random.randn(T_seq)*2)
            rtg_low  = np.maximum(0, 40 - 1.5*np.arange(T_seq) + np.random.randn(T_seq)*2)
            fig_dt, ax_dt = _fig(1,1,5.5,4)
            ax_dt.plot(rtg_high, color="#4caf50", lw=2.5, label="Target return=90 (expert)")
            ax_dt.plot(rtg_low,  color="#ef5350", lw=2.5, label="Target return=40 (medium)")
            ax_dt.set_xlabel("Timestep t", color="white"); ax_dt.set_ylabel("Return-to-go R̂_t", color="white")
            ax_dt.set_title("DT: Return-to-go conditioning\ndecreases as rewards are collected", color="white", fontweight="bold")
            ax_dt.legend(facecolor=CARD, labelcolor="white", fontsize=8); ax_dt.grid(alpha=0.12)
            plt.tight_layout(); st.pyplot(fig_dt); plt.close()

        st.dataframe(pd.DataFrame({
            "Dataset quality": ["Expert only","Medium","Medium-replay","Random+expert mix"],
            "BC score": [107,42,36,88],
            "CQL score": [107,74,45,97],
            "DT score": [107,67,41,95],
            "IQL score": [107,71,47,99],
        }), use_container_width=True, hide_index=True)
        st.markdown(_insight("DT is competitive with CQL/IQL on expert and high-quality data, but underperforms on 'medium' and 'medium-replay' datasets where the sequence structure is less clear. DT's strength is long-horizon tasks and multi-task settings."), unsafe_allow_html=True)

    with tab_td3bc:
        _sec("🎯","TD3+BC — The Simplest Competitive Baseline","Fujimoto & Gu 2021 — add one BC term to TD3, beat complex methods","#e65100")
        st.markdown(_card("#e65100","🎯","TD3+BC: one line of code beats complicated algorithms",
            """TD3+BC (Fujimoto & Gu 2021) makes a stunning point: add a single BC regularisation
            term to TD3 and you get a competitive offline RL algorithm that beats CQL on many tasks.
            The actor loss becomes: minimise λ·(-Q(s,a)) + (a - a_data)² — standard TD3 actor loss
            plus an MSE term that keeps the policy close to the dataset actions.
            The λ term normalises the Q-value scale. That's it. 5 lines of change from standard TD3.
            TD3+BC highlights that simple baselines should always be tried before complex methods.
            It's competitive because the BC term acts as an implicit constraint on OOD actions:
            if the policy deviates far from dataset actions, the MSE penalty increases.
            The Q-function still extrapolates, but the policy is prevented from exploiting it."""), unsafe_allow_html=True)

        st.markdown("**TD3+BC actor loss:**")
        st.latex(r"\mathcal{L}_\pi(\theta) = -\underbrace{\lambda Q_\phi(s,\pi_\theta(s))}_\text{TD3 Q-maximisation} + \underbrace{\|\pi_\theta(s) - a\|^2}_\text{BC regularisation}")
        st.latex(r"\lambda = \frac{\alpha}{\frac{1}{N}\sum_{(s_i,a_i)\sim\mathcal{D}}|Q_\phi(s_i,a_i)|} \quad\text{(normalise Q scale)}")
        st.markdown("α controls the BC strength (typically 2.5). λ adapts as Q-values change during training.")

        st.code("""
# TD3+BC: add just 3 lines to standard TD3 actor update
def actor_update(batch, Q_network, actor_network, alpha=2.5):
    s, a_dataset, _, _, _ = batch
    a_policy = actor_network(s)

    # Standard TD3: maximise Q
    Q_values = Q_network(s, a_policy)

    # Normalise Q scale (prevents BC term being overwhelmed)
    lam = alpha / Q_values.abs().mean().detach()

    # TD3+BC loss = -Q + BC penalty
    actor_loss = -lam * Q_values.mean() + F.mse_loss(a_policy, a_dataset)
    return actor_loss
""", language="python")

    with tab_cmp:
        _sec("📊","Offline RL Benchmark — D4RL Results","Normalised scores across all major algorithms and datasets","#00897b")
        st.dataframe(pd.DataFrame({
            "Algorithm": ["BC","BCQ","CQL","TD3+BC","IQL","Decision Transformer","Cal-QL (offline→online)"],
            "HC medium": [42,61,74,59,71,67,75],
            "HC med-rep": [36,53,45,58,47,41,69],
            "Walker medium": [75,84,83,88,87,74,91],
            "Ant medium": [35,55,48,90,48,63,89],
            "Hopper medium": [52,59,86,91,91,67,97],
            "Requires env?": ["No","No","No","No","No","No","Yes (fine-tune)"],
        }), use_container_width=True, hide_index=True)
        st.caption("Scores are D4RL normalised: 0=random, 100=expert-level. HC=HalfCheetah.")

    with tab_res:
        _sec("📚","Books & Deep-Dive Resources","The best offline RL papers, books, and courses","#546e7a")
        for title, authors, why, url in [
            ("Offline Reinforcement Learning: Tutorial, Review, and Perspectives",
             "Levine, Kumar, Tucker, Fu (2020) — NeurIPS tutorial",
             "THE comprehensive offline RL survey. Covers all approaches, theory, and benchmarks in 100+ pages.",
             "https://arxiv.org/abs/2005.01643"),
            ("Conservative Q-Learning for Offline RL (CQL)",
             "Kumar et al. (2020) — NeurIPS — The most widely used offline RL algorithm",
             "Full derivation of the conservative lower bound. Appendix has convergence proof.",
             "https://arxiv.org/abs/2006.04779"),
            ("Offline RL with Implicit Q-Learning (IQL)",
             "Kostrikov, Nair, Levine (2021) — ICLR 2022",
             "Expectile regression derivation. Clean code available. Transfers to online RL well.",
             "https://arxiv.org/abs/2110.06169"),
            ("Decision Transformer: RL via Sequence Modelling",
             "Chen et al. (2021) — NeurIPS",
             "Reframes offline RL as GPT-style language modelling. Code on GitHub.",
             "https://arxiv.org/abs/2106.01345"),
            ("D4RL: Datasets for Deep Data-Driven RL",
             "Fu et al. (2020) — The standard offline RL benchmark",
             "Defines the benchmark datasets (HalfCheetah-medium, Walker-expert, etc.) used in all papers.",
             "https://arxiv.org/abs/2004.07219"),
            ("A Minimalist Approach to Offline RL (TD3+BC)",
             "Fujimoto & Gu (2021) — NeurIPS",
             "Shows that simple BC regularisation on TD3 beats complex methods. Highly readable.",
             "https://arxiv.org/abs/2106.06860"),
        ]:
            st.markdown(_book(title, authors, why, url), unsafe_allow_html=True)

    offline_notes = [
        (tab_why, "Why Offline RL", "offline_batch_rl"),
        (tab_bc, "Behaviour Cloning", "offline_batch_rl_behaviour_cloning"),
        (tab_cql, "CQL", "offline_batch_rl_cql"),
        (tab_iql, "IQL", "offline_batch_rl_iql"),
        (tab_dt, "Decision Transformer", "offline_batch_rl_decision_transformer"),
        (tab_td3bc, "TD3+BC", "offline_batch_rl_td3_bc"),
        (tab_cmp, "Comparison", "offline_batch_rl_comparison"),
        (tab_res, "Resources", "offline_batch_rl_resources"),
    ]
    for tab, note_title, note_slug in offline_notes:
        with tab:
            render_notes(f"Offline / Batch RL - {note_title}", note_slug)
