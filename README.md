# Deep RL Learning Portal

An interactive, self-contained study environment for learning Reinforcement Learning — from mathematical foundations through frontier research. Built with Streamlit, it runs entirely on your machine with no external services required.

## What it contains

- **16 learning modules** covering the full RL curriculum: Math Foundations → Deep Learning Prerequisites → Dynamic Programming → Monte Carlo → Temporal Difference → Value-Based Deep RL (DQN→Rainbow) → Continuous Control (DDPG, TD3) → Actor-Critic (PPO, SAC, A2C) → Imitation Learning (BC, GAIL, AIRL) → Model-Based RL (Dyna-Q, MuZero, DreamerV3) → Offline RL (CQL, IQL, Decision Transformer) → Exploration (UCB, ICM, RND) → Advanced (MARL, HRL, Safe RL) → Transfer & GRPO → Practical Engineering → Frontier Research (RLHF, Foundation Models)
- **Interactive simulations** — run DP/MC/TD algorithms on live GridWorld and BlackJack environments, visualise value functions, Q-tables, and learning curves
- **60+ algorithms** with formula derivations, pseudocode, and interactive parameter sliders
- **Persistent per-section notes** — write Markdown + LaTeX, attach images; notes survive app restarts
- **Study material tracker** — manage books (PDFs in `StudyMaterial/Books/`) and external links with status and scores
- **Discussion board** — post questions and notes that persist across sessions
- **Learning roadmap** — staged curriculum from Stage −1 (math) through Tier 4 (frontier) with decision trees for algorithm selection

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone <repo-url>
cd RL

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# Install dependencies
pip install streamlit numpy matplotlib pandas
```

## Running the portal

```bash
streamlit run rl_portal.py
```

The portal opens in your browser at `http://localhost:8501`. No internet connection is needed after installation.

## Navigating the portal

### Home page tabs

| Tab | What it does |
|---|---|
| **Learning Roadmap** | Staged curriculum cards — read these to understand the recommended study order |
| **Interactive Map** | Visual knowledge tree; click "Open →" buttons to jump to any module |
| **Method Comparison** | Side-by-side comparison of all algorithm families |
| **When to Use Which** | Decision-tree questionnaire that recommends an algorithm for your problem |
| **All Modules** | Grid of every module with a direct open button |
| **Study Material** | PDF book reader and external link tracker |
| **Discussion Board** | Post and browse notes and questions |

### Navigating inside a module

Each module opens as a full page with its own tab bar. Use the **← Back** button (top-left) to return to the home page.

Every tab inside a module ends with a **Notes** panel where you can:
- Write free-form Markdown text
- Insert LaTeX formulas (wraps input in `$$...$$` automatically)
- Upload and embed images (stored locally in `section_notes/<slug>/assets/`)
- Save and reload notes independently per section

### Recommended study order

Follow the stages in the Learning Roadmap tab:

```
Stage -1  Math & CS Foundations        (~3–4 weeks)
Stage  0  Deep Learning Prerequisites  (~1–2 weeks)
Stage  1  Dynamic Programming          (~1 week)
Stage  2  Monte Carlo Methods          (~1 week)
Stage  3  Temporal-Difference Learning (~1 week)
Stage  4  Value-Based Deep RL          (~2 weeks)
Stage  4b Continuous Control           (~1 week)
Stage  5  Actor-Critic / PPO / SAC     (~2 weeks)
Stage  6  Imitation Learning           (~1 week)
Tier   1  Model-Based / Offline / Exploration
Tier   2  Advanced / Transfer / GRPO
Tier   3  Practical Engineering
Tier   4  Frontier Research
```

Each stage card in the Roadmap tab lists what to study, why it matters, and a concrete milestone to verify understanding before moving on.

## Adding study materials

**Books (PDF):** Place PDF files in `StudyMaterial/Books/`. They appear automatically in the Study Material tab.

**Links:** Use the Study Material tab → "Add new resource" form. Links are saved to `portal_data/study_links.json` and survive restarts.

## Data storage

All user data is stored locally as plain files:

| Path | Contents |
|---|---|
| `section_notes/<slug>/note.md` | Your notes for each section |
| `section_notes/<slug>/assets/` | Images embedded in notes |
| `portal_data/discussion_posts.json` | Discussion board posts |
| `portal_data/study_links.json` | Study link tracker |
| `StudyMaterial/Books/` | PDF books |

## Project structure

```
rl_portal.py            # Entry point and page router
_*_mod.py               # One file per learning module (loaded lazily)
_notes_mod.py           # Shared persistent notes component
_discussion_mod.py      # Discussion board component
_study_material_mod.py  # Book browser and link tracker
portal_data/            # JSON persistence for links and discussion posts
section_notes/          # Markdown notes written by the user
StudyMaterial/Books/    # PDF study books
```

Each `_*_mod.py` exports a single `main_<name>()` function that `rl_portal.py` calls when the corresponding page is selected.
