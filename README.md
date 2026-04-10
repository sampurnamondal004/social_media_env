---
title: Social Media Env
emoji: 📱
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Social Media Feed Ranking RL Environment

A **FastAPI-based reinforcement learning environment** for simulating and optimising social media feed ranking. Built on the [OpenEnv](https://github.com/openenv) framework, this project models the feed ranking problem as a Markov Decision Process — where an RL agent learns to order content to maximise user engagement and relevance scores.

> Directly analogous to the personalisation and feed-ranking pipelines used in large-scale consumer platforms.

---

## What it does

The environment exposes a gym-style RL interface over HTTP, allowing any RL agent to:

- Observe the current feed state (post features, user context, engagement history)
- Submit a ranking action (reordered list of content items)
- Receive a reward signal based on engagement and relevance scoring
- Step through episodes to train and evaluate ranking policies

---

## Architecture

```
RL Agent (external)
       │
       │  HTTP (GET /observe, POST /step, POST /reset)
       ▼
  FastAPI Server  (app.py)
       │
       ├──▶  social_media_env.py   # Core environment logic & state management
       ├──▶  model.py              # Pydantic schemas for Actions, Observations, States
       └──▶  reward.py             # Engagement + relevance reward functions
```

---

## Tech Stack

| Component | Technology |
|---|---|
| API Server | Python, FastAPI |
| Data Validation | Pydantic |
| Numerical Computing | NumPy |
| RL Framework | OpenEnv |
| Containerisation | Docker, docker-compose |
| Deployment Target | Hugging Face Spaces (Docker SDK) |

---

## Reward Design

The reward function in `reward.py` combines two signals:

| Signal | Description |
|---|---|
| **Engagement score** | Models click-through likelihood based on content features and user history |
| **Relevance score** | Measures semantic alignment between content and inferred user interest |

Final reward: `R = α · engagement + (1 - α) · relevance`

where `α` is a tunable weight parameter (default: `0.6`).

---

## Local Setup

### Prerequisites
- Python 3.10+
- Docker & docker-compose (for containerised setup)

### Option 1 — Virtual environment

```bash
# 1. Clone the repo
git clone https://github.com/sampurnamondal004/social_media_env.git
cd social_media_env

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --reload --port 8000
```

### Option 2 — Docker

```bash
docker-compose up --build
```

Server runs at `http://localhost:8000`.

---

## API Reference

### `POST /reset`
Resets the environment and returns the initial observation.

**Response:**
```json
{
  "observation": {
    "feed": [...],
    "user_context": {...},
    "step": 0
  }
}
```

---

### `GET /observe`
Returns the current environment state without stepping.

---

### `POST /step`
Submits a ranking action and returns the next observation, reward, and episode status.

**Request body:**
```json
{
  "ranked_item_ids": ["post_3", "post_1", "post_5", "post_2", "post_4"]
}
```

**Response:**
```json
{
  "observation": { "feed": [...], "user_context": {...}, "step": 1 },
  "reward": 0.74,
  "done": false,
  "info": {
    "engagement_score": 0.81,
    "relevance_score": 0.63
  }
}
```

---

## Project Structure

```
├── app.py                  # FastAPI entry point, route definitions
├── social_media_env.py     # Environment logic, episode management, state transitions
├── model.py                # Pydantic schemas: Action, Observation, State, StepResult
├── reward.py               # Engagement and relevance reward calculation
├── inference.py            # Utility for running inference against a trained policy
├── Dockerfile              # Container config for Hugging Face Spaces deployment
├── docker-compose.yml      # Local multi-container setup
├── openenv.yaml            # OpenEnv framework configuration
├── validate-submission.sh  # Automated environment validation script
├── pyproject.toml          # Python project metadata and dependency specs
└── requirements.txt        # Pip dependencies
```

---

## Validation

Before submitting or deploying, run the included validation script:

```bash
bash validate-submission.sh
```

This checks that all API endpoints respond correctly, reward values are within expected bounds, and the episode lifecycle (reset → step → done) completes without errors.

---

## Key Design Decisions

- **HTTP-based environment interface** — Decouples the RL agent from the environment implementation, allowing agents written in any language or framework to interact with the environment over the network.
- **Pydantic schemas throughout** — Strict input/output validation on all API boundaries prevents silent data errors during training.
- **Modular reward system** — Separating reward logic into `reward.py` makes it easy to experiment with different reward formulations without touching environment or API code.
- **Docker-first deployment** — One-command setup eliminates environment reproducibility issues across machines.

---

## Future Improvements

- [ ] Multi-user simulation with distinct preference profiles
- [ ] Historical session context for sequential recommendation
- [ ] Baseline policy implementations (random, greedy, popularity-based)
- [ ] Wandb / TensorBoard integration for reward tracking during training
- [ ] REST client SDK for Python agents

---

## Author

**Sampurna Mondal** — [github.com/sampurnamondal004](https://github.com/sampurnamondal004)

B.Tech CSE, IIIT Agartala | Summer Research Intern, IIT Guwahati (2024)
