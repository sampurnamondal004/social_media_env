---
title: Social Media Env
emoji: 📱
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Social Media Feed Ranking Environment

This repository contains a **FastAPI-based environment server** for simulating and ranking social media feeds. It is designed to work with the **OpenEnv** framework.

## Deployment Features
- **SDK**: Docker
- **Framework**: FastAPI
- **Core Logic**: Social Media Feed Ranking
- **Dependencies**: OpenEnv, NumPy, Pydantic

## Project Structure
- `app.py`: The main FastAPI entry point.
- `social_media_env.py`: Core environment logic and state management.
- `model.py`: Data schemas for Actions, Observations, and States.
- `reward.py`: Logic for calculating engagement and relevance scores.
- `Dockerfile`: Container configuration for Hugging Face Spaces.

## Local Setup
If you want to run this locally in your VS Code environment:

1. **Activate your virtual environment**:
   ```powershell
   .venv\Scripts\Activate.ps1