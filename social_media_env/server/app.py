"""FastAPI application for the Social Media Feed Ranking Environment."""
import os
from fastapi import Request
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app
from ..models import FeedRankingAction, FeedRankingObservation
from ..social_media_env import FeedRankingEnvironment

HF_API_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

def env_factory():
    return FeedRankingEnvironment(base_url=HF_API_URL)


app = create_app(
    env_factory,
    FeedRankingAction,
    FeedRankingObservation,
    env_name="social_media_env"
)


@app.get("/tasks")
async def get_tasks():
    return {
        "tasks": [
            {"id": "engagement_optimization", "grader": "/grader", "weight": 0.4},
            {"id": "relevance_ranking", "grader": "/grader", "weight": 0.3},
            {"id": "diversity_optimization", "grader": "/grader", "weight": 0.3},
        ]
    }


@app.post("/grader")
async def grader(request: Request):
    body = await request.json()
    task_id = body.get("task_id", "engagement_optimization")
    rewards = body.get("rewards", [])
    steps = body.get("steps", 0)
    episode_id = body.get("episode_id", "")

    if not rewards:
        raw = 0.05
    else:
        raw = sum(rewards) / (len(rewards) * 1.0)

    
    score = max(0.001, min(0.994, raw))

    return JSONResponse({
        "score": score,
        "task_id": task_id,
        "episode_id": episode_id,
        "scenario_id": f"{task_id}_001",
        "breakdown": {
            "relevance": max(0.001, min(0.994, score * 0.4)),
            "quality": max(0.001, min(0.994, score * 0.3)),
            "diversity": max(0.001, min(0.994, score * 0.3)),
        },
        "grader_version": "1.0.0"
    })


def main():
    """Entry point for the server script."""
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("social_media_env.server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()