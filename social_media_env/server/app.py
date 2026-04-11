"""FastAPI application for the Social Media Feed Ranking Environment."""
import os
from fastapi import Request
from fastapi.responses import JSONResponse
from openenv.core.env_server import create_app
from ..models import FeedRankingAction, FeedRankingObservation
from ..social_media_env import FeedRankingEnvironment

#API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")

HF_API_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1" )
def env_factory():
    return FeedRankingEnvironment(base_url=HF_API_URL)


app = create_app(
    env_factory,
    FeedRankingAction, 
    FeedRankingObservation,
    env_name="social_media_env"
    tasks=ALL_TASKS
)


@app.post("/grader")
async def grader(request: Request):
    body = await request.json()
    task_id = body.get("task_id", "engagement_optimization")
    rewards = body.get("rewards", [])
    steps = body.get("steps", 0)

    if not rewards:
        score = 0.001
    else:
        raw = sum(rewards) / (len(rewards) * 1.0)
        score = min(max(raw, 0.001), 0.999)

    return JSONResponse({
        "task_id": task_id,
        "score": score,
        "success": score >= 0.1,
        "steps": steps,
    })

def main():
    """Entry point for the server script."""
    import uvicorn
   
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("social_media_env.server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()