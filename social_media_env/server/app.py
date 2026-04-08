"""FastAPI application for the Social Media Feed Ranking Environment."""
import os
from openenv.core.env_server import create_app
from ..models import FeedRankingAction, FeedRankingObservation
from .. import FeedRankingEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")


def env_factory():
    return FeedRankingEnvironment(base_url=API_BASE_URL)


app = create_app(
    FeedRankingEnvironment, 
    FeedRankingAction, 
    FeedRankingObservation,
    env_name="social_media_env"
)


def main():
    """Entry point for the server script."""
    import uvicorn
   
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("social_media_env.server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()