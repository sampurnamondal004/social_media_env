"""FastAPI application for the Social Media Feed Ranking Environment."""
from openenv.core.env_server import create_app
from ..models import FeedRankingAction, FeedRankingObservation
from .. import FeedRankingEnvironment

# Create the FastAPI app
# Pass the class (factory) instead of an instance for WebSocket session support
app = create_app(
    FeedRankingEnvironment, 
    FeedRankingAction, 
    FeedRankingObservation,
    env_name="social_media_env"
)


def main():
    """Entry point for the server script."""
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("social_media_env.server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()