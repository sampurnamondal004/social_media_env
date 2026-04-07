"""FastAPI application for the Social Media Feed Ranking Environment."""
from openenv.core.env_server import create_app
from model import FeedRankingAction, FeedRankingObservation
from social_media_env import FeedRankingEnvironment

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
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()