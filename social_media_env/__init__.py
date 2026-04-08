"""
Social Media Feed Ranking Environment for OpenEnv.

This module provides OpenEnv integration for social media feed ranking,
simulating user engagement with posts to train recommendation agents.

Example:
    >>> from envs.social_feed_env import SocialFeedEnv, FeedAction
    >>>
    >>> # Connect to a running server or start via Docker
    >>> env = SocialFeedEnv.from_docker_image("social-feed-env:latest")
    >>>
    >>> # Reset and interact
    >>> result = env.reset()
    >>> print(result.observation.user_interests)
    >>> print(result.observation.candidate_posts)
    >>>
    >>> # Agent selects post at index 3 from candidates
    >>> result = env.step(FeedAction(post_index=3))
    >>> print(result.reward, result.done)
    >>>
    >>> # Cleanup
    >>> env.close()
"""
from .client import SocialFeedEnv
from .models import FeedRankingState, FeedRankingAction, FeedRankingObservation
FeedRankingEnvironment = SocialFeedEnv
__all__ = ["SocialFeedEnv", 
           "FeedRankingAction", 
           "FeedRankingObservation", 
           "FeedRankingState", 
           "FeedRankingEnvironment"]