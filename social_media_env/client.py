"""
Social Media Feed Ranking Environment Client.
This module provides the client for connecting to a Social Media Feed Ranking 
Environment server via WebSocket for persistent sessions.
"""
from __future__ import annotations
from typing import Any, Dict
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from social_media_env.models import FeedRankingAction, FeedRankingObservation, FeedRankingState


class SocialFeedEnv(EnvClient[FeedRankingAction, FeedRankingObservation, FeedRankingState]):
    """
    Client for Social Media Feed Ranking Environment.
    
    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    
    The environment simulates a social media feed ranking system where an agent
    learns to select posts that maximize user engagement by understanding user
    preferences and post characteristics.
    
    Example:
        >>> with SocialFeedEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.user_interests)
        ...     print(result.observation.candidate_posts)
        ...
        ...     # Agent selects post index 3 from candidates
        ...     result = client.step(FeedAction(post_index=3))
        ...     print(result.reward, result.done)
    """

    def _step_payload(self, action: FeedRankingAction) -> Dict[str, Any]:
        """
        Convert FeedAction to JSON payload for step request.
        
        Args:
            action: FeedAction instance with selected post index.
        
        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "post_index": action.post_index,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FeedRankingObservation]:
        """
        Parse server response into StepResult[FeedObservation].
        
        Args:
            payload: JSON response from server.
        
        Returns:
            StepResult with FeedObservation.
        """
        obs_data = payload.get("observation", {})
        observation = FeedRankingObservation(
            user_interests=obs_data.get("user_interests", []),
            user_context=obs_data.get("user_context", {}),
            candidate_posts=obs_data.get("candidate_posts", []),
            session_progress=obs_data.get("session_progress", 0.0),
            engagement_history=obs_data.get("engagement_history", []),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FeedRankingState:
        """
        Parse server response into FeedState object.
        
        Args:
            payload: JSON response from /state endpoint.
        
        Returns:
            FeedState object with environment state information.
        """
        return FeedRankingState(
            episode_id=payload.get("episode_id", ""),
            user_profile=payload.get("user_profile", {}),
            shown_post_ids=payload.get("shown_post_ids", []),
            total_engagement=payload.get("total_engagement", 0.0),
            step_count=payload.get("step_count", 0),
            avg_engagement=payload.get("avg_engagement", 0.0),
        )