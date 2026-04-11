from __future__ import annotations
import os
from openai import OpenAI
from typing import Any, Dict, Optional
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from social_media_env.models import FeedRankingAction, FeedRankingObservation, FeedRankingState

class SocialFeedEnv(EnvClient[FeedRankingAction, FeedRankingObservation, FeedRankingState]):
    """
    Client for Social Media Feed Ranking Environment.
    Optimized to work with the OpenEnv Proxy.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
       
        target_url = os.getenv("API_BASE_URL", base_url)
        target_key = os.getenv("HF_TOKEN", api_key)
        
        
        super().__init__(base_url=target_url, api_key=target_key, **kwargs)

    def _step_payload(self, action: FeedRankingAction) -> Dict[str, Any]:
        return {"post_id": action.post_id}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FeedRankingObservation]:
        obs_data = payload.get("observation", {})
        observation = FeedRankingObservation(
            feed=obs_data.get("feed", []),
            candidate_pool=obs_data.get("candidate_pool", []),
            user_interest_vector=obs_data.get("user_interest_vector", {}),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 50),
            reward=obs_data.get("reward", 0.0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            done=obs_data.get("done", False),
            info=obs_data.get("info", {}),
        )
        
        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> FeedRankingState:
        
        return FeedRankingState(
            episode_id=payload.get("episode_id", ""),
            user_interest_vector=payload.get("user_interest_vector", {}), 
            placed_ids=set(payload.get("placed_ids", [])),             
            cumulative_reward=payload.get("cumulative_reward", 0.0),   
            step_count=payload.get("step_count", 0),
            candidate_pool=payload.get("candidate_pool", []),
        )