from __future__ import annotations
import os
from typing import Any, Dict, Optional
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from social_media_env.models import FeedRankingAction, FeedRankingObservation, FeedRankingState

class SocialFeedEnv(EnvClient[FeedRankingAction, FeedRankingObservation, FeedRankingState]):

    def __init__(self, base_url: Optional[str] = None, **kwargs):
        # ✅ base_url is the env Space URL — NOT the LLM API URL
        env_url = base_url or os.getenv(
            "ENV_URL",
            "https://sampurnamondal012-ocial-media-ranking-env.hf.space"
        )
        super().__init__(base_url=env_url, **kwargs)

    def _step_payload(self, action: FeedRankingAction) -> Dict[str, Any]:
        return {"post_id": action.post_id}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FeedRankingObservation]:
        obs_data = payload.get("observation", payload)  # fallback to payload itself
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
            step_count=payload.get("step_count", 0),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            feed=payload.get("feed", []),
            placed_ids=payload.get("placed_ids", []),
            topic_counts=payload.get("topic_counts", {}),
            source_counts=payload.get("source_counts", {}),
            user_interest_vector=payload.get("user_interest_vector", {}),
            candidate_pool=payload.get("candidate_pool", []),
        )