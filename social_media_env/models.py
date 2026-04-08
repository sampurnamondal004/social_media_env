from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class FeedRankingAction(Action):
    post_id: str  # ✅ string, not int


class FeedRankingObservation(Observation):
    feed: List[str] = Field(default_factory=list)
    candidate_pool: List[Dict] = Field(default_factory=list)
    user_interest_vector: Dict[str, float] = Field(default_factory=dict)
    step: int = 0
    max_steps: int = 50
    reward: float = 0.0
    cumulative_reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class FeedRankingState(State):
    episode_id: str = ""
    step_count: int = 0
    cumulative_reward: float = 0.0
    feed: List[str] = Field(default_factory=list)
    placed_ids: List[str] = Field(default_factory=list)  # set → list for Pydantic
    topic_counts: Dict[str, int] = Field(default_factory=dict)
    source_counts: Dict[str, int] = Field(default_factory=dict)
    user_interest_vector: Dict[str, float] = Field(default_factory=dict)
    candidate_pool: List[Any] = Field(default_factory=list)


FeedAction = FeedRankingAction
FeedObservation = FeedRankingObservation
FeedState = FeedRankingState