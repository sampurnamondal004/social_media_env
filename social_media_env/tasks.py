from openenv.core.task import Task, Grader
from social_media_env.models import FeedRankingState
from typing import Any

class EngagementGrader(Grader):
    def grade(self, state: FeedRankingState, episode_data: Any) -> float:
        return 1.0 if state.cumulative_reward > 5.0 else 0.0

class RelevanceGrader(Grader):
    def grade(self, state: FeedRankingState, episode_data: Any) -> float:
        hot_topics = [t for t, v in state.user_interest_vector.items() if v > 0.6]
        return 1.0 if state.cumulative_reward > 3.0 else 0.0 

engagement_task = Task(
    id="engagement_optimization",
    env_kwargs={"max_steps": 15, "pool_size": 40},
    grader=EngagementGrader()
)

relevance_task = Task(
    id="relevance_ranking",
    env_kwargs={"max_steps": 10, "pool_size": 20},
    grader=RelevanceGrader()
)    

diversity_task = Task(
    id="diversity_optimization",
    env_kwargs={"max_steps": 20, "pool_size": 50},
    grader=EngagementGrader()
)

ALL_TASKS = [engagement_task, relevance_task, diversity_task]