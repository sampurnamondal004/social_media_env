"""social_media_env_environment.py"""
import os
import uuid
import random
from typing import List, Dict, Any, Optional

# Framework imports
from openenv.core.env_server import Environment
import social_media_env.reward as reward_
from social_media_env.models import (
    FeedRankingAction,
    FeedRankingObservation,
    FeedRankingState,
)
from .client import EnvClient

class Post:
    """A candidate post that can be placed in the feed."""
    def __init__(self, post_id: str, topic: str, source: str, age_hours: float, 
                 quality_score: float, base_ctr: float, is_clickbait: bool = False):
        self.post_id = post_id
        self.topic = topic
        self.source = source
        self.age_hours = age_hours
        self.quality_score = quality_score
        self.base_ctr = base_ctr
        self.is_clickbait = is_clickbait

class FeedRankingEnvironment(Environment):
    """
    Social Media Feed Ranking Environment.
    Simulates user engagement based on post characteristics and topic interests.
    """
    TOPICS = ["sports", "politics", "tech", "entertainment", "health",
              "science", "travel", "finance", "food", "gaming"]
    SOURCES = [f"publisher_{i}" for i in range(12)]

    def __init__(
        self,
        feed_slots: int = 10,
        pool_size: int = 30,
        max_steps: int = 50,
        user_profile: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        actual_url = base_url or os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        self.client = EnvClient(base_url=actual_url)
        # 1. Configuration & Randomness
        self._feed_slots = feed_slots
        self._pool_size = pool_size
        self._max_steps = max_steps
        self._user_profile = user_profile
        self._rng = random.Random(seed)
        
        # 2. Logic Components
        self._rubric = reward_.FeedRankingRubric()
        
        # 3. Client Initialization (Crucial for avoiding TypeErrors)
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        self.client = EnvClient(base_url=self.base_url)
        
        # 4. State Management
        self._state: Optional[FeedRankingState] = None
        self.reset()

    # --- Public API ---

    def reset(self) -> FeedRankingObservation:
        """Starts a fresh episode with a new candidate pool and profile."""
        pool = self._generate_pool()

        interests = dict(self._user_profile) if self._user_profile else self._generate_user_profile()

        self._state = FeedRankingState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            cumulative_reward=0.0,
            feed=[],
            placed_ids=set(),
            topic_counts={},
            source_counts={},
            user_interest_vector=interests,
            candidate_pool=pool,
        )

        return self._build_obs(reward=0.0, done=False)

    def step(self, action: FeedRankingAction) -> FeedRankingObservation:
        """Places a post and evaluates engagement."""
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        state = self._state

        # Check if feed is already full
        if len(state.feed) >= self._feed_slots:
            penalty, _ = self._rubric.wasted_step()
            state.cumulative_reward += penalty
            return self._build_obs(reward=penalty, done=True, info={"wasted_step": True})

        post = self._find_post(action.post_id)

        # Handle invalid post IDs
        if post is None:
            penalty = -0.15
            state.step_count += 1
            state.cumulative_reward += penalty
            done = state.step_count >= self._max_steps
            return self._build_obs(reward=penalty, done=done, info={"invalid_id": action.post_id})

        # Process valid placement
        slot_index = len(state.feed)
        reward, breakdown = self._rubric(post, state, slot_index)

        state.feed.append(post.post_id)
        state.placed_ids.add(post.post_id)
        state.topic_counts[post.topic] = state.topic_counts.get(post.topic, 0) + 1
        state.source_counts[post.source] = state.source_counts.get(post.source, 0) + 1
        
        # Filter pool and update progression
        state.candidate_pool = [p for p in state.candidate_pool if p.post_id != post.post_id]
        state.step_count += 1
        state.cumulative_reward += reward

        done = (len(state.feed) >= self._feed_slots or 
                state.step_count >= self._max_steps or 
                not state.candidate_pool)

        return self._build_obs(reward=reward, done=done, info={"breakdown": breakdown.as_dict()})

    @property
    def state(self) -> Optional[FeedRankingState]:
        return self._state

    # --- Private Helpers ---

    def _build_obs(self, reward: float, done: bool, info: Optional[dict] = None) -> FeedRankingObservation:
        s = self._state
        candidate_view = [
            {
                "post_id": p.post_id,
                "topic": p.topic,
                "source": p.source,
                "age_hours": round(p.age_hours, 1),
                "quality_score": round(p.quality_score, 3),
                "base_ctr": round(p.base_ctr, 3),
                "is_clickbait": p.is_clickbait,
            } for p in s.candidate_pool
        ]
        return FeedRankingObservation(
            feed=list(s.feed),
            candidate_pool=candidate_view,
            user_interest_vector=dict(s.user_interest_vector),
            step=s.step_count,
            max_steps=self._max_steps,
            reward=reward,
            cumulative_reward=s.cumulative_reward,
            done=done,
            info=info or {},
        )

    def _find_post(self, post_id: str) -> Optional[Post]:
        for p in self._state.candidate_pool:
            if p.post_id == post_id:
                return p
        return None

    def _generate_pool(self) -> List[Post]:
        posts = []
        for i in range(self._pool_size):
            topic = self._rng.choice(self.TOPICS)
            source = self._rng.choice(self.SOURCES)
            quality = self._rng.betavariate(2, 2)
            clickbait = self._rng.random() < 0.12
            if clickbait:
                quality = min(quality, 0.35)

            posts.append(Post(
                post_id=f"post_{i:04d}_{uuid.uuid4().hex[:6]}",
                topic=topic,
                source=source,
                age_hours=self._rng.expovariate(1 / 18),
                quality_score=quality,
                base_ctr=self._rng.betavariate(1.5, 6),
                is_clickbait=clickbait,
            ))
        return posts

    def _generate_user_profile(self) -> Dict[str, float]:
        weights = {t: round(self._rng.random() * 0.15, 3) for t in self.TOPICS}
        hot = self._rng.sample(self.TOPICS, k=self._rng.randint(2, 4))
        for t in hot:
            weights[t] = round(0.6 + self._rng.random() * 0.4, 3)
        max_w = max(weights.values())
        return {t: round(w / max_w, 3) for t, w in weights.items()}