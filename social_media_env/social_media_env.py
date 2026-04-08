"""social_media_env_environment.py"""
import os
import random
import uuid
from dataclasses import dataclass
from typing import List, Any, Dict
from .client import EnvClient
import numpy as num
from openenv.core.env_server import Environment
import social_media_env.reward as reward_  
#class FeedRankingEnvironment:
 #   def __init__(self):
  #      self.rubric = reward_.FeedRankingRubric()
   #     self.breakdown = reward_.RewardBreakdown()

from social_media_env.models import (
    FeedRankingAction,
    FeedRankingObservation,
    FeedRankingState,
)
# Data Models

@dataclass
class Post:
    """A candidate post that can be placed in the feed."""
    post_id: str
    topic: str
    source: str
    age_hours: float
    quality_score: float
    base_ctr: float
    is_clickbait: bool = False



#Environment
class FeedRankingEnvironment(Environment):
    TOPICS   = ["sports", "politics", "tech", "entertainment", "health",
                "science", "travel", "finance", "food", "gaming"]
    SOURCES  = [f"publisher_{i}" for i in range(12)]

    def __init__(
        self,
        feed_slots: int = 10,
        pool_size: int = 30,
        max_steps: int = 50,
        user_profile: dict[str, float] | None = None,
        seed: int | None = None,
        base_url: str | None = None,  
        **kwargs
    ):
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        """
        Parameters
        ----------
        feed_slots   : Number of posts in the final ranked feed.
        pool_size    : Number of candidate posts per episode.
        max_steps    : Hard step budget (>= feed_slots recommended).
        user_profile : topic → interest weight (0–1). Auto-generated if None.
        seed         : Random seed for reproducibility.
        """
        self._feed_slots  = feed_slots
        self._pool_size   = pool_size
        self._max_steps   = max_steps
        self._user_profile = user_profile
        self._rng = random.Random(seed)
        self._rubric = reward_.FeedRankingRubric()
        self.client = EnvClient(base_url=self.base_url) 
        self._state: FeedRankingState | None = None
        self.reset()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def reset(self) -> FeedRankingObservation:
        """
        Start a new episode.

        Generates a fresh candidate pool and (optionally) a new user profile.

        Returns
        -------
        Initial observation (feed is empty, all candidates available).
        """
        pool = self._generate_pool()

        # User interest vector
        if self._user_profile is not None:
            interests = dict(self._user_profile)
        else:
            interests = self._generate_user_profile()

        self._state = FeedRankingState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            feed=[],
            placed_ids=set(),
            topic_counts={},
            source_counts={},
            user_interest_vector=interests,
            candidate_pool=pool,
        )

        return self._build_obs(reward=0.0, done=False)

    def step(self, action: FeedRankingAction) -> FeedRankingObservation:
        """
        Place a post and return the next observation.

        Parameters
        ----------
        action : FeedRankingAction with a valid post_id.

        Returns
        -------
        FeedRankingObservation with per-step reward and info breakdown.
        """
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        state = self._state

        # ── Wasted-step guard (feed already full) ───────────────
        if len(state.feed) >= self._feed_slots:
            penalty, _ = self._rubric.wasted_step()
            state.cumulative_reward += penalty
            return self._build_obs(reward=penalty, done=True, info={"wasted_step": True})

        # ── Find the post in the pool ────────────────
        post = self._find_post(action.post_id)

        # Invalid post_id → small penalty, episode continues
        if post is None:
            penalty = -0.15
            state.step_count += 1
            state.cumulative_reward += penalty
            done = state.step_count >= self._max_steps
            return self._build_obs(
                reward=penalty,
                done=done,
                info={"invalid_post_id": action.post_id},
            )

        slot_index = len(state.feed)

        # ── Compute reward ───────────────────
        reward, breakdown = self._rubric(post, state, slot_index)

        # ── Update state ──────────────────────
        state.feed.append(post.post_id)
        state.placed_ids.add(post.post_id)
        state.topic_counts[post.topic] = state.topic_counts.get(post.topic, 0) + 1
        state.source_counts[post.source] = state.source_counts.get(post.source, 0) + 1
        # Remove placed post from pool
        state.candidate_pool = [p for p in state.candidate_pool if p.post_id != post.post_id]
        state.step_count += 1
        state.cumulative_reward += reward

        # ── Termination ─────────────────────
        feed_full  = len(state.feed) >= self._feed_slots
        step_limit = state.step_count >= self._max_steps
        no_more    = len(state.candidate_pool) == 0
        done = feed_full or step_limit or no_more

        return self._build_obs(reward=reward, done=done, info={"breakdown": breakdown.as_dict()})

    @property
    def state(self) -> FeedRankingState | None:
        """Return the current (mutable) episode state."""
        return self._state

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_obs(
        self,
        reward: float,
        done: bool,
        info: dict | None = None,
    ) -> FeedRankingObservation:
        s = self._state
        candidate_view = [
            {
                "post_id":      p.post_id,
                "topic":        p.topic,
                "source":       p.source,
                "age_hours":    round(p.age_hours, 1),
                "quality_score": round(p.quality_score, 3),
                "base_ctr":     round(p.base_ctr, 3),
                "is_clickbait": p.is_clickbait,
            }
            for p in s.candidate_pool
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

    def _find_post(self, post_id: str) -> Post | None:
        for p in self._state.candidate_pool:
            if p.post_id == post_id:
                return p
        # Also allow re-selecting an already-placed post (rubric applies penalty)
        for pid in self._state.placed_ids:
            if pid == post_id:
                # Return a dummy post to trigger the repeat-penalty path
                return Post(
                    post_id=post_id, topic="unknown", source="unknown",
                    age_hours=0, quality_score=0, base_ctr=0,
                )
        return None

    def _generate_pool(self) -> list[Post]:
        posts = []
        for i in range(self._pool_size):
            topic  = self._rng.choice(self.TOPICS)
            source = self._rng.choice(self.SOURCES)
            quality = self._rng.betavariate(2, 2)         # peaks around 0.5
            clickbait = self._rng.random() < 0.12         # ~12% clickbait
            if clickbait:
                quality = min(quality, 0.35)              # clickbait is low-quality

            posts.append(Post(
                post_id=f"post_{i:04d}_{uuid.uuid4().hex[:6]}",
                topic=topic,
                source=source,
                age_hours=self._rng.expovariate(1 / 18),  # mean ~18 h
                quality_score=quality,
                base_ctr=self._rng.betavariate(1.5, 6),   # realistic CTR skew
                is_clickbait=clickbait,
            ))
        return posts

    def _generate_user_profile() -> dict[str, float]:
        """Sparse interest vector: 2-4 hot topics, rest near zero."""
        weights = {t: round(self._rng.random() * 0.15, 3) for t in self.TOPICS}
        hot = self._rng.sample(self.TOPICS, k=self._rng.randint(2, 4))
        for t in hot:
            weights[t] = round(0.6 + self._rng.random() * 0.4, 3)
        # Normalise to [0,1] without destroying sparsity
        max_w = max(weights.values())
        return {t: round(w / max_w, 3) for t, w in weights.items()}