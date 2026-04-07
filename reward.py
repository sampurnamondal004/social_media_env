"""reward.py"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RewardBreakdown:
    relevance: float = 0.0
    freshness: float = 0.0
    diversity: float = 0.0
    quality: float = 0.0
    position: float = 0.0
    total: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__
    
class FeedRankingRubric:
    def __call__(self, post, state, slot_index) -> Tuple[float, RewardBreakdown]:
        # Your scoring logic goes here
        # Example:
        relevance = state.user_interest_vector.get(post.topic, 0.0)
        total = relevance # Simplification for now
        breakdown = RewardBreakdown(relevance=relevance, total=total)
        return total, breakdown

    def wasted_step(self) -> Tuple[float, dict]:
        return -0.5, {"error": "feed_full"}

try:
    from openenv.core.rubrics.trajectory import ExponentialDiscountingTrajectoryRubric
except ModuleNotFoundError:
    class ExponentialDiscountingTrajectoryRubric:
        """Compatibility fallback when the installed core lacks rubrics."""

        def __init__(self, gamma: float = 0.99, intermediate_reward: float = 0.0):
            self.gamma = gamma
            self.intermediate_reward = intermediate_reward
            self._trajectory: List[Tuple[Any, Any]] = []

        def __call__(self, action: Any, observation: Any) -> float:
            self._trajectory.append((action, observation))
            if getattr(observation, "done", False):
                return self.score_trajectory(self._trajectory)
            return self.intermediate_reward

        def reset(self) -> None:
            self._trajectory = []

        def compute_step_rewards(self) -> List[float]:
            if not self._trajectory:
                return []
            final_score = self.score_trajectory(self._trajectory)
            total_steps = len(self._trajectory)
            return [
                self.gamma ** (total_steps - 1 - step_index) * final_score
                for step_index in range(total_steps)
            ]


# ---------------------------------------------------------------------------
# Rubric 1 — Dense pass-through
# ---------------------------------------------------------------------------

class FeedRankingDenseRubric(ExponentialDiscountingTrajectoryRubric):
    """
    Pass-through rubric that preserves the per-step reward signal.

    The environment's FeedRankingRubric already emits a dense, bounded reward
    every step.  This rubric accumulates the trajectory and, at episode end,
    returns the mean per-step reward as the trajectory score.  Temporal
    discounting is then applied by ``compute_step_rewards()`` for training.

    Terminal score
    --------------
    mean(r_t for t in 0..T-1)   — average per-step reward over the episode.
    Range: roughly [-1, +1] since each r_t is in [-1, +1].

    Per-step reward (training)
    --------------------------
    r_t_discounted = gamma^(T-1-t) * terminal_score

    Parameters
    ----------
    gamma : float
        Exponential discount factor (0 < gamma ≤ 1).  Closer to 1 gives
        more uniform credit; lower values concentrate credit on later steps.
    intermediate_reward : float
        Value returned by __call__ before the episode ends.  Default 0.0
        (the dense signal is baked into obs.reward, not the rubric return).

    Usage
    -----
        rubric = FeedRankingDenseRubric(gamma=0.99)
        rubric.reset()
        for action, obs in episode:
            rubric(action, obs)
        step_rewards = rubric.compute_step_rewards()
    """

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """
        Score as the mean per-step reward over the completed episode.

        Parameters
        ----------
        trajectory : list of (action, observation) pairs.

        Returns
        -------
        float — mean reward in roughly [-1, +1].  Returns 0.0 for an
        empty trajectory.
        """
        if not trajectory:
            return 0.0

        rewards = [
            getattr(obs, "reward", 0.0)
            for _, obs in trajectory
        ]
        return sum(rewards) / len(rewards)


# ---------------------------------------------------------------------------
# Rubric 2 — Holistic feed quality scorer
# ---------------------------------------------------------------------------

class FeedRankingQualityRubric(ExponentialDiscountingTrajectoryRubric):
    """
    Holistic rubric that re-scores the completed feed as a whole.

    Rather than averaging per-step rewards, this rubric evaluates five
    feed-level properties from the terminal observation and combines them
    into a single score in [-1, +1].  It is intentionally independent of
    the step-level FeedRankingRubric so the two can be used together as
    a reward + shaped bonus, or this rubric can be used alone for offline
    evaluation.

    Feed-level dimensions scored
    ----------------------------
    relevance_coverage (weight 0.30)
        Fraction of placed posts whose topic has user interest > threshold,
        mapped to [-1, +1].  Rewards feeds that consistently match the
        user's declared interests.

    diversity (weight 0.25)
        Normalised entropy of the topic distribution in the feed, mapped
        to [-1, +1].  Rewards variety; penalises filter-bubble feeds.

    quality (weight 0.25)
        Mean quality_score of placed posts, mapped to [-1, +1].

    freshness (weight 0.10)
        Mean freshness of placed posts (exponential decay, 24 h half-life),
        mapped to [-1, +1].

    top_slot_integrity (weight 0.10)
        Whether the top-3 slots are free of clickbait.  +1 if none of the
        first three posts is clickbait, -1 if all three are, linear between.

    Terminal score
    --------------
    Weighted sum of the five dimensions above, clamped to [-1, +1].
    Returns 0.0 when the feed is empty.

    Per-step reward (training)
    --------------------------
    gamma^(T-1-t) * terminal_score   (via compute_step_rewards)

    Parameters
    ----------
    gamma : float
        Exponential discount factor.
    interest_threshold : float
        Minimum user interest weight for a topic to count as "relevant".
        Default 0.5.
    freshness_halflife_hours : float
        Half-life for the freshness decay.  Must match the environment's
        setting for consistent scoring.  Default 24.0.
    top_slots : int
        Number of leading feed slots checked for top-slot integrity.
        Default 3.
    """

    # Sub-score weights — must sum to 1.0
    W_RELEVANCE_COVERAGE = 0.30
    W_DIVERSITY          = 0.25
    W_QUALITY            = 0.25
    W_FRESHNESS          = 0.10
    W_TOP_SLOT_INTEGRITY = 0.10

    def __init__(
        self,
        gamma: float = 0.99,
        intermediate_reward: float = 0.0,
        interest_threshold: float = 0.5,
        freshness_halflife_hours: float = 24.0,
        top_slots: int = 3,
    ) -> None:
        super().__init__(gamma=gamma, intermediate_reward=intermediate_reward)
        self.interest_threshold      = interest_threshold
        self.freshness_halflife_hours = freshness_halflife_hours
        self.top_slots               = top_slots

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score_trajectory(self, trajectory: List[Tuple[Any, Any]]) -> float:
        """
        Holistically score the completed feed from the terminal observation.

        Parameters
        ----------
        trajectory : list of (action, observation) pairs.

        Returns
        -------
        float in [-1, +1].  Returns 0.0 for an empty trajectory or an
        empty feed.
        """
        if not trajectory:
            return 0.0

        _, final_obs = trajectory[-1]

        # Pull required fields from the terminal observation
        feed: List[str]             = getattr(final_obs, "feed",                 [])
        pool: List[Dict]            = getattr(final_obs, "candidate_pool",       [])
        interests: Dict[str, float] = getattr(final_obs, "user_interest_vector", {})

        if not feed:
            return 0.0

        # Reconstruct placed post metadata from the episode trajectory.
        # The candidate_pool in the terminal obs only contains *remaining*
        # posts, so we walk back through all observations to collect every
        # post that was ever in the pool, keyed by post_id.
        post_meta: Dict[str, Dict] = {}
        for _, obs in trajectory:
            for p in getattr(obs, "candidate_pool", []):
                post_meta.setdefault(p["post_id"], p)
        # Also include anything still in the terminal pool
        for p in pool:
            post_meta.setdefault(p["post_id"], p)

        placed: List[Dict] = [post_meta[pid] for pid in feed if pid in post_meta]

        if not placed:
            return 0.0

        score = (
            self.W_RELEVANCE_COVERAGE * self._relevance_coverage(placed, interests)
            + self.W_DIVERSITY          * self._diversity(placed)
            + self.W_QUALITY            * self._quality(placed)
            + self.W_FRESHNESS          * self._freshness(placed)
            + self.W_TOP_SLOT_INTEGRITY * self._top_slot_integrity(placed)
        )
        return float(max(-1.0, min(1.0, score)))

    # ------------------------------------------------------------------
    # Sub-scorers (each returns a value in [-1, +1])
    # ------------------------------------------------------------------

    def _relevance_coverage(
        self,
        placed: List[Dict],
        interests: Dict[str, float],
    ) -> float:
        """Fraction of placed posts that match a high-interest topic."""
        if not placed:
            return 0.0
        relevant = sum(
            1 for p in placed
            if interests.get(p.get("topic", ""), 0.0) >= self.interest_threshold
        )
        fraction = relevant / len(placed)   # [0, 1]
        return 2.0 * fraction - 1.0         # [-1, +1]

    def _diversity(self, placed: List[Dict]) -> float:
        """
        Normalised Shannon entropy of the topic distribution.

        entropy = -Σ p_i * log(p_i)
        max_entropy = log(n_topics)   (uniform distribution)
        normalised = entropy / max_entropy   → [0, 1] → [-1, +1]
        """
        from collections import Counter

        if not placed:
            return 0.0

        counts = Counter(p.get("topic", "unknown") for p in placed)
        n = len(placed)
        entropy = -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)
        n_topics = len(counts)
        if n_topics <= 1:
            return -1.0                     # perfectly homogeneous
        max_entropy = math.log(n_topics)
        normalised = entropy / max_entropy  # [0, 1]
        return 2.0 * normalised - 1.0       # [-1, +1]

    def _quality(self, placed: List[Dict]) -> float:
        """Mean quality_score of placed posts, mapped to [-1, +1]."""
        if not placed:
            return 0.0
        mean_q = sum(p.get("quality_score", 0.5) for p in placed) / len(placed)
        return 2.0 * mean_q - 1.0

    def _freshness(self, placed: List[Dict]) -> float:
        """Mean freshness of placed posts (exponential decay), mapped to [-1, +1]."""
        if not placed:
            return 0.0
        decay = math.log(2) / self.freshness_halflife_hours
        mean_fresh = sum(
            math.exp(-decay * p.get("age_hours", 0.0)) for p in placed
        ) / len(placed)
        return 2.0 * mean_fresh - 1.0

    def _top_slot_integrity(self, placed: List[Dict]) -> float:
        """
        Penalise clickbait in the leading feed slots.

        +1.0  — no clickbait in top-k slots
         0.0  — half of top-k slots are clickbait
        -1.0  — all top-k slots are clickbait
        """
        top = placed[: self.top_slots]
        if not top:
            return 0.0
        clickbait_count = sum(1 for p in top if p.get("is_clickbait", False))
        fraction_bad = clickbait_count / len(top)   # [0, 1]
        return 1.0 - 2.0 * fraction_bad             # [+1, -1]


# ---------------------------------------------------------------------------
# Quick smoke-test (run with: python feed_ranking_rubrics.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dataclasses import dataclass, field

    @dataclass
    class _Obs:
        done: bool = False
        reward: float = 0.0
        feed: list = field(default_factory=list)
        candidate_pool: list = field(default_factory=list)
        user_interest_vector: dict = field(default_factory=dict)

    def _make_post(pid, topic, quality, age, clickbait=False):
        return {
            "post_id": pid, "topic": topic, "source": "pub_0",
            "age_hours": age, "quality_score": quality,
            "base_ctr": 0.05, "is_clickbait": clickbait,
        }

    interests = {"tech": 1.0, "science": 0.8, "sports": 0.2}

    posts = [
        _make_post("p0", "tech",    0.9,  2.0),
        _make_post("p1", "science", 0.8,  5.0),
        _make_post("p2", "tech",    0.7, 10.0),
        _make_post("p3", "sports",  0.5, 30.0),
        _make_post("p4", "food",    0.3, 48.0, clickbait=True),
    ]
    feed_ids = [p["post_id"] for p in posts]

    def _make_traj(reward_per_step=0.3):
        traj = []
        for i, pid in enumerate(feed_ids):
            obs = _Obs(
                done=(i == len(feed_ids) - 1),
                reward=reward_per_step,
                feed=feed_ids[: i + 1],
                candidate_pool=posts[i + 1 :],
                user_interest_vector=interests,
            )
            traj.append((None, obs))
        return traj

    traj = _make_traj()

    # ── Dense rubric ────────────────────────────────────────────────────────
    dense = FeedRankingDenseRubric(gamma=0.99)
    dense.reset()
    for action, obs in traj:
        dense(action, obs)
    score_d = dense.score_trajectory(dense._trajectory)
    steps_d = dense.compute_step_rewards()
    print(f"FeedRankingDenseRubric")
    print(f"  trajectory score : {score_d:+.4f}")
    print(f"  step rewards     : {[round(r,4) for r in steps_d]}")

    # ── Quality rubric ──────────────────────────────────────────────────────
    quality_rubric = FeedRankingQualityRubric(gamma=0.99)
    quality_rubric.reset()
    for action, obs in traj:
        quality_rubric(action, obs)
    score_q = quality_rubric.score_trajectory(quality_rubric._trajectory)
    steps_q = quality_rubric.compute_step_rewards()
    print(f"\nFeedRankingQualityRubric")
    print(f"  trajectory score : {score_q:+.4f}")
    print(f"  step rewards     : {[round(r,4) for r in steps_q]}")

    # ── Sub-scorer sanity checks ────────────────────────────────────────────
    placed = [_make_post(p["post_id"], p["topic"], p["quality_score"], p["age_hours"], p["is_clickbait"]) for p in posts]
    assert -1.0 <= quality_rubric._relevance_coverage(placed, interests) <= 1.0
    assert -1.0 <= quality_rubric._diversity(placed) <= 1.0
    assert -1.0 <= quality_rubric._quality(placed) <= 1.0
    assert -1.0 <= quality_rubric._freshness(placed) <= 1.0
    assert -1.0 <= quality_rubric._top_slot_integrity(placed) <= 1.0

    # Homogeneous feed should score diversity = -1
    same_topic = [_make_post(f"x{i}", "tech", 0.8, 1.0) for i in range(5)]
    assert quality_rubric._diversity(same_topic) == -1.0, "Uniform topic should give minimum diversity"

    # All-clickbait top slots
    clickbait_feed = [_make_post(f"c{i}", "tech", 0.2, 1.0, clickbait=True) for i in range(3)]
    assert quality_rubric._top_slot_integrity(clickbait_feed) == -1.0

    # Empty trajectory
    assert dense.score_trajectory([]) == 0.0
    assert quality_rubric.score_trajectory([]) == 0.0

    print("\nAll assertions passed.")