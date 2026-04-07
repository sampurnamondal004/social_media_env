from __future__ import annotations

#import random
#import uuid
#from dataclasses import dataclass, field
from typing import Any
from openenv.core.env_server import Action, Observation, State
#import pandas as pd
#import pyarrow as pa
from pydantic import Field
class FeedRankingAction(Action):
    """
    Action for the Feed Ranking environment.

    The agent selects one post from the candidate pool to place in the
    next available feed slot. A single step = a single placement decision.

    Attributes
    ----------
    post_id : str
        Unique identifier of the post to place (e.g. ``"post_0042_3f9a1b"``).
        Must match a ``post_id`` present in the current candidate pool;
        invalid IDs incur a small penalty without terminating the episode.
    """

    post_id: str


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class FeedRankingObservation(Observation):
    """
    Observation for the Feed Ranking environment.

    Returned by ``reset()`` and ``step()``. Contains everything the agent
    needs to make its next decision, plus reward bookkeeping fields that
    mirror the chess observation pattern.

    Attributes
    ----------
    feed : List[str]
        Ordered list of ``post_id`` values placed so far (index 0 = top slot).
        Empty at episode start; grows by one per valid step.

    candidate_pool : List[Dict]
        Remaining posts the agent may still place. Each entry is a plain dict
        with the keys below — no internal state leaks through:

        .. code-block:: text

            post_id        str    unique post identifier
            topic          str    content category (e.g. "tech", "sports")
            source         str    publisher / account id
            age_hours      float  how old the post is
            quality_score  float  editorial credibility, 0–1
            base_ctr       float  simulated baseline click-through rate, 0–1
            is_clickbait   bool   True if the post is low-quality bait

    user_interest_vector : Dict[str, float]
        Sparse map of topic → interest weight (0–1).  Higher weight means
        the simulated user prefers this topic.  Stays constant within an
        episode; regenerated on ``reset()``.

    step : int
        Number of steps taken so far in this episode (0-indexed).

    max_steps : int
        Hard step budget for this episode.

    reward : float
        Per-step reward for the most recent action.  Bounded in [-1, +1]
        under normal conditions; penalty constants may push it slightly
        below -1 when multiple soft penalties stack.

    cumulative_reward : float
        Sum of all per-step rewards since the last ``reset()``.

    done : bool
        True when the episode has terminated.  Termination conditions:
        feed is full, step budget exhausted, or candidate pool is empty.

    result : Optional[str]
        Human-readable summary of the episode outcome, set only when
        ``done=True``.  Format: ``"feed_complete"`` | ``"step_limit"``
        | ``"pool_exhausted"``.

    info : Dict
        Diagnostic payload.  Keys present after a valid placement step:

        .. code-block:: text

            breakdown  Dict[str, float]  per-component reward decomposition
                Keys: relevance, freshness, diversity, quality, position,
                      rabbit_hole_penalty*, clickbait_penalty*, total
                (* only present when the penalty fired)

        Keys present after an invalid step:

        .. code-block:: text

            invalid_post_id  str   the unrecognised post_id that was submitted
            wasted_step      bool  True if the feed was already full
    """

    feed:                  List[str]         = Field(default_factory=list)
    candidate_pool:        List[Dict]         = Field(default_factory=list)
    user_interest_vector:  Dict[str, float]   = Field(default_factory=dict)
    step:                  int                = 0
    max_steps:             int                = 20
    reward:                float              = 0.0
    cumulative_reward:     float              = 0.0
    done:                  bool               = False
    result:                Optional[str]      = None
    info:                  Dict               = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class FeedRankingState(State):
    """
    Full internal episode state for the Feed Ranking environment.

    Not exposed to the agent directly — the environment builds an
    ``FeedRankingObservation`` (a sanitised, read-only view) from this object
    on every step.  

    Attributes
    ----------
    episode_id : str
        UUID generated at ``reset()``.  Unique across episodes.

    step_count : int
        Total steps taken this episode (includes invalid / wasted steps).

    feed : List[str]
        Ordered ``post_id`` values placed so far. Length ≤ ``feed_slots``.

    placed_ids : List[str]
        Set-semantics membership index (stored as a list for Pydantic
        serialisation).  Used by the rubric to detect repeat placements.

    topic_counts : Dict[str, int]
        topic → number of times a post with that topic has been placed.
        Drives the diversity sub-reward and rabbit-hole penalty.

    source_counts : Dict[str, int]
        source → number of times a post from that source has been placed.
        Drives the source-diversity component of the diversity sub-reward.

    user_interest_vector : Dict[str, float]
        topic → interest weight (0–1). Fixed for the duration of the episode.

    candidate_pool : List[Dict]
        Full pool of candidate posts as plain dicts.  Posts are removed from
        this list as they are placed.  Using dicts (rather than a nested
        Pydantic model) keeps serialisation lightweight.

    cumulative_reward : float
        Running sum of per-step rewards.  Mirrored into every observation.
    """

    episode_id:           str               = ""
    step_count:           int               = 0
    feed:                 List[str]         = Field(default_factory=list)
    placed_ids:           List[str]         = Field(default_factory=list)
    topic_counts:         Dict[str, int]    = Field(default_factory=dict)
    source_counts:        Dict[str, int]    = Field(default_factory=dict)
    user_interest_vector: Dict[str, float]  = Field(default_factory=dict)
    candidate_pool:       List[Dict]        = Field(default_factory=list)
    cumulative_reward:    float             = 0.0

if __name__ == "__main__":
    print("🚀 Environment loaded successfully!")
    # Let's try to create an instance of your action
    test_action = FeedRankingAction(post_id="test_123")
    print(f"Action created: {test_action.post_id}")