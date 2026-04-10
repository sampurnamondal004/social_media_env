import asyncio
import os
import json
from openai import OpenAI
from typing import List
from social_media_env.client import SocialFeedEnv
from social_media_env.models import FeedRankingAction

API_KEY = os.environ["API_KEY"] 
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4"
ENV_URL = os.getenv("ENV_URL") or "https://sampurnamondal012-ocial-media-ranking-env.hf.space"

TASK_NAME = "feed-ranking"
BENCHMARK = "social_media_env"
MAX_STEPS = 5

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def select_post_with_llm(client: OpenAI, obs) -> str:
    """Use LLM to select the best post from candidate pool."""
    candidates = obs.candidate_pool[:10]  
    interests = obs.user_interest_vector

    prompt = f"""You are a social media feed ranking agent.

User interests (topic -> weight): {json.dumps(interests)}

Candidate posts:
{json.dumps(candidates, indent=2)}

Select the single best post_id to show the user next based on:
- High relevance to user interests
- High quality_score
- Low age_hours (fresher is better)
- Avoid is_clickbait=true posts

Reply with ONLY the post_id string, nothing else."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.0,
    )

    chosen = response.choices[0].message.content.strip().strip('"')

    
    valid_ids = {p["post_id"] for p in obs.candidate_pool}
    if chosen not in valid_ids:
        chosen = obs.candidate_pool[0]["post_id"]

    return chosen

async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0
    
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with SocialFeedEnv(base_url=ENV_URL) as env:
            result = await env.reset()
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break
                if not obs.candidate_pool:
                    break

                post_id = select_post_with_llm(llm, obs)
                result = await env.step(FeedRankingAction(post_id=post_id))
                obs = result.observation

                reward = result.reward or 0.0
                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=f"select({post_id})",
                         reward=reward, done=result.done, error=None)

                if result.done:
                    break

        final_score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        success = final_score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        raise

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())