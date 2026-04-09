import asyncio
import os
from typing import List, Optional
from openai import OpenAI
from openenv.core.env_client import EnvClient
from social_media_env.models import FeedRankingAction

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
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


async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0 

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await env.reset()

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

           
            if not obs.candidate_pool:
                break
            post_id = obs.candidate_pool[0]["post_id"]

            obs = await env.step(FeedRankingAction(post_id=post_id))

            reward = obs.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=f"select({post_id})",
                     reward=reward, done=obs.done, error=None)

            if obs.done:
                break

        final_score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        success = final_score > 0.5

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        raise

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())