import asyncio
import os
import json
from typing import List
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


def safe_get(result, key, default):
    """Works whether result is a dict or an object with attributes."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Handle both sync and async reset
        reset_result = env.reset()
        if asyncio.iscoroutine(reset_result):
            reset_result = await reset_result
        obs = safe_get(reset_result, "observation", reset_result)

        for step in range(1, MAX_STEPS + 1):
            error = None
            reward = 0.0
            done = False

            try:
                prompt = f"""You are a social media feed ranking agent.
Current observation: {obs}

Respond with a JSON object with a 'rankings' key containing a list of item IDs in priority order.
Example: {{"rankings": ["item_3", "item_1", "item_2"]}}
Respond with JSON only, no markdown."""

                response = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
                raw_action = response.choices[0].message.content.strip()

                # Strip markdown fences if model wraps in ```json ... ```
                if raw_action.startswith("```"):
                    raw_action = raw_action.split("```")[1]
                    if raw_action.startswith("json"):
                        raw_action = raw_action[4:]
                raw_action = raw_action.strip()

                parsed = json.loads(raw_action)
                action = FeedRankingAction(**parsed)

                # Handle both sync and async step
                step_result = env.step(action)
                if asyncio.iscoroutine(step_result):
                    step_result = await step_result

                obs = safe_get(step_result, "observation", obs)
                reward = float(safe_get(step_result, "reward", 0.0))
                done = bool(safe_get(step_result, "done", False))
                final_score = reward

            except json.JSONDecodeError as e:
                error = f"JSON parse error: {e}"
                done = True
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                done = True

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action="feed_ranking", reward=reward, done=done, error=error)

            if done:
                success = (error is None)
                break

        if steps_taken == MAX_STEPS:
            success = True

    except Exception as e:
        print(f"Fatal error in main: {type(e).__name__}: {e}", flush=True)
        if steps_taken == 0:
            rewards = [0.0]

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards if rewards else [0.0],
        )


if __name__ == "__main__":
    asyncio.run(main())