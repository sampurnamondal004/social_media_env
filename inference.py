import asyncio
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from social_media_env.client import SocialFeedEnv
from social_media_env import FeedRankingAction, FeedRankingEnvironment

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "registry.hf.space/sampurnamondal012-ocial-media-ranking-env:latest")
ENV_URL = os.getenv("ENV_URL", "https://sampurnamondal012-ocial-media-ranking-env.hf.space")

TASK_NAME = os.getenv("SOCIAL_MEDIA_ENV_TASK", "engagement_optimization")
BENCHMARK = os.getenv("SOCIAL_MEDIA_ENV_BENCHMARK", "social_media_env")

MAX_STEPS = 5
TEMPERATURE = 0.0
MAX_TOKENS = 50
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward per step is 1.0 (reward bounded to [-1, 1])
_MAX_REWARD_PER_STEP = 1.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent("""
    You are a social media feed ranking agent.
    Your goal is to select the best post from the candidate pool to maximize user engagement.
    Consider the user's interest vector and prefer:
    - Posts with high relevance to user interests
    - Posts with high quality_score
    - Fresh posts with low age_hours
    - Avoid posts where is_clickbait=true
    Reply with ONLY the post_id string. No explanation, no quotes.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    candidates = obs.candidate_pool[:10] if hasattr(obs, "candidate_pool") else []
    interests = obs.user_interest_vector if hasattr(obs, "user_interest_vector") else {}
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step: {step}
        User interests (topic -> weight): {json.dumps(interests)}
        Candidate posts:
        {json.dumps(candidates, indent=2)}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Reply with ONLY the post_id of the best post.
    """).strip()


def get_model_message(client: OpenAI, step: int, obs, last_reward: float, history: List[str]) -> str:
    candidates = obs.candidate_pool[:10] if hasattr(obs, "candidate_pool") else []
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().strip('"')
        valid_ids = {p["post_id"] for p in candidates}
        return text if text in valid_ids else (candidates[0]["post_id"] if candidates else "")
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return candidates[0]["post_id"] if candidates else ""


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await SocialFeedEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()  # OpenENV.reset()
        last_obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            post_id = get_model_message(client, step, last_obs, last_reward, history)

            result = await env.step(FeedRankingAction(post_id=post_id))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_obs = obs
            last_reward = reward

            log_step(step=step, action=f"select({post_id})", reward=reward, done=done, error=error)

            history.append(f"Step {step}: {post_id!r} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
