import asyncio
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
import httpx


API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
ENV_URL = os.getenv("ENV_URL") or "https://sampurnamondal012-ocial-media-ranking-env.hf.space"

TASK_NAME = "feed-ranking"
BENCHMARK = "social_media_env"
MAX_STEPS = 5
SUCCESS_SCORE_THRESHOLD = 0.1

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def select_post_with_llm(client: OpenAI, candidate_pool: List[Dict], interests: Dict) -> str:
    candidates = candidate_pool[:10]
    prompt = f"""You are a social media feed ranking agent.
User interests (topic -> weight): {json.dumps(interests)}
Candidate posts:
{json.dumps(candidates, indent=2)}
Reply with ONLY the post_id of the best post to show next. No explanation."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        chosen = response.choices[0].message.content.strip().strip('"')
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        chosen = ""

    valid_ids = {p["post_id"] for p in candidate_pool}
    if chosen not in valid_ids:
        chosen = candidate_pool[0]["post_id"]
    return chosen

async def main() -> None:
    
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    http = httpx.AsyncClient(base_url=ENV_URL, timeout=30.0)
    try:
        r = await http.post("/reset")
        r.raise_for_status()
        obs = r.json()

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break
            candidate_pool = obs.get("candidate_pool", [])
            if not candidate_pool:
                break

            interests = obs.get("user_interest_vector", {})
            post_id = select_post_with_llm(client, candidate_pool, interests)

            r = await http.post("/step", json={"post_id": post_id})
            r.raise_for_status()
            obs = r.json()

            reward = float(obs.get("reward", 0.0))
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=f"select({post_id})",
                     reward=reward, done=obs.get("done", False), error=None)

            if obs.get("done", False):
                break

        score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
       
        try:
            await http.aclose()
        except Exception as e:
            print(f"[DEBUG] http.aclose() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())