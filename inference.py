import asyncio
import os
import json
from typing import List, Dict
import httpx

API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
ENV_URL = os.environ.get("ENV_URL", "https://sampurnamondal012-ocial-media-ranking-env.hf.space")

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

def select_post_with_llm(candidate_pool: List[Dict], interests: Dict) -> str:
    """Call LLM via raw httpx to guarantee API_BASE_URL and API_KEY are used."""
    candidates = candidate_pool[:10]
    prompt = f"""You are a social media feed ranking agent.
User interests: {json.dumps(interests)}
Candidate posts: {json.dumps(candidates, indent=2)}
Reply with ONLY the post_id of the best post. No explanation."""

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.0,
    }

    
    base = API_BASE_URL.rstrip("/")
    with httpx.Client(timeout=30.0) as client:
        r = client.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        chosen = r.json()["choices"][0]["message"]["content"].strip().strip('"')

    valid_ids = {p["post_id"] for p in candidate_pool}
    if chosen not in valid_ids:
        chosen = candidate_pool[0]["post_id"]
    return chosen

async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    success = False
    final_score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(base_url=ENV_URL, timeout=30.0) as http:
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

                # ✅ Direct httpx call — no OpenAI client wrapper that might reroute
                post_id = select_post_with_llm(candidate_pool, interests)

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

        final_score = sum(rewards) / MAX_STEPS if MAX_STEPS > 0 else 0.0
        success = final_score > 0.5

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
        raise

    finally:
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())