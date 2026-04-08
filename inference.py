import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI

# Import client 
from social_media_env import SocialFeedEnv, FeedRankingAction

# Environment Variables (Required by Grader)
API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4"
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # For local validation

# Constants
TASK_NAME = "feed-ranking"
BENCHMARK = "social_media_env"
MAX_STEPS = 5  

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize environment
    # The grader uses from_docker_image to test reproduction
    env = await SocialFeedEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            
            
            selected_post_index = 0 
            
            # Perform action
            result = await env.step(FeedRankingAction(post_index=selected_post_index))
            
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=f"select({selected_post_index})", 
                     reward=reward, done=result.done, error=None)

            if result.done:
                break

        # Calculate final score (must be 0.0 to 1.0)
        final_score = sum(rewards) / MAX_STEPS 
        success = final_score > 0.5

    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())