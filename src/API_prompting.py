import pandas as pd
import os
import asyncio
import time
import regex as re
import json
import random

from asyncio import Queue
from pathlib import Path
from tqdm.asyncio import tqdm
from litellm import acompletion
from dataclasses import dataclass, asdict


@dataclass
class PromptClass:
    prompt:    str
    model:     str
    inventory: str
    situation: str
    rep:       int
    item:      str
    preamble:  str
    postamble: str
    options:   str
    timestamp: str
    response:  str
    usage:     dict
    reasoning: str

    def asdict(self):
        return asdict(self)


def build_filename(prompt_class) -> str:
    """Build filename with model name."""

    if isinstance(prompt_class, PromptClass):
        model_name = re.sub(r'[/:.]', '-', prompt_class.model.split("/")[-1])
        filename = f"{model_name}"

    elif isinstance(prompt_class, dict):
        model_name = re.sub(r'[/:.]', '-', prompt_class["model"].split("/")[-1])
        filename = f"{model_name}"

    else:
        raise ValueError("prompt_class must be an instance of PromptClass or a dict with a 'model' key.")

    return filename.lower()


def cache_directory(prompt_class: PromptClass):
    """Check if cache directory exists and if not, create it."""

    cache_dir = Path("./cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_file = cache_dir / "cache.csv"
    if cache_file.exists():
        cache = pd.read_csv(cache_file, low_memory=False)
    else:
        cache = pd.DataFrame(columns = ['prompt', 'model', 'inventory', 'situation', 'rep', 'item', 'preamble', 'postamble', 'options', 'timestamp', 'response', 'usage', 'reasoning'])

    return cache


def save_cache(response_queue: Queue, prompt_class: PromptClass, path: str, final: bool = False):
    """Save cache to csv."""

    cache_file = Path("./cache/cache.csv")

    # Check if cache already exists, if not: create cache
    if cache_file.exists():
        cache = pd.read_csv(cache_file, low_memory=False)
    else:
        cache = pd.DataFrame(columns=['prompt', 'model', 'inventory', 'situation', 'rep', 'item', 'preamble', 'postamble', 'options', 'timestamp', 'response', 'usage', 'reasoning'])

    # Get new responses and add to cache
    new_rows = []
    while not response_queue.empty():
        response = response_queue.get_nowait()
        new_rows.append(response.asdict())

    cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)

    # Drop duplicates, sort, and save cache
    cache = (cache
             .drop_duplicates(subset=["prompt", "model", "inventory", "situation", "rep", "item", "preamble", "postamble", "options", "timestamp"], keep="last")
             .sort_values(by=["model", "inventory", "situation", "prompt", "rep"])
             .reset_index(drop=True))

    cache.to_csv("./cache/cache.csv", index=False)

    # Save final file with given filename in the corresponding folder
    if final and not cache.empty:
        cache_model = cache[cache["model"] == prompt_class.model]
        if not cache_model.empty:
            filename = build_filename(prompt_class)
            os.makedirs(f"./{path}", exist_ok=True)
            cache_model.to_csv(f"./{path}/{filename}.csv", index=False)


def prompt_generator(preamble, item, postamble, inventory) -> str:
    """Generate prompt from preamble, item, and postamble."""
    if inventory == "lmlpa":
        prompt = f"{preamble} 'I {item}'. {postamble}"
    elif inventory == "bfi-llm":
        prompt = f"{preamble} 'I see myself as a chatbot who {item}'. {postamble}"
    elif inventory == "social-desirability":
        prompt = f"{preamble} 'I {item}'. {postamble}"
    else:
        prompt = f"{preamble} '{item}'. {postamble}"
    return prompt


def prompt_generator_options(preamble, item, options_a, options_b, postamble) -> str:
    """Generate prompt from preamble, item, and postamble."""
    prompt_a = f"{preamble} {options_a} {postamble} 'I see myself as a chatbot who {item}'."
    prompt_b = f"{preamble} {options_b} {postamble} 'I see myself as a chatbot who {item}'."
    return prompt_a, prompt_b


async def create_queue(prompts: list):
    """Create prompt queue and response queue."""

    # Create prompt queue and fill with prompts
    prompt_queue = Queue()
    for prompt in prompts:
        await prompt_queue.put(prompt)

    print(f"Prompt queue size: {prompt_queue.qsize()}")

    # Create response queue (empty)
    response_queue = Queue()

    return prompt_queue, response_queue


async def reponse_generator(prompt, model):
    """Generate response for a given prompt and model."""

    # Some anthropic models need a special API call
    thinking_models = [
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
        "claude-opus-4-6",
        "claude-opus-4-20250514"
    ]

    # Normal API call
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "messages": messages,
    }

    # Special API call: additionally put "thinking"
    if model in thinking_models:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 1024}

    # Get response (async)
    response = await acompletion(**kwargs)

    # Extract response
    output_text = response.choices[0].message.content

    # Extract tokens and costs
    usage = {
        "total_tokens": response.usage.total_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "reasoning_tokens": (
            response.usage.completion_tokens_details.reasoning_tokens
            if response.usage.completion_tokens_details is not None
            else 0
        ),
        "response_costs": response._hidden_params["response_cost"],
    }

    # Extract reasoning (if applicable)
    reasoning = getattr(response.choices[0].message, "reasoning_content", None) or ""

    return output_text, usage, reasoning


def calculate_num_workers(tpm_limit: int, rpm_limit: int, total: int, n_reps: int, situation: str):
    """Calculate optimal number of workers based on (average) rate limits."""

    # Conservative estimates of average number of tokens
    avg_tokens_per_request = 500

    # Calculate number of max workers depending on rate limits: tpm & rpm
    max_workers_by_tpm = max(1, tpm_limit // avg_tokens_per_request)
    max_workers_by_rpm = max(1, rpm_limit // 10)  # Conservative estimate: each worker handles ~10 requests/min
    num_workers = min(max_workers_by_tpm, max_workers_by_rpm, total)

    return num_workers


async def worker(name: str, prompt_queue: Queue, response_queue: Queue, cache: pd.DataFrame, pbar: tqdm,
                 rate_dict: dict, total_requests: int, path: str, tpm_limit: int = None, rpm_limit: int = None):
    """Generating workers prompting to model in parallel."""

    while True:
        try:
            prompt_class = await prompt_queue.get()
        except asyncio.CancelledError:
            break

        # Get general info
        prompt = prompt_class.prompt
        model = prompt_class.model
        inventory = prompt_class.inventory
        situation = prompt_class.situation
        rep = prompt_class.rep

        # Check if prompt is already in cache
        if ((cache["prompt"] == prompt) & (cache["rep"] == rep) &
            (cache["inventory"] == inventory) & (cache["situation"] == situation)).any():
            output_text = cache.loc[
                (cache["prompt"] == prompt) & (cache["rep"] == rep) &
                (cache["inventory"] == inventory) & (cache["situation"] == situation),
                "response"
            ].values[0]
            prompt_queue.task_done()
            continue

        # Check rate limits
        current_time = time.time()
        time_passed = (current_time - rate_dict["start_time"]) / 60

        # Token limits
        if tpm_limit is not None and rate_dict["tokens_used"] >= tpm_limit:
            sleep_time = 60 - (current_time - rate_dict["start_time"])
            if sleep_time > 0:
                print(f"\nTPM limit ({tpm_limit}) reached. Sleeping for {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                rate_dict["tokens_used"] = 0
                rate_dict["start_time"] = time.time()

        # Rate limits
        if rpm_limit is not None and rate_dict["requests_made"] >= rpm_limit and time_passed < 1:
            sleep_time = 60 - (current_time - rate_dict["start_time"])
            if sleep_time > 0:
                print(f"\nRPM limit ({rpm_limit}) reached. Sleeping for {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                rate_dict["requests_made"] = 0
                rate_dict["start_time"] = time.time()

        # Reset counts if one minute passed
        if time_passed >= 1:
            rate_dict["tokens_used"] = 0
            rate_dict["requests_made"] = 0
            rate_dict["start_time"] = time.time()

        # Get response from model
        try:
            output_text, usage, reasoning = await reponse_generator(prompt, model=model)
            # Sanity check: randomly print out some responses
            if random.randint(1, 200) == 2:
                print("\nOutput:", output_text)
                print("Reasoning:", reasoning)
                print("Usage:", usage)
        except Exception as e:
            print(f"\nWorker {name} failed on prompt '{prompt[:50]}...': {e}")
            prompt_queue.task_done()
            pbar.update(1)
            continue

        # Update counts for rate limits
        tokens_used = usage["total_tokens"]
        rate_dict["tokens_used"] += tokens_used
        rate_dict["requests_made"] += 1
        rate_dict["average_tokens_used"] = rate_dict["tokens_used"] / rate_dict["requests_made"]
        rate_dict["response_costs"] += usage["response_costs"] or 0
        rate_dict["prompts_completed"] += 1

        # Save output in response queue
        prompt_class.response = output_text
        prompt_class.usage = usage
        prompt_class.reasoning = reasoning
        await response_queue.put(prompt_class)
        prompt_queue.task_done()

        # Update progress bar
        pbar.update(1)

        # Cache every 100 completed prompts
        if random.randint(1, 20) == 2:
            save_cache(response_queue, prompt_class, path)


async def parallel_prompting(prompt_queue: Queue, response_queue: Queue, cache: pd.DataFrame, total: int, tpm_limit: int, rpm_limit: int, n_reps: int, path: str):
    """Run parallel prompting with rate limiting."""

    # Get general info
    if not prompt_queue.empty():
        prompt_class = prompt_queue._queue[0]
        model = prompt_class.model
        inventory = prompt_class.inventory
        situation = prompt_class.situation
    else:
        raise Exception("Prompt queue empty.")

    # Initiate rate dictionary
    rate_dict = {
        "tokens_used": 0,
        "average_tokens_used": 0,
        "requests_made": 0,
        "response_costs": 0,
        "start_time": time.time(),
        "prompts_completed": 0
    }

    # Check how many prompts were already processed and cached
    remaining = sum(
        not ((cache["prompt"] == pc.prompt) &
             (cache["rep"] == pc.rep) &
             (cache["inventory"] == pc.inventory) &
             (cache["situation"] == pc.situation)).any()
        for pc in list(prompt_queue._queue)
    )

    # Print general info
    print(f"\n\nModel: {model}")
    print(f"Num_reps: {n_reps}")
    print(f"Total requests: {prompt_queue.qsize()} ({total})")
    print(f"Remaining requests (not in cache): {remaining}")

    # Skip if all prompts have been processed
    if remaining == 0:
        print("All prompts already cached. Skipping.")
    else:
        # Calculate optimal number of workers
        num_workers = calculate_num_workers(tpm_limit, rpm_limit, remaining, n_reps, situation)
        print(f"Using {num_workers} workers (TPM limit: {tpm_limit}, RPM limit: {rpm_limit})")

        # Spawn workers and process prompts
        with tqdm(total=remaining, desc="Processing prompts", unit="prompt") as pbar:
            workers = [
                asyncio.create_task(worker(f"{i}", prompt_queue, response_queue, cache, pbar, rate_dict, total, path, tpm_limit, rpm_limit))
                for i in range(num_workers)
            ]

            await prompt_queue.join()

            for w in workers:
                w.cancel()

    # Save cache
    save_cache(response_queue, prompt_class, path)

    return response_queue, rate_dict["response_costs"]
