# IMPORTS
import torch
import os
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.API_prompting import PromptClass, build_filename
from huggingface_hub import HfApi, scan_cache_info


# HELPERS
class HFModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )


    def generate(self, prompt, max_new_tokens=1024):
        """Generate response from prompt."""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Undesirable outputs
        task = "### TASK"
        instruction = "### INSTRUCTIONS"
        output = "### OUTPUT"
        json_constraint = """
        You are a strict JSON generator.
        Output ONLY valid JSON.

        The JSON must contain exactly:
        {
          "score": integer (1-5),
          "response": string
        }

        Return exactly one JSON object.
        """

        # Clean output from prompt
        response = decoded.replace(prompt, "").strip()
        response = response.replace(json_constraint, "").strip()
        response = response.replace(task, "").strip()
        response = response.replace(instruction, "").strip()
        response = response.replace(output, "").strip()

        return response


def save_cache_local(response_queue, prompt_class: PromptClass, path: str, final: bool = False):
    """Save cache to csv."""

    cache_file = Path("cache/cache.csv")

    # Load existing cache
    if cache_file.exists():
        cache = pd.read_csv(cache_file, low_memory=False)
    else:
        cache = pd.DataFrame(columns=[
            'prompt', 'model', 'inventory', 'situation', 'rep',
            'item', 'preamble', 'postamble', 'options',
            'timestamp', 'response', 'usage', 'reasoning'
        ])

    new_rows = []

    if isinstance(response_queue, list):
        for response in response_queue:
            new_rows.append(response.asdict())
    else:
        while not response_queue.empty():
            response = response_queue.get_nowait()
            new_rows.append(response.asdict())

    # Append new data
    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)

    # Clean cache
    cache = (
        cache
        .drop_duplicates(
            subset=["prompt", "model", "inventory", "situation", "rep",
                    "item", "preamble", "postamble", "options", "timestamp"],
            keep="last"
        )
        .sort_values(by=["model", "inventory", "situation", "prompt", "rep"])
        .reset_index(drop=True)
    )

    cache.to_csv("./cache/cache.csv", index=False)

    # Save final model file
    if final and not cache.empty:
        cache_model = cache[cache["model"] == prompt_class.model]

        if not cache_model.empty:
            filename = build_filename(prompt_class)
            os.makedirs(f"./responses", exist_ok=True)
            cache_model.to_csv(f"./responses/{filename}.csv", index=False)


def is_cached(pc, cache):
    if cache.empty:
        return False

    return (
            (cache["prompt"] == pc.prompt) &
            (cache["model"] == pc.model) &
            (cache["inventory"] == pc.inventory) &
            (cache["situation"] == pc.situation) &
            (cache["rep"] == pc.rep)
    ).any()


def estimate_model_size_billion(model_name):
    api = HfApi()
    info = api.model_info(model_name, files_metadata=True)
    total_bytes = sum(
        f.size for f in info.siblings
        if f.size and (f.rfilename.endswith(".safetensors") or f.rfilename.endswith(".bin"))
    )
    return total_bytes / 2 / 1e9


def select_quantization(param_billion):
    if param_billion <= 50:
        return None
    elif param_billion <= 83:
        return BitsAndBytesConfig(load_in_8bit=True)
    elif param_billion <= 130:
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    elif param_billion <= 160:
        return BitsAndBytesConfig(load_in_4bit=True)
    else:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=True)


def delete_model_cache(model_name):
    cache_info = scan_cache_info()
    for repo in cache_info.repos:
        if repo.repo_id == model_name:
            revisions = {rev.commit_hash for rev in repo.revisions}
            delete_strategy = cache_info.delete_revisions(*revisions)
            delete_strategy.execute()