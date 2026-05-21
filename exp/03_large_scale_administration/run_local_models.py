


### 0) IMPORTS
import random
import time
import warnings
import pandas as pd
import json
import os

from datetime import datetime
from src.API_prompting import *
from src.huggingface_prompting import *
from tqdm import tqdm
from huggingface_hub import login

warnings.filterwarnings("ignore")

login(os.getenv("HF_TOKEN"))



def main():

    ### 1) INITIALIZATION

    # Initialize lists
    prompt_class_list = []
    inventory_list = []
    prompts = []

    # Set conditions for experiments
    inventories = ["bfi-llm", "social-desirability"]
    situation = "baseline"
    n_reps = 5
    timestamp = datetime.now().strftime("%Y%m%d")

    number_items = 59 # 44 BFI-LLM + 15
    prompt_variations = 2 # option order a and b

    # Models for experiments
    models = [
        # "Qwen/Qwen3-0.6B",
        # "Qwen/Qwen3-0.6B-Base",
        # "Qwen/Qwen3-8B-Base",
        # "Qwen/Qwen3.5-0.8B-Base",
        # "Qwen/Qwen3.5-2B-Base",
        # "Qwen/Qwen3.5-4B-Base",
        # "google/gemma-3-270m",
        # "google/gemma-3-270m-it",
        "google/gemma-3-4b-pt",
        # "meta-llama/Llama-3.1-8B",
    ]

    # Prompt constraint for local models
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



    ### 2) PROMPT GENERATION
    # for all inventories, option orders, models, and repetitions

    # Load winning prompt template with two possible option orders
    with open("../../dat/00_inventories/bfi-llm.json", "r") as f:
        bfi_llm_json = json.load(f)

    item_preamble = bfi_llm_json["item_preamble"]
    item_postamble = bfi_llm_json["item_postamble"]
    options_a = bfi_llm_json["options_a"]
    options_b = bfi_llm_json["options_b"]

    # Load inventories
    for inventory in inventories:
        items = []

        with open(f"../../dat/00_inventories/{inventory}.json") as inv:
            inventory_json = json.load(inv)

        # Collect all items
        for item_dict in inventory_json["items"]:
            items.append(item_dict["text"])
            inventory_list.append(inventory)

        for item in items:
            # Generate prompts with item preamble, item, and item postamble
            # Create two prompts per item: option_a and option_b (reverse order)
            prompt_a_prelim, prompt_b_prelim = prompt_generator_options(
                item_preamble, item, options_a, options_b, item_postamble
            )

            # Add json constraint
            prompt_a = f"""
            ### TASK
            {prompt_a_prelim}

            ### INSTRUCTIONS
            {json_constraint}

            ### OUTPUT
            """

            prompt_b = f"""
            ### TASK
            {prompt_b_prelim}

            ### INSTRUCTIONS
            {json_constraint}

            ### OUTPUT
            """

            prompts.append(prompt_a)
            prompts.append(prompt_b)

            # Create final PromptClass objects for models and repetitions
            for model in models:
                for rep in range(n_reps):
                    # Store all prompts and the corresponding info in a PromptClass object
                    prompt_class_list.append(PromptClass(prompt_a, model, inventory, situation, rep, item, item_preamble, item_postamble, options_a, timestamp, None, None, None))
                    prompt_class_list.append(PromptClass(prompt_b, model, inventory, situation, rep, item, item_preamble, item_postamble, options_b,timestamp, None, None, None))



    ### 3) SANITY CHECK & INFO
    # Check if length of prompt_class_list is correct
    tot_n_requests = number_items * n_reps * prompt_variations

    print("CORRECT NUMBER OF PROMPTS?")
    print(f"Number of models: {str(len(models) == len(set(pc.model for pc in prompt_class_list))).upper()} ({len(models)} vs. {len(set(pc.model for pc in prompt_class_list))})")
    print(f"Number inventories: {str(prompt_variations == len(set(pc.options for pc in prompt_class_list))).upper()} ({prompt_variations} vs. {len(set(pc.options for pc in prompt_class_list))})")
    print(f"Total number of requests: {str(tot_n_requests  == 590).upper()} ({tot_n_requests} vs. 590)")
    print(f"Length prompt_class_list: {str(len(prompt_class_list) == tot_n_requests * len(models)).upper()} ({len(prompt_class_list)} vs. {tot_n_requests * len(models)})\n")



    ### 4) DATA COLLECTION
    total_time = 0

    for model_name in models:

        # Load local model
        print(f"\nLoading model: {model_name}")
        runner = HFModelRunner(model_name)

        # Create cache
        cache_file = Path("cache/cache.csv")

        if cache_file.exists():
            cache = pd.read_csv(cache_file)
        else:
            cache = pd.DataFrame()

        # Get all prompts for the selected model
        model_prompts = [pc for pc in prompt_class_list if pc.model == model_name]

        # Random order for items
        random.shuffle(model_prompts)

        # Check if already cached
        model_prompts = [pc for pc in model_prompts if not is_cached(pc, cache)]
        if len(model_prompts) == 0:
            print(f"{model_name}: all prompts already cached → skipping")
            continue

        # Create list for responses
        results = []

        start = time.time()

        # Generating responses and save them in list
        for i, pc in enumerate(tqdm(model_prompts, desc=model_name)):

            response = runner.generate(pc.prompt, max_new_tokens=200)
            pc.response = response
            results.append(pc)

            # Save cache every 10th prompt
            if i % 10 == 0:
                save_cache_local(results, model_prompts[0], "responses")

        end = time.time()

        # Print info for model
        print(f"{model_name} done in {end - start:.2f} sec")

        total_time += (end - start)

        # Save final responses as csv
        save_cache_local(results, model_prompts[0], "experiments", final=True)

    # Print overall info for all models
    print("\nTotal runtime:", total_time)


if __name__ == "__main__":
    main()