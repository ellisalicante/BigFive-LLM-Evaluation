


### 0) IMPORTS
import random
import warnings
import litellm

from datetime import datetime
from src.API_prompting import *

# Deactivate warnings
litellm.set_verbose = False
litellm.suppress_debug_info = True
warnings.filterwarnings("ignore", category=RuntimeWarning)



async def main():

    ### 1) INITIALIZATION

    # Initialize lists
    prompt_class_list = []
    item_preambles = []
    item_postambles = []
    inventory_list = []
    prompts = []

    # Set conditions for experiments
    inventories = ["bfi-llm", "social-desirability"]
    situation = "baseline"
    n_reps = 3
    total_costs = 0
    tpm_limit = 4000000
    rpm_limit = 1000
    path = "responses_pilot"
    timestamp = datetime.now().strftime("%Y%m%d")

    number_items = 59 # 44 BFI-LLM + 15

    # Get models for pilot
    models = [
        "gpt-4o",
        "claude-haiku-4-5",
        "xai/grok-4-1-fast-reasoning",
        "openrouter/deepseek/deepseek-r1",
        "openrouter/google/gemini-2.5-flash",
        "openrouter/meta-llama/llama-3.1-8b-instruct",
        "openrouter/z-ai/glm-5",
        "openrouter/qwen/qwen3.5-flash-02-23",
        "openrouter/mistralai/mistral-small-2603"
    ]



    ### 2) PROMPT GENERATION
    # for all inventories, option orders, models, and repetitions

    # Load all prompt template candidates
    with open("../../dat/02_pilot_study/prompt-templates.json", "r") as f:
        instructions_json = json.load(f)

    for item_dict in instructions_json["items"]:
        item_preambles.append(item_dict["item_preamble"])
        item_postambles.append(item_dict["item_postamble"])

    # Load inventories
    for inventory in inventories:
        items = []

        # Load inventory from .JSON
        with open(f"../../dat/00_inventories/{inventory}.json") as inv:
            inventory_json = json.load(inv)

        # Collect all items
        for item_dict in inventory_json["items"]:
            items.append(item_dict["text"])
            inventory_list.append(inventory)

        for item in items:
            for instruction in zip(item_preambles, item_postambles, inventory_list):
                # Generate prompts with item preamble, item, and item postamble
                prompt = prompt_generator(instruction[0], item, instruction[1], instruction[2])
                prompts.append(prompt)

                # Create final PromptClass objects for models and repetitions
                for model in models:
                    for rep in range(n_reps):
                        # Store all prompts and the corresponding info in a PromptClass object
                        prompt_class = PromptClass(prompt, model, inventory, situation, rep, item, instruction[0], instruction[1], None, timestamp, None, None, None)
                        prompt_class_list.append(prompt_class)



    ### 3) SANITY CHECK & INFO
    # Check if length of prompt_class_list is correct
    tot_n_requests = number_items * n_reps * len(item_preambles)

    print("CORRECT NUMBER OF PROMPTS?")
    print(f"Number of models: {str(len(models) == len(set(pc.model for pc in prompt_class_list))).upper()} ({len(models)} vs. {len(set(pc.model for pc in prompt_class_list))})")
    print(f"Prompt variations: {str(len(item_preambles) == len(set(pc.preamble for pc in prompt_class_list))).upper()} ({len(item_preambles)} vs. {len(set(pc.preamble for pc in prompt_class_list))})")
    print(f"Total number of requests: {str(tot_n_requests  == 59*3*7).upper()} ({tot_n_requests} vs. {59*3*7})")
    print(f"Length prompt_class_list: {str(len(prompt_class_list) == tot_n_requests * len(models)).upper()} ({len(prompt_class_list)} vs. {tot_n_requests * len(models)})\n")



    ### 4) DATA COLLECTION
    for model in models:

        # Create cache
        cache = cache_directory(prompt_class_list[0])
        cache = cache[cache["model"] == model]

        # Get all prompts for the selected model
        model_prompts = [pc for pc in prompt_class_list if pc.model == model]
        total = len(model_prompts)

        # Random order for items
        model_prompts = random.sample(model_prompts, total)

        # Create queues for prompts and responses
        prompt_queue, response_queue = await create_queue(model_prompts)

        # Parallel prompting and save responses in queue
        start_time = time.time()
        response_queue, response_costs = await parallel_prompting(prompt_queue, response_queue, cache, total, tpm_limit,
                                                  rpm_limit, n_reps, path)

        # Save final responses as csv
        save_cache(response_queue, model_prompts[0], path, final=True)

        end_time = time.time()

        # Print info for model
        print(f"\nProcessed {total} prompts in {end_time - start_time:.2f} seconds.")
        print(f"\nResponse costs for model {model}:", response_costs)
        total_costs += response_costs

    # Print overall info for all models
    print(f"\nTotal costs for all models: {total_costs}")


if __name__ == "__main__":
    asyncio.run(main())
