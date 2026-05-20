


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
    inventory_list = []
    prompts = []

    # Set conditions for experiments
    inventories = ["bfi-llm", "social-desirability"]
    situation = "baseline"
    n_reps = 5
    total_costs = 0
    tpm_limit = 4000000
    rpm_limit = 1000
    path = "responses"
    timestamp = datetime.now().strftime("%Y%m%d")

    number_items = 59 # 44 BFI-LLM + 15
    prompt_variations = 2 # option order a and b

    # Get models for experiments from meta info
    meta = pd.read_csv("../../dat/03_large_scale_administration/meta_info_models.csv")
    models = meta["Model_ID"].dropna().tolist()



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

        # Load inventory from .JSON
        with open(f"../../dat/00_inventories/{inventory}.json") as inv:
            inventory_json = json.load(inv)

        # Collect all items
        for item_dict in inventory_json["items"]:
            items.append(item_dict["text"])
            inventory_list.append(inventory)

        for item in items:
            # Generate prompts with item preamble, item, and item postamble
            # Create two prompts per item: option_a and option_b (reverse order)
            prompt_a, prompt_b = prompt_generator_options(item_preamble, item, options_a, options_b, item_postamble)
            prompts.append(prompt_a)
            prompts.append(prompt_b)

            # Create final PromptClass objects for models and repetitions
            for model in models:
                for rep in range(n_reps):
                    # Store all prompts and the corresponding info in a PromptClass object
                    prompt_class_list.append(PromptClass(prompt_a, model, inventory, situation, rep, item, item_preamble, item_postamble, options_a, timestamp, None, None, None))
                    prompt_class_list.append(PromptClass(prompt_b, model, inventory, situation, rep, item, item_preamble, item_postamble, options_b, timestamp, None, None, None))



    ### 3) SANITY CHECK & INFO
    # Check if length of prompt_class_list is correct
    tot_n_requests = number_items * n_reps * prompt_variations

    print("CORRECT NUMBER OF PROMPTS?")
    print(f"Number of models: {str(len(models) == len(set(pc.model for pc in prompt_class_list))).upper()} ({len(models)} vs. {len(set(pc.model for pc in prompt_class_list))})")
    print(f"Prompt variations: {str(prompt_variations == len(set(pc.options for pc in prompt_class_list))).upper()} ({prompt_variations} vs. {len(set(pc.options for pc in prompt_class_list))})")
    print(f"Total number of requests: {str(tot_n_requests  == 590).upper()} ({tot_n_requests} vs. 590)")
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
