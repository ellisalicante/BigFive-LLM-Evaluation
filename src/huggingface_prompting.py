import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
