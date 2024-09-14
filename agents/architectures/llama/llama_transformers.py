import transformers
import torch
import os

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

class LlamaTransformer:
    def __init__(self, max_new_tokens: int, temperature: float = 0.7, top_p: float = 0.95):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.max_new_tokens = max_new_tokens
        self.weights = "app/model_weights/llama_3_1"
        self.temperature = temperature
        self.top_p = top_p

    def complete(self, system_prompt: str, user_prompt: str):
        # Check if the weights folder exists
        if not os.path.exists(self.weights):
            raise ValueError(f"Model weights folder not found: {self.weights}")

        # Load the model and tokenizer from the local weights
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.weights,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.weights)

        # Create the pipeline with the loaded model and tokenizer
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        outputs = pipeline(messages,
                           max_new_tokens=self.max_new_tokens,
                           do_sample=True,
                           temperature=self.temperature,
                           top_p=self.top_p,
                           )
        return outputs[0]["generated_text"]