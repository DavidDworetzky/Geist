import transformers
import torch
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

class LlamaTransformer:
    def __init__(self, max_new_tokens: int, temperature: float = 0.7, top_p: float = 0.95):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.max_new_tokens = max_new_tokens
        self.weights_dir = "app/model_weights/llama_3_1"
        self.temperature = temperature
        self.top_p = top_p

    def download_model(self):
        # Login to Hugging Face
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError("HUGGING_FACE_HUB_TOKEN not found in environment variables")
        login(token=token)

        # Download model and tokenizer
        print(f"Downloading model to {self.weights_dir}")
        AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir=self.weights_dir)
        AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.weights_dir)

    def complete(self, system_prompt: str, user_prompt: str):
        # Check if model exists, if not, download it
        if not os.path.exists(os.path.join(self.weights_dir, "config.json")):
            self.download_model()

        if not os.path.exists(os.path.join(self.weights_dir, "config.json")):
            raise ValueError(f"Model weights do not exist at {self.weights_dir}")
            
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.weights_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.weights_dir)

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