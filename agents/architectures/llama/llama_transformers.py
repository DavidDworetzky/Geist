import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

class LlamaTransformer:
    def __init__(self, max_new_tokens:int):
        self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.max_new_tokens = max_new_tokens

    def complete(self, system_prompt:str, user_prompt:str):
        pipeline = transformers.pipeline(
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        outputs = pipeline(messages,
                           max_new_tokens = self.max_new_tokens
                           )
        return outputs[0]["generated_text"][-1]