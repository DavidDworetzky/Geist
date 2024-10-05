import transformers
import torch
import os
import logging
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            
        logger.info("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(self.weights_dir, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(self.weights_dir)

        logger.info(f"Model loaded. Model size: {model.num_parameters()} parameters")

        # Check for CUDA, MPS, or CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal) backend")
        else:
            device = torch.device("cpu")
            logger.info("CUDA and MPS not available, using CPU")

        model = model.to(device)

        logger.info("Creating pipeline")
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("Starting text generation")
        try:
            outputs = pipeline(messages,
                               max_new_tokens=self.max_new_tokens,
                               do_sample=True,
                               temperature=self.temperature,
                               top_p=self.top_p,
                               )
            logger.info("Text generation completed successfully")
            return outputs[0]["generated_text"]
        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            raise