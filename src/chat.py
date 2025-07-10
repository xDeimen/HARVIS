
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
import torch

SYSTEM_PROMPT="You are a helpful assistant. Your jobs is to answer questions as corect as possible."


class LLM:
    def __init__(self, path: str, device: int = -1,):
        self.path = path
        self.device = device
        self._load_model()
        self._create_pipeline()
        logging.info("LLM Instantiated")

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForCausalLM.from_pretrained(self.path)

    def _create_pipeline(self):
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="cuda",
            return_full_text=False
        )
        
    def _create_messages(self, user_prompt):
        prompt = f"""
        You are a helpful assistant. Only answer the user's question. Do not continue or generate more questions.
        Q: {user_prompt}
        A: """
        return prompt
        

    def chat(self, user_prompt: str) -> str:
        messages = self._create_messages(user_prompt)
        response = self.pipeline(messages, max_new_tokens=128, temperature=0.7)[0]["generated_text"]
        print(response)
        return response

