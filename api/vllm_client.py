import httpx
import logging
from api.config import settings

class VLLMClient:
    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def complete(self, prompt: str, **kwargs) -> str:
        """Call the completions endpoint."""
        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "stop": kwargs.get("stop", None),
            "n": kwargs.get("n", 1)
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["text"]
        except Exception as e:
            logging.error(f"vLLM completion error: {e}")
            raise

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """Call the chat completions endpoint."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "n": kwargs.get("n", 1)
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"vLLM chat error: {e}")
            raise

    async def generate_n_candidates(self, messages: list[dict], n: int, **kwargs) -> list[str]:
        """Generate multiple candidates using the chat completions endpoint."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 256),
            "temperature": kwargs.get("temperature", 0.8), # Higher temp for variety
            "top_p": kwargs.get("top_p", 0.95),
            "n": n
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return [choice["message"]["content"] for choice in data["choices"]]
        except Exception as e:
            logging.error(f"vLLM batch generation error: {e}")
            raise

# Singleton instance
vllm_client = VLLMClient(settings.VLLM_BASE_URL, settings.VLLM_MODEL, settings.VLLM_TIMEOUT_SECS)
