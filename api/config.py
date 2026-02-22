import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Required
    CLASSIFIER_ARTIFACTS_DIR: str = "./classifier/artifacts"
    VLLM_BASE_URL: str = "http://localhost:8000"
    VLLM_MODEL: str = "meta-llama/Llama-2-7b-chat-hf"
    
    # Optional
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    MAX_INPUT_CHARS: int = 5000
    VLLM_TIMEOUT_SECS: int = 60
    REWRITE_MAX_RETRIES: int = 2
    DEFAULT_TOX_THRESHOLD: float = 0.2
    
    class Config:
        env_file = ".env"

settings = Settings()
