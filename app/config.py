from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "LangChain + Jev Ticket Router"
    debug: bool = True
    
    # TypeSafe AI (Jev) Configuration
    typesafe_api_key: str = ""
    typesafe_base_url: str = "https://api.typesafe.ai/v1"
    jev_model: str = "jev-latest"
    
    # OpenAI Configuration (for LangChain LLM)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    
    # OpenRouter Configuration (alternative for Jev)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
