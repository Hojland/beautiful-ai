from pydantic import BaseSettings


class Settings(BaseSettings):
    HUGGINGFACE_TOKEN: str = "Your Huggingface Token"


settings = Settings()
