from typing import List
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "씨앗AI"
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    
    # OpenAI 설정
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-0125-preview"
    
    # Whisper 설정
    WHISPER_MODEL: str = "large-v3"
    
    # 벡터 DB 설정
    VECTORDB_PATH: str = "./data/vectordb"
    
    # 환경 설정
    ENVIRONMENT: str
    DEBUG: bool = False
    
    # 보안 설정
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()