# config/settings.py
"""
Settings configuration with SQLite option for quick development.
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Reddit
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
    
    # Database - Use SQLite for quick development
    USE_SQLITE = os.getenv("USE_SQLITE", "true").lower() == "true"
    
    if USE_SQLITE:
        # SQLite (no Docker needed!)
        DATABASE_URL = "sqlite:///./rslash.db"
    else:
        # PostgreSQL (requires Docker)
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rslash_user:rslash_pass@localhost:5432/rslash")
    
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # ML
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    
    # API
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # Features
    EXPLORATION_RATE = float(os.getenv("EXPLORATION_RATE", 0.3))
    ENABLE_ONLINE_LEARNING = os.getenv("ENABLE_ONLINE_LEARNING", "true").lower() == "true"
    
settings = Settings()

print(f"Using database: {settings.DATABASE_URL}")