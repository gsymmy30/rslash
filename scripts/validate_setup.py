#!/usr/bin/env python
"""Validate that all dependencies and services are working."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_reddit():
    """Test Reddit API connection."""
    try:
        import praw
        from config.settings import settings
        
        if not settings.REDDIT_CLIENT_ID or settings.REDDIT_CLIENT_ID == "your_client_id":
            print("⚠️  Reddit API: Please update .env with your Reddit credentials")
            print("   Get them from: https://www.reddit.com/prefs/apps")
            return False
            
        reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT
        )
        # Try to fetch one post
        post = next(reddit.subreddit("python").hot(limit=1))
        print(f"✅ Reddit API: Connected (test post: {post.title[:50]}...)")
        return True
    except ImportError:
        print("⚠️  Reddit API: praw not installed. Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Reddit API: {e}")
        return False

def check_postgres():
    """Test PostgreSQL connection."""
    try:
        import psycopg2
        from config.settings import settings
        conn = psycopg2.connect(settings.DATABASE_URL)
        conn.close()
        print("✅ PostgreSQL: Connected")
        return True
    except ImportError:
        print("⚠️  PostgreSQL: psycopg2 not installed. Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL: {e}")
        print("   Make sure Docker is running: docker-compose up -d")
        return False

def check_redis():
    """Test Redis connection."""
    try:
        import redis
        from config.settings import settings
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        print("✅ Redis: Connected")
        return True
    except ImportError:
        print("⚠️  Redis: redis-py not installed. Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Redis: {e}")
        print("   Make sure Docker is running: docker-compose up -d")
        return False

def check_ml_models():
    """Test ML model loading."""
    try:
        from sentence_transformers import SentenceTransformer
        from config.settings import settings
        print("Loading ML model (this may take a moment on first run)...")
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        test_embedding = model.encode("test")
        print(f"✅ ML Models: Loaded (embedding dim: {len(test_embedding)})")
        return True
    except ImportError:
        print("⚠️  ML Models: sentence-transformers not installed. Run: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ ML Models: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Validating rslash setup...\n")
    
    checks = [
        check_postgres(),
        check_redis(),
        check_reddit(),
        check_ml_models()
    ]
    
    if all(checks):
        print("\n✨ All systems go! Ready to build rslash!")
    else:
        print("\n⚠️  Some components need attention. Check the errors above.")
        print("\n📋 Setup checklist:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Start databases: docker-compose up -d")
        print("3. Update Reddit credentials in .env file")
        sys.exit(1)
