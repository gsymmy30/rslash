#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Setting up rslash - Reddit-powered Recommendation System${NC}"
echo "================================================"

# Create directory structure
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p backend/app/{routes,services}
mkdir -p ml
mkdir -p data
mkdir -p frontend
mkdir -p config
mkdir -p scripts

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Data
*.db
*.sqlite
*.pkl
*.npy
*.faiss

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Data folders
data/cache/
data/embeddings/
data/checkpoints/
EOF

# Create .env template
cat > .env.example << 'EOF'
# Reddit API (get from https://www.reddit.com/prefs/apps)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=rslash:v1.0.0 (by /u/yourusername)

# Database
DATABASE_URL=postgresql://rslash_user:rslash_pass@localhost:5432/rslash
REDIS_URL=redis://localhost:6379

# API
API_PORT=8000
FRONTEND_PORT=3000

# ML Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384
BATCH_SIZE=32

# Features
ENABLE_ONLINE_LEARNING=true
EXPLORATION_RATE=0.3
EOF

# Copy .env.example to .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${YELLOW}Created .env file - Please update with your Reddit API credentials${NC}"
fi

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0
python-multipart==0.0.6
aiofiles==23.2.1

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
redis==5.0.1
alembic==1.12.1

# Reddit
praw==7.7.1

# ML/AI
torch==2.1.1
sentence-transformers==2.2.2
numpy==1.24.3
scikit-learn==1.3.2
lightgbm==4.1.0
faiss-cpu==1.7.4

# Data Processing
pandas==2.1.3
scipy==1.11.4

# Utils
tqdm==4.66.1

# Monitoring
prometheus-client==0.19.0
EOF

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: rslash_user
      POSTGRES_PASSWORD: rslash_pass
      POSTGRES_DB: rslash
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
EOF

# Create config/settings.py
cat > config/__init__.py << 'EOF'
EOF

cat > config/settings.py << 'EOF'
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Reddit
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
    
    # Database
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
EOF

# Create validation script
cat > scripts/validate_setup.py << 'EOF'
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
EOF

chmod +x scripts/validate_setup.py

# Create empty __init__.py files
touch backend/__init__.py
touch backend/app/__init__.py
touch backend/app/routes/__init__.py
touch backend/app/services/__init__.py
touch ml/__init__.py
touch data/__init__.py
touch scripts/__init__.py

# Create basic README
cat > README.md << 'EOF'
# rslash - Reddit-powered Recommendation System

A real-time recommendation system inspired by YouTube Shorts, using Reddit content as the data source.

## Features
- Two-tower neural architecture for recommendations
- Real-time personalization
- Multi-stage ranking pipeline
- Online learning from user interactions

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Services
```bash
# Start PostgreSQL and Redis
docker-compose up -d
```

### 3. Configure Reddit API
1. Get credentials from https://www.reddit.com/prefs/apps
2. Update `.env` file with your credentials

### 4. Validate Setup
```bash
python scripts/validate_setup.py
```

### 5. Run the Application
```bash
# Start backend
python backend/app/main.py

# In another terminal, serve frontend
python -m http.server 3000 --directory frontend
```

## Architecture
- **Backend**: FastAPI + PostgreSQL + Redis
- **ML**: PyTorch + Sentence Transformers + LightGBM
- **Frontend**: Vanilla JS with swipeable cards
- **Data**: Real-time Reddit scraping via PRAW

## Development Progress
- [x] Project setup
- [ ] Reddit data pipeline
- [ ] Database models
- [ ] Two-tower architecture
- [ ] Ranking pipeline
- [ ] Frontend interface
- [ ] Online learning
- [ ] Deployment
EOF

# Create a simple frontend placeholder
cat > frontend/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>rslash - Personalized Reddit Feed</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="app">
        <h1>rslash</h1>
        <p>Loading your personalized feed...</p>
    </div>
    <script src="app.js"></script>
</body>
</html>
EOF

cat > frontend/style.css << 'EOF'
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: #0f0f0f;
    color: #ffffff;
    height: 100vh;
    overflow: hidden;
}

#app {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
}

h1 {
    font-size: 3rem;
    margin-bottom: 1rem;
    background: linear-gradient(45deg, #ff4500, #ff8717);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
EOF

cat > frontend/app.js << 'EOF'
console.log('rslash frontend loaded');
// Frontend code will be implemented in the next steps
EOF

# Create basic backend main.py
cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings

app = FastAPI(title="rslash API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "rslash API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=True
    )
EOF

echo -e "${GREEN}✅ Project structure created successfully!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Create and activate virtual environment:"
echo "   python -m venv venv"
echo "   source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo ""
echo "2. Install Python dependencies:"
echo "   pip install -r requirements.txt"
echo ""
echo "3. Start PostgreSQL and Redis:"
echo "   docker-compose up -d"
echo ""
echo "4. Get Reddit API credentials:"
echo "   - Go to https://www.reddit.com/prefs/apps"
echo "   - Create a new app (choose 'script' type)"
echo "   - Update the .env file with your credentials"
echo ""
echo "5. Validate setup:"
echo "   python scripts/validate_setup.py"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"