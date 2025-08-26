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
