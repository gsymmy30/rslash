# backend/app/main.py
"""
FastAPI backend for rslash recommendation system.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import sys
import os
import uuid
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings
from backend.app.database import SessionLocal, get_db
from backend.app.models import User, Post, Interaction, Session
from ml.ranking import RankingPipeline
from ml.embeddings import EmbeddingManager

app = FastAPI(title="rslash API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
ranking_pipeline = RankingPipeline()
embedding_manager = EmbeddingManager()

# Pydantic models
class FeedbackRequest(BaseModel):
    user_id: str
    post_id: str
    interaction_type: str  # like, dislike, skip
    time_spent: float = 0.0

class UserCreate(BaseModel):
    username: Optional[str] = None

# API Routes

@app.get("/")
async def root():
    return {
        "message": "rslash API is running!",
        "endpoints": {
            "feed": "/api/feed/{user_id}",
            "feedback": "/api/feedback",
            "users": "/api/users",
            "stats": "/api/stats/{user_id}"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/users")
async def create_user(user_data: UserCreate, db=Depends(get_db)):
    """Create a new user."""
    user_id = str(uuid.uuid4())
    username = user_data.username or f"user_{user_id[:8]}"
    
    # Check if username exists
    existing = db.query(User).filter_by(username=username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user
    user = User(
        user_id=user_id,
        username=username,
        exploration_rate=0.5  # Start with 50% exploration
    )
    db.add(user)
    db.commit()
    
    return {
        "user_id": user_id,
        "username": username,
        "message": "User created successfully"
    }

@app.get("/api/users")
async def list_users(db=Depends(get_db)):
    """List all available users (for demo)."""
    users = db.query(User).all()
    return {
        "users": [
            {
                "user_id": u.user_id,
                "username": u.username,
                "total_interactions": u.total_interactions
            } for u in users
        ]
    }

@app.get("/api/feed/{user_id}")
async def get_feed(user_id: str, num_items: int = 10, db=Depends(get_db)):
    """Get personalized feed for a user."""
    
    # Check if user exists, create if not
    user = db.query(User).filter_by(user_id=user_id).first()
    if not user:
        # Auto-create user for demo
        user = User(
            user_id=user_id,
            username=f"user_{user_id[:8]}",
            exploration_rate=0.5
        )
        db.add(user)
        db.commit()
    
    # Get recommendations
    recommendations = ranking_pipeline.get_recommendations(user_id, num_items)
    
    if not recommendations:
        # Fallback to popular posts
        posts = db.query(Post).order_by(Post.score.desc()).limit(num_items).all()
        recommendations = [
            {
                'post_id': p.post_id,
                'title': p.title,
                'subreddit': p.subreddit,
                'score': p.score,
                'url': p.url,
                'permalink': p.permalink,
                'content': p.content[:200] if p.content else '',
                'num_comments': p.num_comments,
                'created_utc': p.created_utc,
                'is_video': p.is_video
            } for p in posts
        ]
    
    return {
        "user_id": user_id,
        "recommendations": recommendations,
        "count": len(recommendations)
    }

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest, db=Depends(get_db)):
    """Submit user feedback on a post."""
    
    # Verify user and post exist
    user = db.query(User).filter_by(user_id=feedback.user_id).first()
    post = db.query(Post).filter_by(post_id=feedback.post_id).first()
    
    if not user or not post:
        raise HTTPException(status_code=404, detail="User or post not found")
    
    # Record interaction
    interaction = Interaction(
        user_id=feedback.user_id,
        post_id=feedback.post_id,
        interaction_type=feedback.interaction_type,
        time_spent=feedback.time_spent
    )
    db.add(interaction)
    
    # Update user statistics
    user.total_interactions += 1
    if feedback.interaction_type == 'like':
        user.total_likes += 1
    elif feedback.interaction_type == 'dislike':
        user.total_dislikes += 1
    
    # Update user's average read time
    if feedback.time_spent > 0:
        user.avg_read_time = (
            (user.avg_read_time * (user.total_interactions - 1) + feedback.time_spent) 
            / user.total_interactions
        )
    
    db.commit()
    
    # Update user embedding in real-time
    embedding_manager.update_user_embedding_online(
        feedback.user_id,
        feedback.post_id,
        feedback.interaction_type
    )
    
    return {
        "status": "success",
        "message": f"Feedback recorded: {feedback.interaction_type}"
    }

@app.get("/api/stats/{user_id}")
async def get_user_stats(user_id: str, db=Depends(get_db)):
    """Get user statistics."""
    user = db.query(User).filter_by(user_id=user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get interaction breakdown
    interactions = db.query(Interaction).filter_by(user_id=user_id).all()
    
    interaction_breakdown = {}
    for interaction in interactions:
        interaction_breakdown[interaction.interaction_type] = \
            interaction_breakdown.get(interaction.interaction_type, 0) + 1
    
    # Get favorite subreddits
    favorite_subreddits = {}
    for interaction in interactions:
        if interaction.interaction_type == 'like':
            post = db.query(Post).filter_by(post_id=interaction.post_id).first()
            if post:
                favorite_subreddits[post.subreddit] = \
                    favorite_subreddits.get(post.subreddit, 0) + 1
    
    # Sort favorite subreddits
    favorite_subreddits = dict(
        sorted(favorite_subreddits.items(), key=lambda x: x[1], reverse=True)[:5]
    )
    
    return {
        "user_id": user_id,
        "username": user.username,
        "stats": {
            "total_interactions": user.total_interactions,
            "total_likes": user.total_likes,
            "total_dislikes": user.total_dislikes,
            "avg_read_time": user.avg_read_time,
            "exploration_rate": user.exploration_rate
        },
        "interaction_breakdown": interaction_breakdown,
        "favorite_subreddits": favorite_subreddits
    }

@app.get("/api/posts/{post_id}")
async def get_post(post_id: str, db=Depends(get_db)):
    """Get details of a specific post."""
    post = db.query(Post).filter_by(post_id=post_id).first()
    
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    return {
        "post_id": post.post_id,
        "title": post.title,
        "content": post.content,
        "subreddit": post.subreddit,
        "author": post.author,
        "score": post.score,
        "num_comments": post.num_comments,
        "url": post.url,
        "permalink": post.permalink,
        "created_utc": post.created_utc
    }

if __name__ == "__main__":
    print(f"Starting rslash API on port {settings.API_PORT}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=True
    )