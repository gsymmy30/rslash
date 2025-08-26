# backend/app/models.py
"""
SQLAlchemy models for rslash.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

# Create base class for models directly here
Base = declarative_base()

class Post(Base):
    """Reddit post model."""
    __tablename__ = "posts"
    
    # Primary fields
    post_id = Column(String, primary_key=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    subreddit = Column(String, nullable=False, index=True)
    author = Column(String)
    
    # Reddit metrics
    score = Column(Integer, default=0)
    upvote_ratio = Column(Float, default=0.5)
    num_comments = Column(Integer, default=0)
    awards = Column(Integer, default=0)
    
    # Post metadata
    url = Column(Text)
    permalink = Column(Text)
    is_video = Column(Boolean, default=False)
    is_self = Column(Boolean, default=True)
    nsfw = Column(Boolean, default=False)
    
    # Derived features
    total_engagement = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)
    estimated_read_time = Column(Float, default=0.0)
    text_length = Column(Integer, default=0)
    
    # Timestamps
    created_utc = Column(Float)  # Reddit's timestamp
    fetched_at = Column(DateTime, server_default=func.now())
    post_age_hours = Column(Float)
    
    # Embedding
    embedding = Column(JSON)  # Store as JSON array
    embedding_text = Column(Text)  # Text used for embedding
    
    # Relationships
    interactions = relationship("Interaction", back_populates="post")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_posts_subreddit_score', 'subreddit', 'score'),
        Index('ix_posts_created_utc', 'created_utc'),
    )


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True)  # Optional username
    
    # User embedding (learned preferences)
    user_embedding = Column(JSON)  # Store as JSON array
    
    # User statistics
    total_interactions = Column(Integer, default=0)
    total_likes = Column(Integer, default=0)
    total_dislikes = Column(Integer, default=0)
    avg_read_time = Column(Float, default=0.0)
    
    # User preferences (learned)
    preferred_subreddits = Column(JSON)  # {"programming": 0.8, "funny": 0.3}
    content_length_preference = Column(String, default="medium")  # short/medium/long
    
    # Activity patterns
    active_hours = Column(JSON)  # [7, 8, 20, 21, 22] - hours when user is active
    last_active = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    # Exploration vs exploitation
    exploration_rate = Column(Float, default=0.5)  # Start with 50% exploration
    
    # Relationships
    interactions = relationship("Interaction", back_populates="user")
    sessions = relationship("Session", back_populates="user")


class Interaction(Base):
    """User-Post interaction model."""
    __tablename__ = "interactions"
    
    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign keys
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    post_id = Column(String, ForeignKey("posts.post_id"), nullable=False)
    session_id = Column(String, ForeignKey("sessions.session_id"))
    
    # Interaction data
    interaction_type = Column(String, nullable=False)  # like, dislike, skip, click
    time_spent = Column(Float, default=0.0)  # Seconds spent on post
    
    # Additional signals
    clicked_comments = Column(Boolean, default=False)
    shared = Column(Boolean, default=False)
    
    # Timestamp
    timestamp = Column(DateTime, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    post = relationship("Post", back_populates="interactions")
    session = relationship("Session", back_populates="interactions")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_interactions_user_timestamp', 'user_id', 'timestamp'),
        Index('ix_interactions_post_type', 'post_id', 'interaction_type'),
    )


class Session(Base):
    """User session model for tracking behavior patterns."""
    __tablename__ = "sessions"
    
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    
    # Session metadata
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime)
    duration_seconds = Column(Float)
    
    # Session statistics
    posts_viewed = Column(Integer, default=0)
    posts_liked = Column(Integer, default=0)
    posts_disliked = Column(Integer, default=0)
    avg_time_per_post = Column(Float, default=0.0)
    
    # Device/context (for future use)
    device_type = Column(String)  # mobile, desktop, tablet
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session")


class RecommendationLog(Base):
    """Log of recommendations served to users."""
    __tablename__ = "recommendation_logs"
    
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # What was recommended
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    post_ids = Column(JSON)  # List of recommended post IDs
    
    # Recommendation metadata
    algorithm_version = Column(String)  # Track which algorithm was used
    exploration_items = Column(Integer, default=0)  # How many were exploration
    
    # Performance metrics
    clicks = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    total_time_spent = Column(Float, default=0.0)
    
    # Timestamp
    timestamp = Column(DateTime, server_default=func.now())
    
    # Index for analytics
    __table_args__ = (
        Index('ix_recommendations_user_timestamp', 'user_id', 'timestamp'),
    )