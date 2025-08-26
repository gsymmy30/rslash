# scripts/setup_db.py
"""
Setup database tables and load initial data.
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import engine, SessionLocal
from backend.app.models import Base, Post, User, Interaction, Session
from data.reddit_scraper import RedditScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables."""
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Tables created successfully")

def drop_tables():
    """Drop all tables (use with caution!)."""
    logger.info("Dropping existing tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("Tables dropped")

def load_posts_from_json(filepath='data/reddit_posts.json'):
    """Load posts from JSON file into database."""
    db = SessionLocal()
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} not found. Fetching fresh data...")
            scraper = RedditScraper()
            posts_data = scraper.fetch_diverse_posts(
                num_posts_per_subreddit=30,
                subreddits=['programming', 'funny', 'todayilearned', 'gaming', 'movies']
            )
            scraper.save_data(posts_data, filepath)
        else:
            with open(filepath, 'r') as f:
                posts_data = json.load(f)
        
        logger.info(f"Loading {len(posts_data)} posts into database...")
        
        loaded_count = 0
        for post_data in posts_data:
            # Check if post already exists
            existing = db.query(Post).filter_by(post_id=post_data['post_id']).first()
            if existing:
                continue
            
            # Create post object
            post = Post(
                post_id=post_data['post_id'],
                title=post_data['title'],
                content=post_data.get('content', ''),
                subreddit=post_data['subreddit'],
                author=post_data.get('author', '[deleted]'),
                score=post_data['score'],
                upvote_ratio=post_data['upvote_ratio'],
                num_comments=post_data['num_comments'],
                awards=post_data.get('awards', 0),
                url=post_data.get('url', ''),
                permalink=post_data.get('permalink', ''),
                is_video=post_data.get('is_video', False),
                is_self=post_data.get('is_self', True),
                nsfw=post_data.get('nsfw', False),
                total_engagement=post_data.get('total_engagement', 0),
                engagement_rate=post_data.get('engagement_rate', 0.0),
                estimated_read_time=post_data.get('estimated_read_time', 0.0),
                text_length=post_data.get('text_length', 0),
                created_utc=post_data['created_utc'],
                post_age_hours=post_data.get('post_age_hours', 0),
                embedding_text=post_data.get('embedding_text', '')
            )
            
            db.add(post)
            loaded_count += 1
        
        db.commit()
        logger.info(f"✅ Loaded {loaded_count} new posts successfully")
        
    except Exception as e:
        logger.error(f"Error loading posts: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def create_demo_users():
    """Create demo users for testing."""
    db = SessionLocal()
    
    try:
        demo_users = [
            {'user_id': 'tech_enthusiast', 'username': 'tech_lover',
             'preferred_subreddits': {'programming': 0.9, 'technology': 0.8, 'machinelearning': 0.7}},
            {'user_id': 'entertainment_fan', 'username': 'fun_seeker',
             'preferred_subreddits': {'funny': 0.9, 'memes': 0.8, 'videos': 0.7}},
            {'user_id': 'knowledge_seeker', 'username': 'curious_mind',
             'preferred_subreddits': {'todayilearned': 0.9, 'explainlikeimfive': 0.8, 'askreddit': 0.7}},
            {'user_id': 'new_user', 'username': 'newbie',
             'preferred_subreddits': {}, 'exploration_rate': 0.8},
        ]
        
        created_count = 0
        for user_data in demo_users:
            # Check if user exists
            existing = db.query(User).filter_by(user_id=user_data['user_id']).first()
            if existing:
                continue
            
            user = User(
                user_id=user_data['user_id'],
                username=user_data['username'],
                preferred_subreddits=user_data.get('preferred_subreddits', {}),
                exploration_rate=user_data.get('exploration_rate', 0.3)
            )
            db.add(user)
            created_count += 1
        
        db.commit()
        logger.info(f"✅ Created {created_count} new demo users")
        
    except Exception as e:
        logger.error(f"Error creating users: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def load_synthetic_interactions(filepath='data/synthetic_interactions.json'):
    """Load synthetic interactions."""
    db = SessionLocal()
    
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                interactions_data = json.load(f)
            
            logger.info(f"Loading {len(interactions_data)} interactions...")
            
            loaded_count = 0
            for int_data in interactions_data[:500]:  # Limit for testing
                # Make sure user and post exist
                user = db.query(User).filter_by(user_id=int_data['user_id']).first()
                post = db.query(Post).filter_by(post_id=int_data['post_id']).first()
                
                if user and post:
                    interaction = Interaction(
                        user_id=int_data['user_id'],
                        post_id=int_data['post_id'],
                        interaction_type=int_data['interaction_type'],
                        time_spent=int_data.get('time_spent', 0)
                    )
                    db.add(interaction)
                    loaded_count += 1
            
            db.commit()
            logger.info(f"✅ Loaded {loaded_count} synthetic interactions")
        else:
            logger.info("No synthetic interactions file found, skipping...")
            
    except Exception as e:
        logger.error(f"Error loading interactions: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def verify_setup():
    """Verify database setup."""
    db = SessionLocal()
    
    try:
        post_count = db.query(Post).count()
        user_count = db.query(User).count()
        interaction_count = db.query(Interaction).count()
        
        print("\n📊 Database Statistics:")
        print(f"   Posts: {post_count}")
        print(f"   Users: {user_count}")
        print(f"   Interactions: {interaction_count}")
        
        if post_count > 0:
            sample_post = db.query(Post).first()
            print(f"\n📝 Sample post: {sample_post.title[:80]}...")
            
        return post_count > 0 and user_count > 0
        
    finally:
        db.close()

def main():
    """Main setup function."""
    print("🚀 Setting up rslash database...")
    
    # Drop and recreate tables (comment out if you want to keep existing data)
    # drop_tables()
    
    # Create tables
    create_tables()
    
    # Load data
    print("\n📥 Loading Reddit posts...")
    load_posts_from_json()
    
    print("\n👥 Creating demo users...")
    create_demo_users()
    
    print("\n🔄 Loading interactions...")
    load_synthetic_interactions()
    
    # Verify
    if verify_setup():
        print("\n✨ Database setup complete! Ready to build recommendations!")
    else:
        print("\n⚠️  Setup incomplete. Check the logs above.")

if __name__ == "__main__":
    main()