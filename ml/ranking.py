# ml/ranking.py
"""
Multi-stage ranking pipeline for recommendations.
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
import sys
import os
import random
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app.database import SessionLocal
from backend.app.models import Post, User, Interaction
from ml.embeddings import EmbeddingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RankingPipeline:
    """Multi-stage ranking pipeline."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        # Try to load existing index
        self.embedding_manager.load_index('data/post_index.faiss')
        
    def get_candidates(self, user_id: str, num_candidates: int = 100) -> List[Post]:
        """
        Stage 1: Candidate Generation
        Retrieve initial set of candidates using multiple strategies.
        """
        db = SessionLocal()
        candidates = []
        
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            if not user:
                # New user - return popular posts
                candidates = db.query(Post).order_by(Post.score.desc()).limit(num_candidates).all()
                return candidates
            
            # Get user's interaction history to avoid repeats
            seen_post_ids = set(
                i.post_id for i in db.query(Interaction)
                .filter_by(user_id=user_id)
                .all()
            )
            
            # Strategy 1: Similarity-based (40% of candidates)
            similarity_candidates = []
            if user.user_embedding:
                similar_posts = self.embedding_manager.find_similar_posts(
                    np.array(user.user_embedding),
                    k=int(num_candidates * 0.4)
                )
                for post_id, _ in similar_posts:
                    if post_id not in seen_post_ids:
                        post = db.query(Post).filter_by(post_id=post_id).first()
                        if post:
                            similarity_candidates.append(post)
            
            # Strategy 2: Trending posts (30% of candidates)
            trending_cutoff = datetime.now().timestamp() - (48 * 3600)  # Last 48 hours
            trending_candidates = db.query(Post).filter(
                Post.created_utc > trending_cutoff,
                ~Post.post_id.in_(seen_post_ids)
            ).order_by(
                (Post.score / (Post.post_age_hours + 1)).desc()  # Score velocity
            ).limit(int(num_candidates * 0.3)).all()
            
            # Strategy 3: User's preferred subreddits (20% of candidates)
            subreddit_candidates = []
            if user.preferred_subreddits:
                top_subreddits = sorted(
                    user.preferred_subreddits.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                
                for subreddit, _ in top_subreddits:
                    posts = db.query(Post).filter(
                        Post.subreddit == subreddit,
                        ~Post.post_id.in_(seen_post_ids)
                    ).order_by(Post.score.desc()).limit(10).all()
                    subreddit_candidates.extend(posts)
            
            # Strategy 4: Exploration (10% of candidates)
            from sqlalchemy import func
            exploration_candidates = db.query(Post).filter(
                ~Post.post_id.in_(seen_post_ids)
            ).order_by(
                func.random()  # Random posts for discovery
            ).limit(int(num_candidates * 0.1)).all()
            
            # Combine all strategies
            candidates = (similarity_candidates + trending_candidates + 
                         subreddit_candidates + exploration_candidates)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for post in candidates:
                if post.post_id not in seen:
                    seen.add(post.post_id)
                    unique_candidates.append(post)
            
            return unique_candidates[:num_candidates]
            
        finally:
            db.close()
    
    def score_candidates(self, user_id: str, candidates: List[Post]) -> List[Tuple[Post, float]]:
        """
        Stage 2: Scoring
        Score each candidate based on multiple features.
        """
        db = SessionLocal()
        scored_posts = []
        
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            
            for post in candidates:
                score = 0.0
                
                # 1. Embedding similarity (if available)
                if user and user.user_embedding and post.embedding:
                    similarity = np.dot(
                        np.array(user.user_embedding),
                        np.array(post.embedding)
                    )
                    score += similarity * 0.4  # 40% weight
                
                # 2. Engagement metrics
                engagement_score = (
                    (post.score / 1000) * 0.3 +  # Normalized score
                    (post.num_comments / 100) * 0.2 +  # Comment engagement
                    post.upvote_ratio * 0.1  # Quality signal
                )
                score += min(engagement_score, 1.0) * 0.3  # 30% weight, capped at 1
                
                # 3. Freshness
                hours_old = post.post_age_hours if post.post_age_hours else 24
                freshness_score = max(0, 1 - (hours_old / 168))  # Decay over a week
                score += freshness_score * 0.2  # 20% weight
                
                # 4. Subreddit affinity
                if user and user.preferred_subreddits:
                    subreddit_score = user.preferred_subreddits.get(post.subreddit, 0)
                    score += subreddit_score * 0.1  # 10% weight
                
                scored_posts.append((post, score))
            
            # Sort by score
            scored_posts.sort(key=lambda x: x[1], reverse=True)
            return scored_posts
            
        finally:
            db.close()
    
    def apply_business_rules(self, scored_posts: List[Tuple[Post, float]], 
                           user_id: str, num_items: int = 10) -> List[Post]:
        """
        Stage 3: Business Rules & Diversity
        Apply final filtering and diversity rules.
        """
        db = SessionLocal()
        final_posts = []
        
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            exploration_rate = user.exploration_rate if user else 0.5
            
            # Track diversity metrics
            subreddit_count = {}
            content_types = {'text': 0, 'link': 0, 'video': 0}
            
            for post, score in scored_posts:
                # Diversity rules
                # 1. Max 2 posts per subreddit
                if subreddit_count.get(post.subreddit, 0) >= 2:
                    continue
                
                # 2. Balance content types
                if post.is_video:
                    content_type = 'video'
                elif post.is_self:
                    content_type = 'text'
                else:
                    content_type = 'link'
                
                if content_types[content_type] >= num_items // 2:
                    continue
                
                # 3. Filter NSFW content (optional)
                if post.nsfw:
                    continue
                
                # Add to final list
                final_posts.append(post)
                subreddit_count[post.subreddit] = subreddit_count.get(post.subreddit, 0) + 1
                content_types[content_type] += 1
                
                if len(final_posts) >= num_items:
                    break
            
            # Add exploration items
            num_exploration = int(num_items * exploration_rate)
            if num_exploration > 0 and len(final_posts) > num_exploration:
                # Replace some items with random exploration
                from sqlalchemy import func
                exploration_indices = random.sample(
                    range(len(final_posts) - num_exploration, len(final_posts)),
                    num_exploration
                )
                
                exploration_posts = db.query(Post).order_by(
                    func.random()
                ).limit(num_exploration).all()
                
                for idx, exp_post in zip(exploration_indices, exploration_posts):
                    if idx < len(final_posts):
                        final_posts[idx] = exp_post
            
            return final_posts
            
        finally:
            db.close()
    
    def get_recommendations(self, user_id: str, num_items: int = 10) -> List[Dict]:
        """
        Get final recommendations for a user.
        """
        # Stage 1: Get candidates
        candidates = self.get_candidates(user_id, num_candidates=100)
        
        if not candidates:
            logger.warning(f"No candidates found for user {user_id}")
            return []
        
        # Stage 2: Score candidates
        scored_posts = self.score_candidates(user_id, candidates)
        
        # Stage 3: Apply business rules
        final_posts = self.apply_business_rules(scored_posts, user_id, num_items)
        
        # Format results
        recommendations = []
        for post in final_posts:
            recommendations.append({
                'post_id': post.post_id,
                'title': post.title,
                'subreddit': post.subreddit,
                'score': post.score,
                'url': post.url,
                'permalink': post.permalink,
                'content': post.content[:200] if post.content else '',
                'num_comments': post.num_comments,
                'created_utc': post.created_utc,
                'is_video': post.is_video
            })
        
        return recommendations


def test_ranking():
    """Test the ranking pipeline."""
    pipeline = RankingPipeline()
    
    # Test for different user types
    test_users = ['tech_enthusiast', 'entertainment_fan', 'new_user']
    
    for user_id in test_users:
        print(f"\n{'='*60}")
        print(f"Recommendations for {user_id}:")
        print('='*60)
        
        recommendations = pipeline.get_recommendations(user_id, num_items=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. r/{rec['subreddit']} (score: {rec['score']})")
            print(f"   {rec['title'][:80]}...")
            print(f"   💬 {rec['num_comments']} comments")
    
    print("\n✅ Ranking pipeline ready!")


if __name__ == "__main__":
    test_ranking()