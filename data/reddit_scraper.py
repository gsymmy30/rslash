# data/reddit_scraper.py
"""
Reddit scraper for fetching posts and building our dataset.
"""

import praw
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditScraper:
    """Scrape Reddit posts for recommendation system."""
    
    def __init__(self):
        """Initialize Reddit API connection."""
        self.reddit = praw.Reddit(
            client_id=settings.REDDIT_CLIENT_ID,
            client_secret=settings.REDDIT_CLIENT_SECRET,
            user_agent=settings.REDDIT_USER_AGENT
        )
        
        # Popular subreddits across different categories
        self.default_subreddits = [
            # Technology
            'programming', 'python', 'javascript', 'machinelearning', 'datascience',
            'webdev', 'technology', 'gadgets', 'android', 'apple',
            
            # Entertainment
            'funny', 'memes', 'videos', 'gifs', 'movies', 'television',
            'netflix', 'music', 'hiphopheads', 'gaming',
            
            # Discussion
            'askreddit', 'todayilearned', 'explainlikeimfive', 'showerthoughts',
            'unpopularopinion', 'changemyview', 'nostupidquestions',
            
            # Interests
            'books', 'cooking', 'fitness', 'personalfinance', 'cryptocurrency',
            'wallstreetbets', 'sports', 'nba', 'soccer', 'formula1',
            
            # Lifestyle
            'lifeprotips', 'getmotivated', 'food', 'travel', 'photography',
            'art', 'diy', 'fashion', 'malefashionadvice', 'sneakers'
        ]
        
    def fetch_posts(self, 
                   subreddit_name: str, 
                   limit: int = 100,
                   time_filter: str = 'week') -> List[Dict]:
        """
        Fetch posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Number of posts to fetch
            time_filter: Time filter for top posts ('day', 'week', 'month', 'year', 'all')
        """
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Fetch from different sorting methods for diversity
            fetch_methods = [
                ('hot', subreddit.hot(limit=limit//3)),
                ('top', subreddit.top(time_filter=time_filter, limit=limit//3)),
                ('new', subreddit.new(limit=limit//3))
            ]
            
            for method_name, submissions in fetch_methods:
                for submission in submissions:
                    # Skip stickied posts (usually rules/megathreads)
                    if submission.stickied:
                        continue
                    
                    post_data = self._extract_post_data(submission, method_name)
                    posts.append(post_data)
                    
        except Exception as e:
            logger.error(f"Error fetching from r/{subreddit_name}: {e}")
            
        return posts
    
    def _extract_post_data(self, submission, fetch_method: str = 'hot') -> Dict:
        """Extract relevant data from a Reddit submission."""
        
        # Calculate engagement metrics
        total_engagement = submission.score + submission.num_comments
        engagement_rate = submission.num_comments / (submission.score + 1)  # Comments per upvote
        
        # Estimate read time based on content length
        text_length = len(submission.selftext) if submission.selftext else 0
        title_length = len(submission.title)
        estimated_read_time = (text_length + title_length) / 200  # Assuming 200 words per minute
        
        return {
            'post_id': submission.id,
            'title': submission.title,
            'content': submission.selftext[:1000] if submission.selftext else '',  # Truncate long posts
            'subreddit': submission.subreddit.display_name,
            'author': str(submission.author) if submission.author else '[deleted]',
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'num_comments': submission.num_comments,
            'created_utc': submission.created_utc,
            'url': submission.url,
            'permalink': f"https://reddit.com{submission.permalink}",
            'is_video': submission.is_video,
            'is_self': submission.is_self,  # Text post vs link
            'nsfw': submission.over_18,
            'awards': len(submission.all_awardings),
            
            # Derived features
            'fetch_method': fetch_method,
            'total_engagement': total_engagement,
            'engagement_rate': engagement_rate,
            'estimated_read_time': estimated_read_time,
            'post_age_hours': (time.time() - submission.created_utc) / 3600,
            
            # Content features
            'text_length': text_length,
            'title_length': title_length,
            'has_content': bool(submission.selftext),
            
            # For embeddings
            'embedding_text': f"{submission.title}. {submission.selftext[:500] if submission.selftext else ''}"
        }
    
    def fetch_diverse_posts(self, 
                           num_posts_per_subreddit: int = 100,
                           subreddits: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch diverse posts from multiple subreddits.
        
        Args:
            num_posts_per_subreddit: Number of posts to fetch from each subreddit
            subreddits: List of subreddits to fetch from (uses defaults if None)
        """
        subreddits = subreddits or self.default_subreddits
        all_posts = []
        
        logger.info(f"Fetching posts from {len(subreddits)} subreddits...")
        
        for subreddit_name in tqdm(subreddits, desc="Fetching subreddits"):
            posts = self.fetch_posts(subreddit_name, limit=num_posts_per_subreddit)
            all_posts.extend(posts)
            
            # Be nice to Reddit's API
            time.sleep(0.5)
        
        logger.info(f"Fetched {len(all_posts)} total posts")
        
        # Remove duplicates
        unique_posts = {post['post_id']: post for post in all_posts}
        all_posts = list(unique_posts.values())
        
        logger.info(f"After deduplication: {len(all_posts)} unique posts")
        
        return all_posts
    
    def create_synthetic_interactions(self, posts: List[Dict]) -> List[Dict]:
        """
        Create synthetic user interactions based on post metrics.
        This helps bootstrap our recommendation system.
        """
        interactions = []
        
        # Create synthetic users with different preferences
        user_profiles = [
            {'user_id': 'tech_enthusiast', 'interests': ['programming', 'technology', 'machinelearning', 'gadgets']},
            {'user_id': 'entertainment_lover', 'interests': ['funny', 'memes', 'videos', 'movies', 'gaming']},
            {'user_id': 'knowledge_seeker', 'interests': ['todayilearned', 'explainlikeimfive', 'askreddit', 'books']},
            {'user_id': 'sports_fan', 'interests': ['sports', 'nba', 'soccer', 'formula1', 'fitness']},
            {'user_id': 'lifestyle_guru', 'interests': ['cooking', 'food', 'travel', 'photography', 'fashion']},
            {'user_id': 'finance_focused', 'interests': ['personalfinance', 'wallstreetbets', 'cryptocurrency']},
            {'user_id': 'creative_soul', 'interests': ['art', 'diy', 'photography', 'music']},
            {'user_id': 'casual_browser', 'interests': []},  # Likes popular content
        ]
        
        for post in posts:
            # High-scoring posts get positive interactions
            if post['score'] > 100:
                # Multiple users might like popular posts
                for user in user_profiles:
                    # Check if post matches user interests
                    if post['subreddit'] in user['interests'] or (not user['interests'] and post['score'] > 1000):
                        interaction = {
                            'user_id': user['user_id'],
                            'post_id': post['post_id'],
                            'interaction_type': 'like',
                            'time_spent': min(post['estimated_read_time'] * 60 + 20, 300),  # Convert to seconds
                            'timestamp': time.time() - (post['post_age_hours'] * 3600 * 0.5),  # Synthetic timestamp
                        }
                        interactions.append(interaction)
            
            # Low upvote ratio posts get negative interactions
            if post['upvote_ratio'] < 0.7 and post['score'] < 50:
                interaction = {
                    'user_id': 'casual_browser',
                    'post_id': post['post_id'],
                    'interaction_type': 'dislike',
                    'time_spent': 2,  # Quick skip
                    'timestamp': time.time() - (post['post_age_hours'] * 3600 * 0.3),
                }
                interactions.append(interaction)
        
        logger.info(f"Created {len(interactions)} synthetic interactions")
        return interactions
    
    def save_data(self, posts: List[Dict], filepath: str = 'data/reddit_posts.json'):
        """Save scraped posts to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(posts, f, indent=2)
        
        logger.info(f"Saved {len(posts)} posts to {filepath}")
    
    def load_data(self, filepath: str = 'data/reddit_posts.json') -> List[Dict]:
        """Load posts from JSON file."""
        with open(filepath, 'r') as f:
            posts = json.load(f)
        
        logger.info(f"Loaded {len(posts)} posts from {filepath}")
        return posts


def main():
    """Main function to test the scraper."""
    scraper = RedditScraper()
    
    # Test connection
    print("Testing Reddit connection...")
    test_posts = scraper.fetch_posts('python', limit=5)
    
    if test_posts:
        print(f"✅ Successfully fetched {len(test_posts)} posts from r/python")
        print(f"Sample post: {test_posts[0]['title'][:80]}...")
        
        # Fetch diverse dataset
        print("\nFetching diverse dataset...")
        print("This will take a few minutes...")
        
        # Start with fewer subreddits for testing
        test_subreddits = [
            'programming', 'funny', 'todayilearned', 
            'gaming', 'movies', 'fitness'
        ]
        
        all_posts = scraper.fetch_diverse_posts(
            num_posts_per_subreddit=50,
            subreddits=test_subreddits
        )
        
        # Save the data
        scraper.save_data(all_posts)
        
        # Create synthetic interactions
        interactions = scraper.create_synthetic_interactions(all_posts)
        scraper.save_data(interactions, 'data/synthetic_interactions.json')
        
        print(f"\n✅ Data pipeline complete!")
        print(f"   - Posts: {len(all_posts)}")
        print(f"   - Interactions: {len(interactions)}")
        print(f"   - Saved to: data/reddit_posts.json")
        
    else:
        print("❌ Failed to fetch posts. Check your Reddit API credentials.")


if __name__ == "__main__":
    main()