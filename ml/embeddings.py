# ml/embeddings.py
"""
Generate and manage embeddings for posts and users.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import faiss
import pickle
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.app.database import SessionLocal
from backend.app.models import Post, User, Interaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manage embeddings for posts and users."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.post_index = None
        self.post_id_map = {}  # Map index to post_id
        
    def generate_post_embeddings(self, batch_size: int = 32):
        """Generate embeddings for all posts in database."""
        db = SessionLocal()
        
        try:
            # Fetch all posts
            posts = db.query(Post).all()
            logger.info(f"Generating embeddings for {len(posts)} posts...")
            
            # Prepare texts for embedding
            texts = []
            post_ids = []
            
            for post in posts:
                # Combine title and content for embedding
                text = f"{post.title}. {post.content[:500] if post.content else ''}"
                texts.append(text)
                post_ids.append(post.post_id)
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Update posts with embeddings
            for post, embedding in zip(posts, embeddings):
                post.embedding = embedding.tolist()
            
            db.commit()
            logger.info(f"✅ Generated embeddings for {len(posts)} posts")
            
            # Build FAISS index for fast similarity search
            self._build_faiss_index(embeddings, post_ids)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _build_faiss_index(self, embeddings: np.ndarray, post_ids: List[str]):
        """Build FAISS index for fast similarity search."""
        logger.info("Building FAISS index...")
        
        # Create index
        self.post_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine similarity for normalized vectors
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.post_index.add(embeddings)
        
        # Create mapping
        self.post_id_map = {i: post_id for i, post_id in enumerate(post_ids)}
        
        logger.info(f"✅ Built FAISS index with {len(post_ids)} posts")
        
        # Save index
        self.save_index('data/post_index.faiss')
    
    def save_index(self, filepath: str):
        """Save FAISS index and mappings."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.post_index, filepath)
        
        # Save mappings
        with open(filepath.replace('.faiss', '_map.pkl'), 'wb') as f:
            pickle.dump(self.post_id_map, f)
        
        logger.info(f"Saved index to {filepath}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and mappings."""
        if os.path.exists(filepath):
            self.post_index = faiss.read_index(filepath)
            
            map_file = filepath.replace('.faiss', '_map.pkl')
            if os.path.exists(map_file):
                with open(map_file, 'rb') as f:
                    self.post_id_map = pickle.load(f)
            
            logger.info(f"Loaded index from {filepath}")
            return True
        return False
    
    def find_similar_posts(self, query_embedding: np.ndarray, k: int = 10) -> List[tuple]:
        """Find k most similar posts to query embedding."""
        if self.post_index is None:
            logger.warning("Index not built. Loading or building...")
            if not self.load_index('data/post_index.faiss'):
                self.generate_post_embeddings()
        
        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.post_index.search(query, k)
        
        # Map indices to post IDs
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx in self.post_id_map:
                results.append((self.post_id_map[idx], float(score)))
        
        return results
    
    def generate_user_embedding(self, user_id: str) -> np.ndarray:
        """Generate user embedding based on interaction history."""
        db = SessionLocal()
        
        try:
            # Get user's liked posts
            liked_interactions = db.query(Interaction).filter(
                Interaction.user_id == user_id,
                Interaction.interaction_type == 'like'
            ).all()
            
            if not liked_interactions:
                # Return average embedding for new users
                logger.info(f"User {user_id} has no likes, returning average embedding")
                all_posts = db.query(Post).limit(100).all()
                embeddings = [post.embedding for post in all_posts if post.embedding]
                if embeddings:
                    return np.mean(embeddings, axis=0)
                else:
                    return np.zeros(self.embedding_dim)
            
            # Get embeddings of liked posts
            liked_embeddings = []
            for interaction in liked_interactions:
                post = db.query(Post).filter_by(post_id=interaction.post_id).first()
                if post and post.embedding:
                    # Weight by time spent (engagement)
                    weight = min(interaction.time_spent / 30.0, 2.0)  # Cap at 2x weight
                    liked_embeddings.append(np.array(post.embedding) * weight)
            
            # Get disliked posts for negative signal
            disliked_interactions = db.query(Interaction).filter(
                Interaction.user_id == user_id,
                Interaction.interaction_type == 'dislike'
            ).all()
            
            disliked_embeddings = []
            for interaction in disliked_interactions:
                post = db.query(Post).filter_by(post_id=interaction.post_id).first()
                if post and post.embedding:
                    disliked_embeddings.append(np.array(post.embedding))
            
            # Compute user embedding
            user_embedding = np.zeros(self.embedding_dim)
            
            if liked_embeddings:
                # Positive signal: average of liked posts
                user_embedding += np.mean(liked_embeddings, axis=0)
            
            if disliked_embeddings:
                # Negative signal: move away from disliked posts
                user_embedding -= 0.3 * np.mean(disliked_embeddings, axis=0)
            
            # Normalize
            norm = np.linalg.norm(user_embedding)
            if norm > 0:
                user_embedding = user_embedding / norm
            
            # Update user in database
            user = db.query(User).filter_by(user_id=user_id).first()
            if user:
                user.user_embedding = user_embedding.tolist()
                db.commit()
            
            return user_embedding
            
        finally:
            db.close()
    
    def update_user_embedding_online(self, user_id: str, post_id: str, 
                                    interaction_type: str, learning_rate: float = 0.1):
        """Update user embedding in real-time based on new interaction."""
        db = SessionLocal()
        
        try:
            user = db.query(User).filter_by(user_id=user_id).first()
            post = db.query(Post).filter_by(post_id=post_id).first()
            
            if not user or not post or not post.embedding:
                return
            
            # Get current user embedding
            if user.user_embedding:
                user_embedding = np.array(user.user_embedding)
            else:
                user_embedding = np.zeros(self.embedding_dim)
            
            post_embedding = np.array(post.embedding)
            
            # Update based on interaction
            if interaction_type == 'like':
                # Move towards liked content
                user_embedding = (1 - learning_rate) * user_embedding + learning_rate * post_embedding
            elif interaction_type == 'dislike':
                # Move away from disliked content
                user_embedding = user_embedding - learning_rate * 0.5 * post_embedding
                
            # Normalize
            norm = np.linalg.norm(user_embedding)
            if norm > 0:
                user_embedding = user_embedding / norm
            
            # Save updated embedding
            user.user_embedding = user_embedding.tolist()
            db.commit()
            
        finally:
            db.close()


def main():
    """Test the embedding manager."""
    manager = EmbeddingManager()
    
    # Generate embeddings for all posts
    manager.generate_post_embeddings()
    
    # Test similarity search
    db = SessionLocal()
    sample_post = db.query(Post).first()
    
    if sample_post and sample_post.embedding:
        print(f"\n🔍 Finding posts similar to: {sample_post.title[:50]}...")
        similar = manager.find_similar_posts(np.array(sample_post.embedding), k=5)
        
        print("\nSimilar posts:")
        for post_id, score in similar:
            post = db.query(Post).filter_by(post_id=post_id).first()
            if post:
                print(f"  - {score:.3f}: {post.title[:60]}...")
    
    # Generate user embeddings
    print("\n👤 Generating user embeddings...")
    users = db.query(User).all()
    for user in users:
        embedding = manager.generate_user_embedding(user.user_id)
        print(f"  - {user.user_id}: embedding shape {embedding.shape}")
    
    db.close()
    print("\n✅ Embedding pipeline ready!")


if __name__ == "__main__":
    main()