import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class MovieRecommender:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.item_features = None
        self.user_item_interactions = None
        self.movies_df = None
        self.ratings_df = None
        
    def load_data(self, movies_path, ratings_path, sample_size=None):
        """Load the movies and ratings CSV files"""
        print("Loading data...")
        self.movies_df = pd.read_csv(movies_path)
        
        if sample_size:
            print(f"Sampling {sample_size} ratings for faster processing...")
            self.ratings_df = pd.read_csv(ratings_path, nrows=sample_size)
        else:
            self.ratings_df = pd.read_csv(ratings_path)
        
        # Filter movies to only include those that appear in ratings
        movie_ids_in_ratings = set(self.ratings_df['movieId'].unique())
        self.movies_df = self.movies_df[self.movies_df['movieId'].isin(movie_ids_in_ratings)]
        
        print(f"Movies: {len(self.movies_df)} entries")
        print(f"Ratings: {len(self.ratings_df)} entries")
        print(f"Unique users: {self.ratings_df['userId'].nunique()}")
        print(f"Unique movies: {self.ratings_df['movieId'].nunique()}")
        
    def preprocess_data(self):
        """Preprocess the data for LightFM"""
        print("Preprocessing data...")
        
        # Parse genres and create genre features
        all_genres = set()
        self.movies_df['genre_list'] = self.movies_df['genres'].str.split('|')
        
        for genres in self.movies_df['genre_list']:
            all_genres.update(genres)
        
        print(f"Found {len(all_genres)} unique genres: {sorted(all_genres)}")
        
        # Create item features (movies with their genres)
        item_features = []
        for _, movie in self.movies_df.iterrows():
            movie_id = movie['movieId']
            genres = movie['genre_list']
            # Add movie ID as a feature, plus all its genres
            features = [str(movie_id)] + [f"genre:{genre}" for genre in genres]
            item_features.append((movie_id, features))
        
        return item_features, sorted(all_genres)
    
    def build_dataset(self):
        """Build the LightFM dataset"""
        print("Building LightFM dataset...")
        
        # Get preprocessed data
        item_features, all_genres = self.preprocess_data()
        
        # Get unique users and items
        unique_users = self.ratings_df['userId'].unique()
        unique_items = self.movies_df['movieId'].unique()
        
        # Create all possible item features
        all_item_features = []
        for item_id in unique_items:
            all_item_features.append(str(item_id))
        for genre in all_genres:
            all_item_features.append(f"genre:{genre}")
        
        # Initialize dataset
        self.dataset = Dataset()
        self.dataset.fit(users=unique_users,
                        items=unique_items,
                        item_features=all_item_features)
        
        # Build interactions matrix
        interactions, weights = self.dataset.build_interactions(
            [(row['userId'], row['movieId'], row['rating']) 
             for _, row in self.ratings_df.iterrows()]
        )
        
        # Build item features matrix
        item_features_matrix = self.dataset.build_item_features(item_features)
        
        self.user_item_interactions = interactions
        self.item_features = item_features_matrix
        
        print(f"Interactions matrix shape: {interactions.shape}")
        print(f"Item features matrix shape: {item_features_matrix.shape}")
        
        return interactions, item_features_matrix
    
    def train_model(self, loss='warp', epochs=30):
        """Train the LightFM model"""
        print(f"Training model with {loss} loss for {epochs} epochs...")
        
        if self.user_item_interactions is None:
            self.build_dataset()
        
        # Split data for training and testing
        train, test = random_train_test_split(self.user_item_interactions, 
                                            test_percentage=0.2, 
                                            random_state=42)
        
        # Initialize and train model
        self.model = LightFM(loss=loss, random_state=42)
        self.model.fit(train, 
                      item_features=self.item_features,
                      epochs=epochs, 
                      verbose=True)
        
        # Evaluate model
        train_precision = precision_at_k(self.model, train, 
                                       item_features=self.item_features, k=10).mean()
        test_precision = precision_at_k(self.model, test, 
                                      item_features=self.item_features, k=10).mean()
        
        print(f"Train precision@10: {train_precision:.4f}")
        print(f"Test precision@10: {test_precision:.4f}")
        
        return train, test
    
    def get_recommendations(self, user_id, num_recommendations=10):
        """Get movie recommendations for a specific user"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Get all movie IDs
        all_movies = self.movies_df['movieId'].values
        
        # Get movies the user has already rated
        user_rated_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].values)
        
        # Get predictions for all movies
        user_idx = self.dataset.mapping()[0][user_id]
        movie_indices = [self.dataset.mapping()[2][movie_id] 
                        for movie_id in all_movies 
                        if movie_id in self.dataset.mapping()[2]]
        
        scores = self.model.predict(user_idx, movie_indices, item_features=self.item_features)
        
        # Create movie-score pairs and sort
        movie_scores = list(zip(all_movies[:len(scores)], scores))
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out already rated movies and get top recommendations
        recommendations = []
        for movie_id, score in movie_scores:
            if movie_id not in user_rated_movies and len(recommendations) < num_recommendations:
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                recommendations.append({
                    'movieId': movie_id,
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_score': score
                })
        
        return recommendations
    
    def get_similar_movies_simple(self, movie_id, num_similar=10):
        """Simple approach to find similar movies using item embeddings"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
            
        # Check if movie exists
        movie_mapping = self.dataset.mapping()[2]  # item mapping
        if movie_id not in movie_mapping:
            print(f"Movie {movie_id} not in dataset")
            return []
            
        # Get the movie's embedding index
        target_idx = movie_mapping[movie_id]
        print(f"Target movie {movie_id} has index {target_idx}")
        
        # Get all item embeddings
        embeddings = self.model.item_embeddings
        print(f"Embedding matrix shape: {embeddings.shape}")
        
        # Calculate similarities using cosine similarity
        target_vec = embeddings[target_idx:target_idx+1]  # Keep as 2D
        similarities = cosine_similarity(target_vec, embeddings).flatten()
        
        print(f"Calculated {len(similarities)} similarities")
        print(f"Target movie similarity to itself: {similarities[target_idx]:.4f}")
        print(f"Top 5 similarity scores: {np.sort(similarities)[-5:]}")
        
        # Get indices sorted by similarity (highest first)
        similar_indices = np.argsort(similarities)[::-1]
        
        # Create reverse mapping from index to movie_id
        idx_to_movie = {v: k for k, v in movie_mapping.items()}
        
        results = []
        count = 0
        for i, idx in enumerate(similar_indices):
            if count >= num_similar + 1:  # +1 because we'll skip the target movie
                break
                
            if idx in idx_to_movie:
                movie_id_candidate = idx_to_movie[idx]
                
                # Skip the target movie itself
                if movie_id_candidate == movie_id:
                    print(f"Skipping target movie {movie_id_candidate}")
                    continue
                
                # Find movie info
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id_candidate]
                if len(movie_info) > 0:
                    movie_data = movie_info.iloc[0]
                    results.append({
                        'movieId': movie_id_candidate,
                        'title': movie_data['title'],
                        'genres': movie_data['genres'],
                        'similarity_score': similarities[idx]
                    })
                    count += 1
                    print(f"Added: {movie_data['title']} (score: {similarities[idx]:.4f})")
                else:
                    print(f"Movie {movie_id_candidate} not found in movies dataframe")
        
    def get_similar_movies_correct(self, movie_id, num_similar=10):
        """Correct approach to find similar movies using LightFM representations"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
            
        # Check if movie exists
        movie_mapping = self.dataset.mapping()[2]  # item mapping
        if movie_id not in movie_mapping:
            print(f"Movie {movie_id} not in dataset")
            return []
            
        print(f"Finding movies similar to movie ID {movie_id}...")
        
        # Get movie representations by combining embeddings with item features
        # This is the correct way to get item representations in LightFM
        num_items = len(movie_mapping)
        item_representations = []
        movie_ids_ordered = []
        
        for mid, idx in movie_mapping.items():
            # Get the item representation using LightFM's internal method
            # This combines the item embedding with its features
            item_rep = self.model.get_item_representations(self.item_features)[1][idx]
            item_representations.append(item_rep)
            movie_ids_ordered.append(mid)
        
        item_representations = np.array(item_representations)
        
        # Find the target movie's position in our ordered list
        target_idx = movie_ids_ordered.index(movie_id)
        target_representation = item_representations[target_idx]
        
        print(f"Item representations shape: {item_representations.shape}")
        print(f"Target movie index in representations: {target_idx}")
        
        # Calculate cosine similarities
        similarities = cosine_similarity([target_representation], item_representations)[0]
        
        print(f"Target movie similarity to itself: {similarities[target_idx]:.4f}")
        print(f"Similarity range: {similarities.min():.4f} to {similarities.max():.4f}")
        print(f"Top 5 similarity scores: {np.sort(similarities)[-5:]}")
        
        # Get top similar movies
        similar_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in similar_indices:
            similar_movie_id = movie_ids_ordered[idx]
            
            # Skip the target movie itself
            if similar_movie_id == movie_id:
                print(f"Skipping target movie {similar_movie_id}")
                continue
                
            # Find movie info
            movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
            if len(movie_info) > 0:
                movie_data = movie_info.iloc[0]
                results.append({
                    'movieId': similar_movie_id,
                    'title': movie_data['title'],
                    'genres': movie_data['genres'],
                    'similarity_score': similarities[idx]
                })
                print(f"Added: {movie_data['title']} (score: {similarities[idx]:.4f})")
                
                if len(results) >= num_similar:
                    break
        
        return results
    
    def get_available_movies(self, limit=10):
        """Get a list of available movie IDs and titles"""
        available_movies = []
        for movie_id in list(self.dataset.mapping()[2].keys())[:limit]:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            if len(movie_info) > 0:
                available_movies.append({
                    'movieId': movie_id,
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres']
                })
        return available_movies

# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load data (replace with your file paths)
    # For testing with large dataset, use sample_size parameter
    recommender.load_data('BigMovieData/ml-32m/movies.csv', 'Research/ratings.csv', sample_size=100000)
    
    # Train the model
    recommender.train_model(loss='warp', epochs=20)
    
    # Get recommendations for user 1
    print("\n" + "="*50)
    print("RECOMMENDATIONS FOR USER 1:")
    print("="*50)
    recommendations = recommender.get_recommendations(user_id=1, num_recommendations=5)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Genres: {rec['genres']}")
        print(f"   Predicted Score: {rec['predicted_score']:.3f}")
        print()
    
    # Get available movies to find a valid movie ID
    print("\n" + "="*50)
    print("AVAILABLE MOVIES IN DATASET:")
    print("="*50)
    available_movies = recommender.get_available_movies(10)
    for movie in available_movies:
        print(f"ID: {movie['movieId']} - {movie['title']}")
    
    # Use the first available movie for similarity search
    if available_movies:
        sample_movie_id = available_movies[0]['movieId']
        sample_movie_title = available_movies[0]['title']
        
        print("\n" + "="*50)
        print(f"MOVIES SIMILAR TO {sample_movie_title} (Corrected Method):")
        print("="*50)
        try:
            similar_movies = recommender.get_similar_movies_correct(movie_id=sample_movie_id, num_similar=5)
            if similar_movies and len(similar_movies) > 0:
                for i, movie in enumerate(similar_movies, 1):
                    print(f"{i}. {movie['title']}")
                    print(f"   Genres: {movie['genres']}")
                    print(f"   Similarity Score: {movie['similarity_score']:.3f}")
                    print()
            else:
                print("No similar movies found.")
        except Exception as e:
            print(f"Error finding similar movies: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No available movies found in dataset.")