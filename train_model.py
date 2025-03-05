import pandas as pd
import numpy as np
import time
import os
import sys
import pickle
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(filepath):
    """Load data from the CSV format provided"""
    try:
        # Read CSV with custom column names
        df = pd.read_csv(filepath, names=['Timestamp_x', 'User_ID', 'Movie_Name', 'Timestamp_y', 'Rating'], 
                        skiprows=1)
        
        # Ensure correct data types
        df['User_ID'] = df['User_ID'].astype(str)
        df['Movie_Name'] = df['Movie_Name'].astype(str)
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        
        # Remove rows with missing ratings
        df = df.dropna(subset=['Rating'])
        
        return df
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise

def create_utility_matrix(df):
    """Create a user-item utility matrix from ratings dataframe"""
    # Create pivot table: rows = users, columns = movies, values = ratings
    utility_matrix = df.pivot_table(index='User_ID', columns='Movie_Name', values='Rating')
    
    return utility_matrix

def compute_user_similarity(utility_matrix, max_users=5000):
    """Compute user similarity matrix using cosine similarity"""
    # Record start time
    start_time = time.time()
    
    # For large datasets, sample users to speed up computation
    original_size = len(utility_matrix)
    if len(utility_matrix) > max_users:
        utility_matrix = utility_matrix.sample(max_users, random_state=42)
        print(f"Sampled users from {original_size} to {len(utility_matrix)} for similarity computation")
    
    # Replace NaN with 0 for similarity calculation
    matrix_for_sim = utility_matrix.fillna(0)
    
    # Calculate cosine similarity between users
    similarity = cosine_similarity(matrix_for_sim)
    
    # Create DataFrame for similarity matrix
    similarity_df = pd.DataFrame(similarity, 
                                 index=utility_matrix.index, 
                                 columns=utility_matrix.index)
    
    # Calculate computation time
    compute_time = time.time() - start_time
    
    return similarity_df, compute_time

def predict_rating(user_id, movie_name, utility_matrix, similarity_df, k=40):
    """Predict rating for a user-movie pair using k nearest neighbors"""
    # If user doesn't exist in training data, return None
    if user_id not in utility_matrix.index:
        return None
    
    # If movie doesn't exist in training data, return None
    if movie_name not in utility_matrix.columns:
        return None
    
    # If user has already rated this movie, return that rating
    if not pd.isna(utility_matrix.loc[user_id, movie_name]):
        return utility_matrix.loc[user_id, movie_name]
    
    # Ensure user is in similarity matrix
    if user_id not in similarity_df.index:
        return None
    
    # Get similarity scores for this user compared to all others
    try:
        user_similarities = similarity_df.loc[user_id].drop(user_id)
    except KeyError:
        return None
    
    # Find users who have rated this movie
    movie_ratings = utility_matrix[movie_name].dropna()
    
    # Get the intersection of users who are similar and have rated this movie
    common_users = user_similarities.index.intersection(movie_ratings.index)
    
    # If no common users, return None
    if len(common_users) == 0:
        return None
    
    # Keep only the k most similar users
    if len(common_users) > k:
        similarities = user_similarities.loc[common_users]
        common_users = similarities.nlargest(k).index
    
    # Calculate weighted average rating
    weights = user_similarities.loc[common_users]
    ratings = movie_ratings.loc[common_users]
    
    if weights.sum() == 0:
        return None
    
    weighted_avg = (weights * ratings).sum() / weights.sum()
    
    return weighted_avg

def evaluate_model(utility_matrix, similarity_df, test_df):
    """
    Evaluate model performance with comprehensive metrics
    
    Returns:
    - Dictionary of performance metrics
    """
    start_test_time = time.time()
    
    # Prepare for testing
    actual_ratings = []
    predicted_ratings = []
    
    # Make predictions on test set
    for _, row in test_df.iterrows():
        user_id = row['User_ID']
        movie_name = row['Movie_Name']
        
        # Check if user and movie exist in training data
        if (user_id in utility_matrix.index and 
            user_id in similarity_df.index and 
            movie_name in utility_matrix.columns):
            
            predicted_rating = predict_rating(
                user_id, 
                movie_name, 
                utility_matrix, 
                similarity_df
            )
            
            if predicted_rating is not None:
                actual_ratings.append(row['Rating'])
                predicted_ratings.append(predicted_rating)
    
    # Calculate test metrics
    test_time = time.time() - start_test_time
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings)) if predicted_ratings else None
    mae = mean_absolute_error(actual_ratings, predicted_ratings) if predicted_ratings else None
    
    # Calculate prediction accuracy metrics
    total_test_samples = len(test_df)
    successful_predictions = len(predicted_ratings)
    prediction_coverage = (successful_predictions / total_test_samples) * 100 if total_test_samples > 0 else 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'Test_Time': test_time,
        'Total_Predictions': successful_predictions,
        'Total_Test_Samples': total_test_samples,
        'Prediction_Coverage': prediction_coverage
    }

def get_popular_movies(df, n=20):
    """Get most popular movies based on rating count and average rating"""
    # Calculate average rating and count for each movie
    movie_stats = df.groupby('Movie_Name').agg({
        'Rating': ['mean', 'count']
    })
    
    # Flatten the multi-index columns
    movie_stats.columns = ['avg_rating', 'count']
    movie_stats = movie_stats.reset_index()
    
    # Sort by rating count and average rating
    movie_stats = movie_stats.sort_values(['count', 'avg_rating'], ascending=[False, False])
    
    # Return top N movies
    return movie_stats['Movie_Name'].head(n).tolist()

def calculate_model_size_and_memory(df, utility_matrix, similarity_df):
    """Calculate model size and memory usage"""
    # Calculate file size
    model_file_size = os.path.getsize('final_data.csv') / (1024 * 1024)  # Size in MB
    
    # Calculate memory usage
    utility_matrix_memory = utility_matrix.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    similarity_matrix_memory = similarity_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    return {
        'Model_File_Size_MB': model_file_size,
        'Utility_Matrix_Memory_MB': utility_matrix_memory,
        'Similarity_Matrix_Memory_MB': similarity_matrix_memory
    }

def train_model(filepath, model_path='movie_recommender_model.pkl'):
    """
    Train the recommendation model and save it to a pickle file
    
    Returns:
    - Trained model dictionary with performance metrics
    """
    # Start overall training timing
    start_total_time = time.time()
    
    # 1. Load data
    print("Loading data...")
    df = load_data(filepath)
    print(f"Loaded {len(df)} ratings from {len(df['User_ID'].unique())} users on {len(df['Movie_Name'].unique())} movies")
    
    # 2. Split data for training and testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 3. Create utility matrix
    print("Creating utility matrix...")
    utility_matrix = create_utility_matrix(train_df)
    
    # 4. Compute user similarity
    print("Computing user similarity...")
    similarity_df, similarity_time = compute_user_similarity(utility_matrix)
    
    # 5. Evaluate model performance
    print("Evaluating model performance...")
    performance_metrics = evaluate_model(utility_matrix, similarity_df, test_df)
    
    # 6. Calculate total training time
    total_training_time = time.time() - start_total_time
    
    # 7. Calculate model size and memory usage
    model_size_metrics = calculate_model_size_and_memory(df, utility_matrix, similarity_df)
    
    # 8. Get popular movies for new users
    popular_movies = get_popular_movies(df)
    
    # 9. Prepare model dictionary
    model = {
        'utility_matrix': utility_matrix,
        'similarity_df': similarity_df,
        'popular_movies': popular_movies,
        'performance_metrics': {
            **performance_metrics,
            'Total_Training_Time': total_training_time,
            'Similarity_Computation_Time': similarity_time,
            **model_size_metrics
        }
    }
    
    # 10. Save model
    try:
        # Use joblib for faster serialization
        joblib.dump(model, model_path, compress=3)
        print(f"Model saved successfully to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return None
    
    # 11. Print comprehensive performance report
    print("\n--- Comprehensive Model Performance Report ---")
    print("\n1. Prediction Accuracy:")
    print(f"   RMSE: {performance_metrics['RMSE']:.4f}")
    print(f"   MAE: {performance_metrics['MAE']:.4f}")
    print(f"   Prediction Coverage: {performance_metrics['Prediction_Coverage']:.2f}%")
    
    print("\n2. Training Cost:")
    print(f"   Total Training Time: {total_training_time:.4f} seconds")
    print(f"   Similarity Computation Time: {similarity_time:.4f} seconds")
    
    print("\n3. Inference Cost:")
    print(f"   Test Time: {performance_metrics['Test_Time']:.4f} seconds")
    print(f"   Total Test Samples: {performance_metrics['Total_Test_Samples']}")
    print(f"   Successful Predictions: {performance_metrics['Total_Predictions']}")
    
    print("\n4. Model Size and Memory:")
    print(f"   Model File Size: {model_size_metrics['Model_File_Size_MB']:.2f} MB")
    print(f"   Utility Matrix Memory: {model_size_metrics['Utility_Matrix_Memory_MB']:.2f} MB")
    print(f"   Similarity Matrix Memory: {model_size_metrics['Similarity_Matrix_Memory_MB']:.2f} MB")
    
    return model

def main():
    # Specify the path to your CSV file
    filepath = "final_data.csv"
    
    # Train and save the model
    model = train_model(filepath)

if __name__ == "__main__":
    main()