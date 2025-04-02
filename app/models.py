import os
import time
import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from pathlib import Path


class Base:
    _instances = {}

    def __new__(cls):
        # Ensures singleton pattern
        if cls not in cls._instances:
            cls._instances[cls] = super(Base, cls).__new__(cls)
        return cls._instances[cls]

    def __init__(self):
        self.saved_model_path = None
        self.dataset_path = None
        self.model = None
        self.df = None

    def load_model(self):
        with open(self.saved_model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def load_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def recommend(self, user_id, n=20):
        raise NotImplementedError


class SVDRecommender(Base):

    def __init__(self):
        super().__init__()
        root =  Path(__file__).resolve().parent.parent # Resolves to Movie-Recommendation/
        self.saved_model_path = root / "trained_models" / "trained_model.pkl"
        self.dataset_path = root / "data" / "final_processed_data.csv"
        self.model = self.load_model()
        self.df = self.load_data()

    def load_data(self):
        df_all = pd.read_csv(self.dataset_path)
        df_all.rename(columns={
            "User_ID": "user_id",
            "Movie_Name": "movie_id",
            "Rating": "rating"
        }, inplace=True)

        df_all["user_id"] = df_all["user_id"].astype(int)

        df_all["rating"] = pd.to_numeric(df_all["rating"], errors="coerce")

        movie_avg = df_all.groupby("movie_id")["rating"].transform("mean")
        df_all["rating"] = df_all["rating"].fillna(movie_avg)

        global_avg = df_all["rating"].mean()
        df_all["rating"] = df_all["rating"].fillna(global_avg)

        print(f"Loaded {len(df_all)} rows from {self.dataset_path}.")
        return df_all

    def stratified_cross_validate(self, model_class, df, n_splits=5, reader=Reader(rating_scale=(1, 5))):

        y_discrete = df["rating"].round().astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        X = df.index.values

        rmses = []
        maes = []
        fit_times = []
        test_times = []

        for train_index, test_index in skf.split(X, y_discrete):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            train_data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]], reader)
            trainset = train_data.build_full_trainset()

            model = model_class()

            # Measure fit time for this fold
            fold_fit_start = time.time()
            model.fit(trainset)
            fold_fit_time = time.time() - fold_fit_start
            fit_times.append(fold_fit_time)

            # Measure test time for this fold and compute predictions
            fold_test_start = time.time()
            test_preds = []
            for _, row in test_df.iterrows():
                uid = row["user_id"]
                iid = row["movie_id"]
                true_rating = row["rating"]
                pred = model.predict(uid, iid).est
                test_preds.append((true_rating, pred))
            fold_test_time = time.time() - fold_test_start
            test_times.append(fold_test_time)

            # Compute RMSE and MAE
            se = [(true - pred) ** 2 for true, pred in test_preds]
            ae = [abs(true - pred) for true, pred in test_preds]
            rmse = np.sqrt(np.mean(se))
            mae = np.mean(ae)
            rmses.append(rmse)
            maes.append(mae)

        mean_rmse = np.mean(rmses)
        mean_mae = np.mean(maes)
        mean_fit_time = np.mean(fit_times)
        mean_test_time = np.mean(test_times)
        return mean_rmse, mean_mae, mean_fit_time, mean_test_time

    def save_model(self):
        with open(self.saved_model_path, "wb") as f:
            pickle.dump(self.model, f)

    def train(self):
        reader = Reader(rating_scale=(1, 5))

        print("Performing stratified cross-validation with 5 folds...")
        mean_rmse, mean_mae, mean_fit_time, mean_test_time = self.stratified_cross_validate(SVD, self.df, n_splits=5,
                                                                                       reader=reader)
        print(f"RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}")
        print(
            f"Average Fit Time per fold: {mean_fit_time:.4f} sec, Average Test Time per fold: {mean_test_time:.4f} sec")

        start_time = time.time()
        data = Dataset.load_from_df(self.df[["user_id", "movie_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        model = SVD()
        model.fit(trainset)
        training_time = time.time() - start_time
        print(f"Training time (full training set): {training_time:.4f} seconds")

        self.save_model()

        model_size = os.path.getsize(self.saved_model_path) / 1024.0
        print(f"Model saved as {self.saved_model_path}, size: {model_size:.2f} KB")
        print("Model training complete.")
        return model, self.df, training_time, model_size

    def recommend(self, user_id, n=20):
        all_movies = self.df["movie_id"].unique()
        watched_movies = self.df[self.df["user_id"] == user_id]["movie_id"].tolist()

        inference_start = time.time()
        predictions = [
            self.model.predict(user_id, movie)
            for movie in all_movies if movie not in watched_movies
        ]
        predictions.sort(key=lambda x: x.est, reverse=True)
        recommended = [pred.iid for pred in predictions[:n]]
        inference_time = time.time() - inference_start
        print(f"Inference time for user {user_id}: {inference_time:.4f} seconds")
        return recommended, inference_time


class KNNRecommender(Base):
    def __init__(self):
        super().__init__()
        root = Path(__file__).resolve().parent.parent  # Resolves to Movie-Recommendation/
        self.saved_model_path = root / "trained_models" / "movie_recommender_model.pkl"
        self.dataset_path = root / "data" / "final_data.csv"
        self.model = self.load_model()
        self.df = self.load_data()
        self.utility_matrix = None
        self.similarity_df = None
        self.popular_movies = None

    def load_model(self):
        try:
            return joblib.load(self.saved_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def load_data(self):
        """Load data from the CSV format provided"""
        try:
            # Read CSV with custom column names
            df = pd.read_csv(self.dataset_path, names=['Timestamp_x', 'User_ID', 'Movie_Name', 'Timestamp_y', 'Rating'],
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

    @staticmethod
    def create_utility_matrix(df):
        """Create a user-item utility matrix from ratings dataframe"""
        # Create pivot table: rows = users, columns = movies, values = ratings
        utility_matrix = df.pivot_table(index='User_ID', columns='Movie_Name', values='Rating')

        return utility_matrix

    def compute_user_similarity(self, max_users=5000):
        """Compute user similarity matrix using cosine similarity"""
        # Record start time
        start_time = time.time()

        # For large datasets, sample users to speed up computation
        original_size = len(self.utility_matrix)
        if len(self.utility_matrix) > max_users:
            utility_matrix = self.utility_matrix.sample(max_users, random_state=42)
            print(f"Sampled users from {original_size} to {len(utility_matrix)} for similarity computation")

        # Replace NaN with 0 for similarity calculation
        matrix_for_sim = self.utility_matrix.fillna(0)

        # Calculate cosine similarity between users
        similarity = cosine_similarity(matrix_for_sim)

        # Create DataFrame for similarity matrix
        similarity_df = pd.DataFrame(similarity,
                                     index=self.utility_matrix.index,
                                     columns=self.utility_matrix.index)

        # Calculate computation time
        compute_time = time.time() - start_time

        return similarity_df, compute_time

    def predict_rating(self, user_id, movie_name, k=40):
        """Predict rating for a user-movie pair using k nearest neighbors"""
        # If user doesn't exist in training data, return None
        if user_id not in self.utility_matrix.index:
            return None

        # If movie doesn't exist in training data, return None
        if movie_name not in self.utility_matrix.columns:
            return None

        # If user has already rated this movie, return that rating
        if not pd.isna(self.utility_matrix.loc[user_id, movie_name]):
            return self.utility_matrix.loc[user_id, movie_name]

        # Ensure user is in similarity matrix
        if user_id not in self.similarity_df.index:
            return None

        # Get similarity scores for this user compared to all others
        try:
            user_similarities = self.similarity_df.loc[user_id].drop(user_id)
        except KeyError:
            return None

        # Find users who have rated this movie
        movie_ratings = self.utility_matrix[movie_name].dropna()

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

    def evaluate_model(self, test_df):
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
            if (user_id in self.utility_matrix.index and
                    user_id in self.similarity_df.index and
                    movie_name in self.utility_matrix.columns):

                predicted_rating = self.predict_rating(
                    user_id,
                    movie_name,
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

    def get_popular_movies(self, n=20):
        """Get most popular movies based on rating count and average rating"""
        # Calculate average rating and count for each movie
        movie_stats = self.df.groupby('Movie_Name').agg({
            'Rating': ['mean', 'count']
        })

        # Flatten the multi-index columns
        movie_stats.columns = ['avg_rating', 'count']
        movie_stats = movie_stats.reset_index()

        # Sort by rating count and average rating
        movie_stats = movie_stats.sort_values(['count', 'avg_rating'], ascending=[False, False])

        # Return top N movies
        return movie_stats['Movie_Name'].head(n).tolist()

    def calculate_model_size_and_memory(self):
        """Calculate model size and memory usage"""
        # Calculate file size
        model_file_size = os.path.getsize(self.dataset_path) / (1024 * 1024)  # Size in MB

        # Calculate memory usage
        utility_matrix_memory = self.utility_matrix.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        similarity_matrix_memory = self.similarity_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB

        return {
            'Model_File_Size_MB': model_file_size,
            'Utility_Matrix_Memory_MB': utility_matrix_memory,
            'Similarity_Matrix_Memory_MB': similarity_matrix_memory
        }

    def view_model_metrics(self):
        """
        Load and display the model metrics
        """
        try:
            # Load the model using joblib
            model = joblib.load(self.saved_model_path)

            # Extract performance metrics
            metrics = model['performance_metrics']

            # Print comprehensive performance report
            print("\n--- Saved Model Performance Metrics ---")

            print("\n1. Prediction Accuracy:")
            print(f"   RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
            print(f"   MAE: {metrics.get('MAE', 'N/A'):.4f}")
            print(f"   Prediction Coverage: {metrics.get('Prediction_Coverage', 'N/A'):.2f}%")

            print("\n2. Training Cost:")
            print(f"   Total Training Time: {metrics.get('Total_Training_Time', 'N/A'):.4f} seconds")
            print(f"   Similarity Computation Time: {metrics.get('Similarity_Computation_Time', 'N/A'):.4f} seconds")

            print("\n3. Inference Cost:")
            print(f"   Test Time: {metrics.get('Test_Time', 'N/A'):.4f} seconds")
            print(f"   Total Test Samples: {metrics.get('Total_Test_Samples', 'N/A')}")
            print(f"   Successful Predictions: {metrics.get('Total_Predictions', 'N/A')}")

            print("\n4. Model Size and Memory:")
            print(f"   Model File Size: {metrics.get('Model_File_Size_MB', 'N/A'):.2f} MB")
            print(f"   Utility Matrix Memory: {metrics.get('Utility_Matrix_Memory_MB', 'N/A'):.2f} MB")
            print(f"   Similarity Matrix Memory: {metrics.get('Similarity_Matrix_Memory_MB', 'N/A'):.2f} MB")

            return metrics

        except Exception as e:
            print(f"Error loading model metrics: {e}")
            return None

    def save_model(self):
        try:
            # Use joblib for faster serialization
            joblib.dump(self.model, self.saved_model_path, compress=3)
            print(f"Model saved successfully to {self.saved_model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise e


    def train(self):
        """
        Train the recommendation model and save it to a pickle file

        Returns:
        - Trained model dictionary with performance metrics
        """
        # Start overall training timing
        start_total_time = time.time()

        # 1. Load data
        print("Loading data...")
        self.df = self.load_data()
        print(
            f"Loaded {len(self.df)} ratings from {len(self.df['User_ID'].unique())} users on "
            f"{len(self.df['Movie_Name'].unique())} movies")

        # 2. Split data for training and testing
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)

        # 3. Create utility matrix
        print("Creating utility matrix...")
        self.utility_matrix = self.create_utility_matrix(train_df)

        # 4. Compute user similarity
        print("Computing user similarity...")
        self.similarity_df, similarity_time = self.compute_user_similarity()

        # 5. Evaluate model performance
        print("Evaluating model performance...")
        performance_metrics = self.evaluate_model(test_df)

        # 6. Calculate total training time
        total_training_time = time.time() - start_total_time

        # 7. Calculate model size and memory usage
        model_size_metrics = self.calculate_model_size_and_memory()

        # 8. Get popular movies for new users
        self.popular_movies = self.get_popular_movies()

        # 9. Prepare model dictionary
        self.model = {
            'utility_matrix': self.utility_matrix,
            'similarity_df': self.similarity_df,
            'popular_movies': self.popular_movies,
            'performance_metrics': {
                **performance_metrics,
                'Total_Training_Time': total_training_time,
                'Similarity_Computation_Time': similarity_time,
                **model_size_metrics
            }
        }

        # 10. Save model
        self.save_model()

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

    def recommend(self, user_id, n=20):
        """
        Get movie recommendations for a user with performance tracking
        """
        # Start total timing
        start_time = time.time()

        # Extract model components
        self.utility_matrix = self.model['utility_matrix']
        self.similarity_df = self.model['similarity_df']
        self.popular_movies = self.model.get('popular_movies', [])

        # Recommendation generation
        try:
            # Check if user exists in the model
            if user_id not in self.utility_matrix.index or user_id not in self.similarity_df.index:
                # Return popular movies for new users
                recommendations = self.popular_movies[:n]
                print("\nRecommendations for new user (Popular Movies):")
            else:
                # Get user's rated movies
                user_rated = set(self.utility_matrix.loc[user_id].dropna().index)

                # Get all movies
                all_movies = set(self.utility_matrix.columns)

                # Movies to predict
                movies_to_predict = list(all_movies - user_rated)

                # Get user's similarity vector
                user_similarities = self.similarity_df.loc[user_id].drop(user_id)

                # Efficient prediction
                predictions = []
                for movie in movies_to_predict:
                    # Find users who rated this movie
                    movie_ratings = self.utility_matrix[movie].dropna()

                    # Get similar users who rated this movie
                    similar_raters = user_similarities.index.intersection(movie_ratings.index)

                    if len(similar_raters) == 0:
                        continue

                    # Quick weighted average
                    weights = user_similarities.loc[similar_raters]
                    ratings = movie_ratings.loc[similar_raters]

                    if weights.sum() > 0:
                        predicted_rating = np.average(ratings, weights=weights)
                        predictions.append((movie, predicted_rating))

                # Sort and get top recommendations
                predictions.sort(key=lambda x: x[1], reverse=True)
                recommendations = [movie for movie, _ in predictions[:n]]

                # Fill with popular movies if needed
                if len(recommendations) < n:
                    additional_movies = [m for m in self.popular_movies if m not in recommendations]
                    recommendations.extend(additional_movies[:n - len(recommendations)])

                # Ensure exactly n recommendations
                recommendations = recommendations[:n]
                print(f"\nRecommendations for user {user_id}:")

            # Print recommendations
            for i, movie in enumerate(recommendations, 1):
                print(f"{i}. {movie}")

            # Calculate and print recommendation time
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            print(f"\nRecommendation Time: {inference_time:.2f} ms")

            # Warn if recommendation time exceeds 700 ms
            if inference_time > 700:
                print(f"Warning: Recommendation time exceeded 700 ms (took {inference_time:.2f} ms)")

            return recommendations

        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

if __name__ == "__main__":
    svd = SVDRecommender()
    svd_recommended, _ = svd.recommend(user_id=77386)
    print(f"SVD: {svd_recommended}")

    knn = KNNRecommender()
    knn.view_model_metrics()
    knn_recommended = knn.recommend(user_id=77386)
    print(f"KNN: {knn_recommended}")