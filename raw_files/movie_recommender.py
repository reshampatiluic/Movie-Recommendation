import joblib
import numpy as np
import time
import sys


def get_recommendations(user_id, model_path="movie_recommender_model.pkl", n=20):
    """
    Get movie recommendations for a user with performance tracking
    """
    # Start total timing
    start_time = time.time()

    # Load the model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []

    # Extract model components
    utility_matrix = model["utility_matrix"]
    similarity_df = model["similarity_df"]
    popular_movies = model.get("popular_movies", [])

    # Recommendation generation
    try:
        # Check if user exists in the model
        if user_id not in utility_matrix.index or user_id not in similarity_df.index:
            # Return popular movies for new users
            recommendations = popular_movies[:n]
            print("\nRecommendations for new user (Popular Movies):")
        else:
            # Get user's rated movies
            user_rated = set(utility_matrix.loc[user_id].dropna().index)

            # Get all movies
            all_movies = set(utility_matrix.columns)

            # Movies to predict
            movies_to_predict = list(all_movies - user_rated)

            # Get user's similarity vector
            user_similarities = similarity_df.loc[user_id].drop(user_id)

            # Efficient prediction
            predictions = []
            for movie in movies_to_predict:
                # Find users who rated this movie
                movie_ratings = utility_matrix[movie].dropna()

                # Get similar users who rated this movie
                similar_raters = user_similarities.index.intersection(
                    movie_ratings.index
                )

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
                additional_movies = [
                    m for m in popular_movies if m not in recommendations
                ]
                recommendations.extend(additional_movies[: n - len(recommendations)])

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
            print(
                f"Warning: Recommendation time exceeded 700 ms (took {inference_time:.2f} ms)"
            )

        return recommendations

    except Exception as e:
        print(f"Error generating recommendations: {e}")
        return []


def main():
    # Check if user ID is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide a user ID as a command-line argument.")
        print("Usage: python movie_recommender.py <user_id>")
        sys.exit(1)

    # Get user ID from command-line argument
    user_id = sys.argv[1]

    # Get recommendations
    get_recommendations(user_id)


if __name__ == "__main__":
    main()
