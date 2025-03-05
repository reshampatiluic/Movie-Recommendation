# inference.py
import time
from train import load_data
import pickle

def recommend_movies(user_id, model, df_all, top_n=20):
    all_movies = df_all["movie_id"].unique()
    watched_movies = df_all[df_all["user_id"] == user_id]["movie_id"].tolist()
    
    inference_start = time.time()
    predictions = [
        model.predict(user_id, movie)
        for movie in all_movies if movie not in watched_movies
    ]
    predictions.sort(key=lambda x: x.est, reverse=True)
    recommended = [pred.iid for pred in predictions[:top_n]]
    inference_time = time.time() - inference_start
    print(f"Inference time for user {user_id}: {inference_time:.4f} seconds")
    return recommended, inference_time

def load_model(model_filename="trained_model.pkl"):
    
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    return model
