import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data import load_data

def load_final_model(model_filename="trained_model.pkl"):
    """
    Loads the trained SVD++ model from pickle.
    """
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    return model

def compute_hit_rate(model, train_df, test_df, k=20, rating_threshold=4.0):
    """
    Computes the Hit Rate as the percentage of users in the test set for whom at least one of the top-k
    recommended items is relevant (i.e., has a rating >= rating_threshold).

    For each user in the test set:
      1. Relevant items: items in the test set with rating >= rating_threshold.
      2. Known items: items the user has interacted with in the training set.
      3. Generate predictions for all items not in the user's training set.
      4. Rank these items by predicted rating and select the top k.
      5. Count a "hit" if at least one of the top-k items is in the set of relevant items.
    
    Returns:
      The hit rate as a percentage.
    """
    # All unique items from both train and test (to simulate candidate items)
    all_items = pd.concat([train_df, test_df])["movie_id"].unique()
    
    # Group train and test data by user
    train_groups = train_df.groupby("user_id")
    test_groups = test_df.groupby("user_id")
    
    hit_users = 0
    total_users = 0

    # Iterate over each user in the test set
    for user_id, test_group in test_groups:
        # Define relevant items (those rated >= rating_threshold in the test set)
        relevant_items = set(test_group.loc[test_group["rating"] >= rating_threshold, "movie_id"].unique())
        
        # Get items the user already interacted with in the training set
        if user_id in train_groups.groups:
            train_items = set(train_groups.get_group(user_id)["movie_id"].tolist())
        else:
            train_items = set()
        
        # Predict for all candidate items not in the training set (simulate unseen items)
        predictions = []
        for item_id in all_items:
            if item_id not in train_items:
                pred = model.predict(user_id, item_id).est
                predictions.append((item_id, pred))
                
        if not predictions:
            continue
        
        # Sort predictions by estimated rating in descending order and select top k items
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [p[0] for p in predictions[:k]]
        
        total_users += 1
        # If at least one recommended item is relevant, count it as a hit
        if any(item in relevant_items for item in top_k_items):
            hit_users += 1

    hit_rate = hit_users / float(total_users) if total_users > 0 else 0.0
    return hit_rate

def main():
    # Load the full dataset using your data.py
    df_all = load_data("data/final_processed_data.csv")
    
    # Sample a subset to keep the evaluation manageable; here we sample 2000 rows
    sample_size = 2000
    if len(df_all) > sample_size:
        df_all = df_all.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows from the full dataset for offline evaluation.")
    else:
        print("Dataset is smaller than or equal to 2000 rows; using the full dataset.")
    
    # Split the sampled data into train and test sets (e.g., 80/20 split)
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=42)
    
    # Load the final trained model (assumes the model was trained and saved as 'trained_model.pkl')
    svd_model = load_final_model("trained_models/trained_model.pkl")
    
    # Compute Hit Rate with top-20 recommendations using the specified rating threshold
    k = 20
    rating_threshold = 2.0
    hit_rate = compute_hit_rate(svd_model, train_df, test_df, k=k, rating_threshold=rating_threshold)
    
    print(f"\nHit Rate@{k} : {hit_rate*100:.2f}%")

if __name__ == "__main__":
    main()
