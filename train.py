# train.py
import time
import os
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import StratifiedKFold
from data import load_data
import pickle

def stratified_cross_validate(model_class, df, n_splits=5, reader=Reader(rating_scale=(1, 5))):
    
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

def train_model(data_path="data/final_processed_data.csv"):
    df_all = load_data(data_path)
    reader = Reader(rating_scale=(1, 5))
    
    print("Performing stratified cross-validation with 5 folds...")
    mean_rmse, mean_mae, mean_fit_time, mean_test_time = stratified_cross_validate(SVD, df_all, n_splits=5, reader=reader)
    print(f"RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}")
    print(f"Average Fit Time per fold: {mean_fit_time:.4f} sec, Average Test Time per fold: {mean_test_time:.4f} sec")
 
    start_time = time.time()
    data = Dataset.load_from_df(df_all[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    training_time = time.time() - start_time
    print(f"Training time (full training set): {training_time:.4f} seconds")
    
    model_filename = "trained_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    model_size = os.path.getsize(model_filename) / 1024.0
    print(f"Model saved as {model_filename}, size: {model_size:.2f} KB")
    print("Model training complete.")
    return model, df_all, training_time, model_size
