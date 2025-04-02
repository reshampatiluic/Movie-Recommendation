import time
import os
import numpy as np
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import GridSearchCV
from data import load_data
import pickle

def tune_svdpp_hyperparameters(df, reader, cv=3):
    """
    Tune SVD++ hyperparameters using GridSearchCV.
    
    Parameters:
      df (DataFrame): The full dataset.
      reader (Reader): A Surprise Reader object.
      cv (int): Number of cross-validation folds for grid search.
      
    Returns:
      dict: The best hyperparameters based on RMSE.
    """
    param_grid = {
        'n_factors': [50, 100, 150],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.05, 0.1]
    }
    data = Dataset.load_from_df(df[["user_id", "movie_id", "rating"]], reader)
    gs = GridSearchCV(SVDpp, param_grid, measures=['rmse', 'mae'], cv=cv, n_jobs=-1)
    gs.fit(data)
    
    best_params = gs.best_params['rmse']
    best_rmse = gs.best_score['rmse']
    
    print("Hyperparameter Tuning Results (SVD++):")
    print("Best RMSE: {:.4f}".format(best_rmse))
    print("Best Hyperparameters:", best_params)
    
    return best_params

def train_model(data_path="data/final_processed_data.csv"):
    # Load the full dataset
    df_all = load_data(data_path)
    reader = Reader(rating_scale=(1, 5))
    
    print("Tuning hyperparameters for SVD++ using GridSearchCV...")
    best_params = tune_svdpp_hyperparameters(df_all, reader, cv=3)
    
    print("Training final SVD++ model using tuned hyperparameters...")
    start_time = time.time()
    data = Dataset.load_from_df(df_all[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    
    # Create and train the SVD++ model using the best hyperparameters
    model = SVDpp(n_factors=best_params['n_factors'],
                  lr_all=best_params['lr_all'],
                  reg_all=best_params['reg_all'])
    model.fit(trainset)
    training_time = time.time() - start_time
    print("Final SVD++ model trained in {:.4f} seconds".format(training_time))
    
    # Save the final model
    model_filename = "trained_model.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    model_size = os.path.getsize(model_filename) / 1024.0
    print("Model saved as {}, size: {:.2f} KB".format(model_filename, model_size))
    print("Model training complete.")
    
    return model, df_all, training_time, model_size

if __name__ == "__main__":
    train_model("data/final_processed_data.csv")
