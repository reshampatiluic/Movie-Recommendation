# data.py
import pandas as pd

def load_data(data_path="data/final_processed_data.csv"):
    df_all = pd.read_csv(data_path)
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
    
    print(f"Loaded {len(df_all)} rows from {data_path}.")
    return df_all
