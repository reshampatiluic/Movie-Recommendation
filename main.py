# main.py
from train import train_model
from inference import recommend_movies

def main():
    
    model, df_all, training_time, model_size = train_model("data/final_processed_data.csv")
    
    # Generate recommendations for a sample user (adjust user_id as needed)
    test_user_id = 77386
    recommendations, inference_time = recommend_movies(test_user_id, model, df_all, top_n=20)
    print(f"Recommendations for user {test_user_id}: {recommendations}")

if __name__ == "__main__":
    main()
