import joblib

def view_model_metrics(model_path='movie_recommender_model.pkl'):
    """
    Load and display the model metrics
    """
    try:
        # Load the model using joblib
        model = joblib.load(model_path)
        
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

def main():
    # Load and display metrics
    view_model_metrics()

if __name__ == "__main__":
    main()