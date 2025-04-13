# train_knn.py
from app.models import KNNRecommender

if __name__ == "__main__":
    knn = KNNRecommender()
    knn.train()