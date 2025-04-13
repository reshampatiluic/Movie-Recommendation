import unittest
import pandas as pd
from app.models import SVDRecommender

class TestSVDRecommender(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize the SVD recommender once for all tests"""
        cls.model = SVDRecommender()
        cls.user_id = cls.model.df['user_id'].iloc[0]  # Dynamically choose an existing user

    def test_model_loads(self):
        """Check that the trained model is not None"""
        self.assertIsNotNone(self.model.model)

    def test_data_load(self):
        """Check data is loaded properly"""
        df = self.model.df
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("user_id", df.columns)
        self.assertIn("movie_id", df.columns)
        self.assertIn("rating", df.columns)

    def test_data_types(self):
        """Check for correct data types"""
        df = self.model.df
        self.assertTrue(pd.api.types.is_integer_dtype(df["user_id"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["rating"]))

    def test_train_method(self):
        """Ensure train method runs and returns expected outputs"""
        model, df, training_time, model_size = self.model.train()
        self.assertIsNotNone(model)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertGreater(training_time, 0)
        self.assertGreater(model_size, 0)

    def test_recommend(self):
        """Test recommendations are generated correctly"""
        recs, infer_time = self.model.recommend(user_id=self.user_id, n=10)
        self.assertIsInstance(recs, list)
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 10)
        self.assertGreater(infer_time, 0)

    def test_recommend_does_not_return_watched(self):
        """Ensure recommended movies do not include already watched ones"""
        watched = set(self.model.df[self.model.df["user_id"] == self.user_id]["movie_id"].tolist())
        recs, _ = self.model.recommend(user_id=self.user_id, n=10)
        self.assertIsInstance(recs, list)
        self.assertTrue(all(movie not in watched for movie in recs))


if __name__ == "__main__":
    unittest.main()