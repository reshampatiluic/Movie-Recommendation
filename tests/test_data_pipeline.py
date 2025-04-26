import unittest
import pandas as pd
from raw_files import data


class TestDataPipeline(unittest.TestCase):
    def load_clean_data(self):
        """Helper to load and clean data for tests"""
        df = data.load_data("data/final_processed_data.csv")
        df.dropna(
            subset=["Timestamp_y"], inplace=True
        )  # Handle missing timestamp issue
        return df

    def test_data_loading(self):
        df = self.load_clean_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_data_schema(self):
        df = self.load_clean_data()
        required_columns = ["user_id", "movie_id", "rating"]
        for col in required_columns:
            self.assertIn(col, df.columns)

    def test_no_missing_values(self):
        df = self.load_clean_data()
        print("Missing value counts:\n", df.isnull().sum())
        self.assertFalse(
            df.isnull().any().any()
        )  # Should be True only if all missing data is removed

    def test_rating_range(self):
        df = self.load_clean_data()
        self.assertTrue(df["rating"].between(0, 5).all())

    def test_unique_users_and_movies(self):
        df = self.load_clean_data()
        self.assertGreater(df["user_id"].nunique(), 1)
        self.assertGreater(df["movie_id"].nunique(), 1)

    def test_data_types(self):
        df = self.load_clean_data()
        self.assertTrue(pd.api.types.is_integer_dtype(df["user_id"]))  # int64
        self.assertTrue(pd.api.types.is_object_dtype(df["movie_id"]))  # object (string)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["rating"]))  # float64 or int
