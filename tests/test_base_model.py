import unittest
from app.models import Base, SVDRecommender


class TestBaseModel(unittest.TestCase):
    def test_base_singleton_behavior(self):
        """Test if Base enforces singleton pattern correctly"""
        base1 = Base()
        base2 = Base()
        self.assertIs(base1, base2)

    def test_svd_is_subclass_of_base(self):
        """Ensure SVDRecommender inherits from Base"""
        model = SVDRecommender()
        self.assertIsInstance(model, Base)

    def test_base_recommend_not_implemented(self):
        """Base class should raise NotImplementedError for recommend()"""
        base = Base()
        with self.assertRaises(NotImplementedError):
            base.recommend(user_id=1)

    def test_base_train_not_implemented(self):
        """Base class should raise NotImplementedError for train()"""
        base = Base()
        with self.assertRaises(NotImplementedError):
            base.train()

    def test_base_save_model_not_implemented(self):
        """Base class should raise NotImplementedError for save_model()"""
        base = Base()
        with self.assertRaises(NotImplementedError):
            base.save_model()

    def test_base_load_data_not_implemented(self):
        """Base class should raise NotImplementedError for load_data()"""
        base = Base()
        with self.assertRaises(NotImplementedError):
            base.load_data()


if __name__ == "__main__":
    unittest.main()
