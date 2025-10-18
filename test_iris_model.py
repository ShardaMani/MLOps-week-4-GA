

import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
import os

class TestIrisModel(unittest.TestCase):

    def setUp(self):
        """Set up model and data for testing."""
        # Load the trained model
        self.model = tf.keras.models.load_model('iris_model.h5')
        # Load the dataset
        self.data = pd.read_csv(os.path.join('Data', 'iris.csv'))

    def test_data_validation(self):
        """Test to validate the shape and columns of the input data."""
        print("Running data validation test...")
        # We expect 5 columns: 4 features + 1 target
        self.assertEqual(self.data.shape[1], 5)
        print("Data validation test passed.")

    def test_model_evaluation(self):
        """A simple sanity test for the model prediction."""
        print("Running model evaluation sanity test...")
        # Create a sample input (versicolor)
        sample_input = np.array([[6.0, 2.2, 4.0, 1.0]])

        # Get the prediction
        prediction = self.model.predict(sample_input)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # We expect the model to be able to predict without errors.
        # The output shape should be (1, 3) for 3 classes.
        self.assertEqual(prediction.shape, (1, 3))
        print(f"Prediction successful. Predicted class index: {predicted_class}")
        print("Model evaluation sanity test passed.")

if __name__ == '__main__':
    unittest.main()
