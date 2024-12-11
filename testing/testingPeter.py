import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.peter import CleanCityname

class TestDataCleanFunctions(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'cityname': ['New York', '123City', 'Los Angeles'],
            'state': ['NY', None, 'CA'],
            'latitude': [40.7128, 34.0522, 37.7749],
            'longitude': [-74.0060, -118.2437, -122.4194]
        })

    def test_clean_cityname(self):
        # Expected DataFrame after cleaning
        expected_data = pd.DataFrame({
            'cityname': ['New York', 'Los Angeles'],
            'state': ['NY', 'CA'],
            'latitude': [40.7128, 37.7749],
            'longitude': [-74.0060, -122.4194]
        }).reset_index(drop=True)
        print(expected_data)
        # Call the function
        cleaned_data = CleanCityname(self.sample_data).reset_index(drop=True)
        print(CleanCityname(self.sample_data))
        # Assert that the cleaned data matches the expected data
        pd.testing.assert_frame_equal(cleaned_data, expected_data)

if __name__ == '__main__':
    unittest.main()