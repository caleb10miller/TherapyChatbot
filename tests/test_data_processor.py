import unittest
import pandas as pd
from pathlib import Path
from src.preprocessing.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        
    def test_preprocess_data(self):
        # Create sample data
        data = pd.DataFrame({
            'context': ['I am a failure', 'I can\'t do anything right', None],
            'response': ['That\'s not true', 'You\'re being too hard on yourself', 'You can do this']
        })
        
        # Test preprocessing
        processed_data = self.processor.preprocess_data(data)
        
        # Check that None values are removed
        self.assertEqual(len(processed_data), 2)
        
        # Check that text is converted to lowercase
        self.assertEqual(processed_data['context'].iloc[0], 'i am a failure')
        
    def test_create_training_pairs(self):
        # Create sample data
        data = pd.DataFrame({
            'context': ['I am a failure', 'I can\'t do anything right'],
            'response': ['That\'s not true', 'You\'re being too hard on yourself']
        })
        
        # Test pair creation
        pairs = self.processor.create_training_pairs(data)
        
        # Check that pairs are created correctly
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0], ('I am a failure', 'That\'s not true'))
        
if __name__ == '__main__':
    unittest.main()
