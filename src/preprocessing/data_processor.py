import pandas as pd
from typing import Tuple, List
from pathlib import Path

class DataProcessor:
    """Process and prepare the EmpatheticDialogues dataset for training."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, validation, and test datasets."""
        self.train_data = pd.read_csv(self.data_dir / "train_data.csv")
        self.val_data = pd.read_csv(self.data_dir / "val_data.csv")
        self.test_data = pd.read_csv(self.data_dir / "test_data.csv")
        return self.train_data, self.val_data, self.test_data
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for training."""
        # Remove any rows with missing values
        df = df.dropna()
        
        # Convert to lowercase
        df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
        
        # Add any additional preprocessing steps here
        return df
    
    def create_training_pairs(self, df: pd.DataFrame) -> List[Tuple[str, str]]:
        """Create input-target pairs for training."""
        # Implementation will depend on the specific format of your data
        # This is a placeholder that should be modified based on your data structure
        pairs = []
        for _, row in df.iterrows():
            # Assuming your data has 'context' and 'response' columns
            if 'context' in row and 'response' in row:
                pairs.append((row['context'], row['response']))
        return pairs
    
    def save_processed_data(self, output_dir: str):
        """Save processed data to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.train_data is not None:
            self.train_data.to_csv(output_path / "processed_train.csv", index=False)
        if self.val_data is not None:
            self.val_data.to_csv(output_path / "processed_val.csv", index=False)
        if self.test_data is not None:
            self.test_data.to_csv(output_path / "processed_test.csv", index=False)
