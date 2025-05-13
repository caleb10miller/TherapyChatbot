from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseModel(ABC):
    """Base class for all therapy chatbot models."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any] = None):
        self.model_name = model_name
        self.model_config = model_config or {}
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load the model and its weights."""
        pass
    
    @abstractmethod
    def reframe_thought(self, negative_thought: str) -> str:
        """Reframe a negative thought using CBT techniques."""
        pass
    
    @abstractmethod
    def evaluate_response(self, response: str) -> Dict[str, float]:
        """Evaluate the quality of a response."""
        pass
    
    def save_model(self, path: str):
        """Save the model to disk."""
        if self.model is None:
            raise ValueError("No model loaded to save")
        # Implementation will be model-specific
        pass
    
    def load_model_from_path(self, path: str):
        """Load a saved model from disk."""
        # Implementation will be model-specific
        pass
