from typing import Dict, Any, Optional
from models.base_model import BaseModel

class TherapyChatbot:
    """Interface for interacting with the therapy chatbot."""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        self.conversation_history = []
        
    def start_conversation(self):
        """Start a new conversation."""
        self.conversation_history = []
        return "Hello! I'm here to help you reframe negative thoughts. What's on your mind?"
    
    def process_input(self, user_input: str) -> str:
        """Process user input and generate a response."""
        # Add user input to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Get model response
        response = self.model.reframe_thought(user_input)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.conversation_history
    
    def evaluate_response(self, response: str) -> Dict[str, float]:
        """Evaluate the quality of a response."""
        return self.model.evaluate_response(response)
    
    def save_conversation(self, filepath: str):
        """Save the conversation history to a file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
    
    def load_conversation(self, filepath: str):
        """Load a conversation history from a file."""
        import json
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)
