from typing import Dict, Any, Optional
from models.base_model import BaseModel
from datetime import datetime
import json
import os
from difflib import SequenceMatcher

class TherapyChatbot:
    """Interface for interacting with the therapy chatbot."""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        self.conversation_history = []
        self.session_scores = []
        self.session_start_time = None
        self.model_name = self.model.model_name.replace("/", "-")
        # No system prompt for fine-tuned model
        
    def start_conversation(self):
        self.conversation_history = []
        self.session_scores = []
        self.session_start_time = datetime.now()
        return "Hello! I'm here to help you reframe negative thoughts. What's on your mind?"
    
    def _is_echo_response(self, user_input: str, response: str, threshold: float = 0.8) -> bool:
        user_input_clean = user_input.lower().strip()
        response_clean = response.lower().strip()
        similarity = SequenceMatcher(None, user_input_clean, response_clean).ratio()
        return similarity >= threshold
    
    def _validate_response(self, user_input: str, response: str, max_retries: int = 3) -> str:
        retries = 0
        current_response = response
        while retries < max_retries:
            if not self._is_echo_response(user_input, current_response):
                return current_response
            retries += 1
            try:
                # Regenerate using full history
                current_response = self.model.reframe_thought(
                    self._build_messages(user_input)
                )
            except Exception as e:
                print(f"Error generating alternative response: {e}")
                break
        return current_response
    
    def _build_messages(self, new_user_input: str = None):
        # Only include user and assistant messages, no system prompt
        messages = []
        for entry in self.conversation_history:
            messages.append({"role": entry["role"], "content": entry["content"]})
        if new_user_input is not None:
            messages.append({"role": "user", "content": new_user_input})
        return messages
    
    def process_input(self, user_input: str) -> str:
        self.conversation_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().isoformat()})
        # Build full message history for the model
        messages = self._build_messages()
        # Get model response and validate it
        initial_response = self.model.reframe_thought(messages)
        response = self._validate_response(user_input, initial_response)
        self.session_scores.append({
            "timestamp": datetime.now().isoformat(),
            "scores": self.evaluate_response(response, user_input),
            "is_echo": self._is_echo_response(user_input, response)
        })
        self.conversation_history.append({
            "role": "assistant", 
            "content": response, 
            "timestamp": datetime.now().isoformat(),
            "is_echo": self._is_echo_response(user_input, response)
        })
        return response
    
    def get_conversation_history(self) -> list:
        return self.conversation_history
    
    def evaluate_response(self, response: str, user_input: str = None) -> Dict[str, float]:
        if user_input and self._is_echo_response(user_input, response):
            return {
                "empathy": 0.0,
                "clarity": 0.0,
                "cbt_technique": 0.0,
                "supportiveness": 0.0
            }
        return self.model.evaluate_response(response)
    
    def save_conversation(self, filepath: str = None):
        if not filepath:
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join("logs", self.model_name, f"therapy_session_{timestamp}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        session_data = {
            "model_name": self.model_name,
            "session_start": self.session_start_time.isoformat(),
            "session_end": datetime.now().isoformat(),
            "conversation_history": self.conversation_history,
            "session_scores": self.session_scores,
            "average_scores": self._calculate_average_scores()
        }
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        return filepath
    
    def load_conversation(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.conversation_history = data.get("conversation_history", [])
            self.session_scores = data.get("session_scores", [])
            self.session_start_time = datetime.fromisoformat(data.get("session_start"))
            if "model_name" in data:
                self.model_name = data["model_name"]
    
    def _calculate_average_scores(self) -> Dict[str, float]:
        if not self.session_scores:
            return {}
        total_scores = {}
        count = len(self.session_scores)
        for score_entry in self.session_scores:
            for category, value in score_entry["scores"].items():
                if category not in total_scores:
                    total_scores[category] = 0.0
                total_scores[category] += value
        return {category: value/count for category, value in total_scores.items()}
