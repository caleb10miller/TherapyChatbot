from typing import Dict, Any
from openai import OpenAI
from models.base_model import BaseModel
import os
from dotenv import load_dotenv

class GPTModel(BaseModel):
    """GPT-based model for therapy chatbot."""
    
    def __init__(self, model_name: str = None, model_config: Dict[str, Any] = None):
        # Load environment variables
        load_dotenv()
        
        # Use model from config, env, or default
        self.model_name = (
            model_name or 
            model_config.get('model_name') if model_config else 
            os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
        )
        
        # Get API key from config or env
        self.api_key = (
            model_config.get('api_key') if model_config else 
            os.getenv('OPENAI_API_KEY')
        )
        
        super().__init__(self.model_name, model_config)
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        
    def load_model(self):
        """Load the GPT model."""
        # GPT models are loaded via API, so we just verify the API key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        if not self.client:
            self.client = OpenAI(api_key=self.api_key)
        self.model = self.model_name  # Store the model name for API calls
        
    def reframe_thought(self, messages) -> str:
        """Reframe a negative thought using CBT techniques, using the full conversation history."""
        if not self.model or not self.client:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Only add system prompt for non-fine-tuned models
        if not self.model.startswith("ft:") and not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {
                "role": "system",
                "content": """You are a CBT assistant that helps users reframe negative thoughts using cognitive behavioral therapy techniques. You are NOT a therapist or medical professional.

Your task is to:
1. Briefly acknowledge the user's feeling (1 sentence)
2. Identify one cognitive distortion (1 sentence)
3. Help them challenge that thought (1-2 sentences)
4. Suggest one practical step (1 sentence)

Keep responses under 5 sentences total. Focus on practical CBT techniques and thought reframing. If the user needs professional help, briefly acknowledge this while still providing CBT support."""
            })
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=100,
                presence_penalty=0.6,
                frequency_penalty=0.6
            )
            response_text = response.choices[0].message.content.strip()
            # If response is too short or contains the exact input, try with adjusted parameters
            if len(response_text.split()) < 10:
                # Try again with slightly different parameters
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                response_text = response.choices[0].message.content.strip()
            return response_text
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def evaluate_response(self, response: str) -> Dict[str, float]:
        """Evaluate the quality of a response using GPT."""
        if not self.model or not self.client:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        evaluation_prompt = f"""Evaluate the following therapy response on a scale of 0-1 for:
        1. Empathy
        2. Clarity
        3. CBT technique application
        4. Supportiveness
        
        Response: {response}
        
        Return ONLY a JSON object with these exact keys and float values between 0 and 1:
        {{
            "empathy": 0.0,
            "clarity": 0.0,
            "cbt_technique": 0.0,
            "supportiveness": 0.0
        }}"""
        
        try:
            eval_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of therapy responses. Always respond with valid JSON only."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3,
                max_tokens=150,
                response_format={ "type": "json_object" }
            )
            
            # Parse the response to get scores
            import json
            try:
                scores = json.loads(eval_response.choices[0].message.content.strip())
                # Ensure all required keys are present
                required_keys = ["empathy", "clarity", "cbt_technique", "supportiveness"]
                if not all(key in scores for key in required_keys):
                    # If any key is missing, return default scores
                    return {key: 0.5 for key in required_keys}
                return scores
            except json.JSONDecodeError:
                # If JSON parsing fails, return default scores
                return {
                    "empathy": 0.5,
                    "clarity": 0.5,
                    "cbt_technique": 0.5,
                    "supportiveness": 0.5
                }
        except Exception as e:
            # If any other error occurs, return default scores
            return {
                "empathy": 0.5,
                "clarity": 0.5,
                "cbt_technique": 0.5,
                "supportiveness": 0.5
            }
    
    def save_model(self, path: str):
        """Save model configuration."""
        import json
        config = {
            'model_name': self.model_name,
            'api_key': self.api_key
        }
        with open(path, 'w') as f:
            json.dump(config, f)
    
    def load_model_from_path(self, path: str):
        """Load model configuration."""
        import json
        with open(path, 'r') as f:
            config = json.load(f)
        self.model_name = config['model_name']
        self.api_key = config['api_key']
        self.load_model()
