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
            
        required_keys = ["empathy", "clarity", "cbt_technique", "supportiveness", "response_quality"]
            
        evaluation_prompt = f"""You are a strict evaluator of therapy responses. Score the following response on a scale of 0-1 for each criterion. Be very critical and objective.

Response to evaluate: {response}

Evaluation Criteria:

1. Empathy (0-1):
- 0.0: No emotional validation or understanding shown
- 0.3: Basic acknowledgment of feelings
- 0.6: Good emotional validation and understanding
- 0.8: Excellent emotional validation with specific understanding
- 1.0: Perfect emotional validation with deep understanding

2. Clarity (0-1):
- 0.0: Unclear or confusing response
- 0.3: Basic explanation of concepts
- 0.6: Clear explanation with good structure
- 0.8: Very clear with excellent structure
- 1.0: Perfect clarity and structure

3. CBT Technique Application (0-1):
- 0.0: No CBT technique used
- 0.3: Basic mention of a technique
- 0.6: Good explanation of technique
- 0.8: Excellent application of technique
- 1.0: Perfect application with multiple techniques

4. Supportiveness (0-1):
- 0.0: No practical support or advice
- 0.3: Basic supportive statement
- 0.6: Good practical advice
- 0.8: Excellent actionable steps
- 1.0: Perfect combination of support and action

5. Response Quality (0-1):
- 0.0: Too short (<10 words) or too long (>100 words)
- 0.3: Basic structure, some issues
- 0.6: Good structure and length
- 0.8: Excellent structure and length
- 1.0: Perfect structure and length

Automatic Penalties:
- Responses under 15 words: All scores reduced by 50%
- No CBT technique mentioned: CBT score automatically 0.0
- No practical advice: Supportiveness score automatically 0.0
- Echo of user input: All scores reduced by 30%

Return your evaluation in this exact format:
empathy: [score]
clarity: [score]
cbt_technique: [score]
supportiveness: [score]
response_quality: [score]"""
        
        try:
            eval_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are a strict evaluator of therapy responses.
                    You have extensive experience in CBT and therapy response evaluation.
                    You are very critical and objective in your scoring.
                    Never give perfect scores (1.0) unless absolutely warranted.
                    Always respond with scores in the exact format specified."""},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # Parse the response to get scores
            try:
                response_text = eval_response.choices[0].message.content.strip()
                scores = {}
                
                # Parse each line for scores
                for line in response_text.split('\n'):
                    if ':' in line:
                        key, value = line.split(':')
                        key = key.strip()
                        try:
                            scores[key] = float(value.strip())
                        except ValueError:
                            scores[key] = 0.0
                
                # Ensure all required keys are present
                if not all(key in scores for key in required_keys):
                    return {key: 0.0 for key in required_keys}
                
                # Apply automatic penalties
                words = response.split()
                
                # Length penalty
                if len(words) < 15:
                    for key in scores:
                        scores[key] *= 0.5
                
                # CBT technique penalty
                cbt_keywords = ['cognitive', 'distortion', 'reframe', 'challenge', 'thought', 'perspective', 'belief']
                if not any(keyword in response.lower() for keyword in cbt_keywords):
                    scores['cbt_technique'] = 0.0
                
                # Supportiveness penalty
                if len(words) < 20 or not any(word in response.lower() for word in ['try', 'could', 'would', 'suggest', 'help']):
                    scores['supportiveness'] = 0.0
                
                # Echo penalty
                if any(word in response.lower() for word in ['why would they', 'that\'s just', 'i know right']):
                    for key in scores:
                        scores[key] *= 0.7
                
                return scores
            except Exception as e:
                return {key: 0.0 for key in required_keys}
        except Exception as e:
            return {key: 0.0 for key in required_keys}
    
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
