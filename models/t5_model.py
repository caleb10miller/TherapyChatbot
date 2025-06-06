from typing import Dict, Any
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from models.base_model import BaseModel

class T5Model(BaseModel):
    """T5-based model for therapy chatbot."""
    
    def __init__(self, model_name: str = "google/flan-t5-base", model_config: Dict[str, Any] = None):
        super().__init__(model_name, model_config)
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the T5 model and tokenizer."""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
        
    def reframe_thought(self, negative_thought: str) -> str:
        """Reframe a negative thought using CBT techniques."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare the input text
        input_text = f"""reframe the following negative thought using CBT techniques: {negative_thought}"""
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate response
        try:
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=5,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                length_penalty=1.0
            )
            
            # Decode and return the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def evaluate_response(self, response: str) -> Dict[str, float]:
        """Evaluate the quality of a response using the model."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Prepare evaluation prompts
        eval_prompts = {
            "empathy": f"rate the empathy level of this response (0-1): {response}",
            "clarity": f"rate the clarity of this response (0-1): {response}",
            "cbt_technique": f"rate how well this response applies CBT techniques (0-1): {response}",
            "supportiveness": f"rate how supportive this response is (0-1): {response}"
        }
        
        scores = {}
        try:
            for metric, prompt in eval_prompts.items():
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=10,
                    num_beams=1,
                    temperature=0.3
                )
                
                score_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Extract numerical score from the response
                try:
                    score = float(score_text.strip())
                    scores[metric] = min(max(score, 0), 1)  # Ensure score is between 0 and 1
                except ValueError:
                    scores[metric] = 0.5  # Default score if parsing fails
                    
            return scores
        except Exception as e:
            raise Exception(f"Error evaluating response: {str(e)}")
    
    def save_model(self, path: str):
        """Save the model and tokenizer."""
        if not self.model or not self.tokenizer:
            raise ValueError("No model loaded to save")
            
        try:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")
    
    def load_model_from_path(self, path: str):
        """Load a saved model and tokenizer."""
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(path)
            self.model = T5ForConditionalGeneration.from_pretrained(path)
            self.model.to(self.device)
        except Exception as e:
            raise Exception(f"Error loading model from path: {str(e)}")
