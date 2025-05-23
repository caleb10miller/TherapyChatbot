import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import openai
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
import sys
sys.path.append(project_root)

from models.gpt_model import GPTModel
from src.evaluation.metrics import calculate_bleu_score, calculate_rouge_scores

class ModelComparison:
    def __init__(self):
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.system_prompt = """You are a supportive therapy chatbot specializing in Cognitive Behavioral Therapy (CBT).
        Your responses should be empathetic, clear, and incorporate CBT techniques when appropriate.
        Focus on validating feelings, challenging cognitive distortions, and providing practical support."""
        
        # Initialize models
        self.models = {
            "ft:gpt-3.5-turbo-0125:personal::BWtfMrrB": GPTModel(
                model_name="ft:gpt-3.5-turbo-0125:personal::BWtfMrrB",
                model_config={"api_key": os.getenv("OPENAI_API_KEY")}
            ),
            "gpt-3.5-turbo-0125": GPTModel(
                model_name="gpt-3.5-turbo-0125",
                model_config={"api_key": os.getenv("OPENAI_API_KEY")}
            ),
            "gpt-4o-mini": GPTModel(
                model_name="gpt-4o-mini",
                model_config={"api_key": os.getenv("OPENAI_API_KEY")}
            ),
            "gpt-4": GPTModel(
                model_name="gpt-4",
                model_config={"api_key": os.getenv("OPENAI_API_KEY")}
            ),
            "gpt-4-0125-preview": GPTModel(
                model_name="gpt-4-0125-preview",
                model_config={"api_key": os.getenv("OPENAI_API_KEY")}
            ),
            "gpt-4-turbo-preview": GPTModel(
                model_name="gpt-4-turbo-preview",
                model_config={"api_key": os.getenv("OPENAI_API_KEY")}
            )
        }
        
        # Load test prompts
        with open(os.path.join(project_root, "data", "test_prompts.json"), "r") as f:
            self.test_prompts = json.load(f)["prompts"]
    
    def _is_echo_response(self, user_input: str, response: str, threshold: float = 0.8) -> bool:
        """Check if the response is too similar to the user input."""
        user_input_clean = user_input.lower().strip()
        response_clean = response.lower().strip()
        similarity = SequenceMatcher(None, user_input_clean, response_clean).ratio()
        return similarity >= threshold
    
    def run_model(self, model_name: str, prompt: str, use_system_prompt: bool = True) -> str:
        """Run a single prompt through a model."""
        messages = [{"role": "user", "content": prompt}]
        if use_system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error running model {model_name}: {e}")
            return ""
    
    def evaluate_response(self, response: str, user_input: str) -> Dict[str, float]:
        """Evaluate a response using GPT-4 for consistent evaluation across all models."""
        # Check for echo response first
        if self._is_echo_response(user_input, response):
            return {
                "empathy": 0.0,
                "clarity": 0.0,
                "cbt_technique": 0.0,
                "supportiveness": 0.0,
                "response_quality": 0.0,
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }
        
        # Get base scores from GPT-4 evaluation
        scores = self.models["gpt-4"].evaluate_response(response)
        
        # Add BLEU and ROUGE scores
        bleu_score = calculate_bleu_score(response, [user_input])
        rouge_scores = calculate_rouge_scores(response, user_input)
        
        scores.update({
            "bleu": bleu_score,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"]
        })
        
        return scores
    
    def run_comparison(self) -> pd.DataFrame:
        """Run the full model comparison and return results as a DataFrame."""
        results = []
        total_prompts = len(self.test_prompts)
        total_models = len(self.models)
        
        print(f"\nStarting model comparison with {total_prompts} prompts and {total_models} models...")
        
        for i, prompt in enumerate(self.test_prompts, 1):
            user_input = prompt["user_input"]
            print(f"\nProcessing prompt {i}/{total_prompts}: {user_input}")
            
            # Test each model
            for model_name in self.models:
                print(f"  Testing {model_name}...")
                
                # Test with system prompt
                response = self.run_model(model_name, user_input, use_system_prompt=True)
                if response:
                    scores = self.evaluate_response(response, user_input)
                    results.append({
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "user_input": user_input,
                        "model": f"{model_name} + prompt",
                        "response": response,
                        **scores
                    })
                
                # Test without system prompt
                response = self.run_model(model_name, user_input, use_system_prompt=False)
                if response:
                    scores = self.evaluate_response(response, user_input)
                    results.append({
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "user_input": user_input,
                        "model": model_name,
                        "response": response,
                        **scores
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "logs", "model_comparison")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, f"detailed_results_{timestamp}.csv"), index=False)
        
        # Calculate and save summary statistics
        summary = df.groupby("model").agg({
            "empathy": "mean",
            "clarity": "mean",
            "cbt_technique": "mean",
            "supportiveness": "mean",
            "response_quality": "mean",
            "bleu": "mean",
            "rouge1": "mean",
            "rouge2": "mean",
            "rougeL": "mean"
        }).round(3)
        
        summary.to_csv(os.path.join(output_dir, f"summary_{timestamp}.csv"))
        
        return df

def main():
    comparison = ModelComparison()
    results = comparison.run_comparison()
    
    # Print summary statistics
    print("\nModel Comparison Results:")
    print(results.groupby("model").agg({
        "empathy": "mean",
        "clarity": "mean",
        "cbt_technique": "mean",
        "supportiveness": "mean",
        "response_quality": "mean"
    }).round(3))

if __name__ == "__main__":
    main() 