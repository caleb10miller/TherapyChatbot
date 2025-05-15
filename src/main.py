import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from dotenv import load_dotenv
from models.gpt_model import GPTModel
from src.inference.chatbot import TherapyChatbot
from src.evaluation.metrics import calculate_bleu_score, calculate_rouge_scores

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize GPT model
    model = GPTModel(
        model_name=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        model_config={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    # Load the model
    model.load_model()
    
    # Initialize chatbot
    chatbot = TherapyChatbot(model)
    
    # Start conversation
    print(chatbot.start_conversation())
    
    try:
        # Main conversation loop
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nChatbot: Goodbye! Take care of yourself.")
                break
            
            # Get response
            response = chatbot.process_input(user_input)
            print(f"\nChatbot: {response}")
            
            # Calculate BLEU and ROUGE scores
            bleu_score = calculate_bleu_score(response, [user_input])
            rouge_scores = calculate_rouge_scores(response, user_input)
            
            # Evaluate response
            scores = chatbot.evaluate_response(response)
            
            # Display all scores
            print("\nResponse Evaluation:")
            for metric, score in scores.items():
                print(f"{metric}: {score:.2f}")
            print(f"BLEU score: {bleu_score:.2f}")
            for metric, score in rouge_scores.items():
                print(f"{metric}: {score:.2f}")
    
    finally:
        # Save the conversation when it ends (either normally or due to an error)
        if chatbot.conversation_history:
            log_file = chatbot.save_conversation()
            print(f"\nConversation saved to: {log_file}")

if __name__ == "__main__":
    main() 