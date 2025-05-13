import pandas as pd
import json
from pathlib import Path
from typing import List, Dict
import tiktoken
from tqdm import tqdm

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def prepare_training_data(input_file: str, output_file: str, max_tokens: int = 16000):
    """Prepare training data for GPT-3.5 Turbo fine-tuning."""
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Prepare training examples
    training_examples = []
    skipped_examples = []
    
    print(f"Processing {len(df)} examples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Create the messages
        messages = [
            {
                "role": "system",
                "content": "You are a therapy chatbot trained in Cognitive Behavioral Therapy (CBT) techniques. Your goal is to help users reframe negative thoughts in a supportive and constructive way."
            },
            {
                "role": "user",
                "content": f"Reframe this negative thought using CBT techniques: {row['prompt']}"
            },
            {
                "role": "assistant",
                "content": row['utterance']
            }
        ]
        
        # Count total tokens
        total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
        
        if total_tokens > max_tokens:
            skipped_examples.append((idx, total_tokens))
            continue
            
        # Create a training example
        example = {"messages": messages}
        training_examples.append(example)
    
    # Save as JSONL file
    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\nPrepared {len(training_examples)} training examples")
    if skipped_examples:
        print(f"Skipped {len(skipped_examples)} examples that exceeded token limit:")
        for idx, tokens in skipped_examples:
            print(f"  Example {idx}: {tokens} tokens")
    
    return training_examples

def main():
    # Create output directory if it doesn't exist
    output_dir = Path("data/finetune")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare training data
    train_examples = prepare_training_data(
        "data/train_data.csv",
        output_dir / "train.jsonl"
    )
    
    # Prepare validation data
    val_examples = prepare_training_data(
        "data/val_data.csv",
        output_dir / "val.jsonl"
    )
    
    print(f"\nFinal counts:")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

if __name__ == "__main__":
    main() 