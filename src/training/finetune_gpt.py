import os
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

def upload_file(client: OpenAI, file_path: str) -> str:
    """Upload a file for fine-tuning."""
    with open(file_path, "rb") as file:
        response = client.files.create(
            file=file,
            purpose="fine-tune"
        )
    return response.id

def create_fine_tuning_job(client: OpenAI, training_file_id: str, validation_file_id: str = None) -> str:
    """Create a fine-tuning job."""
    job_params = {
        "model": "gpt-3.5-turbo",
        "training_file": training_file_id,
    }
    
    if validation_file_id:
        job_params["validation_file"] = validation_file_id
    
    response = client.fine_tuning.jobs.create(**job_params)
    return response.id

def check_job_status(client: OpenAI, job_id: str) -> dict:
    """Check the status of a fine-tuning job."""
    response = client.fine_tuning.jobs.retrieve(job_id)
    return {
        "status": response.status,
        "model": response.fine_tuned_model if response.fine_tuned_model else None,
        "error": response.error if response.error else None
    }

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # File paths
    data_dir = Path("data/finetune")
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    
    # Upload files
    print("Uploading training file...")
    training_file_id = upload_file(client, str(train_file))
    print(f"Training file ID: {training_file_id}")
    
    print("Uploading validation file...")
    validation_file_id = upload_file(client, str(val_file))
    print(f"Validation file ID: {validation_file_id}")
    
    # Create fine-tuning job
    print("Creating fine-tuning job...")
    job_id = create_fine_tuning_job(client, training_file_id, validation_file_id)
    print(f"Job ID: {job_id}")
    
    # Monitor job status
    print("\nMonitoring fine-tuning job...")
    while True:
        status = check_job_status(client, job_id)
        print(f"Status: {status['status']}")
        
        if status['status'] == 'succeeded':
            print(f"\nFine-tuning completed! Model: {status['model']}")
            break
        elif status['status'] == 'failed':
            print(f"\nFine-tuning failed! Error: {status['error']}")
            break
        
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main() 