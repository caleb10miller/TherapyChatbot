from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from models.base_model import BaseModel

class TherapyDataset(Dataset):
    """Custom dataset for therapy chatbot training."""
    
    def __init__(self, input_texts: List[str], target_texts: List[str], tokenizer):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize inputs and targets
        inputs = self.tokenizer(input_text, padding='max_length', truncation=True, return_tensors='pt')
        targets = self.tokenizer(target_text, padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class ModelTrainer:
    """Trainer class for therapy chatbot models."""
    
    def __init__(self, model: BaseModel, tokenizer, training_args: Dict[str, Any] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args or {}
        
    def prepare_dataset(self, input_texts: List[str], target_texts: List[str]) -> Dataset:
        """Prepare dataset for training."""
        return TherapyDataset(input_texts, target_texts, self.tokenizer)
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """Train the model."""
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            save_strategy="steps",
            save_steps=500,
            **self.training_args
        )
        
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()
        
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model."""
        trainer = Trainer(
            model=self.model.model,
            args=TrainingArguments(output_dir="./results", per_device_eval_batch_size=8)
        )
        
        return trainer.evaluate(eval_dataset)
