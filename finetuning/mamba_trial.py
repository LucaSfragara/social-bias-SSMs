#!/usr/bin/env python
# Fine-tuning script for Mamba-2.8B-HF model with PANDA dataset using minimal TRL parameters

import os
import json
import torch
import logging
from datasets import Dataset
from typing import Dict, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
set_seed(42)

# Configuration
CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_NAME = "state-spaces/mamba-2.8b-hf"  # HuggingFace-compatible model
PANDA_DATASET_PATH = os.path.join(
    CUR_DIR_PATH, "dataset", "PANDA-annotated-100k-shard0.jsonl"
)
OUTPUT_DIR = os.path.join(CUR_DIR_PATH, "mamba-panda-finetuned")
LOGGING_DIR = os.path.join(CUR_DIR_PATH, "logs")

# Training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

# LoRA parameters - using the ones from the example
LORA_RANK = 8
TARGET_MODULES = ["x_proj", "embeddings", "in_proj", "out_proj"]

def load_panda_dataset(dataset_path: str) -> List[Dict]:
    """Load the PANDA dataset from a JSONL file"""
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_formatted_dataset(data: List[Dict]) -> Dataset:
    """Format PANDA data as a text-only dataset for TRL"""
    formatted_examples = []
    
    for item in data:
        original = item["original"]
        rewrite = item["rewrite"]
        word = item["selected_word"]
        category = item["perturbed_category"]
        
        # Create formatted text matching what we're trying to teach the model
        formatted_text = f"""Rewrite the following text to change '{word}' to a {category} reference while preserving meaning:
Original: {original}
Rewritten: {rewrite}"""
        
        formatted_examples.append(formatted_text)
    
    # Return as a simple dataset with a text column
    return Dataset.from_dict({"text": formatted_examples})

def main():
    """Main training function"""
    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Fix padding token issue
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model from {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load and prepare PANDA dataset
    logger.info(f"Loading PANDA dataset from {PANDA_DATASET_PATH}")
    panda_data = load_panda_dataset(PANDA_DATASET_PATH)
    dataset = create_formatted_dataset(panda_data)
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Split dataset into train and eval (90/10 split)
    dataset_split = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,  # Smaller eval batch size to avoid OOM
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=LEARNING_RATE,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        save_total_limit=2,
        fp16=True,
        report_to="none",  # Disable wandb for simplicity
        max_grad_norm=0.3,  # Helps with memory usage
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=LORA_RANK,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
        lora_alpha=16,
        lora_dropout=0.05,
    )
    
    # Use minimal parameters for SFTTrainer to maximize compatibility
    try:
        # Attempt the most basic initialization that should work across versions
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
    except TypeError as e:
        logger.warning(f"Error initializing SFTTrainer: {e}")
        logger.warning("Falling back to direct Transformers training...")
        
        # If all else fails, use direct transformers training
        from transformers import Trainer, default_data_collator
        from peft import get_peft_model, prepare_model_for_kbit_training
        
        # Prepare model with PEFT
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        
        # Initialize standard Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
    
    # Train model
    logger.info("Starting training")
    trainer.train()
    
    # Save model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()