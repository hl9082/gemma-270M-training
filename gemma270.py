"""

File: gemma270.py

Author: Huy Le (hl9082)
Co-author: Mamadou Keita


Description:

------------

Pretrain google/gemma-3-270m on Zarma text only (resp_lrl column).

At the end, report Loss and Perplexity, and push model to Hugging Face Hub.

"""



# =========================

# Imports and Environment

# =========================

import json
import os

import random

import math

import torch

import pandas as pd



from datasets import load_dataset

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM,

    DataCollatorForLanguageModeling,

    Trainer,

    TrainingArguments,

)



# Hugging Face login (replace with your token or use CLI `huggingface-cli login`)

from huggingface_hub import login,HfApi

import getpass

# Secure HF token prompt

HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ")

login(HF_TOKEN)



# =========================

# Configuration

# =========================

SEED = 42

random.seed(SEED)

torch.manual_seed(SEED)



MODEL_NAME = "google/gemma-3-270m"   # Gemma-270M base model



LEARNING_RATE = 1e-5

BATCH_SIZE = 8

EPOCHS = 3   # adjust as needed



TRAIN_SIZE = 45000

EVAL_SIZE = 5000



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# =========================

# Load dataset

# =========================

ds = load_dataset("27Group/InstructLR_Generate_Datasets", "ZarmaInstruct-50k", split="train", streaming=False)

print(ds)



df = pd.DataFrame(ds[: TRAIN_SIZE + EVAL_SIZE])



# Only keep Zarma text

text_col = "resp_lrl"

if text_col not in df.columns:

    raise ValueError(f"Expected column {text_col} not found. Available: {list(df.columns)}")



df = df[df[text_col].str.strip().astype(bool)]  # drop empty rows

df[text_col] = df[text_col].astype(str)



train_texts = df[text_col].iloc[:TRAIN_SIZE].tolist()

eval_texts  = df[text_col].iloc[TRAIN_SIZE: TRAIN_SIZE + EVAL_SIZE].tolist()



# Convert to Hugging Face Dataset

from datasets import Dataset

train_dataset = Dataset.from_dict({"text": train_texts})

eval_dataset  = Dataset.from_dict({"text": eval_texts})



# =========================

# Tokenizer and model

# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,token=HF_TOKEN).to(DEVICE)



# Tokenize

def tokenize_function(examples):

    return tokenizer(examples["text"], truncation=True, max_length=256)



train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

eval_dataset  = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])



# Data collator for causal LM

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# =========================

# Training arguments

# =========================

training_args = TrainingArguments(

    output_dir="./gemma_zarma_pretrain",

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    eval_strategy="epoch",

    save_strategy="no",

    logging_steps=100,

    report_to=["none"],

    fp16=torch.cuda.is_available(),

    dataloader_pin_memory=False,

    seed=SEED,

)



trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

)



# =========================

# Train and evaluate

# =========================

print("\n=== Pretraining Gemma-270M on Zarma ===")

trainer.train()



print("\n=== Final Evaluation ===")

eval_results = trainer.evaluate()



loss = eval_results["eval_loss"]

perplexity = math.exp(loss)



print(f"Final Loss: {loss:.4f}")

print(f"Final Perplexity: {perplexity:.4f}")



# =========================

# Push to Hugging Face Hub

# =========================

# Make sure you are logged in with `huggingface-cli login`

# Ask user for repo owner + repo name
hf_username = input("Enter your Hugging Face username or organization: ").strip()
hf_repo_name = input("Enter the name of the model repository: ").strip()

repo_name = f"{hf_username}/{hf_repo_name}"


# Save results to JSON
results = {
    "eval_loss": round(loss, 4),
    "perplexity": round(perplexity, 4)
}
with open("Gemma270Mpretrain_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Push model + tokenizer

print(f"\nPushing model to Hugging Face Hub at: {repo_name}")
trainer.push_to_hub(repo_name)

# Push results JSON
api = HfApi()
api.upload_file(
    path_or_fileobj="Gemma270Mpretrain_results.json",
    path_in_repo="Gemma270Mpretrain_results.json",
    repo_id=repo_name,
    repo_type="model",
    create_pr=True  # create a pull request
)
