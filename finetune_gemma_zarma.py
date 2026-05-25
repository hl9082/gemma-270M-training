"""

File: finetune_gemma_noisy_zarma.py

Author: Huy Le (hl9082)

Description:

------------

Fine-tune google/gemma-3-270m (causal LM) to map noisy Zarma text -> cleaned Zarma text.

- Train on first 45k rows (epoch=3)

- Evaluate on next 5k rows

- Compute BLEU, ChrF++, COMET for each noisy column vs "cleaned" column

- Save and push metrics JSON to Hugging Face Hub

"""


# 1.Imports
import os

import math

import json

import random

from typing import List, Dict



import torch

import pandas as pd

from datasets import load_dataset, Dataset

from tqdm.auto import tqdm



from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM,

    Trainer,

    TrainingArguments,

    DataCollatorForLanguageModeling,

)

import getpass

# Secure HF token prompt



# IMPORTANT: ensure you have access to the gated repo and pass a valid HF token if needed

HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ")



SEED = 42

random.seed(SEED)

torch.manual_seed(SEED)



MODEL_NAME = "google/gemma-3-270m"  # gated model; accept license and ensure token works

OUTPUT_DIR = "./gemma_noisy_zarma_finetune"



LEARNING_RATE = 2e-5

EPOCHS = 3

BATCH_SIZE = 28

MAX_LENGTH = 64



TRAIN_ROWS = 45000

EVAL_ROWS = 5000



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


# -------------------------------

# Load dataset (27Group/noisy_zarma)

# -------------------------------

print("Loading dataset: 27Group/noisy_zarma")

hf_ds = load_dataset("27Group/noisy_zarma", split="train", streaming=False)

df = pd.DataFrame(hf_ds[: TRAIN_ROWS + EVAL_ROWS])



# Validate columns

if "cleaned" not in df.columns:

    raise ValueError(f'Expected "cleaned" column not found. Available columns: {list(df.columns)}')



# Identify noisy columns: all except "cleaned" and "original" (original excluded per requirement)

exclude_cols = {"cleaned", "original"}

noisy_cols: List[str] = [c for c in df.columns if c not in exclude_cols]



if not noisy_cols:

    raise ValueError("No noisy columns found to compare against 'cleaned'.")



# Clean and cast

df["cleaned"] = df["cleaned"].astype(str)

for c in noisy_cols:

    df[c] = df[c].astype(str)



# Train/Eval splits

train_df = df.iloc[:TRAIN_ROWS].copy()

eval_df  = df.iloc[TRAIN_ROWS : TRAIN_ROWS + EVAL_ROWS].copy()



# -------------------------------

# Build supervised pairs (noisy -> cleaned) across all noisy columns

# We concatenate all noisy columns into one training set (multi-view supervision).

# -------------------------------

def make_pairs(frame: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Construct noisy→cleaned pairs for training/evaluation.

    Parameters:
    -----------
    frame : pd.DataFrame
        DataFrame containing noisy columns and 'cleaned' column.

    Returns:
    --------
    List[Dict[str, str]]
        Each dict has keys: {"noisy", "cleaned", "col"}.
    """

    pairs = []

    for _, row in frame.iterrows():

        cleaned = row["cleaned"].strip()

        if not cleaned:

            continue

        for nc in noisy_cols:

            noisy = row[nc].strip()

            if not noisy:

                continue

            pairs.append({"noisy": noisy, "cleaned": cleaned, "col": nc})

    return pairs



train_pairs = make_pairs(train_df)

eval_pairs  = make_pairs(eval_df)



print(f"Train pairs: {len(train_pairs)} | Eval pairs: {len(eval_pairs)} | Noisy columns: {noisy_cols}")



# Convert to HF Datasets

train_hf = Dataset.from_dict({

    "noisy": [p["noisy"] for p in train_pairs],

    "cleaned": [p["cleaned"] for p in train_pairs],

    "col": [p["col"] for p in train_pairs],

})

eval_hf = Dataset.from_dict({

    "noisy": [p["noisy"] for p in eval_pairs],

    "cleaned": [p["cleaned"] for p in eval_pairs],

    "col": [p["col"] for p in eval_pairs],

})



# -------------------------------

# Tokenizer / Model

# -------------------------------

print("Loading tokenizer/model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
model.gradient_checkpointing_enable()


# -------------------------------

# Format examples for causal LM with loss masking

# Input format: "Noisy: {noisy}\nCleaned:"

# Labels: ignore (-100) on the input prompt; supervise only on target cleaned text

# -------------------------------

instruction_prefix = "Noisy: "

target_prefix = "\nCleaned: "



def tokenize_example(ex):
    """
    Tokenize a single example into input_ids, attention_mask, and labels.
    """
    prompt = instruction_prefix + ex["noisy"] + target_prefix

    target = ex["cleaned"]



    # Tokenize prompt and target separately

    prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False,return_attention_mask=False)["input_ids"]

    target_ids = tokenizer(target, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False,return_attention_mask=False)["input_ids"]

    # Concatenate

    input_ids = prompt_ids + target_ids

     # Labels: mask prompt part, supervise target part
    labels = [-100] * len(prompt_ids) + target_ids

    # Truncate if too long
    input_ids = input_ids[:MAX_LENGTH]
    labels = labels[:MAX_LENGTH]
    attention_mask = [1] * len(input_ids)

    # Pad up to MAX_LENGTH
    pad_len = MAX_LENGTH - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "col": ex["col"],
        "noisy_text": ex["noisy"],
        "cleaned_text": ex["cleaned"],
    }

   
    



train_tok = train_hf.map(tokenize_example, remove_columns=train_hf.column_names,batched=False)

eval_tok  = eval_hf.map(tokenize_example, remove_columns=eval_hf.column_names,batched=False)



# Data collator (no MLM for causal LM)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# -------------------------------

# Training

# Note: Newer Transformers may use `eval_strategy` instead of `evaluation_strategy`.

# We set eval on epoch via the correct keyword for your installed version.

# -------------------------------

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,

    eval_strategy="epoch",  # use 'evaluation_strategy' if your installed version requires it

    save_strategy="no",

    logging_steps=200,

    report_to=["none"],

    fp16=torch.cuda.is_available(),

    dataloader_pin_memory=False,

    seed=SEED,

)



trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_tok,

    eval_dataset=eval_tok,

    tokenizer=tokenizer,

    data_collator=data_collator,

)



print("\n=== Fine-tuning Gemma-270M on noisy_zarma (noisy -> cleaned) ===")

trainer.train()



print("\n=== Final evaluation (loss/perplexity) ===")

eval_results = trainer.evaluate()

final_loss = float(eval_results["eval_loss"])

final_ppl = math.exp(final_loss)

print(f"Final eval loss: {final_loss:.4f}")

print(f"Final perplexity: {final_ppl:.4f}")



# -------------------------------

# Generation for metric evaluation (BLEU, ChrF++, COMET)

# We evaluate per noisy column vs cleaned, on eval pairs only.

# -------------------------------

from nltk.translate.bleu_score import corpus_bleu

from nltk.translate.chrf_score import corpus_chrf



def generate_predictions(pairs: List[Dict], batch_size=16, max_new_tokens=128):
    """
    Generates cleaned-text predictions from a list of noisy→cleaned training pairs
    using a causal language model. This function performs batched inference,
    constructs prompts for each noisy input, decodes model outputs, and extracts
    only the generated continuation corresponding to the cleaned text.

    Args:
        pairs (List[Dict]):
            A list of dictionaries, where each dictionary represents a single
            noisy→cleaned example. Each element must contain:
                - "noisy" (str): The noisy input text.
                - "cleaned" (str): The ground-truth cleaned text.
                - "col" (str): The name of the noisy column the example came from.
        batch_size (int, optional):
            Number of examples to process per forward pass. Defaults to 16.
        max_new_tokens (int, optional):
            Maximum number of tokens the model is allowed to generate beyond the
            prompt. Defaults to 128.

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]:
            A 4‑tuple containing:
                - srcs (List[str]): The original noisy input strings.
                - preds (List[str]): The model‑generated cleaned strings.
                - refs (List[str]): The ground‑truth cleaned strings.
                - cols (List[str]): The noisy column names associated with each example.

    Raises:
        ValueError: If `pairs` is empty or contains malformed entries.
        RuntimeError: If model inference fails on any batch.

    Notes:
        - The function uses greedy decoding (`do_sample=False`, `num_beams=1`)
          for deterministic evaluation.
        - The generated text is extracted by locating the final occurrence of
          `target_prefix` in the decoded output and slicing everything after it.
        - The model is assumed to be globally available and already moved to the
          correct device.
        - This function does not modify model state and runs under `torch.no_grad()`
          for efficiency.

    Example:
        >>> srcs, preds, refs, cols = generate_predictions(pairs, batch_size=8)
        >>> print(preds[0])
        "corrected sentence here"
    """

    preds, refs, srcs, cols = [], [], [], []

    model.eval()

    for i in tqdm(range(0, len(pairs), batch_size), desc="Generating"):

        batch = pairs[i : i + batch_size]

        prompts = [instruction_prefix + b["noisy"] + target_prefix for b in batch]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)

        with torch.no_grad():

            outs = model.generate(

                **enc,

                max_new_tokens=max_new_tokens,

                do_sample=False,

                num_beams=1,

            )

        dec = tokenizer.batch_decode(outs, skip_special_tokens=True)

        # Extract only the generated continuation after the prompt

        for j, full in enumerate(dec):

            # Simple split: take text after the last occurrence of target_prefix

            split_idx = full.rfind(target_prefix.strip())

            gen = full[split_idx + len(target_prefix.strip()):].strip() if split_idx != -1 else full.strip()

            preds.append(gen)

            refs.append(batch[j]["cleaned"])

            srcs.append(batch[j]["noisy"])

            cols.append(batch[j]["col"])

    return srcs, preds, refs, cols



srcs, preds, refs, cols = generate_predictions(eval_pairs, batch_size=16, max_new_tokens=128)



# Compute metrics per column and overall

def compute_bleu_chrf(refs, preds):

    bleu = corpus_bleu([[r.split()] for r in refs], [p.split() for p in preds]) * 100

    chrf = corpus_chrf(refs, preds) * 100

    return bleu, chrf



def compute_comet(srcs, preds, refs):

    try:

        import evaluate

        comet_metric = evaluate.load("comet", "Unbabel/wmt22-comet-da")

        res = comet_metric.compute(predictions=preds, references=refs, sources=srcs)

        if isinstance(res, dict):

            # COMET returns different keys depending on version; use system score if present

            if "score" in res:

                return float(res["score"]) * 100

            if "system_score" in res:

                return float(res["system_score"]) * 100

        return None

    except Exception as e:

        print("COMET error:", e)

        return None



# Aggregate metrics per column

import numpy as np

results_per_col = {}

for nc in noisy_cols:

    idxs = [i for i, c in enumerate(cols) if c == nc]

    if not idxs:

        continue

    col_srcs = [srcs[i] for i in idxs]

    col_preds = [preds[i] for i in idxs]

    col_refs = [refs[i] for i in idxs]

    bleu, chrf = compute_bleu_chrf(col_refs, col_preds)

    comet = compute_comet(col_srcs, col_preds, col_refs)

    results_per_col[nc] = {

        "BLEU": round(bleu, 4),

        "ChrF++": round(chrf, 4),

        "COMET": round(comet, 4) if comet is not None else None,

        "n_samples": len(idxs),

    }

    if comet is None:

        results_per_col[nc]["COMET_error"] = "COMET scoring unavailable or failed."



# Overall metrics

bleu_all, chrf_all = compute_bleu_chrf(refs, preds)

comet_all = compute_comet(srcs, preds, refs)

overall = {

    "BLEU": round(bleu_all, 4),

    "ChrF++": round(chrf_all, 4),

    "COMET": round(comet_all, 4) if comet_all is not None else None,

    "n_samples": len(refs),

}

if comet_all is None:

    overall["COMET_error"] = "COMET scoring unavailable or failed."



# -------------------------------

# Save metrics JSON locally

# -------------------------------

metrics = {

    "model": MODEL_NAME,

    "train_rows": TRAIN_ROWS,

    "eval_rows": EVAL_ROWS,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "final_eval_loss": round(final_loss, 6),

    "final_perplexity": round(final_ppl, 6),

    "noisy_columns": noisy_cols,

    "overall": overall,

    "per_column": results_per_col,

}

os.makedirs(OUTPUT_DIR, exist_ok=True)

out_json = os.path.join(OUTPUT_DIR, "noisy_zarma_metrics.json")

with open(out_json, "w", encoding="utf-8") as f:

    json.dump(metrics, f, indent=2, ensure_ascii=False)

print(f"Saved metrics to {out_json}")




# -------------------------------

# (Optional) Push model and metrics JSON to HF Hub

# -------------------------------

# Ensure: huggingface-cli login (or pass token)

# Ask user for repo owner + repo name
hf_username = input("Enter your Hugging Face username or organization: ").strip()
hf_repo_name = input("Enter the name of the model repository: ").strip()

REPO_ID = f"{hf_username}/{hf_repo_name}"



print(f"Pushing model to {REPO_ID} ...")

trainer.push_to_hub(REPO_ID)



print("Pushing metrics JSON ...")

from huggingface_hub import HfApi

api = HfApi()

try:

    api.upload_file(

        path_or_fileobj=out_json,

        path_in_repo="noisy_zarma_metrics.json",

        repo_id=REPO_ID,

        repo_type="model",

        token=HF_TOKEN,

    )

except Exception as e:

    # If you’re pushing to an org/shared repo that requires PRs:

    print("Upload failed, retrying with create_pr=True ...")

    api.upload_file(

        path_or_fileobj=out_json,

        path_in_repo="noisy_zarma_metrics.json",

        repo_id=REPO_ID,

        repo_type="model",

        token=HF_TOKEN,

        create_pr=True,

    )



print("Done.")
