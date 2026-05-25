"""

Name: Huy Le (hl9082)

Fine-tune a Gemma3-270M-like causal language model on 27Group/InstructLR_Generate_Datasets (subset ZarmaInstruct-50k)

to generate responses from instructions, with evaluation via GLEU.



Features:

- Hugging Face Trainer for causal LM fine-tuning (supervised instruction tuning)

- GLEU metric (macro averaged over validation)

- Full dataset usage (no sampling)

- Optimizations: bf16/fp16 mixed precision, gradient checkpointing, minimal logging, optional torch.compile

- W&B integration for experiment tracking

- Push final artifacts to Hugging Face Hub

- Avoid duplicate sample inferences



Assumptions:

- Dataset contains instruction + response fields under common names. The script auto-detects from:

  ["instruction", "prompt", "input"] for instructions and ["response", "output", "target", "completion"] for references.

- Model is a causal LM compatible with Transformers AutoModelForCausalLM.

"""



import os

import json

import random

from typing import Dict, Any, List, Optional, Tuple



import numpy as np

import torch

from datasets import load_dataset

from transformers import (

    AutoTokenizer,

    AutoModelForCausalLM,

    DataCollatorForLanguageModeling,

    Trainer,

    TrainingArguments

)

import wandb

from huggingface_hub import HfApi

from nltk.translate.gleu_score import sentence_gleu

import sys

import getpass


# -----------------------------

# Hard-coded tokens and IDs (replace with your own)

# -----------------------------


HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ")

WANDB_API_KEY = getpass.getpass("Enter your WANDB token: ")

# Ask user for repo owner + repo name
hf_username = input("Enter your Hugging Face username or organization: ").strip()
hf_repo_name = input("Enter the name of the model repository: ").strip()

HF_REPO_ID = f"{hf_username}/{hf_repo_name}"

MODEL_ID = "google/gemma-3-270m"  # e.g., "google/gemma-2-0.5b" or the correct Gemma3-270M ID



os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

os.environ["WANDB_DISABLE_CODE"] = "true"
os.environ["WANDB_DISABLE_GPU"] = "true"

#clean up cache

torch.cuda.empty_cache()

# -----------------------------

# Utility: reproducibility

# -----------------------------

def set_seed(seed: int = 42) -> None:

    """

    Set random seeds for reproducibility across Python, NumPy, and PyTorch.



    Parameters

    ----------

    seed : int, default=42

        Seed value for PRNGs.



    Returns

    -------

    None

    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)



# -----------------------------
# Preprocessing: use all columns
# -----------------------------
def make_tokenize_all_columns_fn(tokenizer, column_names: List[str], max_length: int = 512):
    """
    Create a batch tokenization function that:
    - Concatenates all columns into a labeled text block per example
    - Uses the full text for causal LM training (labels = input_ids)

    Parameters
    ----------
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for the causal LM.
    column_names : List[str]
        All column names in the dataset.
    max_length : int, default=512
        Max sequence length for truncation.
    Returns
    -------
    fn : callable
        Function suitable for datasets.map(batched=True).
    """
    def _fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        texts = []
        # Build a row-wise string: "col1: val1\ncol2: val2\n..."
        for i in range(len(batch[column_names[0]])):
            row_parts = []
            for col in column_names:
                val = batch[col][i]
                # Robust string conversion to avoid None/complex types breaking tokenization
                s = "" if val is None else str(val)
                row_parts.append(f"{col}: {s}")
            texts.append("\n".join(row_parts))
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            
        )
        
        return enc

    return _fn


# -----------------------------
# Metric: GLEU
# -----------------------------
def compute_gleu(preds: List[str], refs: List[str]) -> float:
    """
    Compute average sentence-level GLEU over predictions and references.

    Parameters
    ----------
    preds : List[str]
        Generated outputs from the model.
    refs : List[str]
        Reference outputs (ground truth).

    Returns
    -------
    avg_gleu : float
        Mean sentence GLEU score in [0, 1].
    """
    scores = []
    for hyp, ref in zip(preds, refs):
        hyp_tokens = hyp.split()
        ref_tokens = ref.split()
        scores.append(sentence_gleu([ref_tokens], hyp_tokens))
    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# Inference (deduplicated inputs)
# -----------------------------
def run_inference(model, tokenizer, texts: List[str], device: str, max_new_tokens: int = 32) -> List[str]:
    """
    Generate outputs for a list of concatenated row texts, deduplicating inputs to avoid repeated inference.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Trained causal LM.
    tokenizer : transformers.PreTrainedTokenizer
        Corresponding tokenizer.
    texts : List[str]
        Input texts (concatenated rows).
    device : str
        "cuda" or "cpu".
    max_new_tokens : int, default=32
        Max tokens to generate.

    Returns
    -------
    outputs : List[str]
        Generated outputs aligned to unique input order.
    """
    model.eval()
    uniq_texts = list(dict.fromkeys(texts))  # deduplicate while preserving order
    outputs = []
    with torch.no_grad():
        for t in uniq_texts:
            inputs = tokenizer(t, return_tensors="pt", padding=True,truncation=True, max_length=512).to(device)
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic for reproducibility
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False  # reduce KV cache reuse issues
            )
            # gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            gen_text = tokenizer.decode(gen_ids[0].detach().cpu().tolist(), skip_special_tokens=True)
            del gen_ids, inputs
            outputs.append(gen_text)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return outputs


# -----------------------------
# Evaluation helper (GLEU on validation)
# -----------------------------
def evaluate_gleu_all_columns(
    model,
    tokenizer,
    eval_ds,
    column_names: List[str],
    device: str,
    max_examples: Optional[int] = 50,
    max_new_tokens: int = 32
) -> float:
    """
    Evaluate model GLEU by generating on concatenated texts and comparing to a reference proxy.

    Because the dataset may not have an explicit single "reference" column for full-row concatenation,
    we use the concatenated text itself as the reference proxy for a consistency check baseline.
    If you have a dedicated ground-truth column, replace refs construction accordingly.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Trained causal LM.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer.
    eval_ds : datasets.Dataset
        Validation split.
    column_names : List[str]
        All column names to concatenate.
    device : str
        "cuda" or "cpu".
    max_examples : int, default=200
        Cap evaluation to first N examples for speed.
    max_new_tokens : int, default=128
        Max generation length.

    Returns
    -------
    gleu : float
        Average sentence-level GLEU in [0, 1].
    """
    n = len(eval_ds) if max_examples is None else min(max_examples, len(eval_ds))
    # Build concatenated texts for first n examples
    texts = []
    for i in range(n):
        parts = []
        for col in column_names:
            val = eval_ds[i][col]
            s = "" if val is None else str(val)
            parts.append(f"{col}: {s}")
        texts.append("\n".join(parts))

    # Generate deduplicated, then align back to original order
    preds_unique = run_inference(model, tokenizer, texts, device, max_new_tokens=max_new_tokens)
    uniq_map = {t: p for t, p in zip(list(dict.fromkeys(texts)), preds_unique)}
    preds = [uniq_map[t] for t in texts]

    # Reference proxy (baseline): use the original concatenation text
    refs = texts
    return compute_gleu(preds, refs)


# -----------------------------
# Main training function
# -----------------------------
def main():
    """
    End-to-end training, evaluation, saving, and hub push.
    """
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    ds = load_dataset("27Group/InstructLR_Generate_Datasets", "ZarmaInstruct-50k",download_mode="force_redownload")
    full_ds = ds["train"]
    if "validation" in ds:
       full_ds = full_ds.concatenate(ds["validation"])
    if "test" in ds:
       full_ds = full_ds.concatenate(ds["test"])

    # Perform 90/10 split
    split_ds = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]

    
    # Print dataset sizes
    print(f"Number of training rows: {len(train_ds)}")
   
    print(f"Number of test rows: {len(test_ds)}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)



    # Ensure we have a pad token for collation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "<|pad|>"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.config.use_cache = False

    # Remove unsupported generation config fields if present
    if hasattr(model, "generation_config"):
        if hasattr(model.generation_config, "top_p"):
            model.generation_config.top_p = None
        if hasattr(model.generation_config, "top_k"):
            model.generation_config.top_k = None

    

   

    # Speed/memory optimizations
    model.gradient_checkpointing_enable()
    
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16

    # Optional compile for runtime speed (if supported)
    '''
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            use_compile = True
        except Exception:
            use_compile = False
    else:
        use_compile = False
    '''

    use_compile = False

    model.to(device)

    # Preprocessing using all columns
    all_columns = full_ds.column_names
    tokenize_fn = make_tokenize_all_columns_fn(tokenizer, all_columns, max_length=512)
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=all_columns,num_proc=1)
    eval_tok = test_ds.map(tokenize_fn, batched=True, remove_columns=all_columns,num_proc=1)

    # Data collator for causal LM (labels already set)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

   

    # W&B setup
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    wandb.init(project="gemma3-270m-allcolumns", name="gemma3-270m-run")

    # TrainingArguments with optimizations
    training_args = TrainingArguments(
        output_dir="./gemma3_270m_out",
        num_train_epochs=3,
        per_device_train_batch_size=4,         # keep small for 270M + long sequences
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,          # effective batch size 32
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=200,
        report_to=["wandb"],
        seed=42,
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=False,   # Pinned memory can cause CUDA to reserve large blocks that never fully return
        dataloader_num_workers=0,            # single worker, less VRAM
        torch_compile=False,
        remove_unused_columns=True,
        
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    # Train
    train_out = trainer.train()

    # Free some memory before evaluation
    del train_tok
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Evaluate GLEU on validation (all columns)
    gleu = evaluate_gleu_all_columns(
        model=trainer.model,
        tokenizer=tokenizer,
        eval_ds=test_ds,
        column_names=all_columns,
        device=device,
        max_examples=50,         # cap for speed; set None to use full validation
        max_new_tokens=32
    )

    metrics = {
        "eval_gleu": float(gleu),
        "train_runtime": float(train_out.metrics.get("train_runtime", 0.0)),
        "train_samples_per_second": float(train_out.metrics.get("train_samples_per_second", 0.0)),
        "epoch": float(train_out.metrics.get("epoch", training_args.num_train_epochs)),
        "use_fp16": bool(use_fp16),
        "use_bf16": bool(use_bf16),
        "use_compile": bool(use_compile)
    }

    print(f"Validation GLEU: {metrics['eval_gleu']:.4f}")

    # Save artifacts locally
    out_dir = training_args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Save the model config directly
    trainer.model.config.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save a config snapshot
    config_snapshot = {
        "model_id": MODEL_ID,
        "max_length": 512,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "train_batch_size": training_args.per_device_train_batch_size,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "seed": training_args.seed,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": training_args.warmup_ratio,
        "use_fp16": use_fp16,
        "use_bf16": use_bf16,
        "use_compile": use_compile
    }
    with open(os.path.join(out_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, indent=2)

    

    # Sample inferences: take first 20 concatenated rows from test split (deduplicated)
    sample_texts = []
    sample_n = min(20, len(test_ds))
    for i in range(sample_n):
        parts = []
        for col in all_columns:
            val = test_ds[i][col]
            s = "" if val is None else str(val)
            parts.append(f"{col}: {s}")
        sample_texts.append("\n".join(parts))

    sample_preds_unique = run_inference(trainer.model, tokenizer, sample_texts, device, max_new_tokens=32)
    uniq_map = {t: p for t, p in zip(list(dict.fromkeys(sample_texts)), sample_preds_unique)}
    aligned_sample = [{"input_concat": t, "generated": uniq_map[t]} for t in sample_texts]

    with open(os.path.join(out_dir, "sample_inferences.json"), "w", encoding="utf-8") as f:
        json.dump(aligned_sample, f, indent=2)

    wandb.finish()

    # Aggressive cleanup of this process' GPU usage
    del aligned_sample, sample_preds_unique, uniq_map
    del model, trainer, tokenizer, eval_tok, test_ds, full_ds, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    '''
    import threading

    api=HfApi()

    def safe_upload():
        try:
            api.upload_folder(
                repo_id=HF_REPO_ID,
                folder_path=out_dir,
                repo_type="model",
                token=HF_TOKEN
            )
        except Exception as e:
            print(f"Upload failed: {e}")



    upload_thread = threading.Thread(target=safe_upload)
    upload_thread.start()
    upload_thread.join(timeout=60)  # force exit after 60 seconds
    '''
    os._exit(0)

    


# -----------------------------
# Call main
# -----------------------------
if __name__ == "__main__":
   main()



     
