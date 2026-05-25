"""

File: finetune_gemma_noisy_zarma_bleu_gleu.py

Aurthor: Huy Le (hl9082)

Purpose:

--------

Fine-tune google/gemma-3-270m (causal LM) to map noisy Zarma text → cleaned Zarma text.

This version:

- Creates one pair per row per noisy column (except "original").

- Ensures inference is done once per row (batching all noisy variants together).

- Trains on 45k rows, evaluates on 5k rows.

- Computes BLEU and GLEU scores.

"""



import os, math, json, random

import torch

import pandas as pd

from datasets import load_dataset, Dataset

from tqdm.auto import tqdm



from transformers import (

    AutoTokenizer, AutoModelForCausalLM,

    Trainer, TrainingArguments, DataCollatorForLanguageModeling

)



from nltk.translate.bleu_score import corpus_bleu

from nltk.translate.gleu_score import corpus_gleu

import getpass



# from huggingface_hub import HfApi

torch.cuda.empty_cache()

# -----------------------------

# Step 1: Config

# -----------------------------

HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ") # Secure user prompt for HF token

SEED = 42

random.seed(SEED); torch.manual_seed(SEED)

# OPTIMIZATION: Allow TF32 on Ampere+ GPUs for faster matmul/convs, minimal accuracy impact
torch.backends.cuda.matmul.allow_tf32 = True

try:
    # OPTIMIZATION: Set float32 matmul precision to 'high' to enable TF32 where applicable
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

MODEL_NAME = "google/gemma-3-270m"

OUTPUT_DIR = "./gemma_noisy_zarma_bleu_gleu"



LEARNING_RATE = 2e-4

EPOCHS = 3

BATCH_SIZE = 8

MAX_LENGTH = 128

TRAIN_ROWS = 45000

EVAL_ROWS = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# -----------------------------

# Step 2: Load dataset

# -----------------------------

hf_ds = load_dataset("27Group/noisy_zarma", split="train")

df = pd.DataFrame(hf_ds[: TRAIN_ROWS + EVAL_ROWS])



exclude_cols = {"cleaned", "original"}

noisy_cols = [c for c in df.columns if c not in exclude_cols]



df["cleaned"] = df["cleaned"].astype(str)

for c in noisy_cols: df[c] = df[c].astype(str)



train_df = df.iloc[:TRAIN_ROWS].copy()

eval_df  = df.iloc[TRAIN_ROWS: TRAIN_ROWS + EVAL_ROWS].copy()



# -----------------------------

# Step 3: Build pairs (per column)

# -----------------------------

def make_pairs(frame: pd.DataFrame):

    """
    Purpose:
        Build noisy→cleaned pairs for each row and noisy column.
    Parameters:
        frame (pd.DataFrame): DataFrame with noisy columns and 'cleaned'.
    Returns:
        List[Dict[str,str]]: Each dict has {"noisy","cleaned","col"}.
    """

    pairs = []

    for _, row in frame.iterrows():

        cleaned = row["cleaned"].strip()

        if not cleaned: continue

        for nc in noisy_cols:

            noisy = row[nc].strip()

            if not noisy: continue

            pairs.append({"noisy": noisy, "cleaned": cleaned, "col": nc})

    return pairs



train_pairs = make_pairs(train_df)

eval_pairs  = make_pairs(eval_df)



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



# -----------------------------

# Step 4: Tokenizer and model

# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)
model.config.use_cache = False # Disable use_cache when using gradient checkpointing

model.gradient_checkpointing_enable() # OPTIMIZATION: enable gradient checkpointing to reduce memory usage



# -----------------------------

# Step 5: Tokenization

# -----------------------------

instruction_prefix = "Noisy: "

target_prefix = "\nCleaned: "



def tokenize_example(ex):

    """
    Purpose:
        Convert a noisy→cleaned pair into tokenized input/labels.
    Parameters:
        ex (Dict): Example with keys {"noisy","cleaned","col"}.
    Returns:
        Dict:
            - "input_ids": token IDs for prompt+target, padded to MAX_LENGTH.
            - "attention_mask": 1 for real tokens, 0 for padding.
            - "labels": same as input_ids but with -100 masking for prompt tokens
                        (so loss is only computed on target tokens).
            - "col": the noisy column name (for later grouping).
            - "noisy_text": original noisy string (for debugging/eval).
            - "cleaned_text": reference cleaned string.
    """

    prompt = instruction_prefix + ex["noisy"] + target_prefix

    target = ex["cleaned"]



    prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH,

                           add_special_tokens=False)["input_ids"]

    target_ids = tokenizer(target, truncation=True, max_length=MAX_LENGTH,

                           add_special_tokens=False)["input_ids"]



    input_ids = prompt_ids + target_ids

    labels = [-100] * len(prompt_ids) + target_ids


#truncate if too long
    input_ids = input_ids[:MAX_LENGTH]

    labels = labels[:MAX_LENGTH]

    attn = [1] * len(input_ids)


#pad to max length
    pad_len = MAX_LENGTH - len(input_ids)

    if pad_len > 0:

        input_ids += [tokenizer.pad_token_id] * pad_len

        labels += [-100] * pad_len

        attn += [0] * pad_len



    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels,

            "col": ex["col"], "noisy_text": ex["noisy"], "cleaned_text": ex["cleaned"]}



train_tok = train_hf.map(tokenize_example, remove_columns=train_hf.column_names, batched=False)




eval_tok  = eval_hf.map(tokenize_example, remove_columns=eval_hf.column_names, batched=False)

# OPTIMIZATION: Pre-format to torch tensors to reduce collation overhead
train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# -----------------------------

# Step 6: Training setup

# -----------------------------

args = TrainingArguments(

    output_dir=OUTPUT_DIR,
    gradient_checkpointing=True,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,
    dataloader_num_workers=1,                # OPTIMIZATION: parallel data loading
    dataloader_pin_memory=True,              # OPTIMIZATION: faster host→GPU transfer

    num_train_epochs=EPOCHS,

    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",              # smooth schedule, stable at higher LR

    eval_strategy="epoch",

    save_strategy="no",

    logging_steps=2000,
   # disable_tqdm=True,                       # OPTIMIZATION: avoid progress bar overhead

    report_to=["none"],
    # Mixed precision for throughput; bf16 preferred on A100/H100, else fp16
    bf16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8),
    fp16=(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8),


    eval_accumulation_steps=4,                  #reduce eval-time memory

    gradient_accumulation_steps=1,           # keep micro-batch shape; higher would slow wall-clock

    optim="adamw_torch_fused",
    seed=SEED,
    # OPTIMIZATION: compile model for runtime speed (PyTorch 2.x)
    torch_compile=True,

)



trainer = Trainer(

    model=model, args=args,

    train_dataset=train_tok, eval_dataset=eval_tok,

    tokenizer=tokenizer, data_collator=collator,

)



# -----------------------------

# Step 7: Train and evaluate

# -----------------------------

trainer.train()

eval_results = trainer.evaluate()

loss = float(eval_results["eval_loss"])
ppl = math.exp(loss)

print(f"Eval loss={loss:.4f}, Perplexity={ppl:.4f}")



# -----------------------------

# Step 8: Optimized inference (one pass per row)

# -----------------------------

def generate_predictions(eval_df, batch_size=8, max_new_tokens=64):

    """
    Purpose:
        Generate predictions for noisy→cleaned mapping, ensuring each row's
        noisy columns are processed in one pass.

    Parameters:
        eval_df (pd.DataFrame): Evaluation split.
        batch_size (int): Number of rows per batch.
        max_new_tokens (int): Max tokens to generate.

    Returns:
        Tuple[List[str], List[str], List[str]]:
            - preds: list of model-generated cleaned strings.
            - refs: list of ground-truth cleaned strings (same length as preds).
            - cols: list of noisy column names corresponding to each prediction.
              This allows per-column metric analysis if desired.
    """


    preds, refs, cols = [], [], []

    model.eval()

    for i in tqdm(range(0, len(eval_df), batch_size), desc="Generating"):

        batch = eval_df.iloc[i:i+batch_size]

        # For each row, build prompts for all noisy columns

        prompts = []

        ref_texts = []

        col_names = []

        for _, row in batch.iterrows():

            cleaned = row["cleaned"]

            for nc in noisy_cols:

                noisy = row[nc].strip()

                if not noisy: continue

                prompts.append(instruction_prefix + noisy + target_prefix)

                ref_texts.append(cleaned)

                col_names.append(nc)

        if not prompts: continue

        # Why we need padding=True: ensures all prompts in batch have same length

        enc = tokenizer(prompts, return_tensors="pt", padding=True,

                        truncation=True, max_length=MAX_LENGTH).to(DEVICE)

        with torch.no_grad():

        # OPTIMIZATION: Greedy decoding (num_beams=1, do_sample=False)
            # is fastest and consistent for deterministic evaluation

            outs = model.generate(**enc, max_new_tokens=max_new_tokens,

                                  do_sample=False, num_beams=1)

        dec = tokenizer.batch_decode(outs, skip_special_tokens=True)

        for j, full in enumerate(dec):

            split_idx = full.rfind(target_prefix.strip())

            gen = full[split_idx+len(target_prefix.strip()):].strip() if split_idx!=-1 else full.strip()

            preds.append(gen); refs.append(ref_texts[j]); cols.append(col_names[j])

    return preds, refs, cols



preds, refs, cols = generate_predictions(eval_df)



# -----------------------------

# Step 9: BLEU and GLEU

# -----------------------------

bleu = corpus_bleu([[r.split()] for r in refs], [p.split() for p in preds]) * 100

gleu = corpus_gleu([[r.split()] for r in refs], [p.split() for p in preds]) * 100

print(f"BLEU={bleu:.2f}, GLEU={gleu:.2f}")

out_json = os.path.join(OUTPUT_DIR, "metrics_bleu_gleu.json")


metrics = {"loss": loss, "perplexity": ppl, "BLEU": bleu, "GLEU": gleu}

with open(out_json, "w") as f:

    json.dump(metrics, f, indent=2)


# -------------------------------



# (Optional) Push model and metrics JSON to HF Hub



# -------------------------------



# Ensure: huggingface-cli login (or pass token)

hf_username = input("Enter your Hugging Face username or organization: ").strip()
hf_repo_name = input("Enter the name of the model repository: ").strip()

REPO_ID = f"{hf_username}/{hf_repo_name}"



# Save final model + tokenizer

# -----------------------------

# Always save at the end so the folder has weights, config, and tokenizer

trainer.model.save_pretrained(OUTPUT_DIR)

trainer.tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Final model and tokenizer saved to {OUTPUT_DIR}")



print(f"Pushing model to {REPO_ID} ...")



trainer.push_to_hub(REPO_ID)







print("Pushing metrics JSON ...")



from huggingface_hub import HfApi



api = HfApi()



try:



    api.upload_file(



        path_or_fileobj=out_json,



        path_in_repo="metrics_bleu_gleu.json",



        repo_id=REPO_ID,



        repo_type="model",



        token=HF_TOKEN,



    )



except Exception as e:



    # If you’re pushing to an org/shared repo that requires PRs:



    print("Upload failed, retrying with create_pr=True ...")



    api.upload_file(



        path_or_fileobj=out_json,



        path_in_repo="metrics_bleu_gleu.json",



        repo_id=REPO_ID,



        repo_type="model",



        token=HF_TOKEN,



        create_pr=True,



    )

print("Done!")
