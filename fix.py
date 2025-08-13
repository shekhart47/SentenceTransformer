from transformers import AutoTokenizer
import numpy as np
import os

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=True)

def add_triplet_length(batch):
    # Concatenate to one long list to tokenize once.
    n = len(batch['anchor'])
    all_texts = batch['anchor'] + batch['positives'] + batch['negatives']

    # Fast batch tokenization; ask only for lengths
    out = tokenizer(
        all_texts,
        add_special_tokens=False,
        return_length=True,
        padding=False,
        truncation=False
    )
    lens = np.asarray(out['length'], dtype=np.int32)

    # Slice back by role
    anchor_len   = lens[:n]
    positive_len = lens[n:2*n]
    negative_len = lens[2*n:]

    # Max length across the triplet
    triplet_len = np.maximum.reduce([anchor_len, positive_len, negative_len])

    return {'triplet_length': triplet_len.tolist()}

# Choose large batch size; tune based on RAM/CPU cache
BATCHED_SIZE = 8192  # try 8k, 16k, 32k
NUM_PROC = max(2, os.cpu_count() - 2)

train_dataset = train_dataset.map(
    add_triplet_length,
    batched=True,
    batch_size=BATCHED_SIZE,
    num_proc=NUM_PROC,
    remove_columns=[],          # don't drop anything
    desc="Computing triplet lengths"
)
