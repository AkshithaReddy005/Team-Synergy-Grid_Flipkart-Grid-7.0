#!/usr/bin/env python3
"""
Train a G-BERT style sequence model from user interaction logs.
Usage: python train_gbert.py --sessions data/synthetic_sessions.csv --catalog ../SRP-main/Dataset_Final_TeamSynergyGrid.csv --output models/gbert
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

# -------------------------------------------------
# 1. Load data and build sequences
# -------------------------------------------------
def load_catalog(catalog_path: str) -> Dict[str, Dict]:
    df = pd.read_csv(catalog_path)
    catalog = {}
    for _, row in df.iterrows():
        pid = str(row["pid"])
        catalog[pid] = {
            "pid": pid,
            "title": str(row.get("product_name", "")),
            "brand": str(row.get("brand", "")),
            "category": str(row.get("product_category_tree", "")),
        }
    return catalog

def load_sessions(sessions_path: str) -> pd.DataFrame:
    df = pd.read_csv(sessions_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["user_id", "timestamp"])
    return df

def build_sequences(df: pd.DataFrame, catalog: Dict, max_len: int = 16) -> List[Dict]:
    """
    For each user, create training examples:
    - Input: sequence of product tokens up to max_len
    - Label: next product_id (as classification label)
    """
    pid2idx = {pid: i for i, pid in enumerate(catalog.keys())}
    idx2pid = {i: pid for pid, i in pid2idx.items()}
    examples = []
    for user_id, user_df in df.groupby("user_id"):
        pids = user_df["product_id"].tolist()
        # slide window over user's ordered interactions
        for i in range(1, len(pids)):
            seq = pids[max(0, i - max_len):i]
            target_pid = pids[i]
            if target_pid not in pid2idx:
                continue
            examples.append({"user_id": user_id, "sequence": seq, "target_pid": target_pid, "label": pid2idx[target_pid]})
    return examples, pid2idx, idx2pid

# -------------------------------------------------
# 2. Tokenize product metadata into a single string per product
# -------------------------------------------------
def build_product_texts(catalog: Dict) -> Dict[str, str]:
    pid2text = {}
    for pid, rec in catalog.items():
        # Simple concatenation; you can improve with separators
        text = f"{rec['title']} {rec['brand']} {rec['category']}".strip()
        pid2text[pid] = text
    return pid2text

# -------------------------------------------------
# 3. PyTorch Dataset
# -------------------------------------------------
class GBertDataset(Dataset):
    def __init__(self, examples: List[Dict], pid2text: Dict, tokenizer: BertTokenizerFast, max_len: int = 16):
        self.examples = examples
        self.pid2text = pid2text
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq_texts = [self.pid2text[pid] for pid in ex["sequence"]]
        # Join with separator token
        seq_text = " [SEP] ".join(seq_texts)
        encoding = self.tokenizer(
            seq_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        # Squeeze batch dim
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.long)
        return item

# -------------------------------------------------
# 4. Train
# -------------------------------------------------
def train_gbert(
    examples: List[Dict],
    pid2text: Dict,
    idx2pid: Dict,
    output_dir: str,
    model_name: str = "bert-base-uncased",
    max_len: int = 16,
    epochs: int = 3,
    batch_size: int = 32,
):
    num_labels = len(idx2pid)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_dataset = GBertDataset(examples, pid2text, tokenizer, max_len=max_len)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        evaluation_strategy="no",
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    # Save model, tokenizer, and mappings
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(Path(output_dir) / "idx2pid.json", "w") as f:
        json.dump(idx2pid, f)
    with open(Path(output_dir) / "pid2text.json", "w") as f:
        json.dump(pid2text, f)
    print(f"Saved G-BERT model to {output_dir}")

# -------------------------------------------------
# 5. CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", required=True, help="Path to sessions CSV")
    parser.add_argument("--catalog", required=True, help="Path to product catalog CSV")
    parser.add_argument("--output", default="models/gbert", help="Output directory")
    parser.add_argument("--max_len", type=int, default=16, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    catalog = load_catalog(args.catalog)
    sessions_df = load_sessions(args.sessions)
    examples, pid2idx, idx2pid = build_sequences(sessions_df, catalog, max_len=args.max_len)
    pid2text = build_product_texts(catalog)

    print(f"Loaded {len(catalog)} products, {len(sessions_df)} interactions, {len(examples)} training examples")
    train_gbert(examples, pid2text, idx2pid, args.output, max_len=args.max_len, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
