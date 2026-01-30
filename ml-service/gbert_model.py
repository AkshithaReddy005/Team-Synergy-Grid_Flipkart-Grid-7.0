from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification

class GBertRecommender:
    def __init__(self, model_dir: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir or os.getenv("GBERT_MODEL_DIR", "models/gbert")
        self.ready = False
        self.model = None
        self.tokenizer = None
        self.idx2pid = {}
        self.pid2text = {}
        try:
            if Path(self.model_dir).exists():
                self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)
                self.model = BertForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
                self.model.eval()
                with open(Path(self.model_dir) / "idx2pid.json") as f:
                    self.idx2pid = json.load(f)
                with open(Path(self.model_dir) / "pid2text.json") as f:
                    self.pid2text = json.load(f)
                self.ready = True
                print(f"[GBert] Loaded trained model from {self.model_dir}")
            else:
                print("[GBert] Model dir not found; using fallback")
        except Exception as exc:
            print(f"[GBert] Failed to load model: {exc}")

    def recommend(self, user_id: str, history: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, float]]:
        if not self.ready or not history:
            # Fallback deterministic
            return [{"product_id": f"p{i}", "score": 1.0 - i / max(k, 1)} for i in range(1, k + 1)]

        # Build user sequence from history (assume history contains product_id or title)
        seq_texts = []
        for h in history[-16:]:
            pid = str(h.get("product_id") or "")
            if pid in self.pid2text:
                seq_texts.append(self.pid2text[pid])
            else:
                # fallback to title if available
                title = str(h.get("title") or "")
                seq_texts.append(title)
        if not seq_texts:
            return [{"product_id": f"p{i}", "score": 1.0 - i / max(k, 1)} for i in range(1, k + 1)]

        seq_text = " [SEP] ".join(seq_texts)
        inputs = self.tokenizer(
            seq_text,
            truncation=True,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, num_labels)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_labels,)

        # Map top-k indices back to product_ids
        topk_vals, topk_idx = torch.topk(probs, min(k, len(probs)))
        recs = []
        for score, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            pid = self.idx2pid.get(str(idx))
            if pid:
                recs.append({"product_id": pid, "score": float(score)})
        return recs[:k]

    def rerank_candidates(self, user_id: str, history: List[Dict[str, Any]], candidate_pids: List[str]) -> List[Dict[str, float]]:
        """Score a given list of product IDs for the user."""
        if not self.ready or not history or not candidate_pids:
            # Return candidates with neutral scores
            return [{"product_id": pid, "score": 1.0} for pid in candidate_pids]

        seq_texts = []
        for h in history[-16:]:
            pid = str(h.get("product_id") or "")
            if pid in self.pid2text:
                seq_texts.append(self.pid2text[pid])
            else:
                title = str(h.get("title") or "")
                seq_texts.append(title)
        if not seq_texts:
            return [{"product_id": pid, "score": 1.0} for pid in candidate_pids]

        seq_text = " [SEP] ".join(seq_texts)
        inputs = self.tokenizer(
            seq_text,
            truncation=True,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, num_labels)
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # (num_labels,)

        # Pull scores for candidates
        pid2idx = {v: k for k, v in self.idx2pid.items()}
        results = []
        for pid in candidate_pids:
            idx = pid2idx.get(pid)
            if idx is not None and idx < len(probs):
                results.append({"product_id": pid, "score": float(probs[idx])})
            else:
                results.append({"product_id": pid, "score": 0.0})
        return results
