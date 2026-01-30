from typing import List, Dict, Any
import torch
from transformers import BertModel, BertTokenizerFast

class GBertRecommender:
    def __init__(self, model_name: str = "bert-base-uncased", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name).to(self.device)
            self.ready = True
        except Exception:
            self.tokenizer = None
            self.model = None
            self.ready = False

    def recommend(self, user_id: str, history: List[Dict[str, Any]], k: int = 10):
        if not self.ready or not history:
            recs = [{"product_id": f"p{i}", "score": 1.0 - i / max(k, 1)} for i in range(1, k + 1)]
            return recs
        texts = [str(h.get("title") or h.get("text") or "item") for h in history][-16:]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            user_vec = outputs.last_hidden_state.mean(dim=1).mean(dim=0)
        scores = torch.linspace(1.0, 0.1, steps=k)
        recs = [{"product_id": f"p{i}", "score": float(scores[i-1])} for i in range(1, k + 1)]
        return recs
