from __future__ import annotations
import os
from pathlib import Path
from typing import List

import pandas as pd
from symspellpy import SymSpell, Verbosity


class SpellCorrector:
    def __init__(self, max_edit_distance: int = 2, prefix_length: int = 7):
        self.sym = SymSpell(max_edit_distance=max_edit_distance, prefix_length=prefix_length)
        self.ready = False

    def build_from_catalog(self, catalog_path: str):
        df = pd.read_csv(catalog_path)
        corpus_cols = []
        for col in ["product_name", "name", "brand", "product_category_tree", "category", "description"]:
            if col in df.columns:
                corpus_cols.append(col)
        if not corpus_cols:
            self.ready = True
            return
        # Build frequency dictionary from tokens
        for col in corpus_cols:
            series = df[col].dropna().astype(str).str.lower()
            for text in series:
                for token in text.replace("/", " ").replace("|", " ").replace(">", " ").split():
                    token = token.strip(".,:;!()[]{}\"'")
                    if token:
                        self.sym.create_dictionary_entry(token, 1)
        self.ready = True

    def correct(self, text: str, max_suggestions: int = 1) -> str:
        if not text:
            return text
        if not self.ready:
            return text
        suggestions = self.sym.lookup_compound(text.lower(), max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
        # Fallback to word-level
        words: List[str] = []
        for w in text.split():
            sug = self.sym.lookup(w.lower(), Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
            words.append(sug[0].term if sug else w)
        return " ".join(words)
