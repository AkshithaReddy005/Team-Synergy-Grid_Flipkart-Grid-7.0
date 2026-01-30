from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import pandas as pd

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover
    RandomForestRegressor = None  # type: ignore

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore


class PersonalizationModel:
    def __init__(self, rf_path: Optional[str] = None, xgb_path: Optional[str] = None):
        self.rf = None
        self.xgb = None
        rf_path = rf_path or os.getenv("RF_MODEL_PATH", "models/rf_model.joblib")
        xgb_path = xgb_path or os.getenv("XGB_MODEL_PATH", "models/xgb_model.json")
        # Attempt to load RF
        if joblib and os.path.exists(rf_path):
            try:
                self.rf = joblib.load(rf_path)
            except Exception:
                self.rf = None
        # Attempt to load XGB
        if XGBRegressor and os.path.exists(xgb_path):
            try:
                self.xgb = XGBRegressor()
                self.xgb.load_model(xgb_path)
            except Exception:
                self.xgb = None

    def predict_score(self, features: List[float]) -> float:
        x = np.asarray(features, dtype=float).reshape(1, -1)
        preds = []
        if self.rf is not None:
            try:
                preds.append(float(self.rf.predict(x)[0]))
            except Exception:
                pass
        if self.xgb is not None:
            try:
                preds.append(float(self.xgb.predict(x)[0]))
            except Exception:
                pass
        if preds:
            return float(np.mean(preds))
        # Fallback deterministic score
        return float(np.mean(x)) if x.size > 0 else 0.0


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0))


def train_and_save_personalization(dataset_path: str, output_dir: str = "models") -> dict:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(dataset_path)
    # Heuristic feature set from catalog fields
    # Expected columns present in provided dataset: retail_price, discounted_price, product_rating, popularity, brand
    df["retail_price"] = pd.to_numeric(df.get("retail_price"), errors="coerce").fillna(0)
    df["discounted_price"] = pd.to_numeric(df.get("discounted_price"), errors="coerce").fillna(0)
    df["product_rating"] = pd.to_numeric(df.get("product_rating"), errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df.get("popularity"), errors="coerce").fillna(0)
    # Derived features
    df["discount_pct"] = (df["retail_price"] - df["discounted_price"]) / (df["retail_price"].replace(0, np.nan))
    df["discount_pct"] = df["discount_pct"].fillna(0).clip(0, 1)
    # Brand frequency as a proxy popularity feature
    brand_counts = df.get("brand").fillna("unknown").astype(str).value_counts()
    df["brand_freq"] = df.get("brand").fillna("unknown").astype(str).map(brand_counts).fillna(1).astype(float)

    features = np.column_stack([
        _safe_log(df["discounted_price"].to_numpy()),
        _safe_log(df["retail_price"].to_numpy()),
        df["discount_pct"].to_numpy(),
        df["product_rating"].to_numpy(),
        _safe_log(df["popularity"].to_numpy()),
        _safe_log(df["brand_freq"].to_numpy()),
    ])

    # Synthetic target: combine rating and normalized popularity as a proxy for utility
    target = 0.6 * df["product_rating"].to_numpy() + 0.4 * _safe_log(df["popularity"].to_numpy())

    results = {}
    # Train RF
    if RandomForestRegressor:
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(features, target)
        if joblib:
            rf_path = os.path.join(output_dir, "rf_model.joblib")
            joblib.dump(rf, rf_path)
            results["rf_model_path"] = rf_path

    # Train XGB
    if XGBRegressor:
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
        )
        xgb.fit(features, target)
        xgb_path = os.path.join(output_dir, "xgb_model.json")
        xgb.save_model(xgb_path)
        results["xgb_model_path"] = xgb_path

    return results
