#!/usr/bin/env python3
"""
Generate synthetic user interaction sessions from the product catalog.
Output: CSV with columns user_id, timestamp, product_id, action (view/click/purchase)
"""

import csv
import random
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

def load_products(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize product identifiers and text fields
    df["pid"] = df["pid"].astype(str)
    df["product_name"] = df["product_name"].astype(str).fillna("")
    df["brand"] = df["brand"].astype(str).fillna("")
    df["product_category_tree"] = df["product_category_tree"].astype(str).fillna("")
    return df

def sample_product_by_popularity(df: pd.DataFrame) -> str:
    weights = df["popularity"].fillna(1)
    weights = weights + 1e-6
    probs = weights / weights.sum()
    return df.sample(1, weights=probs).iloc[0]["pid"]

def simulate_sessions(df: pd.DataFrame, n_users: int = 500, max_events_per_user: int = 30) -> list[dict]:
    events = []
    start = datetime.utcnow()
    for uid in range(1, n_users + 1):
        user_id = f"user_{uid:04d}"
        # Random start time within last 90 days
        user_start = start - timedelta(days=random.randint(0, 90), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        n_events = random.randint(3, max_events_per_user)
        # Simulate a session with some category/brand affinity
        affinity_brand = random.choice(df["brand"].dropna().unique())
        affinity_category = random.choice(df["product_category_tree"].dropna().unique())
        for i in range(n_events):
            # 70% affinity, 30% random
            if random.random() < 0.7:
                filtered = df[(df["brand"] == affinity_brand) | (df["product_category_tree"].str.contains(affinity_category, na=False, regex=False))]
                if filtered.empty:
                    pid = sample_product_by_popularity(df)
                else:
                    pid = sample_product_by_popularity(filtered)
            else:
                pid = sample_product_by_popularity(df)
            # Choose action: view (60%), click (30%), purchase (10%)
            action = random.choices(["view", "click", "purchase"], weights=[60, 30, 10])[0]
            ts = user_start + timedelta(minutes=random.randint(0, 120 * i))
            events.append({
                "user_id": user_id,
                "timestamp": ts.isoformat() + "Z",
                "product_id": pid,
                "action": action,
            })
    # Sort globally by timestamp
    events.sort(key=lambda e: e["timestamp"])
    return events

def write_sessions(events: list[dict], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "timestamp", "product_id", "action"])
        writer.writeheader()
        writer.writerows(events)
    print(f"Wrote {len(events)} events to {out_path}")

if __name__ == "__main__":
    catalog_path = "../SRP-main/Dataset_Final_TeamSynergyGrid.csv"
    out_path = "data/synthetic_sessions.csv"
    df = load_products(catalog_path)
    events = simulate_sessions(df, n_users=500, max_events_per_user=30)
    write_sessions(events, out_path)
