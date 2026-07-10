#!/usr/bin/env python3
"""
Convenience script to:
1) Generate synthetic sessions from the catalog
2) Train G-BERT on those sessions
3) Start the FastAPI service
"""

import os
import subprocess
import argparse
from pathlib import Path

# Paths (Updated to new monorepo structure)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CATALOG_PATH = BASE_DIR.parent / "data" / "Dataset_Final_TeamSynergyGrid.csv"
SYNTHETIC_SESSIONS_PATH = DATA_DIR / "synthetic_sessions.csv"
GBERT_MODEL_DIR = DATA_DIR / "gbert_model"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(CATALOG_PATH))
    parser.add_argument("--sessions-out", default=str(SYNTHETIC_SESSIONS_PATH))
    parser.add_argument("--gbert-out", default=str(GBERT_MODEL_DIR))
    parser.add_argument("--skip-train", action="store_true", help="Skip training and just start the service")
    parser.add_argument("--skip-gen", action="store_true", help="Skip session generation")
    args = parser.parse_args()

    if not args.skip_gen:
        run_cmd(f"python generate_synthetic_sessions.py --catalog {args.catalog} --out {args.sessions_out}")

    if not args.skip_train:
        run_cmd(f"python train_gbert.py --sessions {args.sessions_out} --catalog {args.catalog} --output {args.gbert_out} --epochs 2")

    # Start the service
    run_cmd("uvicorn main:app --host 0.0.0.0 --port 8001 --reload")

if __name__ == "__main__":
    main()
