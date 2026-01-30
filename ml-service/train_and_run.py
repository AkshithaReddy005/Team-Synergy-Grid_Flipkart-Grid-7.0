#!/usr/bin/env python3
"""
Convenience script to:
1) Generate synthetic sessions from the catalog
2) Train G-BERT on those sessions
3) Start the FastAPI service
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if result.returncode != 0:
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="../SRP-main/Dataset_Final_TeamSynergyGrid.csv")
    parser.add_argument("--sessions-out", default="data/synthetic_sessions.csv")
    parser.add_argument("--gbert-out", default="models/gbert")
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
