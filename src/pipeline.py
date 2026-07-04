"""
pipeline.py

Runs the full pipeline end-to-end:
  data_prep -> retrieve -> generate -> evaluate -> failure_analysis

Usage:
    python src/pipeline.py --n 500
"""

import argparse
import os

from data_prep import build_sample
from retrieve import run_retrieval
from generate import run_generation
from evaluate import run_evaluation
from failure_analysis import run_analysis
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    data_path = os.path.join(args.data_dir, "sample.json")
    retrieved_path = os.path.join(args.out_dir, "retrieved.json")
    predictions_path = os.path.join(args.out_dir, "predictions.json")
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    scored_path = predictions_path.replace(".json", "_scored.json")

    print("=== Step 1: data prep ===")
    sample = build_sample(args.n, args.split)
    with open(data_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"Saved {len(sample)} questions -> {data_path}")

    print("\n=== Step 2: retrieval ===")
    run_retrieval(data_path, retrieved_path, top_k=args.top_k)

    print("\n=== Step 3: generation ===")
    run_generation(retrieved_path, predictions_path)

    print("\n=== Step 4: evaluation ===")
    run_evaluation(predictions_path, metrics_path)

    print("\n=== Step 5: failure analysis (Part B) ===")
    run_analysis(scored_path, args.out_dir)

    print("\nDone. See results/ for metrics.json, failure_analysis.json, and charts.")


if __name__ == "__main__":
    main()
