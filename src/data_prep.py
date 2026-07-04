"""
data_prep.py

Downloads a sample of HotpotQA (distractor setting) and saves it locally
as a JSON list so the rest of the pipeline doesn't need network access
on every run.

HotpotQA distractor setting gives each question its own small corpus:
2 "gold" supporting paragraphs + 8 distractor paragraphs. This is what
makes it a nice fit for a per-question retrieval baseline instead of
needing one giant global index.

Usage:
    python src/data_prep.py --n 500 --split validation --out data/sample.json
"""

import argparse
import json
import os
import random

from datasets import load_dataset


def build_sample(n: int, split: str, seed: int = 42):
    print(f"Loading HotpotQA ({split}, distractor)...")
    ds = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)

    random.seed(seed)
    n = min(n, len(ds))
    indices = random.sample(range(len(ds)), n)

    sample = []
    for i in indices:
        row = ds[i]
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]

        paragraphs = []
        for title, sents in zip(titles, sentences):
            paragraphs.append({"title": title, "text": " ".join(sents)})

        supporting_titles = list(set(row["supporting_facts"]["title"]))

        sample.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row.get("type", "unknown"),        # bridge / comparison
                "level": row.get("level", "unknown"),       # easy/medium/hard
                "paragraphs": paragraphs,                   # 10 candidate paragraphs
                "supporting_titles": supporting_titles,      # gold paragraph titles (2 for bridge)
            }
        )

    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of questions to sample")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--out", type=str, default="data/sample.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample = build_sample(args.n, args.split, args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Saved {len(sample)} questions to {args.out}")
    type_counts = {}
    for row in sample:
        type_counts[row["type"]] = type_counts.get(row["type"], 0) + 1
    print("Question type breakdown:", type_counts)


if __name__ == "__main__":
    main()
