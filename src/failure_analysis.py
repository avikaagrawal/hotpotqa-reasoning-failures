"""
failure_analysis.py  (Part B)

Hypothesis:
  Single-hop dense retrieval fails disproportionately on "bridge" questions
  (where the answer requires a second document whose relevance only becomes
  clear after reading the first) compared to "comparison" questions (where
  both target entities are named directly in the question).

What this script does:
  1. Splits scored results by HotpotQA's own `type` label: bridge vs comparison.
  2. Reports EM/F1 per type.
  3. For each wrong answer, classifies the failure as:
       - "retrieval failure"  -> gold_hit_all is False (model never saw
         all the evidence it needed, so the generator had no chance)
       - "generation failure" -> gold_hit_all is True but the answer was
         still wrong (model saw the evidence but reasoned incorrectly)
  4. Saves a summary table + a bar chart to results/.

Run after evaluate.py, using the *_scored.json file.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def classify_failure(row):
    if row["em"] == 1:
        return "correct"
    return "retrieval_failure" if not row["gold_hit_all"] else "generation_failure"


def run_analysis(scored_path: str, out_dir: str = "results"):
    with open(scored_path) as f:
        data = json.load(f)

    for row in data:
        row["failure_type"] = classify_failure(row)

    by_type = {}
    for row in data:
        by_type.setdefault(row["type"], []).append(row)

    summary = {}
    for qtype, rows in by_type.items():
        n = len(rows)
        em = sum(r["em"] for r in rows) / n
        f1 = sum(r["f1"] for r in rows) / n
        retrieval_recall = sum(r["gold_hit_all"] for r in rows) / n

        failures = [r for r in rows if r["em"] == 0]
        n_fail = len(failures)
        retrieval_fail = sum(1 for r in failures if r["failure_type"] == "retrieval_failure")
        generation_fail = sum(1 for r in failures if r["failure_type"] == "generation_failure")

        summary[qtype] = {
            "n": n,
            "em": round(em, 4),
            "f1": round(f1, 4),
            "gold_hit_all_rate": round(retrieval_recall, 4),
            "n_wrong": n_fail,
            "wrong_due_to_retrieval": retrieval_fail,
            "wrong_due_to_generation": generation_fail,
            "pct_wrong_from_retrieval": round(retrieval_fail / n_fail, 4) if n_fail else None,
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "failure_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

    # --- chart: EM by question type ---
    types = list(summary.keys())
    ems = [summary[t]["em"] for t in types]

    plt.figure(figsize=(6, 4))
    plt.bar(types, ems)
    plt.ylabel("Exact Match")
    plt.title("Accuracy by HotpotQA question type")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "em_by_type.png"))
    plt.close()

    # --- chart: failure attribution (retrieval vs generation) by type ---
    retrieval_pct = [
        summary[t]["pct_wrong_from_retrieval"] or 0 for t in types
    ]
    plt.figure(figsize=(6, 4))
    plt.bar(types, retrieval_pct)
    plt.ylabel("Fraction of wrong answers caused by retrieval miss")
    plt.title("Failure attribution: retrieval vs generation")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "failure_attribution.png"))
    plt.close()

    print(f"\nSaved summary + charts to {out_dir}/")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=str, default="results/predictions_scored.json")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    run_analysis(args.scored, args.out_dir)
