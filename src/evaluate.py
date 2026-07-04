"""
evaluate.py

Standard EM / F1 scoring (same normalization approach as the official
HotpotQA / SQuAD eval scripts), plus retrieval-hit-rate stats.

Outputs results/metrics.json with:
  - overall EM / F1
  - retrieval recall (gold_hit_any, gold_hit_all)
"""

import json
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def run_evaluation(predictions_path: str, out_path: str):
    with open(predictions_path) as f:
        data = json.load(f)

    for row in data:
        row["em"] = exact_match(row["predicted_answer"], row["answer"])
        row["f1"] = f1_score(row["predicted_answer"], row["answer"])

    n = len(data)
    overall_em = sum(r["em"] for r in data) / n
    overall_f1 = sum(r["f1"] for r in data) / n
    retrieval_recall_any = sum(r["gold_hit_any"] for r in data) / n
    retrieval_recall_all = sum(r["gold_hit_all"] for r in data) / n

    metrics = {
        "n_questions": n,
        "overall_em": round(overall_em, 4),
        "overall_f1": round(overall_f1, 4),
        "retrieval_recall_any_gold": round(retrieval_recall_any, 4),
        "retrieval_recall_all_gold": round(retrieval_recall_all, 4),
    }

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(predictions_path.replace(".json", "_scored.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(json.dumps(metrics, indent=2))
    return data, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default="results/predictions.json")
    parser.add_argument("--out", type=str, default="results/metrics.json")
    args = parser.parse_args()

    run_evaluation(args.predictions, args.out)
