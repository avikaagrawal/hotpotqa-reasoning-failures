"""
retrieve.py

Per-question dense retrieval over the 10 candidate paragraphs HotpotQA
provides (2 gold + 8 distractors). We build a fresh small FAISS index
per question rather than one global index, since the distractor setting
is designed to test retrieval within a bounded candidate set.

This keeps the baseline fast and lets Part B cleanly ask:
"did retrieval ever surface the gold/bridge paragraph in top-k?"
"""

import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class Retriever:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def retrieve(self, question: str, paragraphs: list, top_k: int = 3):
        """
        paragraphs: list of {"title": str, "text": str}
        returns: list of retrieved paragraph dicts, in ranked order,
                 each with a 'score' field added.
        """
        texts = [p["text"] for p in paragraphs]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized inner product
        index.add(embeddings.astype(np.float32))

        q_emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        scores, ids = index.search(q_emb.astype(np.float32), min(top_k, len(paragraphs)))

        results = []
        for score, idx in zip(scores[0], ids[0]):
            para = dict(paragraphs[idx])
            para["score"] = float(score)
            results.append(para)
        return results


def run_retrieval(data_path: str, out_path: str, top_k: int = 3):
    with open(data_path) as f:
        data = json.load(f)

    retriever = Retriever()
    output = []

    for row in data:
        retrieved = retriever.retrieve(row["question"], row["paragraphs"], top_k=top_k)
        retrieved_titles = [p["title"] for p in retrieved]

        gold_hit = any(t in retrieved_titles for t in row["supporting_titles"])
        gold_hit_all = all(t in retrieved_titles for t in row["supporting_titles"])

        output.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "supporting_titles": row["supporting_titles"],
                "retrieved": retrieved,
                "gold_hit_any": gold_hit,       # at least one gold paragraph retrieved
                "gold_hit_all": gold_hit_all,   # ALL gold paragraphs retrieved (needed for bridge Qs)
            }
        )

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Retrieved for {len(output)} questions -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sample.json")
    parser.add_argument("--out", type=str, default="results/retrieved.json")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    run_retrieval(args.data, args.out, args.top_k)
