"""
generate.py

Takes retrieved paragraphs + question, produces an answer.

Primary path: Gemini (matches the stack used in your other RAG project).
Fallback path: if GEMINI_API_KEY isn't set, uses a simple extractive
heuristic (most similar sentence to the question) so the pipeline still
runs end-to-end for free / offline. This makes grading/reproducing the
baseline painless for anyone without a key.

Set your key before running:
    export GEMINI_API_KEY="your-key-here"
"""

import json
import os
import time

from sentence_transformers import SentenceTransformer, util

USE_GEMINI = bool(os.environ.get("GEMINI_API_KEY"))

if USE_GEMINI:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")

PROMPT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer is not contained in the context, respond with "unknown".
Give a short, direct answer (a few words), not a full sentence.

Context:
{context}

Question: {question}
Answer:"""


_fallback_model = None


def _get_fallback_model():
    global _fallback_model
    if _fallback_model is None:
        _fallback_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _fallback_model


def generate_answer_gemini(question: str, context: str, retries: int = 3) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    for attempt in range(retries):
        try:
            response = GEMINI_MODEL.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if attempt == retries - 1:
                return f"[ERROR: {e}]"
            time.sleep(2 ** attempt)
    return "[ERROR: unreachable]"


def generate_answer_fallback(question: str, context: str) -> str:
    """
    No-API-key fallback: splits context into sentences, embeds them plus
    the question, and returns the sentence most similar to the question.
    This is a weak baseline on purpose -- it exists so the pipeline is
    runnable without cost/credentials, not to be a good QA model.
    """
    model = _get_fallback_model()
    sentences = [s.strip() for s in context.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return "unknown"

    q_emb = model.encode(question, convert_to_tensor=True)
    s_emb = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, s_emb)[0]
    best_idx = int(scores.argmax())
    return sentences[best_idx]


def generate_answer(question: str, retrieved_paragraphs: list) -> str:
    context = "\n\n".join(
        f"[{p['title']}] {p['text']}" for p in retrieved_paragraphs
    )
    if USE_GEMINI:
        return generate_answer_gemini(question, context)
    return generate_answer_fallback(question, context)


def run_generation(retrieved_path: str, out_path: str):
    with open(retrieved_path) as f:
        data = json.load(f)

    if not USE_GEMINI:
        print("GEMINI_API_KEY not set -- using extractive fallback generator.")
        print("For the real baseline, set the key and re-run this step.")

    for row in data:
        row["predicted_answer"] = generate_answer(row["question"], row["retrieved"])

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated answers for {len(data)} questions -> {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieved", type=str, default="results/retrieved.json")
    parser.add_argument("--out", type=str, default="results/predictions.json")
    args = parser.parse_args()

    run_generation(args.retrieved, args.out)
