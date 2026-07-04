# HotpotQA Reasoning Failures

A retrieval-augmented QA baseline on [HotpotQA](https://hotpotqa.github.io/) (distractor
setting), built to study **where and why multi-hop reasoning fails** rather than just to
hit a benchmark number.

CortiqoLabs take-home — Research & Solution Development Intern.

## Why HotpotQA

Each question ships with its own bounded set of 10 candidate paragraphs (2 gold, 8
distractors), and is pre-labeled as either:
- **bridge** — the answer requires chaining through a second document whose relevance
  isn't obvious from the question alone
- **comparison** — both target entities are named directly in the question

That built-in labeling makes it possible to test a specific hypothesis about *why*
retrieval-augmented QA fails on multi-hop questions, instead of just reporting one
aggregate accuracy number.

## Pipeline

```
data_prep.py        -> samples N questions from HotpotQA, saves to data/sample.json
retrieve.py          -> per-question FAISS retrieval (top-k) over the 10 candidate paragraphs
generate.py           -> answers the question from retrieved context (Gemini, or a
                          no-API-key extractive fallback so the pipeline always runs)
evaluate.py           -> EM / F1 scoring + retrieval recall
failure_analysis.py   -> Part B: bridge vs. comparison breakdown, retrieval- vs.
                          generation-failure attribution, charts
pipeline.py            -> runs all of the above end-to-end
```

## Setup

```bash
pip install -r requirements.txt

# Optional but recommended -- without this the generator falls back to a
# weak extractive heuristic so the pipeline still runs for free:
export GEMINI_API_KEY="your-key-here"
```

## Run

```bash
python src/pipeline.py --n 500
```

Or step by step:

```bash
python src/data_prep.py --n 500 --out data/sample.json
python src/retrieve.py --data data/sample.json --out results/retrieved.json --top_k 3
python src/generate.py --retrieved results/retrieved.json --out results/predictions.json
python src/evaluate.py --predictions results/predictions.json --out results/metrics.json
python src/failure_analysis.py --scored results/predictions_scored.json
```

## Outputs (in `results/`)

- `metrics.json` — overall EM / F1, retrieval recall
- `predictions_scored.json` — every question with retrieved docs, predicted answer, EM/F1
- `failure_analysis.json` — accuracy and failure attribution split by question type
- `em_by_type.png`, `failure_attribution.png` — charts

## Part B: hypothesis and what to look for

**Hypothesis:** dense retrieval fails disproportionately on bridge questions, because the
embedding of the question alone doesn't strongly match the second (bridge) document until
the first document has been read.

`failure_analysis.py` separates every wrong answer into:
- **retrieval failure** — the model never retrieved all the gold paragraphs it needed, so
  the generator had no real chance
- **generation failure** — the model had all the evidence and still answered wrong

See `writeup.md` for the actual result and interpretation once the pipeline has been run.

## Notes on scope

- Uses a sample (default N=500) of the validation split, not the full HotpotQA set — this
  is a baseline/exploration exercise, not a leaderboard submission.
- The extractive fallback generator (used when `GEMINI_API_KEY` is unset) is intentionally
  weak; it exists purely so the repo is runnable without a paid API key. The real numbers
  in `writeup.md` were produced with Gemini as the generator.
