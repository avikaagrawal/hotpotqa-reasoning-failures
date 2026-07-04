# Write-up: Reasoning Failures in Multi-Hop RAG (HotpotQA)

*[Fill in the bracketed parts after running the pipeline — everything below is a
scaffold matching the exact criteria in the assignment rubric.]*

## What I built

A retrieval-augmented QA baseline on HotpotQA (distractor setting): per-question FAISS
retrieval over the 10 candidate paragraphs (top-k=3), answer generation via Gemini
2.5 Flash, scored with EM/F1. N=[500] questions, validation split.

Baseline result: EM = [__], F1 = [__]. Retrieval recall (all gold docs retrieved) = [__].

## Hypothesis

HotpotQA labels every question as **bridge** or **comparison**. Bridge questions need a
second document whose relevance only becomes clear after reading the first — the question
alone doesn't strongly signal it. Comparison questions name both target entities directly.

**Hypothesis:** single-hop dense retrieval underperforms specifically on bridge questions,
and the gap is a *retrieval* problem, not a *reasoning* problem — i.e., when the model
fails on bridge questions, it's mostly because it never saw the second document, not
because it reasoned poorly over evidence it did see.

## What I found

[Fill in with actual numbers from results/failure_analysis.json, e.g.:]

- Bridge questions: EM = [__] vs. comparison questions: EM = [__] — a [__] point gap.
- Of the wrong answers on bridge questions, [__]% were retrieval failures (gold docs never
  retrieved) vs. [__]% generation failures (evidence retrieved, model still wrong).
- Of the wrong answers on comparison questions, [__]% were retrieval failures vs. [__]%
  generation failures.

**Why this matters:** [e.g., "This confirms the hypothesis — bridge-question failures are
overwhelmingly a retrieval-stage problem, not a generation-stage one. A single dense-vector
query embedding of the raw question just doesn't carry enough signal to find a document
whose relevance is contingent on facts from another document. Throwing a better LLM at the
generation step wouldn't fix this — the evidence isn't reaching it in the first place."]

[If the data surprised you in a different direction — e.g. generation failures dominated
even on bridge questions, or comparison questions had unexpectedly low retrieval recall
too — say that instead, and explain what it implies.]

## What I'd try next

A specific, testable follow-up (not "more data" or "more compute"):

**Iterative retrieval:** after the first-pass retrieval, extract named entities from the
top-1 retrieved paragraph, re-query the retriever using (original question + those
entities), and take the union of both retrieval passes as context. Test whether this
closes the bridge-question retrieval-recall gap specifically, without hurting comparison-
question accuracy (which shouldn't need the second pass).

Concretely: re-run `failure_analysis.py` comparing single-pass vs. two-pass retrieval,
holding the generator fixed, and check whether `wrong_due_to_retrieval` drops for bridge
questions.
