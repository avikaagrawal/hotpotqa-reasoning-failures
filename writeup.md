# Write-up: Reasoning Failures in Multi-Hop RAG (HotpotQA)

## What I built

I implemented a Retrieval-Augmented Generation (RAG) baseline on the HotpotQA distractor dataset. Each question contains 10 candidate paragraphs (2 supporting and 8 distractors). I embedded these paragraphs using Sentence Transformers, indexed them with FAISS, retrieved the top-3 most relevant passages, and generated answers using Google Gemini. The generated answers were evaluated using Exact Match (EM) and F1 metrics.

The pipeline was evaluated on **50 validation questions**.

**Results**

- Overall Exact Match (EM): **12.0%**
- Overall F1 Score: **15.38%**
- Retrieval Recall (at least one gold document retrieved): **98%**
- Retrieval Recall (all supporting documents retrieved): **44%**

---

## Hypothesis

HotpotQA contains two reasoning types:

- **Bridge questions**, where answering requires retrieving information across multiple documents.
- **Comparison questions**, where both entities are explicitly mentioned in the question.

My hypothesis was that bridge questions would suffer primarily from retrieval failures because a single dense retrieval query often fails to retrieve all required supporting documents.

---

## What I found

The experimental results support this hypothesis.

### Bridge Questions

- Number of questions: **43**
- Exact Match: **6.98%**
- Retrieval Recall (all supporting documents): **39.53%**
- Wrong answers caused by retrieval failures: **65%**
- Wrong answers caused by generation failures: **35%**

### Comparison Questions

- Number of questions: **7**
- Exact Match: **42.86%**
- Retrieval Recall (all supporting documents): **71.43%**
- Wrong answers caused by retrieval failures: **50%**
- Wrong answers caused by generation failures: **50%**

These results show that bridge questions are considerably harder than comparison questions. Most bridge failures occurred because the retriever failed to retrieve every required supporting document. Once complete evidence was available, the language model performed substantially better.

This indicates that improving retrieval quality is likely to produce larger performance gains than simply replacing the language model.

---

## What I'd try next

As a follow-up experiment, I would implement **iterative retrieval**.

Instead of relying on a single retrieval pass, the system would:

1. Retrieve the top paragraph.
2. Extract important entities from that paragraph.
3. Expand the original query using these entities.
4. Perform a second retrieval pass.
5. Merge results from both retrieval stages before answer generation.

I would then compare bridge-question retrieval recall and EM/F1 against the current single-pass retrieval baseline to determine whether iterative retrieval reduces retrieval-related failures without negatively affecting comparison questions.

---

## Conclusion

This experiment demonstrates that retrieval quality is the primary bottleneck for bridge-style multi-hop reasoning. While the language model is capable of generating correct answers when given sufficient evidence, incomplete retrieval prevents it from accessing the information required for accurate reasoning.

Future work should therefore focus on retrieval improvements such as iterative retrieval, query expansion, or multi-stage retrieval pipelines rather than solely improving the generator.