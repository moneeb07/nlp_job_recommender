# Retrieval benchmark

50 queries over 5,000 postings, 2.9 relevant per query on average. Results are evaluated to depth 20; a posting never retrieves itself.

> **The labels are heuristic, not human judgements.** A posting counts as
> relevant when it shares a normalized title family with the query *and* its
> extracted skills overlap by more than 0.4 Jaccard. Those same signals feed
> the retrievers, so these numbers measure internal consistency rather than
> true relevance, and the absolute values mean little. They are comparable
> between systems, which is what the table is for. See
> `src/build_eval_set.py` for the full list of limitations.


| System | NDCG@5 | NDCG@10 | NDCG@20 | P@5 | P@10 | P@20 | R@5 | R@10 | R@20 | MRR | MAP | latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF baseline | 0.290 | 0.329 | 0.370 | 0.124 | 0.080 | 0.055 | 0.328 | 0.445 | 0.580 | 0.356 | 0.278 | 4 ms |
| Sentence-BERT (retrieval only) | 0.277 | 0.307 | 0.339 | 0.112 | 0.078 | 0.051 | 0.326 | 0.411 | 0.515 | 0.313 | 0.262 | 43 ms |
| Sentence-BERT + re-ranking | 0.433 | 0.464 | 0.482 | 0.188 | 0.126 | 0.072 | 0.526 | 0.601 | 0.657 | 0.479 | 0.384 | 44 ms |

## Offline build cost

Query latency is only half the picture — a dense index has to be built before it can serve anything. Measured when each artefact was built.

| System | build time | artefact |
|---|---|---|
| TF-IDF baseline | 1.9 s (fit) | 11.5 MB, 50,000-term vocabulary |
| Sentence-BERT + FAISS | 194.4 s (embed) + 1 ms (index) | 7.7 MB, 5,000 × 384 vectors |
