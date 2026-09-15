# Re-ranking weight ablation

Each row re-runs the full pipeline with different weights on the three re-ranking components, so the shipped default can be compared against the alternatives instead of being asserted.

> **The labels are heuristic, not human judgements.** A posting counts as
> relevant when it shares a normalized title family with the query *and* its
> extracted skills overlap by more than 0.4 Jaccard. Those same signals feed
> the retrievers, so these numbers measure internal consistency rather than
> true relevance, and the absolute values mean little. They are comparable
> between systems, which is what the table is for. See
> `src/build_eval_set.py` for the full list of limitations.


| Weights (semantic / skills / experience) | NDCG@10 | ± s.e. | P@10 | MAP | wins / losses vs default |
|---|---|---|---|---|---|
| semantic heavy — 0.70 / 0.20 / 0.10 | 0.483 | 0.052 | 0.122 | 0.407 | 11 / 9 |
| default (0.5/0.3/0.2) — 0.50 / 0.30 / 0.20 | 0.464 | 0.051 | 0.126 | 0.384 | 0 / 0 |
| skills only — 0.00 / 1.00 / 0.00 | 0.453 | 0.051 | 0.120 | 0.377 | 6 / 14 |
| equal thirds — 0.33 / 0.33 / 0.33 | 0.439 | 0.051 | 0.126 | 0.358 | 2 / 9 |
| skills heavy — 0.30 / 0.50 / 0.20 | 0.434 | 0.049 | 0.124 | 0.350 | 2 / 14 |
| semantic only — 1.00 / 0.00 / 0.00 | 0.307 | 0.053 | 0.078 | 0.262 | 3 / 27 |
| experience only — 0.00 / 0.00 / 1.00 | 0.259 | 0.050 | 0.072 | 0.223 | 1 / 30 |

The standard errors overlap heavily across the middle of this table, and the win/loss columns show most variants change the ranking for only a handful of the 50 queries. Treat the ordering as indicative, not settled: a difference of a few points here is well inside the noise of a 50-query heuristic sample.
