"""Information-retrieval metrics, implemented from scratch.

Every metric takes `ranked` (the system's ordered result list) and `relevant`
(the set of items the labels call relevant), and returns a number in [0, 1].

No library shortcuts: the formulas are written out so they can be explained.

    python -m src.evaluate

runs the benchmark and the ablation and writes both to `results/`.
"""

import json
import math
import statistics
import time

import pandas as pd

from src.baseline_tfidf import TFIDFRecommender
from src.build_eval_set import load_eval_set
from src.config import (
    JOBS_PARQUET,
    RESULTS_DIR,
    WEIGHT_EXPERIENCE,
    WEIGHT_SEMANTIC,
    WEIGHT_SKILL_OVERLAP,
)
from src.recommender import JobRecommender, as_skill_set

K_VALUES = (5, 10, 20)
EVAL_DEPTH = 20


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def precision_at_k(ranked, relevant, k: int) -> float:
    """Of the top k results, what fraction are relevant?

        P@k = |{relevant} ∩ {top k retrieved}| / k

    Always divides by k, even when fewer than k results came back, so a system
    that returns too little is penalised rather than flattered.
    """
    if k <= 0:
        return 0.0

    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / k


def recall_at_k(ranked, relevant, k: int) -> float:
    """Of everything relevant, what fraction made it into the top k?

        R@k = |{relevant} ∩ {top k retrieved}| / |{relevant}|
    """
    if not relevant:
        return 0.0

    hits = sum(1 for item in ranked[:k] if item in relevant)
    return hits / len(relevant)


def dcg_at_k(ranked, relevant, k: int) -> float:
    """Discounted cumulative gain with binary relevance.

        DCG@k = Σ(i = 1..k) rel_i / log2(i + 1)

    `rel_i` is 1 when the item at rank i is relevant and 0 otherwise. The
    log2(i + 1) divisor is the discount: a hit at rank 1 is worth 1/log2(2) = 1,
    the same hit at rank 10 only 1/log2(11) ≈ 0.29. That is what makes this a
    *ranking* measure — precision@k would score both identically.
    """
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )


def ndcg_at_k(ranked, relevant, k: int) -> float:
    """DCG divided by the best DCG achievable for this query.

        NDCG@k = DCG@k / IDCG@k

    IDCG is the DCG of the perfect ranking — every relevant item packed into the
    top positions. Without that normalisation a query with ten relevant postings
    would always out-score one with a single relevant posting, and the average
    across queries would be meaningless.
    """
    if not relevant:
        return 0.0

    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(ranked, relevant, k) / ideal_dcg


def reciprocal_rank(ranked, relevant) -> float:
    """1 / (rank of the first relevant result), or 0 if there is none.

    Answers "how far down before the user sees something useful?" — the right
    question when someone only looks at the first result or two.
    """
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(ranked, relevant) -> float:
    """Mean of the precisions measured at each relevant hit.

        AP = (1 / |{relevant}|) * Σ(k = 1..n) P@k * rel_k

    Dividing by the total number of relevant items (not by the number found)
    means relevant items the system never returned count as zero, so AP rewards
    both finding relevant results and ranking them early.
    """
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant)


def mean_reciprocal_rank(rankings, relevant_sets) -> float:
    """MRR — reciprocal rank averaged over every query."""
    if not rankings:
        return 0.0
    return sum(
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(rankings, relevant_sets)
    ) / len(rankings)


def mean_average_precision(rankings, relevant_sets) -> float:
    """MAP — average precision averaged over every query."""
    if not rankings:
        return 0.0
    return sum(
        average_precision(ranked, relevant)
        for ranked, relevant in zip(rankings, relevant_sets)
    ) / len(rankings)


def score_rankings(rankings, relevant_sets, k_values=K_VALUES) -> dict:
    """Every metric, averaged over all queries."""
    scores = {}
    for k in k_values:
        scores[f"P@{k}"] = sum(
            precision_at_k(r, s, k) for r, s in zip(rankings, relevant_sets)
        ) / len(rankings)
        scores[f"R@{k}"] = sum(
            recall_at_k(r, s, k) for r, s in zip(rankings, relevant_sets)
        ) / len(rankings)
        scores[f"NDCG@{k}"] = sum(
            ndcg_at_k(r, s, k) for r, s in zip(rankings, relevant_sets)
        ) / len(rankings)

    scores["MRR"] = mean_reciprocal_rank(rankings, relevant_sets)
    scores["MAP"] = mean_average_precision(rankings, relevant_sets)
    return scores


# --------------------------------------------------------------------------
# Running the systems over the evaluation queries
# --------------------------------------------------------------------------

def _drop_self(results, query_position):
    """A posting used as a query trivially retrieves itself — remove it."""
    return [r for r in results if r["index"] != query_position]


def build_pseudo_profile(jobs: pd.DataFrame, position: int, fallback_years: float) -> dict:
    """Treat one posting as if it were a candidate's resume.

    Experience is the posting's own stated requirement; where a posting states
    none, the corpus median is used rather than zero, because "unstated" means
    unknown and zeroing it would punish half the queries for a missing field.
    """
    job = jobs.iloc[position]
    years = job["min_years_experience"]
    return {
        "profile_text": job["job_text"],
        "skills": sorted(as_skill_set(job["skills"])),
        "years_of_experience": float(years) if pd.notna(years) else fallback_years,
    }


def run_system(name, ranker, queries, depth=EVAL_DEPTH) -> tuple[list, float]:
    """Run one system over every query, returning rankings and mean latency."""
    rankings = []
    elapsed = []

    for query in queries:
        started = time.perf_counter()
        results = ranker(query["query_position"])
        elapsed.append((time.perf_counter() - started) * 1000)
        trimmed = _drop_self(results, query["query_position"])[:depth]
        rankings.append([result["index"] for result in trimmed])

    return rankings, sum(elapsed) / len(elapsed)


def run_benchmark(jobs, eval_set, sparse, dense) -> dict:
    queries = eval_set["queries"]
    relevant_sets = [set(query["relevant"]) for query in queries]

    stated = jobs["min_years_experience"].dropna()
    fallback_years = float(stated.median()) if len(stated) else 0.0

    profiles = {
        query["query_position"]: build_pseudo_profile(
            jobs, query["query_position"], fallback_years
        )
        for query in queries
    }

    # The transformer weights load lazily on first use. Without this the first
    # timed system absorbs several seconds of model loading and reports a
    # latency higher than the pipeline that does strictly more work.
    dense.embedder.encode_single("warm up")

    systems = {
        "TF-IDF baseline": lambda position: sparse.recommend(
            jobs.iloc[position]["job_text_clean"], top_k=EVAL_DEPTH + 1
        ),
        "Sentence-BERT (retrieval only)": lambda position: dense.retrieve_only(
            profiles[position], top_k=EVAL_DEPTH + 1
        ),
        "Sentence-BERT + re-ranking": lambda position: dense.recommend(
            profiles[position], top_k=EVAL_DEPTH + 1
        ),
    }

    report = {}
    for name, ranker in systems.items():
        rankings, latency = run_system(name, ranker, queries)
        scores = score_rankings(rankings, relevant_sets)
        scores["latency_ms"] = latency
        report[name] = scores

    return report


def run_ablation(jobs, eval_set, dense) -> dict:
    """Vary the re-ranking weights to show 0.5/0.3/0.2 was not arbitrary."""
    queries = eval_set["queries"]
    relevant_sets = [set(query["relevant"]) for query in queries]

    stated = jobs["min_years_experience"].dropna()
    fallback_years = float(stated.median()) if len(stated) else 0.0
    profiles = {
        query["query_position"]: build_pseudo_profile(
            jobs, query["query_position"], fallback_years
        )
        for query in queries
    }

    variants = {
        "semantic only": (1.0, 0.0, 0.0),
        "skills only": (0.0, 1.0, 0.0),
        "experience only": (0.0, 0.0, 1.0),
        "equal thirds": (1 / 3, 1 / 3, 1 / 3),
        "semantic heavy": (0.70, 0.20, 0.10),
        "skills heavy": (0.30, 0.50, 0.20),
        f"default ({WEIGHT_SEMANTIC}/{WEIGHT_SKILL_OVERLAP}/{WEIGHT_EXPERIENCE})": (
            WEIGHT_SEMANTIC, WEIGHT_SKILL_OVERLAP, WEIGHT_EXPERIENCE
        ),
    }

    default_label = (
        f"default ({WEIGHT_SEMANTIC}/{WEIGHT_SKILL_OVERLAP}/{WEIGHT_EXPERIENCE})"
    )

    report = {}
    per_query = {}
    for label, (semantic, skills, experience) in variants.items():
        variant = dense.with_weights(semantic, skills, experience)

        rankings, _ = run_system(
            label,
            lambda position, variant=variant: variant.recommend(
                profiles[position], top_k=EVAL_DEPTH + 1
            ),
            queries,
        )
        scores = score_rankings(rankings, relevant_sets, k_values=(10,))
        values = [ndcg_at_k(r, s, 10) for r, s in zip(rankings, relevant_sets)]
        per_query[label] = values

        # 50 queries is a small sample. Reporting the standard error stops a
        # difference inside the noise from being read as a real improvement.
        spread = statistics.stdev(values) if len(values) > 1 else 0.0
        report[label] = {
            "weights": (semantic, skills, experience),
            "NDCG@10": scores["NDCG@10"],
            "stderr": spread / math.sqrt(len(values)) if values else 0.0,
            "P@10": scores["P@10"],
            "MAP": scores["MAP"],
        }

    # How often each variant actually beats the shipped default, query by query.
    baseline = per_query.get(default_label, [])
    for label, values in per_query.items():
        wins = sum(1 for value, base in zip(values, baseline) if value > base + 1e-9)
        losses = sum(1 for value, base in zip(values, baseline) if value < base - 1e-9)
        report[label]["wins"] = wins
        report[label]["losses"] = losses

    return report


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

_LIMITATIONS = (
    "> **The labels are heuristic, not human judgements.** A posting counts as\n"
    "> relevant when it shares a normalized title family with the query *and* its\n"
    "> extracted skills overlap by more than 0.4 Jaccard. Those same signals feed\n"
    "> the retrievers, so these numbers measure internal consistency rather than\n"
    "> true relevance, and the absolute values mean little. They are comparable\n"
    "> between systems, which is what the table is for. See\n"
    "> `src/build_eval_set.py` for the full list of limitations.\n"
)


def benchmark_markdown(report: dict, eval_set: dict) -> str:
    metrics = [f"NDCG@{k}" for k in K_VALUES] + [f"P@{k}" for k in K_VALUES]
    metrics += [f"R@{k}" for k in K_VALUES] + ["MRR", "MAP"]

    metadata = eval_set["metadata"]
    lines = [
        "# Retrieval benchmark",
        "",
        f"{metadata['n_queries']} queries over {metadata['corpus_size']:,} postings, "
        f"{metadata['mean_relevant_per_query']:.1f} relevant per query on average. "
        f"Results are evaluated to depth {EVAL_DEPTH}; a posting never retrieves itself.",
        "",
        _LIMITATIONS,
        "",
        "| System | " + " | ".join(metrics) + " | latency |",
        "|---|" + "---|" * (len(metrics) + 1),
    ]

    for name, scores in report.items():
        row = " | ".join(f"{scores[metric]:.3f}" for metric in metrics)
        lines.append(f"| {name} | {row} | {scores['latency_ms']:.0f} ms |")

    return "\n".join(lines) + "\n"


def ablation_markdown(report: dict) -> str:
    lines = [
        "# Re-ranking weight ablation",
        "",
        "Each row re-runs the full pipeline with different weights on the three "
        "re-ranking components, so the shipped default can be compared against "
        "the alternatives instead of being asserted.",
        "",
        _LIMITATIONS,
        "",
        "| Weights (semantic / skills / experience) | NDCG@10 | ± s.e. | P@10 | MAP | wins / losses vs default |",
        "|---|---|---|---|---|---|",
    ]

    for label, scores in sorted(
        report.items(), key=lambda item: -item[1]["NDCG@10"]
    ):
        semantic, skills, experience = scores["weights"]
        lines.append(
            f"| {label} — {semantic:.2f} / {skills:.2f} / {experience:.2f} "
            f"| {scores['NDCG@10']:.3f} | {scores['stderr']:.3f} "
            f"| {scores['P@10']:.3f} | {scores['MAP']:.3f} "
            f"| {scores['wins']} / {scores['losses']} |"
        )

    lines += [
        "",
        "The standard errors overlap heavily across the middle of this table, and "
        "the win/loss columns show most variants change the ranking for only a "
        "handful of the 50 queries. Treat the ordering as indicative, not settled: "
        "a difference of a few points here is well inside the noise of a 50-query "
        "heuristic sample.",
    ]

    return "\n".join(lines) + "\n"


def print_table(report: dict) -> None:
    metrics = ["NDCG@5", "NDCG@10", "P@5", "P@10", "R@10", "MRR", "MAP"]
    header = f"{'System':<32}" + "".join(f"{metric:>9}" for metric in metrics) + f"{'latency':>11}"
    print(header)
    print("-" * len(header))
    for name, scores in report.items():
        row = "".join(f"{scores[metric]:>9.3f}" for metric in metrics)
        print(f"{name:<32}{row}{scores['latency_ms']:>8.0f} ms")


def main() -> None:
    jobs = pd.read_parquet(JOBS_PARQUET)
    eval_set = load_eval_set(jobs=jobs)
    sparse = TFIDFRecommender.load()
    dense = JobRecommender.load()

    print(f"Evaluating {len(eval_set['queries'])} queries…\n")
    report = run_benchmark(jobs, eval_set, sparse, dense)
    print_table(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "benchmark.md").write_text(benchmark_markdown(report, eval_set))
    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(report, indent=2))

    print("\nRunning weight ablation…\n")
    ablation = run_ablation(jobs, eval_set, dense)
    for label, scores in sorted(ablation.items(), key=lambda item: -item[1]["NDCG@10"]):
        print(
            f"  {label:<34} NDCG@10 {scores['NDCG@10']:.3f} "
            f"± {scores['stderr']:.3f}   (better on {scores['wins']} queries, "
            f"worse on {scores['losses']})"
        )

    (RESULTS_DIR / "ablation.md").write_text(ablation_markdown(ablation))
    (RESULTS_DIR / "ablation.json").write_text(json.dumps(ablation, indent=2))

    print(f"\nWrote {RESULTS_DIR / 'benchmark.md'} and {RESULTS_DIR / 'ablation.md'}")


if __name__ == "__main__":
    main()
