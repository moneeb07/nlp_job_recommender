"""Metric tests with hand-computed expectations.

Every expected value below is worked out by hand from the formula in the
docstring of the function under test, so a mistake in the implementation cannot
be masked by comparing it against itself.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    _drop_self,
    average_precision,
    build_pseudo_profile,
    dcg_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_system,
    score_rankings,
)

RANKED = [1, 2, 3, 4, 5]
RELEVANT = {1, 3, 5}


# --- precision --------------------------------------------------------------

@pytest.mark.parametrize(
    "k, expected",
    [
        (1, 1 / 1),    # item 1 is relevant
        (2, 1 / 2),    # item 2 is not
        (3, 2 / 3),    # items 1 and 3
        (5, 3 / 5),    # items 1, 3 and 5
    ],
)
def test_precision_at_k(k, expected):
    assert precision_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)


def test_precision_divides_by_k_even_when_short():
    # Only five results exist, but P@10 still divides by 10: returning too few
    # results should not flatter the score.
    assert precision_at_k(RANKED, RELEVANT, 10) == pytest.approx(0.3)


def test_precision_of_a_perfect_ranking_is_one():
    assert precision_at_k([1, 3, 5], {1, 3, 5}, 3) == 1.0


def test_precision_with_no_hits_is_zero():
    assert precision_at_k([7, 8, 9], RELEVANT, 3) == 0.0


def test_precision_at_zero_k_is_zero():
    assert precision_at_k(RANKED, RELEVANT, 0) == 0.0


# --- recall -----------------------------------------------------------------

@pytest.mark.parametrize("k, expected", [(1, 1 / 3), (3, 2 / 3), (5, 3 / 3)])
def test_recall_at_k(k, expected):
    assert recall_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)


def test_recall_caps_at_one():
    assert recall_at_k(RANKED, RELEVANT, 100) == 1.0


def test_recall_with_no_relevant_items_is_zero():
    assert recall_at_k(RANKED, set(), 5) == 0.0


# --- DCG and NDCG -----------------------------------------------------------

def test_dcg_sums_discounted_hits():
    # Hits at ranks 1 and 3: 1/log2(2) + 1/log2(4) = 1.0 + 0.5
    assert dcg_at_k([1, 2, 3], {1, 3}, 3) == pytest.approx(1.5)


def test_dcg_discount_matches_the_formula():
    # A single hit at rank 4 is worth 1/log2(5).
    assert dcg_at_k([9, 8, 7, 1], {1}, 4) == pytest.approx(1 / math.log2(5))


def test_ndcg_normalises_against_the_ideal_ranking():
    # DCG = 1.5; IDCG packs both hits at the top = 1/log2(2) + 1/log2(3).
    ideal = 1.0 + 1 / math.log2(3)
    assert ndcg_at_k([1, 2, 3], {1, 3}, 3) == pytest.approx(1.5 / ideal)


def test_ndcg_of_a_perfect_ranking_is_one():
    assert ndcg_at_k([1, 3, 5, 2, 4], {1, 3, 5}, 3) == pytest.approx(1.0)


def test_ndcg_is_one_when_all_relevant_come_first():
    assert ndcg_at_k([1, 2], {1, 2}, 2) == pytest.approx(1.0)


def test_ndcg_rewards_ranking_the_hit_earlier():
    """The property that distinguishes NDCG from precision@k."""
    early = [1, 9, 8, 7, 6]
    late = [9, 8, 7, 6, 1]

    assert precision_at_k(early, {1}, 5) == precision_at_k(late, {1}, 5)
    assert ndcg_at_k(early, {1}, 5) > ndcg_at_k(late, {1}, 5)


def test_ndcg_with_no_relevant_items_is_zero():
    assert ndcg_at_k(RANKED, set(), 5) == 0.0


def test_ndcg_stays_within_bounds():
    assert 0.0 <= ndcg_at_k(RANKED, RELEVANT, 5) <= 1.0


def test_ndcg_ideal_accounts_for_the_cutoff():
    # Five relevant items but k=2, so the ideal is only two hits, and finding
    # two in the top two is a perfect score at that depth.
    assert ndcg_at_k([1, 2, 9], {1, 2, 3, 4, 5}, 2) == pytest.approx(1.0)


# --- reciprocal rank --------------------------------------------------------

@pytest.mark.parametrize(
    "ranked, expected",
    [([1, 2, 3], 1.0), ([2, 1, 3], 0.5), ([2, 4, 1], 1 / 3), ([2, 4, 6], 0.0)],
)
def test_reciprocal_rank(ranked, expected):
    assert reciprocal_rank(ranked, {1}) == pytest.approx(expected)


def test_reciprocal_rank_uses_the_first_hit_only():
    assert reciprocal_rank([9, 1, 3], {1, 3}) == pytest.approx(0.5)


# --- average precision ------------------------------------------------------

def test_average_precision_averages_precision_at_each_hit():
    # Hits at ranks 1 and 3: (1/1 + 2/3) / 2 relevant items.
    assert average_precision([1, 2, 3, 4], {1, 3}) == pytest.approx((1.0 + 2 / 3) / 2)


def test_average_precision_of_a_perfect_ranking_is_one():
    assert average_precision([1, 2, 9], {1, 2}) == pytest.approx(1.0)


def test_average_precision_penalises_relevant_items_never_returned():
    """Dividing by |relevant| means a missed item scores zero, not nothing."""
    found_both = average_precision([1, 2], {1, 2})
    found_one = average_precision([1, 9], {1, 2})

    assert found_both == pytest.approx(1.0)
    assert found_one == pytest.approx(0.5)


def test_average_precision_with_no_relevant_items_is_zero():
    assert average_precision([1, 2, 3], set()) == 0.0


# --- aggregates -------------------------------------------------------------

def test_mean_reciprocal_rank_averages_over_queries():
    # 1.0 and 0.5 -> 0.75
    assert mean_reciprocal_rank([[1, 2], [2, 1]], [{1}, {1}]) == pytest.approx(0.75)


def test_mean_average_precision_averages_over_queries():
    # AP 1.0 and AP 0.5 -> 0.75
    assert mean_average_precision([[1], [9, 1]], [{1}, {1}]) == pytest.approx(0.75)


def test_aggregates_of_no_queries_are_zero():
    assert mean_reciprocal_rank([], []) == 0.0
    assert mean_average_precision([], []) == 0.0


def test_score_rankings_returns_every_metric():
    scores = score_rankings([RANKED], [RELEVANT], k_values=(5,))

    assert set(scores) == {"P@5", "R@5", "NDCG@5", "MRR", "MAP"}
    assert scores["P@5"] == pytest.approx(0.6)


def test_score_rankings_averages_across_queries():
    scores = score_rankings([[1, 2], [2, 1]], [{1}, {1}], k_values=(1,))

    # P@1 is 1.0 for the first query and 0.0 for the second.
    assert scores["P@1"] == pytest.approx(0.5)


def test_a_perfect_system_scores_one_everywhere():
    rankings = [[1, 2, 3], [4, 5, 6]]
    relevant = [{1, 2, 3}, {4, 5, 6}]
    scores = score_rankings(rankings, relevant, k_values=(3,))

    assert scores["P@3"] == pytest.approx(1.0)
    assert scores["R@3"] == pytest.approx(1.0)
    assert scores["NDCG@3"] == pytest.approx(1.0)
    assert scores["MRR"] == pytest.approx(1.0)
    assert scores["MAP"] == pytest.approx(1.0)


def test_a_system_that_finds_nothing_scores_zero_everywhere():
    scores = score_rankings([[7, 8, 9]], [{1, 2}], k_values=(3,))

    assert all(value == 0.0 for value in scores.values())


# --- evaluation harness -----------------------------------------------------

def test_a_query_never_scores_against_itself():
    """Every posting is its own nearest neighbour; leaving it in inflates
    every metric for every system."""
    results = [{"index": 5}, {"index": 1}, {"index": 9}]

    assert [r["index"] for r in _drop_self(results, 5)] == [1, 9]


def test_drop_self_leaves_other_results_untouched():
    results = [{"index": 1}, {"index": 2}]

    assert _drop_self(results, 99) == results


JOBS = pd.DataFrame(
    {
        "job_id": [1, 2],
        "job_text": ["Data Analyst at Acme. SQL and Python.", "Chef at Bistro."],
        "skills": [np.array(["Python", "SQL"]), np.array([])],
        "min_years_experience": [4.0, None],
    }
)


def test_pseudo_profile_uses_the_posting_as_the_query():
    profile = build_pseudo_profile(JOBS, 0, fallback_years=3.0)

    assert profile["profile_text"] == JOBS.iloc[0]["job_text"]
    assert profile["skills"] == ["python", "sql"]
    assert profile["years_of_experience"] == 4.0


def test_pseudo_profile_falls_back_when_experience_is_unstated():
    # "Unstated" means unknown, not zero: zeroing it would punish the query
    # against every posting that does state a requirement.
    profile = build_pseudo_profile(JOBS, 1, fallback_years=3.0)

    assert profile["years_of_experience"] == 3.0
    assert profile["skills"] == []


def test_run_system_excludes_self_and_respects_depth():
    queries = [{"query_position": 0}, {"query_position": 1}]

    def ranker(position):
        return [{"index": index} for index in range(5)]

    rankings, latency = run_system(ranker.__name__, ranker, queries, depth=3)

    assert rankings[0] == [1, 2, 3]   # 0 removed, then truncated to 3
    assert rankings[1] == [0, 2, 3]   # 1 removed
    assert latency >= 0.0
