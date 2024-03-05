"""
"""

from collections import defaultdict
import logging
from typing import Any
from statistics import median


def _calculate_f1_per_template(
    scores: dict[Any],
) -> dict[Any]:
    scores_per_template = defaultdict(list)
    for example_id, values in scores.items():
        for template_id, value in values.items():
            scores_per_template[template_id].append(value["weighted avg"]["f1-score"])

    average_scores_per_template = defaultdict(list)
    for template_id, values in scores_per_template.items():
        average_scores_per_template[template_id] = sum(values) / (len(values) + 1e-20)

    return average_scores_per_template


def _calculate_f1_per_template_doc(
    scores: dict[Any],
) -> dict[Any]:
    average_scores_per_template = defaultdict(list)
    for template_id, filename2scores in scores.items():
        values = [
            scores["weighted avg"]["f1-score"] for scores in filename2scores.values()
        ]
        average_scores_per_template[template_id] = sum(values) / (len(values) + 1e-20)

    return average_scores_per_template


def _calculate_score_per_template(
    scores: dict[Any],
) -> dict[Any]:
    scores_per_template = defaultdict(list)
    for example_id, values in scores.items():
        for template_id, value in values.items():
            scores_per_template[template_id].append(value)

    average_scores_per_template = defaultdict(list)
    for template_id, values in scores_per_template.items():
        average_scores_per_template[template_id] = sum(values) / (len(values) + 1e-20)

    return average_scores_per_template


def _calculate_match_per_template_doc(
    scores: dict[Any],
) -> dict[Any]:
    average_match_per_template = defaultdict(list)
    for template_id, matches in scores.items():
        average_match_per_template[template_id] = sum(matches) / (len(matches) + 1e-20)

    return average_match_per_template


def _calculate_score_torque_per_template_doc(
    scores: dict[Any],
) -> dict[Any]:
    average_score_per_template = defaultdict(list)
    for template_id, values in scores.items():
        _values = [value[-1] for value in values]
        average_score_per_template[template_id] = sum(_values) / (len(_values) + 1e-20)

    return average_score_per_template


def calculate_range_template(report: dict[Any], metric_name: str) -> dict[Any]:
    """
    Calculate min, median, max per document
    taking min/max for each example w/ multiple prompt templates

    """
    match metric_name:
        case "pairwise-accuracy-per-template":
            num_per_template = _calculate_f1_per_template(report[metric_name])
        case "temporal-awareness-per-template":
            num_per_template = _calculate_score_per_template(report[metric_name])
        case "document-and-pair-wise-scores" | "document-wise-scores":
            num_per_template = _calculate_f1_per_template_doc(report[metric_name])
        # todo: below three are not tested yet
        case "exact-match" | "exact-match-relaxed":
            num_per_template = _calculate_match_per_template_doc(report[metric_name])
        case "scores" | "scores-relaxed":
            num_per_template = _calculate_score_torque_per_template_doc(
                report[metric_name]
            )
        case "example-wise-scores":
            num_per_template = _calculate_f1_per_template_doc(report[metric_name])
        case _:
            logging.error(f"Undefined metric name: {metric_name}")

    report_range = {
        "range": {
            "min": min(num_per_template.values()),
            "median": median(num_per_template.values()),
            "max": max(num_per_template.values()),
        },
        "individual": num_per_template,
    }
    return report_range


def _calculate_range(
    scores: dict[Any],
    metric_name: str,
) -> dict[Any]:
    """
    Calculate min, median, max per document
    taking min/max for each example w/ multiple prompt templates

    """

    if metric_name == "example-wise-scores":
        nums = [_scores["weighted avg"]["f1-score"] for _scores in scores.values()]
    elif metric_name == "example-wise-match":
        nums = [sum(matches) / (len(matches) + 1e-20) for matches in scores.values()]
    elif metric_name == "example-wise-torque":
        nums = []
        for _scores in scores.values():
            _nums = [_s[-1] for _s in _scores]  # check f1
            avg = sum(_nums) / (len(_nums) + 1e-20)
            nums.append(avg)
    else:
        logging.error(f"Unsupported metric: {metric_name}")

    return min(nums), median(nums), max(nums)
