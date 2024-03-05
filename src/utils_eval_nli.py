from collections import defaultdict, Counter
from typing import Any
import random
import logging
from sklearn.metrics import classification_report
from template_classes import (
    TEMPORALNLI_LABELS,
    CTF_RELATIONS,
)
from utils_eval import _calculate_range, calculate_range_template


def _extract_answer(text: str) -> str:
    """
    extract answer word from the raw LLM generation output

    """
    splits = text.strip().split("\n\n")

    return splits[0].strip().lower()


def _calculate_example_wise_scores_benchmark(
    examples: list[Any], predictions: list[Any], templates: list[Any], dataset_name: str
) -> dict[Any]:
    if dataset_name == "temporal-nli":
        labels = TEMPORALNLI_LABELS
    else:
        logging.error(f"Undefined dataset name: {dataset_name}")

    golds = [labels.index(x["label"]) for x in examples]

    template2preds = defaultdict(lambda: defaultdict(str))
    for pred in predictions:
        example_id = pred["input"]["example_id"]
        template_id = pred["input"]["template_id"]

        _pred = _extract_answer(pred["output"])
        if template_id == 0:
            pred = _pred
        else:
            pred = templates[template_id].map_target2original(_pred)
        pred_id = labels.index(pred) if pred in labels else len(labels)
        template2preds[template_id][example_id] = pred_id

    template2report = {}
    for template_id, predictions in template2preds.items():
        preds = list(dict(sorted(predictions.items())).values())

        # Note:
        # The support is the number of occurrences of each class in y_true
        # so the weighted average does not consider the None classes
        report = classification_report(
            golds,
            preds,
            labels=[x for x in range(len(labels) + 1)],
            target_names=labels + [None],
            output_dict=True,
            zero_division=0.0,
        )

        template2report[template_id] = report

    return template2report


def evaluate_nli_benchmark(
    examples: list[Any], predictions: list[Any], templates: list[Any], dataset_name: str
) -> tuple[dict[Any], dict[Any]]:
    """
    Calculate metrics for tracie dataset

    """

    report = {
        "example-wise-scores": _calculate_example_wise_scores_benchmark(
            examples=examples,
            predictions=predictions,
            templates=templates,
            dataset_name=dataset_name,
        )
    }

    metric_names = [
        "example-wise-scores",
    ]

    report_range = {}
    for metric_name in metric_names:
        _min, _median, _max = _calculate_range(report[metric_name], metric_name)
        report_range[metric_name] = {
            "min": _min,
            "median": _median,
            "max": _max,
        }

    return report_range, report


def majority_vote(pred_ids: list[int]) -> int:
    """
    majority vote
    -> if tie happens, choose one randomly

    """

    count_dict = Counter(pred_ids)
    max_count = max(count_dict.values())
    max_keys = [key for key, value in count_dict.items() if value == max_count]
    if len(max_keys) == 1:
        pred_id = max_keys[0]
    elif len(max_keys) > 1:
        pred_id = random.choice(max_keys)
    else:
        logging.warning(f"No predictions found: {pred_ids}")
        pred_id = None

    return pred_id


def flip_relation(relation):
    if relation == "positive":
        relation_inverse = "negative"
    elif relation == "negative":
        relation_inverse = "positive"
    elif relation == "unclear":
        relation_inverse = "unclear"
    else:
        relation_inverse = None

    return relation_inverse


def _calculate_document_and_pair_wise_scores(
    examples: list[Any],
    predictions: list[Any],
    templates: list[Any],
) -> dict[Any]:
    """
    aggregate predictions for each pair of events
    and calculate scores for each template and document

    TODO:
    create one shared function across formulations w/ post processing part for each

    """
    # register the gold relation of each pair
    filename_pair2gold = defaultdict(lambda: defaultdict(int))
    for example_id, example in enumerate(examples):
        filename = example["filename"]
        for pair, label in example["pairs"]["original"].items():
            filename_pair2gold[filename][pair] = CTF_RELATIONS.index(label)

    # collect preds for each pair
    template_filename_pair2preds = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for pred in predictions:
        example_id = pred["input"]["example_id"]
        template_id = pred["input"]["template_id"]
        filename = examples[example_id]["filename"]
        id_arg1, id_arg2 = (
            examples[example_id]["id_arg1"],
            examples[example_id]["id_arg2"],
        )

        _pred = _extract_answer(pred["output"])
        # convert pred to canonical label
        pred = templates[template_id].map_target2original(_pred)

        if f"{id_arg1}-{id_arg2}" in examples[example_id]["pairs"]["original"]:
            pair_id = f"{id_arg1}-{id_arg2}"
        elif f"{id_arg2}-{id_arg1}" in examples[example_id]["pairs"]["original"]:
            pred = flip_relation(pred)
            pair_id = f"{id_arg2}-{id_arg1}"
        else:
            logging.debug(f"({id_arg1}, {id_arg2}) is not in annotation")

        # TODO: conversion from pred to relation, using both pred and statement.
        # <= seems to be achived... right? Nov 27, Kimi
        if pred == "positive":
            pred_relation_id = CTF_RELATIONS.index(examples[example_id]["keyword"])
            # else:
            #     # TODO: check: is this correct? Probably None is appropriate.
            #     pred_relation_id = len(CTF_RELATIONS)
            template_filename_pair2preds[template_id][filename][pair_id].append(
                pred_relation_id
            )
        else:
            if not template_filename_pair2preds[template_id][filename][pair_id]:
                template_filename_pair2preds[template_id][filename][pair_id] = []

    # identify the best pred for each pair
    template_filename_pair2pred = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for template_id, filename_pair2preds in template_filename_pair2preds.items():
        assert len(filename_pair2preds) == len(filename_pair2gold)
        for filename, pair2preds in filename_pair2preds.items():
            for pair_id, pred_ids in pair2preds.items():
                # ToDo: confidence score-based ILP would be another approach
                # or votes as scores after softmax
                # current: majority-vote -> random guess
                if pred_ids:
                    pred_id = majority_vote(pred_ids)
                else:
                    pred_id = len(CTF_RELATIONS)
                template_filename_pair2pred[template_id][filename][pair_id] = pred_id

    # calculate scores for each template and file
    template_and_filename2report = defaultdict(lambda: defaultdict(list))
    for template_id, filename_pair2pred in template_filename_pair2pred.items():
        for filename, pair2gold in filename_pair2gold.items():
            golds, preds = [], []
            for pair_id, gold in pair2gold.items():
                golds.append(gold)
                if pair_id in filename_pair2pred[filename]:
                    preds.append(filename_pair2pred[filename][pair_id])
                else:
                    preds.append(len(CTF_RELATIONS))

            report = classification_report(
                golds,
                preds,
                labels=[x for x in range(len(CTF_RELATIONS) + 1)],
                target_names=CTF_RELATIONS + [None],
                output_dict=True,
                zero_division=0.0,
            )
            template_and_filename2report[template_id][filename] = report

    return template_and_filename2report


def evaluate_nli_ctf(
    examples: list[Any],
    predictions: list[Any],
    templates: list[Any],
) -> tuple[dict[Any], dict[Any]]:
    """
    Calculate metrics for ctf nli data
    * [x] document and pair wise f1
        * this is basically temporal awareness, right?
    * [ ] temporal awareness (document-level) after reduction
        * run ILP and group events with coex and linerlize

    """

    report = {
        "document-and-pair-wise-scores": (
            _calculate_document_and_pair_wise_scores(examples, predictions, templates)
        )
    }

    metric_names = ["document-and-pair-wise-scores"]
    report_range = {}
    for metric_name in metric_names:
        report_range[metric_name] = calculate_range_template(report, metric_name)

    return report_range, report
