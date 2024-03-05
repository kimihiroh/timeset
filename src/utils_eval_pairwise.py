from collections import defaultdict, Counter
from typing import Any
import random
import logging
from sklearn.metrics import classification_report
from template_classes import (
    CTF_PAIRWISE_RELATIONS,
    MATRES_RELATIONS,
    TDDISCOURSE_RELATIONS,
    CTF_RELATIONS,
)
from utils_eval import _calculate_range, calculate_range_template


def _extract_answer(text: str) -> str:
    """
    extract answer word from the raw LLM generation output

    """
    return text.strip().split("\n\n")[0].strip().upper()


def _calculate_example_wise_scores_benchmark(
    examples: list[Any],
    predictions: list[Any],
    dataset_name: str,
) -> dict[Any]:
    if dataset_name == "matres":
        labels = MATRES_RELATIONS
    elif dataset_name == "tddiscourse":
        labels = TDDISCOURSE_RELATIONS
    else:
        logging.error(f"Undefined dataset name: {dataset_name}")

    golds = [
        labels.index(x["relation"]) if x["relation"] in labels else None
        for x in examples
    ]

    template2predictions = defaultdict(lambda: defaultdict(str))
    for example in predictions:
        example_id = example["input"]["example_id"]
        template_id = example["input"]["template_id"]

        pred = _extract_answer(example["output"]).upper()
        pred_id = labels.index(pred) if pred in labels else len(labels)

        template2predictions[template_id][example_id] = pred_id

    template2report = {}
    for template_id, predictions in template2predictions.items():
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


def evaluate_pairwise_benchmark(
    examples: list[Any], predictions: list[Any], dataset_name: str
) -> tuple[dict[Any], dict[Any]]:
    """
    Calculate metrics for matres dataset
    * [x] example-wise f1

    """

    report = {
        "example-wise-scores": _calculate_example_wise_scores_benchmark(
            examples,
            predictions,
            dataset_name,
        ),
    }

    _min, _median, _max = _calculate_range(
        scores=report["example-wise-scores"],
        metric_name="example-wise-scores",
    )
    report_range = {
        "example-wise-scores": {
            "min": _min,
            "median": _median,
            "max": _max,
        }
    }

    return report_range, report


def flip_relation(relation):
    match relation:
        case "AFTER":
            relation_inverse = "BEFORE"
        case "BEFORE":
            relation_inverse = "AFTER"
        case "VAGUE":
            relation_inverse = "VAGUE"
        case "COEX":
            relation_inverse = "COEX"
        case _:
            relation_inverse = None
    return relation_inverse


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
        logging.error(f"No predictions found: {pred_ids}")

    return pred_id


def _calculate_document_and_pair_wise_scores(
    examples: list[Any],
    predictions: list[Any],
) -> dict[Any]:
    """
    aggregate predictions for each pair of events
    and calculate scores for each template and document

    TODO:
    create one shared function across formulations w/ post processing part for each

    """

    # register the gold relation of each pair
    filename_pair2golds = defaultdict(lambda: defaultdict(int))
    for example_id, example in enumerate(examples):
        filename = example["filename"]
        for pair, label in example["pairs"]["original"].items():
            filename_pair2golds[filename][pair] = CTF_RELATIONS.index(label)

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

        pred = _extract_answer(pred["output"])

        if f"{id_arg1}-{id_arg2}" in examples[example_id]["pairs"]["original"]:
            pair_id = f"{id_arg1}-{id_arg2}"
        elif f"{id_arg2}-{id_arg1}" in examples[example_id]["pairs"]["original"]:
            pred = flip_relation(pred)
            pair_id = f"{id_arg2}-{id_arg1}"
        else:
            logging.error(f"({id_arg1}, {id_arg2}) is not in annotation")

        # is this -1 correct? <= yes, because two "2" exists in the relations
        # so actually, "3" is registered.
        template_filename_pair2preds[template_id][filename][pair_id].append(
            CTF_PAIRWISE_RELATIONS[pred]
            if pred in CTF_PAIRWISE_RELATIONS
            else len(CTF_PAIRWISE_RELATIONS) - 1
        )

    # identify the best pred for each pair
    template_filename_pair2pred = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for template_id, filename_pair2preds in template_filename_pair2preds.items():
        assert len(filename_pair2preds) == len(filename_pair2golds)
        for filename, pair2preds in filename_pair2preds.items():
            assert len(pair2preds) == len(filename_pair2preds[filename])
            for pair_id, preds in pair2preds.items():
                assert len(preds) == len(filename_pair2preds[filename][pair_id])
                pred_ids = [x for x in preds]
                # ToDo: confidence score-based ILP would be another approach
                # current: majority-vote -> random guess
                pred_id = majority_vote(pred_ids)
                template_filename_pair2pred[template_id][filename][pair_id] = pred_id

    # calculate scores for each template and file
    template_and_filename2report = defaultdict(lambda: defaultdict(list))
    for template_id, filename_pair2pred in template_filename_pair2pred.items():
        for filename, pair2pred in filename_pair2pred.items():
            golds, preds = [], []
            assert len(pair2pred) == len(filename_pair2golds[filename])
            for pair_id, gold in filename_pair2golds[filename].items():
                golds.append(gold)
                preds.append(pair2pred[pair_id])

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


def evaluate_pairwise_ctf(
    examples: list[Any],
    predictions: list[Any],
) -> tuple[dict[Any], dict[Any]]:
    """
    Calculate metrics for ctf pairwise data
    * [x] document and pair wise f1
    * [ ] temporal awareness (document-level)

    """

    report = {
        "document-and-pair-wise-scores": (
            _calculate_document_and_pair_wise_scores(examples, predictions)
        )
    }

    metric_names = ["document-and-pair-wise-scores"]
    report_range = {}
    for metric_name in metric_names:
        report_range[metric_name] = calculate_range_template(report, metric_name)

    return report_range, report
