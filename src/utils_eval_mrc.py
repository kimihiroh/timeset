from collections import defaultdict, Counter
from typing import Any
import logging
import random
import json
import re
from math import isclose
from sklearn.metrics import classification_report
from template_classes import CTF_RELATIONS
from utils_eval import _calculate_range, calculate_range_template


def _extract_answers(
    string: str,
) -> list[str]:
    """
    extract answer words from the raw generated outputs by LLMs
    for mrc formulations

    """
    if "-" in string:
        answers = [
            x.strip() for x in string.strip().split("\n\n")[0].split("-") if x.strip()
        ]
    else:
        answers = [
            x.strip() for x in string.strip().split("\n\n")[0].split(",") if x.strip()
        ]

    return answers


def check_if_x_matches_y(x: list[str], y: list[str]) -> bool:
    """
    example:

    check_if_x_matches_y([1,2,3], [1,3,2]): True
    check_if_x_matches_y([1,2,3], [3,2,4]): False

    """

    flag = False
    if len(x) == len(y) and check_if_x_contains_y(x, y) and check_if_x_contains_y(y, x):
        flag = True

    return flag


def _calculate_exact_match_torque(examples, predictions):
    template2matches = defaultdict(list)
    template2matches_relaxed = defaultdict(list)
    for _prediction in predictions:
        example_id = _prediction["input"]["example_id"]
        template_id = _prediction["input"]["template_id"]

        prediction = _extract_answers(string=_prediction["output"])

        # exact match with majority-voted answer
        gold = [x["mention"] for x in examples[example_id]["answers"]]
        template2matches[template_id].append(check_if_x_matches_y(x=prediction, y=gold))

        # exact match with any of individual annotations
        individual_annotations = [
            [answer["mention"] for answer in answers]
            for answers in examples[example_id]["raw_answers_list"]
        ]
        exact_match_relaxed = any(
            [
                check_if_x_matches_y(x=prediction, y=individual_annotation)
                for individual_annotation in individual_annotations
            ]
        )
        template2matches_relaxed[template_id].append(exact_match_relaxed)

    return template2matches, template2matches_relaxed


def check_if_x_contains_y(x: list[str], y: list[str]) -> bool:
    """
    example:
    a = [1,2,3]
    b = [1,2,3,4]

    check_if_x_contains_y(a, b) => False
    check_if_x_contains_y(b, a) => True

    """
    flag = True
    for _y in y:
        if _y in x:
            pass
        else:
            flag = False

    return flag


def check_which_in_y_is_contained_in_x(x: list[str], y: list[str]) -> list[str]:
    """
    example:
    a = [1,2,1]
    b = [1,2,3,4]

    check_which_in_y_is_contained_in_x(a.copy(), b.copy()) => [1,2]
    check_which_in_y_is_contained_in_x(b.copy(), a.copy()) => [1,2]

    """

    contained = []
    for _y in y:
        if _y in x:
            idx = x.index(_y)
            contained.append(x.pop(idx))
        else:
            pass

    return contained


def calc_precision_recall_f1(
    gold: list[str],
    prediction: list[str],
) -> tuple[float, float, float]:
    """
    example:
    prediction = [1,2]
    gold = [1,2,3]
    calc_precision_recall_f1(prediction=prediction, gold=gold)
    => 1.0, 0.6666666666666666, 0.799999999952

    """

    precision = len(
        check_which_in_y_is_contained_in_x(x=gold.copy(), y=prediction.copy())
    ) / (len(prediction) + 1e-100)
    recall = len(
        check_which_in_y_is_contained_in_x(x=prediction.copy(), y=gold.copy())
    ) / (len(gold) + 1e-100)
    f1 = 2 * precision * recall / (precision + recall + 1e-100)

    return precision, recall, f1


def max_by_f1(scores: list[tuple[float, float, float]]) -> int:
    """
    return the element which has the highest f1 score

    example:
    a = [
        (5,1,1),
        (2,8,2),
        (3,3,3)
    ]
    max_f1(a) => (3,3,3)

    """
    scores = list(scores)
    f1s = [score[-1] for score in scores]

    return scores[f1s.index(max(f1s))]


def _calculate_score_torque(examples, predictions):
    template2scores = defaultdict(list)
    template2scores_relaxed = defaultdict(list)
    for _prediction in predictions:
        example_id = _prediction["input"]["example_id"]
        template_id = _prediction["input"]["template_id"]

        prediction = _extract_answers(string=_prediction["output"])

        # score with majority-voted answer
        gold = [x["mention"] for x in examples[example_id]["answers"]]
        score = calc_precision_recall_f1(gold=gold.copy(), prediction=prediction.copy())
        template2scores[template_id].append(score)

        # score with any of individual annotations
        individual_annotations = [
            [answer["mention"] for answer in answers]
            for answers in examples[example_id]["raw_answers_list"]
        ]
        individual_scores = [
            calc_precision_recall_f1(
                prediction=prediction.copy(), gold=individual_annotation.copy()
            )
            for individual_annotation in individual_annotations
        ]
        score_relaxed = max_by_f1(scores=individual_scores)
        template2scores_relaxed[template_id].append(score_relaxed)

    return template2scores, template2scores_relaxed


def evaluate_mrc_torque(
    examples: list[Any],
    predictions: list[Any],
) -> tuple[dict[Any], dict[Any]]:
    """
    Calculate exact match, precision, recall, and f1 scores
    example-wise and template-wise

    """

    exact_matches, exact_matches_relaxed = _calculate_exact_match_torque(
        examples, predictions
    )
    scores, scores_relaxed = _calculate_score_torque(examples, predictions)

    report = {
        "exact-match": exact_matches,
        "exact-match-relaxed": exact_matches_relaxed,
        "scores": scores,
        "scores-relaxed": scores_relaxed,
    }

    metric_names = [
        "exact-match",
        "exact-match-relaxed",
        "scores",
        "scores-relaxed",
    ]
    # todo: replace with _calculate_range_template
    report_range = {}
    for metric_name in metric_names:
        _min, _median, _max = _calculate_range(
            report[metric_name],
            "example-wise-torque" if "scores" in metric_name else "example-wise-match",
        )
        report_range[metric_name] = {
            "min": _min,
            "median": _median,
            "max": _max,
        }

    return report_range, report


def _get_eid_and_representation(text: str) -> tuple[int, str]:
    """
    text: e.g., "[e1]create[/e1]"
    return: e.g., ("1", "create")
    """
    match = re.search(r"\[e(\d+)\](.*?)\[/e(\d+)\]", text)
    if match:
        eid_, representation, _eid = match.group(1), match.group(2), match.group(3)
        if eid_ != _eid:
            logging.warning(f"eid mismatch ({eid_}, {_eid}): use the first one, {eid_}")
        return eid_, representation
    else:
        return None, None


def _get_representation(text: str) -> str:
    """
    text: e.g., "**create**"
    return: e.g., ("create")
    """
    match = re.search(r"\*\*([^*]+)", text)
    if match:
        representation = match.group(1)
    else:
        representation = None
    return representation


def flip_relation(relation):
    """
    reverse relation for MRC
    """
    if relation == "AFTER":
        relation_inverse = "BEFORE"
    elif relation == "BEFORE":
        relation_inverse = "AFTER"
    elif relation == "COEX":
        relation_inverse = "COEX"
    else:
        logging.error(f"Undefined relation: {relation}")
        relation_inverse = None

    return relation_inverse


def majority_vote(pred_ids: list[int]) -> int:
    """
    majority vote
    -> if tie happens, choose one randomly

    """

    assert len(pred_ids) <= 6

    count_dict = Counter(pred_ids)
    max_count = max(count_dict.values())
    max_keys = [key for key, value in count_dict.items() if value == max_count]
    if len(max_keys) == 1:
        pred_id = max_keys[0]
    elif len(max_keys) > 1:
        pred_id = random.choice(max_keys)
    else:
        logging.error(f"No predictions found: {pred_ids}")
        pred_id = None

    return pred_id


def extract_words(string: str) -> list[str]:
    """
    input: e.g., [EVENT]create[ARG0]he[ARG3]school
    output: e.g., [create, he, school]

    """
    return [match for match in re.findall("\[(?:[^\]]*\])?|(\w+)", string) if match]


def check_word_overlap(words1: list[str], words2: list[str]) -> float:
    """
    calc word overlap f1
    """
    if words1 and words2:
        r = sum([any([word in _word for _word in words2]) for word in words1]) / (
            len(words1) + 1e-10
        )
        p = sum([any([word in _word for _word in words1]) for word in words2]) / (
            len(words2) + 1e-10
        )
        f1 = 2 * r * p / (r + p + 1e-10)
    else:
        f1 = 0.0
    return f1


def find_best_match(
    current_eid: str,
    representation: str,
    events: list[Any],
    repr_type: str,
    marker_type: str,
) -> str:
    """
    find the best match considering
    * marker==eid: both eid and representation
        * Default: eid
        * if there is a better representation match, then use the eid of it
    * marker==star: representation
    """

    eid2score = {}
    for eid, event in events.items():
        if repr_type == "structured":
            if representation:
                words = extract_words(representation)
            else:
                words = None
            words_answer = [event["mention"]] + [
                x["mention"] for x in event["arguments"].values()
            ]
        elif repr_type == "mention":
            if representation:
                words = representation.split()
            else:
                words = None
            words_answer = event["mention"].split()
        else:
            logging.error(f"Undefined representation type: {repr_type}")
        eid2score[eid] = check_word_overlap(words, words_answer)

    max_score = max(eid2score.values())
    eids_with_max_score = [
        eid
        for eid, score in eid2score.items()
        if isclose(score, max_score, rel_tol=1e-5)
    ]
    if eids_with_max_score:
        if marker_type == "eid" and current_eid in eids_with_max_score:
            new_eid = current_eid
        else:
            # for marker==star, this is the only choice
            new_eid = random.choice(eids_with_max_score)
    else:
        new_eid = None

    return new_eid


def _calculate_document_and_pair_wise_scores(
    examples: list[Any],
    record: Any,
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
    repr_type, marker_type = record["args"]["representation"], record["args"]["marker"]
    examples_not_repr_or_not_eid = []
    for pred in record["examples"]:
        example_id = pred["input"]["example_id"]
        template_id = pred["input"]["template_id"]
        filename = examples[example_id]["filename"]
        eid = examples[example_id]["target_id"]
        relation = examples[example_id]["relation"].upper()
        if not template_filename_pair2preds[template_id][filename]:
            template_filename_pair2preds[template_id][filename] = defaultdict(list)

        _pred_answers = _extract_answers(pred["output"])
        # add relation for each pair once from one prediction
        # to have this restriction, keep which eid is already registered
        registered_eids = []
        for _pred_answer in _pred_answers:
            if marker_type == "eid":
                eid_answer, representation = _get_eid_and_representation(_pred_answer)
            elif marker_type == "star":
                eid_answer, representation = None, _get_representation(_pred_answer)

            if not representation:
                examples_not_repr_or_not_eid.append(_pred_answer)

            eid_answer = find_best_match(
                current_eid=eid_answer,
                representation=representation,
                events=examples[example_id]["all_events"],
                repr_type=repr_type,
                marker_type=marker_type,
            )
            # skip if this eid is identified in this prediction
            if eid_answer in registered_eids:
                continue

            flag_pair_valid = True
            if f"{eid}-{eid_answer}" in examples[example_id]["pairs"]["original"]:
                pair_id = f"{eid}-{eid_answer}"
                pass
            elif f"{eid_answer}-{eid}" in examples[example_id]["pairs"]["original"]:
                relation = flip_relation(relation)
                pair_id = f"{eid_answer}-{eid}"
            else:
                if str(eid) == str(eid_answer):
                    pass
                else:
                    logging.debug(f"({eid}, {eid_answer}) is not in annotation")
                flag_pair_valid = False

            if flag_pair_valid:
                template_filename_pair2preds[template_id][filename][pair_id].append(
                    CTF_RELATIONS.index(relation)
                )
                registered_eids.append(eid_answer)

    # show examples where answers can be extracted
    for example in list(set(examples_not_repr_or_not_eid)):
        logging.debug(f"cannot extract answer from {example}")

    # identify the best pred for each pair
    template_filename_pair2pred = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for template_id, filename_pair2preds in template_filename_pair2preds.items():
        if len(filename_pair2preds) != len(filename_pair2gold):
            fs = ", ".join(
                [
                    filename
                    for filename in filename_pair2gold.keys()
                    if filename not in filename_pair2preds
                ]
            )
            if fs:
                logging.debug(f"prediction missing files: {fs}")
        for filename, pair2preds in filename_pair2preds.items():
            for pair_id, pred_ids in pair2preds.items():
                # ToDo: confidence score-based ILP would be another approach
                # or votes as scores after softmax
                # current: majority-vote -> random guess
                pred_id = majority_vote(pred_ids)
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


def evaluate_mrc_ctf(
    examples: list[Any],
    record: Any,
) -> tuple[dict[Any], dict[Any]]:
    """
    Calculate metrics for ctf mrc data
    * [x] document and pair wise f1
    * [ ] temporal awareness (document-level)

    """

    report = {
        "document-and-pair-wise-scores": (
            _calculate_document_and_pair_wise_scores(examples, record)
        ),
    }
    metric_names = ["document-and-pair-wise-scores"]
    report_range = {}
    for metric_name in metric_names:
        report_range[metric_name] = calculate_range_template(report, metric_name)

    return report_range, report


if __name__ == "__main__":
    print(_get_eid_and_representation("[e1]create[/e1]"))
    print(_get_representation("**create**"))
    print(_get_representation("**create"))
    print(_extract_answers("**create**, **won, **hey**"))
    print(_get_representation("**[Event]protests[ARG5]protest**"))
    print(_get_eid_and_representation("[e1][EVENT]create[ARG0]he[/e1]"))
    preds = _extract_answers("[e1]create[/e1], [e2]won[/e2]")
    print(preds)
    for pred in preds:
        print(_get_eid_and_representation(pred))
    print(extract_words("[EVENT]create[ARG0]he"))
    print(extract_words("[EVENT]create[ARG0]he[ARG3]school"))

    with open("./data/preprocessed/ctf/test.json", "r") as f:
        example = json.load(f)[0]["annotation"]
    repr_type = "mention"
    marker_type = "eid"
    assert find_best_match("3", "won", example["events"], repr_type, marker_type) == "3"
    assert find_best_match("6", "won", example["events"], repr_type, marker_type) == "6"
    print(
        find_best_match("4", "won", example["events"], repr_type, marker_type)
    )  # id != repr
    repr_type = "structured"
    match = find_best_match(
        "6",
        "[EVENT]won[ARG0]Thibaut Courtois[ARG1]the Adidas Golden Glove[ARG2]this year's FIFA World Cup",  # noqa E501
        example["events"],
        repr_type,
        marker_type,
    )
    assert match == "6", match

    match = find_best_match(
        "4",
        "[EVENT]won[ARG0]Thibaut Courtois[ARG1]the Adidas Golden Glove[ARG2]this year's FIFA World Cup",  # noqa E501
        example["events"],
        repr_type,
        marker_type,
    )
    assert match == "6", match  # case: id != repr

    temp = """- [e2]joined[/e2]
- [e3]won[/e3]
- [e4]loaned[/e4]
- [e5]won[/e5]
- [e6]won[/e6]"""
    print(_extract_answers(temp))
