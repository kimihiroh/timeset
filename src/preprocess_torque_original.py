"""
preprocessing script for TORQUE, stored under https://github.com/qiangning/TORQUE-dataset

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
import random
from typing import List, Dict


def _process_mentions(annotation: Dict, context: str) -> List:
    """
    process annotated mentions

    """

    mentions = []

    spans = annotation["spans"]
    indices = annotation["indices"]
    for span, offset in zip(spans, indices):
        start, end = offset.strip("()").split(",")
        start, end = int(start.strip()), int(end.strip())

        if context[start:end] != span:
            logging.warning(f"Index Mismatch: {context[start:end]} != {span}")
        else:
            mentions.append({"mention": span, "start": start, "end": end})

    return mentions


def preprocess_train(data: List) -> List:
    """
    preprocess torque train set w/o modifying annotation

    """

    examples = []
    for item in data:
        for passage in item["passages"]:
            context = passage["passage"]

            # process annotated events
            events = []
            if len(passage["events"]) != 1:
                logging.warning(f"Unexpected format of events: {passage['events']}")
                continue
            else:
                events = _process_mentions(passage["events"][0]["answer"], context)

            # process annotated question-answer pairs
            for qa in passage["question_answer_pairs"]:
                if qa["passageID"] != passage["events"][0]["passageID"]:
                    logging.warning(
                        f"passageID Mismatch: {qa['passageID']} "
                        f"!= {passage['events'][0]['passageID']}"
                    )
                    continue

                if not qa["isAnswered"]:
                    logging.warning(
                        f"The question, {qa['question_id']}, is not answered."
                    )
                else:
                    answers = _process_mentions(qa["answer"], context)

                examples.append(
                    {
                        "context": context.strip(),
                        "events": events,
                        "answers": answers,
                        "question": qa["question"].strip(),
                        "passage_id": qa["passageID"],
                        "question_id": qa["question_id"],
                        "is_default": qa["is_default_question"],
                    }
                )

    logging.info(f"#examples: {len(examples)}")

    return examples


def preprocess_dev(data: Dict) -> List:
    """
    preprocess torque dev set w/o modifying annotation

    """

    examples = []
    for doc_id, item in data.items():
        context = item["passage"]

        # process annotated events
        events = _process_mentions(item["events"]["answer"], context)

        # process annotated question-answer pairs
        for question, answers_indices in item["question_answer_pairs"].items():
            if answers_indices["passageID"] != item["events"]["passageID"]:
                logging.warning(
                    f"passageID Mismatch: {answers_indices['passageID']} "
                    f"!= {item['events']['passageID']}"
                )
                continue

            if answers_indices["validated_by"] != 3:
                logging.warning(
                    f"This question is annotated by "
                    f"{answers_indices['validated_by']}, not the default 3."
                )

            # process aggregated answer annotation
            answers = _process_mentions(answers_indices["answer"], context)

            # process individual answer annotations
            raw_answers_list = [
                _process_mentions(individual_answer, context)
                for individual_answer in answers_indices["individual_answers"]
            ]

            examples.append(
                {
                    "context": context.strip(),
                    "events": events,
                    "answers": answers,
                    "raw_answers_list": raw_answers_list,
                    "question": question.strip(),
                    "passage_id": doc_id,
                    "cluster_id": answers_indices["cluster_id"],
                    "is_default": answers_indices["is_default_question"],
                }
            )

    logging.info(f"#examples: {len(examples)}")

    return examples


def main(args):
    with open(args.dirpath_input / "train.json", "r") as f:
        data_train = json.load(f)
    examples_train = preprocess_train(data_train)

    with open(args.dirpath_output / "train.json", "w") as f:
        json.dump(examples_train, f, indent=4)
        f.write("\n")

    random.seed(args.seed)
    random.shuffle(examples_train)
    threshold_20percent = int(len(examples_train) * 0.2)
    examples_train_dev = examples_train[:threshold_20percent]
    examples_train_train = examples_train[threshold_20percent:]
    with open(args.dirpath_output / "train_train.json", "w") as f:
        json.dump(examples_train_train, f, indent=4)
        f.write("\n")
    with open(args.dirpath_output / "train_dev.json", "w") as f:
        json.dump(examples_train_dev, f, indent=4)
        f.write("\n")

    with open(args.dirpath_input / "dev.json", "r") as f:
        data_dev = json.load(f)
    examples_dev = preprocess_dev(data_dev)

    with open(args.dirpath_output / "dev.json", "w") as f:
        json.dump(examples_dev, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Preprocess TORQUE dataset w/o annotation modification"
    )
    parser.add_argument("--dirpath_input", type=Path, help="dirpath to input data")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")
    parser.add_argument("--seed", type=int, help="random seed", default=7)
    args = parser.parse_args()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "preprocess_torque_original.log"),
        ],
    )

    main(args)
