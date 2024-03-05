"""
utils for dataset classes

"""

import logging
import numpy as np
import random
from typing import Any


def add_emarkers(example: list[Any]) -> str:
    """
    add event markers

    """

    _context = example["context"]

    if example["arg1"]["start"] < example["arg2"]["start"]:
        context = (
            _context[: example["arg1"]["start"]]
            + "[e1]"
            + example["arg1"]["mention"]
            + "[/e1]"
            + _context[example["arg1"]["end"] : example["arg2"]["start"]]
            + "[e2]"
            + example["arg2"]["mention"]
            + "[/e2]"
            + _context[example["arg2"]["end"] :]
        )
    else:
        context = (
            _context[: example["arg2"]["start"]]
            + "[e2]"
            + example["arg2"]["mention"]
            + "[/e2]"
            + _context[example["arg2"]["end"] : example["arg1"]["start"]]
            + "[e1]"
            + example["arg1"]["mention"]
            + "[/e1]"
            + _context[example["arg1"]["end"] :]
        )

    return context


def _select_demonstrations(examples: list, num_demonstration: int, criteria: str):
    """
    weighted random choices based on context length
    shorter docs get chosen more easily

    """
    outputs = []
    if criteria == "short":
        lens = [len(x["context"]) for x in examples]
        avg_len = sum(lens) / len(lens)
        weights = [avg_len / len(x["context"]) for x in examples]
        p = [x / sum(weights) for x in weights]

        outputs = np.random.choice(examples, num_demonstration, replace=False, p=p)
    elif criteria == "random":
        outputs = random.sample(examples, k=num_demonstration)
    else:
        logging.error(f"Undefined demonstration selection criteria: {criteria}")

    return outputs


def _preprocess_example(annotations, task):
    """ """
    examples = []
    for annotation in annotations:
        for raw_example in annotation["examples"][task]:
            match task:
                case "pairwise":
                    example = {
                        "context": annotation["text"],
                        "id_arg1": raw_example["id_arg1"],
                        "arg1": annotation["annotation"]["events"][
                            str(raw_example["id_arg1"])
                        ],
                        "id_arg2": raw_example["id_arg2"],
                        "arg2": annotation["annotation"]["events"][
                            str(raw_example["id_arg2"])
                        ],
                        "relation": raw_example["relation"],
                        "pairs": annotation["pairs"],
                        "filename": annotation["filename"],
                    }
                case "nli":
                    example = {
                        "context": annotation["text"],
                        "id_arg1": raw_example["id_arg1"],
                        "arg1": annotation["annotation"]["events"][
                            str(raw_example["id_arg1"])
                        ],
                        "keyword": raw_example["keyword"],
                        "id_arg2": raw_example["id_arg2"],
                        "arg2": annotation["annotation"]["events"][
                            str(raw_example["id_arg2"])
                        ],
                        "label": raw_example["label"],
                        "pairs": annotation["pairs"],
                        "filename": annotation["filename"],
                    }
                case "mrc":
                    example = {
                        "context": annotation["text"],
                        "all_events": annotation["annotation"]["events"],
                        "target_id": raw_example["target"],
                        "target": annotation["annotation"]["events"][
                            str(raw_example["target"])
                        ],
                        "answer_ids": raw_example["answers"],
                        "answers": {
                            idx: annotation["annotation"]["events"][str(idx)]
                            for idx in raw_example["answers"]
                        },
                        "relation2target": raw_example["relation2answers"],
                        "relation": raw_example["relation"],
                        "pairs": annotation["pairs"],
                        "filename": annotation["filename"],
                    }
                case "timeline":
                    example = {
                        "context": annotation["text"],
                        "all_events": annotation["annotation"]["events"],
                        "timeline": [
                            {
                                eid: annotation["annotation"]["events"][str(eid)]
                                for eid in layer
                            }
                            for layer in raw_example
                        ],
                        "all_relations": annotation["annotation"]["relations"],
                        "pairs": annotation["pairs"],
                        "filename": annotation["filename"],
                    }
                case _:
                    logging.error(f"Undefined task: {task}")

            examples.append(example)

    return examples
