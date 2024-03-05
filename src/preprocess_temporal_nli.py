"""
preprocessing script for Temporal NLI, https://github.com/sidsvash26/temporal_nli/

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any


def preprocess(data: list[Any], metadata: list[Any]) -> list[Any]:
    """
    preprocess temporal nli

    """

    examples = []
    for (
        _example,
        _example_meta,
    ) in zip(data, metadata):
        if _example["label"] == "entailed":
            label = "positive"
        elif _example["label"] == "not-entailed":
            label = "negative"
        else:
            logging.error(f"Undefined label: {_example['label']}")

        corpus, filename, arg1, arg2 = _example_meta["corpus-sent-id"].split("|")

        examples.append(
            {
                "context": _example["context"],
                "statement": _example["hypothesis"],
                "label": label,
                "corpus": corpus,
                "filename": filename,
                "arg1": arg1,
                "arg2": arg2,
            }
        )

    return examples


def main(args):
    for split in ["train", "dev", "test"]:
        dirpath = args.dirpath_input / split

        with open(dirpath / "recast_tempeval3_data.json", "r") as f:
            data = json.load(f)
        with open(dirpath / "recast_tempeval3_metadata.json", "r") as f:
            metadata = json.load(f)

        examples = preprocess(data, metadata)

        logging.info(f"[{split}] #examples: {len(examples)}")

        with open(args.dirpath_output / f"{split}.json", "w") as f:
            json.dump(examples, f, indent=4)
            f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess temporal nli (tempeval3)")
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
            logging.FileHandler(
                args.dirpath_log / "preprocess_temporal_nli_tempeval3.log"
            ),
        ],
    )

    main(args)
