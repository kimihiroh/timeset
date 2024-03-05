"""
Preprocess brat .ann files

"""
from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any
from utils_brat import (
    parse_ann_file,
    parse_txt_file,
)
from utils_ctf import (
    create_pairwise_examples,
    create_nli_examples,
    create_mrc_examples,
    create_timeline_examples,
    get_pair_labels,
)


def _preprocess(
    eid2event_with_arguments,
    rid2relation,
) -> dict[Any]:
    # remapping eid in textual order
    eids_sorted = [
        eid
        for eid, event in sorted(
            eid2event_with_arguments.items(),
            key=lambda item: int(item[1].event.textbound.start),
        )
    ]
    # restructure elements
    events = {}
    eid_mapping_old2new = {}
    for new_id, old_eid in enumerate(eids_sorted):
        eid_mapping_old2new[old_eid] = new_id

        event_with_arguments = eid2event_with_arguments[old_eid]
        role2argument = {}
        for argument in event_with_arguments.arguments:
            role = argument.role.strip("-")
            role2argument[role] = {
                "mention": argument.textbound.text,
                "start": argument.textbound.start,
                "end": argument.textbound.end,
            }
        events[str(new_id)] = {
            "mention": event_with_arguments.event.textbound.text,
            "start": event_with_arguments.event.textbound.start,
            "end": event_with_arguments.event.textbound.end,
            "arguments": role2argument,
        }

    relations = []
    for _, relation in rid2relation.items():
        relations.append(
            {
                "arg1": str(eid_mapping_old2new[relation.arg1]),
                "arg2": str(eid_mapping_old2new[relation.arg2]),
                "relation": relation.relation,
            }
        )
    # TODO: better to create canonical relations from a graph
    # for chain of thought
    annotation = {
        "events": events,
        "relations": relations,
        # "relations_canonical": relations_canonical,
    }

    return annotation


def main(args):
    for split, dirpath in [("dev", args.dirpath_dev), ("test", args.dirpath_test)]:
        logging.info(f"Split: {split}")
        outputs = []
        for filepath in dirpath.glob("*.ann"):
            logging.debug(f"[filepath] {filepath}")

            # parse .txt file
            text = parse_txt_file(filepath)

            # parse .ann file
            eid2event_with_arguments, tid2entity, rid2relation = parse_ann_file(
                filepath
            )
            # preprocess
            annotation = _preprocess(eid2event_with_arguments, rid2relation)

            # create examples
            examples = {
                "pairwise": create_pairwise_examples(annotation),
                "nli": create_nli_examples(annotation),
                "mrc": create_mrc_examples(annotation),
                "timeline": create_timeline_examples(annotation),
            }
            # create gold labels
            pairs = {
                "original": get_pair_labels(annotation, flag_linearized=False),
                "linearized": get_pair_labels(annotation, flag_linearized=True),
            }

            outputs.append(
                {
                    "text": text.rstrip(),
                    "annotation": annotation,
                    "examples": examples,
                    "pairs": pairs,
                    "filename": filepath.stem,
                }
            )

        filepath = args.dirpath_output / f"{split}.json"
        with open(filepath, "w") as f:
            json.dump(outputs, f, indent=4)
            f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess brat data")
    parser.add_argument(
        "--dirpath_dev", type=Path, help="dirpath to dev data", default=False
    )
    parser.add_argument(
        "--dirpath_test", type=Path, help="dirpath to test data", default=False
    )
    parser.add_argument(
        "--dirpath_output", type=Path, help="dirpath to output data", default=False
    )
    parser.add_argument(
        "--dirpath_log", type=Path, help="dirpath to log", default=False
    )
    args = parser.parse_args()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "preprocess_timeset.log"),
        ],
    )
    logging.info(f"Arguments: {args}")

    main(args)
