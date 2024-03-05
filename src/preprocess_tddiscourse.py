"""

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup
from collections import defaultdict
import re
import spacy
import pandas as pd
from copy import deepcopy

RELATION_MAPPING = {
    "a": "AFTER",
    "b": "BEFORE",
    "s": "SIMULTANEOUS",
    "i": "INCLUDE",
    "ii": "INCLUDED",
}


def parse_annotation(dirpath: Path) -> dict[str, Any]:
    """ """

    annotation = {}
    for filepath in dirpath.glob("*.tsv"):
        if "Phenomena" in filepath.name:
            continue
        data = pd.read_csv(
            filepath,
            delimiter="\t",
            header=None,
            names=["filename", "arg1", "arg2", "relation"],
        )

        filename2annotation = defaultdict(list)
        for idx, row in data.iterrows():
            filename2annotation[row["filename"]].append(
                {
                    "arg1": row["arg1"],
                    "arg2": row["arg2"],
                    "relation": RELATION_MAPPING[row["relation"]],
                }
            )
        annotation[filepath.stem.replace("TDDMan", "").lower()] = filename2annotation

    return annotation


def parse_tags(parsed_data) -> dict[str, Any]:
    """ """

    text_with_tags = str(parsed_data.find("TEXT"))
    text = parsed_data.find("TEXT").text
    tags = (
        parsed_data.find("TEXT").find_all("EVENT")
        + parsed_data.find("TEXT").find_all("TIMEX3")
        + parsed_data.find("TEXT").find_all("SIGNAL")
    )

    # sort tags
    tags_positions = []
    for tag in tags:
        if str(tag) in text_with_tags:
            tags_positions.append(
                {
                    "eid": tag.get("eid"),
                    "event_text": tag.text,
                    "tag_offset": text_with_tags.index(str(tag)),
                    "tag_length": len(str(tag)),
                }
            )
        else:
            logging.error(f"{tag} not found in text.")

    sorted_tags = sorted(tags_positions, key=lambda x: x["tag_offset"])

    and_markers = [x.start() for x in re.finditer("\&", text_with_tags)]

    eid2events = {}
    char_count_tags = len("<TEXT>")
    modifier = 0
    for item in sorted_tags:
        eid, event_text, tag_offset, tag_length = (
            item["eid"],
            item["event_text"],
            item["tag_offset"],
            item["tag_length"],
        )

        while and_markers:
            if and_markers[0] < tag_offset:
                modifier += 4
                and_markers.pop(0)
            if not and_markers or and_markers[0] > tag_offset:
                break

        if eid:
            start = tag_offset - char_count_tags - modifier
            end = start + len(event_text)

            # hack to deal with annotation error, ' Taking'
            # in APW19980227.0487 in timebank split
            if event_text.startswith(" "):
                event_text = event_text[1:]
                start += 1
                char_count_tags -= 1

            eid2events[eid] = {"mention": event_text, "start": start, "end": end}

            if event_text != text[start:end]:
                logging.error(
                    f"[offset mismatch] {eid}: {event_text} != {text[start:end]}"
                )

        char_count_tags += tag_length - len(event_text)

    return eid2events


def parse(filepath: Path, pipeline) -> tuple[str, dict[str, Any]]:
    """ """

    with open(filepath, "r") as f:
        raw_data = f.read()

    parsed_data = BeautifulSoup(raw_data, "xml")
    text = parsed_data.find("TEXT").text
    eid2events = parse_tags(parsed_data)

    return text, eid2events


def make_example(
    text: str, eid2events: dict[str, Any], raw_annotation: dict[str, Any]
) -> dict[str, Any]:
    if raw_annotation["arg1"] in eid2events and raw_annotation["arg2"] in eid2events:
        event1 = deepcopy(eid2events[raw_annotation["arg1"]])
        event2 = deepcopy(eid2events[raw_annotation["arg2"]])
        while text.startswith("\n"):
            text = text[1:]
            event1["start"], event1["end"] = event1["start"] - 1, event1["end"] - 1
            event2["start"], event2["end"] = event2["start"] - 1, event2["end"] - 1
        text = text.rstrip("\n")

        relation = raw_annotation["relation"]

        if event1["mention"] != text[event1["start"] : event1["end"]]:
            logging.error(
                f"text mismatch: {event1['mention']} "
                f"!= {text[event1['start']:event1['end']]}"
            )
        if event2["mention"] != text[event2["start"] : event2["end"]]:
            logging.error(
                f"text mismatch: {event2['mention']} "
                f"!= {text[event2['start']:event2['end']]}"
            )

        example = {
            "context": text,
            "arg1": event1,
            "arg2": event2,
            "relation": relation,
        }

    else:
        logging.error(
            f"{raw_annotation['arg1']} or {raw_annotation['arg2']}"
            f" not identified in the tml file."
        )
        example = None

    return example


def main(args):
    split2annotation = parse_annotation(args.dirpath_annotation)

    pipeline = spacy.load(args.spacy_model)

    examples = defaultdict(list)
    for split_name, filename2annotation in split2annotation.items():
        logging.info(f"[data split] {split_name}")

        for filename, raw_annotations in filename2annotation.items():
            logging.info(f"[filename]: {filename}")

            filepath = args.dirpath_timebank / f"{filename}.tml"
            assert filepath.exists() is True
            text, eid2events = parse(filepath, pipeline)

            for raw_annotation in raw_annotations:
                example = make_example(text, eid2events, raw_annotation)

                if example:
                    examples[split_name].append(
                        {
                            "filename": filename,
                            "context": example["context"],
                            "arg1": example["arg1"],
                            "arg2": example["arg2"],
                            "relation": example["relation"],
                        }
                    )
                else:
                    logging.error(f"skip annotation: {raw_annotation}")

    with open(args.dirpath_output / "train.json", "w") as f:
        json.dump(examples["train"], f, indent=4)
        f.write("\n")

    with open(args.dirpath_output / "dev.json", "w") as f:
        json.dump(examples["dev"], f, indent=4)
        f.write("\n")

    with open(args.dirpath_output / "test.json", "w") as f:
        json.dump(examples["test"], f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess TDDiscourse data")
    parser.add_argument(
        "--dirpath_timebank", type=Path, help="dirpath to timebank docs"
    )
    parser.add_argument("--dirpath_annotation", type=Path, help="dirpath to annotation")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")
    parser.add_argument(
        "--spacy_model", type=str, help="spacy model", default="en_core_web_sm"
    )
    args = parser.parse_args()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "preprocess_tddiscourse.log"),
        ],
    )
    main(args)
