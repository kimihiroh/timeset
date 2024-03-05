"""

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup
from collections import defaultdict
import random
import re
import spacy


def parse_annotation(dirpath: Path) -> dict[str, Any]:
    annotation = {}
    for filepath in dirpath.glob("*.txt"):
        with open(filepath, "r") as f:
            lines = f.readlines()

        filename2annotation = defaultdict(list)
        for line in lines:
            splits = line.strip().split("\t")
            filename2annotation[splits[0]].append(
                {"eiid1": splits[3], "eiid2": splits[4], "relation": splits[5]}
            )
        annotation[filepath.stem] = filename2annotation

    return annotation


def parse_text(text: str, pipeline) -> dict[str, Any]:
    sentences = []
    for paragraph in [x for x in text.split("\n") if x]:
        for sentence in pipeline(paragraph).sents:
            sentence = sentence.text.strip()

            if not sentence:
                continue

            sentences.append(
                {
                    "text": sentence,
                    "start": text.index(sentence),
                    "end": text.index(sentence) + len(sentence),
                }
            )

    return sentences


def parse_tags(parsed_data) -> dict[str, Any]:
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

    eid2eiid = {}
    for tag in parsed_data.find_all("MAKEINSTANCE"):
        eid = tag.get("eventID")
        eiid = tag.get("eiid")
        eid2eiid[eid] = eiid

    and_markers = [x.start() for x in re.finditer("\&", text_with_tags)]

    eiid2events = {}
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

        if eid and eid in eid2eiid:
            start = tag_offset - char_count_tags - modifier
            end = start + len(event_text)
            eiid = eid2eiid[eid].replace("ei", "")

            # hack to deal with annotation error, ' Taking'
            # in APW19980227.0487 in timebank split
            if event_text.startswith(" "):
                event_text = event_text[1:]
                start += 1
                char_count_tags -= 1

            eiid2events[eiid] = {"mention": event_text, "start": start, "end": end}

            if event_text != text[start:end]:
                logging.error(
                    f"[offset mismatch] {eid}: {event_text} != {text[start:end]}"
                )

        char_count_tags += tag_length - len(event_text)

    return eiid2events


def parse(filepath: Path, pipeline) -> tuple[list, dict[str, Any]]:
    with open(filepath, "r") as f:
        raw_data = f.read()

    parsed_data = BeautifulSoup(raw_data, "xml")
    sentences = parse_text(parsed_data.find("TEXT").text, pipeline)
    eiid2events = parse_tags(parsed_data)

    return sentences, eiid2events


def find_source_sentence(sentences: list, event: dict[str, Any]) -> int:
    corresponding_sent_id = -1
    for sent_id, sentence in enumerate(sentences):
        if sentence["start"] <= event["start"] and event["end"] <= sentence["end"]:
            corresponding_sent_id = sent_id
            break

    return corresponding_sent_id


def update_offset(event: dict[str, Any], sentence: dict[str, Any]) -> dict[str, Any]:
    start = event["start"] - sentence["start"]
    end = event["end"] - sentence["start"]

    if event["mention"] != sentence["text"][start:end]:
        print(f"text mismatch: {event['text']} != {sentence['text'][start:end]}")

    new_event = {
        "mention": event["mention"],
        "start": start,
        "end": end,
    }
    return new_event


def make_example(
    sentences: list, eiid2events: dict[str, Any], raw_annotation: dict[str, Any]
) -> dict[str, Any]:
    if (
        raw_annotation["eiid1"] in eiid2events
        and raw_annotation["eiid2"] in eiid2events
    ):
        event1 = eiid2events[raw_annotation["eiid1"]]
        event2 = eiid2events[raw_annotation["eiid2"]]

        relation = raw_annotation["relation"]

        sent_id_event1 = find_source_sentence(sentences, event1)
        sent_id_event2 = find_source_sentence(sentences, event2)

        if sent_id_event1 > -1 and sent_id_event2 > -1:
            event = {
                "arg1": update_offset(event1, sentences[sent_id_event1]),
                "arg2": update_offset(event2, sentences[sent_id_event2]),
            }

            # note: spaCy's sentence segmentation splits more then
            # what MATRES used.
            sent_id_begin, sent_id_end = None, None
            later_event = None
            if sent_id_event1 < sent_id_event2:
                sent_id_begin = sent_id_event1
                sent_id_end = sent_id_event2
                later_event = "arg2"
            elif sent_id_event1 > sent_id_event2:
                sent_id_begin = sent_id_event2
                sent_id_end = sent_id_event1
                later_event = "arg1"
            elif sent_id_event1 == sent_id_event2:
                sent_id_begin, sent_id_end = sent_id_event1, sent_id_event2
            else:
                logging.error("something wrong")

            context = ""
            for sentence in sentences[sent_id_begin:sent_id_end]:
                context += f"{sentence['text']} "
            if later_event:
                event[later_event]["start"] += len(context)
                event[later_event]["end"] += len(context)
            context += sentences[sent_id_end]["text"]

            if (
                event["arg1"]["mention"]
                != context[event["arg1"]["start"] : event["arg1"]["end"]]
            ):
                logging.error(
                    f"text mismatch: {event['arg2']['mention']} "
                    f"!= {context[event['arg1']['start']:event['arg1']['end']]}"
                )
            if (
                event["arg2"]["mention"]
                != context[event["arg2"]["start"] : event["arg2"]["end"]]
            ):
                logging.error(
                    f"text mismatch: {event['arg2']['mention']} "
                    f"!= {context[event['arg2']['start']:event['arg2']['end']]}"
                )

            example = {
                "context": context,
                "arg1": event["arg1"],
                "arg2": event["arg2"],
                "relation": relation,
            }

        else:
            logging.error(
                f"[corresponding sentence not found] "
                f"e1: {sent_id_event1}, e2: {sent_id_event2}"
            )
            example = None
    else:
        logging.error(
            f"ei{raw_annotation['eiid1']} or ei{raw_annotation['eiid2']}"
            f" not identified in the tml file."
        )
        example = None

    return example


def main(args):
    split2annotation = parse_annotation(args.dirpath_annotation)

    split2dirpath = {
        "timebank": args.dirpath_timebank,
        "aquaint": args.dirpath_aquaint,
        "platinum": args.dirpath_platinum,
    }

    pipeline = spacy.load(args.spacy_model)

    examples = []
    for split_name, filename2annotation in split2annotation.items():
        logging.info(f"[data split] {split_name}")

        dirpath_source_document = split2dirpath[split_name]

        for filename, raw_annotations in filename2annotation.items():
            logging.info(f"[filename]: {filename}")

            sentences, eiid2events = parse(
                dirpath_source_document / f"{filename}.tml", pipeline
            )

            for raw_annotation in raw_annotations:
                example = make_example(sentences, eiid2events, raw_annotation)

                if example:
                    examples.append(
                        {
                            "split_name": split_name,
                            "filename": filename,
                            "context": example["context"],
                            "arg1": example["arg1"],
                            "arg2": example["arg2"],
                            "relation": example["relation"],
                        }
                    )
                else:
                    logging.error(f"skip annotation: {raw_annotation}")

    examples_train_and_dev, examples_test = [], []
    for example in examples:
        if example["split_name"] == "platinum":
            examples_test.append(example)
        else:
            examples_train_and_dev.append(example)

    random.seed(args.seed)
    random.shuffle(examples_train_and_dev)
    threshold_20percent = int(len(examples_train_and_dev) * 0.2)
    examples_dev = examples_train_and_dev[:threshold_20percent]
    examples_train = examples_train_and_dev[threshold_20percent:]

    with open(args.dirpath_output / "train.json", "w") as f:
        json.dump(examples_train, f, indent=4)
        f.write("\n")

    with open(args.dirpath_output / "dev.json", "w") as f:
        json.dump(examples_dev, f, indent=4)
        f.write("\n")

    with open(args.dirpath_output / "test.json", "w") as f:
        json.dump(examples_test, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Preprocess MATRES data")
    parser.add_argument(
        "--dirpath_timebank", type=Path, help="dirpath to timebank docs"
    )
    parser.add_argument("--dirpath_aquaint", type=Path, help="dirpath to aquaint docs")
    parser.add_argument(
        "--dirpath_platinum", type=Path, help="dirpath to platinum docs"
    )
    parser.add_argument("--dirpath_annotation", type=Path, help="dirpath to annotation")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath to log")
    parser.add_argument(
        "--spacy_model", type=str, help="spacy model", default="en_core_web_sm"
    )
    parser.add_argument("--seed", type=int, help="random seed", default=7)
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
            logging.FileHandler(args.dirpath_log / "preprocess_matres.log"),
        ],
    )
    main(args)
