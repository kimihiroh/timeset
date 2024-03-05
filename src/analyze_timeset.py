"""

"""

from argparse import ArgumentParser
from collections import defaultdict
import json
import logging
from pathlib import Path
import spacy
from tqdm import tqdm


PARENT_TOPICS = {
    "computers",
    "crime_and_law_accuse",
    "culture_and_entertainment",
    "disasters_and_accidents",
    "economy_and_business",
    "environment",
    "government_and_politics",
    "health",
    "infectious_disease",
    "internet",
    "mining",
    "politics_and_conflicts",
    "rail_transport",
    "space",
    "sports",
    "weather",
}


def get_parent_topic(text: str):
    output = ""
    for topic in PARENT_TOPICS:
        if topic in text:
            if output:
                logging.warning("matched")
            else:
                output = topic
    return output


def convert_category(num: str) -> tuple[str, str]:
    match num:
        case "1":
            return "New", "Short"
        case "2":
            return "New", "Long"
        case "3":
            return "Old", "Short"
        case "4":
            return "Old", "Long"
        case _:
            logging.error(f"Undefined category: {num}")


def extract_topic(text: str):
    text = text.replace("_extra", "").split("_")
    topic = "_".join(text[:-1])
    category = convert_category(text[-1])

    parent_topic = get_parent_topic(topic)
    child_topic = topic.replace(parent_topic, "").strip("_")

    return category[0], category[1], parent_topic, child_topic


def convert_month(text: str) -> str:
    output = None
    match text:
        case "January":
            output = "01"
        case "February":
            output = "02"
        case "March":
            output = "03"
        case "April":
            output = "04"
        case "May":
            output = r"05"
        case "June":
            output = "06"
        case "July":
            output = "07"
        case "August":
            output = "08"
        case "September":
            output = "09"
        case "October":
            output = "10"
        case "November":
            output = "11"
        case "December":
            output = "12"
        case _:
            logging.error(f"{text} cannot be converted.")
    return output


def extract_date(text: str) -> str:
    """
    extract date
    input: Wikinews article
    output: yyyy-mm-dd
    """

    _, date, *_ = text.split("\n")
    _, month_day, year = date.split(",")
    month, day = month_day.split()
    month_num = convert_month(month)

    if not month_num:
        return None

    year, day = year.strip(), day.strip()
    date_str = f"{year}-{month_num}-{day}"

    return date_str


def main(args):
    parser = spacy.load("en_core_web_sm")

    for filepath in args.dirpath_input.glob("*.json"):
        logging.info(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        metadata = defaultdict(lambda: defaultdict(str))
        for _data in tqdm(data):
            filename = _data["filename"]
            new_or_old, length, *topic = extract_topic(filename)
            metadata[filename]["date(binary)"] = new_or_old
            metadata[filename]["length(binary)"] = length
            metadata[filename]["topic"] = topic

            date = extract_date(_data["text"])
            metadata[filename]["date"] = date

            events = _data["annotation"]["events"]
            metadata[filename]["#event"] = len(events)

            metadata[filename]["#word"] = len(parser(_data["text"]))

            metadata[filename]["#sent"] = len([x for x in parser(_data["text"]).sents])

        with open(args.dirpath_output / filepath.name, "w") as f:
            json.dump(metadata, f, indent=4)
            f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract metadata from Wikinews articles")
    parser.add_argument("--dirpath_input", type=Path, help="dirpath to input data")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    args = parser.parse_args()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )

    logging.info(args)

    main(args)
