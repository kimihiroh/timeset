"""
Metrics calculation code

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from utils_eval_nli import (
    evaluate_nli_benchmark,
    evaluate_nli_ctf,
)
from utils_eval_pairwise import (
    evaluate_pairwise_benchmark,
    evaluate_pairwise_ctf,
)
from utils_eval_mrc import (
    evaluate_mrc_torque,
    evaluate_mrc_ctf,
)
from utils_eval_timeline import evaluate_timeline_ctf
from template import TEMPLATES
from my_datasets import load_examples


def evaluate(
    dataset_name: str,
    examples,
    record,
):
    match dataset_name:
        case "matres" | "tddiscourse":
            report_avg, report = evaluate_pairwise_benchmark(
                examples=examples,
                predictions=record["examples"],
                dataset_name=dataset_name,
            )
        case "torque":
            report_avg, report = evaluate_mrc_torque(
                examples=examples,
                predictions=record["examples"],
            )
        case "temporal-nli":
            report_avg, report = evaluate_nli_benchmark(
                examples=examples,
                predictions=record["examples"],
                templates=TEMPLATES[dataset_name],
                dataset_name=dataset_name,
            )
        case "ctf-nli":
            report_avg, report = evaluate_nli_ctf(
                examples=examples,
                predictions=record["examples"],
                templates=TEMPLATES[dataset_name],
            )
        case "ctf-pairwise":
            report_avg, report = evaluate_pairwise_ctf(
                examples=examples,
                predictions=record["examples"],
            )
        case "ctf-mrc" | "ctf-mrc-cot":
            report_avg, report = evaluate_mrc_ctf(
                examples=examples,
                record=record,
            )
        case "ctf-timeline" | "ctf-timeline-cot" | "ctf-timeline-code":
            report_avg, report = evaluate_timeline_ctf(
                examples=examples,
                record=record,
            )
        case _:
            logging.error(f"Undefined dataset name: {record['args']['dataset_name']}")

    return report, report_avg


def main(args):
    with open(args.filepath_pred, "r") as f:
        record = json.load(f)

    assert "dataset_name" in record["args"]

    raw_examples_test = load_examples(
        dataset_name=record["args"]["dataset_name"], filepath=args.filepath_test
    )

    report, report_avg = evaluate(
        dataset_name=record["args"]["dataset_name"],
        examples=raw_examples_test,
        record=record,
    )

    output = {"args": record["args"], "average": report_avg, "individuals": report}

    filepath_score = Path(str(args.filepath_pred).replace("output", "output_score"))
    if not filepath_score.parent.exists():
        filepath_score.parent.mkdir(parents=True)
    with open(filepath_score, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="evaluation code to calc scores")

    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    parser.add_argument("--filepath_test", type=Path, help="filepath to test data")
    parser.add_argument("--filepath_pred", type=Path, help="filepath to pred data")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / "evaluate.log"),
        ],
    )

    logging.info(f"Arguments: {args}")

    main(args)
