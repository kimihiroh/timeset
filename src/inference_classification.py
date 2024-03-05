"""
Inference code for classification model

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from peft import PeftModel
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Any
from my_datasets import (
    InferenceDatasetForFinetunedClassificationModel,
    load_examples,
)
from template_classes import (
    MATRES_RELATIONS,
    TDDISCOURSE_RELATIONS,
    TEMPORALNLI_LABELS,
)

from utils_model import (
    get_date,
    set_seed,
    args2dict,
    load_tokenizer,
    load_classification_model,
)
from evaluate import evaluate


def inference(
    categories: list,
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str,
    tokenizer: PreTrainedTokenizer,
) -> list[Any]:
    """
    inference

    """

    record = []
    example_id = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
            )

            predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            input_text_batch = tokenizer.batch_decode(batch.input_ids)
            for in_text, pred in zip(input_text_batch, predictions):
                record.append(
                    {
                        "input": {
                            "example_id": example_id,
                            "template_id": 0,
                            "input": in_text,
                        },
                        "output": categories[int(pred)],
                    }
                )
                example_id += 1

    return record


def main(args):
    set_seed(args.seed, args.num_gpu, is_eval=True)

    tokenizer, _ = load_tokenizer(args.model_id)
    logging.debug(f"tokenizer: {tokenizer}")

    if args.dataset_name in ["matres", "tddiscourse"]:
        # add emarkers
        special_tokens_dict = {
            "additional_special_tokens": ["[e1]", "[/e1]", "[e2]", "[/e2]"]
        }
        tokenizer.add_special_tokens(special_tokens_dict)

    logging.info("Create dataset for sequence classification model")
    raw_examples_test = load_examples(
        dataset_name=args.dataset_name,
        filepath=args.filepath_test,
    )
    dataset_eval = InferenceDatasetForFinetunedClassificationModel(
        dataset_name=args.dataset_name,
        raw_examples_test=raw_examples_test,
        tokenizer=tokenizer,
    )
    logging.info(f"#dataset_eval is {len(dataset_eval)}")
    logging.debug(f"dataset_eval[0]: {dataset_eval[0]}")

    if args.dataset_name == "tddiscourse":
        batch_size_eval = 4
    else:
        batch_size_eval = args.batch_size

    dataloader_eval = DataLoader(
        dataset=dataset_eval,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=dataset_eval.collate_fn,
    )

    logging.info(f"Load model weight with {args.precision_type} precision")

    if args.dataset_name == "temporal-nli":
        categories = TEMPORALNLI_LABELS
    elif args.dataset_name == "matres":
        categories = MATRES_RELATIONS
    elif args.dataset_name == "tddiscourse":
        categories = TDDISCOURSE_RELATIONS
    else:
        logging.error(f"Undefined dataset name (category): {args.dataset_name}")

    model = load_classification_model(
        model_id=args.model_id,
        model_id_or_path=(
            args.dirpath_model if args.inference_type == "ft" else args.model_id
        ),
        dataset_name=args.dataset_name,
        num_labels=len(categories),
        precision_type=args.precision_type,
    )
    model.to(args.device)
    # logging.info(f"device map: {model.hf_device_map}")

    if args.dataset_name in ["matres", "tddiscourse"]:
        model.set_emarker_ids(tokenizer.convert_tokens_to_ids(["[e1]", "[e2]"]))

    if args.inference_type == "peft":
        logging.info("Load as PeftModel")
        model = PeftModel.from_pretrained(model, args.peft_model_path)

    logging.info("Inference starts")
    record = inference(
        categories=categories,
        model=model,
        dataloader=dataloader_eval,
        device=args.device,
        tokenizer=tokenizer,
    )

    logging.info("Save outputs")
    output = {
        "args": args2dict(args),
        "examples": record,
    }
    dirpath_output = (
        args.dirpath_output
        / args.dataset_name
        / (
            f"{Path(args.model_id).name}_{args.precision_type}"
            f"_{args.inference_type}_classification"
        )
    )
    if not dirpath_output.exists():
        dirpath_output.mkdir(parents=True)

    filename_output = f"{args.seed}.json"
    with open(dirpath_output / filename_output, "w") as f:
        json.dump(output, f, indent=4)
        f.write("\n")

    # eval
    report, report_avg = evaluate(
        dataset_name=args.dataset_name,
        examples=raw_examples_test,
        record=output,
    )
    output_eval = {
        "args": args2dict(args),
        "average": report_avg,
        "individuals": report,
    }
    dirpath_output_score = (
        args.dirpath_output_score
        / args.dataset_name
        / (
            f"{Path(args.model_id).name}_{args.precision_type}"
            f"_{args.inference_type}_classification"
        )
    )
    if not dirpath_output_score.exists():
        dirpath_output_score.mkdir(parents=True)

    with open(dirpath_output_score / filename_output, "w") as f:
        json.dump(output_eval, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code for classification model")

    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument(
        "--device", type=str, help="device: cuda or cpu", default="cuda"
    )
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    parser.add_argument(
        "--dirpath_model", type=Path, help="dirpath to trained model", default=None
    )
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument(
        "--dirpath_output_score",
        type=Path,
        help="dirpath to output data",
        default="./output_score/benchmark/",
    )
    parser.add_argument("--filepath_test", type=Path, help="filepath to test data")
    parser.add_argument(
        "--filepath_dev",
        type=Path,
        help="filepath to dev data for demonstration",
        default=None,
    )
    parser.add_argument(
        "--inference_type",
        type=str,
        help="inference type",
        choices=[
            "ft",
            "peft",
        ],
    )
    parser.add_argument(
        "--max_new_tokens", type=int, help="max new tokens", default=None
    )
    parser.add_argument("--model_id", type=str, help="model id in HuggingFace library")
    parser.add_argument(
        "--num_demonstration", type=int, help="#demonstration", default=None
    )
    parser.add_argument("--num_gpu", type=int, help="#gpu", default=1)
    parser.add_argument(
        "--peft_model_path", type=Path, help="dirpath to peft model", default=None
    )
    parser.add_argument(
        "--precision_type", type=str, help="precision type", default="full"
    )

    parser.add_argument(
        "--seed", type=int, help="random seed for initialization", default=7
    )
    parser.add_argument(
        "--temperature", type=float, help="temperature for decoding", default=None
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
            logging.FileHandler(
                args.dirpath_log / f"inference_classification_{get_date()}.log"
            ),
        ],
    )

    logging.info(f"Arguments: {args}")

    main(args)
