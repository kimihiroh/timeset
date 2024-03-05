"""
Inference code for sequence tagging model

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Any
from my_datasets import (
    InferenceDatasetForFinetunedSequenceTaggingModel,
    load_examples,
)

from utils_model import (
    get_date,
    set_seed,
    args2dict,
    load_tokenizer,
    load_sequence_tagging_model,
    decode,
)
from evaluate import evaluate


def inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader,
    device: str,
) -> list[Any]:
    """
    inference

    """
    record = []
    example_id = 0
    for input_encodings_batch in tqdm(dataloader):
        input_ids_batch = input_encodings_batch.input_ids
        attention_mask_batch = input_encodings_batch.attention_mask

        outputs = model(
            input_ids=input_ids_batch.to(device),
            attention_mask=attention_mask_batch.to(device),
        )
        predicted_text_batch = decode(input_encodings_batch, outputs.logits, tokenizer)
        input_text_batch = tokenizer.batch_decode(input_ids_batch)
        for in_text, pred_text in zip(input_text_batch, predicted_text_batch):
            record.append(
                {
                    "input": {
                        "example_id": example_id,
                        "template_id": 0,
                        "input": in_text,
                    },
                    "output": pred_text,
                }
            )
            example_id += 1

    return record


def main(args):
    set_seed(args.seed, args.num_gpu, is_eval=True)

    tokenizer, num_new_tokens = load_tokenizer(args.model_id)

    logging.info("Create dataset for sequence tagging model")
    raw_examples_test = load_examples(
        dataset_name=args.dataset_name,
        filepath=args.filepath_test,
    )
    dataset_eval = InferenceDatasetForFinetunedSequenceTaggingModel(
        raw_examples_test=raw_examples_test,
        tokenizer=tokenizer,
    )
    logging.info(f"#dataset_eval is {len(dataset_eval)}")
    logging.debug(f"dataset_eval[0]: {dataset_eval[0]}")

    dataloader_eval = DataLoader(
        dataset=dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset_eval.collate_fn,
    )

    logging.info(f"Load model weight with {args.precision_type} precision")

    model = load_sequence_tagging_model(
        model_id=args.model_id,
        model_id_or_path=(
            args.dirpath_model if args.inference_type == "ft" else args.model_id
        ),
        precision_type=args.precision_type,
    )
    model.to(args.device)
    # logging.info(f"device map: {model.hf_device_map}")

    if args.inference_type == "peft":
        logging.info("Load as PeftModel")
        model = PeftModel.from_pretrained(model, args.peft_model_path)

    logging.info("Inference starts")
    record = inference(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader_eval,
        device=args.device,
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
            f"_{args.inference_type}_tagging"
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
            f"_{args.inference_type}_tagging"
        )
    )
    if not dirpath_output_score.exists():
        dirpath_output_score.mkdir(parents=True)

    with open(dirpath_output_score / filename_output, "w") as f:
        json.dump(output_eval, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code for sequence tagging model")

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
    parser.add_argument("--filepath_test", type=Path, help="filepath to input data")
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
                args.dirpath_log / f"inference_tagging_{get_date()}.log"
            ),
        ],
    )

    logging.info(f"Arguments: {args}")

    main(args)
