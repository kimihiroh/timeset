"""
Inference code

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from evaluate import evaluate

from my_datasets import (
    InferenceDatasetForFewShotLearning,
    InferenceDatasetForFinetunedSeq2SeqModel,
    InferenceDatasetForFinetunedDecoderModel,
    load_examples,
)

from utils_model import (
    get_date,
    set_seed,
    args2dict,
    load_tokenizer,
    load_model,
    SEQ2SEQ_MODELS,
    truncate,
)


def main(args):
    set_seed(args.seed, args.num_gpu, is_eval=True)

    tokenizer, num_new_tokens = load_tokenizer(args.model_id)
    logging.debug(f"tokenizer: {tokenizer}")

    if args.inference_type == "few-shot":
        logging.info("Create Dataset for Few-shot Learning")
        raw_examples_dev = load_examples(
            dataset_name=args.dataset_name,
            filepath=args.filepath_dev,
        )
        raw_examples_test = load_examples(
            dataset_name=args.dataset_name,
            filepath=args.filepath_test,
        )
        examples = InferenceDatasetForFewShotLearning(
            dataset_name=args.dataset_name,
            raw_examples_test=raw_examples_test,
            raw_examples_dev=raw_examples_dev,
            marker=args.marker,
            num_demonstration=args.num_demonstration,
            representation=args.representation,
        )
    else:
        if args.model_id in SEQ2SEQ_MODELS:
            logging.info("Create Datase for Finetuned Seq2Seq Model")
            raw_examples_test = load_examples(
                dataset_name=args.dataset_name,
                filepath=args.filepath_test,
            )
            examples = InferenceDatasetForFinetunedSeq2SeqModel(
                dataset_name=args.dataset_name,
                raw_examples_test=raw_examples_test,
            )
        else:
            logging.info("Create Datase for Finetuned Decoder Model")
            raw_examples_test = load_examples(
                dataset_name=args.dataset_name,
                filepath=args.filepath_test,
            )
            examples = InferenceDatasetForFinetunedDecoderModel(
                dataset_name=args.dataset_name,
                raw_examples_test=raw_examples_test,
            )

    logging.info("Truncate if needed")
    examples = truncate(
        examples=examples,
        tokenizer=tokenizer,
        max_input_length=tokenizer.model_max_length - args.max_new_tokens,
        num_cpu=args.num_cpu,
    )

    logging.info(f"#example is {len(examples)}")
    logging.debug(f"examples[0]['input']: {examples[0]['input']}")

    # todo: finetuned model may be able to load generation config from path
    model, generation_config = load_model(
        model_id=args.model_id,
        model_id_or_path=(
            args.dirpath_model if args.inference_type == "ft" else args.model_id
        ),
        precision_type=args.precision_type,
        flag_sample=False,
        tokenizer=tokenizer,
        num_new_tokens=num_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map="auto",
    )
    if args.inference_type == "peft":
        logging.info("Load as PeftModel")
        model = PeftModel.from_pretrained(model, args.peft_model_path)

    logging.info(f"device map: {model.hf_device_map}")

    logging.debug(f"model.config: {model.config}")

    # TODO: better to setup stopping criteria like in the script with vLLM?
    generator = pipeline(
        "text2text-generation"
        if args.model_id in SEQ2SEQ_MODELS
        else "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    logging.info("Inference starts")
    generated_texts = []
    for output in tqdm(
        generator(
            KeyDataset(examples, "input"),
            batch_size=args.batch_size,
            # truncation=True if args.model_id in SEQ2SEQ_MODELS else False,
        ),
        total=len(examples),
    ):
        generated_texts.extend(output)

    logging.info("Postprocess")
    record = []
    for input, generated_text in zip(examples, generated_texts):
        record.append(
            {
                "input": input,
                "output": (
                    generated_text["generated_text"].replace(input["input"], "")
                ),
            }
        )

    logging.info("Save outputs")
    output = {
        "args": args2dict(args),
        "examples": record,
    }

    if args.dataset_name.startswith("ctf-"):
        dataset_name = args.dataset_name.replace("ctf-", "")
    else:
        dataset_name = args.dataset_name

    dirpath_output = (
        args.dirpath_output
        / dataset_name
        / (
            f"{Path(args.model_id).name}_{args.precision_type}"
            f"_{args.inference_type}_generation"
        )
    )
    if not dirpath_output.exists():
        dirpath_output.mkdir(parents=True)

    filename_output = (
        f"{args.num_demonstration}_{args.representation}"
        f"_{args.marker}_{args.seed}.json"
    )

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
        / dataset_name
        / (
            f"{Path(args.model_id).name}_{args.precision_type}"
            f"_{args.inference_type}_generation"
        )
    )
    if not dirpath_output_score.exists():
        dirpath_output_score.mkdir(parents=True)
    filepath_score = dirpath_output_score / filename_output
    with open(filepath_score, "w") as f:
        json.dump(output_eval, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Inference code")

    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--dataset_name", type=str, help="dataset_name")
    parser.add_argument(
        "--device", type=str, help="device: cuda or cpu", default="cuda"
    )
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    parser.add_argument(
        "--dirpath_model", type=Path, help="dirpath to finetuned model", default=None
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
            "few-shot",
            "peft",
        ],
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--marker", type=str, default=None)
    parser.add_argument(
        "--max_new_tokens", type=int, help="max new tokens", default=256
    )
    parser.add_argument("--model_id", type=str, help="model id in HuggingFace library")
    parser.add_argument(
        "--num_demonstration", type=int, help="#demonstration", default=0
    )
    parser.add_argument("--num_gpu", type=int, help="#gpu", default=1)
    parser.add_argument("--num_cpu", type=int, help="#cpu", default=2)
    parser.add_argument(
        "--peft_model_path", type=Path, help="dirpath to peft model", default=None
    )
    parser.add_argument(
        "--precision_type", type=str, help="precision type", default="full"
    )
    parser.add_argument(
        "--representation", type=str, help="event representation", default=None
    )
    parser.add_argument(
        "--seed", type=int, help="random seed for initialization", default=7
    )
    parser.add_argument(
        "--temperature", type=float, help="temperature for decoding", default=1.0
    )
    parser.add_argument(
        "--dirpath_model_cache", type=Path, help="vllm cache/download dir", default=None
    )

    args = parser.parse_args()

    if not args.dirpath_output.exists():
        args.dirpath_output.mkdir(parents=True)

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir(parents=True)

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.dirpath_log / f"inference_{get_date()}.log"),
        ],
    )

    logging.info(f"Arguments: {args}")

    main(args)
