"""
Inference code

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from vllm import LLM, SamplingParams
from evaluate import evaluate

from my_datasets import (
    InferenceDatasetForFewShotLearning,
    InferenceDatasetForFinetunedDecoderModel,
    load_examples,
)

from utils_model import (
    get_date,
    set_seed,
    args2dict,
    load_tokenizer,
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
        model_id_or_path = args.model_id
    elif args.inference_type == "ft":
        logging.info("Create Datase for Finetuned Decoder Model")
        examples = InferenceDatasetForFinetunedDecoderModel(
            dataset_name=args.dataset_name,
            filepath_data=args.filepath_test,
        )
        model_id_or_path = args.dirpath_model
    else:
        logging.error(f"Undefined inference type: {args.inference_type}")

    logging.info("Truncate if needed")
    examples = truncate(
        examples=examples,
        tokenizer=tokenizer,
        max_input_length=tokenizer.model_max_length - args.max_new_tokens,
        num_cpu=args.num_cpu,
    )

    logging.info(f"#example is {len(examples)}")
    logging.debug(f"examples[0]['input']: {examples[0]['input']}")

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=["\n\n", "\n\n\n"],
    )

    model = LLM(
        model=model_id_or_path,
        tensor_parallel_size=args.num_gpu,
        dtype=args.precision_type,
        seed=args.seed,
        download_dir=args.dirpath_model_cache,
    )

    logging.info("Prediction start")

    prompts = [x["input"] for x in examples]
    outputs = []
    for i in range(0, len(prompts), args.chunk_size):
        """
        apparrently, vLLM stores all prompts in CPU RAM before prediction,
        which requires a large amount of CPU RAM,
        e.g., nli formulation w/ 40 our docs, it can go up to 25GB.
        To reduce it, chunk the prompts into hard-coded size.
        """
        logging.info(f"Chunk {i}/{len(prompts)}")
        outputs += model.generate(prompts[i : i + args.chunk_size], sampling_params)

    logging.info("Postprocessing")
    record = []
    for example, output in zip(examples, outputs):
        record.append(
            {
                "input": example,
                "output": output.outputs[0].text,
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
            f"_{args.inference_type}_generation_vllm"
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
            f"_{args.inference_type}_generation_vllm"
        )
    )
    if not dirpath_output_score.exists():
        dirpath_output_score.mkdir(parents=True)
    filepath_score = dirpath_output_score / filename_output
    with open(filepath_score, "w") as f:
        json.dump(output_eval, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Inference code w/ vLLM. " "This code can be used for only Decoder models"
        )
    )

    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--chunk_size", type=int, help="vLLM chunk size", default=4096)
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
    parser.add_argument("--num_cpu", type=int, help="#cpu", default=4)
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
        "--temperature", type=float, help="temperature for decoding", default=0.0
    )
    parser.add_argument(
        "--dirpath_model_cache", type=Path, help="vllm cache/download dir", default=None
    )
    parser.add_argument("--top_p", type=float, help="top_p for decoding", default=1)

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
            logging.FileHandler(args.dirpath_log / f"inference_vllm_{get_date()}.log"),
        ],
    )

    logging.info(f"Arguments: {args}")

    main(args)
