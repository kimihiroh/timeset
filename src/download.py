"""
Download pretrained weights from huggingface

"""

from argparse import ArgumentParser
import logging
from pathlib import Path
import traceback
from transformers import (
    GenerationConfig,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from utils_model import SEQ2SEQ_MODELS


def main(args):
    logging.info(f"Download {args.model_id} if not cached.")

    try:
        AutoConfig.from_pretrained(args.model_id)
        AutoTokenizer.from_pretrained(args.model_id)
        if args.model_id in SEQ2SEQ_MODELS:
            AutoModelForSeq2SeqLM.from_pretrained(
                args.model_id,
                resume_download=True,
                device_map="auto",
            )
        else:
            AutoModelForCausalLM.from_pretrained(
                args.model_id,
                resume_download=True,
                device_map="auto",
            )
        if args.model_id not in ["t5-3b", "t5-11b"]:
            GenerationConfig.from_pretrained(args.model_id)

        logging.info("Done for HuggingFace")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    parser = ArgumentParser(description="Download LLM")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    parser.add_argument("--model_id", type=str, help="model id in HuggingFace library")
    parser.add_argument("--num_gpu", type=int, help="#gpus")

    args = parser.parse_args()

    if not args.dirpath_log.exists():
        args.dirpath_log.mkdir()

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )

    main(args)
