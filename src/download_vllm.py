"""
Download pretrained weights for vLLM

"""

from argparse import ArgumentParser
import logging
from pathlib import Path
import traceback
from vllm import LLM
from utils_model import SEQ2SEQ_MODELS


def main(args):
    logging.info(f"Download {args.model_id} if not cached for vLLM.")

    try:
        if args.model_id not in SEQ2SEQ_MODELS:
            LLM(
                model=args.model_id,
                tensor_parallel_size=args.num_gpu,
                download_dir=args.dirpath_model_cache,
            )
            logging.info("Done for vLLM")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    parser = ArgumentParser(description="Download LLM for vLLM")
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    parser.add_argument(
        "--dirpath_model_cache", type=Path, help="dirpath for model cache"
    )
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
