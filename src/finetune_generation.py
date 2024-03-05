"""
Finetune/PEFT code

"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    # prepare_model_for_int8_training,
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_datasets import (
    FinetuneDatasetForSeq2SeqModel,
    FinetuneDatasetForDecoderModel,
)

from utils_eval_mrc import (
    check_if_x_matches_y,
)

from utils_model import (
    get_date,
    set_seed,
    args2dict,
    load_tokenizer,
    load_model,
    SEQ2SEQ_MODELS,
)
from transformers import get_linear_schedule_with_warmup


def extract_answers(string: str, dataset_name: str):
    if dataset_name == "torque":
        answers = [x.strip() for x in string.strip().split(",")]
    else:
        answers = string.strip()
    return answers


def validate_model(
    dataloader,
    dataset_name: str,
    device,
    generation_config,
    max_new_tokens,
    model,
    tokenizer,
) -> tuple[float, float]:
    """
    Calculate loss and performance
    """

    avg_loss = 0.0
    generation_all = []
    exact_match = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            (
                input_ids,
                attention_mask,
                labels,
                input_ids_eval,
                attention_mask_eval,
            ) = batch

            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=labels.to(device),
            )
            avg_loss += outputs.loss.item()

            generated_ids = model.generate(
                input_ids=input_ids_eval.to(device),
                attention_mask=attention_mask_eval.to(device),
                generation_config=generation_config,
            )
            generation_batch = tokenizer.batch_decode(
                generated_ids.detach().cpu().numpy(), skip_special_tokens=True
            )
            generation_all += generation_batch

    avg_loss /= len(dataloader)
    idx = 0
    for example, generation in zip(dataloader.dataset, generation_all):
        gold = extract_answers(string=example["output"], dataset_name=dataset_name)
        generation_wo_input = generation.replace(example["input"], "")
        prediction = extract_answers(
            string=generation_wo_input, dataset_name=dataset_name
        )
        if dataset_name in ["torque"]:
            exact_match.append(check_if_x_matches_y(x=prediction, y=gold))
        else:
            exact_match.append(gold == prediction)

        if idx in [0]:
            logging.debug(f"evaluation set[{idx}]")
            logging.debug(f"raw generation: {generation}")
            logging.debug(f"raw generation w/o input: {generation_wo_input}")
            logging.debug(f"prediction(extracted answers): {prediction}")
            logging.debug(f"gold: {gold}")
            if dataset_name in ["torque"]:
                match = check_if_x_matches_y(x=prediction, y=gold)
            else:
                match = gold == prediction
            logging.debug(f"exact match: {match}")
        idx += 1

    score_exact_match = sum(exact_match) / len(exact_match)

    logging.info(f"Loss (dev): {avg_loss:.4f}")
    logging.info(f"EM (dev): {score_exact_match:.4f}")

    return avg_loss, score_exact_match


def train_one_epoch(
    dataloader,
    device,
    model,
    optimizer,
    scheduler,
    gradient_accumulation_steps: int = 1,
) -> float:
    """
    train a model for one epoch
    """

    avg_loss = 0.0

    model.train()
    for idx, (input_ids, attention_mask, labels, *_) in enumerate(tqdm(dataloader)):
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
        )
        loss = outputs.loss
        avg_loss += loss.item()

        loss.backward()

        if (idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    avg_loss /= len(dataloader)

    logging.info(f"Loss (train): {avg_loss:.4f}")

    return avg_loss


def main(args):
    set_seed(args.seed, args.num_gpu)

    tokenizer, num_new_tokens = load_tokenizer(args.model_id)

    if args.model_id in SEQ2SEQ_MODELS:
        logging.info("Create Dataset for Se2Seq model")
        dataset_train = FinetuneDatasetForSeq2SeqModel(
            dataset_name=args.dataset_name,
            filepath_data=args.filepath_data_train,
            tokenizer=tokenizer,
        )
        dataset_dev = FinetuneDatasetForSeq2SeqModel(
            dataset_name=args.dataset_name,
            filepath_data=args.filepath_data_dev,
            tokenizer=tokenizer,
        )
    else:
        logging.info("Create dataset for Decoder model")
        dataset_train = FinetuneDatasetForDecoderModel(
            dataset_name=args.dataset_name,
            filepath_data=args.filepath_data_train,
            tokenizer=tokenizer,
            is_eval=False,
        )
        dataset_dev = FinetuneDatasetForDecoderModel(
            dataset_name=args.dataset_name,
            filepath_data=args.filepath_data_dev,
            tokenizer=tokenizer,
            is_eval=True,
        )
    logging.info(f"[#examples] train: {len(dataset_train)}, dev: {len(dataset_dev)}")
    logging.debug(f"dataset_dev[0]: {dataset_dev[0]}")

    if args.dataset_name == "tddiscourse":
        gradient_accumulation_steps = args.batch_size
        batch_size = 1
        batch_size_eval = 4
    else:
        gradient_accumulation_steps = 1
        batch_size = args.batch_size
        batch_size_eval = args.batch_size

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset_train.collate_fn,
    )
    dataloader_dev = DataLoader(
        dataset=dataset_dev,
        batch_size=batch_size_eval,
        shuffle=False,
        collate_fn=dataset_dev.collate_fn,
    )

    # === debug === #
    for idx, batch in enumerate(dataloader_dev):
        input_ids, _, labels, input_ids_eval, _ = batch
        decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        if idx == 0:
            logging.debug("dataloader_dev[0]")
            logging.debug(f"input_ids[0]: {input_ids[0].tolist()}")
            logging.debug(f"decoded[0]: {decoded[0]}")
            logging.debug(f"labels[0]: {labels[0].tolist()}")
            logging.debug(f"input_ids_eval[0]: {input_ids_eval[0].tolist()}")
            break
    # === debug === #

    model, generation_config = load_model(
        model_id=args.model_id,
        model_id_or_path=args.model_id,
        precision_type=args.precision_type,
        flag_sample=False,
        tokenizer=tokenizer,
        num_new_tokens=num_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device_map="auto",
    )
    logging.info(f"device map: {model.hf_device_map}")

    if args.finetune_type == "peft":
        if args.peft_type == "lora":
            peft_config = LoraConfig(
                task_type=(
                    TaskType.SEQ_2_SEQ_LM
                    if args.model_id in SEQ2SEQ_MODELS
                    else TaskType.CAUSAL_LM
                ),
                inference_mode=False,
                r=args.lora_dimension,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
        else:
            logging.error(f"Undefined peft type: {args.peft_type}")

        logging.info(f"peft config: {peft_config}")

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    else:
        logging.info("normal finetuning")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(
            int(len(dataloader_train) * args.num_epoch * 0.1) if args.warmup else 0
        ),
        num_training_steps=len(dataloader_train) * args.num_epoch,
    )
    logging.info(
        f"warmup: {args.warmup}, init learning rate: {scheduler.get_last_lr()}"
    )

    # create a folder to store model weight
    _dirpath_model_weight = (
        args.dirpath_output
        / args.dataset_name
        / f"{Path(args.model_id).name}_{args.precision_type}"
        f"_{args.finetune_type}_generation"
    )
    if not _dirpath_model_weight.exists():
        _dirpath_model_weight.mkdir(parents=True)

    dirpath_model_weight = (
        _dirpath_model_weight / f"seed{args.seed}_bs{args.batch_size}"
        f"_lr{args.learning_rate}_dim{args.lora_dimension}"
        f"_alpha{args.lora_alpha}_drop{args.lora_dropout}"
    )
    filepath_training_log = (
        _dirpath_model_weight / f"seed{args.seed}_bs{args.batch_size}"
        f"_lr{args.learning_rate}_dim{args.lora_dimension}"
        f"_alpha{args.lora_alpha}_drop{args.lora_dropout}.log"
    )

    # training
    logging.info("Training start")
    best_score, best_epoch = 0.0, 0
    log = {
        "args": args2dict(args),
        "best": None,
        "train": [],
        "dev": [],
    }

    for epoch in range(args.num_epoch):
        logging.info(f"Epoch: {epoch}")

        loss_train = train_one_epoch(
            dataloader=dataloader_train,
            device=args.device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        log["train"].append(loss_train)

        logging.info(f"learning rate: {scheduler.get_last_lr()}")

        loss_dev, exact_match = validate_model(
            dataloader=dataloader_dev,
            dataset_name=args.dataset_name,
            device=args.device,
            generation_config=generation_config,
            max_new_tokens=args.max_new_tokens,
            model=model,
            tokenizer=tokenizer,
        )
        log["dev"].append(
            {
                "loss": loss_dev,
                "exact_match": exact_match,
            }
        )

        # save model
        if best_score < exact_match:
            best_score, best_epoch = exact_match, epoch
            if args.save_model_weight:
                logging.info(f"[Save model] epoch: {epoch}")
                model.save_pretrained(dirpath_model_weight)

        # save log so far
        logging.info("Save training log so far")
        with open(filepath_training_log, "w") as f:
            json.dump(log, f, indent=4)

    logging.info("Save best score")
    log["best"] = {"score": best_score, "epoch": best_epoch}
    with open(filepath_training_log, "w") as f:
        json.dump(log, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Finetune code for LLM")

    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument("--dataset_name", type=str, help="dataset name")
    parser.add_argument(
        "--device", type=str, help="device: cuda or cpu", default="cuda"
    )
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath to output data")
    parser.add_argument(
        "--filepath_data_train", type=Path, help="filepath to train data"
    )
    parser.add_argument("--filepath_data_dev", type=Path, help="filepath to dev data")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--finetune_type",
        type=str,
        help="finetune type",
        choices=[
            "ft",
            "peft",
        ],
    )
    parser.add_argument(
        "--peft_type", type=str, help="PEFT type", choices=["lora", None], default=None
    )
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate", default=1e-5
    )
    parser.add_argument(
        "--lora_dimension",
        type=int,
        help="the dimension of the low-rank matrices",
        default=None,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        help="the scaling factor for the low-rank matrices",
        default=None,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        help="the dropout probability of the LoRA layers",
        default=None,
    )
    parser.add_argument("--max_new_tokens", type=int, help="max new tokens", default=64)
    parser.add_argument("--model_id", type=str, help="model id in HuggingFace library")
    parser.add_argument(
        "--num_demonstration", type=int, help="#demonstration", default=0
    )
    parser.add_argument("--num_epoch", type=int, help="#training epoch", default=10)
    parser.add_argument("--num_gpu", type=int, help="#gpu", default=1)
    parser.add_argument(
        "--precision_type", type=str, help="precision type", default="float32"
    )
    parser.add_argument(
        "--save_model_weight",
        action="store_true",
        help="True if you want to save model weight",
    )
    parser.add_argument(
        "--seed", type=int, help="random seed for initialization", default=7
    )
    parser.add_argument(
        "--temperature", type=float, help="temperature for decoding", default=0.0
    )
    parser.add_argument(
        "--warmup", action="store_true", help="True if warmup is activated"
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
            logging.FileHandler(args.dirpath_log / f"ft_generation_{get_date()}.log"),
        ],
    )

    logging.info(f"Arguments: {args}")

    main(args)
