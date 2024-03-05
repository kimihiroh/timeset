"""
Finetune sequence tagging model
for TORQUE

"""

from argparse import ArgumentParser
import json
import logging
import numpy as np
from pathlib import Path
from peft import LoraConfig, TaskType, get_peft_model
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from my_datasets import FinetuneDatasetForSequenceTaggingModel

from utils_model import (
    get_date,
    set_seed,
    args2dict,
    load_tokenizer,
    decode,
    load_sequence_tagging_model,
)

from utils_eval_mrc import (
    check_if_x_matches_y,
)


def extract_answers(string: str):
    answers = [x.strip() for x in string.strip().split(",")]
    return answers


def calc_exact_match_id(
    logits_batch,
    labels_batch,
):
    preds_batch = torch.argmax(logits_batch, dim=-1).cpu().numpy()
    golds_batch = labels_batch.cpu().numpy()

    exact_match_id = [
        np.array_equal(preds[np.where(golds >= 0)], golds[np.where(golds >= 0)])
        for preds, golds in zip(preds_batch, golds_batch)
    ]

    return exact_match_id


def validate_model(
    dataloader: DataLoader,
    device: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
) -> tuple[float, float]:
    """
    Calculate loss and performance
    """

    avg_loss = 0.0
    prediction_all = []
    exact_match = []
    exact_match_id = []

    model.eval()
    with torch.no_grad():
        for input_encodings_batch, labels_batch in tqdm(dataloader):
            input_ids_batch = input_encodings_batch.input_ids
            attention_mask_batch = input_encodings_batch.attention_mask

            outputs = model(
                input_ids=input_ids_batch.to(device),
                attention_mask=attention_mask_batch.to(device),
                labels=labels_batch.to(device),
            )

            avg_loss += outputs.loss.item()

            exact_match_id += calc_exact_match_id(outputs.logits, labels_batch)
            prediction_all += decode(input_encodings_batch, outputs.logits, tokenizer)

    avg_loss /= len(dataloader)

    idx = 0
    for example, raw_prediction in zip(dataloader.dataset, prediction_all):
        gold = [x["mention"] for x in example["answers"]]
        prediction = extract_answers(raw_prediction)
        exact_match.append(check_if_x_matches_y(x=prediction, y=gold))

        if idx in [0]:
            logging.debug(f"evaluation set[{idx}]")
            logging.debug(f"raw prediction: {raw_prediction}")
            logging.debug(f"prediction (extracted): {prediction}")
            logging.debug(f"gold: {gold}")
            logging.debug(f"exact match: {check_if_x_matches_y(x=prediction, y=gold)}")
        idx += 1

    score_exact_match = sum(exact_match) / len(exact_match)
    score_exact_match_id = sum(exact_match_id) / len(exact_match_id)

    logging.info(f"Loss (dev): {avg_loss:.4f}")
    logging.info(f"EM (dev): {score_exact_match:.4f}")
    logging.info(f"EM (id) (dev): {score_exact_match_id:.4f}")

    return avg_loss, score_exact_match


def train_one_epoch(
    dataloader,
    device,
    model,
    optimizer,
    scheduler,
) -> float:
    """
    train a model for one epoch
    """

    avg_loss = 0.0

    model.train()
    for input_encodings_batch, labels_batch in tqdm(dataloader):
        input_ids_batch = input_encodings_batch.input_ids
        attention_mask_batch = input_encodings_batch.attention_mask

        outputs = model(
            input_ids=input_ids_batch.to(device),
            attention_mask=attention_mask_batch.to(device),
            labels=labels_batch.to(device),
        )

        loss = outputs.loss
        avg_loss += loss.item()

        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss /= len(dataloader)

    logging.info(f"Loss (train): {avg_loss:.4f}")

    return avg_loss


def main(args):
    set_seed(args.seed, args.num_gpu)

    tokenizer, _ = load_tokenizer(args.model_id)

    logging.info("Create dataset")
    dataset_train = FinetuneDatasetForSequenceTaggingModel(
        dataset_name=args.dataset_name,
        filepath_data=args.filepath_data_train,
        tokenizer=tokenizer,
    )
    dataset_dev = FinetuneDatasetForSequenceTaggingModel(
        dataset_name=args.dataset_name,
        filepath_data=args.filepath_data_dev,
        tokenizer=tokenizer,
    )
    logging.info(f"[#examples] train: {len(dataset_train)}, dev: {len(dataset_dev)}")
    logging.debug(f"dataset_dev[0]: {dataset_dev[0]}")

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset_train.collate_fn,
    )
    dataloader_dev = DataLoader(
        dataset=dataset_dev,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset_dev.collate_fn,
    )

    # === debug === #
    for idx, (input_ids, labels) in enumerate(dataloader_dev):
        input_ids = input_ids.input_ids
        decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        if idx == 0:
            logging.debug("dataloader_dev[0]")
            logging.debug(f"input_ids[0]: {input_ids[0].tolist()}")
            logging.debug(f"decoded[0]: {decoded[0]}")
            logging.debug(f"labels[0]: {labels[0].tolist()}")
            break
    # === debug === #

    model = load_sequence_tagging_model(
        model_id=args.model_id,
        model_id_or_path=args.model_id,
        precision_type=args.precision_type,
    )
    model.to(args.device)
    # logging.info(f"device map: {model.hf_device_map}")

    if args.finetune_type == "peft":
        if args.peft_type == "lora":
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
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
        f"warmup: {args.warmup},  init learning rate: {scheduler.get_last_lr()}"
    )

    # create a folder to store model weight
    _dirpath_model_weight = (
        args.dirpath_output
        / args.dataset_name
        / f"{Path(args.model_id).name}_{args.precision_type}"
        f"_{args.finetune_type}_sequence_tagging"
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
        )
        log["train"].append(loss_train)

        logging.info(f"learning rate: {scheduler.get_last_lr()}")

        loss_dev, exact_match = validate_model(
            dataloader=dataloader_dev,
            device=args.device,
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
    parser = ArgumentParser(description="Finetuning for sequence tagging")

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
    parser.add_argument("--peft_type", type=str, help="PEFT type", default=None)
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
    parser.add_argument("--model_id", type=str, help="model id in HuggingFace library")
    parser.add_argument("--num_epoch", type=int, help="#training epoch", default=10)
    parser.add_argument("--num_gpu", type=int, help="#gpu", default=1)
    parser.add_argument(
        "--precision_type", type=str, help="precision type", default="full"
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
            logging.FileHandler(args.dirpath_log / f"peft_{get_date()}.log"),
        ],
    )

    logging.info(f"Arguments: {args}")

    main(args)
