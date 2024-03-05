"""
Dataset class

"""
import sys
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Any
from template import TEMPLATES
from template_classes import MATRES_RELATIONS, TEMPORALNLI_LABELS, TDDISCOURSE_RELATIONS
from utils_dataset import (
    add_emarkers,
    _select_demonstrations,
    _preprocess_example,
)


"""
dataset for few-shot learning
"""


def load_examples(
    dataset_name: str,
    filepath: str,
) -> list[Any]:
    with open(filepath, "r") as f:
        annotations = json.load(f)

    # load examples for this task
    if dataset_name.startswith("ctf-"):
        task, *_ = dataset_name.replace("ctf-", "").split("-")
        raw_examples = _preprocess_example(annotations, task)
    else:
        raw_examples = annotations

    return raw_examples


def _create_fewshot_examples(
    dataset_name: str,
    raw_examples_test,
    raw_examples_dev,
    num_demonstration: int,
    marker: str,
    representation: str,
):
    """ """

    # get templates
    templates = TEMPLATES[dataset_name]

    examples = []
    for example_id, raw_example_test in enumerate(raw_examples_test):
        # get demonstrations randomly for each example
        demonstrations = _select_demonstrations(
            raw_examples_dev,
            num_demonstration,
            criteria="random",
        )
        for template_name, template in templates.items():
            prompt = ""
            for demonstration in demonstrations:
                prompt += template.get_demonstration(
                    example=demonstration,
                    representation=representation,
                    marker=marker,
                )
            prompt += template.get_prompt(
                example=raw_example_test,
                representation=representation,
                marker=marker,
            )
            examples.append(
                {
                    "example_id": example_id,
                    "template_id": template_name,
                    "representation": representation,
                    "marker": marker,
                    "input": prompt,
                }
            )

    return examples


class InferenceDatasetForFewShotLearning(Dataset):
    """
    dataset class for inference
    """

    def __init__(
        self,
        dataset_name: str,
        raw_examples_test: list[Any],
        raw_examples_dev: list[Any],
        num_demonstration: int,
        marker: str = None,
        representation: str = None,
    ):
        self.examples = _create_fewshot_examples(
            dataset_name=dataset_name,
            raw_examples_test=raw_examples_test,
            raw_examples_dev=raw_examples_dev,
            num_demonstration=num_demonstration,
            marker=marker,
            representation=representation,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


"""
finetune generation model
"""


def _create_example_finetune_generation_mrc(
    example: list[Any],
) -> str:
    """
    create input and output for finetuning models for MRC task

    """
    qst, ctx = example["question"], example["context"]
    input = f"question: {qst} context: {ctx} answer: "
    output = ", ".join([x["mention"] for x in example["answers"]])
    return input, output


def _create_example_finetune_generation_nli(
    example: list[Any],
) -> str:
    """
    create input and output for finetuning models for pairwise task

    """

    ctx, stm = example["context"], example["statement"]
    input = f"context: {ctx} statement: {stm} answer: "
    output = example["label"]

    return input, output


def _create_example_finetune_generation_pairwise(
    example: list[Any],
) -> str:
    """
    create input and output for finetuning models for pairwise task

    """

    ctx = add_emarkers(example)
    input = f"context: {ctx} answer: "
    output = example["relation"]

    return input, output


def _create_examples_finetune_generation(
    dataset_name: str,
    examples: list[Any],
) -> list[Any]:
    """
    create example for finetuning generation models

    """
    new_examples = []
    for example_id, example in enumerate(examples):
        if dataset_name == "torque":
            input, output = _create_example_finetune_generation_mrc(example)
        elif dataset_name == "temporal-nli":
            input, output = _create_example_finetune_generation_nli(example)
        elif dataset_name == "matres":
            input, output = _create_example_finetune_generation_pairwise(example)
        elif dataset_name == "tddiscourse":
            input, output = _create_example_finetune_generation_pairwise(example)
        else:
            logging.error(f"Undefined dataset_name: {dataset_name}")

        new_examples.append(
            {
                "example_id": example_id,
                "template_id": 0,
                "input": input,
                "output": output,
            }
        )

    return new_examples


class InferenceDatasetForFinetunedSeq2SeqModel(Dataset):
    """
    dataset for finetuned/peft model inference
    """

    def __init__(
        self,
        dataset_name: str,
        raw_examples_test: list[Any],
    ):
        self.examples = _create_examples_finetune_generation(
            dataset_name=dataset_name,
            examples=raw_examples_test,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class FinetuneDatasetForSeq2SeqModel(Dataset):
    """
    dataset for finetune seq2seq model
    """

    def __init__(
        self,
        dataset_name: str,
        filepath_data: Path,
        tokenizer: PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

        with open(filepath_data, "r") as f:
            examples = json.load(f)

        self.examples = _create_examples_finetune_generation(
            dataset_name=dataset_name,
            examples=examples,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        input_encoding_batch = self.tokenizer(
            [x["input"] for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids_batch = input_encoding_batch.input_ids
        attention_mask_batch = input_encoding_batch.attention_mask

        target_encoding_batch = self.tokenizer(
            [x["output"] for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        labels_batch = target_encoding_batch.input_ids
        labels_batch[labels_batch == self.tokenizer.pad_token_id] = -100

        return (
            input_ids_batch,
            attention_mask_batch,
            labels_batch,
            input_ids_batch,
            attention_mask_batch,
        )


class InferenceDatasetForFinetunedDecoderModel(Dataset):
    """
    inference dataset for finetuned/peft decoder-only model
    """

    def __init__(
        self,
        dataset_name: str,
        raw_examples_test: list[Any],
    ):
        self.examples = _create_examples_finetune_generation(
            dataset_name=dataset_name,
            examples=raw_examples_test,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class FinetuneDatasetForDecoderModel(Dataset):
    """
    dataset for finetune decoder-only model
    """

    def __init__(
        self,
        dataset_name: str,
        filepath_data: Path,
        tokenizer: PreTrainedTokenizer,
        is_eval: bool = False,
    ):
        with open(filepath_data, "r") as f:
            examples = json.load(f)

        self.examples = _create_examples_finetune_generation(
            dataset_name=dataset_name,
            examples=examples,
        )

        self.tokenizer = tokenizer
        self.is_eval = is_eval

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        # TODO: is this correct?
        # set left in load_tokenizer
        # I think so. Nov23
        self.tokenizer.padding_side = "right"

        encoding_batch = self.tokenizer(
            [(x["input"], x["output"]) for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids_batch = encoding_batch.input_ids
        attention_mask_batch = encoding_batch.attention_mask

        # create labels for decoder-only model
        labels_batch = []
        for i in range(len(batch)):
            labels = []
            for idx, seq_id in enumerate(encoding_batch.sequence_ids(i)):
                flag_eos = input_ids_batch[i][idx].item() == self.tokenizer.eos_token_id
                if seq_id == 1 or (idx > 0 and flag_eos):
                    # calc loss only from target tokens and eos_token
                    # note: for the models whose bos == eos,
                    # only calc loss from the eos at the end.
                    labels.append(input_ids_batch[i][idx])
                else:
                    labels.append(-100)
            labels_batch.append(labels)
        labels_batch = torch.LongTensor(labels_batch)

        # create input for eval as well
        input_ids_batch_eval, attention_mask_batch_eval = None, None
        if self.is_eval:
            self.tokenizer.padding_side = "left"
            encoding_batch_eval = self.tokenizer(
                [x["input"] for x in batch],
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids_batch_eval = encoding_batch_eval.input_ids
            attention_mask_batch_eval = encoding_batch_eval.attention_mask

        return (
            input_ids_batch,
            attention_mask_batch,
            labels_batch,
            input_ids_batch_eval,
            attention_mask_batch_eval,
        )


"""
finetune sequence tagging model
"""


class InferenceDatasetForFinetunedSequenceTaggingModel(Dataset):
    """
    dataset for finetuned sequence tagging model inference
    """

    def __init__(
        self,
        raw_examples_test: list[Any],
        tokenizer: PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

        self.examples = []
        for example_id, example in enumerate(raw_examples_test):
            self.examples.append(
                {
                    "example_id": example_id,
                    "question": example["question"],
                    "context": example["context"],
                    "answers": example["answers"],
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        input_encoding_batch = self.tokenizer(
            [(x["question"], x["context"]) for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return input_encoding_batch


class FinetuneDatasetForSequenceTaggingModel(Dataset):
    """
    dataset for finetune sequence tagging models
    specifically for TORQUE dataset

    """

    def __init__(
        self,
        dataset_name: str,
        filepath_data: Path,
        tokenizer: PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

        with open(filepath_data, "r") as f:
            examples = json.load(f)

        self.examples = []
        for example_id, example in enumerate(examples):
            if dataset_name == "torque":
                self.examples.append(
                    {
                        "example_id": example_id,
                        "template_id": 0,
                        "question": example["question"],
                        "context": example["context"],
                        "answers": example["answers"],
                    }
                )
            else:
                logging.error(f"Undefined dataset name: {dataset_name}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        input_encoding_batch = self.tokenizer(
            [(x["question"], x["context"]) for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # input_ids_batch = input_encoding_batch.input_ids
        # attention_mask_batch = input_encoding_batch.attention_mask

        labels_batch = []

        # create labels for sequence labeling task
        for i in range(len(batch)):
            # init: 0 for context tokens;
            # otherwise, question and special tokens, -100
            labels = [
                0 if x == 1 else -100 for x in input_encoding_batch.sequence_ids(i)
            ]

            for answer in batch[i]["answers"]:
                token_id_start = input_encoding_batch.char_to_token(
                    i, answer["start"], 1
                )
                token_id_end = input_encoding_batch.char_to_token(
                    i, answer["end"] - 1, 1
                )

                if token_id_start and token_id_end:
                    labels[token_id_start : token_id_end + 1] = [1] * (
                        token_id_end + 1 - token_id_start
                    )
                else:
                    # 3 cases in TORQUE training dataset
                    # due to annotation error.
                    logging.debug(
                        f"token_id_start: {token_id_start}, "
                        f"or token_id_end: {token_id_end} not identified."
                    )

            labels_batch.append(labels)

        return input_encoding_batch, torch.LongTensor(labels_batch)


"""
finetune classification model
"""


def _create_example_finetune_classification_nli(
    example: list[Any],
) -> str:
    """
    create input and output for finetuning classification models
    for nli task

    """

    ctx, stm = example["context"], example["statement"]
    input = f"context: {ctx} statement: {stm}"
    output = example["label"]

    return input, output


def _create_example_finetune_classification_pairwise(
    example: list[Any],
) -> str:
    """
    create input and output for finetuning classification models
    for pairwise task

    """

    input = add_emarkers(example)
    output = example["relation"]

    return input, output


def _create_examples_finetune_classification(
    dataset_name: str,
    examples: list[Any],
) -> tuple[list[Any], list[str]]:
    new_examples, label_list = [], None

    for example_id, example in enumerate(examples):
        if dataset_name in ["matres", "tddiscourse"]:
            input, output = _create_example_finetune_classification_pairwise(example)
        elif dataset_name in ["temporal-nli"]:
            input, output = _create_example_finetune_classification_nli(example)
        else:
            logging.error(f"Undefined dataset_name (create example): {dataset_name}")

        new_examples.append(
            {
                "example_id": example_id,
                "template_id": 0,
                "input": input,
                "output": output,
            }
        )

    if dataset_name == "matres":
        label_list = MATRES_RELATIONS
    elif dataset_name == "temporal-nli":
        label_list = TEMPORALNLI_LABELS
    elif dataset_name == "tddiscourse":
        label_list = TDDISCOURSE_RELATIONS
    else:
        logging.error(f"Undefined dataset name (label): {dataset_name}")
        sys.exit("stop")

    return new_examples, label_list


class InferenceDatasetForFinetunedClassificationModel(Dataset):
    """
    dataset for finetuned classification model inference
    """

    def __init__(
        self,
        dataset_name: str,
        raw_examples_test: list[Any],
        tokenizer: PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

        self.examples, self.label_list = _create_examples_finetune_classification(
            dataset_name=dataset_name,
            examples=raw_examples_test,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        input_encoding_batch = self.tokenizer(
            [x["input"] for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return input_encoding_batch


class FinetuneDatasetForClassificationModel(Dataset):
    """
    dataset for finetune classification models

    """

    def __init__(
        self,
        dataset_name: str,
        filepath_data: Path,
        tokenizer: PreTrainedTokenizer,
    ):
        self.tokenizer = tokenizer

        with open(filepath_data, "r") as f:
            examples = json.load(f)

        self.examples, self.label_list = _create_examples_finetune_classification(
            dataset_name=dataset_name,
            examples=examples,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        input_encoding_batch = self.tokenizer(
            [x["input"] for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels_batch = [self.label_list.index(x["output"]) for x in batch]

        return input_encoding_batch, torch.LongTensor(labels_batch)
