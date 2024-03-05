"""
shared functions

"""
from argparse import ArgumentParser
from datetime import datetime
import logging
import random
import multiprocessing
import numpy as np
from pathlib import Path
from typing import Any, Optional
import torch
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from tokenizers import processors

from my_models import (
    T5ForTokenClassification,
    DebertaForSequenceClassificationEMarker,
)

DEFAULT_PAD_TOKEN = "[PAD]"

SEQ2SEQ_MODELS = [
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]

T5_FAMILY = [
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "google/flan-t5-xl",
    "google/flan-t5-xxl",
]


def get_date(granularity: Optional[str] = "min") -> str:
    """
    get date

    """
    date_time = datetime.now()
    if granularity == "min":
        str_data_time = date_time.strftime("%Y%m%d-%H%M")
    elif granularity == "day":
        str_data_time = date_time.strftime("%Y%m%d")
    else:
        logging.error(f"Undefined timestamp granularity: {granularity}")

    return str_data_time


def set_seed(
    seed: int,
    num_gpu: int,
    is_eval: bool = False,
):
    """
    set random seed

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if num_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    if is_eval:
        torch.backends.cudnn.deterministic = True


def args2dict(args: ArgumentParser) -> dict[str, Any]:
    """
    convert passed arguments to dict

    """

    new_args = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        new_args[key] = value

    return new_args


def truncate_one_input(string: str, max_input_length: int, tokenizer) -> str:
    """
    char-based rough truncation, instead of tokenizer's function
    because one of tokenizers, flan-t5, cannot recover newline.
    ids = tokenizer(truncate=True)
    text = tokenizer.decode(ids) <= does not have newlines

    """

    input_str_len = len(string)
    input_ids_len = len(tokenizer(string).input_ids)  # buffer just in case
    ratio = max_input_length / input_ids_len
    if ratio < 1:
        boundary = int(input_str_len * (1 - ratio))
        string = string[boundary:]
    return string


def truncate(
    examples: list[Any], tokenizer, max_input_length: int, num_cpu: int = 4
) -> list[Any]:
    """ """
    inputs = [[x["input"], max_input_length, tokenizer] for x in examples]
    inputs_truncated = []
    with multiprocessing.Pool(processes=num_cpu) as pool:
        for result in pool.starmap(truncate_one_input, tqdm(inputs, total=len(inputs))):
            inputs_truncated.append(result)

    logging.info("Update input")
    for example, input_truncated in zip(examples, inputs_truncated):
        assert input_truncated in example["input"]
        example["input"] = input_truncated

    return examples


def _add_pad_token(tokenizer: PreTrainedTokenizer) -> tuple[PreTrainedTokenizer, int]:
    """
    Add pad token
    return: num_new_tokens

    Note:
    maybe better to add 63 more special tokens
    so that this is divisible by 64
    """

    logging.info("the tokenizer does not have a pad token, so add [PAD] as pad token")
    special_tokens_dict = {"pad_token": DEFAULT_PAD_TOKEN}
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    return tokenizer, num_new_tokens


def _update_post_processor(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """
    Update tokenizer's post_processor

    """

    logging.info("update tokenizer's post_processor")

    bos, bos_token_id = tokenizer.bos_token, tokenizer.bos_token_id
    eos, eos_token_id = tokenizer.eos_token, tokenizer.eos_token_id

    single = f"{(bos+':0 ') * True}$A:0{(' '+eos+':0') * False}"
    pair = (
        f"{(bos+':0 ') * True}$A:0{(' '+eos+':0') * False}"
        f"{(' '+bos+':1') * False} $B:1{(' '+eos+':1') * True}"
    )

    special_tokens = [(bos, bos_token_id), (eos, eos_token_id)]
    tokenizer._tokenizer.post_processor = processors.TemplateProcessing(
        single=single, pair=pair, special_tokens=special_tokens
    )

    return tokenizer


def load_tokenizer(model_id: str) -> tuple[PreTrainedTokenizer, int]:
    """
    load tokenizer

    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_new_tokens = 0

    if any([x in model_id for x in ["llama", "gpt2", "alpaca"]]):
        if tokenizer.pad_token is None:
            tokenizer, num_new_tokens = _add_pad_token(tokenizer)
        tokenizer = _update_post_processor(tokenizer)
        tokenizer.padding_side = "left"

    tokenizer.model_max_length = 4096
    tokenizer.truncation_side = "left"

    return tokenizer, num_new_tokens


def _map_precision_type(precision_type_original: str) -> tuple[str, str]:
    """
    map precision type to pytorch type

    """
    if precision_type_original == "float16":
        precision_type = torch.float16
        flag_load_in_8bit = False
    elif precision_type_original == "bfloat16":
        precision_type = torch.bfloat16
        flag_load_in_8bit = False
    elif precision_type_original == "int8":
        precision_type = torch.float16
        flag_load_in_8bit = True
    elif precision_type_original == "float32":
        precision_type = torch.float32
        flag_load_in_8bit = False
    else:
        logging.error(f"Undefined precision type: {precision_type_original}")
        return None, None

    return precision_type, flag_load_in_8bit


def _load_generation_config(
    model_id: str,
    precision_type: str,
    max_new_tokens: int,
    tokenizer: PreTrainedTokenizer,
    temperature: float,
    flag_sample: bool = False,
) -> tuple[GenerationConfig, bool]:
    """
    load generation config

    """

    if "falcon" in model_id:
        flag_trust_remote_code = True
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            max_new_tokens=max_new_tokens,
            do_sample=flag_sample,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
        )
        generation_config.update(**{"eos_token_id": tokenizer.eos_token_id})
        # generation_config.update(**{"temperature": temperature})
    elif "gpt" in model_id:
        flag_trust_remote_code = False
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            max_new_tokens=max_new_tokens,
            do_sample=flag_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    elif "Llama-2" in model_id:
        flag_trust_remote_code = False
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            max_new_tokens=max_new_tokens,
        )
        generation_config.update(
            **{
                "pad_token_id": tokenizer.pad_token_id,
                "temperature": temperature,
                "do_sample": flag_sample,
            }
        )
    elif any([x in model_id for x in ["llama", "alpaca"]]):
        flag_trust_remote_code = False
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            max_new_tokens=max_new_tokens,
            do_sample=flag_sample,
        )
        generation_config.update(
            **{
                "pad_token_id": tokenizer.pad_token_id,
                "temperature": temperature,
            }
        )
    else:
        if model_id in ["t5-3b", "t5-11b"]:
            model_id = "t5-large"  # use t5-large's config
        flag_trust_remote_code = False
        generation_config = GenerationConfig.from_pretrained(
            model_id,
            max_new_tokens=max_new_tokens,
            do_sample=flag_sample,
            # temperature=temperature,
        )
        # Note: this seems to be not working
        if "t5" in model_id:
            generation_config.update(**{"max_length": None})

    logging.info(f"generation config: {generation_config}")

    return generation_config, flag_trust_remote_code


def load_model(
    model_id: str,
    model_id_or_path: str,
    precision_type,
    flag_sample: bool,
    tokenizer: PreTrainedTokenizer,
    num_new_tokens: int,
    max_new_tokens: int,
    temperature: float,
    device_map: str = "auto",
) -> tuple[AutoModel, GenerationConfig]:
    """
    load model
    """

    flag_trust_remote_code = True if "falcon" in model_id else False

    precision_type, flag_load_in_8bit = _map_precision_type(
        precision_type,
    )
    logging.info(f"Load model weight with {precision_type} precision")

    generation_config, flag_trust_remote_code = _load_generation_config(
        model_id=model_id,
        precision_type=precision_type,
        max_new_tokens=max_new_tokens,
        tokenizer=tokenizer,
        temperature=temperature,
        flag_sample=flag_sample,
    )

    logging.info(f"Load model params from {model_id_or_path}")
    if model_id in SEQ2SEQ_MODELS:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id_or_path,
            torch_dtype=precision_type,
            trust_remote_code=flag_trust_remote_code,
            load_in_8bit=flag_load_in_8bit,
            device_map=device_map,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=precision_type,
            trust_remote_code=flag_trust_remote_code,
            load_in_8bit=flag_load_in_8bit,
            device_map=device_map,
        )
        if "Llama-2" in model_id:
            model.config.pad_token_id = tokenizer.pad_token_id
        if "gpt" in model_id:
            pass
            # ToDo: better to set sample=False here as well?

    if num_new_tokens > 0:
        logging.debug(f"Resize model's embedding to {len(tokenizer)}.")
        model.resize_token_embeddings(len(tokenizer))

    model.generation_config = generation_config

    return model, generation_config


def load_sequence_tagging_model(
    model_id: str,
    model_id_or_path: str,
    precision_type: str,
) -> PreTrainedModel:
    """
    load sequence tagging model

    Note: deberta-v3 does not support device_map="auto"

    """

    torch_dtype, flag_load_in_8bit = _map_precision_type(
        precision_type,
    )
    logging.info(f"Load model weight with {precision_type} precision")

    logging.info(f"Load model params from {model_id_or_path}")
    if model_id in T5_FAMILY:
        model = T5ForTokenClassification.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            load_in_8bit=flag_load_in_8bit,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            load_in_8bit=flag_load_in_8bit,
        )

    return model


def decode(input_encodings_batch, logits_batch, tokenizer) -> list[list[str]]:
    """ """

    prediction_batch = []

    input_ids_batch = input_encodings_batch.input_ids.cpu().numpy()
    prediction_labels_batch = torch.argmax(logits_batch, dim=-1).cpu().numpy()

    for batch_idx in range(len(prediction_labels_batch)):
        indices = [
            token_idx
            for token_idx in np.where(prediction_labels_batch[batch_idx] == 1)[0]
            if input_encodings_batch.token_to_sequence(batch_idx, token_idx) == 1
        ]

        if len(indices) > 0:
            word_ids = [[input_ids_batch[batch_idx][indices[0]]]]
            prev_token_idx = indices[0]
            for token_idx in indices[1:]:
                token_id = input_ids_batch[batch_idx][token_idx]
                if abs(token_idx - prev_token_idx) == 1:
                    word_ids[-1].append(token_id)
                else:
                    word_ids.append([token_id])
                prev_token_idx = token_idx

            prediction_batch.append(
                [tokenizer.decode(word_id).strip() for word_id in word_ids]
            )
        else:
            prediction_batch.append([])

    return [",".join(pred) for pred in prediction_batch]


def load_classification_model(
    model_id: str,
    model_id_or_path: str,
    dataset_name: str,
    num_labels: int,
    precision_type: str,
    device_map: str = "auto",
) -> PreTrainedModel:
    """
    load sequence classification model

    Note: deberta-v3 does not support device_map="auto"

    """

    torch_dtype, flag_load_in_8bit = _map_precision_type(
        precision_type,
    )
    logging.info(f"Load model weight with {precision_type} precision")

    logging.info(f"Load model params from {model_id_or_path}")
    if dataset_name == "temporal-nli":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            load_in_8bit=flag_load_in_8bit,
            # device_map=device_map,
            num_labels=num_labels,
        )
    elif dataset_name in ["matres", "tddiscourse"]:
        model = DebertaForSequenceClassificationEMarker.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            load_in_8bit=flag_load_in_8bit,
            # device_map=device_map,
            num_labels=num_labels,
        )
    else:
        logging.error(
            f"No classification model is defined for the dataset, {dataset_name}"
        )

    return model
