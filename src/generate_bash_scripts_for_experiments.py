"""

"""

from argparse import ArgumentParser

from pathlib import Path

model2param = {
    "t5-large": {"num_gpu": 1, "batch_size": 4, "code": "inference_generation"},
    "t5-3b": {"num_gpu": 1, "batch_size": 4, "code": "inference_generation"},
    "t5-11b": {"num_gpu": 2, "batch_size": 2, "code": "inference_generation"},
    "google/flan-t5-large": {
        "num_gpu": 1,
        "batch_size": 4,
        "code": "inference_generation",
    },
    "google/flan-t5-xl": {
        "num_gpu": 1,
        "batch_size": 4,
        "code": "inference_generation",
    },
    "google/flan-t5-xxl": {
        "num_gpu": 2,
        "batch_size": 2,
        "code": "inference_generation",
    },
    "meta-llama/Llama-2-7b-hf": {
        "num_gpu": 1,
        "batch_size": 8,
        "code": "inference_generation_vllm",
    },
    "meta-llama/Llama-2-13b-hf": {
        "num_gpu": 2,
        "batch_size": 4,
        "code": "inference_generation_vllm",
    },
    "meta-llama/Llama-2-70b-hf": {
        "num_gpu": 4,
        "batch_size": 4,
        "code": "inference_generation_vllm",
    },
    "meta-llama/Llama-2-7b-chat-hf": {
        "num_gpu": 1,
        "batch_size": 8,
        "code": "inference_generation_vllm",
    },
    "meta-llama/Llama-2-13b-chat-hf": {
        "num_gpu": 2,
        "batch_size": 4,
        "code": "inference_generation_vllm",
    },
    "meta-llama/Llama-2-70b-chat-hf": {
        "num_gpu": 4,
        "batch_size": 4,
        "code": "inference_generation_vllm",
    },
    "codellama/CodeLlama-7b-hf": {
        "num_gpu": 1,
        "batch_size": 8,
        "code": "inference_generation_vllm",
    },
}


def generate_scripts_for_benchmark(dirpath_output):
    with open("src/template_script/benchmark_inference_few-shot.txt", "r") as f:
        template = f.read()
    params_benchmark = [
        {
            "dataset_name": "torque",
            "filepath_test": "data/preprocessed/torque/dev.json",
            "filepath_dev": "data/preprocessed/torque/train.json",
            "max_new_tokens": 64,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "matres",
            "filepath_test": "data/preprocessed/matres/test.json",
            "filepath_dev": "data/preprocessed/matres/dev.json",
            "max_new_tokens": 64,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "tddiscourse",
            "filepath_test": "data/preprocessed/tddiscourse/test.json",
            "filepath_dev": "data/preprocessed/tddiscourse/dev.json",
            "max_new_tokens": 64,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "temporal-nli",
            "filepath_test": "data/preprocessed/temporal_nli/test.json",
            "filepath_dev": "data/preprocessed/temporal_nli/dev.json",
            "max_new_tokens": 64,
            "precision_type": "bfloat16",
        },
    ]

    model_ids = [
        "t5-3b",
        # "t5-11b",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        # "meta-llama/Llama-2-13b-chat-hf",
        # "meta-llama/Llama-2-70b-chat-hf",
    ]

    for param in params_benchmark:
        for model_id in model_ids:
            num_gpu = model2param[model_id]["num_gpu"]
            batch_size = model2param[model_id]["batch_size"]
            code = model2param[model_id]["code"]
            script = (
                template.replace("{dataset_name}", param["dataset_name"])
                .replace("{filepath_test}", param["filepath_test"])
                .replace("{filepath_dev}", param["filepath_dev"])
                .replace("{max_new_tokens}", str(param["max_new_tokens"]))
                .replace("{batch_size}", str(batch_size))
                .replace("{code}", code)
                .replace("{precision_type}", param["precision_type"])
                .replace("{num_gpu}", str(num_gpu))
                .replace("{num_cpu}", "4")
                .replace("{model_id}", model_id)
                .replace(
                    "{gpu_ids}",
                    ",".join([str(x) for x in range(num_gpu)]) if num_gpu > 0 else "0",
                )
                .replace("{model_id_short}", Path(model_id).name)
            )
            filename = (
                f'inference_few-shot_{Path(model_id).name}_{param["precision_type"]}.sh'
            )
            dirpath_output = args.dirpath_output / "benchmark" / param["dataset_name"]

            if not dirpath_output.exists():
                dirpath_output.mkdir(parents=True)
            with open(dirpath_output / filename, "w") as f:
                f.write(script)

    return None


# ========== #


def generate_scripts_for_benchmark_finetune(dirpath: Path) -> None:
    with open("src/template_script/benchmark_finetune.txt", "r") as f:
        TEMPLATE = f.read()

    params = [
        {
            "dataset_name": "torque",
            "filepath_train": "data/preprocessed/torque/train_train.json",
            "filepath_dev": "data/preprocessed/torque/train_dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_sequence_tagging", "bfloat16"),
            ],
        },
        {
            "dataset_name": "matres",
            "filepath_train": "data/preprocessed/matres/train.json",
            "filepath_dev": "data/preprocessed/matres/dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_classification", "bfloat16"),
            ],
        },
        {
            "dataset_name": "tddiscourse",
            "filepath_train": "data/preprocessed/tddiscourse/train.json",
            "filepath_dev": "data/preprocessed/tddiscourse/dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_classification", "bfloat16"),
            ],
        },
        {
            "dataset_name": "temporal-nli",
            "filepath_train": "data/preprocessed/temporal_nli/train.json",
            "filepath_dev": "data/preprocessed/temporal_nli/dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_classification", "bfloat16"),
            ],
        },
    ]

    for param in params:
        for model_id, code, precision in param["models"]:
            script = (
                TEMPLATE.replace("{dataset_name}", param["dataset_name"])
                .replace("{filepath_data_train}", param["filepath_train"])
                .replace("{filepath_data_dev}", param["filepath_dev"])
                .replace("{model_id}", model_id)
                .replace("{code}", code)
                .replace("{precision_type}", precision)
            )
            filename = f"ft_{Path(model_id).name}_{precision}.sh"
            dirpath_output = args.dirpath_output / "benchmark" / param["dataset_name"]

            if not dirpath_output.exists():
                dirpath_output.mkdir(parents=True)
            with open(dirpath_output / filename, "w") as f:
                f.write(script)


# ========== #


def generate_scripts_for_benchmark_peft(dirpath: Path) -> None:
    with open("src/template_script/benchmark_peft.txt", "r") as f:
        TEMPLATE = f.read()

    params = [
        {
            "dataset_name": "torque",
            "filepath_train": "data/preprocessed/torque/train_train.json",
            "filepath_dev": "data/preprocessed/torque/train_dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("google/flan-t5-xl", "finetune_generation", "bfloat16"),
                ("meta-llama/Llama-2-7b-hf", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_sequence_tagging", "bfloat16"),
            ],
        },
        {
            "dataset_name": "matres",
            "filepath_train": "data/preprocessed/matres/train.json",
            "filepath_dev": "data/preprocessed/matres/dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("google/flan-t5-xl", "finetune_generation", "bfloat16"),
                ("meta-llama/Llama-2-7b-hf", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_classification", "bfloat16"),
            ],
        },
        {
            "dataset_name": "tddiscourse",
            "filepath_train": "data/preprocessed/tddiscourse/train.json",
            "filepath_dev": "data/preprocessed/tddiscourse/dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("google/flan-t5-xl", "finetune_generation", "bfloat16"),
                ("meta-llama/Llama-2-7b-hf", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_classification", "bfloat16"),
            ],
        },
        {
            "dataset_name": "temporal-nli",
            "filepath_train": "data/preprocessed/temporal_nli/train.json",
            "filepath_dev": "data/preprocessed/temporal_nli/dev.json",
            "models": [
                ("google/flan-t5-large", "finetune_generation", "bfloat16"),
                ("google/flan-t5-xl", "finetune_generation", "bfloat16"),
                ("meta-llama/Llama-2-7b-hf", "finetune_generation", "bfloat16"),
                ("microsoft/deberta-v3-large", "finetune_classification", "bfloat16"),
            ],
        },
    ]

    for param in params:
        for model_id, code, precision in param["models"]:
            script = (
                TEMPLATE.replace("{dataset_name}", param["dataset_name"])
                .replace("{filepath_data_train}", param["filepath_train"])
                .replace("{filepath_data_dev}", param["filepath_dev"])
                .replace("{model_id}", model_id)
                .replace("{code}", code)
                .replace("{precision_type}", precision)
            )
            filename = f"peft_{Path(model_id).name}_{precision}.sh"
            dirpath_output = args.dirpath_output / "benchmark" / param["dataset_name"]

            if not dirpath_output.exists():
                dirpath_output.mkdir(parents=True)
            with open(dirpath_output / filename, "w") as f:
                f.write(script)


# ========== #


def generate_scripts_for_benchmark_inference(dirpath_output):
    with open("src/template_script/benchmark_inference_tuned.txt", "r") as f:
        TEMPLATE = f.read()
    params = [
        {
            "dataset_name": "torque",
            "filepath_test": "data/preprocessed/torque/dev.json",
            "peft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha64_drop0.1",
                ),
                (
                    "google/flan-t5-xl",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha64_drop0.1",
                ),
                (
                    "meta-llama/Llama-2-7b-hf",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim16_alpha64_drop0.1",
                ),
                (
                    "microsoft/deberta-v3-large",
                    "inference_sequence_tagging",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim16_alpha64_drop0.1",
                ),
            ],
            "ft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs16_lr0.0001_dimNone_alphaNone_dropNone",
                ),
                (
                    "microsoft/deberta-v3-large",
                    "inference_sequence_tagging",
                    "bfloat16",
                    "seed7_bs32_lr0.0001_dimNone_alphaNone_dropNone",
                ),
            ],
        },
        {
            "dataset_name": "matres",
            "filepath_test": "data/preprocessed/matres/test.json",
            "peft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim16_alpha64_drop0.1",
                ),
                (
                    "google/flan-t5-xl",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha64_drop0.1",
                ),
                (
                    "meta-llama/Llama-2-7b-hf",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim16_alpha64_drop0.1",
                ),
                (
                    "microsoft/deberta-v3-large",
                    "inference_classification",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha64_drop0.1",
                ),
            ],
            "ft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs16_lr0.0001_dimNone_alphaNone_dropNone",
                ),
                (
                    "microsoft/deberta-v3-large",
                    "inference_classification",
                    "bfloat16",
                    "seed7_bs32_lr0.0001_dimNone_alphaNone_dropNone",
                ),
            ],
        },
        {
            "dataset_name": "tddiscourse",
            "filepath_test": "data/preprocessed/tddiscourse/test.json",
            "peft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha16_drop0.1",
                ),
                (
                    "google/flan-t5-xl",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha64_drop0.1",
                ),
                ("meta-llama/Llama-2-7b-hf", "inference_generation", "bfloat16", None),
                (
                    "microsoft/deberta-v3-large",
                    "inference_classification",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha16_drop0.1",
                ),
            ],
            "ft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs16_lr0.0001_dimNone_alphaNone_dropNone",
                ),
                (
                    "microsoft/deberta-v3-large",
                    "inference_classification",
                    "bfloat16",
                    "seed7_bs16_lr0.0001_dimNone_alphaNone_dropNone",
                ),
            ],
        },
        {
            "dataset_name": "temporal-nli",
            "filepath_test": "data/preprocessed/temporal_nli/test.json",
            "peft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim16_alpha16_drop0.1",
                ),
                (
                    "google/flan-t5-xl",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha64_drop0.1",
                ),
                (
                    "meta-llama/Llama-2-7b-hf",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim64_alpha16_drop0.1",
                ),
                (
                    "microsoft/deberta-v3-large",
                    "inference_classification",
                    "bfloat16",
                    "seed7_bs8_lr0.0001_dim16_alpha64_drop0.1",
                ),
            ],
            "ft": [
                (
                    "google/flan-t5-large",
                    "inference_generation",
                    "bfloat16",
                    "seed7_bs16_lr0.0001_dimNone_alphaNone_dropNone",
                ),
                (
                    "microsoft/deberta-v3-large",
                    "inference_classification",
                    "bfloat16",
                    "seed7_bs16_lr0.0001_dimNone_alphaNone_dropNone",
                ),
            ],
        },
    ]
    for param in params:
        for tuned_type in ["peft", "ft"]:
            for model_id, code, precision, dirname_model in param[tuned_type]:
                if "deberta" in model_id:
                    if param["dataset_name"] == "torque":
                        folder_name = (
                            f"{Path(model_id).name}_{precision}_{tuned_type}"
                            "_sequence_tagging"
                        )
                    else:
                        folder_name = (
                            f"{Path(model_id).name}_{precision}_{tuned_type}"
                            "_sequence_classification"
                        )
                else:
                    folder_name = (
                        f"{Path(model_id).name}_{precision}_{tuned_type}_generation"
                    )

                dirpath_model = (
                    "/usr1/datasets/kimihiro/llm-for-event-temporal-ordering/models/"
                    f'{param["dataset_name"]}/{folder_name}/{dirname_model}'
                )
                script = (
                    TEMPLATE.replace("{dataset_name}", param["dataset_name"])
                    .replace("{filepath_data_test}", param["filepath_test"])
                    .replace("{inference_type}", tuned_type)
                    .replace(
                        "{dirpath_model}",
                        "None" if tuned_type == "peft" else dirpath_model,
                    )
                    .replace(
                        "{peft_model_path}",
                        "None" if tuned_type == "ft" else dirpath_model,
                    )
                    .replace("{model_id}", model_id)
                    .replace("{code}", code)
                    .replace("{precision_type}", precision)
                )
                filename = (
                    f"inference_{tuned_type}_{Path(model_id).name}_{precision}.sh"
                )
                dirpath_output = (
                    args.dirpath_output / "benchmark" / param["dataset_name"]
                )

                if not dirpath_output.exists():
                    dirpath_output.mkdir(parents=True)
                with open(dirpath_output / filename, "w") as f:
                    f.write(script)


# ========== #


def generate_scripts_for_formulation_comprison(dirpath_output):
    with open("src/template_script/comparison_inference_few-shot.txt", "r") as f:
        TEMPLATE = f.read()
    with open(
        "src/template_script/comparison_inference_few-shot_marker_repr.txt", "r"
    ) as f:
        TEMPLATE_marker_repr = f.read()
    params_ctf = [
        {
            "dataset_name": "ctf-pairwise",
            "max_new_tokens": 512,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "ctf-nli",
            "max_new_tokens": 512,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "ctf-mrc",
            "max_new_tokens": 512,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "ctf-mrc-cot",
            "max_new_tokens": 512,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "ctf-timeline",
            "max_new_tokens": 512,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "ctf-timeline-cot",
            "max_new_tokens": 512,
            "precision_type": "bfloat16",
        },
        {
            "dataset_name": "ctf-timeline-code",
            "max_new_tokens": 512,
            "precision_type": "bfloat16",
        },
    ]
    model_ids = [
        # "t5-large",
        "t5-3b",
        # "t5-11b",
        "google/flan-t5-large",
        "google/flan-t5-xl",
        "google/flan-t5-xxl",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        # "meta-llama/Llama-2-13b-chat-hf",
        # "meta-llama/Llama-2-70b-chat-hf",
        "codellama/CodeLlama-7b-hf",
    ]

    for param in params_ctf:
        for model_id in model_ids:
            num_gpu = model2param[model_id]["num_gpu"]
            batch_size = model2param[model_id]["batch_size"]
            code = model2param[model_id]["code"]
            gpu_ids = ",".join([str(x) for x in range(num_gpu)]) if num_gpu > 0 else "0"

            if "Llama-2-7b-hf" in model_id and "timeline" in param["dataset_name"]:
                template = TEMPLATE_marker_repr
            else:
                template = TEMPLATE

            script = (
                template.replace("{dataset_name}", param["dataset_name"])
                .replace("{max_new_tokens}", str(param["max_new_tokens"]))
                .replace("{batch_size}", str(batch_size))
                .replace("{code}", code)
                .replace("{precision_type}", param["precision_type"])
                .replace("{num_gpu}", str(num_gpu))
                .replace("{model_id}", model_id)
                .replace("{gpu_ids}", gpu_ids)
                .replace("{num_cpu}", "4")
                .replace("{model_id_short}", Path(model_id).name)
            )
            filename = (
                f'inference_few-shot_{Path(model_id).name}_{param["precision_type"]}.sh'
            )
            formulation = param["dataset_name"].replace("ctf-", "")
            dirpath_output = args.dirpath_output / "comparison" / formulation
            if not dirpath_output.exists():
                dirpath_output.mkdir(parents=True)
            with open(dirpath_output / filename, "w") as f:
                f.write(script)

    return None


if __name__ == "__main__":
    parser = ArgumentParser(description="generate scripts for experiments")
    parser.add_argument("--dirpath_output", type=Path, help="dirpath for output")
    parser.add_argument(
        "--task", type=str, choices=["comparison", "benchmark"], help="task name"
    )
    parser.add_argument("--dirpath_log", type=Path, help="dirpath for log")
    args = parser.parse_args()

    if args.task == "comparison":
        generate_scripts_for_formulation_comprison(args.dirpath_output)
    elif args.task == "benchmark":
        generate_scripts_for_benchmark(args.dirpath_output)
        generate_scripts_for_benchmark_finetune(args.dirpath_output)
        generate_scripts_for_benchmark_peft(args.dirpath_output)
        generate_scripts_for_benchmark_inference(args.dirpath_output)
    else:
        print("Error")
