# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor. # noqa: E501
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import datasets


_CITATION = """\
@article{hasegawa-etal-2024-formulation,
    title={Formulation Comparison for Timeline Construction using LLMs},
    author={Hasegawa, Kimihiro and Kandukuri, Nikhil and Holm, Susan and Yamakawa, Yukari and Mitamura, Teruko},
    publisher = {arXiv},
    year={2024},
    url={https://arxiv.org/abs/2403.00990},
}
"""  # noqa: E501

_DESCRIPTION = """\
TimeSET is an evaluation dataset for timeline construction from text.
"""

_HOMEPAGE = "https://github.com/kimihiroh/timeset"

_LICENSE = "CC BY 4.0"

_URL = "https://huggingface.co/datasets/kimihiroh/timeset/raw/main/"
_URLS = {
    "dev": _URL + "full_dev.json",
    "test": _URL + "full_test.json",
}

_VERSION = "1.0.0"


class TimeSETConfig(datasets.BuilderConfig):
    def __init__(self, features, **kwargs):
        super().__init__(version=datasets.Version(_VERSION), **kwargs)
        self.features = datasets.Features(features)


class TimeSET(datasets.GeneratorBasedBuilder):
    """TimeSET for automatic timeline construction task"""

    VERSION = datasets.Version(_VERSION)
    _URLS = _URLS

    BUILDER_CONFIGS = [
        TimeSETConfig(
            name="nli",
            description="NLI formulation",
            features={
                "context": datasets.Value("string"),
                "id_arg1": datasets.Value("string"),
                "arg1": {
                    "mention": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                },
                "keyword": datasets.Value("string"),
                "id_arg2": datasets.Value("string"),
                "arg2": {
                    "mention": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                },
                "label": datasets.Value("string"),
                "filename": datasets.Value("string"),
            },
        ),
        TimeSETConfig(
            name="pairwise",
            description="Pairwise formulation",
            features={
                "context": datasets.Value("string"),
                "id_arg1": datasets.Value("string"),
                "arg1": {
                    "mention": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                },
                "keyword": datasets.Value("string"),
                "id_arg2": datasets.Value("string"),
                "arg2": {
                    "mention": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                },
                "relation": datasets.Value("string"),
                "filename": datasets.Value("string"),
            },
        ),
        TimeSETConfig(
            name="mrc",
            description="MRC formulation",
            features={
                "context": datasets.Value("string"),
                "target_id": datasets.Value("string"),
                "target": {
                    "mention": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                },
                "answers": datasets.Sequence(
                    {
                        "mention": datasets.Value("string"),
                        "start": datasets.Value("int32"),
                        "end": datasets.Value("int32"),
                    }
                ),
                "relation": datasets.Value("string"),
                "filename": datasets.Value("string"),
            },
        ),
        TimeSETConfig(
            name="timeline",
            description="Timeline formulation",
            features={
                "context": datasets.Value("string"),
                "timeline": datasets.Sequence(
                    datasets.Sequence(
                        {
                            "mention": datasets.Value("string"),
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                        }
                    )
                ),
                "filename": datasets.Value("string"),
            },
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION + " " + self.config.description,
            features=self.config.features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = self._URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": downloaded_files["dev"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": downloaded_files["test"],
                },
            ),
        ]

    def _generate_examples_nli(self, filepath):
        with open(filepath, "r") as f:
            annotations = json.load(f)
            idx = 0
            for annotation in annotations:
                for raw_example in annotation["examples"]["nli"]:
                    ev1 = annotation["annotation"]["events"][
                        str(raw_example["id_arg1"])
                    ]
                    ev2 = annotation["annotation"]["events"][
                        str(raw_example["id_arg2"])
                    ]
                    yield idx, {
                        "context": annotation["text"],
                        "id_arg1": raw_example["id_arg1"],
                        "arg1": {
                            "mention": ev1["mention"],
                            "start": ev1["start"],
                            "end": ev1["end"],
                        },
                        "keyword": raw_example["keyword"],
                        "id_arg2": raw_example["id_arg2"],
                        "arg2": {
                            "mention": ev2["mention"],
                            "start": ev2["start"],
                            "end": ev2["end"],
                        },
                        "label": raw_example["label"],
                        "filename": annotation["filename"],
                    }
                    idx += 1

    def _generate_examples_pairwise(self, filepath):
        with open(filepath, "r") as f:
            annotations = json.load(f)
            idx = 0
            for annotation in annotations:
                for raw_example in annotation["examples"]["pairwise"]:
                    ev1 = annotation["annotation"]["events"][
                        str(raw_example["id_arg1"])
                    ]
                    ev2 = annotation["annotation"]["events"][
                        str(raw_example["id_arg2"])
                    ]
                    yield idx, {
                        "context": annotation["text"],
                        "id_arg1": raw_example["id_arg1"],
                        "arg1": {
                            "mention": ev1["mention"],
                            "start": ev1["start"],
                            "end": ev1["end"],
                        },
                        "id_arg2": raw_example["id_arg2"],
                        "arg2": {
                            "mention": ev2["mention"],
                            "start": ev2["start"],
                            "end": ev2["end"],
                        },
                        "relation": raw_example["relation"],
                        "filename": annotation["filename"],
                    }
                    idx += 1

    def _generate_examples_mrc(self, filepath):
        with open(filepath, "r") as f:
            annotations = json.load(f)
            idx = 0
            for annotation in annotations:
                for raw_example in annotation["examples"]["mrc"]:
                    ev_target = annotation["annotation"]["events"][
                        str(raw_example["target"])
                    ]
                    answers = [
                        annotation["annotation"]["events"][idx]
                        for idx in raw_example["answers"]
                    ]
                    yield idx, {
                        "context": annotation["text"],
                        "target_id": raw_example["target"],
                        "target": {
                            "mention": ev_target["mention"],
                            "start": ev_target["start"],
                            "end": ev_target["end"],
                        },
                        "answers": [
                            {
                                "mention": answer["mention"],
                                "start": answer["start"],
                                "end": answer["end"],
                            }
                            for answer in answers
                        ],
                        "relation": raw_example["relation"],
                        "filename": annotation["filename"],
                    }
                    idx += 1

    def _generate_examples_timeline(self, filepath):
        with open(filepath, "r") as f:
            annotations = json.load(f)
            idx = 0
            for annotation in annotations:
                for raw_example in annotation["examples"]["timeline"]:
                    events = annotation["annotation"]["events"]
                    yield idx, {
                        "context": annotation["text"],
                        "timeline": [
                            [
                                {
                                    "mention": events[eid]["mention"],
                                    "start": events[eid]["start"],
                                    "end": events[eid]["end"],
                                }
                                for eid in layer
                            ]
                            for layer in raw_example
                        ],
                        "filename": annotation["filename"],
                    }
                    idx += 1

    def _generate_examples(self, filepath):
        match self.config.name:
            case "nli":
                yield from self._generate_examples_nli(filepath)
            case "pairwise":
                yield from self._generate_examples_pairwise(filepath)
            case "mrc":
                yield from self._generate_examples_mrc(filepath)
            case "timeline":
                yield from self._generate_examples_timeline(filepath)
            case _:
                print(f"Undefined formulation: {self.config.name}")
