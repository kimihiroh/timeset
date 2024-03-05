"""

"""

from dataclasses import dataclass
import logging
from typing import Any
from utils_template import (
    add_marker_to_context_pair,
    add_marker_to_context_all,
    get_representation,
    get_marker_single,
    get_event_list,
    get_event_list_code,
    get_cot,
    get_cot_timeline,
    get_target_events,
    get_edges,
    get_timeline_code,
    get_timeline,
)

MATRES_RELATIONS = ["AFTER", "BEFORE", "EQUAL", "VAGUE"]
TDDISCOURSE_RELATIONS = ["AFTER", "BEFORE", "SIMULTANEOUS", "INCLUDE", "INCLUDED"]
# TRACIE_LABELS = ["negative", "positive"]
TEMPORALNLI_LABELS = ["negative", "positive"]
CTF_PAIRWISE_RELATIONS = {
    "AFTER": 0,
    "BEFORE": 1,
    "VAGUE": 2,
    "COEX": 2,
}
CTF_NLI_LABELS = ["positive", "negative"]
CTF_RELATIONS = ["AFTER", "BEFORE", "COEX"]


# BASE
@dataclass
class BASE_TEMPLATE:
    name: str
    inputs_prefix: str
    inputs: str
    x_y_delimiter: str
    targets_prefix: str
    targets: str
    example_separator: str

    def get_input(self):
        return self.inputs_prefix + self.inputs + self.x_y_delimiter

    def get_target(self):
        return self.targets_prefix + self.targets + self.example_separator

    def get_template_w_target(self):
        return self.get_input() + self.get_target()

    def get_template_wo_target(self):
        return self.get_input() + self.targets_prefix


# MATRES
@dataclass
class MATRES_TEMPLATE(BASE_TEMPLATE):
    choices = MATRES_RELATIONS
    event_representation: str  # eid or mention

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str = None,
        marker: str = None,
    ):
        template = self.get_template_w_target()
        context_with_marker = add_marker_to_context_pair(example, "eid")
        choices_str = ", ".join(self.choices)
        if self.event_representation == "eid":
            arg1_str, arg2_str = "[e1]", "[e2]"
        elif self.event_representation == "mention":
            arg1_str = f"[e1]{example['arg1']['mention']}[/e1]"
            arg2_str = f"[e2]{example['arg2']['mention']}[/e2]"
        else:
            logging.error(
                f"Undefined event representation: {self.event_representation}"
            )
        one_demonstration_in_template = (
            template.replace("{context}", context_with_marker)
            .replace("{arg1}", arg1_str)
            .replace("{arg2}", arg2_str)
            .replace("{choices}", choices_str)
            .replace("{target}", example["relation"])
        )
        return one_demonstration_in_template

    def get_prompt(
        self,
        example: dict[str, Any],
        representation: str = None,
        marker: str = None,
    ):
        template = self.get_template_wo_target()
        context_with_marker = add_marker_to_context_pair(example, "eid")
        choices_str = ", ".join(self.choices)
        if self.event_representation == "eid":
            arg1_str, arg2_str = "[e1]", "[e2]"
        elif self.event_representation == "mention":
            arg1_str = f"[e1]{example['arg1']['mention']}[/e1]"
            arg2_str = f"[e2]{example['arg2']['mention']}[/e2]"
        else:
            logging.error(
                f"Undefined event representation: {self.event_representation}"
            )
        one_prompt_in_template = (
            template.replace("{context}", context_with_marker)
            .replace("{arg1}", arg1_str)
            .replace("{arg2}", arg2_str)
            .replace("{choices}", choices_str)
        )
        return one_prompt_in_template


# TDDISCOURSE
@dataclass
class TDDISCOURSE_TEMPLATE(MATRES_TEMPLATE):
    choices = TDDISCOURSE_RELATIONS


# TORQUE
@dataclass
class TORQUE_TEMPLATE(BASE_TEMPLATE):
    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str = None,
        marker: str = None,
    ):
        template = self.get_template_w_target()
        target = ", ".join([x["mention"] for x in example["answers"]])
        if not target:
            target = "None"
        one_demonstration_in_template = (
            template.replace("{context}", example["context"])
            .replace("{question}", example["question"])
            .replace("{target}", target)
        )
        return one_demonstration_in_template

    def get_prompt(
        self,
        example: dict[str, Any],
        representation: str = None,
        marker: str = None,
    ):
        template = self.get_template_wo_target()
        one_prompt_in_template = template.replace(
            "{context}", example["context"]
        ).replace("{question}", example["question"])
        return one_prompt_in_template


# TemporalNLI
@dataclass
class TEMPORALNLI_TEMPLATE(BASE_TEMPLATE):
    target2original: dict[str, str]
    original2target: dict[str, str]

    def map_target2original(self, label):
        if label in self.target2original:
            return self.target2original[label]
        else:
            logging.debug(f"{label} not in {self.target2original.keys()}")
            return None

    def map_original2target(self, label):
        return self.original2target[label]

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str = None,
        marker: str = None,
    ):
        template = self.get_template_w_target()
        one_demonstration_in_template = (
            template.replace("{premise}", example["context"])
            .replace("{hypothesis}", example["statement"])
            .replace("{target}", self.map_original2target(example["label"]))
        )
        return one_demonstration_in_template

    def get_prompt(
        self,
        example: dict[str, Any],
        representation: str = None,
        marker: str = None,
    ):
        template = self.get_template_wo_target()
        one_prompt_in_template = template.replace(
            "{premise}", example["context"]
        ).replace("{hypothesis}", example["statement"])
        return one_prompt_in_template


# ======= classes for formulation comparison ======= #
# NLI
@dataclass
class NLI_TEMPLATE(BASE_TEMPLATE):
    target2original: dict[str, str]
    original2target: dict[str, str]
    hypothesis_mapping: dict[str, str]

    def map_target2original(self, label):
        if label in self.target2original:
            return self.target2original[label]
        else:
            logging.debug(f"{label} not in {self.target2original.keys()}")
            return None

    def map_original2target(self, label):
        return self.original2target[label]

    def map_hypothesis(self, relation):
        return self.hypothesis_mapping[relation]

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str,
        marker: str,
    ):
        template = self.get_template_w_target()
        premise = add_marker_to_context_pair(example, marker)
        hypothesis = (
            self.map_hypothesis(example["keyword"])
            .replace(
                "{arg1}",
                get_representation(
                    event=example["arg1"],
                    eid="1",
                    representation=representation,
                    marker=marker,
                ),
            )
            .replace(
                "{arg2}",
                get_representation(
                    event=example["arg2"],
                    eid="2",
                    representation=representation,
                    marker=marker,
                ),
            )
        )
        one_demonstration_in_template = (
            template.replace("{premise}", premise)
            .replace("{hypothesis}", hypothesis)
            .replace("{target}", self.map_original2target(example["label"]))
        )
        return one_demonstration_in_template

    def get_prompt(self, example: dict[str, Any], representation: str, marker: str):
        template = self.get_template_wo_target()
        premise = add_marker_to_context_pair(example, marker)
        hypothesis = (
            self.map_hypothesis(example["keyword"])
            .replace(
                "{arg1}",
                get_representation(
                    event=example["arg1"],
                    eid="1",
                    representation=representation,
                    marker=marker,
                ),
            )
            .replace(
                "{arg2}",
                get_representation(
                    event=example["arg2"],
                    eid="2",
                    representation=representation,
                    marker=marker,
                ),
            )
        )
        one_prompt_in_template = template.replace("{premise}", premise).replace(
            "{hypothesis}", hypothesis
        )
        return one_prompt_in_template


# PAIRWISE
@dataclass
class PAIRWISE_TEMPLATE(BASE_TEMPLATE):
    choices: dict[str, str]

    def _get_choices(self):
        output = ""
        for label, description in self.choices.items():
            output += f"- {label}: {description}\n"
        return output.rstrip()

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str,
        marker: str,
    ):
        template = self.get_template_w_target()
        context_with_marker = add_marker_to_context_pair(example, marker)
        choices = self._get_choices()
        if example["relation"] == "COEX":
            target = "COEX" if "COEX" in self.choices else "VAGUE"
        else:
            target = example["relation"]
        one_demonstration_in_template = (
            template.replace("{context}", context_with_marker)
            .replace(
                "{arg1}",
                get_representation(
                    event=example["arg1"],
                    eid="1",
                    representation=representation,
                    marker=marker,
                ),
            )
            .replace(
                "{arg2}",
                get_representation(
                    event=example["arg2"],
                    eid="2",
                    representation=representation,
                    marker=marker,
                ),
            )
            .replace("{choices}", choices)
            .replace("{target}", target)
        )
        return one_demonstration_in_template

    def get_prompt(self, example: dict[str, Any], representation: str, marker: str):
        template = self.get_template_wo_target()
        context_with_marker = add_marker_to_context_pair(example, marker)
        choices = self._get_choices()
        one_prompt_in_template = (
            template.replace("{context}", context_with_marker)
            .replace(
                "{arg1}",
                get_representation(
                    event=example["arg1"],
                    eid="1",
                    representation=representation,
                    marker=marker,
                ),
            )
            .replace(
                "{arg2}",
                get_representation(
                    event=example["arg2"],
                    eid="2",
                    representation=representation,
                    marker=marker,
                ),
            )
            .replace("{choices}", choices)
        )
        return one_prompt_in_template


# MRC
@dataclass
class MRC_TEMPLATE(BASE_TEMPLATE):
    question_mapping: dict[str, str]
    flag_list_events: bool
    flag_marker: bool
    flag_cot: bool = False
    cot_mapping: dict[str, str] = None

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str,
        marker: str,
    ):
        one_demonstration_in_template = self.get_template_w_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        if self.flag_marker:
            one_demonstration_in_template = one_demonstration_in_template.replace(
                "{marker}", get_marker_single(marker)
            )
        if self.flag_list_events:
            one_demonstration_in_template = one_demonstration_in_template.replace(
                "{events}", get_event_list(example, representation, marker)
            )
        if self.flag_cot:
            one_demonstration_in_template = one_demonstration_in_template.replace(
                "{cot}",
                get_cot(
                    example=example,
                    representation=representation,
                    marker=marker,
                    cot_mapping=self.cot_mapping,
                ),
            )
        question_str = self.question_mapping[example["relation"]].replace(
            "{event}",
            get_representation(
                event=example["target"],
                eid=example["target_id"],
                representation=representation,
                marker=marker,
            ),
        )
        target_str = get_target_events(
            events=example["answers"],
            representation=representation,
            marker=marker,
            flag_list=self.flag_list_events,
        )
        one_demonstration_in_template = (
            one_demonstration_in_template.replace("{context}", context_with_marker)
            .replace("{question}", question_str)
            .replace("{target}", target_str)
        )

        return one_demonstration_in_template

    def get_prompt(self, example: dict[str, Any], representation: str, marker: str):
        one_prompt_in_template = self.get_template_wo_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        if self.flag_marker:
            one_prompt_in_template = one_prompt_in_template.replace(
                "{marker}", get_marker_single(marker)
            )
        if self.flag_list_events:
            one_prompt_in_template = one_prompt_in_template.replace(
                "{events}", get_event_list(example, representation, marker)
            )
        question_str = self.question_mapping[example["relation"]].replace(
            "{event}",
            get_representation(
                event=example["target"],
                eid=example["target_id"],
                representation=representation,
                marker=marker,
            ),
        )
        one_prompt_in_template = one_prompt_in_template.replace(
            "{context}", context_with_marker
        ).replace("{question}", question_str)

        return one_prompt_in_template


# TIMELINE
@dataclass
class TIMELINE_TEMPLATE(BASE_TEMPLATE):
    flag_marker: bool = False
    flag_list_events: bool = False
    flag_cot: bool = False
    cot_mapping: dict[str, str] = None

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str,
        marker: str,
    ):
        one_demonstration_in_template = self.get_template_w_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        if self.flag_marker:
            one_demonstration_in_template = one_demonstration_in_template.replace(
                "{marker}", get_marker_single(marker)
            )
        if self.flag_list_events:
            one_demonstration_in_template = one_demonstration_in_template.replace(
                "{events}", get_event_list(example, representation, marker)
            )
        if self.flag_cot:
            one_demonstration_in_template = one_demonstration_in_template.replace(
                "{cot}",
                get_cot_timeline(
                    example=example,
                    representation=representation,
                    marker=marker,
                    cot_mapping=self.cot_mapping,
                ),
            )
        one_demonstration_in_template = one_demonstration_in_template.replace(
            "{context}", context_with_marker
        ).replace(
            "{target}",
            get_timeline(
                timeline=example["timeline"],
                representation=representation,
                marker=marker,
                flag_list=self.flag_list_events,
            ),
        )
        return one_demonstration_in_template

    def get_prompt(self, example: dict[str, Any], representation: str, marker: str):
        one_prompt_in_template = self.get_template_wo_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        if self.flag_marker:
            one_prompt_in_template = one_prompt_in_template.replace(
                "{marker}", get_marker_single(marker)
            )
        if self.flag_list_events:
            one_prompt_in_template = one_prompt_in_template.replace(
                "{events}", get_event_list(example, representation, marker)
            )
        one_prompt_in_template = one_prompt_in_template.replace(
            "{context}", context_with_marker
        )

        return one_prompt_in_template


@dataclass
class TIMELINE_CODE_GRAPH_TEMPLATE(BASE_TEMPLATE):
    target_template: str
    target_event_template: str
    event_template: str

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str,
        marker: str,
    ):
        one_demonstration_in_template = self.get_template_w_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        events = get_event_list_code(
            example=example,
            representation=representation,
            event_template=self.event_template,
            marker=marker,
        )
        target = get_edges(
            example=example,
            target_template=self.target_template,
            target_event_template=self.target_event_template,
        )
        one_demonstration_in_template = (
            one_demonstration_in_template.replace("{context}", context_with_marker)
            .replace("{events}", events)
            .replace("{target}", target)
        )
        return one_demonstration_in_template

    def get_prompt(self, example: dict[str, Any], representation: str, marker: str):
        one_prompt_in_template = self.get_template_wo_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        events = get_event_list_code(
            example=example,
            representation=representation,
            event_template=self.event_template,
            marker=marker,
        )
        one_prompt_in_template = one_prompt_in_template.replace(
            "{context}", context_with_marker
        ).replace("{events}", events)
        return one_prompt_in_template


@dataclass
class TIMELINE_CODE_TIMELINE_TEMPLATE(BASE_TEMPLATE):
    target_template: str
    target_event_template: str
    event_template: str

    def get_demonstration(
        self,
        example: dict[str, Any],
        representation: str,
        marker: str,
    ):
        one_demonstration_in_template = self.get_template_w_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        events = get_event_list_code(
            example=example,
            representation=representation,
            event_template=self.event_template,
            marker=marker,
        )
        target = get_timeline_code(
            example=example,
            target_template=self.target_template,
            target_event_template=self.target_event_template,
        )
        one_demonstration_in_template = (
            one_demonstration_in_template.replace("{context}", context_with_marker)
            .replace("{events}", events)
            .replace("{target}", target)
        )
        return one_demonstration_in_template

    def get_prompt(self, example: dict[str, Any], representation: str, marker: str):
        one_prompt_in_template = self.get_template_wo_target()
        context_with_marker = add_marker_to_context_all(example, marker)
        events = get_event_list_code(
            example=example,
            representation=representation,
            event_template=self.event_template,
            marker=marker,
        )
        one_prompt_in_template = one_prompt_in_template.replace(
            "{context}", context_with_marker
        ).replace("{events}", events)
        return one_prompt_in_template
