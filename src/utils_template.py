"""

"""

import logging
from typing import Any


def get_edges(
    example,
    target_template: str,
    target_event_template: str,
) -> str:
    output = ""
    for relation in example["all_relations"]:
        if "{idx}" in target_event_template:
            arg1_in_template = target_event_template.replace("{idx}", relation["arg1"])
            arg2_in_template = target_event_template.replace("{idx}", relation["arg2"])
        elif "{mention}" in target_event_template:
            arg1_in_template = target_event_template.replace(
                "{idx}", example["all_events"][str(relation["arg1"])]
            )
            arg2_in_template = target_event_template.replace(
                "{idx}", example["all_events"][str(relation["arg2"])]
            )
        else:
            logging.error(f"Undefined placeholder: {target_event_template}")

        edge_in_template = (
            target_template.replace("{arg1}", arg1_in_template)
            .replace("{arg2}", arg2_in_template)
            .replace("{relation}", relation["relation"])
        )
        output += f"        {edge_in_template}\n"

    return output


def get_timeline_code(
    example,
    target_template: str,
    target_event_template: str,
) -> str:
    output = ""
    for layer_id, layer in enumerate(example["timeline"]):
        events_str = ""
        for eid, event in layer.items():
            event_in_template = target_event_template
            if "{idx}" in target_event_template:
                event_in_template = event_in_template.replace("{idx}", eid)
            elif "{mention}" in target_event_template:
                event_in_template = event_in_template.replace(
                    "{mention}", event["mention"]
                )
            else:
                logging.error(f"Undefined event template: {target_event_template}")

            events_str += f"{event_in_template}, "
        events_str = events_str.strip(", ")
        layer_in_template = target_template.replace("{idx}", str(layer_id)).replace(
            "{events}", events_str
        )
        output += f"        {layer_in_template}\n"

    return output.rstrip()


def get_cot_timeline(
    example,
    representation: str,
    marker: str,
    cot_mapping: dict[str, str],
) -> str:
    output = ""
    for relation in example["all_relations"]:
        arg1_str = get_representation(
            event=example["all_events"][relation["arg1"]],
            eid=relation["arg1"],
            representation=representation,
            marker=marker,
        )
        arg2_str = get_representation(
            event=example["all_events"][relation["arg2"]],
            eid=relation["arg2"],
            representation=representation,
            marker=marker,
        )
        one_relation = (
            cot_mapping[relation["relation"]]
            .replace("{arg1}", arg1_str)
            .replace("{arg2}", arg2_str)
        )
        output += f"{one_relation} "

    return output


def get_cot(
    example,
    representation: str,
    marker: str,
    cot_mapping: dict[str, str],
) -> str:
    answer2relation = {
        answer: relation
        for relation, answers in example["relation2target"].items()
        for answer in answers
    }
    target_event = get_representation(
        event=example["target"],
        eid=example["target_id"],
        representation=representation,
        marker=marker,
    )

    output = ""
    for idx, event in example["all_events"].items():
        if idx == example["target_id"]:
            continue
        answer_event = get_representation(
            event=event, eid=idx, representation=representation, marker=marker
        )
        relation = answer2relation[idx]

        one_thought = (
            cot_mapping[relation]
            .replace("{answer}", answer_event)
            .replace("{event}", target_event)
        )
        output += f"{one_thought} "

    return output


def get_timeline(
    timeline,
    representation: str,
    marker: str,
    flag_list: bool,
) -> str:
    output = ""
    for idx, layer in enumerate(timeline):
        layer_str = f"T{idx}:"
        for eid, event in layer.items():
            event_str = get_representation(
                event=event,
                eid=eid,
                representation=representation,
                marker=marker,
            )
            if flag_list:
                layer_str += f"\n- {event_str}"
            else:
                layer_str += f" {event_str},"
        output += layer_str.rstrip(",") + "\n"

    return output


def get_target_events(events, representation: str, marker: str, flag_list: bool) -> str:
    output = ""
    for idx, event in events.items():
        event_str = get_representation(
            event=event,
            eid=idx,
            representation=representation,
            marker=marker,
        )
        if flag_list:
            output += f"- {event_str}\n"
        else:
            output += f"{event_str}, "

    if not output:
        output = "None"

    return output.rstrip().rstrip("\n").rstrip(", ")


def get_marker_single(marker: str) -> str:
    output = ""
    if marker == "eid":
        output = "[e]"
    elif marker == "star":
        output = "**"
    else:
        logging.error(f"Undefined marker: {marker}")
    return output


def get_marker_pair(idx: str, marker: str) -> tuple[str, str]:
    if marker == "eid":
        marker_start, marker_end = f"[e{idx}]", f"[/e{idx}]"
    elif marker == "star":
        marker_start, marker_end = "**", "**"
    else:
        logging.error(f"Undefined marker: {marker}")
    return marker_start, marker_end


def get_event_list_code(
    example,
    representation: str,
    event_template: str,
    marker: str,
) -> str:
    output = ""
    for idx, event in example["all_events"].items():
        event_str = get_representation(
            event=event,
            eid=idx,
            representation=representation,
            marker=marker,
        )
        event_in_template = event_template
        if "{idx}" in event_in_template:
            event_in_template = event_in_template.replace("{idx}", idx)
        if "{mention}" in event_in_template:
            event_in_template = event_in_template.replace("{mention}", event["mention"])
        if "{representation}" in event_in_template:
            event_in_template = event_in_template.replace("{representation}", event_str)
        output += f"        {event_in_template}\n"
    return output.rstrip()


def get_event_list(example, representation: str, marker: str) -> str:
    output = ""
    for idx, event in example["all_events"].items():
        event_str = get_representation(
            event=event,
            eid=idx,
            representation=representation,
            marker=marker,
        )
        output += f"- {event_str}\n"
    return output.rstrip()


def add_marker_to_context_all(
    example: dict[str, Any],
    marker: str,
):
    _context = example["context"]
    offset_diff = 0
    for idx, event in example["all_events"].items():
        marker_start, marker_end = get_marker_pair(idx=idx, marker=marker)
        _context = (
            _context[: event["start"] + offset_diff]
            + f"{marker_start}{event['mention']}{marker_end}"
            + _context[event["end"] + offset_diff :]
        )
        offset_diff += len(marker_start + marker_end)

    return _context


def add_marker_to_context_pair(example, marker) -> str:
    """
    add event markers

    """

    _context = example["context"]

    if example["arg1"]["start"] < example["arg2"]["start"]:
        context = (
            _context[: example["arg1"]["start"]]
            + ("[e1]" if marker == "eid" else "**")
            + example["arg1"]["mention"]
            + ("[/e1]" if marker == "eid" else "**")
            + _context[example["arg1"]["end"] : example["arg2"]["start"]]
            + ("[e2]" if marker == "eid" else "**")
            + example["arg2"]["mention"]
            + ("[/e2]" if marker == "eid" else "**")
            + _context[example["arg2"]["end"] :]
        )
    else:
        context = (
            _context[: example["arg2"]["start"]]
            + ("[e2]" if marker == "eid" else "**")
            + example["arg2"]["mention"]
            + ("[/e2]" if marker == "eid" else "**")
            + _context[example["arg2"]["end"] : example["arg1"]["start"]]
            + ("[e1]" if marker == "eid" else "**")
            + example["arg1"]["mention"]
            + ("[/e1]" if marker == "eid" else "**")
            + _context[example["arg1"]["end"] :]
        )

    return context


def get_representation(
    event: dict[str, Any],
    eid: str,
    representation: str,
    marker: str,
) -> str:
    """
    get event representation
    * mention:
    * structured: SRL-like structured representation for an event w/ arguments
    * sentence:

    """

    match representation:
        case "mention":
            output = event["mention"]
        case "structured":
            # event -> ARG0 -> ARG1 -> ...
            output = f"[Event]{event['mention']}"
            for role, argument in event["arguments"].items():
                output += f"[{role}]{argument['mention']}"
        case "sentence":
            pass
            # output = event["sentence"]
        case _:
            logging.error(f"Undefined representation type: {representation}")

    if marker == "eid":
        output = f"[e{eid}]{output}[/e{eid}]"
    elif marker == "star":
        output = f"**{output}**"

    return output
