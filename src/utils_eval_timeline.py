from collections import defaultdict
import logging
import json
import networkx as nx
from typing import Any
import re
from sklearn.metrics import classification_report
from template_classes import CTF_RELATIONS
from utils_eval_mrc import find_best_match
from utils_eval import calculate_range_template
from template import TEMPLATES


def _extract_answer(text: str, marker_type: str) -> str:
    """
    extract generated timelines from the raw LLM generation output

    ToDo: check this function with other representations than mention

    """

    # split into each timespan
    splits = re.split(r"(?=\sT\d+:)", text.strip())
    tid_and_events_list = [x.strip() for x in splits if x]

    # from each timespan, extract eid and event text
    timeline = defaultdict(list)
    for tid_and_events in tid_and_events_list:
        splits = tid_and_events.split(":")
        if len(splits) != 2:
            continue
        tid, raw_events = splits

        if marker_type == "eid":
            matches = re.findall(r"(?:\[e(\d+)\](.*?)\[/e\d+\])", raw_events)
            for match in matches:
                eid = match[0] if match[0] else ""
                text = match[1] if match[1] else ""
                if eid and text:
                    timeline[tid].append([eid, text])
        elif marker_type == "star":
            matches = re.findall(r"\*\*(.*?)\*\*", raw_events)
            for match in matches:
                timeline[tid].append([None, match])
        else:
            logging.error(f"Undefined marker type: {marker_type}")

    return timeline


def _extract_answer_graph(
    text: str,
) -> list[Any]:
    """
    extract generated timelines from the raw LLM generation output with graph template

    ToDo: check this function with other representations than mention

    """
    matches = re.findall(r'add_edge\(([^,]+),\s*"([^"]+)",\s*([^)]+)\)', text)
    edges = []
    for match in matches:
        if match[0] and match[1] and match[2]:
            id_arg1 = re.search(r"\d+", match[0])
            id_arg2 = re.search(r"\d+", match[2])
            if id_arg1 and id_arg2:
                edges.append(
                    {
                        "arg1": id_arg1.group(),
                        "relation": match[1],
                        "arg2": id_arg2.group(),
                    }
                )
            else:
                logging.warning(f"ids cannot be found: {match[0]} or {match[2]}")
        else:
            logging.warning(f"complete edge is not found in {match}")

    return edges


def _extract_answer_timeline(
    text: str,
) -> list[Any]:
    """
    extract generated timelines from the raw LLM generation output with graph template

    ToDo: check this function with other representations than mention

    """
    splits = re.split(r"(?=\sT\d+ = )", text.strip())
    tid_and_events_list = [x.strip() for x in splits if x]

    timeline = []
    for tid_and_events in tid_and_events_list:
        splits = tid_and_events.split("=")
        if len(splits) != 2:
            continue
        tid, raw_events = splits

        matches = re.findall(r"self.event(\d+)", raw_events)
        timeline.append([match[0] for match in matches if match])

    return timeline


def post_process(
    timeline: list[Any],
    events: list[Any],
    repr_type: str,
    marker_type: str,
) -> list[Any]:
    """
    post process predicted timeline to
    find the best match considering
    * marker==eid: both eid and representation
        * Default: eid
        * if there is a better representation match, then use the eid of it
    * marker==star: representation
    """
    new_timeline = []
    for layer_id, layer in timeline.items():
        new_layer = []
        for eid, representation in layer:
            new_eid = find_best_match(
                current_eid=eid,
                representation=representation,
                events=events,
                repr_type=repr_type,
                marker_type=marker_type,
            )
            new_layer.append(new_eid)
        new_timeline.append(new_layer)

    # remove duplicate eids
    # eids in earlier layers are prioritized
    eids_found = []
    new_timeline_deduplicated = []
    for layer in new_timeline:
        new_layer = []
        for eid in layer:
            if eid not in eids_found:
                new_layer.append(eid)
                eids_found.append(eid)
        if new_layer:
            new_timeline_deduplicated.append(new_layer)

    return new_timeline_deduplicated


def _create_graph(
    timeline: list[list[str]],
) -> nx.DiGraph:
    G = nx.DiGraph()

    for i in range(len(timeline) - 1):
        current_events, next_events = timeline[i], timeline[i + 1]

        for current_event in current_events:
            for next_event in next_events:
                if current_event == next_event:
                    continue
                G.add_edge(current_event, next_event)

    return G


def get_directed_graph_linearized_no_reduction(relations: list[dict[str, Any]]):
    """
    create directed graph from prediction w/o reduction

    """

    # create coex clusters
    G_coex = nx.Graph()
    for relation in relations:
        if relation["relation"] != "COEX":
            continue
        G_coex.add_edge(relation["arg1"], relation["arg2"])

    idx2cluster = {}
    coex_pairs = []
    for node in G_coex.nodes:
        nodes = list(nx.node_connected_component(nx.transitive_closure(G_coex), node))
        idx2cluster[node] = nodes
        coex_pairs += [(node, _node) for _node in nodes]
    coex_pairs = list(set(coex_pairs))

    G = nx.DiGraph()
    for relation in relations:
        if relation["relation"] != "AFTER":
            continue

        if (relation["arg1"], relation["arg2"]) in coex_pairs:
            continue

        # add the original relation
        if not G.has_edge(relation["arg1"], relation["arg2"]):
            G.add_edge(relation["arg1"], relation["arg2"])

        # augmented relation
        augmented_pairs = []
        if relation["arg1"] in idx2cluster and relation["arg2"] in idx2cluster:
            for cluster_idx1 in idx2cluster[relation["arg1"]]:
                for cluster_idx2 in idx2cluster[relation["arg2"]]:
                    augmented_pairs.append((cluster_idx1, cluster_idx2))
        elif relation["arg1"] in idx2cluster:
            for cluster_idx1 in idx2cluster[relation["arg1"]]:
                augmented_pairs.append((cluster_idx1, relation["arg2"]))
        elif relation["arg2"] in idx2cluster:
            for cluster_idx2 in idx2cluster[relation["arg2"]]:
                augmented_pairs.append((relation["arg1"], cluster_idx2))
        else:
            augmented_pairs.append((relation["arg1"], relation["arg2"]))

        # add augmented relations
        for pair in augmented_pairs:
            if not G.has_edge(pair[0], pair[1]):
                G.add_edge(pair[0], pair[1])

    return G


def _check_cycle(graph, graph_name: str) -> bool:
    flag_cycle = False
    if list(nx.simple_cycles(graph)):
        logging.info(
            f"Cycle found in {graph_name}: {list(nx.simple_cycles(graph))}."
            " Stop calculating temporal awareness score."
        )
        flag_cycle = True

    return flag_cycle


def _calculate_temporal_awareness(
    G1,
    G2,
) -> float:
    """
    calculate temporal awareness

    """

    # check cycle
    if _check_cycle(G1, "G1") or _check_cycle(G2, "G2"):
        return 0.0

    # create closure and reduction for both
    G1_cls = nx.transitive_closure(G1)
    G1_rdc = nx.transitive_reduction(G1)
    G2_cls = nx.transitive_closure(G2)
    G2_rdc = nx.transitive_reduction(G2)

    # calc pairwise F1
    # p/r = transitive reduction of graph 1/2 & transitive closure of graph 2/1
    # / transitive reduction of graph 1/2
    p_G1_in_G2 = len([True for edge in G1_rdc.edges if edge in G2_cls.edges]) / (
        len(G1_rdc.edges) + 1e-20
    )
    r_G2_in_G1 = len([True for edge in G2_rdc.edges if edge in G1_cls.edges]) / (
        len(G2_rdc.edges) + 1e-20
    )
    f1 = 2 * p_G1_in_G2 * r_G2_in_G1 / (p_G1_in_G2 + r_G2_in_G1 + 1e-20)

    return f1


def check_relation(G, node1, node2):
    if G.has_node(node1) and G.has_node(node2):
        if nx.has_path(G, node1, node2):
            relation = "BEFORE"
        elif nx.has_path(G, node2, node1):
            relation = "AFTER"
        else:
            relation = "COEX"
    else:
        relation = None

    return relation


def _calculate_pairwise_accuracy(
    pairs_gold,
    G_pred,
) -> float:
    """
    calculate pairwise accuracy

    """

    labels = CTF_RELATIONS + [None]

    golds = []
    preds = []
    for pair, label in pairs_gold.items():
        node1, node2 = pair.split("-")
        golds.append(labels.index(label))
        preds.append(labels.index(check_relation(G_pred, node1, node2)))

    # Note:
    # The support is the number of occurrences of each class in y_true
    # so the weighted average does not consider the None classes
    report = classification_report(
        golds,
        preds,
        labels=[x for x in range(len(CTF_RELATIONS) + 1)],
        target_names=CTF_RELATIONS + [None],
        output_dict=True,
        zero_division=0.0,
    )

    return report


def evaluate_timeline_ctf(
    examples: list[Any],
    record: Any,
) -> tuple[dict[Any], dict[Any]]:
    """
    calculate metrics for timeline formulation
    """

    report = {
        "document-and-pair-wise-scores": defaultdict(lambda: defaultdict(float)),
        # "temporal-awareness-per-template": defaultdict(lambda: defaultdict(float)),
    }
    repr_type = record["args"]["representation"]
    marker_type = record["args"]["marker"]

    for prediction in record["examples"]:
        example_id = prediction["input"]["example_id"]
        template_id = prediction["input"]["template_id"]

        pairs_gold = examples[example_id]["pairs"]

        if record["args"]["dataset_name"] == "ctf-timeline-code":
            template = TEMPLATES["ctf-timeline-code"][template_id]
            if "graph" in template.name:
                edges = _extract_answer_graph(text=prediction["output"])
                graph_pred = get_directed_graph_linearized_no_reduction(edges)
            elif "timeline" in template.name:
                timeline_pred = _extract_answer_timeline(text=prediction["output"])
                graph_pred = _create_graph(timeline_pred)
            else:
                logging.error(f"Undefined type of template: {template.name}")
        else:
            timeline_pred = _extract_answer(
                text=prediction["output"],
                marker_type=marker_type,
            )
            timeline_pred = post_process(
                timeline=timeline_pred,
                events=examples[example_id]["all_events"],
                repr_type=repr_type,
                marker_type=marker_type,
            )
            graph_pred = _create_graph(timeline_pred)

        score_pairwise_accuracy = _calculate_pairwise_accuracy(
            pairs_gold=pairs_gold["original"],
            G_pred=graph_pred,
        )

        # score_temporal_awareness = _calculate_temporal_awareness(
        #     G1=graph_gold,
        #     G2=graph_pred,
        # )
        filename = examples[example_id]["filename"]
        report["document-and-pair-wise-scores"][template_id][
            filename
        ] = score_pairwise_accuracy
        # report["temporal-awareness-per-template"][example_id][
        #     template_id
        # ] = score_temporal_awareness

    # calc range
    metric_names = [
        "document-and-pair-wise-scores",
        # "temporal-awareness-per-template",
    ]
    report_range = {}
    for metric_name in metric_names:
        report_range[metric_name] = calculate_range_template(report, metric_name)

    return report_range, report


if __name__ == "__main__":
    """
    test

    """
    representation = "mention"
    marker_type = "eid"

    example_output = (
        "T0: [e1]doping[/e1] T1: [e4]loss[/e4] "
        "T2: [e3]testing[/e3] T3: [e2]retired[/e2] T4: [e0]ban[/e0]"
    )
    timelines = _extract_answer(example_output, marker_type)
    print(timelines)

    example_output = (
        "T0: [e2]joined[/e2]\nT1: [e4]loaned[/e4]\nT2: [e5]won[/e5]\n"
        "T3: [e3]won[/e3]\nT4: [e6]won[/e6]\nT5: [e0]agreement[/e0] "
        "[e7]loaned[/e7] [e1]transfer[/e1]"
    )
    timelines = _extract_answer(example_output, marker_type)
    print(timelines)

    example_output = """
T4: [e6][EVENT]won[ARG0]John[/e6],
T5: [e0]agreement[/e0], [e7]loaned[/e7], [e1]transfer[/e1],
    """
    timelines = _extract_answer(example_output, marker_type)
    print(timelines)

    example_output = """
T4: **[EVENT]won[ARG0]John**
T5: **agreement**, **loaned**, **transfer**
    """
    timelines = _extract_answer(example_output, "star")
    print(timelines)

    example_output = """
T0:
  - [e2]joined[/e2]
T5:
  - [e0]agreement[/e0]
  - [e7]loaned[/e7]
  - [e1]transfer[/e1]
"""
    print(example_output)
    timelines = _extract_answer(example_output, marker_type)
    print(timelines)

    raise NotImplementedError
    # TODO: replace < with [

    example_output = """T0:
- <e2>joined</e2>
T1:
- <e4>loaned</e4>
T2:
- <e5>won</e5>
T3:
- <e3>won</e3>
T4:
- <e1>won</e1>
T5:
- <e2>agreement</e2>
- <e7>loaned</e7>
- <e2>transfer</e2>
T6:
- <e0>agreement</e0>
T7:
- <e2>joined</e2>"""  # noqa: E501
    print(example_output)
    timelines = _extract_answer(example_output, marker_type)
    print(timelines)
    with open("./data/preprocessed/ctf/test.json") as f:
        example = json.load(f)[0]
    timelines = post_process(
        timelines,
        example["annotation"]["events"],
        repr_type="mention",
        marker_type="eid",
    )
    print(timelines)

    example_output = """T0:
- **joined**
T1:
- **loaned**
T2:
- **won**
T3:
- **won**
T4:
- **won**
T5:
- **agreement**
- **loaned**
- **transfer**
T6:
- **agreement**
T7:
- **joined**"""  # noqa: E501
    print(example_output)
    timelines = _extract_answer(example_output, "star")
    print(timelines)
    timelines = post_process(
        timelines,
        example["annotation"]["events"],
        repr_type="mention",
        marker_type="eid",
    )
    print(timelines)

    example_output = """<e0>airlifted</e0> started after <e1>flying</e1>. <e1>flying</e1> started after <e2>fell ill</e2>. <e2>fell ill</e2> started after <e3>redirected</e3>. <e3>redirected</e3> started after <e4>drank</e4>. <e4>drank</e4> started after <e0>airlifted</e0>. \nTimeline:\nT0: <e0>airlifted</e0>\nT1: <e1>flying</e1>\nT2: <e2>fell ill</e2>\nT3: <e3>redirected</e3>\nT4: <e4>drank</e4>\nT5: <e0>airlifted</e0>"""  # noqa: E501
    timelines = _extract_answer(example_output, "eid")
    print(timelines)
    # with open("./data/brat/preprocessed/ctf_timeline_7_sampled.json", "r") as f:
    #     examples = json.load(f)

    # graph = _create_graph(examples[0]["timeline"], representation)
    # print(nx.nx_pydot.to_pydot(graph))

    # G1 = nx.DiGraph()
    # nx.add_path(G1, [0, 1, 2])
    # G2 = nx.DiGraph()
    # nx.add_path(G2, [1, 0, 2])
    # f1 = _calculate_temporal_awareness(G1, G2)
    # print(f"temporal awareness: {f1:.3f}")
    # acc = _calculate_pairwise_accuracy(G1, G2)
    # print(f"pairwise accuracy: {acc: .3f}")
