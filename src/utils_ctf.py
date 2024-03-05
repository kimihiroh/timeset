"""
preprocessing script for brat-annotated data

"""

from collections import defaultdict
from itertools import combinations
import networkx as nx
from typing import Any


def get_directed_graph(relations: list[dict[str, Any]]):
    """
    create directed graph from annotation

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

        # add the original relation
        if not G.has_edge(relation["arg1"], relation["arg2"]):
            G.add_edge(relation["arg1"], relation["arg2"])

        # augmented relation
        if (relation["arg1"], relation["arg2"]) not in coex_pairs:
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

    G = nx.transitive_reduction(G)
    G_inverse = G.reverse()

    return G, G_inverse


def get_directed_graph_linearized(relations: list[dict[str, Any]]):
    """
    create directed graph from annotation

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

    G = nx.transitive_reduction(G)
    G_inverse = G.reverse()

    return G, G_inverse


def create_pairwise_examples(annotation):
    """
    pairwise
    """

    G, G_inverse = get_directed_graph(annotation["relations"])
    ids = sorted(list(annotation["events"].keys()))

    examples = []
    for id_arg1 in ids:
        for id_arg2 in ids:
            if id_arg1 == id_arg2:
                continue

            # Note: edge(e1, e2) means e1 -> e2
            # in English, e1 happens before e2
            if nx.has_path(G, id_arg1, id_arg2):
                relation = "BEFORE"
            elif nx.has_path(G_inverse, id_arg1, id_arg2):
                relation = "AFTER"
            else:
                relation = "COEX"

            examples.append(
                {"id_arg1": str(id_arg1), "id_arg2": str(id_arg2), "relation": relation}
            )

    return examples


def create_nli_examples(annotation):
    """
    nli
    """

    G, G_inverse = get_directed_graph(annotation["relations"])
    ids = sorted(list(annotation["events"].keys()))

    examples = []
    for id_arg1 in ids:
        for id_arg2 in ids:
            if id_arg1 == id_arg2:
                continue

            # Note: edge(e1, e2) means e1 -> e2
            # in English, e1 happens before e2
            if nx.has_path(G, id_arg1, id_arg2):
                relation = "BEFORE"
            elif nx.has_path(G_inverse, id_arg1, id_arg2):
                relation = "AFTER"
            else:
                relation = "COEX"

            examples_from_one_pair = []
            for temporal_keyword in ["BEFORE", "AFTER", "COEX"]:
                if temporal_keyword == relation:
                    label = "positive"
                else:
                    label = "negative"

                examples_from_one_pair.append(
                    {
                        "id_arg1": str(id_arg1),
                        "keyword": temporal_keyword,
                        "id_arg2": str(id_arg2),
                        "label": label,
                    }
                )

            examples += examples_from_one_pair

    return examples


def create_mrc_examples(annotation):
    """
    mrc
    """

    G, G_inverse = get_directed_graph(annotation["relations"])
    ids = sorted(list(annotation["events"].keys()))

    examples = []
    for id_arg1 in ids:
        answers = defaultdict(list)
        for id_arg2 in ids:
            if id_arg1 == id_arg2:
                continue

            # Note: edge(e1, e2) means e1 -> e2
            # in English, e1 happens before e2
            if nx.has_path(G, id_arg1, id_arg2):
                answers["AFTER"].append(str(id_arg2))
            elif nx.has_path(G_inverse, id_arg1, id_arg2):
                answers["BEFORE"].append(str(id_arg2))
            else:
                answers["COEX"].append(str(id_arg2))

        for relation in ["BEFORE", "AFTER", "COEX"]:
            examples.append(
                {
                    "target": str(id_arg1),
                    "answers": answers[relation],
                    "relation2answers": answers,
                    "relation": relation,
                }
            )

    return examples


def create_timeline_examples(annotation):
    """
    timeline
    """

    G, G_inverse = get_directed_graph_linearized(annotation["relations"])

    # find starting nodes, in_degree == 0
    init_layer = [x for x in G.nodes() if G.in_degree(x) == 0]
    layers = [
        [str(idx) for idx in layer] for layer in list(nx.bfs_layers(G, init_layer))
    ]

    examples = [layers]

    return examples


def get_pair_labels(annotation, flag_linearized: bool = False):
    if flag_linearized:
        G, G_inverse = get_directed_graph_linearized(annotation["relations"])
    else:
        G, G_inverse = get_directed_graph(annotation["relations"])
    ids = sorted(list(annotation["events"].keys()))

    pair2relation = {}
    for id_arg1, id_arg2 in combinations(ids, 2):
        # Note: edge(e1, e2) means e1 -> e2
        # in English, e1 happens before e2
        if nx.has_path(G, id_arg1, id_arg2):
            relation = "BEFORE"
        elif nx.has_path(G_inverse, id_arg1, id_arg2):
            relation = "AFTER"
        else:
            relation = "COEX"

        pair2relation[f"{id_arg1}-{id_arg2}"] = relation

    return pair2relation


if __name__ == "__main__":
    # test code

    relations = [
        {"arg1": 0, "arg2": 1, "relation": "AFTER"},
        {"arg1": 1, "arg2": 2, "relation": "AFTER"},
        {"arg1": 1, "arg2": 3, "relation": "COEX"},
        {"arg1": 2, "arg2": 3, "relation": "COEX"},
        {"arg1": 2, "arg2": 4, "relation": "AFTER"},
        {"arg1": 4, "arg2": 5, "relation": "AFTER"},
        {"arg1": 5, "arg2": 6, "relation": "COEX"},
        {"arg1": 6, "arg2": 7, "relation": "COEX"},
        {"arg1": 7, "arg2": 8, "relation": "AFTER"},
    ]
    G, _G = get_directed_graph(relations)
    print(nx.nx_pydot.to_pydot(G))
    print(nx.nx_pydot.to_pydot(_G))
    G, _ = get_directed_graph_linearized(relations)
    print(nx.nx_pydot.to_pydot(G))
