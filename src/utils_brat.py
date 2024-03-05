"""
Utils to parse .ann files

"""

from dataclasses import asdict, dataclass
import logging
from pathlib import Path

# from typing import Any


@dataclass
class Textbound:
    tid: str
    text: str
    start: int
    end: int
    qnode: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Event:
    textbound: Textbound
    realis: str

    def to_dict(self):
        output = {"textbound": self.textbound.to_dict(), "realis": self.realis}
        return output


@dataclass
class Argument:
    textbound: Textbound
    role: str

    def to_dict(self):
        output = {"textbound": self.textbound.to_dict(), "role": self.role}
        return output


@dataclass
class EventWithArguments:
    event: Event
    arguments: list[Argument]

    def to_dict(self):
        output = {
            "event": self.event.to_dict(),
            "arguments": [x.to_dict() for x in self.arguments],
        }
        return output


@dataclass
class Relation:
    relation: str
    arg1: str
    arg2: str

    def to_dict(self):
        return asdict(self)


def extract_textbounds(lines: list[str]) -> dict[str, Textbound]:
    """
    get textbound: T
    e.g. T54	contact_contact 2133 2139;2146 2151	placed blame

    """

    tid2textbound: dict[str, Textbound] = {}

    for line in lines:
        if not line.startswith("T"):
            continue

        tid, items, text = line.strip().split("\t")

        if ";" in line:  # discontinuous event
            # Note: [current] chose the first part only
            qnode, start, end, *_ = items.replace(";", " ").split()
            text = text[: int(end) - int(start)]
        else:
            qnode, start, end = items.split()

        tid2textbound[tid] = Textbound(
            tid=tid, text=text, start=int(start), end=int(end), qnode=qnode
        )

    return tid2textbound


def extract_event_with_arguments(
    lines: list[str],
    tid2textbound: dict[str, Textbound],
) -> tuple[dict[str, EventWithArguments], dict[str, Textbound]]:
    """
    get event: E
    e.g. E4	ManArt_Build:T4 Place:T19 Artifact:T18

    """

    eid2event_with_arguments = {}
    tid2entity = {}
    event_tids = []
    eid2tid = {}
    for line in lines:
        if not line.startswith("E"):
            continue

        eid, items = line.strip().split("\t")
        current_event, *current_arguments = items.split()
        qnode, event_tid = current_event.split(":")

        assert qnode == tid2textbound[event_tid].qnode

        event = Event(
            textbound=tid2textbound[event_tid],
            realis="ACTUAL",  # default realis
        )
        event_tids.append(event_tid)
        eid2tid[eid] = event_tid

        eid2event_with_arguments[eid] = EventWithArguments(
            event=event,
            arguments=None,
        )

    for line in lines:
        if not line.startswith("E"):
            continue

        eid, items = line.strip().split("\t")
        current_event, *current_arguments = items.split()

        # get arguments
        # e.g., ... Place:T19, Artifact:T18
        arguments: list[Argument] = []
        for arg in current_arguments:
            role, tid_or_eid = arg.split(":")
            # note: event can be an argument
            if tid_or_eid.startswith("E"):
                tid = eid2tid[tid_or_eid]
            else:
                tid = tid_or_eid

            if tid in tid2textbound:
                textbound = tid2textbound[tid]

                arguments.append(Argument(textbound=textbound, role=role))

                if tid not in tid2entity:  # store this argument unless already seen
                    tid2entity[tid] = textbound
            else:
                logging.warning(f"tid, {tid}, for argument, {arg}, does not exist.")

        eid2event_with_arguments[eid].arguments = arguments

    # add entities that are not event arguments
    for line in lines:
        if not line.startswith("T"):
            continue

        tid, items, text = line.strip().split("\t")
        if tid not in event_tids and tid not in tid2entity:
            tid2entity[tid] = tid2textbound[tid]

    return eid2event_with_arguments, tid2entity


def update_realis(
    lines: list[str],
    eid2event_with_arguments: dict[str, EventWithArguments],
) -> dict[str, EventWithArguments]:
    """
    get realis
    e.g. A6	Realis E27 NEGATED

    """

    for line in lines:
        if not line.startswith("A"):
            continue

        aid, splits = line.strip().split("\t")
        _, eid, realis = splits.split()

        if realis not in ["NEGATED", "GENERIC", "HEDGED", "IRREALIS"]:
            logging.error(f"Unknown realis type: {realis}")

        if eid in eid2event_with_arguments:
            eid2event_with_arguments[eid].event.realis = realis

    return eid2event_with_arguments


def extract_relations(lines: list[str]) -> dict[str, Relation]:
    """
    get relation: R
    time order: event1 -> event2
    e.g. R19	After Arg1:E4 Arg2:E8
    e.g. R15	EntityCoref Arg1:T86 Arg2:T87

    """

    rid2relation = {}
    for line in lines:
        if not line.startswith("R"):
            continue

        rid, items = line.strip().split("\t")
        relation, arg1, arg2 = items.split()
        _, eid_or_tid_1 = arg1.split(":")
        _, eid_or_tid_2 = arg2.split(":")
        rid2relation[rid] = Relation(
            relation=relation,
            arg1=eid_or_tid_1,
            arg2=eid_or_tid_2,
        )

    return rid2relation


def parse_ann_file(
    filepath: Path,
    text_bound: bool = False,
) -> tuple[dict[str, EventWithArguments], dict[str, Textbound], dict[str, Relation]]:
    """
    parse .ann file

    """

    with open(filepath, "r") as f:
        lines = f.readlines()

    tid2textbound = extract_textbounds(lines)
    eid2event_with_arguments, tid2entity = extract_event_with_arguments(
        lines, tid2textbound
    )
    eid2event_with_arguments = update_realis(lines, eid2event_with_arguments)
    rid2relation = extract_relations(lines)

    if text_bound:
        return eid2event_with_arguments, tid2entity, rid2relation, tid2textbound
    else:
        return eid2event_with_arguments, tid2entity, rid2relation


def parse_txt_file(
    filepath_ann: Path,
) -> str:
    """
    parse .txt file

    """

    assert filepath_ann.suffix == ".ann"

    with open(
        filepath_ann.parent / filepath_ann.name.replace(".ann", ".txt"), "r"
    ) as f:
        text = f.read()

    return text
