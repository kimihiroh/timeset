"""

"""

import json
from utils_dataset import _preprocess_example
from template_classes import (
    MATRES_TEMPLATE,
    TDDISCOURSE_TEMPLATE,
    TORQUE_TEMPLATE,
    TEMPORALNLI_TEMPLATE,
    NLI_TEMPLATE,
    PAIRWISE_TEMPLATE,
    MRC_TEMPLATE,
    TIMELINE_TEMPLATE,
    TIMELINE_CODE_GRAPH_TEMPLATE,
    TIMELINE_CODE_TIMELINE_TEMPLATE,
)

# =========== prompt templates for benchmarking =========== #

TEMPLATES_MATRES = {
    "temporal_eid": MATRES_TEMPLATE(
        name="temporal_eid",
        inputs_prefix="",
        inputs=(
            """Context: {context}
Question: What is the temporal relationship between {arg1} and {arg2}?
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "temporal_mention": MATRES_TEMPLATE(
        name="temporal_mention",
        inputs_prefix="",
        inputs=(
            """Context: {context}
Question: What is the temporal relationship between {arg1} and {arg2}?
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "find_eid": MATRES_TEMPLATE(
        name="find_eid",
        inputs_prefix="Read the following text and answer the question.\n",
        inputs=(
            """Text:
{context}
Question: Find the temporal relationship between {arg1} and {arg2}.
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "find_mention": MATRES_TEMPLATE(
        name="find_mention",
        inputs_prefix="Read the following text and answer the question.\n",
        inputs=(
            """Choices:
{choices}
Text:
{context}
Question: Find the temporal relationship between {arg1} and {arg2}.
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "semantic_eid": MATRES_TEMPLATE(
        name="semantic_eid",
        inputs_prefix="",
        inputs=(
            """Given the sentences, {context}

Out of the options
{choices}
What is the semantic relation between {arg1} and {arg2} in the sentences:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "semantic_mention": MATRES_TEMPLATE(
        name="semantic_mention",
        inputs_prefix="",
        inputs=(
            """Given the sentences, {context}

Out of the options
{choices}
What is the semantic relation between {arg1} and {arg2} in the sentences:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "please_eid": MATRES_TEMPLATE(
        name="please_eid",
        inputs_prefix="",
        inputs=(
            """Sentences:
{context}
How temporally related are the two events, {arg1} and {arg2}, in the sentences?
Please answer with one of the following options:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "please_mention": MATRES_TEMPLATE(
        name="please_mention",
        inputs_prefix="",
        inputs=(
            """Sentences:
{context}
How temporally related are the two events, {arg1} and {arg2}, in the sentences?
Please answer with one of the following options:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "choose_eid": MATRES_TEMPLATE(
        name="choose_eid",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Based on the passage, choose the temporal relation of the two events, {arg1} and {arg2}?
Please answer with one of the following options:
{choices}
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "choose_mention": MATRES_TEMPLATE(
        name="choose_mention",
        inputs_prefix="",
        inputs=(
            """Please answer with one of the following options:
{choices}
Passage:
{context}
Based on the passage, choose the temporal relation of the two events, {arg1} and {arg2}?
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
}

TEMPLATES_TDDISCOURSE = {
    "temporal_eid": TDDISCOURSE_TEMPLATE(
        name="temporal_eid",
        inputs_prefix="",
        inputs=(
            """Context: {context}
Question: What is the temporal relationship between {arg1} and {arg2}?
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "temporal_mention": TDDISCOURSE_TEMPLATE(
        name="temporal_mention",
        inputs_prefix="",
        inputs=(
            """Context: {context}
Question: What is the temporal relationship between {arg1} and {arg2}?
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "find_eid": TDDISCOURSE_TEMPLATE(
        name="find_eid",
        inputs_prefix="Read the following text and answer the question.\n",
        inputs=(
            """Text:
{context}
Question: Find the temporal relationship between {arg1} and {arg2}.
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "find_mention": TDDISCOURSE_TEMPLATE(
        name="find_mention",
        inputs_prefix="Read the following text and answer the question.\n",
        inputs=(
            """Choices:
{choices}
Text:
{context}
Question: Find the temporal relationship between {arg1} and {arg2}.
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "semantic_eid": TDDISCOURSE_TEMPLATE(
        name="semantic_eid",
        inputs_prefix="",
        inputs=(
            """Given the sentences, {context}

Out of the options
{choices}
What is the semantic relation between {arg1} and {arg2} in the sentences:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "semantic_mention": TDDISCOURSE_TEMPLATE(
        name="semantic_mention",
        inputs_prefix="",
        inputs=(
            """Given the sentences, {context}

Out of the options
{choices}
What is the semantic relation between {arg1} and {arg2} in the sentences:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "please_eid": TDDISCOURSE_TEMPLATE(
        name="please_eid",
        inputs_prefix="",
        inputs=(
            """Sentences:
{context}
How temporally related are the two events, {arg1} and {arg2}, in the sentences?
Please answer with one of the following options:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "please_mention": TDDISCOURSE_TEMPLATE(
        name="please_mention",
        inputs_prefix="",
        inputs=(
            """Sentences:
{context}
How temporally related are the two events, {arg1} and {arg2}, in the sentences?
Please answer with one of the following options:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
    "choose_eid": TDDISCOURSE_TEMPLATE(
        name="choose_eid",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Based on the passage, choose the temporal relation of the two events, {arg1} and {arg2}?
Please answer with one of the following options:
{choices}
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="eid",
    ),
    "choose_mention": TDDISCOURSE_TEMPLATE(
        name="choose_mention",
        inputs_prefix="",
        inputs=(
            """Please answer with one of the following options:
{choices}
Passage:
{context}
Based on the passage, choose the temporal relation of the two events, {arg1} and {arg2}?
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        event_representation="mention",
    ),
}

TEMPLATES_TORQUE = {
    "simple": TORQUE_TEMPLATE(
        name="simple",
        inputs_prefix="",
        inputs=(
            """Context:
{context}
Question: {question}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "extract": TORQUE_TEMPLATE(
        name="extract",
        inputs_prefix="",
        inputs=(
            """Extract the answer to the question from the following context.
Question: {question}
Context: {context}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "note": TORQUE_TEMPLATE(
        name="note",
        inputs_prefix="",
        inputs=(
            """Given the following passage
"{context}"
answer the following question.
Question: {question}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "I_know": TORQUE_TEMPLATE(
        name="I_know",
        inputs_prefix="",
        inputs=(
            """I know that the answer to the question "{question}" is in "{context}". Can you tell me what they are?
Answer:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "qa": TORQUE_TEMPLATE(
        name="qa",
        inputs_prefix="",
        inputs=(
            """{question}
Answer the above question based on the context below:
{context}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "friend": TORQUE_TEMPLATE(
        name="friend",
        inputs_prefix="",
        inputs=(
            """A friend asked me to answer this question: {question}, using the article:
{context}
What would be the answer(s)?
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "refer": TORQUE_TEMPLATE(
        name="refer",
        inputs_prefix="",
        inputs=(
            """Refer to the passage below and answer the following question.
Passage:
{context}
Question:
{question}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "refer_above": TORQUE_TEMPLATE(
        name="refer_above",
        inputs_prefix="",
        inputs=(
            """{context}
Q: {question}
Referring to the passage above, the correct answers to the given question is:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "info": TORQUE_TEMPLATE(
        name="info",
        inputs_prefix="",
        inputs=(
            """Answer the following question, "{question}" using the information below.
{context}
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "pick": TORQUE_TEMPLATE(
        name="pick",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Based on the passage above, answer the question: {question}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
}

TEMPLATES_TEMPORALNLI = {
    "simple": TEMPORALNLI_TEMPLATE(
        name="simple",
        inputs_prefix="",
        inputs=(
            """Context: {premise}
Statement: {hypothesis}
Yes or No:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "yes": "positive",
            "no": "negative",
        },
        original2target={
            "positive": "Yes",
            "negative": "No",
        },
    ),
    "unclear": TEMPORALNLI_TEMPLATE(
        name="unclear",
        inputs_prefix="",
        inputs=(
            """Context: {premise}
Statement: {hypothesis}
Based on the context above, is the statement true, false, or inconclusive?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "true": "positive",
            "false": "negative",
            "inconclusive": "negative",
        },
        original2target={
            "positive": "True",
            "negative": "False",
        },
    ),
    "take": TEMPORALNLI_TEMPLATE(
        name="take",
        inputs_prefix="",
        inputs=(
            """Take the following as truth: {premise}
Then the following statement: "{hypothesis}" is true, false, or inconclusive?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "true": "positive",
            "false": "negative",
            "inconclusive": "negative",
        },
        original2target={
            "positive": "True",
            "negative": "False",
        },
    ),
    "only": TEMPORALNLI_TEMPLATE(
        name="only",
        inputs_prefix="",
        inputs=(
            """{premise}
Using only the above description and what you know about the world, "{hypothesis}" is definitely correct, incorrect, or inconclusive?"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "correct": "positive",
            "incorrect": "negative",
            "inconclusive": "negative",
        },
        original2target={
            "positive": "Correct",
            "negative": "Incorrect",
        },
    ),
    "assume": TEMPORALNLI_TEMPLATE(
        name="assume",
        inputs_prefix="",
        inputs=(
            """Assume it is true that {premise}
Therefore, "{hypothesis}" is guaranteed, possible, or impossible?"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "guaranteed": "positive",
            "impossible": "negative",
            "possible": "negative",
        },
        original2target={
            "positive": "Guaranteed",
            "negative": "Impossible",
        },
    ),
    "based": TEMPORALNLI_TEMPLATE(
        name="based",
        inputs_prefix="",
        inputs=(
            """{premise}
Based on the previous passage, is it true that "{hypothesis}"? Yes, no, or maybe?"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "yes": "positive",
            "no": "negative",
            "maybe": "negative",
        },
        original2target={
            "positive": "Yes",
            "negative": "No",
        },
    ),
    "keep": TEMPORALNLI_TEMPLATE(
        name="keep",
        inputs_prefix="",
        inputs=(
            """{premise}
Keeping in mind the above text, consider: {hypothesis}
Is this always, sometimes, or never correct?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "always": "positive",
            "never": "negative",
            "sometimes": "negative",
        },
        original2target={
            "positive": "Always",
            "negative": "Never",
        },
    ),
    "imply": TEMPORALNLI_TEMPLATE(
        name="imply",
        inputs_prefix="",
        inputs=(
            """{premise}
Does this imply
{hypothesis}
Please answer yes, no, or maybe."""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "yes": "positive",
            "no": "negative",
            "maybe": "negative",
        },
        original2target={
            "positive": "Yes",
            "negative": "No",
        },
    ),
    "entailment": TEMPORALNLI_TEMPLATE(
        name="entailment",
        inputs_prefix="",
        inputs=(
            """We say that one sentence "entails" another sentence when the first sentence implies the second sentence. Consider the following two sentences:
{premise}
{hypothesis}
Is the relationship from the first to the second sentence "entailment" or "not entailment"?"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "entailment": "positive",
            "not entailment": "negative",
        },
        original2target={
            "positive": "Entailment",
            "negative": "Not entailment",
        },
    ),
    "confident": TEMPORALNLI_TEMPLATE(
        name="confident",
        inputs_prefix="",
        inputs=(
            """Suppose it's true that
{premise}
how confident should I be that "{hypothesis}"
very confident or not confident?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "very confident": "positive",
            "not confident": "negative",
        },
        original2target={
            "positive": "Very confident",
            "negative": "Not confident",
        },
    ),
}

# =========== prompt templates for formulation comparison =========== #

TEMPLATES_NLI = {
    "simple": NLI_TEMPLATE(
        name="simple",
        inputs_prefix="",
        inputs=(
            """Context: {premise}
Statement: {hypothesis}
Yes or No:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "yes": "positive",
            "no": "negative",
        },
        original2target={
            "positive": "Yes",
            "negative": "No",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} happened after {arg2}.",
            "BEFORE": "{arg1} happened before {arg2}.",
            "COEX": "{arg1} happened around the same time as {arg2}.",
        },
    ),
    "unclear": NLI_TEMPLATE(
        name="unclear",
        inputs_prefix="",
        inputs=(
            """Context: {premise}
Statement: {hypothesis}
Based on the context above, is the statement true, false, or inconclusive?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "true": "positive",
            "false": "negative",
            "inconclusive": "negative",
        },
        original2target={
            "positive": "True",
            "negative": "False",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} started after {arg2} started.",
            "BEFORE": "{arg1} started before {arg2} started.",
            "COEX": "The temporal relationship between {arg1} and {arg2} is unclear.",
        },
    ),
    "take": NLI_TEMPLATE(
        name="take",
        inputs_prefix="",
        inputs=(
            """Take the following as truth: {premise}
Then the following statement: "{hypothesis}" is true, false, or inconclusive?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "true": "positive",
            "false": "negative",
            "inconclusive": "negative",
        },
        original2target={
            "positive": "True",
            "negative": "False",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} started after {arg2} started.",
            "BEFORE": "{arg1} started before {arg2} started.",
            "COEX": "You cannot tell which started first, {arg1} or {arg2}.",
        },
    ),
    "only": NLI_TEMPLATE(
        name="only",
        inputs_prefix="",
        inputs=(
            """{premise}
Using only the above description and what you know about the world, "{hypothesis}" is definitely correct, incorrect, or inconclusive?"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "correct": "positive",
            "incorrect": "negative",
            "inconclusive": "negative",
        },
        original2target={
            "positive": "Correct",
            "negative": "Incorrect",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} happened after {arg2}.",
            "BEFORE": "{arg1} happened before {arg2}.",
            "COEX": "{arg1} happened around the same time as {arg2}.",
        },
    ),
    "assume": NLI_TEMPLATE(
        name="assume",
        inputs_prefix="",
        inputs=(
            """Assume it is true that {premise}
Therefore, "{hypothesis}" is guaranteed, possible, or impossible?"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "guaranteed": "positive",
            "impossible": "negative",
            "possible": "negative",
        },
        original2target={
            "positive": "Guaranteed",
            "negative": "Impossible",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} happened after {arg2}.",
            "BEFORE": "{arg1} happened before {arg2}.",
            "COEX": "{arg1} happened around the same time as {arg2}.",
        },
    ),
    "based": NLI_TEMPLATE(
        name="based",
        inputs_prefix="",
        inputs=(
            """{premise}
Based on the previous passage, is it true that "{hypothesis}"? Yes, no, or maybe?"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "yes": "positive",
            "no": "negative",
            "maybe": "negative",
        },
        original2target={
            "positive": "Yes",
            "negative": "No",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} started after {arg2} started.",
            "BEFORE": "{arg1} started before {arg2} started.",
            "COEX": "{arg1} started around the same time as {arg2} started.",
        },
    ),
    "keep": NLI_TEMPLATE(
        name="keep",
        inputs_prefix="",
        inputs=(
            """{premise}
Keeping in mind the above text, consider: {hypothesis}
Is this always, sometimes, or never correct?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "always": "positive",
            "never": "negative",
            "sometimes": "negative",
        },
        original2target={
            "positive": "Always",
            "negative": "Never",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} started after {arg2} started.",
            "BEFORE": "{arg1} started before {arg2} started.",
            "COEX": "{arg1} started around the same time as {arg2} started.",
        },
    ),
    "imply": NLI_TEMPLATE(
        name="imply",
        inputs_prefix="",
        inputs=(
            """{premise}
Does this imply
{hypothesis}
Please answer yes, no, or maybe."""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "yes": "positive",
            "no": "negative",
            "maybe": "negative",
        },
        original2target={
            "positive": "Yes",
            "negative": "No",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} started after {arg2} started.",
            "BEFORE": "{arg1} started before {arg2} started.",
            "COEX": "The start time of {arg1} is not clear ame time as {arg2} started.",
        },
    ),
    "entailment": NLI_TEMPLATE(
        name="entailment",
        inputs_prefix="",
        inputs=(
            """We say that one sentence "entails" another sentence when the first sentence implies the second sentence. Consider the following two sentences:
{premise}
{hypothesis}
Is the relationship from the first to the second sentence "entailment" or "not entailment"?"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "entailment": "positive",
            "not entailment": "negative",
        },
        original2target={
            "positive": "Entailment",
            "negative": "Not entailment",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} started after {arg2} started.",
            "BEFORE": "{arg1} started before {arg2} started.",
            "COEX": "The start time of {arg1} is uncertain when compared to that of {arg2}.",  # noqa: E501
        },
    ),
    "confident": NLI_TEMPLATE(
        name="confident",
        inputs_prefix="",
        inputs=(
            """Suppose it's true that
{premise}
how confident should I be that "{hypothesis}"
very confident or not confident?"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target2original={
            "very confident": "positive",
            "not confident": "negative",
        },
        original2target={
            "positive": "Very confident",
            "negative": "Not confident",
        },
        hypothesis_mapping={
            "AFTER": "{arg1} happened after {arg2}.",
            "BEFORE": "{arg1} happened before {arg2}.",
            "COEX": "Not enough information exists which happened first, {arg1} or {arg2}.",  # noqa: E501
        },
    ),
}

TEMPLATES_PAIRWISE = {
    "start_vague": PAIRWISE_TEMPLATE(
        name="start_vague",
        inputs_prefix="",
        inputs=(
            """Context: {context}
Question: What is the temporal relationship between {arg1} and {arg2}?
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event started before the second event started.",  # noqa: E501
            "AFTER": "The first event started after the second event started.",  # noqa: E501
            "VAGUE": "The temporal relationship between the first and second event is unclear.",  # noqa: E501
        },
    ),
    "start_coex": PAIRWISE_TEMPLATE(
        name="start_coex",
        inputs_prefix="",
        inputs=(
            """Context: {context}
Choices:
{choices}
Question: What is the temporal relationships between {arg1} and {arg2}?
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event started before the second event started.",  # noqa: E501
            "AFTER": "The first event started after the second event stared.",  # noqa: E501
            "COEX": "The first and second event started around the same time, but the temporal relationship between them is not clear.",  # noqa: E501
        },
    ),
    "happen_vague": PAIRWISE_TEMPLATE(
        name="happen_vague",
        inputs_prefix="Read the following text and answer the question.\n",
        inputs=(
            """Text:
{context}
Question: Find the temporal relationship between {arg1} and {arg2}.
Choices:
{choices}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event happened before the second event happened.",  # noqa: E501
            "AFTER": "The first event happened after the second event happened.",  # noqa: E501
            "VAGUE": "The temporal relationship between the first and second event is unclear.",  # noqa: E501
        },
    ),
    "happen_coex": PAIRWISE_TEMPLATE(
        name="happen_coex",
        inputs_prefix="Read the following text and answer the question.\n",
        inputs=(
            """Choices:
{choices}
Text:
{context}
Question: Find the temporal relationship between {arg1} and {arg2}.
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event happened before the second event happened.",  # noqa: E501
            "AFTER": "The first event happened after the second event happened.",  # noqa: E501
            "COEX": "The first and second event happened around the same time, but the temporal relationship between them is not clear.",  # noqa: E501
        },
    ),
    "semantic_start_vague": PAIRWISE_TEMPLATE(
        name="semantic_start_vague",
        inputs_prefix="",
        inputs=(
            """Given the sentences, {context}

Out of the options
{choices}
What is the semantic relation between {arg1} and {arg2} in the sentences:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event started before the second event started.",  # noqa: E501
            "AFTER": "The first event started after the second event started.",  # noqa: E501
            "VAGUE": "The temporal relationship between the first and second event is unclear.",  # noqa: E501
        },
    ),
    "semantic_start_coex": PAIRWISE_TEMPLATE(
        name="semantic_start_coex",
        inputs_prefix="",
        inputs=(
            """Given the sentences, {context}

Out of the options
{choices}
What is the semantic relation between {arg1} and {arg2} in the sentences:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event started before the second event started.",  # noqa: E501
            "AFTER": "The first event started after the second event stared.",  # noqa: E501
            "COEX": "The first and second event started around the same time, but the temporal relationship between them is not clear.",  # noqa: E501
        },
    ),
    "please_start_coex": PAIRWISE_TEMPLATE(
        name="please_start_coex",
        inputs_prefix="",
        inputs=(
            """Sentences:
{context}
How temporally related are the two events, {arg1} and {arg2}, in the sentences?
Please answer with one of the following options:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event started before the second event started.",  # noqa: E501
            "AFTER": "The first event started after the second event stared.",  # noqa: E501
            "COEX": "The first and second event started around the same time, but the temporal relationship between them is not clear.",  # noqa: E501
        },
    ),
    "please_happen_coex": PAIRWISE_TEMPLATE(
        name="please_happen_coex",
        inputs_prefix="",
        inputs=(
            """Sentences:
{context}
How temporally related are the two events, {arg1} and {arg2}, in the sentences?
Please answer with one of the following options:
{choices}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event happened before the second event happened.",  # noqa: E501
            "AFTER": "The first event happened after the second event happened.",  # noqa: E501
            "COEX": "The first and second event happened around the same time, but the temporal relationship between them is not clear.",  # noqa: E501
        },
    ),
    "choose_start_vague": PAIRWISE_TEMPLATE(
        name="choose_start_vague",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Based on the passage, choose the temporal relation of the two events, {arg1} and {arg2}?
Please answer with one of the following options:
{choices}
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event started before the second event started.",  # noqa: E501
            "AFTER": "The first event started after the second event started.",  # noqa: E501
            "VAGUE": "The temporal relationship between the first and second event is unclear.",  # noqa: E501
        },
    ),
    "choose_happen_vague_reverse": PAIRWISE_TEMPLATE(
        name="choose_happen_vague_reverse",
        inputs_prefix="",
        inputs=(
            """Please answer with one of the following options:
{choices}
Passage:
{context}
Based on the passage, choose the temporal relation of the two events, {arg1} and {arg2}?
Answer:"""  # noqa: E501
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        choices={
            "BEFORE": "The first event happened before the second event happened.",  # noqa: E501
            "AFTER": "The first event happened after the second event happened.",  # noqa: E501
            "VAGUE": "The temporal relationship between the first and second event is unclear.",  # noqa: E501
        },
    ),
}

TEMPLATES_MRC = {
    "simple": MRC_TEMPLATE(
        name="simple",
        inputs_prefix="",
        inputs=(
            """Context:
{context}
Question: {question}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events started after {event}?",  # noqa: E501
            "BEFORE": "Which events started before {event}?",  # noqa: E501
            "COEX": "Which events do not have a clear temporal relation with {event}?",  # noqa: E501
        },
        flag_list_events=False,
        flag_marker=False,
    ),
    "simple_events": MRC_TEMPLATE(
        name="simple_events",
        inputs_prefix="",
        inputs=(
            """Context:
{context}
Question: {question}
Answer candidates:
{events}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events started after {event}?",  # noqa: E501
            "BEFORE": "Which events started before {event}?",  # noqa: E501
            "COEX": "Which events started around the same time but without clear temporal relations with {event}?",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
    ),
    "note": MRC_TEMPLATE(
        name="note",
        inputs_prefix="",
        inputs=(
            """Given the following passage
"{context}"
answer the following question. Note that events are marked with "{marker}".
Question: {question}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events happened after {event}?",  # noqa: E501
            "BEFORE": "Which events happened before {event}?",  # noqa: E501
            "COEX": "Which events happened around the same time but without clear temporal relations with {event}?",  # noqa: E501
        },
        flag_list_events=False,
        flag_marker=True,
    ),
    "note_events": MRC_TEMPLATE(
        name="note_events",
        inputs_prefix="",
        inputs=(
            """Given the following passage
"{context}"
answer the following question. Note that following are the events:
{events}
Question: {question}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events happened after {event}?",  # noqa: E501
            "BEFORE": "Which events happened before {event}?",  # noqa: E501
            "COEX": "Which events do not have a clear temporal relation with {event}?",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
    ),
    "qa": MRC_TEMPLATE(
        name="qa",
        inputs_prefix="",
        inputs=(
            """{question}
Answer the above question based on the context below:
{context}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events started after {event}?",  # noqa: E501
            "BEFORE": "Which events started before {event}?",  # noqa: E501
            "COEX": "Which events do not have a clear temporal relation with {event}?",  # noqa: E501
        },
        flag_list_events=False,
        flag_marker=False,
    ),
    "qa_events": MRC_TEMPLATE(
        name="qa_events",
        inputs_prefix="",
        inputs=(
            """{question}
Answer the above question based on the context below:
{context}
Note: Events are marked with "{marker}".
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events started after {event}?",  # noqa: E501
            "BEFORE": "Which events started before {event}?",  # noqa: E501
            "COEX": "Which events do not have a clear temporal relation with {event}?",  # noqa: E501
        },
        flag_list_events=False,
        flag_marker=True,
    ),
    "refer": MRC_TEMPLATE(
        name="refer",
        inputs_prefix="",
        inputs=(
            """Refer to the passage below and answer the following question.
Passage:
{context}
Note: Events are marked with "{marker}".
Question:
{question}
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events happened after {event}?",  # noqa: E501
            "BEFORE": "Which events happened before {event}?",  # noqa: E501
            "COEX": "Which events happened around the same time but without clear temporal relations with {event}?",  # noqa: E501
        },
        flag_list_events=False,
        flag_marker=True,
    ),
    "refer_events": MRC_TEMPLATE(
        name="refer_events",
        inputs_prefix="",
        inputs=(
            """{context}
Events:
{events}
Q: {question}
Referring to the passage above, the correct answers to the given question is:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "List events that started after {event}.",  # noqa: E501
            "BEFORE": "List events that started before {event}.",  # noqa: E501
            "COEX": "List events that started around the same time, but the temporal relationship with {event} is not clear.",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
    ),
    "hint": MRC_TEMPLATE(
        name="hint",
        inputs_prefix="",
        inputs=(
            """Answer the following question, "{question}" using the information below.
{context}
Hint: events are marked with "{marker}".
Answer:"""
        ),
        x_y_delimiter=" ",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events happened after {event}?",  # noqa: E501
            "BEFORE": "Which events happened before {event}?",  # noqa: E501
            "COEX": "Which events do not have a clear temporal relation with {event}?",
        },
        flag_list_events=False,
        flag_marker=True,
    ),
    "pick": MRC_TEMPLATE(
        name="pick",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Based on the passage above, answer the question: {question}
Pick answer events from the following list:
{events}
Answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "List events that started after {event}.",  # noqa: E501
            "BEFORE": "List events that started before {event}.",  # noqa: E501
            "COEX": "List events that started around the same time but without clear temporal relations with {event}.",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
    ),
}

TEMPLATES_MRC_COT = {
    "hint_cot": MRC_TEMPLATE(
        name="hint_cot",
        inputs_prefix="",
        inputs=(
            """Answer the following question, "{question}" using the information below.
{context}
Hint: events are marked with "{marker}".
Reasonings:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
So, the answer events are: {target}
""",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events started after {event}?",  # noqa: E501
            "BEFORE": "Which events started before {event}?",  # noqa: E501
            "COEX": "Which events started around the same time but without clear temporal relations with {event}?",  # noqa: E501
        },
        flag_list_events=False,
        flag_marker=True,
        flag_cot=True,
        cot_mapping={
            "AFTER": "{answer} started after {event}.",  # noqa: E501
            "BEFORE": "{answer} started before {event}.",  # noqa: E501
            "COEX": "{answer} started around the same time but without clear temporal relations with {event}.",  # noqa: E501
        },
    ),
    "note_events_cot": MRC_TEMPLATE(
        name="note_events_cot",
        inputs_prefix="",
        inputs=(
            """Given the following passage
"{context}"
answer the following question. Note that following are the events:
{events}
Question: {question}
Step-by-step thoughts:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
Answer:
{target}""",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "List events that started after {event}.",  # noqa: E501
            "BEFORE": "List events that started before {event}.",  # noqa: E501
            "COEX": "List events that started around the same time but without clear temporal relations with {event}.",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
        flag_cot=True,
        cot_mapping={
            "AFTER": "{answer} started after {event}.",  # noqa: E501
            "BEFORE": "{answer} started before {event}.",  # noqa: E501
            "COEX": "{answer} started around the same time but without clear temporal relations with {event}.",  # noqa: E501
        },
    ),
    "refer_events_cot": MRC_TEMPLATE(
        name="refer_events_cot",
        inputs_prefix="",
        inputs=(
            """{context}
Events:
{events}
Q: {question}
Referring to the passage above:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
So, the correct answers to the given question is:
{target}""",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events happened after {event}?",  # noqa: E501
            "BEFORE": "Which events happened before {event}?",  # noqa: E501
            "COEX": "Which events happened around the same time but without clear temporal relations with {event}?",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
        flag_cot=True,
        cot_mapping={
            "AFTER": "{answer} happened after {event}.",  # noqa: E501
            "BEFORE": "{answer} happened before {event}.",  # noqa: E501
            "COEX": "{answer} happened around the same time but without clear temporal relations with {event}",  # noqa: E501
        },
    ),
    "careful_cot": MRC_TEMPLATE(
        name="careful_cot",
        inputs_prefix="",
        inputs=(
            """Given the following passage
"{context}"
answer the following question. Note that following are the events:
{events}
Question: {question}
Think slowly and carefully, before giving your answer:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
Answer:
{target}""",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "List events that started after {event}.",  # noqa: E501
            "BEFORE": "List events that started before {event}.",  # noqa: E501
            "COEX": "List events that started around the same time, but the temporal relationship with {event} is not clear.",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
        flag_cot=True,
        cot_mapping={
            "AFTER": "{answer} started after {event}.",  # noqa: E501
            "BEFORE": "{answer} started before {event}.",  # noqa: E501
            "COEX": "{answer} started around the same time, but the temporal relationship with {event} is not clear.",  # noqa: E501
        },
    ),
    "simple_cot": MRC_TEMPLATE(
        name="simple_cot",
        inputs_prefix="",
        inputs=(
            """Context:
{context}
Question: {question}
Answer candidates:
{events}
Chain of thoughts:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
Answer:
{target}""",
        example_separator="\n\n",
        question_mapping={
            "AFTER": "Which events happened after {event}?",  # noqa: E501
            "BEFORE": "Which events happened before {event}?",  # noqa: E501
            "COEX": "Which events do not have a clear temporal relation with {event}?",  # noqa: E501
        },
        flag_list_events=True,
        flag_marker=False,
        flag_cot=True,
        cot_mapping={
            "AFTER": "{answer} happened after {event}.",
            "BEFORE": "{answer} happened before {event}.",
            "COEX": "{answer} does not have a clear temporal relation with {event}.",
        },
    ),
}

TEMPLATES_TIMELINE = {
    "simple": TIMELINE_TEMPLATE(
        name="simple",
        inputs_prefix="",
        inputs=(
            """Context:
{context}
Timeline:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "based": TIMELINE_TEMPLATE(
        name="based",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Based on the above passage, create a timeline:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
    ),
    "note": TIMELINE_TEMPLATE(
        name="note",
        inputs_prefix="",
        inputs=(
            """Context:
{context}
Based on the context, create a timeline. Note events are marked with "{marker}."
Timeline:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_marker=True,
    ),
    "list": TIMELINE_TEMPLATE(
        name="list",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Events:
{events}
Based on the above passage and the events, create a timeline:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_list_events=True,
    ),
    "chronological": TIMELINE_TEMPLATE(
        name="chronological",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Events:
{events}
Based on the above passage and the events, create a timeline. Events in a timeline should be chronologically ordered:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_list_events=True,
    ),
    "coex": TIMELINE_TEMPLATE(
        name="coex",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Events:
{events}
Based on the above passage and the events, create a timeline. Events in a timeline should be chronologically ordered. You can put multiple events in one time range if they started around the same time without clear temporal relations:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_list_events=True,
    ),
    "read": TIMELINE_TEMPLATE(
        name="read",
        inputs_prefix="",
        inputs=(
            """Read the following passage:
{context}
In the passage, there are following events:
{events}
Based on the above passage and the events, create a timeline. Events in a timeline should be chronologically ordered. You can put multiple events in one time range if they started around the same time without clear temporal relations:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_list_events=True,
    ),
    "def": TIMELINE_TEMPLATE(
        name="def",
        inputs_prefix="",
        inputs=(
            """A timeline consists of multiple time spans, ordered chronologically. Each time span contains one or more events that started around the same time without clear temporal relations. Based on the passage below, create a timeline.
First, please read the passage:
{context}
Next, please identify events:
{events}
Timeline:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_list_events=True,
    ),
    "you": TIMELINE_TEMPLATE(
        name="you",
        inputs_prefix="",
        inputs=(
            """Please create a timeline based on a context. A timeline is a linear representation of events, ordered chronologically. You can put multiple events in one time span if they happened around the same time.
Given the passage below:
{context}
You can identify the following events:
{events}
Now, create a timeline with the events:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_list_events=True,
    ),
    "suppose": TIMELINE_TEMPLATE(
        name="suppose",
        inputs_prefix="",
        inputs=(
            """A timeline consists of multiple time spans, which ordered chronologically. Each time span contains one or more events that happened around the same time without clear temporal relations. Your task is to create a timeline from the next article.
Article:
{context}
Suppose you identified the following events from the article above:
{events}
create a timeline with the events:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        flag_list_events=True,
    ),
}

TEMPLATES_TIMELINE_COT = {
    "simple": TIMELINE_TEMPLATE(
        name="simple",
        inputs_prefix="",
        inputs=(
            """Context:
{context}
Chain of Thoughts:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
Timeline:
{target}""",
        example_separator="\n\n",
        flag_cot=True,
        cot_mapping={
            "AFTER": "{arg2} started after {arg1}.",  # noqa: E501
            "COEX": "{arg1} does not have a clear temporal relation with {arg2}.",  # noqa: E501
        },
    ),
    "list": TIMELINE_TEMPLATE(
        name="list",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Events:
{events}
Step-by-step thoughts:"""
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
Based on the passage, events, and step-by-step thoughts, create a timeline:
{target}""",
        example_separator="\n\n",
        flag_cot=True,
        flag_list_events=True,
        cot_mapping={
            "AFTER": "{arg2} happened after {arg1}.",  # noqa: E501
            "COEX": "{arg1} happened around the same time but without clear temporal relations with {arg2}.",  # noqa: E501
        },
    ),
    "chronological": TIMELINE_TEMPLATE(
        name="chronological",
        inputs_prefix="",
        inputs=(
            """Passage:
{context}
Events:
{events}
Referring to the passage and the events,"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        flag_cot=True,
        targets_prefix="",
        targets="""{cot}
Based on the information, create a timeline:
{target}""",
        example_separator="\n\n",
        flag_list_events=True,
        cot_mapping={
            "AFTER": "{arg2} started after {arg1}.",  # noqa: E501
            "COEX": "{arg1} started around the same time, but the temporal relationship with {arg2} is not clear.",  # noqa: E501
        },
    ),
    "you": TIMELINE_TEMPLATE(
        name="you",
        inputs_prefix="",
        inputs=(
            """Please create a timeline based on a context. A timeline is a linear representation of events, ordered chronologically. You can put multiple events in one time span if they happened around the same time.
Given the passage below:
{context}
You can identify the following events:
{events}
Think slowly and carefully, before giving your answer:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
Now, create a timeline with the events:
{target}""",
        example_separator="\n\n",
        flag_list_events=True,
        flag_cot=True,
        cot_mapping={
            "AFTER": "{arg2} happened after {arg1}.",  # noqa: E501
            "COEX": "{arg1} happened around the same time, but the temporal relationship with {arg2} is not clear.",  # noqa: E501
        },
    ),
    "suppose": TIMELINE_TEMPLATE(
        name="suppose",
        inputs_prefix="",
        inputs=(
            """A timeline consists of multiple time spans, which ordered chronologically. Each time span contains one or more events that happened around the same time without clear temporal relations. Your task is to create a timeline from the next article.
Article:
{context}
Suppose you identified the following events from the article above:
{events}
Additionally, you identified the following temporal relations in the article:"""  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="""{cot}
Then, the timeline is:
{target}""",
        example_separator="\n\n",
        flag_list_events=True,
        flag_cot=True,
        cot_mapping={
            "AFTER": "{arg2} started after {arg1}.",  # noqa: E501
            "COEX": "{arg1} started around the same time, but the temporal relationship with {arg2} is not clear.",  # noqa: E501
        },
    ),
}

TEMPLATES_TIMELINE_CODE = {
    "code_graph_id": TIMELINE_CODE_GRAPH_TEMPLATE(
        name="code_graph_id",
        inputs_prefix="",
        inputs=(
            '''"""
{context}
"""
# [Relations]
# "AFTER": The first event happened. After that, the second event happened.
# "COEX": The first and second event happened around the same time, but the temporal relationship between them is not clear.
class TemporalRelationGraph:
    def __init__(self):
        # Events
{events}

        # Edges'''  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target_template='add_edge({arg1}, "{relation}", {arg2})',
        target_event_template="self.event{idx}",
        event_template='self.event{idx} = "{representation}"',
    ),
    "code_graph_comment": TIMELINE_CODE_GRAPH_TEMPLATE(
        name="code_graph_comment",
        inputs_prefix="",
        inputs=(
            '''# [Relations]
# "AFTER": The first event started. After that, the second event started.
# "COEX": The first and second event started around the same time, but the temporal relationship between them is not clear.
"""
{context}
"""
class TemporalRelationGraph:
    def __init__(self, events):
        # Events
{events}

        # Edges'''  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target_template='add_edge({arg1}, "{relation}", {arg2})',
        target_event_template="self.event{idx}",
        event_template="self.event{idx} = events[{idx}] # {representation}",
    ),
    "code_timeline_id": TIMELINE_CODE_TIMELINE_TEMPLATE(
        name="code_timeline_id",
        inputs_prefix="",
        inputs=(
            '''"""
{context}
"""
class Timeline:
    def __init__(self):
        # Events
{events}

        # Timeline'''
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target_template="T{idx} = [{events}]",
        target_event_template="self.event{idx}",
        event_template='self.event{idx} = "{representation}"',
    ),
    "code_timeline_comment": TIMELINE_CODE_TIMELINE_TEMPLATE(
        name="code_timeline_comment",
        inputs_prefix="",
        inputs=(
            '''"""
{context}
"""
class Timeline:
    def __init__(self):
        # Events
{events}

        # Timeline'''
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target_template="T{idx} = [{events}]",
        target_event_template="self.event{idx}",
        event_template="self.event{idx} = events[{idx}] # {representation}",
    ),
    "code_timeline_comment_plus": TIMELINE_CODE_TIMELINE_TEMPLATE(
        name="code_timeline_comment_plus",
        inputs_prefix="",
        inputs=(
            '''"""
Context:
{context}
"""
class Timeline:
    """
    This class is the structured representation of the context above.
    * Events are the actions or happenings in it.
    * Timeline is a chronologically ordered events, where two events
        that happened around the same time are placed in the same time span.
        A time span is represented by T<num>.

    """
    def __init__(self):
        # Events
{events}

        # Timeline'''  # noqa: E501
        ),
        x_y_delimiter="\n",
        targets_prefix="",
        targets="{target}",
        example_separator="\n\n",
        target_template="T{idx} = [{events}]",
        target_event_template="self.event{idx}",
        event_template="self.event{idx} = events[{idx}] # {representation}",
    ),
}

TEMPLATES = {
    # benchmark
    "matres": TEMPLATES_MATRES,
    "tddiscourse": TEMPLATES_TDDISCOURSE,
    "torque": TEMPLATES_TORQUE,
    "temporal-nli": TEMPLATES_TEMPORALNLI,
    # formulation comparison
    "ctf-nli": TEMPLATES_NLI,
    "ctf-pairwise": TEMPLATES_PAIRWISE,
    "ctf-mrc": TEMPLATES_MRC,
    "ctf-mrc-cot": TEMPLATES_MRC_COT,
    "ctf-timeline": TEMPLATES_TIMELINE,
    "ctf-timeline-cot": TEMPLATES_TIMELINE_COT,
    "ctf-timeline-code": TEMPLATES_TIMELINE_CODE,
}


if __name__ == "__main__":
    tmp = [
        ("./data/preprocess/matres/dev.json", TEMPLATES_MATRES),
        ("./data/preprocess/torque/dev.json", TEMPLATES_TORQUE),
        ("./data/preprocess/tddiscourse/dev.json", TEMPLATES_TDDISCOURSE),
        ("./data/preprocess/temporal_nli/dev.json", TEMPLATES_TEMPORALNLI),
    ]
    # benchmarking:
    for path, templates in tmp:
        with open(path, "r") as f:
            examples = json.load(f)
        example = examples[4]
        print("#" * 50 + f"{path}" + "#" * 50)
        for template_name, template in templates.items():
            print("=" * 50 + f"[{template_name}]" + "=" * 50)
            assert template_name == template.name, template_name
            print(template.get_template_w_target())
            print("=" * 50)
            print(template.get_demonstration(example))
            print("=" * 50)
            print(template.get_prompt(example))

    # formulation comparison
    path = "./data/preprocessed/ctf/dev.json"
    with open(path, "r") as f:
        annotations = json.load(f)
    representation = "mention"
    marker = "eid"

    tmp = [
        (TEMPLATES_NLI, "nli"),
        (TEMPLATES_PAIRWISE, "pairwise"),
        (TEMPLATES_MRC, "mrc"),
        (TEMPLATES_MRC_COT, "mrc"),
        (TEMPLATES_TIMELINE, "timeline"),
        (TEMPLATES_TIMELINE_COT, "timeline"),
        (TEMPLATES_TIMELINE_CODE, "timeline"),
    ]
    for templates, task_name in tmp:
        print("#" * 50 + task_name + "#" * 50)
        example = _preprocess_example(annotations, task_name)[0]
        for template_name, template in templates.items():
            print("=" * 50 + f"[{template_name}]" + "=" * 50)
            assert template_name == template.name, template_name
            print(template.get_template_w_target())
            print("=" * 50)
            print(template.get_demonstration(example, representation, marker))
            print("=" * 50)
            print(template.get_prompt(example, representation, marker))
