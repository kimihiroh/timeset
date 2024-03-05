"""
Model class

"""

import copy
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    T5PreTrainedModel,
    T5Config,
    DebertaV2PreTrainedModel,
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    StableDropout,
)
from transformers.models.t5.modeling_t5 import (
    T5Stack,
)
from transformers.modeling_outputs import (
    TokenClassifierOutput,
    SequenceClassifierOutput,
)
from typing import Optional, Union, Tuple


class T5ForTokenClassification(T5PreTrainedModel):
    """
    Reference:
    * https://github.com/huggingface/transformers/blob/ee88ae59940fd4b2c8fc119373143d7a1175c651/src/transformers/models/roberta/modeling_roberta.py#L1370
    * https://github.com/osainz59/t5-encoder/blob/main/t5_encoder/modeling_t5.py

    """  # noqa: E501

    _keys_to_ignore_on_load_unexpected = [r"decoder"]
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        classifier_dropout = config.dropout_rate

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.d_model, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune:
        dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (
            `torch.LongTensor` of shape `(batch_size, sequence_length)`,
            *optional*
            ):
            Labels for computing the token classification loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class DebertaClassificationHeadEMarker(nn.Module):
    """Head for emarker-based classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        # *2 because inputs <e1> and <e2> concat embeds
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def set_emarker_ids(self, e1_e2_ids):
        self.e1_id, self.e2_id = e1_e2_ids

    def forward(self, input_ids, features, **kwargs):
        input_ids_flatten = input_ids.flatten()
        outputs_hidden_states = features.view(-1, self.config.hidden_size)
        e1_output = outputs_hidden_states[input_ids_flatten == self.e1_id]
        e2_output = outputs_hidden_states[input_ids_flatten == self.e2_id]
        x = torch.cat((e1_output, e2_output), 1)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class DebertaForSequenceClassificationEMarker(DebertaV2PreTrainedModel):
    """
    ref:
    - https://github.com/huggingface/transformers/blob/ee88ae59940fd4b2c8fc119373143d7a1175c651/src/transformers/models/roberta/modeling_roberta.py#L1175
    - https://github.com/kimihiroh/kairos-extraction/blob/master/pairwise-relation-classification/models/modeling_roberta2.py
    """  # noqa: E501

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.deberta = DebertaV2Model(config)
        self.classifier = DebertaClassificationHeadEMarker(self.config)

        # Initialize weights and apply final processing
        self.post_init()

    def set_emarker_ids(self, e1_e2_ids):
        self.classifier.set_emarker_ids(e1_e2_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification loss.
            Indices should be in `[0, ..., config.num_labels - 1]`.
            classification loss is computed (Cross-Entropy).
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(input_ids, sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
