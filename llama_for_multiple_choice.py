from transformers import LlamaPreTrainedModel, LlamaModel, RobertaPreTrainedModel, RobertaModel
from torch import nn
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutputWithPast,
)
from transformers.cache_utils import (
    Cache,
    DynamicCache,
)


class LlamaForMultipleChoice(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        # self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # @can_return_tuple
    # @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> MultipleChoiceModelOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is labelscomputed (Cross-Entropy).
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None

        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]


        # transformer_outputs: BaseModelOutputWithPast = self.model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        # )

        transformer_outputs: BaseModelOutputWithPast = self.model(
            flat_input_ids,
            attention_mask=flat_attention_mask,
            position_ids=flat_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=flat_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = transformer_outputs.last_hidden_state

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (flat_input_ids != self.config.pad_token_id).to(hidden_states.device, torch.int32)
            token_indices = torch.arange(flat_input_ids.shape[-1], device=hidden_states.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1

        last_token_logits = hidden_states[torch.arange(batch_size * 2, device=hidden_states.device), last_non_pad_token]

        logits = self.classifier(last_token_logits)
        reshaped_logits = logits.reshape(-1, num_choices)


        loss = None
        if labels is not None:
            # loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)
            labels = labels.to(reshaped_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)


        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )