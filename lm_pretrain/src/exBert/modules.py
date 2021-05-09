"""PyTorch exBERT model."""

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEmbeddings,
    BertEncoder,
    BertForPreTraining,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertIntermediate,
    BertModel,
    BertOnlyMLMHead,
    BertOutput,
    BertPooler,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertPreTrainingHeads,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.configuration_utils import PretrainedConfig

import os
from typing import Union


class exBertEmbeddings(BertEmbeddings):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.config = config
        self.word_embeddings_ADD = nn.Embedding(config_ADD.vocab_size, config.hidden_size, padding_idx=0)

    def forward(self, input_ids, *args, **kwargs):
        if kwargs.get("inputs_embeds", None) is None:
            Input_ids_new = input_ids.clone()
            Input_ids_new[input_ids <= (self.config.vocab_size - 1)] = 0
            Input_ids_new[input_ids > (self.config.vocab_size - 1)] = Input_ids_new[input_ids > (self.config.vocab_size - 1)] - self.config.vocab_size
            Input_ids_old = input_ids.clone()
            Input_ids_old[input_ids > (self.config.vocab_size - 1)] = 0
            words_embeddings = self.word_embeddings(Input_ids_old.long())
            words_embeddings.view(-1, self.config.hidden_size)[(input_ids == 0).reshape(-1)] = 0
            words_embeddings_ADD = self.word_embeddings_ADD(Input_ids_new.long())
            words_embeddings_ADD.view(-1, self.config.hidden_size)[(input_ids == 0).reshape(-1)] = 0
            kwargs["inputs_embeds"] = words_embeddings + words_embeddings_ADD
        assert kwargs["inputs_embeds"].size()[-1] == self.word_embeddings_ADD.weight.size()[-1]
        return super().forward(*args, **kwargs)


class exBertSelfAttention(BertSelfAttention):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.num_attention_heads = config_ADD.num_attention_heads
        self.attention_head_size = int(config_ADD.hidden_size / config_ADD.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


class exBertSelfOutput(BertSelfOutput):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.dense = nn.Linear(config_ADD.hidden_size, config.hidden_size)


class exBertAttention(BertAttention):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.self = exBertSelfAttention(config, config_ADD)
        self.output = exBertSelfOutput(config, config_ADD)


class exBertIntermediate(BertIntermediate):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.dense = nn.Linear(config.hidden_size, config_ADD.intermediate_size)


class exBertOutput(BertOutput):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.dense = nn.Linear(config_ADD.intermediate_size, config.hidden_size)


class exBertLayer(nn.Module):
    def __init__(self, config, config_ADD):
        assert config_ADD.hidden_size > 0
        super().__init__()
        self.attention = exBertAttention(config, config)
        self.intermediate = exBertIntermediate(config, config)
        self.output = exBertOutput(config, config)
        self.attention_ADD = exBertAttention(config, config_ADD)
        self.intermediate_ADD = exBertIntermediate(config, config_ADD)
        self.output_ADD = exBertOutput(config, config_ADD)
        self.gate_ADD = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        gated_value_layer = torch.sigmoid(self.gate_ADD(hidden_states))
        attention_output = self.attention(hidden_states, attention_mask)[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        attention_output_ADD = self.attention_ADD(hidden_states, attention_mask)[0]
        intermediate_output_ADD = self.intermediate_ADD(attention_output_ADD)
        layer_output_ADD = self.output_ADD(intermediate_output_ADD, attention_output_ADD)
        Layer_output = gated_value_layer * layer_output + (1 - gated_value_layer) * layer_output_ADD
        return (Layer_output, )


class exBertEncoder(BertEncoder):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.layer = nn.ModuleList([exBertLayer(config, config_ADD) for _ in range(config.num_hidden_layers)])


class exBertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, bert_model_embedding_ADD_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

        # extension decoder
        self.decoder_ADD = nn.Linear(config.hidden_size, bert_model_embedding_ADD_weights.size(0), bias=False)
        self.decoder_ADD.weight = bert_model_embedding_ADD_weights
        self.bias_ADD = nn.Parameter(torch.zeros(bert_model_embedding_ADD_weights.size(0)))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder_ADD.bias = self.bias_ADD

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = (
            torch.cat([self.decoder(hidden_states), self.decoder_ADD(hidden_states)], dim=2)
            + torch.cat([self.bias, self.bias_ADD])
        )
        return hidden_states


class BertOnlyMLMHead(BertOnlyMLMHead):
    def __init__(self, config, bert_model_embedding_weights, bert_model_embedding_ADD_weights):
        self.predictions = exBertLMPredictionHead(
            config,
            bert_model_embedding_weights,
            bert_model_embedding_ADD_weights)


class exBertPreTrainingHeads(BertPreTrainingHeads):
    def __init__(self, config, bert_model_embedding_weights, bert_model_embedding_ADD_weights):
        super().__init__(config)
        self.predictions = exBertLMPredictionHead(
            config,
            bert_model_embedding_weights,
            bert_model_embedding_ADD_weights)


class exBertModel(BertModel):
    def __init__(self, config, config_ADD=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.embeddings = exBertEmbeddings(config, config_ADD)
        self.encoder = exBertEncoder(config, config_ADD)


class exBertForPreTraining(BertForPreTraining):
    def __init__(self, config, config_ADD):
        assert config_ADD.vocab_size > 0
        super().__init__(config)

        self.bert = exBertModel(config, config_ADD)
        self.cls = exBertPreTrainingHeads(
            config,
            self.bert.embeddings.word_embeddings.weight,
            self.bert.embeddings.word_embeddings_ADD.weight)

        # create a copies of the configs
        self.config = self.config_class.from_dict(config.to_dict())
        self.config_ADD = self.config_class.from_dict(config_ADD.to_dict())

        # resize the vocabulary
        self.config.vocab_size = config.vocab_size + config_ADD.vocab_size

        self.init_weights()

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        *args,
        **kwargs,
    ):
        super().save_pretrained(save_directory, save_config, *args, **kwargs)
        if save_config:
            self.config_ADD.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        *model_args,
        **kwargs,
    ):
        config_ADD = kwargs.pop("config_ADD", None)
        cache_dir = kwargs.get("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        resume_download = kwargs.get("resume_download", False)
        proxies = kwargs.get("proxies", None)
        local_files_only = kwargs.get("local_files_only", False)
        use_auth_token = kwargs.get("use_auth_token", None)
        revision = kwargs.get("revision", None)
        from_pipeline = kwargs.get("_from_pipeline", None)
        from_auto_class = kwargs.get("_from_auto", False)

        # Load config if we don't provide a configuration
        if not isinstance(config_ADD, PretrainedConfig):
            assert isinstance(config_ADD, (str, os.PathLike))
            config_ADD, _ = cls.config_class.from_pretrained(
                config_ADD,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline
            )
        kwargs["config_ADD"] = config_ADD

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class exBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.bert = exBertModel(config, config_ADD)


class exBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config, config_ADD):
        super().__init__(config)
        self.bert = exBertModel(config, config_ADD)
