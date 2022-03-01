import torch
import torch.nn as nn
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
except:
    pass
from modeling.modeling_roberta import RobertaModel, BertPreTrainedModel

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'lstm': ['lstm'],
}
try:
    MODEL_CLASS_TO_NAME['albert'] =  list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
except:
    pass

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}


class TextEncoder(BertPreTrainedModel):
    def __init__(self, config, model_name, **kwargs):
        super().__init__(config)
        if model_name == "roberta-base":
            self.roberta = RobertaModel(config, kwargs["with_naive_feature"], kwargs["entity_structure"])
        else:
            raise ValueError("Unsupported model name", model_name)
        self.reduced_dim = 128
        self.dim_reduction = nn.Linear(config.hidden_size, self.reduced_dim)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feature_size = self.reduced_dim
        if kwargs["with_naive_feature"]:
            self.feature_size += 20
            self.distance_emb = nn.Embedding(20, 20, padding_idx=10)
        self.bili = torch.nn.Bilinear(self.feature_size, self.feature_size, 97)
        self.hidden_size = config.hidden_size

        self.init_weights()
        print("Finish init_weights")

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            ent_mask=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            ent_ner=None,
            ent_pos=None,
            ent_distance=None,
            structure_mask=None,
            label=None,
            label_mask=None,
            output_attentions=False
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            ner_ids=ent_ner,
            ent_ids=ent_pos,
            structure_mask=structure_mask.float(),
        )

        attentions = outputs[-1]
        outputs = outputs[0]

        # # projection: dim reduction
        # outputs = torch.relu(self.dim_reduction(outputs))
        # ent_rep = torch.matmul(ent_mask, outputs)

        return outputs if not output_attentions else (outputs, attentions)
