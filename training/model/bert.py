import torch

from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class BertForSpanClassification(BertPreTrainedModel):
    
    def __init__(self, config=None, model_args=None):
        super(BertForSpanClassification, self).__init__(config)
        self.config = config
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout((
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        ))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        weights = torch.tensor([0.5, 3.0, 4.0, 1.0, 1.0, 10.0, 1.0, 5.0]).to(model_args.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        
        self.post_init()
    
    def forward(self, inputs_dict=None, entities_ids=None, labels=None, entities_mask=None):
        outputs = self.bert(**inputs_dict).last_hidden_state

        sequence_output = self.dropout(outputs)
        
        batch_size, max_len, hidden_dim = sequence_output.shape
        entity_count = entities_ids.shape[1]
        entities_mask = entities_mask.reshape(-1)
        
        entity_ids = entities_ids.reshape(batch_size, -1, 1).repeat(1, 1, hidden_dim)
        entity_states = torch.gather(
            sequence_output, dim=1, index=entity_ids
        ).reshape(batch_size, entity_count, -1, hidden_dim)
        entity_states = torch.mean(entity_states, dim=2)  # 取实体首尾hidden_states的均值
        entity_logit = self.classifier(entity_states)  # [btz, 实体个数，实体类型数]
        entity_logit = entity_logit.reshape(-1, entity_logit.shape[-1])
        
        loss = None
        if labels is not None:
            labels = labels.reshape(-1)
            entity_logit_filted = None
            
            for i, item in enumerate(entities_mask):
                if item == 1:
                    if entity_logit_filted is None:
                        entity_logit_filted = entity_logit[i, :].unsqueeze(0)
                    else:
                        entity_logit_filted = torch.cat([entity_logit_filted, entity_logit[i, :].unsqueeze(0)], dim=0)
                        
            loss = self.loss_fn(entity_logit_filted, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=entity_logit
        )
            