import torch
import torch.nn.functional as F

from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class BertForSpanClassification(BertPreTrainedModel):
    
    def __init__(self, config=None, model_args=None):
        super(BertForSpanClassification, self).__init__(config)
        self.config = config
        self.model_args = model_args
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout((
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        ))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        weights = torch.tensor([0.5, 3.0, 4.0, 1.0, 1.0, 10.0, 1.0, 5.0]).to(model_args.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)
        
        self.post_init()
    
    def forward(self, inputs_dict=None, entities_ids=None, labels=None, entities_mask=None):
        sequence_output_1 = self.dropout(self.bert(**inputs_dict).last_hidden_state)
        sequence_output_2 = self.dropout(self.bert(**inputs_dict).last_hidden_state)
        
        batch_size, max_len, hidden_dim = sequence_output_1.shape
        entity_count, entities_mask = entities_ids.shape[1], entities_mask.reshape(-1)
        entity_ids = entities_ids.reshape(batch_size, -1, 1).repeat(1, 1, hidden_dim)
        
        entity_states_1 = torch.gather(
            sequence_output_1, dim=1, index=entity_ids
        ).reshape(batch_size, entity_count, -1, hidden_dim)
        entity_states_1 = torch.max(entity_states_1, dim=2).values
        entity_logit_1 = self.classifier(entity_states_1)
        entity_logit_1 = entity_logit_1.reshape(-1, entity_logit_1.shape[-1])
        
        entity_states_2 = torch.gather(
            sequence_output_2, dim=1, index=entity_ids
        ).reshape(batch_size, entity_count, -1, hidden_dim)
        entity_states_2 = torch.max(entity_states_2, dim=2).values
        entity_logit_2 = self.classifier(entity_states_2)
        entity_logit_2 = entity_logit_2.reshape(-1, entity_logit_2.shape[-1])
        
        out_logit = torch.max(torch.stack([entity_logit_1, entity_logit_2], dim=-1), dim=-1).values
        
        loss = None
        if labels is not None:
            labels = labels.reshape(-1)
            entity_logit_filted_1, entity_logit_filted_2 = None, None
            
            for i, item in enumerate(entities_mask):
                if item == 1:
                    if entity_logit_filted_1 is None:
                        entity_logit_filted_1 = entity_logit_1[i, :].unsqueeze(0)
                        entity_logit_filted_2 = entity_logit_2[i, :].unsqueeze(0)
                    else:
                        entity_logit_filted_1 = torch.cat([entity_logit_filted_1, entity_logit_1[i, :].unsqueeze(0)], dim=0)
                        entity_logit_filted_2 = torch.cat([entity_logit_filted_2, entity_logit_2[i, :].unsqueeze(0)], dim=0)
            
            p_loss = F.kl_div(
                entity_logit_filted_1.log_softmax(dim=-1), entity_logit_filted_2.softmax(dim=-1), reduction='none'
            )
            q_loss = F.kl_div(
                entity_logit_filted_2.log_softmax(dim=-1), entity_logit_filted_1.softmax(dim=-1), reduction='none'
            )
            
            kl_loss = (p_loss.mean() + q_loss.mean()) * 0.5
            
            ce_loss = self.loss_fn(
                torch.max(torch.stack([entity_logit_filted_1, entity_logit_filted_2], dim=-1), dim=-1).values, labels
            )
            
            loss = ce_loss + self.model_args.alpha * kl_loss

        return TokenClassifierOutput(
            loss=loss,
            logits=out_logit
        )
            