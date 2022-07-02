import torch

from torch import nn
from model.dice_loss import DiceLoss
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput

class BertForTokenClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout((
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        ))
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.loss_fct = DiceLoss()
        
        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            targets = torch.masked_select(labels, (attention_mask==1))
            
            logits_list = list()
            attention_mask, logits = attention_mask.view(-1), logits.view(-1, logits.shape[-1])
            for i in range(attention_mask.shape[0]):
                if attention_mask[i]:
                    logits_list.append(logits[i, :])

            loss = self.loss_fct(torch.stack(logits_list), targets)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )