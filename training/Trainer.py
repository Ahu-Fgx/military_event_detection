import torch
import logging
import operator

from tqdm import tqdm
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score

class Trainer():
    
    def __init__(self, model=None, train_dataloader=None, valid_dataloader=None, label2id=None, config=None):
        self.model = model
        self.config = config
        self.label2id = label2id
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.best_score = 0
        
        self.logger = self.set_logger()
        
        self.optimizer = AdamW(self.get_grouped_params(config, self.model))
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 0, len(self.train_dataloader) * self.config.epoches)
    
    def train(self):
        step = 0
        for epoch in range(self.config.epoches):
            self.model.train()

            loss_list = list()
            train_bar = tqdm(self.train_dataloader)
            for batch_input_encoded, batch_entities_ids_padded, batch_label_padded, entities_mask in train_bar:
                self.optimizer.zero_grad()
                loss = self.model(batch_input_encoded, batch_entities_ids_padded, batch_label_padded, entities_mask).loss
                loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_bar.set_description(
                    "Training [ %d|%d ] Avg Ls = %.4f" % (epoch+1, self.config.epoches, sum(loss_list) / len(loss_list))
                )
                
                if step % 50 == 0:
                    if self.valid_dataloader is not None:
                        dev_score = self.validate(epoch, step)
                        if dev_score > self.best_score:
                            self.best_score = dev_score
                            self.model.save_pretrained(self.config.save_path)
                step += 1

            self.logger.info(
                f"Training Epoch: [ {epoch + 1} / {self.config.epoches} ] | Avg Loss is {sum(loss_list) / len(loss_list)}"
            )
        
    def validate(self, epoch, step):
        self.model.eval()
        score = list()
        self.confusion_matrix = torch.zeros((len(self.label2id), len(self.label2id)))
        dev_bar = tqdm(self.valid_dataloader)
        with torch.no_grad():
            for batch_input_encoded, batch_entities_ids_padded, batch_label_padded, entities_mask in dev_bar:
                logits = self.model(batch_input_encoded, batch_entities_ids_padded, entities_mask=entities_mask).logits
                prediction = torch.argmax(logits[1:-1], dim=-1).tolist()

                score_metrics, macro_f1 = self.metrics(prediction, batch_label_padded.tolist())
                score.append(macro_f1)
                dev_bar.set_description(
                    "Validing [ {}|{} ] Step [ {} ] Macro F1 is {:.4f}".format(
                        epoch + 1, self.config.epoches, step, sum(score) / len(score)
                    )
                )
        
        torch.set_printoptions(precision=2,sci_mode=False)

        if (sum(score) / len(score)) > self.best_score:
            self.logger.info(
                f"Validing Epoch: [ {epoch} / {self.config.epoches} ] Step [ {step} ] | AvgScore is {sum(score) / len(score)}  ↑↑↑"
            )
        else:
            self.logger.info(
                f"Validing Epoch: [ {epoch} / {self.config.epoches} ] Step [ {step} ] | AvgScore is {sum(score) / len(score)}"
            )
            
        self.logger.info(f"Score Metrics is \n {score_metrics}")
        
        return sum(score) / len(score)

    def metrics(self, pre, truth):
        """ O标签一定在第一个 """
        
        assert len(pre) == len(truth)
        for i in range(len(pre)):
            self.confusion_matrix[pre[i], truth[i]] += 1
        
        scores = torch.zeros((len(self.label2id), 3))

        for i in range(len(self.label2id)):
            if torch.sum(self.confusion_matrix[:, i]) != 0:
                recall = self.confusion_matrix[i, i] / torch.sum(self.confusion_matrix[:, i])
            else:
                recall = 0

            if torch.sum(self.confusion_matrix[i, :]) != 0:
                precission = self.confusion_matrix[i, i] / torch.sum(self.confusion_matrix[i, :])
            else:
                precission = 0

            if recall + precission != 0:
                f1 = 2 * (recall * precission) / (recall + precission)
            else:
                f1 = 0

            scores[i, 0], scores[i, 1], scores[i, 2] = recall, precission, f1

        return [scores, torch.sum(scores[1:, -1]) / (scores.shape[0] - 1)]

    def get_grouped_params(self, args, model):
        no_decay = ["bias", "LayerNorm.weight"]
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
        group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
        group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                 'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']

        bert_named_params = dict()
        base_named_params = dict()
        for n, p in model.named_parameters():
            if "bert" in n:
                bert_named_params.update({ n: p })
            else:
                base_named_params.update({ n: p })
                
        optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_named_params.items() if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_named_params.items() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr / 2},
        {'params': [p for n, p in bert_named_params.items() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_named_params.items() if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr * 2},

        {'params': [p for n, p in bert_named_params.items() if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)], 'weight_decay': 0.0},
        {'params': [p for n, p in bert_named_params.items() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay': 0.0,
         'lr': args.bert_lr / 2},
        {'params': [p for n, p in bert_named_params.items() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay': 0.0,
         'lr': args.bert_lr},
        {'params': [p for n, p in bert_named_params.items() if
                    any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay': 0.0,
         'lr': args.bert_lr * 2},
        {'params': [p for n, p in base_named_params.items()], 'lr': args.learning_rate, "weight_decay": args.weight_decay}]
        
        return optimizer_grouped_parameters
    
    def set_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        
        handler = logging.FileHandler(self.config.logger_path, mode="w")
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        return logger