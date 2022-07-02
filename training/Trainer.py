import torch
import logging

from tqdm import tqdm
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score

class Trainer():
    
    def __init__(self, model=None, train_dataloader=None, valid_dataloader=None, config=None):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        self.logger = self.set_logger()
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 0, len(self.train_dataloader) * self.config.epoches)
    
    def train(self):
        best_score = 0
        for epoch in range(self.config.epoches):
            self.model.train()
            loss_list = list()
            train_bar = tqdm(self.train_dataloader)
            for inputs_encoded, batch_label in train_bar:
                self.optimizer.zero_grad()
                inputs_encoded, batch_label = { k: v.to(self.config.device) for k, v in inputs_encoded.items() }, batch_label.to(self.config.device)
                inputs_encoded.update({ "labels": batch_label })

                loss = self.model(**inputs_encoded).loss
                loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_bar.set_description("Training [ %d|%d ] Avg Ls = %.4f" % (epoch+1, self.config.epoches, sum(loss_list) / len(loss_list)))

            self.logger.info(f"Training Epoch: [ {epoch} / {self.config.epoches} ] | Avg Loss is {sum(loss_list) / len(loss_list)}")
        
            if self.valid_dataloader is not None:
                dev_score = self.validate(epoch)
                if dev_score > best_score:
                    best_score = dev_score
                    self.model.save_pretrained(self.config.save_path)
    
    def validate(self, epoch):
        self.model.eval()
        score = list()
        dev_bar = tqdm(self.valid_dataloader)
        with torch.no_grad():
            for inputs_encoded, batch_label in dev_bar:
                inputs_encoded = { k: v.to(self.config.device) for k, v in inputs_encoded.items() }
                
                if self.config.eva_method == "f1":
                    batch_label = batch_label[0].tolist()[1:-1]
                    prediction = torch.argmax(self.model(**inputs_encoded).logits[0][1:-1], dim=-1).tolist()
                    score.append(f1_score(batch_label, prediction, average="macro"))
                    dev_bar.set_description("Validing [ {}|{} ] Macro F1 is {:.4f}".format(epoch + 1, self.config.epoches, sum(score) / len(score)))
                
                elif self.config.eva_method == "acc":
                    batch_label = batch_label[0].tolist()[1:-1]
                    prediction = torch.argmax(self.model(**inputs_encoded).logits[0][1:-1], dim=-1).tolist()
                    score.append(1 if operator.eq(prediction, batch_label) else 0)
                    dev_bar.set_description("Validing [ {}|{} ] Acc is {:.2f}%".format(epoch + 1, self.config.epoches, (sum(score) / len(score))*100))
        
        self.logger.info(f"Validing Epoch: [ {epoch} / {self.config.epoches} ] | AvgScore is {sum(score) / len(score)}")
            
        return sum(score) / len(score)
    
    def set_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        
        handler = logging.FileHandler(self.config.logger_path, mode="w")
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        
        return logger