from torch.utils.data import DataLoader
from configuration import Config
from data_processer.data_process import myDataset, myFn
from transformers import AutoTokenizer, BertConfig
from model.bert_dice import BertForTokenClassification
from Trainer import Trainer

config = Config()

tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)

id2label = eval(open(config.id2label_path, "r", encoding="utf-8").readline())
label2id = { label: i for i, label in enumerate(id2label) }

train_dataset = myDataset(config, mode="train")
valid_dataset = myDataset(config, mode="valid")

train_dataloader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              collate_fn=lambda x: myFn(x, tokenizer, label2id, config))
valid_dataloader = DataLoader(valid_dataset,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=lambda x: myFn(x, tokenizer, label2id, config))

model_config = BertConfig(vocab_size=config.vocab_size, num_labels=len(id2label))
model = BertForTokenClassification.from_pretrained(config.pretrained_model_path, config=model_config).to(config.device)

trainer = Trainer(model, train_dataloader, valid_dataloader, config)
trainer.train()