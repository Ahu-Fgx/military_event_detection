import json
import torch
import operator
import collections

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def pad_sequence_ls(ls):
    max_len = max(list(map(len, ls)))
    ls = list(map(lambda x: x + [[0, 0]] * (max_len - len(x)), ls))
    entities_mask = list(map(lambda x: [0 if operator.eq(ids, [0, 0]) else 1 for ids in x], ls))
    
    return torch.tensor(ls, dtype=torch.long), torch.tensor(entities_mask, dtype=torch.long)

class myDataset(Dataset):

    def __init__(self, config, mode):
        self.data = list()

        if mode == "train":
            data = json.load(open(config.train_path, "r", encoding="utf-8"))

        elif mode == "valid":
            data = json.load(open(config.dev_path, "r", encoding="utf-8"))

        for data_item in tqdm(data, desc=f"Reading From {mode} Data"):
            sentence = list(data_item["sentence"].strip())
            participle_sentence = data_item["tokens"]
            
            begin, part_token_ids, labels = 1, list(), list()
            
            if len(data_item["event_mention"]) != 0:
                token_type = data_item["event_mention"]["event_type"]
                trigger_offset = data_item["event_mention"]["trigger"]["offset"]
                
                for i, token in enumerate(participle_sentence):
                    part_token_ids.append([begin, begin + len(token) - 1])
                    if i >= trigger_offset[0] and i < trigger_offset[1]:
                        labels.append(token_type)
                    else:
                        labels.append("O")
                    begin += len(token)
                    
            else:
                for i, token in enumerate(participle_sentence):
                    part_token_ids.append([begin, begin + len(token) - 1])
                    labels.append("O")
                    begin += len(token)
            
            self.data.append([sentence, part_token_ids, labels])
        

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def myFn(batch, tokenizer, label2id, config):
    batch_sentence, batch_entities_ids, batch_label = list(map(lambda x: list(x), list(zip(*batch))))
    
    batch_entities_ids = list(map(lambda x: [[0, 0]] + x + [[0, 0]], batch_entities_ids))
    batch_label = list(
        map(
            lambda x: list(
                map(lambda y: label2id[y], x)
            ),
            batch_label
        )
    )
    
    batch_input_encoded = tokenizer(
        batch_sentence,
        padding=True,
        truncation=True,
        max_length=config.max_len,
        is_split_into_words=True,
        return_tensors="pt"
    )
    batch_entities_ids_padded, entities_mask = pad_sequence_ls(batch_entities_ids)

    batch_input_encoded = { k: v.to(config.device) for k, v in batch_input_encoded.items() }
    
    return [
        batch_input_encoded,
        batch_entities_ids_padded.to(config.device),
        torch.tensor(sum(batch_label, list()), dtype=torch.long).view(-1).to(config.device),
        entities_mask.to(config.device)
    ]