import torch

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class myDataset(Dataset):

    def __init__(self, config, mode):
        self.data = list()

        if mode == "train":
            data = open(config.train_path, "r", encoding="utf-8").readlines()

        elif mode == "valid":
            data = open(config.dev_path, "r", encoding="utf-8").readlines()

        for data_item in data:
            sentence, label = list(map(lambda x: x.split(), data_item.split("\t")))

            self.data.append({
                "sentence": sentence,
                "label": label
            })

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def myFn(batch, tokenizer, label2id, config):
    if "label" in batch[0].keys():
        sentences = [item['sentence'] for item in batch]
        labels = [item['label'] for item in batch]
        
        inputs_encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=config.max_len,
            is_split_into_words=True,
            return_tensors="pt"
        )
        
        labels_encoded = pad_sequence(
            [
                torch.tensor(
                    [label2id["O"]] + list(
                        map(lambda x: label2id[x], label if len(label) <= config.max_len else label[:config.max_len])
                    ) + [label2id["O"]]
                ) for label in labels
            ],
            batch_first = True,
            padding_value = label2id["O"]
        )

        return inputs_encoded, labels_encoded
    
    else:
        sentences = [item['sentence'] for item in batch]
        
        inputs_encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=config.max_len,
            is_split_into_words=True,
            return_tensors="pt"
        )

        return inputs_encoded, None