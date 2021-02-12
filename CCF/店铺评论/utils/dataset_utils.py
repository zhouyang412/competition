import torch
from torch.utils.data import Dataset


class ComDataset(Dataset):
    def __init__(self, features, mode):

        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = None

        if mode != 'test':
            self.labels = [torch.tensor(example.label) for example in features]
        else:
            self.id_ = [example.id_ for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]
        if self.id_ is not None:
            data['id_'] = self.id_[index]

        return data
