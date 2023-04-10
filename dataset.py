import torch
from torch.utils.data import Dataset

# Dataset 클래스
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, label_idx):
        self.data = data
        self.tokenizer = tokenizer
        self.label_idx = label_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['question']
        inputs = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        label = torch.tensor(self.data.iloc[idx][f'label{self.label_idx}'], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }
