import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.utils import CfgNode as CN
from transformers import GPT2Tokenizer
from tqdm import tqdm
import pickle
import json

class PileDataset(Dataset):
    def __init__(self, data, max_length=1024):
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        text = self.data[idx]['text']
        # print(f"initial length: {len(text)}")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        tokens = self.tokenizer.encode(text, truncation=True, max_length=1024, padding='max_length')

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y