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
        self.tokenizer.pad_token_id = 50256
        self.max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]['text']
        # encode every character to an integer
        tokens = self.tokenizer.encode(chunk, add_special_tokens=True, max_length=self.max_length, truncation=True, padding=True)
        
        # return as tensors
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y