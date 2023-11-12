import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.utils import CfgNode as CN
from transformers import GPT2Tokenizer
from tqdm import tqdm
import json

class PileDataset(Dataset):

    def __init__(self, data_path, max_length=1024, collated=True):
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size

        if collated == True:
            with open(data_path, "r") as file:
                self.data = []
                for line in tqdm(file):
                    self.data += self.tokenizer.encode(line, max_length=max_length, truncation=True)
                    self.data += [self.tokenizer.eos_token_id]
        else:
            with open(data_path, "r") as file:
                self.data = []
                for line in tqdm(file):
                    self.data.append(self.tokenizer.encode(line, max_length=max_length, truncation=True) + [self.tokenizer.eos_token_id])


    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.max_length + 1]
        # encode every character to an integer
        # return as tensors
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y