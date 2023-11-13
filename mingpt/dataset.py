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

    @staticmethod
    def process_json(json_path, max_length=1024):
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        with open(json_path, "r") as file:
            data = []
            for line in tqdm(file):
                text = json.loads(line)['text']
                # print(f"text is {text}")
                data += tokenizer.encode(text, max_length=max_length, truncation=True)
                data += [tokenizer.eos_token_id]
                # print(tokenizer.decode(data))
                # exit()
        with open("dataset.pkl", "wb") as f: 
            pickle.dump(data, f)

    def __init__(self, data, max_length=1024):
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size
        self.data = data

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