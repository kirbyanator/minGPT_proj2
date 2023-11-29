import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from mingpt.utils import CfgNode as CN
from transformers import GPT2Tokenizer
from tqdm import tqdm
import random
import numpy as np
from scipy.stats import norm

class PileDataset(Dataset):
    def __init__(self, data, max_length=1024, ul2=False):
        self.data = data
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_length = max_length

        self.ul2 = ul2
        print(f"tokens: {len(self.tokenizer)}")
        self.tokenizer.add_tokens(['[S2S]', '[NLU]', '[NLG]'])
        self.mask_tokens = [f'mask_id_{i}' for i in range(200)]
        for token in self.mask_tokens:
            self.tokenizer.add_tokens(token)
        print(f"new tokens: {len(self.tokenizer)}")
        self.denoiser_tokens = {'s':'[S2S]', 'x':'[NLU]', 'r':'[NLG]'}
        self.vocab_size = len(self.tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.ul2:
            self.tokenizer.pad_token = self.tokenizer.eos_token

            text = self.data[idx]['text']
            # print(f"len text: {len(text)}")

            # print("doing ul2")

            # pick which corruption we're doing
            denoiser = random.choices(population=['s', 'x', 'r'], weights=[.5, .25, .25])[0]
            # print(f"denoiser is {denoiser}")
            denoiser_token = self.denoiser_tokens[denoiser]

            text = denoiser_token + ' ' + text
            # print(text)

            tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length, return_tensors='pt', add_special_tokens=True, padding=True)
            # print(tokens.shape)

            if denoiser == 's':
                x, y = self.__s_denoising(tokens)
            elif denoiser == 'x':
                x, y = self.__x_denoising(tokens)
            elif denoiser == 'r':
                x, y = self.__r_denoising(tokens)

            x = x.squeeze()
            y = y.squeeze()
            # print(f"x shape: {x.shape}")
            # print(f"y shape: {y.shape}")

            return x, y


        else:
            # grab a chunk of (block_size + 1) characters from the data
            text = self.data[idx]['text']
            # print(f"initial length: {len(text)}")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_length, padding='max_length')

            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            return x, y
        
    def __s_denoising(self, tokens):
        # get n tokens
        n_tokens = tokens.shape[1]

        gauss_value = random.gauss(.5, .2)
        remove_index = int(n_tokens * gauss_value)
        if remove_index == 0:
            remove_index += 1
        # print(f"remove index is {remove_index}")

        mask_token = self.tokenizer.convert_tokens_to_ids(self.mask_tokens[0])

        target_tokens = torch.cat((torch.tensor([[mask_token]]), tokens[:, remove_index:].clone()), dim=1)

        tokens = tokens[:, :remove_index + 1].clone()
        tokens[:, -1] = mask_token

        tokens = torch.cat((tokens, torch.tensor([[self.tokenizer.eos_token_id] * (self.max_length - tokens.shape[1] - 1)])), dim=1)
        target_tokens = torch.cat((target_tokens, torch.tensor([[self.tokenizer.eos_token_id] * (self.max_length - target_tokens.shape[1] - 1)])), dim=1)

        return tokens, target_tokens
    
    def __x_denoising(self, tokens, corruption_percent=.5, span_lengths=np.arange(2,6)):
        return self.__r_denoising(tokens, corruption_percent, span_lengths)
    
    def __r_denoising(self, tokens, corruption_percent=.15, span_lengths=np.arange(2,6)):
        chance = (corruption_percent / np.mean(span_lengths))
        target_tokens = None

        mask_tokens_used = 0
        span_offset = 0

        n_tokens = tokens.shape[1]

        for idx in range(1, n_tokens): # skip prepended id token
            # roll dice
            roll = np.random.random()

            if span_offset > 0:
                span_offset -= 1 # skips over rest of corrupted offset until back to 0
                continue

            if roll < chance:
                # corrupt stuff
                mask_token = self.tokenizer.convert_tokens_to_ids(self.mask_tokens[mask_tokens_used])
                corruption_span = np.random.choice(span_lengths)
                # print(f'corrupting with span of {corruption_span} at {idx} with mask token {mask_token}')

                if target_tokens is None:
                    target_tokens = torch.tensor([[mask_token]])
                    target_tokens = torch.cat((target_tokens, tokens[:, idx:idx+corruption_span]), dim=1)

                else:
                    target_tokens = torch.cat((target_tokens, torch.tensor([[mask_token]]), tokens[:, idx:idx + corruption_span]), dim=1)

                # tokens = torch.cat((tokens[:, :idx], torch.tensor([[mask_token] * corruption_span]), tokens[:, idx + corruption_span:]), dim=1)
                tokens = torch.cat((tokens[:, :idx], torch.tensor([[mask_token]]), tokens[:, idx + corruption_span:]), dim=1)
                span_offset = corruption_span
                mask_tokens_used += 1

        # padding
        tokens = torch.cat((tokens, torch.tensor([[self.tokenizer.eos_token_id] * (self.max_length - tokens.shape[1] - 1)])), dim=1)
        if target_tokens == None:
            print("NO TARGETS")
            target_tokens = torch.tensor([self.tokenizer.eos_token_id] * (self.max_length - 1))
        else:
            target_tokens = torch.cat((target_tokens, torch.tensor([[self.tokenizer.eos_token_id] * (self.max_length - target_tokens.shape[1] - 1)])), dim=1)

        # print(f"tokens: {(tokens)}")
        # print(f"target tokens: {target_tokens}")

        # print(f"string: {self.tokenizer.decode(tokens.squeeze())}")
        # print(f"target string: {self.tokenizer.decode(target_tokens.squeeze())}")
        return (tokens, target_tokens)