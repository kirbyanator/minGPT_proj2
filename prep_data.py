from mingpt.dataset import PileDataset
from datasets import load_dataset
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", 'plain_text', cache_dir='datasets')
# dataset = load_dataset("json", data_files="minipile.jsonl")
dataset = dataset['train']
# print(dataset)
dataset

dataset = PileDataset(dataset, max_length=10)
x, y = dataset[0]
print(x)
print(y)

print(len(x))
print(len(y))

print(tokenizer.decode(x.squeeze()))
print(tokenizer.decode(y.squeeze()))