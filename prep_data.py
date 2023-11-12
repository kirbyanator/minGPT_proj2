from datasets import load_dataset
from mingpt.dataset import PileDataset
import pickle


data = load_dataset("json", data_files="minipile.jsonl")
data = data['train']

dataset = PileDataset(data, collated=True)

with open("dataset.pkl", "wb") as f: 
    pickle.dump(dataset, f)