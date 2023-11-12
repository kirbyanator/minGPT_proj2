from datasets import load_dataset
from mingpt.dataset import PileDataset
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path for json file")

args = parser.parse_args()

path = args.path

data = load_dataset("json", data_files=path)
data = data['train']['text']


dataset = PileDataset(data, collated=True)

with open("dataset.pkl", "wb") as f: 
    pickle.dump(dataset, f)