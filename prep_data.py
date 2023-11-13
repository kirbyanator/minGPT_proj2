from mingpt.dataset import PileDataset
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path for json file")

args = parser.parse_args()

path = args.path

PileDataset.process_json(path)