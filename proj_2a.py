import pickle
from mingpt.dataset import PileDataset
from mingpt.trainer import Trainer
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from transformers import GPT2Tokenizer
import torch

def main():
    with open('dataset.pkl', "rb") as dataset_file:
        data = pickle.load(dataset_file)
    print(type(data))
    dataset = PileDataset(data, max_length=50)


    gpt_config = GPT.get_default_config()

    gpt_config.vocab_size = dataset.vocab_size
    gpt_config.block_size = 50
    gpt_config.model_type = 'gpt-nano'

    model = GPT(gpt_config)
    model.train()

    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = 500

    my_trainer = Trainer(trainer_config, model, dataset)

    my_trainer.run()

    test_prompt = "Davis Forster is a guy who "
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    test_tokens = tokenizer.encode(test_prompt) + [tokenizer.eos_token_id]
    print(f"test tokens are: {test_tokens}")

    model.eval()
    model.generate(torch.tensor(test_tokens), 10)



if __name__ == "__main__":
    main()