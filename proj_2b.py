from mingpt.dataset import PileDataset
from datasets import load_dataset
import numpy as np
from mingpt.trainer import Trainer
from mingpt.model import GPT


def main():
    print(np.arange(2, 6))
    print(np.mean(np.arange(2,6)))
    dataset = load_dataset("json", data_files="minipile.jsonl")
    dataset = dataset['train']

    normal_dataset = PileDataset(dataset, max_length=1024, ul2=False)
    ul2_dataset = PileDataset(dataset, max_length=1024, ul2=True)

    x, y = ul2_dataset[0]
    print(f"x is {x}")
    print(f"y is {y}")
    print(f"len of x: {len(x)}")
    print(f'shape of x is {x.shape}')
    print(f"len of y: {len(y)}")
    print(f"shape of y is {y.shape}")


    # run training, 500 iters on normal and 500 on ul2

    gpt_config = GPT.get_default_config()
    
    gpt_config.vocab_size = normal_dataset.vocab_size
    gpt_config.block_size = 1024
    gpt_config.model_type = 'gpt-nano'

    normal_gpt = GPT(gpt_config)

    ul2_config = GPT.get_default_config()

    ul2_config.vocab_size = ul2_dataset.vocab_size
    ul2_config.block_size = 1024
    ul2_config.model_type = 'gpt-nano'
    ul2_config.ul2 = True

    ul2_gpt = GPT(ul2_config)

    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = 500
    trainer_config.batch_size = 8
    trainer_config.num_workers = 1

    normal_trainer = Trainer(trainer_config, normal_gpt, normal_dataset)
    ul2_trainer = Trainer(trainer_config, ul2_gpt, ul2_dataset)

    normal_gpt.train()
    ul2_gpt.train()

    normal_trainer.run()
    ul2_trainer.run()

    print(normal_trainer.losses)
    print(ul2_trainer.losses)

    with open("normal_losses.txt", 'w') as output:
        for item in normal_trainer.losses:
            output.write(str(item) + '\n')

    with open("ul2_losses.txt", 'w') as output:
        for item in ul2_trainer.losses:
            output.write(str(item) + '\n')

    return 0

if __name__ == "__main__":
    main()