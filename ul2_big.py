from mingpt.dataset import PileDataset
from datasets import load_dataset
import numpy as np
from mingpt.trainer import Trainer
from mingpt.model import GPT
import os
import torch


def main():
    print(np.arange(2, 6))
    print(np.mean(np.arange(2,6)))
    dataset = load_dataset("json", data_files="minipile.jsonl")
    dataset = dataset['train']

    normal_dataset = PileDataset(dataset, max_length=1024, ul2=False)
    ul2_dataset = PileDataset(dataset, max_length=1024, ul2=True)

    x, y = normal_dataset[0]
    print(f"x is {x}")
    print(f"y is {y}")
    print(f"len of x: {len(x)}")
    print(f'shape of x is {x.shape}')
    print(f"len of y: {len(y)}")
    print(f"shape of y is {y.shape}")

    # run training on saved checkpoint

    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = 500
    trainer_config.batch_size = 8
    trainer_config.num_workers = 1

    gpt_config = GPT.get_default_config()
    
    gpt_config.vocab_size = normal_dataset.vocab_size
    gpt_config.block_size = 1024
    gpt_config.model_type = 'gpt-nano'

    model = GPT(gpt_config)

    my_trainer = Trainer(trainer_config, model, normal_dataset)

    my_trainer.run()

    iter_num = my_trainer.iter_num
    loss = my_trainer.loss.item()
    filepath = "checkpoints/plain_model.pt"

    print("Saving model")

    torch.save({
                'epoch': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': my_trainer.optimizer.state_dict(),
                'loss': loss,
                'train_config':trainer_config,
                'model_config':gpt_config

    }, filepath)

    ul2_trainer = Trainer(trainer_config, model, ul2_dataset)

    ul2_trainer.run()

    iter_num = ul2_trainer.iter_num
    loss = ul2_trainer.loss.item()
    filepath = "checkpoints/model_plus_ul2.pt"

    print("Saving model")

    torch.save({
                'epoch': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': ul2_trainer.optimizer.state_dict(),
                'loss': loss,
                'train_config':trainer_config,
                'model_config':gpt_config

    }, filepath)

    return 0

if __name__ == "__main__":
    main()