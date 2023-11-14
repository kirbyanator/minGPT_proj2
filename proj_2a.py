import pickle
from mingpt.dataset import PileDataset
from mingpt.trainer import Trainer
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from transformers import GPT2Tokenizer
import torch
from datasets import load_dataset
import os

from inference import generate


def main():
    # dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", 'plain_text', cache_dir='datasets')
    # dataset = load_dataset("json", data_files="/lustre/scratch/usr/dw87/pile_data_10.jsonl", cache_dir='pile')
    dataset = load_dataset("json", data_files="minipile.jsonl")
    dataset = dataset['train']
    # print(dataset)

    dataset = PileDataset(dataset, max_length=1024)


    gpt_config = GPT.get_default_config()

    gpt_config.vocab_size = dataset.vocab_size
    gpt_config.block_size = 1024
    gpt_config.model_type = 'gpt2'

    # check for checkpoints
    checkpoints = os.listdir('./checkpoints')
    checkpoints.sort()
    checkpoint = 'checkpoints/' + \
        checkpoints[-1] if checkpoints else None

    model = GPT(gpt_config)

    model.train()

    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = 500
    trainer_config.batch_size = 8
    trainer_config.num_workers = 1

    if checkpoint != None:
        model.load_state_dict(checkpoint.model_state_dict)
        trainer_config.load_optimizer = checkpoint.opimizer_state_dict

    def batch_end_callback(trainer):
        if trainer.iter_num % 1000 == 0 and trainer.iter_num !=0:      
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 10000 == 0 and trainer.iter_num !=0:
            loss = trainer.loss.item()
            iter_num = trainer.iter_num
            filepath = "checkpoints/davis_mingpt_2"+str(iter_num)+".pt"
            torch.save({
                        'epoch': iter_num,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'loss': loss,
                        'train_config':trainer_config,
                        'model_config':gpt_config

            }, filepath)
            print("Saving model")

    my_trainer = Trainer(trainer_config, model, dataset)

    my_trainer.set_callback('on_batch_end', batch_end_callback)

    my_trainer.run()

    loss = my_trainer.loss.item()
    iter_num = my_trainer.iter_num
    filepath = "checkpoints/davis_mingpt_2"+str(iter_num)+".pt"
    torch.save({
                'epoch': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': my_trainer.optimizer.state_dict(),
                'loss': loss,
                'train_config':trainer_config,
                'model_config':gpt_config

    }, filepath)
    print("Saving model")

    model.eval()
    with torch.no_grad():
        generate(model, "Davis Forster is a guy who", num_samples=10, steps=30)



if __name__ == "__main__":
    main()